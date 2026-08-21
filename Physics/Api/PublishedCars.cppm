module;

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:PublishedCars;

import :Contact;
import :ContactPatch;
import :Coupling;
import :Driveline;
import :RigidBody;
import :Suspension;
import :Telemetry;
import :Tyre;
import :Vehicle;

namespace raceengine
{

// A real car, from its own published physics data rather than from plausible invention.
//
// The source is a publicly distributed Assetto Corsa model of a Volkswagen Golf GTI Mk7.5, unpacked
// and parsed by `scripts/import-ac-car.py`. What is here is what that data states, or arithmetic on
// what it states; where AC does not model something this engine does, the number is marked and the
// reason is given. Nothing in this file is a number somebody thought looked about right, and the
// distinction from `placeholderSedan()` is exactly that — the placeholder is a *fixture*, calibrated
// so the model can be validated against a car whose every figure is chosen, and it stays.
//
// **The conversion of the pickup points is verified rather than assumed.** AC states them relative
// to the wheel centre, x measured inboard from the wheel's centre plane, y up and z forward:
//
//     x = track / 2 - x_ac
//     y = y_ac + tyreRadius        (so the design contact patch sits on y = 0)
//     z = z_ac + axleStation
//
// Converting with that and reading the track back off the resulting wheel centre reproduces the
// TRACK the same file states, on both axles, to four decimal places. Any other convention misses by
// tens of centimetres, which is why that check is a test rather than a comment.
//
// The model frame is the engine's: **+x is the car's left** (`outboardSign`), +y up, +z forward,
// origin midway between the axles at
// road level. AC's own frame has its origin at the centre of gravity, so everything AC states in it
// — aero surfaces, the collider — is carried across by the centre of gravity's position below.

// suspensions.ini [BASIC] WHEELBASE, and the axles placed symmetrically about the origin.
inline constexpr auto golfWheelbase = 2.638;
inline constexpr auto golfFrontAxle = 0.5 * golfWheelbase;
inline constexpr auto golfRearAxle = -0.5 * golfWheelbase;

// suspensions.ini [FRONT]/[REAR] TRACK, and tyres.ini RADIUS.
inline constexpr auto golfFrontTrack = 1.539;
inline constexpr auto golfRearTrack = 1.516;
inline constexpr auto golfTyreRadius = 0.3298;

// car.ini [BASIC] TOTALMASS, and suspensions.ini HUB_MASS per corner.
inline constexpr auto golfTotalMass = 1348.0;
inline constexpr auto golfFrontHubMass = 80.0;
inline constexpr auto golfRearHubMass = 85.0;

// Front: MacPherson strut, from suspensions.ini [FRONT]. The strut's top bearing is STRUT_CAR and
// its lower end is the lower ball joint, which is WBTYRE_BOTTOM — the same point STRUT_TYRE names,
// and the two agreeing is the file's own statement that the strut picks up on the ball joint.
export [[nodiscard]] CornerHardpoints golfMk7FrontCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * (0.5 * golfFrontTrack - x), y + golfTyreRadius, z + golfFrontAxle);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;
    corner.kind = SuspensionKind::MacPhersonStrut;

    corner.lower = Wishbone{.frontPivot = at(0.40769, -0.10955, 0.01423),
                            .rearPivot = at(0.40944, -0.12017, -0.27373),
                            .ballJoint = at(0.07174, -0.13710, 0.01213)};
    corner.strutTop = at(0.25366, 0.54933, -0.04296);
    corner.steeringRackOuter = at(0.31000, -0.06546, -0.22200);
    corner.steeringArm = at(0.05200, -0.06757, -0.12586);
    corner.wheelCentre = at(0.0, 0.0, 0.0);
    corner.wheelRadius = golfTyreRadius;

    corner.droopAngle = -0.16;
    corner.bumpAngle = 0.16;

    return corner;
}

// Rear: double wishbone, from suspensions.ini [REAR].
//
// A production Mk7.5 has a multi-link rear rather than a wishbone; the model approximates it as one,
// which is why the lower arm's pivot axis is 0.74 m long — that is a trailing arm wearing a
// wishbone's clothes. The curves it produces are a road car's, but it is a model of a model.
export [[nodiscard]] CornerHardpoints golfMk7RearCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * (0.5 * golfRearTrack - x), y + golfTyreRadius, z + golfRearAxle);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;

    corner.lower = Wishbone{.frontPivot = at(0.17702, -0.11686, 0.50420),
                            .rearPivot = at(0.57702, -0.11686, -0.11540),
                            .ballJoint = at(0.04602, -0.12706, 0.01870)};
    corner.upper = Wishbone{.frontPivot = at(0.34842, 0.09370, 0.22140),
                            .rearPivot = at(0.39800, 0.08540, 0.03080),
                            .ballJoint = at(0.06622, 0.10070, -0.00650)};
    corner.steeringRackOuter = at(0.58002, -0.09806, -0.10540);
    corner.steeringArm = at(0.07202, -0.10406, -0.09540);
    corner.wheelCentre = at(0.0, 0.0, 0.0);
    corner.wheelRadius = golfTyreRadius;

    // **AC states no damper pickup points for a double wishbone**, because it states spring and
    // damper rates at the wheel and never needs them. Left defaulted, both mounts land on the origin
    // and the motion ratio that comes out is a plausible-looking number describing a linkage that
    // does not exist — `validateCornerSetup` refuses exactly that, and `PublishedGeometryTests` pins
    // the refusal.
    //
    // So this is the one piece of geometry here that is not from the file, and it is placed rather
    // than fitted: the Mk7's rear damper picks up on the wheel carrier and stands very nearly
    // upright, so the lower eye goes on the ball joint and the upper mount directly above it. That
    // arrangement has a motion ratio of about one, which is what makes AC's at-the-wheel rates pass
    // through the conversion below essentially unchanged — the least the choice can be made to
    // matter. Only the *length* is free, and nothing reads it but the spring's rest length.
    corner.damperWishbone = corner.lower.ballJoint;
    corner.damperChassis = corner.lower.ballJoint + glm::dvec3(0.0, 0.40, 0.0);

    corner.droopAngle = -0.14;
    corner.bumpAngle = 0.14;

    return corner;
}

namespace
{

inline constexpr auto standardGravity = 9.80665;
inline constexpr auto rpmToRadiansPerSecond = 0.10471975511965977;

// AC states its damper as a rate with a knee: `rate` up to `knee` metres per second of *wheel*
// velocity and `fastRate` above it. The model evaluates the damper at the shaft, and the shaft moves
// `ratio` times as fast as the wheel while carrying `1/ratio` of its force — so a wheel-referred
// coefficient becomes a shaft one divided by the square of the ratio, and the speed the knee sits at
// is multiplied by it. Both halves of that have to be done or the knee lands at the wrong speed.
[[nodiscard]] Curve kneedDamper(const double bumpRate, const double fastBumpRate, const double bumpKnee,
                                const double reboundRate, const double fastReboundRate, const double reboundKnee,
                                const double motionRatio)
{
    const auto ratio = std::abs(motionRatio);
    const auto square = std::max(ratio * ratio, 1e-9);
    const auto span = 5.0;

    const auto bumpKneeSpeed = ratio * bumpKnee;
    const auto bumpKneeForce = (bumpRate / square) * bumpKneeSpeed;
    const auto bumpEnd = bumpKneeForce + (fastBumpRate / square) * (span - bumpKneeSpeed);

    const auto reboundKneeSpeed = ratio * reboundKnee;
    const auto reboundKneeForce = (reboundRate / square) * reboundKneeSpeed;
    const auto reboundEnd = reboundKneeForce + (fastReboundRate / square) * (span - reboundKneeSpeed);

    return Curve{.points = {glm::dvec2(-span, -reboundEnd), glm::dvec2(-reboundKneeSpeed, -reboundKneeForce),
                            glm::dvec2(0.0, 0.0), glm::dvec2(bumpKneeSpeed, bumpKneeForce), glm::dvec2(span, bumpEnd)}};
}

// The rack travel that steers the front wheels to the lock the car states, **signed**. AC gives the
// steering wheel's own lock and the ratio between the two, so the road wheel's angle is data; the
// travel that produces it is a property of this linkage and is solved for rather than guessed.
// Monotonic in the magnitude over the range a rack has, so a bisection is enough and cannot pick a
// second root.
//
// The sign is the half that used to be missing, and it was missing invisibly. The bisection below
// searches positive travel against `std::abs(toe)`, so it can only ever answer *how far*; which way
// was then taken from `copysign(..., steerRatio)` — the sign of a steering *ratio*, which is a
// different quantity that happens to carry one. On this car that produced a rack that steered left
// when the driver asked for right, on both the wheel and the keyboard, and nothing downstream could
// tell: every stage from the kinematic solve to the force feedback was faithfully consistent with a
// rack going the wrong way.
//
// Taken from the linkage instead, and **read off the wheel's own nose rather than off its toe**.
// Toe is documented as positive when the front of the wheel points toward the car on both sides,
// which makes it a mirrored quantity and therefore the worst possible thing to read a left-or-right
// answer from: two separate attempts reasoned from it and both came out inverted. The second was
// worse than the first, because it was "calibrated" against a chassis yaw rate whose own sign was
// read in a frame nobody had pinned down — the answer looked measured and was not.
//
// What it reads now is where the nose of the wheel moves, against `outboardSign`, which is the one
// place the frame's handedness is stated and the only thing here that has been checked against a
// picture. No mirrored quantity is involved and no car-specific assumption either: a rack behind the
// axle steers the opposite way to one ahead of it, and this measures whichever this car has.
//
// The magnitude is searched **along the direction that steers right** rather than along +x. Rack
// travel is applied toward +x on both corners, so it is outboard on one and inboard on the other,
// and a linkage that reaches full lock one way need not reach it the other within a rack's travel —
// searching the wrong way is an honest-looking "cannot reach the lock the car states".
[[nodiscard]] std::expected<double, std::string> rackTravelForSteer(const CornerHardpoints& hardpoints,
                                                                    const double roadWheelAngle)
{
    // Where this wheel is pointing, as a direction rather than as an angle with a convention on it.
    const auto noseAt = [&hardpoints](const double travel) -> std::expected<glm::dvec3, std::string>
    {
        const auto solved = solveCorner(hardpoints, 0.0, travel);
        if (!solved)
        {
            return std::unexpected(solved.error());
        }

        return solved->uprightOrientation * glm::dvec3(0.0, 0.0, 1.0);
    };

    // Small enough to be unambiguously in the linear part of the linkage, large enough to be clear
    // of the solve's own noise.
    constexpr auto probeTravel = 0.01;

    const auto centred = noseAt(0.0);
    const auto probed = noseAt(probeTravel);
    if (!centred || !probed)
    {
        return std::unexpected("the steering linkage will not solve at a small rack travel: " +
                               (centred ? probed.error() : centred.error()));
    }

    // The car's right, from the one function allowed to know which way that is.
    const auto towardRight = outboardSign(CornerSide::Right);
    const auto direction = (probed->x - centred->x) * towardRight > 0.0 ? 1.0 : -1.0;

    auto low = 0.0;

    const auto steerAt = [&hardpoints, direction](const double travel) -> std::expected<double, std::string>
    {
        const auto solved = solveCorner(hardpoints, 0.0, direction * travel);
        if (!solved)
        {
            return std::unexpected(solved.error());
        }

        return std::abs(solved->toe);
    };

    const auto steersRightForPositiveTravel = direction > 0.0;

    // The bracket is **found rather than assumed**, and 0.15 m is past the end of this linkage. A
    // steering arm offset from the kingpin reaches further one way than the other — the Golf's front
    // corner solves to about 110 mm of rack in one direction and 90 in the other — so a fixed upper
    // bracket is a geometry question dressed as a constant, and it answers "the rack cannot reach
    // the lock the car states" for a car whose lock is at 70 mm and perfectly reachable.
    auto high = 0.15;
    while (high > 0.01 && !steerAt(high))
    {
        high *= 0.5;
    }

    const auto reach = steerAt(high);
    if (!reach)
    {
        return std::unexpected("the steering linkage will not solve at any usable rack travel: " + reach.error());
    }

    if (reach.value() < roadWheelAngle)
    {
        return std::unexpected("the steering linkage cannot reach the stated lock within a rack's travel");
    }

    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto steer = steerAt(middle);
        if (!steer)
        {
            return std::unexpected(steer.error());
        }

        if (steer.value() < roadWheelAngle)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }

    const auto magnitude = 0.5 * (low + high);

    return steersRightForPositiveTravel ? magnitude : -magnitude;
}

// The parallel-axis contribution of a point mass offset from the axis it is being measured about.
[[nodiscard]] glm::dmat3 offsetInertia(const double mass, const glm::dvec3& offset)
{
    auto contribution = glm::dmat3(0.0);
    contribution[0][0] = mass * (offset.y * offset.y + offset.z * offset.z);
    contribution[1][1] = mass * (offset.x * offset.x + offset.z * offset.z);
    contribution[2][2] = mass * (offset.x * offset.x + offset.y * offset.y);

    return contribution;
}

} // namespace

// The Golf GTI Mk7.5, assembled. Fallible for the same reasons `placeholderSedan` is: every corner's
// geometry is swept and its setup validated before the car is handed back, so a bad number is an
// error at load time rather than a car that flies away on the tick that first reaches it.
export [[nodiscard]] std::expected<VehicleSetup, std::string> golfGtiMk7()
{
    auto setup = VehicleSetup{};

    setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints = golfMk7FrontCorner(CornerSide::Left);
    setup.corners[static_cast<std::size_t>(Corner::FrontRight)].hardpoints = golfMk7FrontCorner(CornerSide::Right);
    setup.corners[static_cast<std::size_t>(Corner::RearLeft)].hardpoints = golfMk7RearCorner(CornerSide::Left);
    setup.corners[static_cast<std::size_t>(Corner::RearRight)].hardpoints = golfMk7RearCorner(CornerSide::Right);

    // --- the mass ledger ---
    //
    // TOTALMASS is the whole car and HUB_MASS is per corner, so the sprung mass is what is left. The
    // hub masses are high for a hatchback — 330 kg of unsprung against 1348 kg all in — and that is
    // the file's figure rather than a reading of it.
    const auto unsprung = 2.0 * golfFrontHubMass + 2.0 * golfRearHubMass;
    const auto sprungMass = golfTotalMass - unsprung;

    // Where the centre of gravity is. **Neither coordinate is stated cleanly by AC and both are
    // marked**: the height comes from car.ini's GRAPHICS_OFFSET, which places the graphics origin
    // relative to the physics body and so says how far the centre of gravity sits above the road;
    // the longitudinal position comes from suspensions.ini's CG_LOCATION read as the front weight
    // fraction. See the note in the tests for what the alternatives were and why these two won.
    const auto centreHeight = 0.572;
    const auto frontFraction = 0.53;
    const auto centreStation = golfWheelbase * (frontFraction - 0.5);
    const auto centre = glm::dvec3(0.0, centreHeight, centreStation);

    // The unsprung masses ride at their own wheel centres, so the sprung centre is whatever puts the
    // *assembled* car's centre of gravity where the data says. Solved rather than authored, because
    // authoring it would be stating the same fact twice and letting the two disagree.
    const auto hubMass = [](const std::size_t index)
    {
        return index < 2 ? golfFrontHubMass : golfRearHubMass;
    };

    auto unsprungMoment = glm::dvec3(0.0);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        unsprungMoment += hubMass(index) * setup.corners[index].hardpoints.wheelCentre;
    }

    const auto sprungCentre = (golfTotalMass * centre - unsprungMoment) / sprungMass;

    // car.ini's INERTIA is the box whose inertia the whole car has, stated as its three dimensions —
    // 1.54 x 1.452 x 4.27 m, which are the car's track, its height and its length. That is the
    // *assembled* car's tensor about its own centre of gravity, so the shell carries it less every
    // parallel-axis term the assembly puts back: four unsprung masses at their wheel centres and the
    // sprung mass at its own offset. Stating the box on the shell instead would over-state the whole
    // car's yaw inertia by a third.
    auto shell = glm::dmat3(0.0);
    {
        const auto box = glm::dvec3(1.54, 1.452, 4.27);
        const auto twelfth = golfTotalMass / 12.0;

        shell[0][0] = twelfth * (box.y * box.y + box.z * box.z); // pitch, about the lateral axis
        shell[1][1] = twelfth * (box.x * box.x + box.z * box.z); // yaw
        shell[2][2] = twelfth * (box.x * box.x + box.y * box.y); // roll, about the longitudinal axis

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            shell -= offsetInertia(hubMass(index), setup.corners[index].hardpoints.wheelCentre - centre);
        }

        shell -= offsetInertia(sprungMass, sprungCentre - centre);
    }

    setup.sprung = {MassComponent{.mass = sprungMass, .centre = sprungCentre, .inertia = shell}};

    // tyres.ini WIDTH. The patch's length is not stated by AC and stays the model's own; the sample
    // counts stay at three by three, which is part of this car's configuration rather than free to
    // change under it — the penetration is a quadrature and converges rather than being exact.
    setup.sampling.width = 0.235;

    // --- aero, from aero.ini ---
    //
    // Each wing's area is its CHORD by its SPAN and its coefficient is its own lookup table read at
    // zero angle of attack, times the GAIN beside it — so the body carries all the drag and none of
    // the lift and the two ends carry lift and no drag, which is what a road car is. AC's sign
    // convention is that a positive CL is *downforce*, so the rear's -0.124 is lift and comes across
    // with its sign turned over. Positions are AC's, which are relative to the centre of gravity.
    const auto wingArea = 1.03 * 2.07;
    const auto fromCentre = [&centre](const double x, const double y, const double z)
    {
        return centre + glm::dvec3(x, y, z);
    };

    setup.aero = {AeroSurface{.centre = fromCentre(0.0, 0.23, -0.40), .dragArea = 0.340 * wingArea, .liftArea = 0.0},
                  AeroSurface{.centre = fromCentre(0.0, -0.13, 1.10), .dragArea = 0.0, .liftArea = -0.00107 * wingArea},
                  AeroSurface{.centre = fromCentre(0.0, 0.86, -1.55), .dragArea = 0.0, .liftArea = 0.124 * wingArea}};

    // --- bodywork ---
    //
    // AC's only collider is a floor slab rather than a body box, so it gives the height the bodywork
    // starts at and nothing else: COLLIDER_0 is 0.1 m thick centred 0.20 m below the centre of
    // gravity, so its underside is the number taken. The length and height are the INERTIA box's,
    // which are the car's own; the width is the front track plus a tyre, since the track is measured
    // between the wheels' centre planes. Where along the car the box sits is not stated and it is
    // centred on the wheelbase.
    const auto floor = centreHeight - 0.20 - 0.05;
    const auto roof = 1.452;
    setup.body =
        CollisionBox{.centre = glm::dvec3(0.0, 0.5 * (floor + roof), 0.0),
                     .halfExtents = glm::dvec3(0.5 * (golfFrontTrack + 0.235), 0.5 * (roof - floor), 0.5 * 4.27)};

    // car.ini CONTROLS: the steering wheel's lock in degrees each way, and the ratio between it and
    // the road wheel. The travel that produces that angle is solved off the linkage below.
    //
    // **The ratio's sign is not decoration.** This car's steering arm sits behind the kingpin, so a
    // rack moving to the right turns it left — the opposite of the placeholder's front-steer
    // geometry, and the opposite of what every consumer of `VehicleInput::steering` assumes. AC
    // states the ratio negative, and honouring that sign is what puts a right-hand turn back on
    // positive steering.
    const auto steerLock = 378.0;
    const auto steerRatio = -14.1;
    const auto roadWheelLock = std::abs(steerLock / steerRatio) * (3.14159265358979323846 / 180.0);

    // suspensions.ini: rates at the wheel, dampers with a knee, and the anti-roll bars. The spring
    // and damper are carried onto the shaft by the motion ratio the linkage reports at design.
    const auto wheelRate = std::array{35000.0, 35000.0, 57000.0, 57000.0};
    const auto bumpRate = std::array{4600.0, 4600.0, 6200.0, 6200.0};
    const auto fastBumpRate = std::array{1834.0, 1834.0, 1842.0, 1842.0};
    const auto bumpKnee = std::array{0.070, 0.070, 0.100, 0.100};
    const auto reboundRate = std::array{5300.0, 5300.0, 6700.0, 6700.0};
    const auto fastReboundRate = std::array{2589.0, 2589.0, 2700.0, 2700.0};
    const auto reboundKnee = std::array{0.140, 0.140, 0.140, 0.140};
    const auto antiRoll = std::array{34000.0, 34000.0, 15000.0, 15000.0};

    // brakes.ini: one total torque split by FRONT_SHARE and then between the two wheels of an axle.
    const auto brakeTorque =
        std::array{4200.0 * 0.75 / 2.0, 4200.0 * 0.75 / 2.0, 4200.0 * 0.25 / 2.0, 4200.0 * 0.25 / 2.0};

    // Sprung load by statics about the other axle, from the sprung centre solved above.
    const auto frontSprung = sprungMass * standardGravity * (sprungCentre.z - golfRearAxle) / golfWheelbase;
    const auto rearSprung = sprungMass * standardGravity - frontSprung;
    const auto sprungLoad = std::array{frontSprung / 2.0, frontSprung / 2.0, rearSprung / 2.0, rearSprung / 2.0};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& corner = setup.corners[index];

        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        if (!design)
        {
            return std::unexpected("corner " + std::string(cornerAbbreviation(static_cast<Corner>(index))) + ": " +
                                   design.error());
        }

        const auto ratio = std::abs(design->motionRatio);
        corner.springRate = wheelRate[index] / (ratio * ratio);
        corner.damper = kneedDamper(bumpRate[index], fastBumpRate[index], bumpKnee[index], reboundRate[index],
                                    fastReboundRate[index], reboundKnee[index], ratio);
        corner.antiRollRate = antiRoll[index];
        corner.brakeTorque = brakeTorque[index];
        corner.unsprungMass = hubMass(index);

        // tyres.ini, the Semislicks compound — AC's own default for this car, taken whole so the
        // grip and the carcass describe the same tyre: carcass rate and damping, angular inertia,
        // and rolling resistance. AC's ROLLING_RESISTANCE_0 of 12 is not a load fraction and the
        // file does not carry the formula it goes into; read as one part in a thousand it comes to
        // 0.012, and the speed-squared term beside it adds under a tenth of that at motorway speed
        // and is not modelled here.
        corner.tireVerticalRate = 298926.0;
        corner.tireVerticalDamping = 500.0;
        corner.wheelInertia = 1.45;
        corner.rollingResistance = 0.012;

        // The tyre. Reference load, load sensitivity and both friction peaks are the file's:
        // DY_REF/DX_REF state the friction at FZ0, which is exactly this model's
        // peak-at-nominal-load convention, and AC states the force as going with load to the LS_EXP
        // power, so the exponent is that power less one with the sign turned over. The lateral
        // exponent serves both axes because this model carries one (the file's longitudinal 0.8756
        // is close). 1.28 lateral sits under the car's own rollover threshold — about 1.33 g for a
        // 0.572 m centre of gravity on this track width — so the handling cases still measure the
        // tyre and not the wrong failure.
        corner.tyre.nominalLoad = 2939.0;
        corner.tyre.loadSensitivity = 1.0 - 0.8074;
        corner.tyre.lateralPeak = 1.28;
        corner.tyre.longitudinalPeak = 1.30;

        // Stated on the shaft, and sized inside the travel the linkage has. The bump stops AC states
        // do not carry across: its front BUMPSTOP_UP of 0.80 m is not a travel any suspension has and
        // its rear BUMPSTOP_DN of 0 would have the droop stop touching at rest.
        corner.bumpStop = TravelStop{.gap = 0.020, .rate = 900000.0, .progression = 3.0, .damping = 40000.0};
        corner.droopStop = TravelStop{.gap = 0.020, .rate = 600000.0, .progression = 3.0, .damping = 30000.0};

        const auto restLength = springFreeLengthForLoad(corner, sprungLoad[index]);
        if (!restLength)
        {
            return std::unexpected(restLength.error());
        }

        corner.springFreeLength = restLength.value();

        if (const auto validated = validateCornerSetup(corner); !validated)
        {
            return std::unexpected("corner " + std::string(cornerAbbreviation(static_cast<Corner>(index))) + ": " +
                                   validated.error());
        }
    }

    // **The front left, and the side is load-bearing because of Ackermann.** A positive demand is a
    // right turn, so the left front is the *outer* wheel and Ackermann steers it less — this linkage
    // wants 70.0 mm of rack to put the outer wheel on the stated lock and 64.5 mm to put the inner
    // one there, an 8% difference in the car's whole steering ratio depending on which is asked.
    //
    // It reads the outer one because that is the wheel this car has always been derived from and
    // driven with: before the corner sides were un-mirrored (`outboardSign`) this line said
    // `corners[1]` and that index *was* the left front. Which of the two AC's single STEER_RATIO is
    // quoted against is a real question and it is not this change's to answer — un-mirroring the
    // labels must not quietly re-gear the car by eight percent.
    const auto rackTravel =
        rackTravelForSteer(setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints, roadWheelLock);
    if (!rackTravel)
    {
        return std::unexpected(rackTravel.error());
    }

    // Signed by the linkage, not by the ratio. `steerRatio` says how many degrees of rim make a
    // degree of road wheel and its sign is a convention about which the rack knows nothing.
    setup.rackTravelPerInput = rackTravel.value();

    return setup;
}

// The same car's driveline, from engine.ini, drivetrain.ini and the turbocharged torque curve the
// extractor computed off power.lut.
//
// **power.lut is torque in newton metres and it is the naturally aspirated curve**, which AC layers
// boost onto per turbo as min(wastegate, maxBoost * (rpm/reference)^gamma). Read as final it reports
// 159 N.m and 110 bhp for a car that makes 350 and 243 — a wrong number that looks like a right one,
// which is the only kind worth guarding against. The points below are the boosted curve.
export [[nodiscard]] DrivelineSetup golfGtiMk7Driveline()
{
    auto setup = DrivelineSetup{};

    const auto atRpm = [](const double rpm, const double torque)
    {
        return glm::dvec2(rpm * rpmToRadiansPerSecond, torque);
    };

    setup.engine.torque =
        Curve{.points = {atRpm(0.0, 95.0), atRpm(500.0, 129.92491), atRpm(1000.0, 146.745119), atRpm(1500.0, 193.70088),
                         atRpm(1768.0, 235.290486), atRpm(1904.0, 260.621688), atRpm(2000.0, 272.8),
                         atRpm(2500.0, 316.8), atRpm(2650.0, 325.6), atRpm(3000.0, 338.8), atRpm(3500.0, 334.4),
                         atRpm(4000.0, 334.4), atRpm(4500.0, 349.8), atRpm(5000.0, 343.2), atRpm(5500.0, 314.6),
                         atRpm(6300.0, 272.8), atRpm(6500.0, 264.0), atRpm(6800.0, 248.16)}};

    setup.engine.inertia = 0.150;
    setup.engine.idleSpeed = 850.0 * rpmToRadiansPerSecond;
    setup.engine.limiterSpeed = 6800.0 * rpmToRadiansPerSecond;
    // engine.ini [COAST_REF]: what it absorbs at the limiter with the throttle shut.
    setup.engine.coastTorque = 75.0;

    // drivetrain.ini [GEARS]. Seven forward, and reverse as its magnitude — `reduction` puts the sign
    // on for it, so carrying GEAR_R's own -2.9 through would give a reverse gear that drives forward.
    setup.gearbox.ratios = {3.19, 2.08, 1.47, 1.20, 0.99, 0.80, 0.65};
    setup.gearbox.finalDrive = 4.37;
    setup.gearbox.reverseRatio = 2.9;

    setup.coupling.kind = DriveCouplingKind::FrictionClutch;

    // [TRACTION] TYPE=FWD, and [DIFFERENTIAL] POWER/COAST/PRELOAD. AC's two ramp numbers are the
    // fraction of the torque going through the diff that becomes locking torque on power and off it,
    // which is exactly what `clutchPackLsd` takes — so this fits the interface as it stands and needs
    // nothing built. A preload of zero does not make it open: it still locks with a quarter of
    // whatever is being put through it.
    setup.driven = DrivenAxle::Front;
    setup.differential = clutchPackLsd(0.0, 0.25, 0.25);

    return setup;
}

} // namespace raceengine
