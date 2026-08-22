// PublishedCars bodies. Declarations are in Api/PublishedCars.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] CornerHardpoints golfMk7FrontCorner(const CornerSide side)
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

[[nodiscard]] CornerHardpoints golfMk7RearCorner(const CornerSide side)
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

[[nodiscard]] std::expected<VehicleSetup, std::string> golfGtiMk7()
{
    auto setup = VehicleSetup{};

    setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints = golfMk7FrontCorner(CornerSide::Left);
    setup.corners[static_cast<std::size_t>(Corner::FrontRight)].hardpoints = golfMk7FrontCorner(CornerSide::Right);
    setup.corners[static_cast<std::size_t>(Corner::RearLeft)].hardpoints = golfMk7RearCorner(CornerSide::Left);
    setup.corners[static_cast<std::size_t>(Corner::RearRight)].hardpoints = golfMk7RearCorner(CornerSide::Right);

    // --- the mass ledger ---
    //
    // TOTALMASS is the whole car and the corner masses are per corner, so the sprung mass is what is
    // left.
    //
    // **Everything below this line re-derives itself when the split changes, and that is the point.**
    // Correcting the unsprung mass was specified as a four-step job — set the corner figure,
    // recompute sprung mass to preserve the total, recompute the sprung centre of gravity to preserve
    // the *assembled* centre, recompute the inertia tensor from the corrected distribution — and only
    // the first step is an edit here. The other three were already solved rather than authored, for
    // the reason given at each: stating the same fact twice is how two statements of it come to
    // disagree. So `sprungMass` is a subtraction, `sprungCentre` is solved from the assembled centre
    // the data states, and `shell` is the assembled tensor less every parallel-axis term the assembly
    // puts back. Change a corner mass and all three follow, and `GolfGtiTests` asserts that the
    // assembled car still has the centre and the tensor the file states.
    const auto unsprung = 2.0 * golfFrontHubMass + 2.0 * golfRearHubMass;
    const auto sprungMass = golfTotalMass - unsprung;

    // Where the centre of gravity is. The height comes from car.ini's GRAPHICS_OFFSET, which places
    // the graphics origin relative to the physics body and so says how far the centre of gravity
    // sits above the road. That one is the mod's and is plausible: 0.572 m against a typical
    // C-segment hatch's 0.55.
    const auto centreHeight = 0.572;

    // **The longitudinal position is the third number taken away from the mod** (2026-08-22).
    //
    // suspensions.ini states `CG_LOCATION=0.53 ; Front Weight distribution in percentance` — the
    // file names its own convention, so `scripts/import-ac-car.py` reads it correctly and the source
    // is what disagrees. A transverse-engined front-drive hatchback does not sit at 53% front, and a
    // DSG Mk7 GTI measures **61.4/38.6**. At 1348 kg that is 114 kg per axle, 8.5% of the car.
    //
    // **It is a bigger change than the radius or the unsprung mass and it is a different kind.**
    // Those two corrected levers — a scale factor on speed and gearing, a split that preserved every
    // total. This moves the centre of gravity 222 mm forward, which moves the car's *balance*: front
    // tyre load, understeer, the aligning moment, and therefore where the steering limit falls in
    // rack torque. Nothing downstream is unaffected. It is here because the mod disagreed with the
    // published car three times out of three and Dominic's call was to source the chassis from the
    // specification rather than keep patching it.
    const auto frontFraction = 0.614;
    const auto centreStation = golfWheelbase * (frontFraction - 0.5);
    const auto centre = glm::dvec3(0.0, centreHeight, centreStation);

    // **And AC's own frame origin, which is not the same point any more.**
    //
    // Everything AC states positionally — the aero surfaces, the collider — it states relative to
    // *its* centre of gravity, and its centre of gravity is the one at 53%. A wing is bolted to the
    // bodywork: correcting where the car balances does not slide the radiator inlet 222 mm up the
    // chassis. So AC-relative positions are carried across from where AC put its origin, and only
    // the mass ledger uses the corrected centre.
    //
    // Getting this wrong is invisible in every total — the aero would still integrate to the same
    // drag — and would show up only as a pitching moment nobody authored, which is exactly the class
    // of fault this file keeps being bitten by.
    const auto acCentreStation = golfWheelbase * (0.53 - 0.5);
    const auto acOrigin = glm::dvec3(0.0, centreHeight, acCentreStation);

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
    // with its sign turned over.
    //
    // **Positions are AC's, relative to AC's origin** — see `acOrigin` above. They were relative to
    // `centre` until the weight distribution was corrected, and the two were the same point until
    // then, which is why this reads as a change to nothing.
    const auto wingArea = 1.03 * 2.07;
    const auto fromCentre = [&acOrigin](const double x, const double y, const double z)
    {
        return acOrigin + glm::dvec3(x, y, z);
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

    // Where a ride height for this car is quoted from. **Not in AC's data**: its only collider is a
    // coarse body shell whose underside is 0.32 m up, which is a real surface of the car and not the
    // one anybody means. A Mk7 GTI is quoted at about 135 mm unladen, so that is the design figure
    // taken here — a placeholder in the same sense as every other number without a source.
    setup.rideHeightReference = 0.135;

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

        // **The fourth number taken away from the mod** (2026-08-22), after the wheel radius, the
        // unsprung mass and `CG_LOCATION`. The file states DY_REF 1.28 and DX_REF 1.30 for its
        // Semislicks — and **1.23/1.26 for a compound it calls "Street"**, which is not a street tyre:
        // a 225/40 R18 performance road tyre peaks nearer 1.0 to 1.15. Both of its compounds are
        // track-tyre numbers.
        //
        // Scaled by 0.87, which is **derived rather than chosen**, and the derivation is the point:
        //
        //   1. Criteria 6 and 7 were fixed from physical reasoning first, and they can be because
        //      neither is a grip figure — 6 is response shape and 7 is a ratio between two runs that
        //      share their grip. Measured across a 20% grip cut, 6's four channels move under 1.5%
        //      and 7's ratio moves 0.9%. They then hold at whatever grip is chosen.
        //   2. Criterion 5's threshold comes from the real car: a Mk7 GTI on OEM tyres skidpads at
        //      **0.90 to 0.95 g**. Grip is the free parameter that lands it, so nothing about the
        //      model may set it.
        //   3. 0.87 puts criterion 5 at **0.9230 g**, mid-band.
        //   4. **0-100 is then checked independently** on the launch fixture against the published
        //      6.4-6.7 s for a DSG without launch control — a different measurement against a
        //      different external reference, with no shared parameter but this one.
        //
        // Steps 3 and 4 agreeing is the validation. Neither number was used to set the other.
        //
        // Applied to the peaks and **not to `gripScale`**: that is the runtime seam a thermal,
        // pressure or wear model multiplies through, and spending it on a static data correction
        // would leave those models nowhere to act.
        //
        // **Driven and accepted, 2026-08-22** — Dominic's verdict was that it feels okay. That is a
        // pass on the car's character with 13% less grip than the mod stated, which is what the
        // derivation could not tell us.
        corner.tyre.lateralPeak = 1.114;
        corner.tyre.longitudinalPeak = 1.131;

        // **The longitudinal curve's shape, and these three are fitted to target behaviour rather
        // than sourced.** Labelled that way deliberately: `tyres.ini` carries `DX_REF` and nothing
        // whatever about fall-off, so claiming provenance for a shape factor would be inventing it.
        //
        // What was here were the model's defaults — C = 1.65, K = 20, E = -1.0 — and they were never
        // calibrated against anything. A Magic Formula's sliding force tends to `D * sin(C * pi/2)`
        // as slip grows, so C alone caps it: 1.65 gives sin(148.5 deg) = **52.2% of peak**, measured
        // at 54.8% by slip ratio 3. A standing start on this car reaches slip ratios of 2.5 to 3.8,
        // so it was spending the whole launch on half the grip it had. **The lateral curve's
        // C = 1.35 gives 85% and is healthy, which is why nothing caught it** — every validation this
        // project runs is lateral, and the longitudinal curve had never been checked against
        // anything at all.
        //
        // **Judged at a slip ratio of 1, not at 3**, and that correction matters more than the
        // numbers it produced. A tyre rig runs to about 0.2-0.3, and to 1.0 for locked-wheel work;
        // nobody measures one at 500% slip. Past that the formula is pure extrapolation and its
        // asymptote is a property of the functional form rather than of any tyre — which is how a
        // published C_x of 1.65 coexists with real tyres that plainly do not lose half their grip
        // when they spin. The 70-80%-of-peak figure this is aimed at is locked-wheel data, so slip 1
        // is where it is read: **77.7% here against 60.2% as shipped.**
        //
        // **K moves with C and E and is not free.** B is `K / (C * mu)`, so both shift where the peak
        // sits and K is what holds it at 0.120, where a semislick's belongs. 28 does that and stays
        // inside the 10-30 a road tyre runs. It is also what rules the alternatives out: E = +0.90 at
        // C = 1.50 wants K = 81, and C = 1.45 with E = 0.95 wants 120, against the low forties of a
        // race construction. A curve needing those is not this tyre.
        //
        // E = 0 rather than the +0.50 first tried. Holding the peak makes the two nearly equivalent
        // where the data lives — 0.777 against 0.765 at slip 1 — so the choice is near-peak breadth:
        // +0.50 makes 91.5% of peak at half the peak slip against this curve's 87.7%, and a semislick
        // should be the peakier of the two. Chosen on that rather than as a by-product of the K
        // constraint, and E = 0 needs no defending either way.
        //
        // Swept in `[.tyre-shape]`; **driven and accepted 2026-08-22**, in the same session as the
        // grip correction below — the two went to the seat together and cannot be separated by it.
        corner.tyre.longitudinalShape = 1.50;
        corner.tyre.longitudinalCurvature = 0.0;
        corner.tyre.longitudinalStiffness = 28.0;

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

[[nodiscard]] DrivelineSetup golfGtiMk7Driveline()
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
