// Vehicle bodies. Declarations are in Api/Vehicle.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.physics;

namespace raceengine
{

void seedDiscTemperatures(VehicleState& state, const double celsius)
{
    for (auto& corner : state.corners)
    {
        corner.discTemperature = celsius;
        corner.wheelTemperature = celsius;
    }
}

void seedTyreTemperatures(VehicleState& state, const double celsius)
{
    for (auto& corner : state.corners)
    {
        seedTyreTemperature(corner.tyre, celsius);
    }
}

[[nodiscard]] Curve linearDamper(const double bumpRate, const double reboundRate)
{
    return Curve{
        .points = {glm::dvec2(-5.0, -5.0 * reboundRate), glm::dvec2(0.0, 0.0), glm::dvec2(5.0, 5.0 * bumpRate)}};
}

[[nodiscard]] Curve kneedDamper(const double bumpRate, const double fastBumpRate, const double bumpKnee,
                                const double reboundRate, const double fastReboundRate, const double reboundKnee,
                                const DamperKinematics& kinematics)
{
    const auto ratio = std::abs(kinematics.motionRatio);
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

[[nodiscard]] std::array<double, cornerCount> wheelInertias(const VehicleSetup& setup)
{
    auto inertias = std::array<double, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        inertias[index] = setup.corners[index].wheelInertia;
    }

    return inertias;
}

[[nodiscard]] std::array<double, cornerCount> brakeCircuitPressures(const VehicleSetup& setup, const double pedal)
{
    const auto master = brakeLinePressure(setup.brakeHydraulics, pedal);
    const auto rear = proportionedPressure(setup.rearBrakeValve, master);

    auto pressures = std::array<double, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        pressures[index] = rearAxle(static_cast<Corner>(index)) ? rear : master;
    }

    return pressures;
}

[[nodiscard]] double brakePedalResponse(const VehicleSetup& setup, const std::size_t corner, const double pedal)
{
    const auto full = brakeCircuitPressures(setup, 1.0);

    if (corner >= cornerCount || full[corner] <= 0.0)
    {
        return 0.0;
    }

    return brakeCircuitPressures(setup, pedal)[corner] / full[corner];
}

[[nodiscard]] std::array<double, cornerCount> roadTorques(const VehicleStep& step)
{
    auto torques = std::array<double, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        torques[index] = -step.corners[index].contact.tyre.longitudinal * step.corners[index].contact.effectiveRadius;
    }

    return torques;
}

[[nodiscard]] std::expected<double, std::string> springFreeLengthForLoad(const CornerSetup& corner,
                                                                         const double sprungLoad)
{
    const auto spring = solveSpringKinematics(corner.hardpoints, springElementOf(corner.hardpoints), 0.0, 0.0);
    if (!spring)
    {
        return std::unexpected(spring.error());
    }

    if (std::abs(spring->motionRatio) < 1e-9 || corner.springRate <= 0.0)
    {
        return std::unexpected("a corner with no motion ratio or no spring cannot be given a rest length");
    }

    // At equilibrium the spring's force along its own axis, resolved to the wheel by the spring's
    // motion ratio, carries the sprung corner load.
    return spring->length + sprungLoad / (corner.springRate * std::abs(spring->motionRatio));
}

[[nodiscard]] SpringSolution solveSpringForce(const CornerSetup& corner, const SuspensionState& suspension)
{
    const auto evaluated =
        solveElement(corner.hardpoints, springElementOf(corner.hardpoints), suspension.wishboneAngle);

    return SpringSolution{.length = evaluated.length,
                          .lengthPerAngle = evaluated.lengthPerAngle,
                          .force = corner.springRate * (corner.springFreeLength - evaluated.length)};
}

[[nodiscard]] DamperSolution solveDamperGeometry(const CornerSetup& corner, const SuspensionState& suspension)
{
    const auto evaluated =
        solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), suspension.wishboneAngle);

    return DamperSolution{.length = evaluated.length, .lengthPerAngle = evaluated.lengthPerAngle};
}

[[nodiscard]] DamperForceSolution solveDamperForce(const CornerSetup& corner, const SuspensionState& suspension,
                                                   const double wishboneRate)
{
    const auto geometry = solveDamperGeometry(corner, suspension);
    const auto velocity = -geometry.lengthPerAngle * wishboneRate;
    const auto viscous = corner.damper.at(velocity);

    return DamperForceSolution{.length = geometry.length,
                               .lengthPerAngle = geometry.lengthPerAngle,
                               .velocity = velocity,
                               // Seal and rod friction on top of the curve, opposing the shaft at any non-zero velocity
                               // and not scaling with it. `tanh` and not `sign` deliberately — a hard sign term at this
                               // tick rate makes a limit cycle rather than a dead band, and the width is a numerical
                               // choice stated as one (`CornerSetup::damperFrictionSpeed`).
                               //
                               // Branched rather than added, so a car with no friction stated runs the expression it
                               // always ran and not one that happens to add a zero to it. Every car here states none.
                               .force =
                                   corner.damperFriction > 0.0
                                       ? viscous + corner.damperFriction *
                                                       std::tanh(velocity / std::max(corner.damperFrictionSpeed, 1e-9))
                                       : viscous};
}

[[nodiscard]] double damperShaftCompression(const CornerSetup& corner, const DamperForceSolution& damper)
{
    const auto design = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0);

    return design.length - damper.length;
}

[[nodiscard]] SteeringLimitLoads steeringLimitLoad(const VehicleSetup& setup)
{
    auto mass = 0.0;
    auto heightMoment = 0.0;
    auto stationMoment = 0.0;

    for (const auto& component : setup.sprung)
    {
        mass += component.mass;
        heightMoment += component.mass * component.centre.y;
        stationMoment += component.mass * component.centre.z;
    }

    for (const auto& corner : setup.corners)
    {
        mass += corner.unsprungMass;
        heightMoment += corner.unsprungMass * corner.hardpoints.wheelCentre.y;
        stationMoment += corner.unsprungMass * corner.hardpoints.wheelCentre.z;
    }

    if (mass <= 0.0)
    {
        return SteeringLimitLoads{};
    }

    const auto centreHeight = heightMoment / mass;
    const auto centreStation = stationMoment / mass;

    const auto frontStation = setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints.wheelCentre.z;
    const auto rearStation = setup.corners[static_cast<std::size_t>(Corner::RearLeft)].hardpoints.wheelCentre.z;
    const auto wheelbase = frontStation - rearStation;

    const auto track =
        2.0 * std::abs(setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints.wheelCentre.x);

    if (std::abs(wheelbase) < 1e-9 || track < 1e-9)
    {
        return SteeringLimitLoads{};
    }

    const auto frontMass = mass * (centreStation - rearStation) / wheelbase;
    const auto staticLoad = frontMass * earthGravity / 2.0;

    const auto& tyre = setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].tyre;

    auto load = staticLoad;
    for (auto pass = 0; pass < 6; pass++)
    {
        const auto friction = tyreFriction(tyre, TyreAxis::Lateral, load, 1.0);
        load = staticLoad + friction * frontMass * earthGravity * centreHeight / track;
    }

    // The inside wheel gets what the outside did not take, floored at zero rather than allowed
    // negative: a car whose transfer lifts the inside wheel has an inside tyre carrying nothing,
    // which is the physical answer and not a clamp hiding one.
    return SteeringLimitLoads{
        .staticPerWheel = staticLoad, .outside = load, .inside = std::max(2.0 * staticLoad - load, 0.0)};
}

[[nodiscard]] CornerHardpoints placeholderCorner(const CornerSide side, const double axleZ)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror, axleZ](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * x, y, z + axleZ);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;

    corner.lower = Wishbone{
        .frontPivot = at(0.30, 0.13, 0.15), .rearPivot = at(0.30, 0.13, -0.15), .ballJoint = at(0.62, 0.12, 0.0)};
    corner.upper = Wishbone{
        .frontPivot = at(0.35, 0.36, 0.12), .rearPivot = at(0.35, 0.36, -0.12), .ballJoint = at(0.58, 0.42, 0.0)};

    corner.damperChassis = at(0.32, 0.55, 0.0);
    corner.damperWishbone = at(0.50, 0.14, 0.0);
    corner.antiRollBarChassis = at(0.45, 0.30, 0.25);
    corner.antiRollBarWishbone = at(0.45, 0.13, 0.25);
    corner.steeringRackOuter = at(0.30, 0.16, 0.16);
    corner.steeringArm = at(0.60, 0.16, 0.14);

    // Wheel centre one radius up, so the design contact patch sits exactly on y = 0 and every
    // height in the setup is measured from the road rather than from an arbitrary datum.
    corner.wheelCentre = at(0.72, 0.30, 0.0);
    corner.wheelRadius = 0.30;

    corner.droopAngle = -0.26;
    corner.bumpAngle = 0.26;

    return corner;
}

[[nodiscard]] std::expected<void, std::string> validateCornerSetup(const CornerSetup& corner)
{
    if (const auto geometry = validateCorner(corner.hardpoints); !geometry)
    {
        return std::unexpected(geometry.error());
    }

    // A damper whose two ends coincide is not a damper, and the arithmetic downstream does not say
    // so: the "length" of a zero-length damper still varies as the wishbone swings its end about,
    // so a motion ratio comes out, and it is a plausible-looking number describing nothing.
    if (corner.hardpoints.kind != SuspensionKind::MacPhersonStrut &&
        glm::distance(corner.hardpoints.damperChassis, corner.hardpoints.damperWishbone) < 1e-6)
    {
        return std::unexpected("the damper's two mounts are the same point, so it has no axis; a corner "
                               "imported from data that states its spring rate at the wheel needs damper "
                               "hardpoints supplied or a direct-acting rate declared");
    }

    // The motion ratio must not change sign across the travel. A real linkage compresses its damper
    // monotonically as the wheel rises; one that reverses would have the spring pushing the wrong way
    // over half its range. This is what catches a damper attached to the wrong thing — the geometry
    // still solves, the curves still look like curves, and the ratio quietly passes through zero.
    // Since step 14 the ratio is the damper element's, read at the same forty-one positions the
    // sweep used to supply — the same bits, so the refusals print the numbers they always did.
    const auto damperElement = damperElementOf(corner.hardpoints);
    auto sign = 0.0;
    for (auto index = 0; index < 41; index++)
    {
        const auto through = static_cast<double>(index) / 40.0;
        const auto angle =
            corner.hardpoints.droopAngle + through * (corner.hardpoints.bumpAngle - corner.hardpoints.droopAngle);

        const auto kinematics = solveDamperKinematics(corner.hardpoints, damperElement, angle, 0.0);
        if (!kinematics)
        {
            return std::unexpected(kinematics.error());
        }

        if (std::abs(kinematics->motionRatio) < 1e-9)
        {
            continue;
        }

        const auto here = kinematics->motionRatio < 0.0 ? -1.0 : 1.0;
        if (sign != 0.0 && here != sign)
        {
            return std::unexpected("the motion ratio changes sign across the travel, so the damper "
                                   "reverses direction relative to the wheel; its mounts are almost "
                                   "certainly not where the linkage thinks they are");
        }

        sign = here;

        if (std::abs(kinematics->motionRatio) > 2.0)
        {
            return std::unexpected("the motion ratio reaches " + std::to_string(kinematics->motionRatio) +
                                   ", so the damper is being asked to move twice as far as the wheel");
        }
    }

    // The design and end lengths off the damper element — the same bits the full solves used to
    // produce (step 14), and the subtraction order the messages were always built from. The travel
    // itself is already proven solvable by `validateCorner` above, and element evaluation is
    // closed form, so there is nothing left here that can fail to solve.
    const auto element = damperElementOf(corner.hardpoints);
    const auto designLength = solveElement(corner.hardpoints, element, 0.0).length;
    const auto compression =
        designLength - solveElement(corner.hardpoints, element, corner.hardpoints.bumpAngle).length;
    const auto extension = solveElement(corner.hardpoints, element, corner.hardpoints.droopAngle).length - designLength;

    if (corner.bumpStop.gap >= compression)
    {
        return std::unexpected("the bump stop's gap of " + std::to_string(corner.bumpStop.gap) +
                               " m never closes: the damper only compresses " + std::to_string(compression) +
                               " m before the linkage reaches its limit");
    }

    if (corner.droopStop.gap >= extension)
    {
        return std::unexpected("the droop stop's gap of " + std::to_string(corner.droopStop.gap) +
                               " m never closes: the damper only extends " + std::to_string(extension) +
                               " m before the linkage reaches its limit");
    }

    // A drop link that is stated has to be a drop link. A corner that states none is not checked and
    // is not a mistake — most cars here state none, because AC's data has no bar geometry at all —
    // but one that states half of it, with the chassis end left on the origin, produces a link that
    // sweeps most of a metre for a few millimetres of wheel and a bar rate referred through it that is
    // nonsense. The force path falls back rather than dividing by it, so without this the mistake is
    // silent.
    if (dropLinkStated(corner.hardpoints))
    {
        const auto link = dropLinkElementOf(corner.hardpoints);
        const auto kinematics = solveSpringKinematics(corner.hardpoints, link, 0.0, 0.0);
        if (!kinematics)
        {
            return std::unexpected("the anti-roll bar's drop link cannot be evaluated: " + kinematics.error());
        }

        const auto ratio = std::abs(kinematics->motionRatio);
        if (ratio < 0.05 || ratio > 2.0)
        {
            return std::unexpected("the anti-roll bar's drop link has a motion ratio of " +
                                   std::to_string(kinematics->motionRatio) +
                                   ", so it barely moves with the wheel or moves twice as far as it; its two "
                                   "hardpoints are almost certainly not where the linkage thinks they are");
        }
    }

    return {};
}

[[nodiscard]] std::expected<VehicleSetup, std::string> placeholderSedan()
{
    constexpr auto frontAxle = 1.35;
    constexpr auto rearAxle = -1.35;
    constexpr auto sprungMass = 1200.0;
    // Placed so that the whole car, unsprung included, sits 60% on the front axle.
    constexpr auto sprungCentre = 0.3042;

    auto setup = VehicleSetup{};

    // Placeholder: a 1200 kg body's inertia about its own centre, about what a 4.5 x 1.8 x 1.4 m
    // shell of that mass comes to. Diagonal in body axes, which is what a symmetric car very nearly
    // is; the products of inertia a real one has are small enough to be somebody else's milestone.
    // The body frame is +x lateral, +y up, +z forward, so the long axis is z and not x: the roll
    // term belongs on [2][2]. Written the other way round the car is nearly four times too
    // roll-resistant and as much too willing to pitch, which distorts only the transients — yaw is
    // on the right axis either way and steady-state cornering never asks.
    auto shell = glm::dmat3(0.0);
    shell[0][0] = 1900.0; // pitch, about the lateral axis
    shell[1][1] = 2900.0; // yaw
    shell[2][2] = 500.0;  // roll, about the longitudinal axis

    setup.sprung = {MassComponent{.mass = sprungMass, .centre = glm::dvec3(0.0, 0.52, sprungCentre), .inertia = shell}};

    setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints =
        placeholderCorner(CornerSide::Left, frontAxle);
    setup.corners[static_cast<std::size_t>(Corner::FrontRight)].hardpoints =
        placeholderCorner(CornerSide::Right, frontAxle);
    setup.corners[static_cast<std::size_t>(Corner::RearLeft)].hardpoints =
        placeholderCorner(CornerSide::Left, rearAxle);
    setup.corners[static_cast<std::size_t>(Corner::RearRight)].hardpoints =
        placeholderCorner(CornerSide::Right, rearAxle);

    // **Negative, and the sign belongs to this linkage rather than to the field's default.** Rack
    // travel is applied toward +x on both corners, so it is outboard on one and inboard on the
    // other, and which way that steers the car depends on where the steering arm sits relative to
    // the kingpin. For this geometry a positive demand — which every input path produces for a
    // right turn — needs the rack going negative. The real car derives the same fact from its own
    // hardpoints in `rackTravelForSteer`; a fixture states it, and states it here rather than
    // leaning on a struct default that cannot know which linkage it is about to describe.
    setup.rackTravelPerInput = -0.055;

    // Sprung load on each axle, by statics about the other one. The springs are then sized from
    // those loads rather than from a guess at them.
    const auto wheelbase = frontAxle - rearAxle;
    const auto frontSprung = sprungMass * earthGravity * (sprungCentre - rearAxle) / wheelbase;
    const auto rearSprung = sprungMass * earthGravity - frontSprung;

    // One surface, at the centre of pressure a sedan's body has — a little behind the centre of
    // gravity, which is what makes a road car stable in a straight line at speed.
    setup.aero = {AeroSurface{.centre = glm::dvec3(0.0, 0.55, -0.20), .dragArea = 0.66, .liftArea = 0.10}};

    // Bodywork: 4.3 m by 1.7 m, its floor 0.15 m off the ground at design ride height, so the tires
    // carry the car and the box only ever meets something the car has actually hit.
    setup.body = CollisionBox{.centre = glm::dvec3(0.0, 0.72, 0.0), .halfExtents = glm::dvec3(0.85, 0.57, 2.15)};

    const auto rates = std::array{73000.0, 73000.0, 54000.0, 54000.0};
    const auto bump = std::array{7000.0, 7000.0, 5600.0, 5600.0};
    const auto rebound = std::array{12000.0, 12000.0, 9600.0, 9600.0};
    const auto loads = std::array{frontSprung / 2.0, frontSprung / 2.0, rearSprung / 2.0, rearSprung / 2.0};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& corner = setup.corners[index];
        corner.springRate = rates[index];
        corner.damper = linearDamper(bump[index], rebound[index]);

        // Stated on the damper shaft, where a real bump stop lives, and sized to engage *inside*
        // the linkage's own range: this geometry gives about 31 mm of shaft compression and 29 mm
        // of extension, so a gap larger than that is a stop that never touches. `validateSetup`
        // below is what turns that from a silent flying car into an error at load time.
        corner.bumpStop = TravelStop{.gap = 0.020, .rate = 900000.0, .progression = 3.0, .damping = 40000.0};
        corner.droopStop = TravelStop{.gap = 0.020, .rate = 600000.0, .progression = 3.0, .damping = 30000.0};

        const auto restLength = springFreeLengthForLoad(corner, loads[index]);
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

    return setup;
}

namespace
{

// The two corners of each axle, for the anti-roll bar — which is the one element that cannot be
// computed from a corner on its own, because what it resists is the *difference* across an axle.
constexpr std::array<std::size_t, cornerCount> acrossAxle = {1, 0, 3, 2};

[[nodiscard]] double curveSlopeAt(const Curve& curve, const double x)
{
    constexpr auto step = 1e-4;

    return (curve.at(x + step) - curve.at(x - step)) / (2.0 * step);
}

} // namespace

[[nodiscard]] double damperDampingCoefficient(const CornerSetup& corner, const DamperForceSolution& damper)
{
    // The exact expression the force pass always used, with the Jacobian and the velocity now the
    // damper element's own. Same factors in the same order: the element's values are the state's
    // bit for bit, so the coefficient is too.
    if (corner.damperFriction <= 0.0)
    {
        return damper.lengthPerAngle * damper.lengthPerAngle *
               std::max(0.0, curveSlopeAt(corner.damper, damper.velocity));
    }

    // Friction has to reach here or the implicit integration fights it. Its slope is
    // `d/dv [f · tanh(v/s)] = (f/s)·sech²(v/s)`, written analytically rather than differenced because
    // the curve's own differencing step is a hundred times the smoothing width and would read the
    // whole term as flat — and the slope at zero, which is `f/s`, is the largest number in this
    // expression and the entire reason the term needs solving rather than stepping towards.
    const auto smoothing = std::max(corner.damperFrictionSpeed, 1e-9);
    const auto shaped = std::tanh(damper.velocity / smoothing);
    const auto frictionSlope = (corner.damperFriction / smoothing) * (1.0 - shaped * shaped);

    return damper.lengthPerAngle * damper.lengthPerAngle *
           std::max(0.0, curveSlopeAt(corner.damper, damper.velocity) + frictionSlope);
}

namespace
{

// The drop link's motion ratio at the corner's design position, dLinkLength/dWheelTravel. A property
// of the hardpoints alone, so it is the constant the authored wheel rate is referred through — and
// evaluating it at design rather than at the tick's own travel is what keeps `antiRollRate` meaning
// what it says. Zero where the corner does not move vertically, which the caller treats as "no bar".
[[nodiscard]] double dropLinkDesignRatio(const CornerHardpoints& hardpoints)
{
    const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
    if (!design || std::abs(design->travelPerAngle) < 1e-9)
    {
        return 0.0;
    }

    return solveElement(hardpoints, dropLinkElementOf(hardpoints), 0.0).lengthPerAngle / design->travelPerAngle;
}

} // namespace

[[nodiscard]] AntiRollBarSolution solveAntiRollBar(const CornerSetup& corner, const SuspensionState& suspension,
                                                   const CornerSetup& across, const SuspensionState& acrossSuspension)
{
    // The wheel-referred bar: a rate times the difference in wheel travel across the axle. This is
    // the model every car in this project has been on, and a corner that states no drop link stays on
    // it — the expression below is the one that was inline in the force pass, factor for factor.
    const auto wheelReferred = corner.antiRollRate * (acrossSuspension.wheelTravel - suspension.wheelTravel);

    if (corner.antiRollRate == 0.0 || !dropLinkStated(corner.hardpoints) || !dropLinkStated(across.hardpoints))
    {
        return AntiRollBarSolution{.wheelForce = wheelReferred};
    }

    const auto selfRatio = dropLinkDesignRatio(corner.hardpoints);
    const auto acrossRatio = dropLinkDesignRatio(across.hardpoints);

    if (std::abs(selfRatio) < 1e-9 || std::abs(acrossRatio) < 1e-9 || std::abs(suspension.travelPerAngle) < 1e-9)
    {
        // A link that does not move with the wheel cannot carry a wheel rate onto itself. Falling
        // back rather than dividing by it, because a degenerate drop link is an authoring mistake and
        // the wheel-referred bar is still a bar.
        return AntiRollBarSolution{.wheelForce = wheelReferred};
    }

    // `k_wheel = k_link · ratio²` is the standard referral of a rate through a motion ratio, and it is
    // read backwards here because the wheel rate is what the data states. Taking one ratio from each
    // end makes the referral a property of the axle: whatever the two corners' geometries are, the
    // pair's link forces come out exactly equal and opposite, which a torsion bar's must be.
    const auto linkRate = corner.antiRollRate / (selfRatio * acrossRatio);

    const auto selfLink = dropLinkElementOf(corner.hardpoints);
    const auto acrossLink = dropLinkElementOf(across.hardpoints);

    const auto here = solveElement(corner.hardpoints, selfLink, suspension.wishboneAngle);
    const auto there = solveElement(across.hardpoints, acrossLink, acrossSuspension.wishboneAngle);

    // Displacement from design, positive as the link lengthens. The bar carries the *difference*, so
    // an axle in pure heave — both links moving together — makes no force at all, whatever the two
    // corners' ratios are.
    const auto selfDisplacement = here.length - solveElement(corner.hardpoints, selfLink, 0.0).length;
    const auto acrossDisplacement = there.length - solveElement(across.hardpoints, acrossLink, 0.0).length;

    const auto linkForce = linkRate * (acrossDisplacement - selfDisplacement);

    return AntiRollBarSolution{.linkForce = linkForce,
                               .lengthPerAngle = here.lengthPerAngle,
                               // What the same virtual work would take as a vertical force at the
                               // wheel, so that the reported number means one thing in both models.
                               .wheelForce = linkForce * here.lengthPerAngle / suspension.travelPerAngle,
                               .geometric = true};
}

[[nodiscard]] std::expected<VehicleStep, std::string>
stepVehicle(const VehicleSetup& setup, VehicleState& state, const VehicleInput& input,
            const std::array<double, cornerCount>& driveTorques, const PhysicsWorld& world, const double deltaTime,
            const BrakeCommand& brakes, const AmbientConditions& ambient)
{
    RACEENGINE_ZONE_N("stepVehicle");

    auto result = VehicleStep{};

    // Recomputed every tick rather than cached, because burning fuel changes all of it and a cache
    // is exactly what makes that a restructure later.
    // The chassis body carries the **whole** car's mass, sprung and unsprung, and that is a
    // correction rather than a convenience.
    //
    // The unsprung masses have a vertical degree of freedom each and no horizontal one — they are
    // rigidly carried fore-and-aft and side-to-side. Left out of the body, the car had 1200 kg of
    // horizontal inertia against 1352 kg of vertical, so every longitudinal and lateral
    // acceleration came out 12.7% too strong. A coastdown is what showed it: the deceleration
    // exceeded what the drag and rolling-resistance coefficients predicted by a ratio that was
    // suspiciously constant at every speed, which is the signature of a wrong mass rather than a
    // missing force.
    //
    // The vertical balance is unchanged by this. The corner's own equation still carries m_u for
    // the wheel's motion *relative* to the body, which is what wheel hop is, and the unsprung
    // weight now reaches the ground through the body's gravity instead of through a separate term.
    auto ledger = std::array<MassComponent, cornerCount + 1>{};
    {
        const auto sprung = computeMassProperties(setup.sprung);
        if (!sprung)
        {
            return std::unexpected(sprung.error());
        }

        ledger[0] = MassComponent{.mass = sprung->mass, .centre = sprung->centreOfMass, .inertia = sprung->inertia};

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            ledger[index + 1] = MassComponent{.mass = setup.corners[index].unsprungMass,
                                              .centre = setup.corners[index].hardpoints.wheelCentre,
                                              .inertia = glm::dmat3(0.0)};
        }
    }

    const auto properties = computeMassProperties(ledger);
    if (!properties)
    {
        return std::unexpected(properties.error());
    }

    applyMassProperties(state.chassis, properties.value());

    const auto rackTravel = input.steering * setup.rackTravelPerInput;

    // --- Pass one: where every corner is, and what road is under it -----------------------------
    //
    // All four are solved before any force is computed, because the anti-roll bar needs both ends
    // of an axle and because one batched ray cast beats four.
    auto allOrigins = std::vector<glm::dvec3>{};
    auto allDirections = std::vector<glm::dvec3>{};
    auto geometries = std::array<ContactSampleGeometry, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        const auto steered = index < 2 ? rackTravel : 0.0;

        auto solved = solveCornerWithJacobian(corner.hardpoints, state.corners[index].wishboneAngle, steered);
        if (!solved)
        {
            return std::unexpected("corner " + std::string(cornerAbbreviation(static_cast<Corner>(index))) + ": " +
                                   solved.error());
        }

        result.corners[index].suspension = solved.value();

        // And the twist the bushes are holding from the previous tick, which is the one force effect
        // the kinematic solve is not told about. Branched rather than applied with a zero angle, so a
        // car that states no compliance runs the solve it always ran, to the bit.
        if (corner.lateralForceSteer != 0.0)
        {
            applyComplianceSteer(corner.hardpoints, result.corners[index].suspension,
                                 state.corners[index].complianceSteer);
        }

        // The wheel, in the world. The suspension solved it in chassis coordinates; the chassis
        // says where those are.
        const auto& suspension = result.corners[index].suspension;
        const auto outboard = outboardSign(corner.hardpoints.side);

        const auto pose = WheelPose{
            .centre = bodyToWorld(state.chassis, suspension.wheelCentre),
            .spinAxis = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(outboard, 0.0, 0.0)),
            .forward = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(0.0, 0.0, 1.0)),
            .radius = corner.hardpoints.wheelRadius};

        geometries[index] = contactPatchSamples(pose, setup.sampling);
        allOrigins.insert(allOrigins.end(), geometries[index].origins.begin(), geometries[index].origins.end());
        allDirections.insert(allDirections.end(), geometries[index].directions.begin(),
                             geometries[index].directions.end());
    }

    auto allHits = std::vector<SurfaceHit>{};
    world.castRays(allOrigins, allDirections, setup.sampling.searchDistance * 2.0, allHits);

    // The world's own table, not the generator's. A `SurfaceHit` reports an index into the mesh the
    // world was built from, so reading it against any other table is reading the wrong row.
    const auto& materials = world.materials();
    auto consumed = std::size_t{0};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto count = geometries[index].origins.size();
        const auto slice = std::vector<SurfaceHit>(allHits.begin() + static_cast<std::ptrdiff_t>(consumed),
                                                   allHits.begin() + static_cast<std::ptrdiff_t>(consumed + count));
        consumed += count;

        result.corners[index].patch = aggregateContactPatch(geometries[index], slice, materials, setup.sampling);
    }

    // --- Pass two: what every element contributes ----------------------------------------------
    auto chassisForces = ForceAccumulator{};
    chassisForces.force = glm::dvec3(0.0, -earthGravity * state.chassis.mass, 0.0);

    const auto worldCentreOfMass = state.chassis.position;
    auto cornerDamping = std::array<double, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        auto& solution = result.corners[index];
        const auto& suspension = solution.suspension;

        // The damper's velocity and force read the damper's own element (`solveDamperForce`): the
        // same curve at the same compression-positive velocity, only the geometry source is the
        // element evaluator, whose Jacobian is the state's `damperLengthPerAngle` bit for bit.
        const auto damper = solveDamperForce(corner, suspension, state.corners[index].wishboneRate);
        solution.damperVelocity = damper.velocity;

        // Compression of the damper against its design length, positive in bump. Everything on the
        // damper axis — spring, damper, and both stops — is measured here, because that is where a
        // real bump stop lives: on the shaft, not at the wheel. Since step 12 both lengths are the
        // damper element's (`damperShaftCompression`), the same bits the design solve produced;
        // the stop force law below and the stops' generalised contribution are untouched.
        const auto compression = damperShaftCompression(corner, damper);

        // The spring reads its own element (`solveSpringForce`), which on every current car is the
        // damper's element and the same numbers it always was. Both stops stay on the damper axis
        // deliberately — the shaft is where a real bump stop lives.
        const auto spring = solveSpringForce(corner, suspension);
        solution.forces.spring = spring.force;
        solution.forces.damper = damper.force;
        solution.forces.bumpStop = corner.bumpStop.force(compression - corner.bumpStop.gap, solution.damperVelocity);
        solution.forces.droopStop =
            -corner.droopStop.force(-compression - corner.droopStop.gap, -solution.damperVelocity);

        // The tire, as a spring in series with the suspension and the unsprung mass between them.
        // It cannot pull: a wheel off the ground has no vertical force, however far the aggregate
        // says the road is below it.
        const auto wheelWorld = bodyToWorld(state.chassis, suspension.wheelCentre);
        const auto wheelVelocity =
            state.chassis.linearVelocity + glm::cross(angularVelocity(state.chassis), wheelWorld - worldCentreOfMass) +
            state.chassis.orientation *
                glm::dvec3(0.0, suspension.travelPerAngle * state.corners[index].wishboneRate, 0.0);

        // **Along the road's normal, not along the world's up**, and that is the whole of why a car
        // in this model now rolls down a hill. A vertical force on a slope exactly cancels vertical
        // gravity and leaves nothing along the surface, so the car was held up *and* held still: in
        // neutral on a 45% slope with no brakes it moved 0.0000 m in five seconds, and creeping in
        // gear it climbed one at walking pace on a quarter of the power that would take. The in-plane
        // tyre forces were always taken in the patch's frame — which is why banking and kerbs worked
        // and why this survived as long as it did — and only the load was flat-world.
        //
        // `patch.penetration` stays what it is: the **vertical** overlap, measured by vertical rays,
        // documented as such and depended on by the enveloping work. What the tyre's spring actually
        // compresses by is the perpendicular distance to the road, which for a vertical overlap
        // against a plane is that overlap times the normal's own vertical component. So the
        // conversion is one cosine and it lives here, at the point of use, rather than changing an
        // aggregate three other things read.
        //
        // Flat ground is unaffected to the bit: the normal is world up there, the cosine is exactly
        // one, and this is the same arithmetic it was.
        const auto normal = solution.patch.inContact ? solution.patch.normal : glm::dvec3(0.0, 1.0, 0.0);
        const auto normalOfVertical = std::max(normal.y, 0.0);

        const auto closingSpeed = -glm::dot(wheelVelocity, normal);
        solution.forces.tireVertical =
            solution.patch.inContact
                ? std::max(0.0, corner.tireVerticalRate * solution.patch.penetration * normalOfVertical +
                                    corner.tireVerticalDamping * closingSpeed)
                : 0.0;

        // The anti-roll bar resists the difference across its axle and nothing else, so a car
        // hitting a kerb with one wheel feels it and a car on a level road does not.
        //
        // Two models behind one call, chosen by whether the corner states drop-link hardpoints —
        // see `solveAntiRollBar`. A corner that states none reproduces the expression that was
        // inline here, factor for factor, which is every car in this project today.
        const auto& other = result.corners[acrossAxle[index]];
        const auto bar = solveAntiRollBar(corner, suspension, setup.corners[acrossAxle[index]], other.suspension);
        solution.forces.antiRoll = bar.wheelForce;
        solution.forces.antiRollLink = bar.linkForce;

        // --- the tire ---
        //
        // The contact patch's own frame: the wheel's heading and its perpendicular, both laid flat
        // in the plane the patch aggregated. Taking them from the *patch* normal rather than from
        // the world's up is what makes a banked or kerbed surface work — the tire pulls along the
        // road it is standing on, not along the horizon.
        if (solution.patch.inContact && solution.forces.tireVertical > 0.0)
        {
            // The same `normal` the load above is taken along, which is now one statement of it for
            // both halves of what the tyre does rather than two.
            const auto heading =
                state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(0.0, 0.0, 1.0));

            auto forward = heading - glm::dot(heading, normal) * normal;
            forward = glm::length(forward) > 1e-9 ? glm::normalize(forward) : glm::dvec3(0.0, 0.0, 1.0);
            const auto lateral = glm::cross(normal, forward);

            const auto patchVelocity =
                state.chassis.linearVelocity +
                glm::cross(angularVelocity(state.chassis), solution.patch.centre - worldCentreOfMass);

            solution.contact.forward = forward;
            solution.contact.lateral = lateral;
            solution.contact.longitudinalVelocity = glm::dot(patchVelocity, forward);
            solution.contact.lateralVelocity = glm::dot(patchVelocity, lateral);

            // Effective rolling radius sits between the free radius and the squashed one, nearer the
            // free: the tread belt is inextensible, so a loaded tire rolls further per revolution
            // than its loaded radius would suggest. Two thirds and a third is the usual figure.
            const auto loaded = std::max(corner.hardpoints.wheelRadius - solution.patch.penetration, 1e-3);
            solution.contact.effectiveRadius = (2.0 * corner.hardpoints.wheelRadius + loaded) / 3.0;

            // SAE: slip ratio is positive when the wheel is turning faster than the road, so a
            // driven wheel makes a forward force and a braked one makes a backward force with no
            // sign convention to remember at the call site.
            const auto longitudinalSlipVelocity = state.corners[index].wheelSpeed * solution.contact.effectiveRadius -
                                                  solution.contact.longitudinalVelocity;

            const auto deflectionRate =
                relaxTyre(corner.tyre, state.corners[index].tyre, solution.contact.longitudinalVelocity,
                          longitudinalSlipVelocity, solution.contact.lateralVelocity, deltaTime);

            // **What the tread's temperature is worth, read off the state this tick starts with.**
            // The multiplier has to be applied before the forces are evaluated and the temperature
            // cannot be advanced until the slip power those forces produce is known, so grip carries
            // one tick of explicit lag exactly as `complianceSteer` does. It is harmless here for a
            // different reason than there: the loop is positive rather than negative — hotter is
            // more grip is more slip power — but the tread's fastest time constant is a couple of
            // seconds against a 360 Hz tick, so the lag is three parts in a thousand of the
            // smallest thing in the loop.
            //
            // Copied and multiplied rather than branched, and that is what makes the switch inert:
            // with `tyreThermal` off the copy is the setup's own bits and `gripScale` is untouched,
            // and with it on inside the curve's plateau the multiplier is exactly 1.0, which in IEEE
            // arithmetic leaves the value alone rather than nearly alone.
            auto tyre = corner.tyre;
            if (setup.tyreThermal)
            {
                tyre.gripScale *= tyreTemperatureGrip(corner.tyre.thermal, state.corners[index].tyre);
            }

            solution.contact.slip = tyreSlip(tyre, state.corners[index].tyre);
            solution.contact.tyre =
                evaluateTyre(tyre, solution.forces.tireVertical, solution.contact.slip, solution.patch.gripMultiplier,
                             longitudinalSlipVelocity, solution.contact.lateralVelocity, deflectionRate);
        }
        else
        {
            // A wheel in the air carries no deflection into the moment it lands with. **The
            // temperatures survive**, which is the whole difference between a tyre that has a state
            // and one that does not: a wheel over a kerb does not forget how hot it is.
            state.corners[index].tyre.longitudinalDeflection = 0.0;
            state.corners[index].tyre.lateralDeflection = 0.0;
            solution.contact = WheelContact{};
        }

        // --- the tread's heat balance ---
        //
        // Outside the contact branch because a wheel in the air still cools, and after the forces
        // because the friction power it feeds on is one of them. Both generation terms were already
        // computed by this tick and thrown away until now: `slipPower` at the patch, and rolling
        // resistance against load and speed.
        //
        // The patch's footprint is the tyre's own deflection rather than a stated size — the chord a
        // wheel of this radius makes at this penetration — so the road-conduction term carries no
        // constant of its own. It over-states the true footprint, because a real tread does not
        // conform right out to the geometric chord, and therefore over-states road cooling by about
        // the square root of that: the conductance goes as `sqrt(patch length)`.
        if (setup.tyreThermal)
        {
            const auto penetration = solution.patch.inContact ? solution.patch.penetration : 0.0;
            const auto radius = corner.hardpoints.wheelRadius;
            const auto chord =
                penetration > 0.0
                    ? 2.0 * std::sqrt(std::max(2.0 * radius * penetration - penetration * penetration, 0.0))
                    : 0.0;

            stepTyreThermal(corner.tyre.thermal, state.corners[index].tyre,
                            TyreThermalInput{.slipPower = solution.contact.tyre.slipPower,
                                             .verticalLoad = solution.forces.tireVertical,
                                             .rollingResistance = corner.rollingResistance,
                                             .roadSpeed = solution.contact.longitudinalVelocity,
                                             .airSpeed = glm::length(state.chassis.linearVelocity),
                                             .patchLength = chord,
                                             .patchWidth = setup.sampling.width,
                                             // **The rim, and only when there is a brake model to
                                             // heat it.** The conductance is the wheel's own and it
                                             // reaches the carcass, not the tread. One tick of lag
                                             // on the wheel's temperature, which is read here before
                                             // the brake loop advances it — three parts in a million
                                             // of a node whose time constant is an hour.
                                             .wheelTemperature = state.corners[index].wheelTemperature,
                                             .wheelConductance = setup.brakeThermal ? corner.wheel.toTyre : 0.0,
                                             .ambient = ambient},
                            deltaTime);
        }

        // What the bushes will be holding on the next tick. The lateral force is resolved into the
        // **body's** own lateral axis rather than taken as the tyre's scalar, because compliance
        // steer is an axle steering away from the turn and that is a direction in the car, not in the
        // patch: on a banked road the two are not the same, and on a kerb they are nowhere near.
        //
        // A wheel in the air relaxes to nothing, which is the same statement the tyre's carcass makes
        // one branch up.
        if (corner.lateralForceSteer != 0.0)
        {
            const auto sideways = glm::dot(glm::inverse(state.chassis.orientation) *
                                               (solution.contact.tyre.lateral * solution.contact.lateral),
                                           glm::dvec3(1.0, 0.0, 0.0));

            state.corners[index].complianceSteer = corner.lateralForceSteer * sideways;
        }

        // --- generalised force on the corner's one degree of freedom ---
        //
        // **Assembled after the tyre rather than before it**, which is the one ordering change the
        // geometric load path needs: the in-plane tyre forces are inputs to it now, and reading last
        // tick's would put a 360 Hz delay inside a feedback loop that carries the jacking force.
        // Nothing between the two blocks reads either of them, so the move is arithmetic-neutral —
        // and the parity gates were held to byte-identity across it.
        const auto axisForce =
            solution.forces.spring + solution.forces.damper + solution.forces.bumpStop + solution.forces.droopStop;

        // Each element's generalised contribution is its force times its **own** element's
        // Jacobian — the spring's and, since step 8, the damper's; the stops still ride the
        // state's Jacobian until their own migration. On a coaxial car every Jacobian here is the
        // same bits — `solveElement` and the corner solve run identical arithmetic — and the fused
        // pre-split sum is kept for exactly that case, because regrouping `s·x + d·x` as `(s+d)·x`
        // moves the last ulp and thirty seconds of launch amplify an ulp into a parity failure.
        // The branch is on bit equality, which is the condition under which the two expressions
        // are the same value, not merely close.
        // Since step 13 every projection here is the element's: the stops ride the damper
        // element's Jacobian (the stop lives on the shaft), and the fused product does too. The
        // branch is keyed on the authored coil-over condition (`coaxialSpring`) since step 14:
        // when the spring's element IS the damper's element, every Jacobian in the sum is the
        // same arithmetic on the same points — the same bits — and the fused pre-split grouping
        // is kept, because regrouping `s·x + d·x` as `(s+d)·x`
        // moves the last ulp and thirty seconds of launch amplify an ulp into a parity failure.
        if (setup.geometricLoadPath)
        {
            // **The road force through the derivative of the point it acts on** — the whole dot
            // product rather than the vertical component of one term of it. A corner is one degree
            // of freedom, so this is not a model of the load path, it is the definition of the
            // generalised force; roll centre, jacking, anti-dive, anti-squat and anti-lift are
            // readings of `patchPerAngle`'s three components and are not implemented separately
            // anywhere.
            //
            // The tyre force is assembled in the world, where its three parts are stated, and
            // rotated into the chassis frame the linkage is solved in.
            const auto tyreInBody =
                glm::inverse(state.chassis.orientation) *
                (solution.forces.tireVertical * normal + solution.contact.tyre.longitudinal * solution.contact.forward +
                 solution.contact.tyre.lateral * solution.contact.lateral);

            // **The point the road pushes on is a material point of the wheel, and
            // `SuspensionState::contactPatch` is not one.** That patch is reconstructed every solve
            // from the world's own down direction projected into the wheel's plane, so it depends on
            // the spin axis and on nothing else about how the upright is standing: turn the upright
            // about the wheel's own axis and the constructed patch does not move, while the real one
            // rolls forward or back by a radius times the angle.
            //
            // Measured, that omission is **the whole of the longitudinal channel and none of the
            // lateral one** — `patchPerAngle - wheelCentrePerAngle` comes out exactly (lateral, 0, 0)
            // on all four corners, which is why the load-path brief found the patch's side-view ratio
            // and the wheel centre's agreeing to every digit it printed and read that as evidence the
            // term was negligible. It is not evidence about the term at all. `[.driveline-path]`
            // section 3 is the demonstration and asserts it.
            //
            // This model already treats the patch as a material point in the *other* coordinate,
            // where the same force's moment about the wheel centre is `−Fx·r` and is applied to the
            // wheel's spin every tick. Saying it here as well is one model written down consistently,
            // not a second one — and it corrects the load-path brief's section 4, which held that
            // applying the road force at the patch was already complete for an outboard brake. It was
            // complete for an *inboard* one: the constructed patch's side-view line is the wheel
            // centre's.
            //
            // The radius carried is the **loaded** one, which is item 6 of
            // docs/suspension-fidelity-brief.md and comes free here rather than costing a term: the
            // road is at the loaded radius, so that is where the force is applied. Same statement of
            // the deflection as the effective rolling radius above uses, so the two cannot disagree.
            const auto belowCentre = suspension.contactPatch - suspension.wheelCentre;
            const auto freeRadius = std::max(glm::length(belowCentre), 1e-12);
            const auto roadRadius = std::max(corner.hardpoints.wheelRadius - solution.patch.penetration, 1e-3);

            const auto patchPerAngle =
                suspension.wheelCentrePerAngle +
                glm::cross(suspension.uprightRatePerAngle, belowCentre * (roadRadius / freeRadius));

            // The bar and the unsprung weight keep the wheel's own vertical Jacobian, because that
            // is where they act: the drop link pulls the wishbone, and the unsprung weight is a
            // vertical force on a mass whose height is the wheel centre's. Neither has an in-plane
            // component to lose. The unsprung weight stays `m·g` rather than `m·g·(ŷ·bodyUp)` for
            // the reason it always did — a separate flat-world approximation, second order against
            // a 49 kg corner, and nothing to do with the load path.
            //
            // The bar's is not an approximation any more even so: a corner that states a drop link
            // reports the wheel force that does the same virtual work as its link force, so
            // `wheelForce · travelPerAngle` *is* `linkForce · lengthPerAngle` and the bar rides its
            // own element after all, through one multiplication instead of a second term.
            const auto unsprungUpForce = solution.forces.antiRoll - corner.unsprungMass * earthGravity;

            // And the shaft torque, which is the other half of the same statement. A chassis-mounted
            // transaxle drives the hub, so its torque does virtual work in this coordinate as the
            // upright turns about the wheel's spin axis; an outboard brake's couple is internal to
            // the wheel assembly and does none, which is why braking needs no counterpart here.
            //
            // The two terms together collapse exactly onto the wheel-centre force line, which is the
            // textbook statement of an inboard differential and is what `[.driveline-path]` section 5
            // asserts to 1e-9. The axis is `upright · (+1, 0, 0)` on **every** corner and deliberately
            // not the outboard-pointing axis the contact patch is posed with: both wheels of a car
            // going forward turn the same way.
            const auto shaftWork =
                setup.drivelineReaction
                    ? driveTorques[index] * glm::dot(suspension.uprightRatePerAngle,
                                                     suspension.uprightOrientation * glm::dvec3(1.0, 0.0, 0.0))
                    : 0.0;

            if (coaxialSpring(corner.hardpoints))
            {
                solution.generalisedForce = axisForce * damper.lengthPerAngle + glm::dot(tyreInBody, patchPerAngle) +
                                            unsprungUpForce * suspension.travelPerAngle + shaftWork;
            }
            else
            {
                const auto stopAxisForce = solution.forces.bumpStop + solution.forces.droopStop;

                solution.generalisedForce =
                    spring.force * spring.lengthPerAngle + damper.force * damper.lengthPerAngle +
                    stopAxisForce * damper.lengthPerAngle + glm::dot(tyreInBody, patchPerAngle) +
                    unsprungUpForce * suspension.travelPerAngle + shaftWork;
            }
        }
        else
        {
            // The corner's one degree of freedom is the wheel travelling along the **body's** up
            // axis, so what reaches it is the tyre load projected onto that axis rather than the
            // load itself. On flat level ground the normal, the body's up and the world's up are one
            // direction and this is the number it always was.
            //
            // The unsprung weight beside it is deliberately left as `m·g`, not `m·g·(ŷ·bodyUp)`.
            // That is a separate flat-world approximation with its own reason to exist, it is a
            // second-order term against a 49 kg corner on a 1348 kg car, and folding it in here
            // would move every cambered and rolling frame for something that has nothing to do with
            // slopes.
            //
            // **Every statement below is byte-for-byte the arithmetic this model has always run**,
            // which is what makes `geometricLoadPath` a control rather than a rewrite: every figure
            // in docs/ was measured here.
            const auto bodyUp = state.chassis.orientation * glm::dvec3(0.0, 1.0, 0.0);

            const auto wheelUpForce = solution.forces.tireVertical * std::max(glm::dot(normal, bodyUp), 0.0) +
                                      solution.forces.antiRoll - corner.unsprungMass * earthGravity;

            if (coaxialSpring(corner.hardpoints))
            {
                solution.generalisedForce =
                    axisForce * damper.lengthPerAngle + wheelUpForce * suspension.travelPerAngle;
            }
            else
            {
                const auto stopAxisForce = solution.forces.bumpStop + solution.forces.droopStop;

                solution.generalisedForce =
                    spring.force * spring.lengthPerAngle + damper.force * damper.lengthPerAngle +
                    stopAxisForce * damper.lengthPerAngle + wheelUpForce * suspension.travelPerAngle;
            }
        }

        // The generalised inertia of a one-degree-of-freedom assembly is `m·|dC/dq|²`, and taking
        // only the wheel centre's *vertical* rate drops the lateral and longitudinal parts of its
        // motion — so the corner is modelled as lighter than it is, its natural frequency comes out
        // too high, and it answers a kerb faster than it should.
        //
        // **Tied to the same switch as the force path, and that is the point rather than a
        // convenience.** With the geometric path on, the two sides of `F = ma` were written in
        // different models: the force used the whole vector and the inertia one component of it.
        // With it off, the force path is vertical-only and a vertical-only inertia is the matching
        // statement, which is also every figure this project has measured. So each position of the
        // switch is now internally consistent, and `OSR_LOAD_PATH=springs` is still the control.
        //
        // The upright's own rotational term is deliberately absent and is where this stops: it needs
        // an inertia tensor for the upright, which is data no car here carries.
        // docs/suspension-fidelity-brief.md, item 4.
        solution.generalisedInertia =
            setup.geometricLoadPath
                ? std::max(corner.unsprungMass *
                               glm::dot(suspension.wheelCentrePerAngle, suspension.wheelCentrePerAngle),
                           1e-6)
                : std::max(corner.unsprungMass * suspension.travelPerAngle * suspension.travelPerAngle, 1e-6);

        // The damper's contribution to the corner's damping, as a coefficient rather than a force,
        // so the integration below can solve it instead of stepping towards it. Damper rates are
        // exactly where an explicit treatment falls over. Since step 9 the coefficient reads the
        // damper element's own Jacobian and velocity — the same bits, through the same law.
        cornerDamping[index] = damperDampingCoefficient(corner, damper);

        // --- reaction on the chassis ---
        //
        // Newton's third law, taken over the whole corner rather than element by element. A corner
        // is a one-degree-of-freedom assembly hanging off the chassis; summing the forces on it,
        //
        //     F_on_chassis = F_tire + m_u * g - m_u * a_u
        //
        // which needs to know nothing about how the load path is arranged inside the linkage, and
        // is exact for the force balance whatever the geometry does.
        //
        // The obvious alternative — apply the spring and damper along the damper axis at its
        // chassis mount, as the brief says — is *not* on its own the same thing, and the difference
        // is not small. That axis is 0.915 vertical here, but virtual work says the corner
        // transmits the axis force times the motion ratio, 0.535: the wishbone is a lever and its
        // pivot reactions carry the rest, straight into the chassis at the pivots. Applying only
        // the damper's share over-supports the body by a factor of 1.7, and the car settles high on
        // its droop stops carrying 60% of its own weight. Modelling it the brief's way needs the
        // wishbone's constraint forces solved as well; the resultant below is what is exact until
        // they are.
        //
        // The tire force goes on at the **contact patch**, and the free body above is why that is
        // exact: with the caliper on the upright the brake couple is internal to the assembly, so
        // moment balance puts the chassis reaction on the same line of action as the road force.
        //
        // **What it is not is the jacking force**, and the comment that said so was wrong in a way
        // that took a brief to unpick. The moment this makes about the centre of gravity is the
        // *total* roll couple, which is right in both load-path models and is fixed by the whole-car
        // free body anyway. Jacking is a statement about how much of that couple the springs have to
        // react, and that is decided on the corner's side of the linkage — `patchPerAngle` and
        // `VehicleSetup::geometricLoadPath`, above. Do not "fix" this side; it is not broken.
        //
        // **And it goes on along the road's normal.** Applied along world up it balanced gravity
        // exactly on any slope and left the car nothing to roll down — see the load above.
        const auto contactPoint = solution.patch.inContact ? solution.patch.centre : wheelWorld;
        chassisForces.addForceAtPoint(solution.forces.tireVertical * normal, contactPoint, worldCentreOfMass);

        // The unsprung mass's weight, which the chassis carries whenever the tire is not, and its
        // inertia, which is what a wheel snatched upward by a kerb kicks back into the body with.
        //
        // Only the *relative* acceleration: the unsprung weight is already in the body's own gravity,
        // since the body carries the whole car's mass. This term is the cross coupling.
        //
        // **Which way the kick points is the same question the inertia above asks**, so it is on the
        // same switch. Vertical-only, a wheel snatched sideways by a kerb kicks the body straight up
        // and the sideways part is thrown away — and it is thrown away along the *world's* vertical
        // rather than the body's, which is a second flat-world approximation on top of the first. On
        // the geometric path the wheel centre's own direction of travel carries it, rotated out of
        // the chassis frame the linkage was solved in, and both approximations go together.
        const auto acceleration = solution.generalisedForce / solution.generalisedInertia;

        if (setup.geometricLoadPath)
        {
            chassisForces.addForceAtPoint(state.chassis.orientation *
                                              (-corner.unsprungMass * acceleration * suspension.wheelCentrePerAngle),
                                          wheelWorld, worldCentreOfMass);
        }
        else
        {
            const auto wheelAcceleration = suspension.travelPerAngle * acceleration;

            chassisForces.addForceAtPoint(glm::dvec3(0.0, -corner.unsprungMass * wheelAcceleration, 0.0), wheelWorld,
                                          worldCentreOfMass);
        }

        // And the tire's in-plane forces, at the contact patch, which is where they act. Their
        // moment about the centre of gravity is the roll couple and the pitch couple — the *total*
        // load transfer, which is `m·a·h/t` and `m·a·h/L` whatever the linkage does with it. The
        // jacking force is not here and never was; it is the same forces read through
        // `patchPerAngle` on the corner's side.
        chassisForces.addForceAtPoint(solution.contact.tyre.longitudinal * solution.contact.forward +
                                          solution.contact.tyre.lateral * solution.contact.lateral,
                                      solution.patch.centre, worldCentreOfMass);

        // The aligning moment is a couple about the patch normal, so it goes on as one rather than
        // as a force somewhere.
        chassisForces.torque += solution.contact.tyre.aligningMoment * solution.patch.normal;
    }

    // --- Ride height, then aero, and in that order ----------------------------------------------
    //
    // The ordering is the seam rather than a preference: a rake- or ride-height-sensitive map is
    // deferred, and the only thing that makes adding it a change to the coefficients instead of a
    // change to the loop is that the suspension has already been solved by the time this runs.
    //
    // See `RideHeight` for what is measured and for the three faults in the expression this
    // replaces. Nothing reads it yet; it is reported rather than consumed.
    {
        const auto floor = setup.rideHeightReference;

        // The plane the touching corners define, for any corner that is in the air.
        auto meanPoint = glm::dvec3(0.0);
        auto meanNormal = glm::dvec3(0.0);

        for (const auto& solution : result.corners)
        {
            if (solution.patch.inContact)
            {
                meanPoint += solution.patch.centre;
                meanNormal += solution.patch.normal;
                result.rideHeight.grounded++;
            }
        }

        if (result.rideHeight.grounded > 0)
        {
            const auto count = static_cast<double>(result.rideHeight.grounded);
            meanPoint /= count;
            meanNormal = glm::length(meanNormal) > 1e-9 ? glm::normalize(meanNormal) : glm::dvec3(0.0, 1.0, 0.0);

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& solution = result.corners[index];
                const auto& patch = solution.patch;

                // Directly above this corner's wheel centre, on the floor. The wheel centre's plan
                // position is what puts the measurement at the corner rather than at the middle of
                // the car, which is the whole reason this is four numbers.
                const auto& design = setup.corners[index].hardpoints.wheelCentre;
                const auto reference = bodyToWorld(state.chassis, glm::dvec3(design.x, floor, design.z));

                const auto point = patch.inContact ? patch.centre : meanPoint;
                const auto normal = patch.inContact ? patch.normal : meanNormal;

                result.rideHeight.corners[index] = glm::dot(reference - point, normal);
            }

            result.rideHeight.front = 0.5 * (result.rideHeight.corners[0] + result.rideHeight.corners[1]);
            result.rideHeight.rear = 0.5 * (result.rideHeight.corners[2] + result.rideHeight.corners[3]);
            result.rideHeight.rake = result.rideHeight.rear - result.rideHeight.front;
        }
    }

    const auto airspeed = glm::length(state.chassis.linearVelocity);
    if (airspeed > 1e-6)
    {
        const auto flow = state.chassis.linearVelocity / airspeed;
        const auto pressure = 0.5 * setup.airDensity * airspeed * airspeed;

        for (const auto& surface : setup.aero)
        {
            // Drag opposes the direction of travel, lift acts along the world's vertical. Applied
            // at the surface's own point, because a splitter and a rear wing pitch the car opposite
            // ways and that is most of what an aero balance *is*.
            const auto force = -pressure * surface.dragArea * flow + glm::dvec3(0.0, pressure * surface.liftArea, 0.0);

            chassisForces.addForceAtPoint(force, bodyToWorld(state.chassis, surface.centre), worldCentreOfMass);
        }
    }

    // --- Pass three: integrate ------------------------------------------------------------------
    const auto previousVelocity = state.chassis.linearVelocity;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& corner = state.corners[index];
        const auto& solution = result.corners[index];

        const auto acceleration = solution.generalisedForce / solution.generalisedInertia;
        auto rate = corner.wishboneRate + acceleration * deltaTime;
        rate /= 1.0 + (cornerDamping[index] / solution.generalisedInertia) * deltaTime;

        corner.wishboneRate = rate;
        corner.wishboneAngle += rate * deltaTime;

        // --- wheel spin ---
        //
        // Two torques and one integration. The road's is the tire's own reaction and is
        // self-correcting — a wheel turning too fast makes a forward force and is slowed by it, a
        // wheel turning too slow is spun up — and the driveline's is whatever the chain handed in.
        //
        // They are summed here rather than applied in two places, and that is a correction. The
        // driveline used to advance this field itself before the tick ran, which put it upstream of
        // the tire's slip and upstream of the brake clamp below while the road's torque was
        // downstream of both: the brake sized itself against one of the two and knew nothing of the
        // other, so a wheel held on the brake still spun up under power for the tire to read as
        // slip. Launch, creep and converter stall are all that case.
        //
        // The inertia is `setup.corners[i].wheelInertia` and only that. It was also stated a second
        // time as an array the caller built for the driveline, with nothing to make the two agree.
        const auto& setupCorner = setup.corners[index];
        const auto roadTorque = -solution.contact.tyre.longitudinal * solution.contact.effectiveRadius;
        corner.wheelSpeed +=
            ((roadTorque + driveTorques[index]) / std::max(setupCorner.wheelInertia, 1e-6)) * deltaTime;

        // Braking, clamped so it can bring the wheel to a stop and not past it. Without the clamp a
        // hard brake application at low wheel speed drives the wheel *backwards* within one tick,
        // which the tire then reads as enormous slip in the other direction — a lock-up that
        // oscillates instead of locking.
        //
        // Rolling resistance rides in the same clamp, and for the same reason. It is a torque and
        // not a force at the patch: the contact pressure is higher at the leading edge of the patch
        // than the trailing one, so the vertical resultant acts *ahead* of the wheel centre and the
        // moment that makes is what a coastdown is measuring.
        //
        // The brake torque itself is the assist layer's when one ran and the driver's demand against
        // this corner's peak when none did. Written as a branch rather than as an always-present
        // command because an unassisted car has to reach the *same expression* it always did: the two
        // arms are one multiplication apart and a car with the electronics switched off must be
        // identical to the bit, not to a tolerance.
        //
        // **The demand is the pedal's own pressure response and not the pedal** (2026-08-23). A car
        // with no servo and no proportioning valve — which is every car here that does not state
        // hydraulics — gets exactly `brakeTorque * pedal` back out of it, so the change is inert for
        // them. A car that states them gets the servo's runout on both axles and the valve's knee on
        // the rear, which is the only place a *pedal-dependent* brake bias can come from.
        //
        // **And fade multiplies it** (2026-08-28). The pad's friction falls with the disc's
        // temperature, and both arms of the branch above are subject to it — the assist layer's
        // command is a torque the ECU asked for through a valve, and a valve cannot make a hot pad
        // grip. Exactly 1.0 with the switch off, and exactly 1.0 anywhere on the curve's own flat
        // part, which is everything below about 350 °C. docs/brake-thermal-brief.md.
        const auto rolling =
            setupCorner.rollingResistance * solution.forces.tireVertical * solution.contact.effectiveRadius;
        const auto fade = setup.brakeThermal ? setupCorner.disc.couple.fade.at(corner.discTemperature) : 1.0;
        const auto braking =
            (brakes.commanded ? std::max(0.0, brakes.wheels[index])
                              : setupCorner.brakeTorque * brakePedalResponse(setup, index, input.brake)) *
            fade;
        const auto commanded = braking + rolling;

        // The speed the brake is about to act at, read before the arrest below changes it. It is
        // what turns a torque into watts, and it is why a **locked** wheel makes no heat at the disc:
        // a pad clamped to a disc that is not turning dissipates nothing. That energy goes into the
        // road instead, which the tyre's own thermal model already has.
        const auto spinBefore = std::abs(corner.wheelSpeed);

        auto arrested = 0.0;

        // Whether the pad was **sliding** against a turning disc this tick, or **holding** a wheel it
        // had already stopped. The clamp below is what separates them: it fires exactly when the
        // demand would have driven the wheel backwards, which is a wheel being held.
        auto sliding = false;

        if (commanded > 0.0 && std::abs(corner.wheelSpeed) > 0.0)
        {
            // The sign is read before the wheel is slowed, because a brake that brings it exactly to
            // rest leaves nothing to read it from afterwards.
            const auto turning = corner.wheelSpeed;
            const auto arresting = std::abs(corner.wheelSpeed) * setupCorner.wheelInertia / deltaTime;
            const auto applied = std::min(commanded, arresting);
            sliding = commanded <= arresting;
            corner.wheelSpeed -= std::copysign(applied / setupCorner.wheelInertia * deltaTime, corner.wheelSpeed);
            arrested = std::copysign(applied, turning);
        }

        // --- the disc's heat balance ---
        //
        // The friction power is the **brake's** share of what was actually applied, times the speed
        // it was applied at. Rolling resistance rides in the same clamp and is not a brake, so it is
        // taken out by the share rather than left to warm the disc: a coasting car would otherwise
        // heat its brakes for ever.
        //
        // **And only while the pad is sliding**, which is not the same as "while the brake is on".
        // A locked wheel is re-arrested every tick after the road's torque has nudged it, so a naive
        // `torque × speed` reads a small power that is pure discretisation — it goes as the square of
        // the timestep and vanishes as the tick shrinks. Measured at 360 Hz it was worth 1.8 °C over
        // three seconds of a locked stop, about 4% of a real stop's rise, in the one case the answer
        // should be exactly zero. A held wheel's energy goes into the road through the tyre, and the
        // tyre's own thermal model already has it.
        if (setup.brakeThermal)
        {
            const auto share = commanded > 0.0 ? braking / commanded : 0.0;
            const auto airSpeed = glm::length(state.chassis.linearVelocity);

            // --- and the wheel between the disc and the tyre (stage 3) ---
            //
            // **The coupling is computed once and handed to both nodes**, because it carries a
            // radiation term that depends on both their temperatures and two independent evaluations
            // of it would let the disc lose what the wheel does not gain. Conduction through the
            // hat's neck in series with the bolted joint, plus that radiation.
            //
            // Both nodes read the temperatures the tick started with and are written afterwards, so
            // the answer does not depend on which is solved first — `stepTyreThermal`'s treatment,
            // for its reason.
            //
            // A car that states no wheel gets a coupling of exactly zero, and both steps then
            // reproduce stage 2's arithmetic expression for expression.
            const auto disc = corner.discTemperature;
            const auto wheel = corner.wheelTemperature;
            const auto coupling = discToWheelCoupling(setupCorner.disc, setupCorner.wheel, disc, wheel);

            stepBrakeThermal(setupCorner.disc, corner.discTemperature,
                             BrakeThermalInput{.frictionPower = sliding ? std::abs(arrested) * share * spinBefore : 0.0,
                                               .airSpeed = airSpeed,
                                               .wheelTemperature = wheel,
                                               .wheelConductance = coupling,
                                               .ambient = ambient},
                             deltaTime);

            // The tyre side is live only when the tyre carries a temperature. Letting the wheel
            // exchange with a carcass nothing is simulating would warm it against a constant and
            // create the energy the carcass never lost.
            stepWheelThermal(setupCorner.wheel, corner.wheelTemperature,
                             WheelThermalInput{.discTemperature = disc,
                                               .discConductance = coupling,
                                               .tyreTemperature = corner.tyre.carcassTemperature,
                                               .tyreConductance = setup.tyreThermal ? setupCorner.wheel.toTyre : 0.0,
                                               .airSpeed = airSpeed,
                                               .ambient = ambient},
                             deltaTime);
        }

        // --- what spinning the wheel up takes out of the body -----------------------------------
        //
        // A wheel gaining or losing angular momentum takes it from somewhere, and this model gives
        // it nowhere to come from. The chassis receives the road force at the contact patch, which
        // is exactly right while the wheel's spin is steady — an outboard brake's couple is internal
        // to the wheel assembly, so moment balance puts the chassis reaction on the road force's own
        // line, which is what the comment beside that call says and it is correct. What is missing is
        // the case where the spin is *not* steady: the difference between the two treatments is
        // precisely `I·alpha`, the torque that went into the wheel instead of into the body.
        //
        // Checked against the whole car rather than argued: for a car accelerating at `a`, angular
        // momentum about the centre of gravity requires the load transfer to exceed `m·a·h/L` by
        // `sum(I·a/r)` over the four wheels, and it is exactly this term that closes it. On this car
        // that is about 1.8% of the transfer under acceleration, zero at a steady speed, and its
        // moment is largest during a launch, a shift and a lock-up — which are the three places the
        // wheels' speeds move fastest.
        //
        // The axis is `upright · (+1, 0, 0)` and not the outboard-pointing axis the wheel is posed
        // with, because both wheels of a car going forward turn the same way.
        if (setup.drivelineReaction)
        {
            const auto& suspension = solution.suspension;
            const auto spinAxis =
                state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(1.0, 0.0, 0.0));

            chassisForces.torque -= (roadTorque + driveTorques[index] - arrested) * spinAxis;
        }

        // The linkage has no solutions past its own geometric limits, and the stops exist so this
        // is never reached — but a clamp is what keeps a bad setup reporting a stiff car rather than
        // an error, and the sweep at load time is where a bad setup is supposed to be caught.
        const auto& hardpoints = setup.corners[index].hardpoints;
        if (corner.wishboneAngle > hardpoints.bumpAngle)
        {
            corner.wishboneAngle = hardpoints.bumpAngle;
            corner.wishboneRate = std::min(corner.wishboneRate, 0.0);
        }
        else if (corner.wishboneAngle < hardpoints.droopAngle)
        {
            corner.wishboneAngle = hardpoints.droopAngle;
            corner.wishboneRate = std::max(corner.wishboneRate, 0.0);
        }
    }

    // Contact, resolved as impulses on the velocity and *before* the integrator moves the body.
    // A contact is a velocity constraint: expressed instead as a force over the tick it is either
    // spongy or explosive depending on the timestep, and there is no setting that is neither.
    //
    // Only the bodywork comes through here. The tires reach the ground through their own model, and
    // a rigid constraint underneath them would be fighting a carefully shaped force with something
    // that knows nothing about slip.
    result.contacts = collideBody(world, state.chassis, setup.body);
    resolveContacts(state.chassis, result.contacts, setup.contact, deltaTime);

    integrate(state.chassis, chassisForces, deltaTime);

    // --- Telemetry ------------------------------------------------------------------------------
    auto& frame = result.telemetry;
    frame.position = state.chassis.position;
    frame.velocity = state.chassis.linearVelocity;

    const auto rotation = glm::mat3_cast(state.chassis.orientation);
    const auto forward = rotation * glm::dvec3(0.0, 0.0, 1.0);
    const auto right = rotation * glm::dvec3(1.0, 0.0, 0.0);

    // **Into the car's own frame**, because that is what a channel called `G Force Long` means. The
    // difference from a world-frame answer is invisible on a straight aimed along +z and is most of
    // the signal on a circuit — see `TelemetryFrame::acceleration`.
    frame.acceleration =
        glm::conjugate(state.chassis.orientation) * ((state.chassis.linearVelocity - previousVelocity) / deltaTime);

    frame.yaw = std::atan2(forward.x, forward.z);
    frame.pitch = std::asin(std::clamp(forward.y, -1.0, 1.0));
    frame.roll = -std::asin(std::clamp(right.y, -1.0, 1.0));

    const auto bodyRates = glm::conjugate(state.chassis.orientation) * angularVelocity(state.chassis);
    frame.yawRate = bodyRates.y;
    frame.pitchRate = bodyRates.x;
    frame.rollRate = bodyRates.z;

    // The axle means the ride-height solve above already produced. Copied rather than recomputed:
    // one expression, whose three faults are recorded at `RideHeight`, and a second one here would
    // be a second place for them to come back.
    frame.rideHeightFront = result.rideHeight.front;
    frame.rideHeightRear = result.rideHeight.rear;

    frame.steering = input.steering;
    frame.steeringWheelAngle = input.steering * 0.5 * setup.steeringLockToLock;
    frame.throttle = input.throttle;
    frame.brake = input.brake;
    frame.gear = input.gear;
    // `TelemetryFrame::engineSpeed`, `clutch` and the driveline channels beside them are filled by whoever
    // stepped the driveline: this partition does not import `:Driveline` and must not, and the
    // frame is a by-value member of the returned step for exactly that reason.

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& solution = result.corners[index];

        frame.wheels[index].verticalLoad = solution.forces.tireVertical;
        frame.wheels[index].slipRatio = solution.contact.slip.slipRatio;
        frame.wheels[index].slipAngle = solution.contact.slip.slipAngle;
        frame.wheels[index].forceLongitudinal = solution.contact.tyre.longitudinal;
        frame.wheels[index].forceLateral = solution.contact.tyre.lateral;
        frame.wheels[index].aligningMoment = solution.contact.tyre.aligningMoment;
        frame.wheels[index].suspensionTravel = solution.suspension.wheelTravel;
        frame.wheels[index].damperVelocity = solution.damperVelocity;
        frame.wheels[index].angularVelocity = state.corners[index].wheelSpeed;
        frame.wheels[index].camber = solution.suspension.camber;
        frame.wheels[index].gripMultiplier = solution.patch.gripMultiplier;
        frame.wheels[index].inContact = solution.patch.inContact;
        frame.wheels[index].contactingSamples = solution.patch.contactingSamples;
        frame.wheels[index].patchDepthSpread = solution.patch.depthSpread;
        frame.wheels[index].tyreSurfaceTemperature = state.corners[index].tyre.surfaceTemperature;
        frame.wheels[index].tyreCoreTemperature = state.corners[index].tyre.coreTemperature;
        frame.wheels[index].tyreCarcassTemperature = state.corners[index].tyre.carcassTemperature;
        frame.wheels[index].discTemperature = state.corners[index].discTemperature;
        frame.wheels[index].wheelTemperature = state.corners[index].wheelTemperature;
    }

    return result;
}

} // namespace raceengine
