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

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] Curve linearDamper(const double bumpRate, const double reboundRate)
{
    return Curve{
        .points = {glm::dvec2(-5.0, -5.0 * reboundRate), glm::dvec2(0.0, 0.0), glm::dvec2(5.0, 5.0 * bumpRate)}};
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
    const auto solved = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
    if (!solved)
    {
        return std::unexpected(solved.error());
    }

    if (std::abs(solved->motionRatio) < 1e-9 || corner.springRate <= 0.0)
    {
        return std::unexpected("a corner with no motion ratio or no spring cannot be given a rest length");
    }

    // At equilibrium the spring's force along the damper, resolved to the wheel by the motion
    // ratio, carries the sprung corner load.
    return solved->damperLength + sprungLoad / (corner.springRate * std::abs(solved->motionRatio));
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
        const auto friction = tyreFriction(tyre, tyre.lateralPeak, load, 1.0);
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

    const auto design = solveCorner(corner.hardpoints, 0.0, 0.0);
    const auto atBump = solveCorner(corner.hardpoints, corner.hardpoints.bumpAngle, 0.0);
    const auto atDroop = solveCorner(corner.hardpoints, corner.hardpoints.droopAngle, 0.0);

    if (!design || !atBump || !atDroop)
    {
        return std::unexpected("the corner cannot be solved across its own travel");
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
    const auto swept = sweepCorner(corner.hardpoints, 41);
    if (!swept)
    {
        return std::unexpected(swept.error());
    }

    auto sign = 0.0;
    for (const auto& sample : swept->samples)
    {
        if (std::abs(sample.motionRatio) < 1e-9)
        {
            continue;
        }

        const auto here = sample.motionRatio < 0.0 ? -1.0 : 1.0;
        if (sign != 0.0 && here != sign)
        {
            return std::unexpected("the motion ratio changes sign across the travel, so the damper "
                                   "reverses direction relative to the wheel; its mounts are almost "
                                   "certainly not where the linkage thinks they are");
        }

        sign = here;

        if (std::abs(sample.motionRatio) > 2.0)
        {
            return std::unexpected("the motion ratio reaches " + std::to_string(sample.motionRatio) +
                                   ", so the damper is being asked to move twice as far as the wheel");
        }
    }

    const auto compression = design->damperLength - atBump->damperLength;
    const auto extension = atDroop->damperLength - design->damperLength;

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

[[nodiscard]] std::expected<VehicleStep, std::string> stepVehicle(const VehicleSetup& setup, VehicleState& state,
                                                                  const VehicleInput& input,
                                                                  const std::array<double, cornerCount>& driveTorques,
                                                                  const PhysicsWorld& world, const double deltaTime)
{
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

        // Compression of the damper against its design length, positive in bump. Everything on the
        // damper axis — spring, damper, and both stops — is measured here, because that is where a
        // real bump stop lives: on the shaft, not at the wheel.
        const auto designSolve = solveCorner(corner.hardpoints, 0.0, 0.0);
        if (!designSolve)
        {
            return std::unexpected(designSolve.error());
        }

        const auto compression = designSolve->damperLength - suspension.damperLength;
        solution.damperVelocity = -suspension.damperLengthPerAngle * state.corners[index].wishboneRate;

        solution.forces.spring = corner.springRate * (corner.springFreeLength - suspension.damperLength);
        solution.forces.damper = corner.damper.at(solution.damperVelocity);
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

        const auto closingSpeed = -wheelVelocity.y;
        solution.forces.tireVertical = solution.patch.inContact
                                           ? std::max(0.0, corner.tireVerticalRate * solution.patch.penetration +
                                                               corner.tireVerticalDamping * closingSpeed)
                                           : 0.0;

        // The anti-roll bar resists the difference across its axle and nothing else, so a car
        // hitting a kerb with one wheel feels it and a car on a level road does not.
        const auto& other = result.corners[acrossAxle[index]];
        solution.forces.antiRoll = corner.antiRollRate * (other.suspension.wheelTravel - suspension.wheelTravel);

        // --- generalised force on the corner's one degree of freedom ---
        const auto axisForce =
            solution.forces.spring + solution.forces.damper + solution.forces.bumpStop + solution.forces.droopStop;

        const auto wheelUpForce =
            solution.forces.tireVertical + solution.forces.antiRoll - corner.unsprungMass * earthGravity;

        solution.generalisedForce =
            axisForce * suspension.damperLengthPerAngle + wheelUpForce * suspension.travelPerAngle;

        solution.generalisedInertia =
            std::max(corner.unsprungMass * suspension.travelPerAngle * suspension.travelPerAngle, 1e-6);

        // The damper's contribution to the corner's damping, as a coefficient rather than a force,
        // so the integration below can solve it instead of stepping towards it. Damper rates are
        // exactly where an explicit treatment falls over.
        cornerDamping[index] = suspension.damperLengthPerAngle * suspension.damperLengthPerAngle *
                               std::max(0.0, curveSlopeAt(corner.damper, solution.damperVelocity));

        // --- the tire ---
        //
        // The contact patch's own frame: the wheel's heading and its perpendicular, both laid flat
        // in the plane the patch aggregated. Taking them from the *patch* normal rather than from
        // the world's up is what makes a banked or kerbed surface work — the tire pulls along the
        // road it is standing on, not along the horizon.
        if (solution.patch.inContact && solution.forces.tireVertical > 0.0)
        {
            const auto normal = solution.patch.normal;
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

            solution.contact.slip = tyreSlip(corner.tyre, state.corners[index].tyre);
            solution.contact.tyre = evaluateTyre(corner.tyre, solution.forces.tireVertical, solution.contact.slip,
                                                 solution.patch.gripMultiplier, longitudinalSlipVelocity,
                                                 solution.contact.lateralVelocity, deflectionRate);
        }
        else
        {
            // A wheel in the air carries no deflection into the moment it lands with.
            state.corners[index].tyre = TyreState{};
            solution.contact = WheelContact{};
        }

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
        // The tire force goes on at the **contact patch**, which is what the brief is really after:
        // the moment it makes about the centre of gravity is where jacking and the lateral load
        // path come from, and that is preserved here.
        const auto contactPoint = solution.patch.inContact ? solution.patch.centre : wheelWorld;
        chassisForces.addForceAtPoint(glm::dvec3(0.0, solution.forces.tireVertical, 0.0), contactPoint,
                                      worldCentreOfMass);

        // The unsprung mass's weight, which the chassis carries whenever the tire is not, and its
        // inertia, which is what a wheel snatched upward by a kerb kicks back into the body with.
        const auto wheelAcceleration =
            suspension.travelPerAngle * (solution.generalisedForce / solution.generalisedInertia);
        // Only the *relative* acceleration now: the unsprung weight is already in the body's own
        // gravity, since the body carries the whole car's mass. This term is the cross coupling —
        // what a wheel snatched upward by a kerb kicks back into the body with.
        chassisForces.addForceAtPoint(glm::dvec3(0.0, -corner.unsprungMass * wheelAcceleration, 0.0), wheelWorld,
                                      worldCentreOfMass);

        // And the tire's in-plane forces, at the contact patch — which is the whole of what the
        // brief means by applying them where they act. Their moment about the centre of gravity is
        // the roll couple and the jacking force, and neither is modelled anywhere else.
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
        const auto rolling =
            setupCorner.rollingResistance * solution.forces.tireVertical * solution.contact.effectiveRadius;
        const auto commanded = setupCorner.brakeTorque * std::clamp(input.brake, 0.0, 1.0) + rolling;

        if (commanded > 0.0 && std::abs(corner.wheelSpeed) > 0.0)
        {
            const auto arresting = std::abs(corner.wheelSpeed) * setupCorner.wheelInertia / deltaTime;
            const auto applied = std::min(commanded, arresting);
            corner.wheelSpeed -= std::copysign(applied / setupCorner.wheelInertia * deltaTime, corner.wheelSpeed);
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
    }

    return result;
}

} // namespace raceengine
