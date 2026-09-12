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
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.physics;

namespace raceengine
{

namespace
{

// How many raw per-triangle contacts one wheel's cylinder may report before the exclusion rule
// runs. It has to cover the flat road under the tyre as well as whatever the tyre has run into: a
// 0.64 by 0.23 m footprint on a quarter-metre grid is a dozen triangles, and a kerb is four more.
constexpr auto obstacleQueryLimit = std::uint32_t{64};

// The sliding speed at which the kerb-contact path's Coulomb friction is fully developed, m/s.
// **Numerical, not physical**, and the same shape of choice as `CornerSetup::damperFrictionSpeed`:
// `mu N tanh(v / this)` rather than `mu N sign(v)`, because a hard sign term at 360 Hz makes a
// limit cycle rather than a stuck wheel. A tenth of a metre per second is small against any kerb
// strike and large enough that one tick cannot step across it.
constexpr auto obstacleFrictionSpeed = 0.1;

// The coupled corner solve's convergence (docs/frame-acceleration-brief.md, 2026-09-08 latest of
// all): the Gauss–Seidel sweep over the four corners stops when no corner's change of rate moved by
// more than this between two sweeps, radians per second — a tenth of a picometre of wheel travel
// per tick at 360 Hz — and never runs past the limit. The coupling between corners is a few per cent
// of a corner's own inertia, so the sweep contracts by that per pass and the Golf settles in well
// under ten.
constexpr auto frameSweepTolerance = 1e-13;
constexpr auto frameSweepLimit = std::uint32_t{32};

// The kinematic reactions' own convergence inside the same sweep (docs/chassis-kinematic-reaction-brief.md):
// the change of the chassis's acceleration they account for, over a tick, metres per second — 4e-8 m/s²
// at 360 Hz, a twentieth of a millinewton on a wheel. Their secant curvature is read off two corner
// solves a tick's travel apart, and the solve's rounding divided by that travel puts a floor of about
// 2e-9 m/s² under the fixed point (measured on the hanging-wheel witness): the corners' 1e-13 above is
// under that floor and a tolerance there runs every tick to the sweep cap without getting closer.
constexpr auto kinematicSweepTolerance = 1e-10;

// ...and the change of rate under which the corners count as settled for that hold, radians per
// second: the hold waits for both, so a range or stop decision that moves a corner's rate after the
// reactions settled — a landing decided a sweep later — re-opens them at the rate that acts.
constexpr auto kinematicRateTolerance = 1e-9;

} // namespace

[[nodiscard]] std::uint32_t keepObstacleContacts(const std::span<const ObstacleContact> found,
                                                 const WheelCylinder& wheel, const glm::dvec3& gridNormal,
                                                 std::array<ObstacleContact, maxObstacleContacts>& kept)
{
    auto count = std::uint32_t{0};

    const auto axisLength = glm::length(wheel.spinAxis);
    if (!(axisLength > 0.0) || !(wheel.radius > 0.0) || !(wheel.halfWidth > 0.0))
    {
        return 0;
    }

    const auto spin = wheel.spinAxis / axisLength;

    // The rim rounding the backend actually built, which is Jolt's default lowered to fit.
    const auto rounding = std::min({obstacleConvexRadius, wheel.radius, wheel.halfWidth});

    // The wheel plane's down direction, constructed exactly as `contactPatchSamples` constructs its
    // `inPlaneDown` — the same expression, so that the two models of one tyre agree about which
    // direction is the grid's. A wheel lying flat has no in-plane down and no grid either; nothing
    // can be excluded against it, so nothing is kept.
    const auto worldDown = glm::dvec3(0.0, -1.0, 0.0);
    const auto inPlane = worldDown - glm::dot(worldDown, spin) * spin;
    const auto inPlaneLength = glm::length(inPlane);
    if (!(inPlaneLength > 1e-9))
    {
        return 0;
    }

    const auto inPlaneUp = -(inPlane / inPlaneLength);

    // What the grid's contacts point along: its own patch normal while the wheel touches, and the
    // wheel's up while it does not. On flat road the two are one direction; on a hill they are not,
    // and the hill is the grid's.
    const auto gridLength = glm::length(gridNormal);
    const auto reference = gridLength > 0.5 ? gridNormal / gridLength : inPlaneUp;

    for (const auto& contact : found)
    {
        // Separation rather than contact cannot come back with the query's zero separation
        // distance, and a first touch reports exactly zero, which is kept: a contact that exists at
        // zero depth and grows continuously from there is the whole point.
        if (!(contact.depth >= 0.0))
        {
            continue;
        }

        // The exclusion rule, in two halves. The axis points out of the road into the tyre, so on
        // the road the grid is standing on it is the grid's own normal and is dropped.
        if (glm::dot(contact.axis, reference) >= obstacleExclusionCosine)
        {
            continue;
        }

        // And a contact against a triangle whose own face is within the cone is dropped **whatever
        // its push-out points along**, because that triangle is road the grid already carries and
        // its edges are triangulation, not surface. The case this exists for was measured on the
        // E2 chamfer crossing (stage 2, 2026-09-08): a rim standing over the crease between the
        // flat road and the 9.46 degree chamfer overlapped a triangle by a sliver, the shortest way
        // out of a sliver is *sideways* off the triangle's edge, that edge is active (9.46 > 5
        // degrees) so Jolt kept the sideways axis, and 367 sidewall pushes a few millimetres deep
        // landed on a road with no face on it. A kerb's top edge is the other case exactly: the
        // triangle it belongs to is the face, whose normal is nowhere near the cone, so an edge
        // contact against it is kept.
        if (glm::dot(contact.normal, reference) >= obstacleExclusionCosine)
        {
            continue;
        }

        // The depth-consistency check, and the artefact it exists for. Jolt reports per triangle,
        // and with active-edge mode on it hands a contact whose closest feature is an *inactive*
        // edge — the diagonal a quad's two triangles share — the triangle's face normal in place of
        // the edge's own push-out, **keeping the edge's depth**. On a kerb face split into two
        // triangles that is a second contact, horizontal, a few tenths of a millimetre deep, six
        // centimetres after first touch, that would grow to centimetres and tens of kilonewtons
        // while the real contact off the top edge is already carrying the wheel. Measured in
        // `KerbContactTests` before this existed.
        //
        // A true minimum-translation contact satisfies an identity: the cylinder's support point
        // in the push-out's opposite direction protrudes past the contact point's plane by exactly
        // the reported depth. The swapped-normal contact does not — its depth was measured along a
        // direction its axis no longer names — so the identity is the test. The support point is
        // Jolt's own construction of its rounded cylinder: the inner sharp cylinder's support plus
        // the rounding along the direction.
        const auto into = -contact.axis;
        const auto along = glm::dot(into, spin);
        const auto perpendicular = into - along * spin;
        const auto perpendicularLength = glm::length(perpendicular);

        auto support = wheel.centre + rounding * into;
        if (along > 0.0)
        {
            support += (wheel.halfWidth - rounding) * spin;
        }
        else if (along < 0.0)
        {
            support -= (wheel.halfWidth - rounding) * spin;
        }

        if (perpendicularLength > 1e-9)
        {
            support += (perpendicular / perpendicularLength) * (wheel.radius - rounding);
        }

        const auto protrusion = glm::dot(contact.point - support, contact.axis);
        if (std::abs(protrusion - contact.depth) > obstacleDepthTolerance + 0.02 * contact.depth)
        {
            continue;
        }

        // The hub-height clause: at or above the wheel centre it is a wall, and a wall is the
        // bodywork's. A kerb's top edge is below the hub by construction of "a kerb a tyre can
        // climb", and a sidewall push on a kerb face lower still.
        if (glm::dot(contact.point - wheel.centre, inPlaneUp) >= 0.0)
        {
            continue;
        }

        // The per-face merge: a face split into two triangles reports twice with the same axis,
        // and the deeper of the two is the face's own depth.
        auto merged = false;
        for (auto index = std::uint32_t{0}; index < count; index++)
        {
            if (glm::dot(kept[index].axis, contact.axis) >= obstacleMergeCosine)
            {
                if (contact.depth > kept[index].depth)
                {
                    kept[index] = contact;
                    const auto towardsUp = glm::dot(contact.axis, reference);
                    kept[index].weight = std::clamp((obstacleExclusionCosine - towardsUp) /
                                                        (obstacleExclusionCosine - obstacleFullWeightCosine),
                                                    0.0, 1.0);
                }

                merged = true;
                break;
            }
        }

        if (merged)
        {
            continue;
        }

        if (count < maxObstacleContacts)
        {
            kept[count] = contact;
            // The hand-over weight: 1 beyond thirty degrees from the wheel's up, 0 at the fifteen
            // degree cone, linear in the cosine between. Written on the kept copy only.
            const auto towardsUp = glm::dot(contact.axis, reference);
            kept[count].weight = std::clamp(
                (obstacleExclusionCosine - towardsUp) / (obstacleExclusionCosine - obstacleFullWeightCosine), 0.0, 1.0);
            count++;
        }
    }

    return count;
}

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

void seedTyreGasPressures(const VehicleSetup& setup, VehicleState& state)
{
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        seedTyreGasAtIdealPressure(setup.corners[index].tyre.pressure, state.corners[index].tyre);
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

double damperFrictionShapeAt(const Curve& shape, const double speed)
{
    // A corner that states no shape has no velocity dependence, and the answer is exactly one — not
    // `Curve::at`'s empty-curve zero, which would delete the friction rather than leave it alone.
    return shape.points.empty() ? 1.0 : shape.at(speed);
}

Curve macPhersonStrutFrictionShape()
{
    // Deubel et al., Tribology International 215 (2026) 111328, Fig. 4, design-deflection panel,
    // the side-force column whose 0.5 mm/s value is the shipped 107 N (≈500 N of side force).
    // Newtons as read: 107 at 0.5 mm/s, then 140, 136, 126, 118, 109, 99, 92, 84, 80, 77, 74 at
    // 5, 10, 20, 30, 50, 75, 100, 150, 200, 250 and 300 mm/s. Divided by the 107 and written to
    // three places, which is finer than the reading and coarser than nothing — the figures below
    // are ratios of two readings from one plot, so the third place carries no claim.
    //
    // Both ends are held rather than extrapolated (`Curve::at`), and both are deliberate: below
    // 0.5 mm/s the source measures no sliding at all and the regularisation owns the answer, and
    // above 300 mm/s the source stops because *"the maximum velocity was limited to 300 mm/s, based
    // on the analysis of real SA stroke data, which is regarded sufficient for analysing vertical
    // dynamics on good- and medium-conditioned roads"*. A suspension crossing a kerb goes faster
    // than that and this curve says the last measured thing about it, which is the honest answer and
    // not a good one.
    return Curve{.points = {glm::dvec2(0.0005, 1.000), glm::dvec2(0.005, 1.308), glm::dvec2(0.010, 1.271),
                            glm::dvec2(0.020, 1.178), glm::dvec2(0.030, 1.103), glm::dvec2(0.050, 1.019),
                            glm::dvec2(0.075, 0.925), glm::dvec2(0.100, 0.860), glm::dvec2(0.150, 0.785),
                            glm::dvec2(0.200, 0.748), glm::dvec2(0.250, 0.720), glm::dvec2(0.300, 0.692)}};
}

namespace
{

// The whole friction-carrying damper force, in the three expressions the data selects between. The
// first two are the shipped ones and are written out character for character; the third is the
// velocity shape, which no car states.
[[nodiscard]] double damperForceWithFriction(const CornerSetup& corner, const double viscous, const double velocity)
{
    if (corner.damperFriction <= 0.0)
    {
        return viscous;
    }

    if (corner.damperFrictionShape.points.empty())
    {
        return viscous + corner.damperFriction * std::tanh(velocity / std::max(corner.damperFrictionSpeed, 1e-9));
    }

    // Magnitude times shape times direction, and the direction factor is the *same* regularised
    // `tanh` the shipped law uses. Keeping it is not conservatism: the shape is a steady-state
    // sliding measurement that says nothing about the presliding regime, so something still has to
    // carry the force through zero, and this is the term whose numerical behaviour at 360 Hz this
    // project has already accepted. The shape reads `|velocity|`, so a symmetric shape gives a
    // symmetric force — which is what the source licenses, its rebound-to-compression eccentricity
    // being *"not exceeding 12 % in absolute means"*.
    return viscous + corner.damperFriction * damperFrictionShapeAt(corner.damperFrictionShape, std::abs(velocity)) *
                         std::tanh(velocity / std::max(corner.damperFrictionSpeed, 1e-9));
}

} // namespace

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
                               //
                               // The third branch is the velocity shape (`CornerSetup::damperFrictionShape`), and it is
                               // branched for the same reason and to the same standard: **no car states one**, so every
                               // car in this project takes one of the two expressions above, unchanged. The magnitude
                               // does not scale with velocity; the shape says how the *measured* magnitude does, and
                               // the two are separate because their sources are.
                               .force = damperForceWithFriction(corner, viscous, velocity)};
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

namespace
{

// The C1 fade-in of one Maxwell element around its contact point. Below `contact - half` the
// element is not touching and its input is exactly zero; above `contact + half` its input is the
// deflection past the contact point; between them a polynomial joins the two with matching value
// and slope at both ends.
//
// The source calls for an order-3 polynomial "because of the resulting four constraints". Those
// four constraints — value 0 and slope 0 at the lower end, value `half` and slope 1 at the upper —
// are satisfied by a quadratic, and solving the cubic for them returns a zero cubic term. So this
// is that cubic, written in the form it collapses to rather than in the form that hides it.
[[nodiscard]] double elementInput(const double past, const JounceElement& element, const double half)
{
    const auto beyond = past - element.contactPoint;

    if (half <= 0.0)
    {
        return std::max(0.0, beyond);
    }

    if (beyond <= -half)
    {
        return 0.0;
    }

    if (beyond >= half)
    {
        return beyond;
    }

    const auto through = (beyond + half) / (2.0 * half);
    return half * through * through;
}

} // namespace

TravelStopTerms TravelStop::dynamicTerms(const double past, const double rateOfChange, const double deltaTime,
                                         TravelStopState& state) const
{
    // Out of contact the stop carries nothing and forgets everything. **The forgetting is the
    // source's**: it resets its Maxwell states after each load event, so that the elements attach
    // at their contact points on the next strike rather than carrying a residue from the last one.
    // A friction element is the same argument — a bumper that has let go is not part-way through a
    // stroke.
    if (past <= 0.0)
    {
        state = TravelStopState{};
        return {};
    }

    const auto elastic = rate * std::pow(past, progression) / std::pow(std::max(gap, 1e-6), progression - 1.0);

    // --- the modified Dahl element ---------------------------------------------------------------
    //
    // The state is normalised to ±1 and the saturation is `hysteresis` times the elastic force, so
    // this branch reaches exactly the same force the `tanh` path reaches when it is fully
    // developed. What it does differently is *how it gets there*: over a displacement set by the
    // Dahl coefficient rather than over a velocity set by a smoothing constant. That is the whole
    // substance of the change, because a bump stop's hysteresis loop has a measured **width** in
    // millimetres and no width at all in metres per second.
    auto hysteretic = 0.0;
    if (dahlSigmaMax > 0.0 && hysteresis != 0.0)
    {
        // The source's displacement-dependent coefficient. Their weighting function follows the
        // static characteristic, so `w(x)/w(xref)` is `(past/xref)^progression` here, and the
        // ratio is clamped at one because past the reference deflection they measured nothing.
        auto coefficient = dahlSigmaMax;
        if (dahlSigmaMax > dahlSigmaMin && dahlReference > 0.0)
        {
            const auto weighting = std::min(1.0, std::pow(past / dahlReference, progression));
            coefficient = dahlSigmaMin + std::pow(weighting, dahlSigmaExponent) * (dahlSigmaMax - dahlSigmaMin);
        }

        const auto step = rateOfChange * deltaTime;
        const auto direction = step > 0.0 ? 1.0 : (step < 0.0 ? -1.0 : 0.0);

        state.dahl = std::clamp(state.dahl + coefficient * (1.0 - state.dahl * direction) * step, -1.0, 1.0);
        hysteretic = elastic * hysteresis * state.dahl;
    }
    else if (hysteresis != 0.0)
    {
        hysteretic = elastic * hysteresis * std::tanh(rateOfChange / hysteresisSpeed);
    }

    // --- the Maxwell branch ----------------------------------------------------------------------
    //
    // Each element is stepped by the source's own discrete form: the continuous transfer function
    // `c·s / (s + w)` through the bilinear transform. That is chosen rather than an explicit
    // integration of the same equation because the bilinear map takes the whole left half plane
    // into the unit disc — the element cannot go unstable at any timestep, which an explicit step
    // can and which the source has to constrain its fit to avoid.
    auto dynamic = 0.0;
    for (auto index = std::size_t{0}; index < jounceElementCount; index++)
    {
        const auto& element = elements[index];
        if (element.stiffness <= 0.0 || element.cutoff <= 0.0)
        {
            continue;
        }

        const auto span = element.cutoff * deltaTime + 2.0;
        const auto decay = (element.cutoff * deltaTime - 2.0) / span;
        const auto gain = 2.0 * element.stiffness / span;

        const auto input = elementInput(past, element, elementSmoothing);
        const auto force =
            std::max(0.0, -decay * state.elementForce[index] + gain * (input - state.elementInput[index]));

        state.elementForce[index] = force;
        state.elementInput[index] = input;
        dynamic += force;
    }

    // The viscous share is not here: it is the corner's to solve, at the velocity it arrives at,
    // exactly as for the plain law (`TravelStop::terms`). Every part above is non-negative or
    // bounded by the elastic force, so the explicit part cannot go negative on its own.
    return TravelStopTerms{.explicitForce = elastic + hysteretic + dynamic, .viscousCoefficient = viscousCoefficient(past)};
}

double TravelStop::dynamicForce(const double past, const double rateOfChange, const double deltaTime,
                                TravelStopState& state) const
{
    const auto split = dynamicTerms(past, rateOfChange, deltaTime, state);

    return std::max(0.0, split.explicitForce + split.viscousCoefficient * rateOfChange);
}

TravelStop jounceBumperCandidate(const TravelStop& stop, const double cutoff)
{
    // The source's specimen, read off its own figures and written here in the two dimensionless
    // forms the transfer needs. Contact points as fractions of the reference deflection, and
    // stiffnesses as `c · xref / Fref` — so a stiffness is "how much of the reference force this
    // element carries over the whole reference deflection".
    //
    // Read from Figure 16: contact points at 2.0, 38.0, 49.0, 63.5 and 69.0 mm against a reference
    // deflection of 71 mm — **an inference, not a printed displacement-at-9-kN figure; see
    // `TravelStop::dahlReference`** — and segment gradients of 6.11, 17.27, 23.45, 109.09 and 280.0 N/mm,
    // differenced into per-element stiffnesses. Summed at the reference deflection these carry
    // 1.91 kN of dynamic stiffening against the figure's own envelope of about 2.0 kN there and
    // the paper's stated 2.8 kN peak at full stroke, which is the arithmetic checking out.
    constexpr auto fractions = std::array<double, jounceElementCount>{0.02817, 0.53521, 0.69014, 0.89437, 0.97183};
    constexpr auto stiffnesses = std::array<double, jounceElementCount>{0.04820, 0.08804, 0.04876, 0.67561, 1.34829};

    // This stop's own reference deflection: where its static law reaches the source's 9 kN.
    // `elastic = rate · x^p / gap^(p-1)`, so inverting it is one root.
    const auto scale = std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);
    const auto reference = std::pow(jounceReferenceForce * scale / std::max(stop.rate, 1e-6), 1.0 / stop.progression);

    auto candidate = stop;
    candidate.dahlReference = reference;

    // The saturation: the source limits its Dahl output at half the weighting function's value at
    // the reference deflection, "which is about 0.9 kN for this specimen" against a 9 kN reference
    // force. So the hysteresis half-height is a tenth of the elastic force, which is exactly what
    // this field means — and it lands at the top of BASF's own 10-20% loop-share band for the
    // material, from a completely independent measurement.
    candidate.hysteresis = 0.10;

    // **Placed, not sourced.** The three coefficients of the source's Equation 3 are not published,
    // so the candidate runs the conventional constant-coefficient Dahl (min equal to max) at a
    // transition displacement of a twentieth of the reference deflection. The displacement
    // dependence is implemented and switched off rather than invented.
    candidate.dahlSigmaMin = 20.0 / std::max(reference, 1e-6);
    candidate.dahlSigmaMax = candidate.dahlSigmaMin;
    candidate.dahlSigmaExponent = 1.0;

    // **Placed too**: the source optimises its fade-in half-width and publishes no value. A
    // fiftieth of the reference deflection is well inside the closest pair of contact points,
    // which is what the parameter has to be for the elements to stay distinguishable.
    candidate.elementSmoothing = 0.02 * reference;

    for (auto index = std::size_t{0}; index < jounceElementCount; index++)
    {
        candidate.elements[index] = JounceElement{.contactPoint = fractions[index] * reference,
                                                  .stiffness = stiffnesses[index] * jounceReferenceForce / reference,
                                                  .cutoff = cutoff};
    }

    return candidate;
}

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

    if (corner.damperFrictionShape.points.empty())
    {
        return damper.lengthPerAngle * damper.lengthPerAngle *
               std::max(0.0, curveSlopeAt(corner.damper, damper.velocity) + frictionSlope);
    }

    // With a shape stated the friction is `f · s(|v|) · tanh(v/w)`, so the product rule gives two
    // terms and **the first of them can be negative**:
    //
    //     d/dv = f · [ s'(|v|)·sgn(v)·tanh(v/w) + s(|v|)·(1/w)·sech²(v/w) ]
    //
    // Negative because the source's curve *falls* with speed above its maximum, which is velocity
    // weakening and is a real destabilising slope rather than an artefact — friction that drops as
    // the shaft speeds up is what makes stick-slip possible. It is written here rather than left
    // out precisely because the integration must see it: the `std::max(0.0, ...)` below is the same
    // floor the shipped path has always had, so a corner whose total slope goes negative is stepped
    // explicitly through that tick rather than being handed a negative implicit divisor.
    //
    // The speed derivative is differenced off the shape curve at the same step every other curve
    // here is differenced at. That step is 0.1 mm/s, a fifth of the first knot's spacing, so unlike
    // the `tanh` term this one is resolvable by differencing and does not need writing out.
    const auto speed = std::abs(damper.velocity);
    const auto direction = damper.velocity < 0.0 ? -1.0 : 1.0;
    const auto shape = damperFrictionShapeAt(corner.damperFrictionShape, speed);
    const auto shapeSlope = curveSlopeAt(corner.damperFrictionShape, speed) * direction * shaped;

    return damper.lengthPerAngle * damper.lengthPerAngle *
           std::max(0.0, curveSlopeAt(corner.damper, damper.velocity) + shape * frictionSlope +
                             corner.damperFriction * shapeSlope);
}

double solveCornerRate(const CornerRateStep& step)
{
    // The affine intercept of the damper's linearisation, `G_d(q̇₀) − G_d'(q̇₀)·q̇₀`, added to the
    // explicit force that already carries `G_d(q̇₀)`: the slope's own share of the old-rate force is
    // taken back out of the numerator, because the divisor is about to apply it at the new rate.
    // Branched rather than added so that a corner with no coefficient runs the expression it always
    // ran, bit for bit, rather than one that happens to add a zero to it.
    const auto numerator = step.damperCoefficient > 0.0
                               ? step.generalisedForce + step.damperCoefficient * step.previousRate
                               : step.generalisedForce;

    auto rate = step.previousRate + (numerator / step.generalisedInertia) * step.deltaTime;
    rate /= 1.0 + ((step.damperCoefficient + step.stopCoefficient) / step.generalisedInertia) * step.deltaTime;

    return rate;
}

RangeConstraintSolution constrainCornerRange(const RangeConstraintStep& step)
{
    // The coordinate the step would land on, in the same arithmetic pass three advances it with, so
    // that "inside" here and "inside" after the advance are one statement. A landing exactly on a
    // limit is inside: the corner arrives at its own rate, and the next tick's free step from there
    // is what decides whether it is held.
    const auto predicted = step.previousAngle + step.rate * step.deltaTime;

    if (predicted > step.bumpAngle)
    {
        return RangeConstraintSolution{.rate = (step.bumpAngle - step.previousAngle) / step.deltaTime,
                                       .limit = RangeLimit::Bump};
    }

    if (predicted < step.droopAngle)
    {
        return RangeConstraintSolution{.rate = (step.droopAngle - step.previousAngle) / step.deltaTime,
                                       .limit = RangeLimit::Droop};
    }

    return RangeConstraintSolution{.rate = step.rate, .limit = RangeLimit::None};
}

double attachmentMobility(const double mass, const glm::dmat3& worldInverseInertia, const glm::dvec3& armFrom,
                          const glm::dvec3& jacobianFrom, const glm::dvec3& armTo, const glm::dvec3& jacobianTo)
{
    return glm::dot(jacobianTo, jacobianFrom) / mass +
           glm::dot(glm::cross(armTo, jacobianTo), worldInverseInertia * glm::cross(armFrom, jacobianFrom));
}

glm::dvec3 attachmentAcceleration(const ChassisMotion& chassis, const glm::dvec3& arm)
{
    return chassis.acceleration + glm::cross(chassis.angularAcceleration, arm) +
           glm::cross(chassis.angularVelocity, glm::cross(chassis.angularVelocity, arm));
}

std::expected<double, std::string> cornerGeneralisedInertia(const CornerSetup& corner, const double wishboneAngle,
                                                            const double rackTravel, const bool geometricLoadPath)
{
    const auto solved = solveCornerWithJacobian(corner.hardpoints, wishboneAngle, rackTravel);
    if (!solved)
    {
        return std::unexpected(solved.error());
    }

    return geometricLoadPath
               ? corner.unsprungMass * glm::dot(solved->wheelCentrePerAngle, solved->wheelCentrePerAngle)
               : corner.unsprungMass * solved->travelPerAngle * solved->travelPerAngle;
}

namespace
{

// The corner solve's Jacobian at an angle, on the load path's own statement of it: the whole vector on
// the geometric path, the vertical component alone on the springs path.
[[nodiscard]] std::expected<glm::dvec3, std::string> cornerJacobianAt(const CornerSetup& corner, const double wishboneAngle,
                                                                      const double rackTravel, const bool geometricLoadPath)
{
    const auto solved = solveCornerWithJacobian(corner.hardpoints, wishboneAngle, rackTravel);
    if (!solved)
    {
        return std::unexpected(solved.error());
    }

    return geometricLoadPath ? solved->wheelCentrePerAngle : glm::dvec3(0.0, solved->travelPerAngle, 0.0);
}

// The wheel centre and the Jacobian at an angle together, on the load path's own statement of them.
struct CornerPoint
{
    glm::dvec3 centre{0.0};
    glm::dvec3 jacobian{0.0};
};

[[nodiscard]] std::expected<CornerPoint, std::string> cornerPointAt(const CornerSetup& corner, const double wishboneAngle,
                                                                    const double rackTravel, const bool geometricLoadPath)
{
    const auto solved = solveCornerWithJacobian(corner.hardpoints, wishboneAngle, rackTravel);
    if (!solved)
    {
        return std::unexpected(solved.error());
    }

    return CornerPoint{.centre = geometricLoadPath ? solved->wheelCentre : glm::dvec3(0.0, solved->wheelCentre.y, 0.0),
                       .jacobian = geometricLoadPath ? solved->wheelCentrePerAngle : glm::dvec3(0.0, solved->travelPerAngle, 0.0)};
}

} // namespace

double cornerInertiaSlope(const CornerSetup& corner, const double wishboneAngle, const double rackTravel,
                          const bool geometricLoadPath)
{
    const auto ahead = cornerGeneralisedInertia(corner, wishboneAngle + inertiaSlopeStep, rackTravel, geometricLoadPath);
    const auto behind = cornerGeneralisedInertia(corner, wishboneAngle - inertiaSlopeStep, rackTravel, geometricLoadPath);

    if (ahead && behind)
    {
        return (ahead.value() - behind.value()) / (2.0 * inertiaSlopeStep);
    }

    const auto here = cornerGeneralisedInertia(corner, wishboneAngle, rackTravel, geometricLoadPath);
    if (!here)
    {
        return 0.0;
    }

    if (ahead)
    {
        return (ahead.value() - here.value()) / inertiaSlopeStep;
    }

    if (behind)
    {
        return (here.value() - behind.value()) / inertiaSlopeStep;
    }

    return 0.0;
}

glm::dvec3 cornerJacobianCurvature(const CornerSetup& corner, const double wishboneAngle, const double rackTravel,
                                   const bool geometricLoadPath)
{
    const auto ahead = cornerJacobianAt(corner, wishboneAngle + inertiaSlopeStep, rackTravel, geometricLoadPath);
    const auto behind = cornerJacobianAt(corner, wishboneAngle - inertiaSlopeStep, rackTravel, geometricLoadPath);

    if (ahead && behind)
    {
        return (ahead.value() - behind.value()) / (2.0 * inertiaSlopeStep);
    }

    const auto here = cornerJacobianAt(corner, wishboneAngle, rackTravel, geometricLoadPath);
    if (!here)
    {
        return glm::dvec3(0.0);
    }

    if (ahead)
    {
        return (ahead.value() - here.value()) / inertiaSlopeStep;
    }

    if (behind)
    {
        return (here.value() - behind.value()) / inertiaSlopeStep;
    }

    return glm::dvec3(0.0);
}

KinematicReaction unsprungKinematicReaction(const UnsprungKinematics& wheel, const ChassisMotion& chassis,
                                            const glm::dvec3& gravity)
{
    const auto offset = wheel.arm - wheel.designArm;
    const auto& omega = chassis.angularVelocity;
    const auto& alpha = chassis.angularAcceleration;

    // The mass's acceleration beyond its design point's rigid motion and beyond the coordinate's own
    // `R·C'·q̈`, which the force pass applies on its own.
    const auto relative = glm::cross(alpha, offset) + glm::cross(omega, glm::cross(omega, offset)) +
                          2.0 * glm::cross(omega, wheel.jacobian * wheel.rate) + wheel.curvature * (wheel.rate * wheel.rate);

    // The design point's own acceleration against free fall: what the body's tensor already reacts at
    // the design point, and what the couple puts back there from the wheel's true point.
    const auto design = chassis.acceleration + glm::cross(alpha, wheel.designArm) +
                        glm::cross(omega, glm::cross(omega, wheel.designArm)) - gravity;

    return KinematicReaction{.force = -wheel.mass * relative, .couple = -wheel.mass * glm::cross(offset, design)};
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
            const BrakeCommand& brakes, const AmbientConditions& ambient,
            const std::span<const DynamicObstacle> dynamicObstacles)
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
    auto poses = std::array<WheelPose, cornerCount>{};

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

        // And the lean the same bushes are holding, about the chassis's forward axis, branched the
        // same way for the same inertness. Applied after the steer twist; the two rotations are
        // about different axes, so their order matters only at second order in angles of a
        // hundredth of a radian.
        if (corner.lateralForceCamber != 0.0)
        {
            applyComplianceCamber(corner.hardpoints, result.corners[index].suspension,
                                  state.corners[index].complianceCamber);
        }

        // And the fore-aft deflection they are holding — the translation third of the same bush,
        // branched the same way for the same inertness. Applied after the two rotations; a
        // translation commutes with both, so the order is a convention rather than a choice.
        if (corner.longitudinalForceRecession != 0.0)
        {
            applyComplianceRecession(corner.hardpoints, result.corners[index].suspension,
                                     state.corners[index].complianceRecession);
        }

        // The wheel, in the world. The suspension solved it in chassis coordinates; the chassis
        // says where those are.
        const auto& suspension = result.corners[index].suspension;
        const auto outboard = outboardSign(corner.hardpoints.side);

        poses[index] = WheelPose{
            .centre = bodyToWorld(state.chassis, suspension.wheelCentre),
            .spinAxis = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(outboard, 0.0, 0.0)),
            .forward = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(0.0, 0.0, 1.0)),
            .radius = corner.hardpoints.wheelRadius};

        geometries[index] = contactPatchSamples(poses[index], setup.sampling);
        allOrigins.insert(allOrigins.end(), geometries[index].origins.begin(), geometries[index].origins.end());
        allDirections.insert(allDirections.end(), geometries[index].directions.begin(),
                             geometries[index].directions.end());
    }

    auto allHits = std::vector<SurfaceHit>{};
    world.castRays(allOrigins, allDirections, setup.sampling.searchDistance * 2.0, allHits);

    // --- and the tyres' side view of the road, when the car states it ---------------------------
    //
    // The kerb-contact path: each tyre as a cylinder of its own radius and section width, collided
    // against the same mesh the rays sample, so that a face the rays cannot see — a 150 mm kerb, a
    // wall — pushes on the carcass instead of arriving as a road step under the leading samples.
    // Behind the switch and outside the ray batch, so a car with it off makes no second query and
    // runs the tick it always ran. docs/kerb-contact-brief.md.
    auto obstacleHits = std::vector<ObstacleContact>{};
    auto obstacleCounts = std::vector<std::uint32_t>{};
    auto cylinders = std::array<WheelCylinder, cornerCount>{};

    if (setup.kerbContact)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            cylinders[index] = WheelCylinder{.centre = poses[index].centre,
                                             .spinAxis = poses[index].spinAxis,
                                             .radius = setup.corners[index].hardpoints.wheelRadius,
                                             .halfWidth = 0.5 * setup.corners[index].tyreSectionWidth};
        }

        world.collideCylinders(cylinders, obstacleQueryLimit, obstacleHits, obstacleCounts);
    }

    // The world's own table, not the generator's. A `SurfaceHit` reports an index into the mesh the
    // world was built from, so reading it against any other table is reading the wrong row.
    const auto& materials = world.materials();
    auto consumed = std::size_t{0};
    auto consumedObstacles = std::size_t{0};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto count = geometries[index].origins.size();
        const auto slice = std::vector<SurfaceHit>(allHits.begin() + static_cast<std::ptrdiff_t>(consumed),
                                                   allHits.begin() + static_cast<std::ptrdiff_t>(consumed + count));
        consumed += count;

        result.corners[index].patch = aggregateContactPatch(geometries[index], slice, materials, setup.sampling);

        // The cylinder's contacts for this wheel, through the exclusion rule and the per-face merge.
        // Read against this wheel's own spin axis, because "the wheel plane's down" is a property of
        // the wheel and a cambered wheel's is not the world's.
        if (setup.kerbContact)
        {
            const auto found = obstacleCounts[index];
            const auto obstacles = std::span<const ObstacleContact>(obstacleHits).subspan(consumedObstacles, found);
            consumedObstacles += found;

            const auto& patch = result.corners[index].patch;
            result.corners[index].obstacleCandidates = found;
            result.corners[index].obstacleCount =
                keepObstacleContacts(obstacles, cylinders[index], patch.inContact ? patch.normal : glm::dvec3(0.0),
                                     result.corners[index].obstacles);
        }
    }

    // --- Pass two: what every element contributes ----------------------------------------------
    auto chassisForces = ForceAccumulator{};
    chassisForces.force = glm::dvec3(0.0, -earthGravity * state.chassis.mass, 0.0);

    const auto worldCentreOfMass = state.chassis.position;
    auto cornerDamping = std::array<double, cornerCount>{};
    auto stopDamping = std::array<double, cornerCount>{};
    auto cornerRate = std::array<double, cornerCount>{};

    // The pass is two loops since 2026-09-08 latest of all (docs/frame-acceleration-brief.md): the
    // first solves every corner's elements, its tyre and its explicit generalised force; the second
    // takes the four velocity steps — coupled through the chassis's own response to their reactions
    // when the car states `frameAcceleration` — and hands the chassis what each corner did. What the
    // second needs of the first is carried here rather than recomputed. **The chassis accumulator is
    // written in the second loop only, in the order the one loop wrote it** — a corner's kerb faces,
    // then the tyre's load, the corner's reaction, the tyre's in-plane forces and its aligning moment,
    // corner by corner, then the air — so that a car with the switch off sums the same operands in
    // the same order and reads the same bits.
    struct AppliedForce
    {
        glm::dvec3 force{0.0};
        glm::dvec3 point{0.0};
    };

    struct CornerScratch
    {
        DamperForceSolution damper;
        TravelStopTerms bumpTerms;
        TravelStopTerms droopTerms;
        glm::dvec3 normal{0.0, 1.0, 0.0};
        glm::dvec3 wheelWorld{0.0};
        glm::dvec3 contactPoint{0.0};
        std::array<AppliedForce, maxObstacleContacts> obstacleForces{};
        std::uint32_t obstacleForceCount = 0;
        glm::dvec3 curvature{0.0}; // C'' at the tick's angle, body frame, on the load path's statement
        glm::dvec3 centre{0.0};    // the wheel centre on the same statement, body frame
    };

    auto scratch = std::array<CornerScratch, cornerCount>{};

    // The forces the chassis is known to feel before any corner has decided its acceleration — the
    // road's, the kerb faces' and the air's, about the centre of mass — for the corners' reading of
    // the chassis's motion. Gravity is deliberately not in it: the corner reads gravity as the
    // acceleration it is, so that a car with nothing else on it reads exactly g rather than
    // `(g·M)/M`. A separate accumulator, for the reason above.
    auto knownForces = ForceAccumulator{};

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

        // The stops, split the way the corner integrates them (2026-09-08,
        // docs/stop-element-brief.md): the part known now — the elastic law, and the sourced branch
        // where a car states one — goes into the generalised force below exactly as the stop force
        // always did; the viscous coefficient goes into the implicit divisor beside the damper's
        // slope, and the viscous share is read off the velocity the step arrives at, at the velocity
        // step below, where the two forces are finished. Until then `forces.bumpStop` and
        // `forces.droopStop` carry the explicit part alone.
        //
        // The bump stop's dynamic branch, where a car states one. **The branch is on the setup's
        // own data and no car in this project states it**, so every shipped car takes the second
        // expression and never touches `bumpStopState`. That is the inertness proof, and it is a
        // property of the data rather than of a flag somebody has to remember to leave alone.
        const auto bumpTerms =
            corner.bumpStop.statesDynamicBranch()
                ? corner.bumpStop.dynamicTerms(compression - corner.bumpStop.gap, solution.damperVelocity, deltaTime,
                                               state.corners[index].bumpStopState)
                : corner.bumpStop.terms(compression - corner.bumpStop.gap, solution.damperVelocity);

        // The droop stop has no branch: on a strut it is the damper topping out, and the source
        // measured a jounce bumper. Its compression is the shaft's extension and its velocity the
        // extension rate, so both arguments are negated on the way in and its force on the way out.
        const auto droopTerms =
            corner.droopStop.terms(-compression - corner.droopStop.gap, -solution.damperVelocity);

        solution.forces.bumpStop = bumpTerms.explicitForce;
        solution.forces.droopStop = -droopTerms.explicitForce;

        // The engaged stop's viscous coefficient on the shaft, N·s/m. At most one of the two is in
        // contact — both gaps are positive — so the sum is whichever it is, and a literal zero when
        // neither is.
        stopDamping[index] = bumpTerms.viscousCoefficient + droopTerms.viscousCoefficient;

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

        // **The carcass is an air spring, so its rate follows the air.** Linear in gauge pressure —
        // Rhyne's constant-belt model — and exactly 1.0 at the ideal pressure the car's own
        // `tireVerticalRate` is quoted at, which is what makes the switch inert. This is the coupling
        // Dominic will feel, because it changes ride and it changes the contact patch.
        const auto verticalRate =
            setup.tyrePressure ? corner.tireVerticalRate *
                                     tyrePressureVerticalRateScale(corner.tyre.pressure, state.corners[index].tyre)
                               : corner.tireVerticalRate;

        solution.forces.tireVertical =
            solution.patch.inContact ? std::max(0.0, verticalRate * solution.patch.penetration * normalOfVertical +
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
            // **One call, because the tyre is one tyre.** `tyreStateGrip` is the only place the
            // thermal and pressure factors meet, so neither subsystem assigns `gripScale` on its own
            // and the answer cannot depend on which stepper ran last. It returns exactly 1.0 with
            // both switches off, which is what lets this multiply unconditionally.
            auto tyre = corner.tyre;
            tyre.gripScale *=
                tyreStateGrip(corner.tyre, state.corners[index].tyre, setup.tyreThermal, setup.tyrePressure);

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
                                             // **The beads alone once the air is its own node**, or
                                             // the whole lumped path when it is not. Leaving the
                                             // whole path in with `tyrePressure` on would carry the
                                             // rim to the carcass twice — through here and again
                                             // through the gas — and roughly double a term the brake
                                             // work measured as worth about a degree.
                                             .wheelConductance =
                                                 setup.brakeThermal
                                                     ? corner.wheel.toTyre *
                                                           (setup.tyrePressure ? 1.0 - corner.wheel.cavityShare : 1.0)
                                                     : 0.0,
                                             .ambient = ambient},
                            deltaTime);
        }

        // --- and the air inside it ---
        //
        // **Its own switch and its own call, after the carcass has moved.** The gas reads the carcass
        // and the rim and generates nothing itself, so it is a one-node problem with two neighbours
        // and it belongs outside the tread's stepper — the two models switch independently and must
        // not acquire each other's switch.
        //
        // The rim reaches it through the wheel well, on the same condition the carcass's own rim term
        // has: only when there is a brake model to supply a temperature.
        if (setup.tyrePressure)
        {
            stepTyreGas(corner.tyre.pressure, state.corners[index].tyre,
                        TyreThermalInput{.wheelTemperature = state.corners[index].wheelTemperature,
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
        if (corner.lateralForceSteer != 0.0 || corner.lateralForceCamber != 0.0)
        {
            const auto sideways = glm::dot(glm::inverse(state.chassis.orientation) *
                                               (solution.contact.tyre.lateral * solution.contact.lateral),
                                           glm::dvec3(1.0, 0.0, 0.0));

            // Each written only when its coefficient is stated, so a car stating one of the two
            // leaves the other's state at the exact 0.0 it was constructed with — `0.0 * sideways`
            // is `-0.0` for a negative force, and a state hash reads bits.
            if (corner.lateralForceSteer != 0.0)
            {
                state.corners[index].complianceSteer = corner.lateralForceSteer * sideways;
            }

            if (corner.lateralForceCamber != 0.0)
            {
                state.corners[index].complianceCamber = corner.lateralForceCamber * sideways;
            }
        }

        // The recession the same bushes will be holding, from the longitudinal force resolved into
        // the **body's** own forward axis for the same banked-road reason the lateral one is. A
        // wheel in the air relaxes to nothing here too. Through `recessionDisplacement` rather
        // than a bare product, so a kerb strike's one-tick force spike saturates at the map's
        // stated travel instead of teleporting the hub.
        if (corner.longitudinalForceRecession != 0.0)
        {
            const auto fore = glm::dot(glm::inverse(state.chassis.orientation) *
                                           (solution.contact.tyre.longitudinal * solution.contact.forward),
                                       glm::dvec3(0.0, 0.0, 1.0));

            state.corners[index].complianceRecession = recessionDisplacement(corner.longitudinalForceRecession, fore);
        }

        // --- the kerb-contact path: what the road's faces are pushing on this tyre ---------------
        //
        // Each surviving contact is a spring on the carcass with Coulomb friction at the point of
        // contact, and it reaches three things: the chassis, whole, at the contact point with its
        // moment — the same free body the tyre's own load takes, which puts the whole road force on
        // the body and lets the corner's coordinate decide how the linkage reacts it; the corner's
        // one degree of freedom, through the same Jacobian the tyre force takes on whichever load
        // path the car is on, so it passes through the spring, the damper and the stops; and the
        // wheel's spin, through the friction's moment about the spin axis, which is how a driven
        // wheel climbs a kerb from rest.
        //
        // **What it does not reach**: `forces.tireVertical`, the belt bed, `patch.normal`,
        // `patch.penetration` and `evaluateTyre`. The vertical load is also the tyre model's load,
        // the rolling-resistance load, the tread's thermal load and the `Tyre Load` channel, and a
        // kerb face is not the road under the tread. The share that reached the corner's coordinate
        // is reported on its own channel instead.
        //
        // Everything here is behind the switch *and* behind a non-empty contact set, and it adds to
        // the accumulators after the fact, so a car with the path off — or on and on flat road —
        // runs the expressions it always ran, to the bit.
        auto obstacleGeneralised = 0.0;

        if (setup.kerbContact && solution.obstacleCount > 0)
        {
            // The signed spin axis, `upright · (+1, 0, 0)` on every corner: both wheels of a car
            // going forward turn the same way about it, which is the sign `wheelSpeed` carries and
            // the axis the driveline reaction is taken about.
            const auto spin = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(1.0, 0.0, 0.0));
            const auto omega = state.corners[index].wheelSpeed * spin;
            const auto bodyUp = state.chassis.orientation * glm::dvec3(0.0, 1.0, 0.0);
            const auto toBody = glm::inverse(state.chassis.orientation);

            auto totalNormal = 0.0;
            auto largestNormal = 0.0;
            auto largestAxis = glm::dvec3(0.0, 1.0, 0.0);
            auto spinTorque = 0.0;

            for (auto slot = std::uint32_t{0}; slot < solution.obstacleCount; slot++)
            {
                const auto& contact = solution.obstacles[slot];

                // Which surface of the tyre the axis leaves through. Zero is the tread — the axis
                // lies in the wheel plane — and one is the sidewall, along the spin axis; the rate
                // and the friction are blended linearly in between, across the shoulder. The
                // tread's rate is the vertical rate the bottom grid uses, pressure scale included.
                const auto sidewall = std::clamp(std::abs(glm::dot(contact.axis, spin)), 0.0, 1.0);
                const auto rate = verticalRate + (corner.tireLateralRate - verticalRate) * sidewall;

                const auto material =
                    contact.surface < materials.size() ? materials[contact.surface] : SurfaceMaterial{};
                const auto treadFriction = corner.tyre.longitudinalPeak * material.gripMultiplier;
                const auto friction = treadFriction + (corner.sidewallFriction - treadFriction) * sidewall;

                // The carcass's closing speed along the axis is the hub's, which is what the
                // vertical path reads too; spin does not compress a carcass. The spring cannot pull.
                // Scaled by the hand-over weight: full strength clear of the cone, fading to nothing
                // at it, so the seam between this contact and the grid's is a ramp and not a step.
                const auto closing = -glm::dot(wheelVelocity, contact.axis);
                const auto normalForce =
                    contact.weight * std::max(0.0, rate * contact.depth + corner.tireVerticalDamping * closing);

                if (!(normalForce > 0.0))
                {
                    continue;
                }

                // Coulomb friction against the tyre's surface velocity at the contact point, spin
                // included, in the plane of the contact. Regularised through `tanh` for
                // `obstacleFrictionSpeed`'s reason.
                const auto arm = contact.point - wheelWorld;
                const auto surfaceVelocity = wheelVelocity + glm::cross(omega, arm);
                const auto sliding = surfaceVelocity - glm::dot(surfaceVelocity, contact.axis) * contact.axis;
                const auto slidingSpeed = glm::length(sliding);
                const auto frictionForce =
                    slidingSpeed > 1e-12
                        ? -(friction * normalForce * std::tanh(slidingSpeed / obstacleFrictionSpeed) / slidingSpeed) *
                              sliding
                        : glm::dvec3(0.0);

                const auto force = normalForce * contact.axis + frictionForce;

                // Onto the chassis in the second loop, in this place in its order; into the
                // corners' reading of the chassis's motion now.
                scratch[index].obstacleForces[scratch[index].obstacleForceCount++] =
                    AppliedForce{.force = force, .point = contact.point};
                knownForces.addForceAtPoint(force, contact.point, worldCentreOfMass);

                // Into the corner's own coordinate. On the geometric path the contact point is a
                // material point of the wheel at `arm` from the hub, so its Jacobian is the wheel
                // centre's plus the upright's rotation rate crossed with that arm — the construction
                // `patchPerAngle` uses for the tyre's own patch. On the springs path it is the
                // body-up component through the wheel's vertical Jacobian, as the tyre load is.
                if (setup.geometricLoadPath)
                {
                    const auto contactPerAngle =
                        suspension.wheelCentrePerAngle + glm::cross(suspension.uprightRatePerAngle, toBody * arm);

                    obstacleGeneralised += glm::dot(toBody * force, contactPerAngle);
                }
                else
                {
                    obstacleGeneralised += glm::dot(force, bodyUp) * suspension.travelPerAngle;
                }

                spinTorque += glm::dot(glm::cross(arm, frictionForce), spin);

                totalNormal += normalForce;
                if (normalForce > largestNormal)
                {
                    largestNormal = normalForce;
                    largestAxis = contact.axis;
                }
            }

            solution.forces.obstacleTravel =
                std::abs(suspension.travelPerAngle) > 1e-12 ? obstacleGeneralised / suspension.travelPerAngle : 0.0;
            solution.obstacleNormalForce = totalNormal;
            solution.obstacleAxisElevation = totalNormal > 0.0 ? std::asin(std::clamp(largestAxis.y, -1.0, 1.0)) : 0.0;
            solution.obstacleSpinTorque = spinTorque;
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

        // The kerb-contact path's share, added after the fact for the reason given where it was
        // computed: the expressions above are the ones every figure in docs/ was measured on.
        if (setup.kerbContact && solution.obstacleCount > 0)
        {
            solution.generalisedForce += obstacleGeneralised;
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

        // --- the corner's configuration-dependent inertia (2026-09-08 latest of all (c)) -------
        //
        // The inertia above varies with travel, and Lagrange's equation from `T = ½·I(q)·q̇²` is
        // `I·q̈ + ½·I'(q)·q̇² = Q` — the `−m_u·(C'·C'')·q̇²` the unsprung mass's own acceleration
        // carries along a curving, stretching coordinate (docs/nonlinear-geometry-brief.md). Without
        // it a wheel coasting along its arc gained or lost world speed for no force: 4.5 % of its
        // weight moment at the 60 mm dip's 5 rad/s, under 1 % elsewhere. Explicit, at the rate the
        // tick starts with — its slope in the rate is `I'·q̇₀`, a thousandth of the inertia over a
        // tick on this car, and a coefficient that could go negative has no business in the implicit
        // divisor. Independent of the chassis's motion, so it neither touches nor duplicates the
        // frame term below. Exactly `−0.0` where `|C'|` does not vary, which is what keeps a corner
        // with a constant Jacobian on the old bits.
        if (setup.nonlinearGeometry)
        {
            solution.inertiaSlope = cornerInertiaSlope(corner, state.corners[index].wishboneAngle,
                                                       index < 2 ? rackTravel : 0.0, setup.geometricLoadPath);
            const auto rate = state.corners[index].wishboneRate;
            const auto geometry = -0.5 * solution.inertiaSlope * rate * rate;
            solution.generalisedForce += geometry;
            solution.forces.geometry =
                std::abs(suspension.travelPerAngle) > 1e-12 ? geometry / suspension.travelPerAngle : 0.0;
        }

        // And the Jacobian's own derivative, for the chassis's side of the same acceleration
        // (2026-09-08 latest of all (d)): the same pair of solves the slope came from.
        if (setup.kinematicReaction)
        {
            scratch[index].curvature = cornerJacobianCurvature(corner, state.corners[index].wishboneAngle,
                                                               index < 2 ? rackTravel : 0.0, setup.geometricLoadPath);
            scratch[index].centre = setup.geometricLoadPath ? suspension.wheelCentre
                                                            : glm::dvec3(0.0, suspension.wheelCentre.y, 0.0);
        }

        // The damper's contribution to the corner's damping, as a coefficient rather than a force,
        // so the integration below can solve it instead of stepping towards it. Damper rates are
        // exactly where an explicit treatment falls over. Since step 9 the coefficient reads the
        // damper element's own Jacobian and velocity — the same bits, through the same law.
        cornerDamping[index] = damperDampingCoefficient(corner, damper);

        // What the second loop needs of this one.
        scratch[index].damper = damper;
        scratch[index].bumpTerms = bumpTerms;
        scratch[index].droopTerms = droopTerms;
        scratch[index].normal = normal;
        scratch[index].wheelWorld = wheelWorld;
        scratch[index].contactPoint = solution.patch.inContact ? solution.patch.centre : wheelWorld;

        // The road's part of what the chassis is known to feel this tick, for the corners' reading
        // of its motion: the load along the normal at the patch, the in-plane forces at the patch and
        // the aligning couple, as the second loop will hand them to the chassis.
        knownForces.addForceAtPoint(solution.forces.tireVertical * normal, scratch[index].contactPoint,
                                    worldCentreOfMass);
        knownForces.addForceAtPoint(solution.contact.tyre.longitudinal * solution.contact.forward +
                                        solution.contact.tyre.lateral * solution.contact.lateral,
                                    solution.patch.centre, worldCentreOfMass);
        knownForces.torque += solution.contact.tyre.aligningMoment * solution.patch.normal;
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

    // The air, computed here so the corners can read it and applied to the chassis after them, in
    // the place and the order it always was. Drag opposes the direction of travel, lift acts along
    // the world's vertical, and each surface's force goes on at its own point, because a splitter
    // and a rear wing pitch the car opposite ways and that is most of what an aero balance *is*.
    auto aeroForces = std::vector<AppliedForce>{};
    aeroForces.reserve(setup.aero.size());

    const auto airspeed = glm::length(state.chassis.linearVelocity);
    if (airspeed > 1e-6)
    {
        const auto flow = state.chassis.linearVelocity / airspeed;
        const auto pressure = 0.5 * setup.airDensity * airspeed * airspeed;

        for (const auto& surface : setup.aero)
        {
            const auto force = -pressure * surface.dragArea * flow + glm::dvec3(0.0, pressure * surface.liftArea, 0.0);
            const auto point = bodyToWorld(state.chassis, surface.centre);

            aeroForces.push_back(AppliedForce{.force = force, .point = point});
            knownForces.addForceAtPoint(force, point, worldCentreOfMass);
        }
    }

    // --- the chassis's motion, as the corners read it -------------------------------------------
    //
    // (2026-09-08 latest of all, docs/frame-acceleration-brief.md.) Each corner's coordinate is
    // relative to the chassis, so the unsprung mass's absolute acceleration is its attachment
    // point's plus its own relative one, and Newton's law for it projected onto the corner's
    // Jacobian carries `−m_u·(R·C')·a_attach` beside the weight. The attachment point's
    // acceleration is the chassis's linear acceleration, the angular acceleration crossed with the
    // arm and the centripetal term — everything a body-fixed point does — from the forces known
    // before the corners decide, plus the four corners' own reactions, which are solved together
    // with the corners below. Gravity as an acceleration, the road, the kerb faces and the air are
    // known here; the bodywork's contact impulses and the driveline's reaction torque are applied to
    // the chassis after this pass and are not read.
    //
    // Nothing here runs with the switch off: the corner solve below is then the one loop's
    // arithmetic, operand for operand.
    auto jacobianWorld = std::array<glm::dvec3, cornerCount>{};
    auto armWorld = std::array<glm::dvec3, cornerCount>{};
    auto knownAcceleration = std::array<glm::dvec3, cornerCount>{};
    auto frameKnown = std::array<double, cornerCount>{};
    auto reducedInertia = std::array<double, cornerCount>{};
    auto linearResponse = std::array<glm::dvec3, cornerCount>{};
    auto angularResponse = std::array<glm::dvec3, cornerCount>{};
    auto mobility = std::array<std::array<double, cornerCount>, cornerCount>{};

    // The chassis's side of the unsprung masses' acceleration (2026-09-08 latest of all (d),
    // docs/chassis-kinematic-reaction-brief.md): each wheel's kinematics for `unsprungKinematicReaction`,
    // and what the four reactions do to the chassis's acceleration — solved inside the same sweep as
    // the frame term, because their offset terms read the chassis's own acceleration and the frame
    // term reads theirs. Zero everywhere with the switch off.
    const auto predictChassis = setup.frameAcceleration || setup.kinematicReaction;
    auto rateChange = std::array<double, cornerCount>{}; // Δq̇ per corner, the latest, rad/s
    const auto gravityVector = glm::dvec3(0.0, -earthGravity, 0.0);
    auto motion = ChassisMotion{};
    auto inverseInertia = glm::dmat3(0.0);
    auto wheelKinematics = std::array<UnsprungKinematics, cornerCount>{};
    auto kinematic = std::array<KinematicReaction, cornerCount>{};
    auto kinematicLinear = glm::dvec3(0.0);  // the four kinematic reactions' share of the chassis's acceleration
    auto kinematicAngular = glm::dvec3(0.0); // and of its angular acceleration

    if (predictChassis)
    {
        const auto omega = angularVelocity(state.chassis);
        inverseInertia = worldInverseInertia(state.chassis);

        // The angular acceleration is Euler's, `I⁻¹·(τ − ω × L)`: the integrator carries the
        // gyroscopic term through the tensor turning with the body, so a prediction without it
        // would be short by exactly that.
        motion = ChassisMotion{.acceleration = gravityVector + knownForces.force / state.chassis.mass,
                               .angularVelocity = omega,
                               .angularAcceleration =
                                   inverseInertia * (knownForces.torque - glm::cross(omega, state.chassis.angularMomentum))};

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto& suspension = result.corners[index].suspension;

            // The direction the unsprung mass moves in per unit of the coordinate, in the world: the
            // wheel centre's whole Jacobian on the geometric path, and on the springs path the
            // world's vertical through the vertical Jacobian — the statement that path's inertia
            // and its reaction on the chassis already make.
            jacobianWorld[index] = setup.geometricLoadPath
                                       ? state.chassis.orientation * suspension.wheelCentrePerAngle
                                       : glm::dvec3(0.0, suspension.travelPerAngle, 0.0);
            armWorld[index] = scratch[index].wheelWorld - worldCentreOfMass;

            wheelKinematics[index] = UnsprungKinematics{
                .mass = corner.unsprungMass,
                .jacobian = jacobianWorld[index],
                .curvature = state.chassis.orientation * scratch[index].curvature,
                .arm = armWorld[index],
                .designArm = bodyToWorld(state.chassis, corner.hardpoints.wheelCentre) - worldCentreOfMass,
                .rate = state.corners[index].wishboneRate};

            // What a unit force along this corner's Jacobian at its wheel centre does to the body:
            // its linear acceleration and its angular one.
            linearResponse[index] = jacobianWorld[index] / state.chassis.mass;
            angularResponse[index] = inverseInertia * glm::cross(armWorld[index], jacobianWorld[index]);
        }
    }

    // What the four kinematic reactions, evaluated at a chassis motion, do to the chassis's own
    // acceleration: the whole of what the corners and the reactions themselves then read back. The
    // rate they read is the tick's **solved** rate, `q̇₀ + Δq̇`, not the one it started with: the
    // coordinate advances by `q̇₁·dt` (the step is semi-implicit), so the wheel's momentum along the
    // turning, curving Jacobian changes by `m_u·(ω × R·C' + R·C''·q̇₁)·q̇₁·dt`, and an impulse read at
    // `q̇₀` would miss `m_u·R·C''·(q̇₁² − q̇₀²)·dt` a tick — a fifth of the term where a spring throws a
    // wheel out at a hundred rad/s².
    //
    // And the curvature it reads is the **secant** of the Jacobian over the tick's own travel,
    // `(C'(q₀ + q̇₁·dt) − C'(q₀)) / (q̇₁·dt)`, one more solve per corner per sweep: with it the wheel's
    // change of momentum along the curving Jacobian is what the chassis is handed exactly, where the
    // curvature at `q₀` alone is short by `½·m_u·C'''·q̇³·dt²` a tick — a hundredth of a newton-second
    // a tick on a strut thrown out at three radians a second, which was most of what the true ledger
    // then failed to close by. The tangent, `cornerJacobianCurvature`, is what it converges to, and
    // what a wheel that does not move reads.
    //
    // The same solve gives where the wheel centre lands, and with it the couple the impulse's moment
    // needs: the impulse is exact but acts at the wheel's start point, and the wheel does not land
    // where its end velocity says. Its end velocity reads the end Jacobian, a whole `C''·q̇·dt` past
    // the start one, while its path over the tick curves by half that, so the wheel's displacement is
    // `ẋ₁·dt − ε` with `ε = R₀·[C(q₁) − C(q₀) − C'(q₀)·q̇₁·dt]`, and its angular momentum moves by
    // `−m_u·ε × ẋ₁` beyond the moment of its momentum change. The body receives the impulse
    // `+m_u·ε × ẋ₁` — second order per tick and all one sign on a wheel swinging on its arc — handed
    // over as the couple `m_u·ε × ẋ₁ / dt`. Measured on the arc: the isolated pair's angular momentum
    // closes to 1e-6 with it and drifts 5e-2 without.
    const auto kinematicResponse = [&](const ChassisMotion& estimate)
    {
        auto linear = glm::dvec3(0.0);
        auto torque = glm::dvec3(0.0);
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            auto& wheel = wheelKinematics[index];
            wheel.rate = state.corners[index].wishboneRate + rateChange[index];

            const auto travel = wheel.rate * deltaTime;
            wheel.curvature = state.chassis.orientation * scratch[index].curvature;
            auto pathCouple = glm::dvec3(0.0);
            if (std::abs(travel) > 1e-9)
            {
                const auto ahead = cornerPointAt(setup.corners[index], state.corners[index].wishboneAngle + travel,
                                                 index < 2 ? rackTravel : 0.0, setup.geometricLoadPath);
                if (ahead)
                {
                    wheel.curvature = (state.chassis.orientation * ahead->jacobian - jacobianWorld[index]) / travel;

                    const auto& here = scratch[index].centre;
                    const auto departure = state.chassis.orientation * (ahead->centre - here) - jacobianWorld[index] * travel;
                    const auto endVelocity = state.chassis.linearVelocity + estimate.acceleration * deltaTime +
                                             glm::cross(motion.angularVelocity + estimate.angularAcceleration * deltaTime, armWorld[index]) +
                                             jacobianWorld[index] * wheel.rate;
                    pathCouple = wheel.mass * glm::cross(departure, endVelocity) / deltaTime;
                }
            }

            // The spin the terms read is the tick's end spin, `ω₀ + α·dt`: the integrator turns the body
            // by the spin its new momentum gives it, so the Jacobian and the offset turn by that, and
            // a Coriolis impulse read at `ω₀` is short by `m_u·α × R·C'·q̇·dt²` a tick.
            auto endSpin = estimate;
            endSpin.angularVelocity = motion.angularVelocity + estimate.angularAcceleration * deltaTime;
            kinematic[index] = unsprungKinematicReaction(wheel, endSpin, gravityVector);
            kinematic[index].couple += pathCouple;
            linear += kinematic[index].force;
            torque += glm::cross(armWorld[index], kinematic[index].force) + kinematic[index].couple;
        }
        kinematicLinear = linear / state.chassis.mass;
        kinematicAngular = inverseInertia * torque;
    };

    // The chassis's motion as the corners' reactions and the kinematic reactions leave it, from the
    // latest changes of rate and the latest kinematic response.
    const auto chassisEstimate = [&]
    {
        auto estimate = motion;
        for (auto other = std::size_t{0}; other < cornerCount; other++)
        {
            const auto force = -setup.corners[other].unsprungMass * rateChange[other] / deltaTime;
            estimate.acceleration += force * linearResponse[other];
            estimate.angularAcceleration += force * angularResponse[other];
        }
        estimate.acceleration += kinematicLinear;
        estimate.angularAcceleration += kinematicAngular;
        return estimate;
    };

    if (setup.frameAcceleration)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto& suspension = result.corners[index].suspension;

            knownAcceleration[index] = attachmentAcceleration(motion, armWorld[index]);

            // The weight's projection, made exact on the same Jacobian. The force pass carries
            // `−m_u·g·travelPerAngle`, the body-frame vertical component alone, and the invariance
            // this term exists for — a car accelerating uniformly at g has no relative suspension
            // motion — needs the weight and the frame term on one Jacobian. Exactly zero on a level
            // body, and on the springs path, whose Jacobian is that component.
            const auto weightCorrection = earthGravity * (jacobianWorld[index].y - suspension.travelPerAngle);

            frameKnown[index] =
                -corner.unsprungMass * (glm::dot(jacobianWorld[index], knownAcceleration[index]) + weightCorrection);
        }

        // The chassis's response at each wheel centre to a unit generalised force at each, symmetric
        // by construction and computed once per pair so that it is symmetric to the bit.
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            for (auto other = index; other < cornerCount; other++)
            {
                mobility[index][other] =
                    attachmentMobility(state.chassis.mass, inverseInertia, armWorld[other], jacobianWorld[other],
                                       armWorld[index], jacobianWorld[index]);
                mobility[other][index] = mobility[index][other];
            }

            // A corner's inertia against a body that gives way under its own reaction: the reduced
            // one, `m_u·|C'|² − m_u²·G`. Positive on any body heavier than its wheel; the floor is
            // the inertia's own.
            const auto& corner = setup.corners[index];
            reducedInertia[index] = std::max(result.corners[index].generalisedInertia -
                                                 corner.unsprungMass * corner.unsprungMass * mobility[index][index],
                                             1e-6);
        }
    }

    // --- the corners' velocity steps --------------------------------------------------------------
    //
    // One corner's step, from an explicit generalised force and an inertia to the rate that acts,
    // pure in its inputs because the coupled solve runs it more than once a tick; what it decided is
    // folded into the corner's forces and its generalised force once, in the second loop, in the
    // order the one loop folded it.
    //
    // One linearised backward-Euler step on the corner's rate (`solveCornerRate`;
    // docs/damper-integrator-brief.md, 2026-09-08 later): the explicit generalised force — the
    // damper's own force at the old rate included, as pass three always assembled it — plus the
    // damper's affine intercept `C·q̇₀` in the numerator, and the damper's slope beside an engaged
    // stop's viscous coefficient in the divisor. For a damper law linear through the origin the
    // intercept cancels the explicit force exactly and the step is backward Euler's `1/(1 + β)`.
    // Before this the numerator carried the force whole, whose map is `(1 − β)/(1 + β)`: every
    // damper slope acted about twice, and a corner past `β = 1` flipped its velocity every tick.
    //
    // An engaged stop's viscous coefficient (2026-09-08, docs/stop-element-brief.md) is linear in
    // the rate at a fixed compression and has no intercept: it sits in the divisor alone and its
    // share is read off the solved velocity — a backward-Euler treatment that is exact for such a
    // term and cannot change the velocity's sign, which is what keeps a corner resting on its
    // stop from alternating at the tick rate (the characterisation's D1). **No viscous stop force
    // appears in the explicit numerator.** The damper's own share is read off the solved velocity
    // the same way, below, so that the chassis reaction and `forces.damper` carry the force that
    // acted rather than the force at the old rate.
    //
    // With no damper coefficient and no stop engaged every quantity here is the bits pass three
    // produced: the divisor gains a literal zero and the generalised force is not touched.
    struct RateSolution
    {
        double rate = 0.0;
        double releasedAxis = 0.0; // the released stop's explicit force on the shaft, signed as it is removed
        bool stopReleased = false;
        bool stopHeld = false;
        double viscous = 0.0;
        RangeLimit limit = RangeLimit::None;
    };

    const auto solveRate = [&](const std::size_t index, const double generalisedForce, const double inertia)
    {
        const auto& hardpoints = setup.corners[index].hardpoints;
        const auto& damper = scratch[index].damper;
        const auto& bumpTerms = scratch[index].bumpTerms;
        const auto& droopTerms = scratch[index].droopTerms;
        const auto previousRate = state.corners[index].wishboneRate;
        const auto stopGeneralised = stopDamping[index] * damper.lengthPerAngle * damper.lengthPerAngle;

        auto step = RateSolution{};
        auto generalised = generalisedForce;

        auto rate = solveCornerRate(CornerRateStep{.previousRate = previousRate,
                                                   .generalisedForce = generalised,
                                                   .generalisedInertia = inertia,
                                                   .damperCoefficient = cornerDamping[index],
                                                   .stopCoefficient = stopGeneralised,
                                                   .deltaTime = deltaTime});

        // The engaged stop, and whether it holds. The shaft's compression rate is
        // `solveDamperForce`'s own expression for it; the droop stop reads the extension rate,
        // which is its negative.
        const auto bump = bumpTerms.viscousCoefficient > 0.0;
        const auto& engaged = bump ? bumpTerms : droopTerms;

        if (stopDamping[index] > 0.0)
        {
            // The engaged stop's viscous share at the velocity just solved.
            const auto stopRate = (bump ? -1.0 : 1.0) * damper.lengthPerAngle * rate;
            step.viscous = engaged.viscousCoefficient * stopRate;

            if (engaged.explicitForce + step.viscous < 0.0)
            {
                // The stop lets go: the shaft is leaving faster than the elastic force can
                // follow, and a stop cannot pull. Its whole force is zero this tick — the
                // push-only clamp `force()` has always applied — and the step is re-solved
                // without it. The re-solve cannot disagree with this branch: the solved total is
                // the stop-free total over a positive divisor, so the two read the same sign and
                // this is the complementarity condition rather than a guess.
                step.releasedAxis = bump ? engaged.explicitForce : -engaged.explicitForce;
                step.stopReleased = true;
                generalised -= step.releasedAxis * damper.lengthPerAngle;
                rate = solveCornerRate(CornerRateStep{.previousRate = previousRate,
                                                      .generalisedForce = generalised,
                                                      .generalisedInertia = inertia,
                                                      .damperCoefficient = cornerDamping[index],
                                                      .stopCoefficient = 0.0,
                                                      .deltaTime = deltaTime});
            }
            else
            {
                step.stopHeld = true;
            }
        }

        // --- the linkage's range ---
        //
        // The authored range as a unilateral constraint (`constrainCornerRange`, 2026-09-08 later
        // still, docs/range-constraint-brief.md): the rate the step solved is kept, bit for bit,
        // whenever it lands the coordinate inside the range, and is replaced by the one rate that
        // lands it exactly on the limit when it does not. Decided here, before the implicit
        // shares are read, so that the damper and the stop report the force of the rate that
        // acts and not of one the range then threw away — which was the 586 N read on a
        // stationary shaft in the droop audit. A corner resting on its limit reads a rate of
        // exactly zero, and with the previous rate also zero the damper's linearised force is its
        // force at rest and the stop's viscous share is nothing, while their elastic laws and
        // the spring stand; the constraint below carries the rest.
        //
        // The held stop cannot let go here: on the side the constraint acts, it only ever
        // reduces the rate into the limit the stop sits before, so the viscous share it re-reads
        // is smaller than the one that passed the test above and of the same sign. The other
        // side's stop is out of contact (both gaps are positive), short of a corner crossing its
        // whole range in one tick, which is tens of metres per second on the shaft.
        const auto constrained =
            constrainCornerRange(RangeConstraintStep{.previousAngle = state.corners[index].wishboneAngle,
                                                     .rate = rate,
                                                     .droopAngle = hardpoints.droopAngle,
                                                     .bumpAngle = hardpoints.bumpAngle,
                                                     .deltaTime = deltaTime});

        if (constrained.limit != RangeLimit::None)
        {
            rate = constrained.rate;

            if (step.stopHeld)
            {
                step.viscous = engaged.viscousCoefficient * ((bump ? -1.0 : 1.0) * damper.lengthPerAngle * rate);
            }
        }

        step.limit = constrained.limit;
        step.rate = rate;
        return step;
    };

    auto solved = std::array<RateSolution, cornerCount>{};
    auto sweeps = std::uint32_t{0};
    auto kinematicSettled = false;
    auto ratesMoved = HUGE_VAL;

    if (setup.frameAcceleration)
    {
        // The four steps together. Each corner's equation carries the chassis's acceleration at its
        // attachment, and that acceleration carries every corner's reaction, so the four are one
        // linear system in the changes of rate — with the stops' complementarity and the range's
        // deciding per corner. Gauss–Seidel over the corners: a corner takes the others' latest
        // changes of rate through the mobility, its own through the reduced inertia (exactly, which
        // is what keeps the sweep from having to iterate its own coupling), and the sweep repeats
        // until no change of rate moves. The coupling between corners is a few per cent of a
        // corner's inertia, so it contracts fast; the count is reported on every corner.
        for (sweeps = 1; sweeps <= frameSweepLimit; sweeps++)
        {
            auto largest = 0.0;

            // The kinematic reactions at the chassis motion the last sweep left, and their share of
            // the chassis's acceleration, which every corner's attachment then carries. Their offset
            // terms read the chassis's own acceleration, which they change — a fixed point of its
            // own, `m_u·|Δr|²` against the body's inertia, a few parts in ten thousand on the Golf but
            // a tenth on a wheel a metre off its design point — so the sweep watches their change as
            // well as the corners' and does not stop on one alone. Nothing here with the switch off:
            // the share is the zero vector and each corner adds `−m_u·0`.
            //
            // Once their change and the corners' are under their tolerances they are held for the
            // rest of the sweep: re-read every sweep they would jitter at the corner solve's rounding,
            // a hundredth of their tolerance, and that jitter reaches the corners' rates two decades
            // above the corners' own tolerance, so the sweep would run to its cap with nothing left
            // to converge. Held, the corners converge on a fixed forcing and are handed exactly the
            // motion they assumed; a rate that moves again — a limit or a stop decided late — lifts
            // the hold.
            if (setup.kinematicReaction && (!kinematicSettled || ratesMoved > kinematicRateTolerance))
            {
                const auto linearBefore = kinematicLinear;
                const auto angularBefore = kinematicAngular;
                kinematicResponse(chassisEstimate());
                const auto moved = std::max(glm::length(kinematicLinear - linearBefore), glm::length(kinematicAngular - angularBefore)) * deltaTime;
                kinematicSettled = moved <= kinematicSweepTolerance && ratesMoved <= kinematicRateTolerance;
                if (!kinematicSettled)
                {
                    largest = std::max(largest, moved);
                }
            }

            auto ratesLargest = 0.0;
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& corner = setup.corners[index];
                auto coupled = 0.0;

                for (auto other = std::size_t{0}; other < cornerCount; other++)
                {
                    if (other != index)
                    {
                        coupled += setup.corners[other].unsprungMass * mobility[index][other] * rateChange[other];
                    }
                }

                auto generalised = result.corners[index].generalisedForce + frameKnown[index] +
                                   corner.unsprungMass * coupled / deltaTime;
                generalised += -corner.unsprungMass *
                               glm::dot(jacobianWorld[index], kinematicLinear + glm::cross(kinematicAngular, armWorld[index]));

                solved[index] = solveRate(index, generalised, reducedInertia[index]);

                const auto change = solved[index].rate - state.corners[index].wishboneRate;
                ratesLargest = std::max(ratesLargest, std::abs(change - rateChange[index]));
                rateChange[index] = change;
            }

            ratesMoved = ratesLargest;
            largest = std::max(largest, ratesLargest);
            if (largest <= frameSweepTolerance)
            {
                break;
            }
        }

        sweeps = std::min(sweeps, frameSweepLimit);
    }
    else
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            solved[index] =
                solveRate(index, result.corners[index].generalisedForce, result.corners[index].generalisedInertia);
            rateChange[index] = solved[index].rate - state.corners[index].wishboneRate;
        }
    }

    // The kinematic reactions the chassis is handed are the ones the corners read: held from the
    // sweep where they settled, or — where the sweep ran out — read once more at the motion the four
    // changes of rate leave the chassis with. With the frame term off there is no sweep and the
    // estimate is the known forces and the four reactions of the uncoupled solve, iterated until the
    // offset terms have seen their own effect.
    if (setup.kinematicReaction && !kinematicSettled)
    {
        for (auto pass = std::uint32_t{0}; pass < frameSweepLimit; pass++)
        {
            const auto linearBefore = kinematicLinear;
            const auto angularBefore = kinematicAngular;
            kinematicResponse(chassisEstimate());
            const auto moved = std::max(glm::length(kinematicLinear - linearBefore), glm::length(kinematicAngular - angularBefore)) * deltaTime;
            if (moved <= kinematicSweepTolerance || setup.frameAcceleration)
            {
                break;
            }
        }
    }

    // --- the second loop: settle each corner, and hand the chassis what it did ------------------
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup.corners[index];
        auto& solution = result.corners[index];
        const auto& suspension = solution.suspension;
        const auto& damper = scratch[index].damper;
        const auto& bumpTerms = scratch[index].bumpTerms;
        const auto& step = solved[index];
        const auto& normal = scratch[index].normal;
        const auto& wheelWorld = scratch[index].wheelWorld;
        const auto previousRate = state.corners[index].wishboneRate;
        const auto inertia = solution.generalisedInertia;
        const auto rate = step.rate;
        const auto bump = bumpTerms.viscousCoefficient > 0.0;

        if (step.stopReleased)
        {
            solution.generalisedForce -= step.releasedAxis * damper.lengthPerAngle;

            if (bump)
            {
                solution.forces.bumpStop = 0.0;
            }
            else
            {
                solution.forces.droopStop = 0.0;
            }
        }

        if (setup.frameAcceleration)
        {
            // The frame term at the rates that act — the four reactions of this tick included, its
            // own through the same mobility the sweep solved it against — folded into the
            // generalised force before the shares, so that what the chassis reaction below reads is
            // the corner's discrete acceleration on its own inertia, `I·(q̇₁ − q̇₀)/dt`, to rounding.
            // And the acceleration the corner answered to, for the record.
            auto coupled = 0.0;
            auto reactions = glm::dvec3(0.0);

            for (auto other = std::size_t{0}; other < cornerCount; other++)
            {
                const auto force = -setup.corners[other].unsprungMass * rateChange[other] / deltaTime;
                coupled += setup.corners[other].unsprungMass * mobility[index][other] * rateChange[other];
                reactions += force * (linearResponse[other] + glm::cross(angularResponse[other], armWorld[index]));
            }

            auto frame = frameKnown[index] + corner.unsprungMass * coupled / deltaTime;
            const auto kinematicShare = kinematicLinear + glm::cross(kinematicAngular, armWorld[index]);
            frame += -corner.unsprungMass * glm::dot(jacobianWorld[index], kinematicShare);
            solution.generalisedForce += frame;
            solution.attachmentAcceleration = knownAcceleration[index] + reactions + kinematicShare;
            solution.forces.frame = std::abs(suspension.travelPerAngle) > 1e-12 ? frame / suspension.travelPerAngle : 0.0;
            solution.frameSweeps = sweeps;
        }

        if (step.stopHeld)
        {
            // The force that acted — for the chassis reaction below, which reads the
            // generalised force, and for whoever reads the forces: the explicit part they
            // already carry plus the viscous share.
            const auto axisViscous = bump ? step.viscous : -step.viscous;
            solution.generalisedForce += axisViscous * damper.lengthPerAngle;

            if (bump)
            {
                solution.forces.bumpStop += axisViscous;
            }
            else
            {
                solution.forces.droopStop += axisViscous;
            }
        }

        // The damper's implicit share: its linearised force moved by the slope times the change
        // of rate the step solved, on the corner and on the shaft. What the chassis reaction
        // below reads is then the corner's own discrete acceleration, `I·(q̇₁ − q̇₀)/dt`, exactly,
        // and `forces.damper` is the force that acted. A coefficient means a non-zero Jacobian,
        // so the division is safe; with no coefficient nothing here runs.
        if (cornerDamping[index] > 0.0)
        {
            const auto share = -cornerDamping[index] * (rate - previousRate);
            solution.generalisedForce += share;
            solution.forces.damper += share / damper.lengthPerAngle;
        }

        if (step.limit != RangeLimit::None)
        {
            // The constraint's own generalised force: the residual between the corner's
            // discrete acceleration at the rate that acts and every other force at that rate.
            // Folded into the generalised force so that the chassis reaction below — which is
            // the corner's `−m_u · q̈ · dC/dq`, whatever produced the acceleration — carries the
            // constraint's equal and opposite reaction through the same Jacobian as every other
            // corner force: a hanging wheel pulls the body down by what it hangs with, a wheel
            // driven into its bump limit pushes it up. On a held tick the corner's acceleration
            // is exactly zero, so the reaction cancels every other force and the chassis reads
            // the tyre alone. Reported in the corner's coordinate and at the wheel; the
            // wheel-referred figure follows `obstacleTravel`'s guard on the Jacobian.
            const auto reaction = inertia * (rate - previousRate) / deltaTime - solution.generalisedForce;
            solution.generalisedForce += reaction;
            solution.rangeLimit = step.limit;
            solution.rangeImpulse = reaction * deltaTime;
            solution.forces.rangeLimit =
                std::abs(suspension.travelPerAngle) > 1e-12 ? reaction / suspension.travelPerAngle : 0.0;
        }

        cornerRate[index] = rate;

        // The kerb faces' forces, first, as the one loop applied them where it computed them.
        for (auto slot = std::uint32_t{0}; slot < scratch[index].obstacleForceCount; slot++)
        {
            const auto& applied = scratch[index].obstacleForces[slot];
            chassisForces.addForceAtPoint(applied.force, applied.point, worldCentreOfMass);
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
        chassisForces.addForceAtPoint(solution.forces.tireVertical * normal, scratch[index].contactPoint,
                                      worldCentreOfMass);

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
            solution.chassisReaction =
                state.chassis.orientation * (-corner.unsprungMass * acceleration * suspension.wheelCentrePerAngle);
        }
        else
        {
            const auto wheelAcceleration = suspension.travelPerAngle * acceleration;
            solution.chassisReaction = glm::dvec3(0.0, -corner.unsprungMass * wheelAcceleration, 0.0);
        }

        chassisForces.addForceAtPoint(solution.chassisReaction, wheelWorld, worldCentreOfMass);

        // And the rest of the unsprung mass's acceleration — the Coriolis, centripetal and
        // design-offset parts — with the couple that takes the design point's inertial force back
        // to its own point (2026-09-08 latest of all (d), docs/chassis-kinematic-reaction-brief.md).
        // A second application at the same point, so the reaction above keeps its bits.
        if (setup.kinematicReaction)
        {
            chassisForces.addForceAtPoint(kinematic[index].force, wheelWorld, worldCentreOfMass);
            chassisForces.torque += kinematic[index].couple;
            solution.kinematicForce = kinematic[index].force;
            solution.kinematicCouple = kinematic[index].couple;
            solution.chassisReaction += kinematic[index].force;
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

    // The air, in its place.
    for (const auto& applied : aeroForces)
    {
        chassisForces.addForceAtPoint(applied.force, applied.point, worldCentreOfMass);
    }

    // --- Pass three: integrate ------------------------------------------------------------------
    const auto previousVelocity = state.chassis.linearVelocity;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& corner = state.corners[index];
        const auto& solution = result.corners[index];

        // The velocity step was solved in pass two, where the stops' viscous share needed it; the
        // arithmetic is the one that used to live here, unchanged (2026-09-08).
        const auto rate = cornerRate[index];

        corner.wishboneRate = rate;

        // The coordinate: the step's own advance inside the range, and the limit itself — exactly,
        // rather than to within the rounding of `q₀ + ((limit − q₀)/dt)·dt` — when the constraint
        // held (`constrainCornerRange`). The rate above already is the one that lands there, so this
        // is a placement of the last ulp and not a correction: nothing is moved that the velocity
        // step did not account for, and no momentum goes with it.
        if (solution.rangeLimit == RangeLimit::Bump)
        {
            corner.wishboneAngle = setup.corners[index].hardpoints.bumpAngle;
        }
        else if (solution.rangeLimit == RangeLimit::Droop)
        {
            corner.wishboneAngle = setup.corners[index].hardpoints.droopAngle;
        }
        else
        {
            corner.wishboneAngle += rate * deltaTime;
        }

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

        // And the kerb-contact path's friction, as its own statement so the line above stays the
        // line it was. Positive spins the wheel forward, as the road's own torque does.
        if (setup.kerbContact && solution.obstacleCount > 0)
        {
            corner.wheelSpeed += (solution.obstacleSpinTorque / std::max(setupCorner.wheelInertia, 1e-6)) * deltaTime;
        }

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
        // **And pressure multiplies rolling resistance** (2026-08-28). A soft tyre bends its sidewalls
        // further every revolution and hysteresis follows the deformation, so the coefficient rises
        // as pressure falls. Exactly 1.0 at the ideal pressure the car's own figure is quoted at.
        const auto rollingScale =
            setup.tyrePressure ? tyrePressureRollingResistanceScale(setupCorner.tyre.pressure, corner.tyre) : 1.0;
        const auto rolling = setupCorner.rollingResistance * rollingScale * solution.forces.tireVertical *
                             solution.contact.effectiveRadius;
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

            // The kerb-contact path's spin torque is part of the same net, and closes the same
            // balance: a wheel climbing a kerb on its friction is being spun down by the kerb.
            if (setup.kerbContact && solution.obstacleCount > 0)
            {
                chassisForces.torque -= solution.obstacleSpinTorque * spinAxis;
            }
        }

        // The linkage's authored range was clamped here until 2026-09-08 later still — the state
        // snapped onto the limit and the rate floored, after the chassis had been given the reaction
        // of an acceleration that then never happened. It is a constraint in the velocity step now
        // (`constrainCornerRange`, pass two), and the placement above is all that is left of it.
    }

    // Contact, resolved as impulses on the velocity and *before* the integrator moves the body.
    // A contact is a velocity constraint: expressed instead as a force over the tick it is either
    // spongy or explosive depending on the timestep, and there is no setting that is neither.
    //
    // Only the bodywork comes through here. The tires reach the ground through their own model, and
    // a rigid constraint underneath them would be fighting a carefully shaped force with something
    // that knows nothing about slip.
    result.contacts = collideBody(world, state.chassis, setup.body);

    // ...and whatever of the traffic is standing in the same place, added to the same manifold. It
    // is a second call rather than a parameter to the first because the world query and this one are
    // different questions asked of different owners — one crosses the Jolt bridge and one reads a
    // span the caller filled a moment ago. Empty on a circuit, and empty is exactly the manifold
    // above.
    collideObstacles(state.chassis, setup.body, dynamicObstacles, result.contacts);

    // Whatever the bodywork is about to hit, decided **before** it is resolved. A prop is bolted
    // down, so the solver below would otherwise put the car's whole momentum into an anchor that is
    // about to give way — right for a bollard, and for a wheelie bin it is both a car stopped dead
    // and, once the anchor did break, a bin handed an impulse sized to stop a car. This asks first.
    // Nothing happens here on a track with no street furniture on it.
    releaseBrokenProps(state.chassis, result.contacts, deltaTime);

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

        // The state as this tick leaves it — the deflection the next solve will be handed, one
        // tick ahead of what this frame's geometry carried, which at 360 Hz is a distinction a
        // trace cannot resolve. It reads exactly 0.0 on a car stating no coefficient — which is
        // the trace's way of answering "was it on", the question the pressure channels existed to
        // answer and were added too late for.
        frame.wheels[index].complianceRecession = state.corners[index].complianceRecession;
        frame.wheels[index].gripMultiplier = solution.patch.gripMultiplier;
        frame.wheels[index].inContact = solution.patch.inContact;
        frame.wheels[index].contactingSamples = solution.patch.contactingSamples;
        frame.wheels[index].patchDepthSpread = solution.patch.depthSpread;
        frame.wheels[index].tyreSurfaceTemperature = state.corners[index].tyre.surfaceTemperature;
        frame.wheels[index].tyreCoreTemperature = state.corners[index].tyre.coreTemperature;
        frame.wheels[index].tyreCarcassTemperature = state.corners[index].tyre.carcassTemperature;
        frame.wheels[index].tyreGasTemperature = state.corners[index].tyre.gasTemperature;

        // Published in psi because it is the one pressure in this model a driver sets by hand, and a
        // trace column exists to be read. It is computed from the state whether or not the model is
        // switched on, which is the same contract the temperature channels keep: the column says what
        // the rest of the model is assuming.
        frame.wheels[index].tyrePressurePsi =
            tyrePressureAt(setup.corners[index].tyre.pressure, state.corners[index].tyre) * psiPerPascal;
        frame.wheels[index].discTemperature = state.corners[index].discTemperature;
        frame.wheels[index].wheelTemperature = state.corners[index].wheelTemperature;

        // The kerb-contact path's three: exactly zero with the switch off and on every flat-road
        // tick with it on, which is the trace's way of answering "did it engage".
        frame.wheels[index].obstacleContacts = solution.obstacleCount;
        frame.wheels[index].obstacleNormalForce = solution.obstacleNormalForce;
        frame.wheels[index].obstacleAxisElevation = solution.obstacleAxisElevation;
    }

    return result;
}

} // namespace raceengine
