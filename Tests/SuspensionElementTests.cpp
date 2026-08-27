#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::coaxialSpring;
using raceengine::CornerSide;
using raceengine::Curve;
using raceengine::damperDampingCoefficient;
using raceengine::damperShaftCompression;
using raceengine::damperElementOf;
using raceengine::DamperForceSolution;
using raceengine::DamperKinematics;
using raceengine::golfGtiMk7;
using raceengine::kneedDamper;
using raceengine::golfMk7FrontCorner;
using raceengine::linearDamper;
using raceengine::golfMk7RearCorner;
using raceengine::outboardSign;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperForce;
using raceengine::solveDamperGeometry;
using raceengine::solveDamperKinematics;
using raceengine::solveElement;
using raceengine::solveSpringForce;
using raceengine::solveSpringKinematics;
using raceengine::springFreeLengthForLoad;
using raceengine::SpringMount;
using raceengine::springElementOf;
using raceengine::SpringKinematics;
using raceengine::SuspensionElement;
using raceengine::SuspensionState;
using raceengine::validateCornerSetup;
using raceengine::Wishbone;

// The element abstraction is the geometry API for separating the rear spring from the rear damper
// (docs/suspension-geometry-audit.md, steps 2-6). The production SPRING path consumes it as of
// step 6 — free length, spring force and the spring's generalised contribution all read
// `springElementOf` — while the damper path deliberately still runs on the solver's own fields.
// These cases gate both halves: the abstraction reproduces the production damper geometry exactly,
// it carries two elements at different stations with genuinely different ratios, the two roles are
// not interchangeable types, and the migrated spring path is bit-identical on coaxial cars.

// The roles must not convert into one another — a wheel rate belongs to the spring's ratio and a
// shaft speed to the damper's, and after the rear split those are different numbers. Asserted at
// compile time, which is the strongest gate a test can hold.
static_assert(!std::is_convertible_v<SpringKinematics, DamperKinematics>);
static_assert(!std::is_convertible_v<DamperKinematics, SpringKinematics>);
static_assert(!std::is_assignable_v<SpringKinematics&, DamperKinematics>);
static_assert(!std::is_assignable_v<DamperKinematics&, SpringKinematics>);

namespace
{

// The same placeholder double-wishbone corner the suspension tests use: plausible mid-size-car
// numbers that exist to exercise the solver, not to describe a vehicle.
CornerHardpoints splitCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * x, y, z);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;

    corner.lower = Wishbone{
        .frontPivot = at(0.30, 0.13, 0.15), .rearPivot = at(0.30, 0.13, -0.15), .ballJoint = at(0.62, 0.12, 0.0)};
    corner.upper = Wishbone{
        .frontPivot = at(0.35, 0.36, 0.12), .rearPivot = at(0.35, 0.36, -0.12), .ballJoint = at(0.58, 0.42, 0.0)};

    corner.damperChassis = at(0.32, 0.55, 0.0);
    corner.damperWishbone = at(0.50, 0.14, 0.0);
    corner.steeringRackOuter = at(0.30, 0.16, 0.16);
    corner.steeringArm = at(0.60, 0.16, 0.14);
    corner.wheelCentre = at(0.72, 0.30, 0.0);
    corner.wheelRadius = 0.31;

    corner.droopAngle = -0.16;
    corner.bumpAngle = 0.16;

    return corner;
}

// A spring seated at a different station on the same arm — the sourced Mk7 rear topology, in
// placeholder numbers. Inboard of the damper pickup, so its motion ratio must come out smaller.
SuspensionElement syntheticSpring(const CornerSide side)
{
    const auto mirror = outboardSign(side);

    return SuspensionElement{.chassis = glm::dvec3(mirror * 0.36, 0.55, 0.0),
                             .wishbone = glm::dvec3(mirror * 0.40, 0.135, 0.0)};
}

// The production slope arithmetic, replicated so the coefficient tests compare against the exact
// factors the force pass multiplies — same step, same expression order as `curveSlopeAt`.
double slopeAt(const Curve& curve, const double x)
{
    constexpr auto step = 1e-4;

    return (curve.at(x + step) - curve.at(x - step)) / (2.0 * step);
}

// An independent re-derivation of the element arithmetic — the swing axis with the solver's sign
// rule, the rotation about it, the lengths, and the raw-difference ratio — sharing no code with
// the production evaluator. The solver stopped carrying its own damper channel when the legacy
// path was retired (step 14), so this replica is the second implementation the element API is
// held against, and it stands in wherever these cases used to read the state's copy: the values
// are the same bits, because the arithmetic is the same expressions in the same order.
[[nodiscard]] glm::dvec3 replicaAxis(const CornerHardpoints& hardpoints)
{
    auto axis = glm::normalize(hardpoints.lower.rearPivot - hardpoints.lower.frontPivot);
    const auto toBallJoint = hardpoints.lower.ballJoint - hardpoints.lower.frontPivot;
    const auto radius = toBallJoint - glm::dot(toBallJoint, axis) * axis;
    if (glm::cross(axis, radius).y < 0.0)
    {
        axis = -axis;
    }

    return axis;
}

[[nodiscard]] double replicaLength(const CornerHardpoints& hardpoints, const SuspensionElement& element,
                                   const double angle)
{
    const auto outboard = hardpoints.lower.frontPivot + glm::angleAxis(angle, replicaAxis(hardpoints)) *
                                                            (element.wishbone - hardpoints.lower.frontPivot);

    return glm::distance(element.chassis, outboard);
}

[[nodiscard]] double replicaLengthPerAngle(const CornerHardpoints& hardpoints, const SuspensionElement& element,
                                           const double angle)
{
    constexpr auto step = 1e-6;

    return (replicaLength(hardpoints, element, angle + step) - replicaLength(hardpoints, element, angle - step)) /
           (2.0 * step);
}

[[nodiscard]] double replicaRatio(const CornerHardpoints& hardpoints, const SuspensionElement& element,
                                  const double angle)
{
    constexpr auto step = 1e-6;
    const auto centre = raceengine::solveCorner(hardpoints, angle, 0.0);
    REQUIRE(centre.has_value());
    const auto behind = raceengine::solveCorner(hardpoints, angle - step, 0.0, &centre.value());
    const auto ahead = raceengine::solveCorner(hardpoints, angle + step, 0.0, &centre.value());
    REQUIRE(behind.has_value());
    REQUIRE(ahead.has_value());

    return (replicaLength(hardpoints, element, angle + step) - replicaLength(hardpoints, element, angle - step)) /
           (ahead->wheelCentre.y - behind->wheelCentre.y);
}

std::vector<SuspensionState> sweepStates(const CornerHardpoints& corner, const std::size_t samples)
{
    auto states = std::vector<SuspensionState>{};
    for (auto index = std::size_t{0}; index < samples; index++)
    {
        const auto through = static_cast<double>(index) / static_cast<double>(samples - 1);
        const auto angle = corner.droopAngle + through * (corner.bumpAngle - corner.droopAngle);

        const auto solved = solveCornerWithJacobian(corner, angle, 0.0, states.empty() ? nullptr : &states.back());
        REQUIRE(solved.has_value());
        states.push_back(solved.value());
    }

    return states;
}

} // namespace

TEST_CASE("the element evaluator reproduces the production damper geometry", "[physics][suspension][element]")
{
    // The proof the abstraction is the same mathematics: across every Golf corner's whole stated
    // travel the element's length, Jacobian and motion ratio must match this file's independent
    // replica of the legacy arithmetic — the second implementation, now that the solver no longer
    // carries a damper channel of its own.
    const auto corners = {golfMk7FrontCorner(CornerSide::Left), golfMk7FrontCorner(CornerSide::Right),
                          golfMk7RearCorner(CornerSide::Left), golfMk7RearCorner(CornerSide::Right)};

    for (const auto& corner : corners)
    {
        const auto element = damperElementOf(corner);

        for (const auto& state : sweepStates(corner, 21))
        {
            // Exact on every channel: the element evaluator runs the same arithmetic as the
            // solver's damper, and both migrations' byte-identity rests on it.
            const auto evaluated = solveElement(corner, element, state.wishboneAngle);
            REQUIRE(evaluated.length == replicaLength(corner, element, state.wishboneAngle));
            REQUIRE(evaluated.lengthPerAngle == replicaLengthPerAngle(corner, element, state.wishboneAngle));

            // Exactly, not approximately: the kinematics ratio differences the same solves at the
            // same step as the solver's own, and the spring migration's byte-identity rests on it.
            const auto kinematics = solveDamperKinematics(corner, element, state.wishboneAngle);
            REQUIRE(kinematics.has_value());
            REQUIRE(kinematics->motionRatio == replicaRatio(corner, element, state.wishboneAngle));
        }
    }
}

TEST_CASE("a spring and a damper on one member carry different motion ratios", "[physics][suspension][element]")
{
    // The property the coil-over model cannot express and the reason the abstraction exists. The
    // synthetic spring picks up inboard of the damper on the same arm, so it must move less per
    // metre of wheel travel — a genuinely different ratio, valid across the whole range.
    const auto corner = splitCorner(CornerSide::Right);
    const auto spring = syntheticSpring(CornerSide::Right);
    const auto damper = damperElementOf(corner);

    for (const auto& state : sweepStates(corner, 21))
    {
        const auto springSide = solveSpringKinematics(corner, spring, state.wishboneAngle);
        const auto damperSide = solveDamperKinematics(corner, damper, state.wishboneAngle);

        REQUIRE(springSide.has_value());
        REQUIRE(damperSide.has_value());

        // Both healthy: signed negative (the element shortens as the wheel rises), magnitude in a
        // road car's range.
        REQUIRE(springSide->motionRatio < 0.0);
        REQUIRE(damperSide->motionRatio < 0.0);
        REQUIRE(std::abs(springSide->motionRatio) > 0.05);
        REQUIRE(std::abs(damperSide->motionRatio) < 1.2);

        // And measurably different, which is the point.
        REQUIRE(std::abs(damperSide->motionRatio) - std::abs(springSide->motionRatio) > 0.05);
    }
}

TEST_CASE("each element's Jacobian agrees with its own finite difference", "[physics][suspension][element]")
{
    // The evaluator differences at 1e-6; this check differences across the sweep's own much larger
    // steps and compares the secant against the mean of the endpoint tangents — the trapezoid
    // rule, second-order, the same discipline the corner's motion-ratio test uses.
    const auto corner = splitCorner(CornerSide::Right);
    const auto states = sweepStates(corner, 21);

    for (const auto& element : {syntheticSpring(CornerSide::Right), damperElementOf(corner)})
    {
        for (auto index = std::size_t{1}; index < states.size(); index++)
        {
            const auto before = solveElement(corner, element, states[index - 1].wishboneAngle);
            const auto after = solveElement(corner, element, states[index].wishboneAngle);

            const auto angleStep = states[index].wishboneAngle - states[index - 1].wishboneAngle;
            const auto secant = (after.length - before.length) / angleStep;
            const auto mean = 0.5 * (before.lengthPerAngle + after.lengthPerAngle);

            REQUIRE(secant == Catch::Approx(mean).epsilon(1e-4));

            // The motion ratio obeys the same identity against wheel travel.
            const auto travelStep = states[index].wheelTravel - states[index - 1].wheelTravel;
            const auto ratioSecant = (after.length - before.length) / travelStep;
            const auto ratioBefore = before.lengthPerAngle / states[index - 1].travelPerAngle;
            const auto ratioAfter = after.lengthPerAngle / states[index].travelPerAngle;

            REQUIRE(ratioSecant == Catch::Approx(0.5 * (ratioBefore + ratioAfter)).epsilon(1e-4));
        }
    }
}

TEST_CASE("coaxial cars answer the same element for both roles", "[physics][suspension][element]")
{
    // Today's coil-over assumption, stated by `springElementOf` and proved inert: on every Golf
    // corner the spring's element is the damper's element, so both roles report identical
    // kinematics — a statement about the authored data, which a sourced rear seat will break.
    const auto corners = {golfMk7FrontCorner(CornerSide::Right), golfMk7RearCorner(CornerSide::Right)};

    for (const auto& corner : corners)
    {
        const auto spring = springElementOf(corner);
        const auto damper = damperElementOf(corner);
        REQUIRE(spring.chassis == damper.chassis);
        REQUIRE(spring.wishbone == damper.wishbone);

        for (const auto& state : sweepStates(corner, 11))
        {
            const auto springSide = solveSpringKinematics(corner, spring, state.wishboneAngle);
            const auto damperSide = solveDamperKinematics(corner, damper, state.wishboneAngle);

            REQUIRE(springSide.has_value());
            REQUIRE(damperSide.has_value());
            REQUIRE(springSide->length == damperSide->length);
            REQUIRE(springSide->lengthPerAngle == damperSide->lengthPerAngle);
            REQUIRE(springSide->motionRatio == damperSide->motionRatio);
        }
    }
}

TEST_CASE("spring kinematics answer spring questions", "[physics][suspension][element]")
{
    // The role-specific use each type exists for, exercised end to end: the wheel rate this car's
    // shaft rate implies through the *spring's* ratio must reproduce the at-the-wheel figures the
    // source states — 35000 front, 57000 rear.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto statedWheelRate = std::array{35000.0, 35000.0, 57000.0, 57000.0};

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());

        const auto spring = solveSpringKinematics(corner.hardpoints, springElementOf(corner.hardpoints), 0.0);
        REQUIRE(spring.has_value());

        const auto wheelRate = corner.springRate * spring->motionRatio * spring->motionRatio;
        REQUIRE(wheelRate == Catch::Approx(statedWheelRate[index]).epsilon(1e-6));
    }
}

TEST_CASE("damper kinematics answer damper questions", "[physics][suspension][element]")
{
    // The damper-side counterpart: the shaft speed per metre-per-second of wheel travel is the
    // magnitude of the *damper's* ratio, and it must equal what the production damper-velocity
    // path computes from the state's own Jacobians.
    const auto corner = golfMk7RearCorner(CornerSide::Right);
    const auto design = solveCornerWithJacobian(corner, 0.0, 0.0);
    REQUIRE(design.has_value());

    const auto damper = solveDamperKinematics(corner, damperElementOf(corner), 0.0);
    REQUIRE(damper.has_value());

    // Production: damperVelocity = -damperLengthPerAngle * wishboneRate. Per unit wheel speed the
    // wishbone rate is 1/travelPerAngle, so the shaft speed is |motionRatio|.
    const auto wishboneRateForUnitWheelSpeed = 1.0 / design->travelPerAngle;
    const auto productionShaftSpeed =
        std::abs(-replicaLengthPerAngle(corner, damperElementOf(corner), 0.0) * wishboneRateForUnitWheelSpeed);

    REQUIRE(std::abs(damper->motionRatio) == Catch::Approx(productionShaftSpeed).margin(1e-9));
}

TEST_CASE("the left corner's elements mirror the right's", "[physics][suspension][element]")
{
    // Element lengths, Jacobians and ratios are mirror-invariant scalars, exactly as the corner's
    // damper length already is in the solver's own mirror test.
    SECTION("the Golf's authored elements")
    {
        const auto pairs = {std::pair{golfMk7FrontCorner(CornerSide::Left), golfMk7FrontCorner(CornerSide::Right)},
                            std::pair{golfMk7RearCorner(CornerSide::Left), golfMk7RearCorner(CornerSide::Right)}};

        for (const auto& [left, right] : pairs)
        {
            for (const auto angle : {-0.1, 0.0, 0.08})
            {
                const auto onLeft = solveElement(left, damperElementOf(left), angle);
                const auto onRight = solveElement(right, damperElementOf(right), angle);

                REQUIRE(onLeft.length == Catch::Approx(onRight.length).margin(1e-12));
                REQUIRE(onLeft.lengthPerAngle == Catch::Approx(onRight.lengthPerAngle).margin(1e-9));
            }
        }
    }

    SECTION("the synthetic split spring")
    {
        const auto left = splitCorner(CornerSide::Left);
        const auto right = splitCorner(CornerSide::Right);

        for (const auto angle : {-0.1, 0.0, 0.08})
        {
            const auto onLeft = solveElement(left, syntheticSpring(CornerSide::Left), angle);
            const auto onRight = solveElement(right, syntheticSpring(CornerSide::Right), angle);

            REQUIRE(onLeft.length == Catch::Approx(onRight.length).margin(1e-12));
            REQUIRE(onLeft.lengthPerAngle == Catch::Approx(onRight.lengthPerAngle).margin(1e-9));
        }
    }
}

TEST_CASE("the migrated spring path is the old spring path, exactly", "[physics][suspension][element]")
{
    // The coaxial regression contract, held to bit-identity rather than tolerance: every value the
    // spring migration touched — the shaft rate, the free length, the per-tick spring force and
    // the spring's Jacobian — must be the same bits the pre-migration formulas produce. This is
    // what lets both parity gates stand as the migration's regression oracle.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto statedWheelRate = std::array{35000.0, 35000.0, 57000.0, 57000.0};

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        REQUIRE(corner.hardpoints.springMount == SpringMount::CoaxialWithDamper);

        const auto element = damperElementOf(corner.hardpoints);

        // The shaft rate, as the pre-migration conversion wrote it (the replica is that
        // arithmetic, bit for bit).
        const auto oldRatio = std::abs(replicaRatio(corner.hardpoints, element, 0.0));
        REQUIRE(corner.springRate == statedWheelRate[index] / (oldRatio * oldRatio));

        // The free length, as the pre-migration formula wrote it, at an arbitrary load.
        const auto load = 3200.0;
        const auto oldFreeLength = replicaLength(corner.hardpoints, element, 0.0) + load / (corner.springRate * oldRatio);
        const auto newFreeLength = springFreeLengthForLoad(corner, load);
        REQUIRE(newFreeLength.has_value());
        REQUIRE(newFreeLength.value() == oldFreeLength);

        // The per-tick force and Jacobian, across the whole stated travel.
        for (const auto& state : sweepStates(corner.hardpoints, 11))
        {
            const auto spring = solveSpringForce(corner, state);
            REQUIRE(spring.length == replicaLength(corner.hardpoints, element, state.wishboneAngle));
            REQUIRE(spring.lengthPerAngle == replicaLengthPerAngle(corner.hardpoints, element, state.wishboneAngle));
            REQUIRE(spring.force ==
                    corner.springRate *
                        (corner.springFreeLength - replicaLength(corner.hardpoints, element, state.wishboneAngle)));
        }
    }
}

TEST_CASE("a separately mounted spring drives the spring path and only the spring path",
          "[physics][suspension][element]")
{
    // The proof the production spring path consumes SpringKinematics rather than wrapping the old
    // damper values: mount the synthetic spring on the placeholder corner, and the spring's force,
    // free length and Jacobian all move while the solved state — the damper's world — does not.
    auto split = CornerSetup{};
    split.hardpoints = splitCorner(CornerSide::Right);
    split.hardpoints.springMount = SpringMount::OnLowerWishbone;
    split.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    split.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    split.springRate = 30000.0;

    auto coaxial = CornerSetup{};
    coaxial.hardpoints = splitCorner(CornerSide::Right);
    coaxial.springRate = 30000.0;

    const auto load = 3000.0;
    const auto splitFree = springFreeLengthForLoad(split, load);
    const auto coaxialFree = springFreeLengthForLoad(coaxial, load);
    REQUIRE(splitFree.has_value());
    REQUIRE(coaxialFree.has_value());
    split.springFreeLength = splitFree.value();
    coaxial.springFreeLength = coaxialFree.value();

    // The free length is the spring element's: it reproduces from spring kinematics exactly, and
    // the damper-based formula lands somewhere else entirely.
    const auto springDesign = solveSpringKinematics(split.hardpoints, springElementOf(split.hardpoints), 0.0);
    REQUIRE(springDesign.has_value());
    REQUIRE(splitFree.value() ==
            springDesign->length + load / (split.springRate * std::abs(springDesign->motionRatio)));

    const auto splitDamperElement = damperElementOf(split.hardpoints);
    const auto damperBasedFree =
        replicaLength(split.hardpoints, splitDamperElement, 0.0) +
        load / (split.springRate * std::abs(replicaRatio(split.hardpoints, splitDamperElement, 0.0)));
    REQUIRE(std::abs(splitFree.value() - damperBasedFree) > 0.01);

    // Away from design the force reads the spring element's length, and the solved state under it
    // is untouched by the spring fields — the spring contribution changed, the damper's did not.
    const auto splitState = solveCornerWithJacobian(split.hardpoints, 0.06, 0.0);
    const auto coaxialState = solveCornerWithJacobian(coaxial.hardpoints, 0.06, 0.0);
    REQUIRE(splitState.has_value());
    REQUIRE(coaxialState.has_value());
    REQUIRE(replicaLength(split.hardpoints, splitDamperElement, 0.06) ==
            replicaLength(coaxial.hardpoints, damperElementOf(coaxial.hardpoints), 0.06));
    REQUIRE(replicaLengthPerAngle(split.hardpoints, splitDamperElement, 0.06) ==
            replicaLengthPerAngle(coaxial.hardpoints, damperElementOf(coaxial.hardpoints), 0.06));

    const auto splitSpring = solveSpringForce(split, *splitState);
    const auto coaxialSpring = solveSpringForce(coaxial, *coaxialState);

    const auto element = solveElement(split.hardpoints, springElementOf(split.hardpoints), 0.06);
    REQUIRE(splitSpring.length == element.length);
    REQUIRE(splitSpring.lengthPerAngle == element.lengthPerAngle);
    REQUIRE(splitSpring.force == split.springRate * (split.springFreeLength - element.length));

    // The spring's Jacobian is its own, not the damper's.
    REQUIRE(std::abs(splitSpring.lengthPerAngle - replicaLengthPerAngle(split.hardpoints, splitDamperElement, 0.06)) >
            0.01);
    REQUIRE(std::abs(splitSpring.force - coaxialSpring.force) > 1.0);
}

TEST_CASE("the spring free length puts equilibrium at the design position", "[physics][suspension][element]")
{
    // With the free length solved for a load, the spring's generalised contribution at q = 0
    // cancels that load's: force times the spring's own Jacobian against load times the wheel's.
    // Held on both mounts, so the invariant is the formula's and not the coaxial coincidence's.
    auto corners = std::vector<CornerSetup>{};

    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    corners.push_back(golf->corners[0]);

    auto split = CornerSetup{};
    split.hardpoints = splitCorner(CornerSide::Right);
    split.hardpoints.springMount = SpringMount::OnLowerWishbone;
    split.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    split.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    split.springRate = 30000.0;
    corners.push_back(split);

    for (auto corner : corners)
    {
        const auto load = 3300.0;
        const auto freeLength = springFreeLengthForLoad(corner, load);
        REQUIRE(freeLength.has_value());
        corner.springFreeLength = freeLength.value();

        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());

        const auto spring = solveSpringForce(corner, *design);
        REQUIRE(spring.force * spring.lengthPerAngle == Catch::Approx(-load * design->travelPerAngle).epsilon(1e-9));
    }
}

TEST_CASE("the migrated damper velocity is the old damper velocity, exactly", "[physics][suspension][element]")
{
    // Step 7's coaxial regression contract: the damper velocity now reads the element evaluator,
    // and on every Golf corner across the whole travel it must be the same bits the state's own
    // Jacobian produced — at rest and at speed, in bump and in rebound.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto element = damperElementOf(corner.hardpoints);

        for (const auto& state : sweepStates(corner.hardpoints, 11))
        {
            const auto geometry = solveDamperGeometry(corner, state);
            const auto legacyJacobian = replicaLengthPerAngle(corner.hardpoints, element, state.wishboneAngle);
            REQUIRE(geometry.length == replicaLength(corner.hardpoints, element, state.wishboneAngle));
            REQUIRE(geometry.lengthPerAngle == legacyJacobian);

            for (const auto wishboneRate : {-1.7, -0.3, 0.0, 0.4, 2.1})
            {
                REQUIRE(-geometry.lengthPerAngle * wishboneRate == -legacyJacobian * wishboneRate);
            }
        }
    }
}

TEST_CASE("each role reads its own attachment and ignores the other's", "[physics][suspension][element]")
{
    // Independence, both ways, through the production seams: on a split corner, moving the spring
    // seat moves only the spring's outputs, and moving the damper pickup moves only the damper's.
    auto base = CornerSetup{};
    base.hardpoints = splitCorner(CornerSide::Right);
    base.hardpoints.springMount = SpringMount::OnLowerWishbone;
    base.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    base.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    base.springRate = 30000.0;
    base.springFreeLength = 0.5;

    auto movedSpring = base;
    movedSpring.hardpoints.springWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    auto movedDamper = base;
    movedDamper.hardpoints.damperWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    const auto state = solveCornerWithJacobian(base.hardpoints, 0.05, 0.0);
    REQUIRE(state.has_value());

    const auto baseSpring = solveSpringForce(base, *state);
    const auto baseDamper = solveDamperGeometry(base, *state);

    // Moving the spring seat: the spring's outputs move, the damper's are the same bits.
    const auto springMovedSpring = solveSpringForce(movedSpring, *state);
    const auto springMovedDamper = solveDamperGeometry(movedSpring, *state);
    REQUIRE(std::abs(springMovedSpring.lengthPerAngle - baseSpring.lengthPerAngle) > 1e-4);
    REQUIRE(springMovedDamper.length == baseDamper.length);
    REQUIRE(springMovedDamper.lengthPerAngle == baseDamper.lengthPerAngle);

    // Moving the damper pickup: the damper's outputs move, the spring's are the same bits.
    const auto damperMovedSpring = solveSpringForce(movedDamper, *state);
    const auto damperMovedDamper = solveDamperGeometry(movedDamper, *state);
    REQUIRE(std::abs(damperMovedDamper.lengthPerAngle - baseDamper.lengthPerAngle) > 1e-4);
    REQUIRE(damperMovedSpring.length == baseSpring.length);
    REQUIRE(damperMovedSpring.lengthPerAngle == baseSpring.lengthPerAngle);
    REQUIRE(damperMovedSpring.force == baseSpring.force);
}

TEST_CASE("the damper-rate conversion consumes the damper's ratio", "[physics][suspension][element]")
{
    // The knee of the converted curve sits at `|ratio| x knee` of shaft speed, so which role's
    // ratio fed the conversion is readable straight off the curve — and on the split corner the
    // two ratios differ enough that the wrong role lands the knee visibly elsewhere.
    auto split = splitCorner(CornerSide::Right);
    split.springMount = SpringMount::OnLowerWishbone;
    split.springChassis = syntheticSpring(CornerSide::Right).chassis;
    split.springWishbone = syntheticSpring(CornerSide::Right).wishbone;

    const auto springSide = solveSpringKinematics(split, springElementOf(split), 0.0);
    const auto damperSide = solveDamperKinematics(split, damperElementOf(split), 0.0);
    REQUIRE(springSide.has_value());
    REQUIRE(damperSide.has_value());
    REQUIRE(std::abs(springSide->motionRatio - damperSide->motionRatio) > 0.05);

    const auto curve = kneedDamper(4600.0, 1834.0, 0.070, 5300.0, 2589.0, 0.140, *damperSide);

    // The bump knee is the fourth point; its speed is the damper ratio's doing.
    REQUIRE(curve.points[3].x == std::abs(damperSide->motionRatio) * 0.070);
    REQUIRE(std::abs(curve.points[3].x - std::abs(springSide->motionRatio) * 0.070) > 1e-3);

    // And the production Golf's built curves reproduce through the same call, point for point.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto frontKinematics =
        solveDamperKinematics(setup->corners[0].hardpoints, damperElementOf(setup->corners[0].hardpoints), 0.0);
    const auto rearKinematics =
        solveDamperKinematics(setup->corners[2].hardpoints, damperElementOf(setup->corners[2].hardpoints), 0.0);
    REQUIRE(frontKinematics.has_value());
    REQUIRE(rearKinematics.has_value());

    const auto front = kneedDamper(4600.0, 1834.0, 0.070, 5300.0, 2589.0, 0.140, *frontKinematics);
    const auto rear = kneedDamper(6200.0, 1842.0, 0.100, 6700.0, 2700.0, 0.140, *rearKinematics);

    REQUIRE(front.points.size() == setup->corners[0].damper.points.size());
    REQUIRE(rear.points.size() == setup->corners[2].damper.points.size());
    for (auto point = std::size_t{0}; point < front.points.size(); point++)
    {
        REQUIRE(front.points[point] == setup->corners[0].damper.points[point]);
        REQUIRE(rear.points[point] == setup->corners[2].damper.points[point]);
    }
}

TEST_CASE("the migrated damper force is the old damper force, exactly", "[physics][suspension][element]")
{
    // Step 8's coaxial regression contract: the damper force now reads the element evaluator, and
    // on every Golf corner, across the travel and across rebound, rest and compression, it must be
    // the same bits the state-field arithmetic produced — force, velocity, and the generalised
    // contribution through the damper's own Jacobian.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];

        const auto element = damperElementOf(corner.hardpoints);
        for (const auto& state : sweepStates(corner.hardpoints, 11))
        {
            const auto legacyJacobian = replicaLengthPerAngle(corner.hardpoints, element, state.wishboneAngle);

            // The fused generalised-force branch's precondition, which is what keeps the whole
            // assembly the pre-split expression on this car.
            const auto spring = solveSpringForce(corner, state);
            REQUIRE(spring.lengthPerAngle == legacyJacobian);

            for (const auto wishboneRate : {-1.7, 0.0, 0.4, 2.1})
            {
                const auto oldVelocity = -legacyJacobian * wishboneRate;
                const auto oldForce = corner.damper.at(oldVelocity);

                const auto damper = solveDamperForce(corner, state, wishboneRate);
                REQUIRE(damper.lengthPerAngle == legacyJacobian);
                REQUIRE(damper.velocity == oldVelocity);
                REQUIRE(damper.force == oldForce);
                REQUIRE(damper.force * damper.lengthPerAngle == oldForce * legacyJacobian);
            }

            // The sign convention, pinned: positive wishbone rate is bump, so the shaft closes
            // (positive velocity) and the damper resists (positive force); rebound mirrors it.
            const auto compressing = solveDamperForce(corner, state, 2.1);
            const auto rebounding = solveDamperForce(corner, state, -1.7);
            REQUIRE(compressing.velocity > 0.0);
            REQUIRE(compressing.force > 0.0);
            REQUIRE(rebounding.velocity < 0.0);
            REQUIRE(rebounding.force < 0.0);
        }
    }
}

TEST_CASE("the damper force follows the damper element through the production path", "[physics][suspension][element]")
{
    // Split-role isolation for the force itself: on the split corner, the spring pickup cannot
    // reach the damper force, the damper pickup moves it, and the generalised contribution rides
    // the damper's Jacobian — visibly not the spring's, whose ratio differs by construction.
    auto base = CornerSetup{};
    base.hardpoints = splitCorner(CornerSide::Right);
    base.hardpoints.springMount = SpringMount::OnLowerWishbone;
    base.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    base.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    base.springRate = 30000.0;
    base.springFreeLength = 0.5;

    auto movedSpring = base;
    movedSpring.hardpoints.springWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    auto movedDamper = base;
    movedDamper.hardpoints.damperWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    const auto state = solveCornerWithJacobian(base.hardpoints, 0.05, 0.0);
    REQUIRE(state.has_value());

    const auto wishboneRate = 0.8;
    const auto baseDamper = solveDamperForce(base, *state, wishboneRate);

    // Moving the spring seat: the damper force path is the same bits.
    const auto springMoved = solveDamperForce(movedSpring, *state, wishboneRate);
    REQUIRE(springMoved.velocity == baseDamper.velocity);
    REQUIRE(springMoved.force == baseDamper.force);
    REQUIRE(springMoved.lengthPerAngle == baseDamper.lengthPerAngle);

    // Moving the damper pickup: the damper force path moves.
    const auto damperMoved = solveDamperForce(movedDamper, *state, wishboneRate);
    REQUIRE(std::abs(damperMoved.velocity - baseDamper.velocity) > 1e-4);
    REQUIRE(std::abs(damperMoved.force - baseDamper.force) > 0.1);

    // The generalised contribution rides the damper's Jacobian, and consuming the spring's
    // instead is a visibly different number — which is what makes miswiring a failing test
    // rather than a quiet bias.
    const auto spring = solveSpringForce(base, *state);
    REQUIRE(std::abs(spring.lengthPerAngle - baseDamper.lengthPerAngle) > 0.01);
    REQUIRE(std::abs(baseDamper.force * baseDamper.lengthPerAngle - baseDamper.force * spring.lengthPerAngle) > 1.0);
}

TEST_CASE("the migrated implicit damping coefficient is the old coefficient, exactly",
          "[physics][suspension][element]")
{
    // Step 9's coaxial regression contract: the coefficient reads the damper element's Jacobian
    // and velocity now, and on every Golf corner — rebound, design, moderate bump and the edge of
    // the bump stop's entry — it must be the same bits the state-field expression produced. The
    // implicit divisor the integration solves against is checked with it, which is the stability
    // statement: identical coefficient, identical integrated response.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto& hardpoints = corner.hardpoints;

        for (const auto through : {0.75 * hardpoints.droopAngle, 0.0, 0.4 * hardpoints.bumpAngle,
                                   0.9 * hardpoints.bumpAngle})
        {
            const auto state = solveCornerWithJacobian(hardpoints, through, 0.0);
            REQUIRE(state.has_value());
            const auto legacyJacobian = replicaLengthPerAngle(hardpoints, damperElementOf(hardpoints), through);

            for (const auto wishboneRate : {-1.7, 0.0, 0.4, 2.1})
            {
                const auto oldVelocity = -legacyJacobian * wishboneRate;
                const auto oldCoefficient = legacyJacobian * legacyJacobian *
                                            std::max(0.0, slopeAt(corner.damper, oldVelocity));

                const auto damper = solveDamperForce(corner, *state, wishboneRate);
                const auto coefficient = damperDampingCoefficient(corner, damper);
                REQUIRE(coefficient == oldCoefficient);

                // The implicit update's divisor, both ways — the integration sees the same number.
                const auto inertia = std::max(corner.unsprungMass * state->travelPerAngle * state->travelPerAngle,
                                              1e-6);
                const auto tick = 1.0 / 360.0;
                REQUIRE(1.0 + (coefficient / inertia) * tick == 1.0 + (oldCoefficient / inertia) * tick);
            }
        }
    }
}

TEST_CASE("the implicit damping coefficient follows the damper element only", "[physics][suspension][element]")
{
    // Split-role isolation for the coefficient, plus the law/geometry separation: the spring seat
    // cannot reach it, the damper pickup moves it, the spring Jacobian wired in by mistake is a
    // visibly different number, and swapping the damper curve moves it through the slope alone
    // while the geometry factors hold still.
    auto base = CornerSetup{};
    base.hardpoints = splitCorner(CornerSide::Right);
    base.hardpoints.springMount = SpringMount::OnLowerWishbone;
    base.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    base.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    base.springRate = 30000.0;
    base.springFreeLength = 0.5;

    auto movedSpring = base;
    movedSpring.hardpoints.springWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    auto movedDamper = base;
    movedDamper.hardpoints.damperWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    auto stifferCurve = base;
    stifferCurve.damper = linearDamper(8400.0, 7600.0);

    const auto state = solveCornerWithJacobian(base.hardpoints, 0.05, 0.0);
    REQUIRE(state.has_value());

    const auto wishboneRate = 0.8;
    const auto baseDamper = solveDamperForce(base, *state, wishboneRate);
    const auto baseCoefficient = damperDampingCoefficient(base, baseDamper);

    // The spring seat cannot reach the coefficient.
    const auto springMoved = damperDampingCoefficient(movedSpring, solveDamperForce(movedSpring, *state, wishboneRate));
    REQUIRE(springMoved == baseCoefficient);

    // The damper pickup moves it.
    const auto damperMoved = damperDampingCoefficient(movedDamper, solveDamperForce(movedDamper, *state, wishboneRate));
    REQUIRE(std::abs(damperMoved - baseCoefficient) > 1.0);

    // Cross-wiring the spring Jacobian is a visibly different number.
    const auto spring = solveSpringForce(base, *state);
    const auto springWired =
        spring.lengthPerAngle * spring.lengthPerAngle * std::max(0.0, slopeAt(base.damper, baseDamper.velocity));
    REQUIRE(std::abs(springWired - baseCoefficient) > 1.0);

    // A different curve moves the coefficient through the slope alone: the geometry factors are
    // the same bits, so what changed is the law's input and not the migrated geometry.
    const auto stifferDamper = solveDamperForce(stifferCurve, *state, wishboneRate);
    REQUIRE(stifferDamper.lengthPerAngle == baseDamper.lengthPerAngle);
    REQUIRE(stifferDamper.velocity == baseDamper.velocity);

    const auto stifferCoefficient = damperDampingCoefficient(stifferCurve, stifferDamper);
    REQUIRE(std::abs(stifferCoefficient - baseCoefficient) > 1.0);
    REQUIRE(stifferCoefficient == baseDamper.lengthPerAngle * baseDamper.lengthPerAngle *
                                      std::max(0.0, slopeAt(stifferCurve.damper, baseDamper.velocity)));
}

namespace
{

// The wishbone angle at which the damper element's compression reaches a target, by bisection —
// positive targets close the bump gap, negative targets open the droop gap.
[[nodiscard]] double angleForShaftCompression(const CornerHardpoints& hardpoints, const double target)
{
    const auto designLength = solveElement(hardpoints, damperElementOf(hardpoints), 0.0).length;
    auto low = 0.0;
    auto high = target > 0.0 ? hardpoints.bumpAngle : hardpoints.droopAngle;

    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto compression = designLength - solveElement(hardpoints, damperElementOf(hardpoints), middle).length;
        (std::abs(compression) < std::abs(target) ? low : high) = middle;
    }

    return 0.5 * (low + high);
}

} // namespace

TEST_CASE("the migrated stop geometry is the old stop geometry, exactly", "[physics][suspension][element]")
{
    // Step 12's coaxial regression contract: the stops' compression now reads the damper element
    // at both ends — design and current — and on every Golf corner, from full droop through both
    // stop entries to deep bump, it must be the same bits the per-tick design solve produced. The
    // past-gap quantities and the unchanged force law are held with it.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto& hardpoints = corner.hardpoints;

        // The legacy sources, computed as the force pass computed them — via the replica, which is
        // that arithmetic written independently.
        const auto element = damperElementOf(hardpoints);
        REQUIRE(solveElement(hardpoints, element, 0.0).length == replicaLength(hardpoints, element, 0.0));

        const auto droopEntry = angleForShaftCompression(hardpoints, -corner.droopStop.gap);
        const auto bumpEntry = angleForShaftCompression(hardpoints, corner.bumpStop.gap);

        for (const auto angle :
             {hardpoints.droopAngle, droopEntry, 0.5 * (hardpoints.droopAngle + droopEntry), 0.0,
              0.4 * hardpoints.bumpAngle, bumpEntry, 0.95 * hardpoints.bumpAngle})
        {
            const auto state = solveCornerWithJacobian(hardpoints, angle, 0.0);
            REQUIRE(state.has_value());

            const auto oldCompression =
                replicaLength(hardpoints, element, 0.0) - replicaLength(hardpoints, element, angle);

            const auto damper = solveDamperForce(corner, *state, 0.6);
            const auto compression = damperShaftCompression(corner, damper);
            REQUIRE(compression == oldCompression);

            // The past-gap quantities the force law consumes, both stops.
            REQUIRE(compression - corner.bumpStop.gap == oldCompression - corner.bumpStop.gap);
            REQUIRE(-compression - corner.droopStop.gap == -oldCompression - corner.droopStop.gap);

            // And the unchanged law over the unchanged inputs is the unchanged force.
            REQUIRE(corner.bumpStop.force(compression - corner.bumpStop.gap, damper.velocity) ==
                    corner.bumpStop.force(oldCompression - corner.bumpStop.gap, damper.velocity));
            REQUIRE(corner.droopStop.force(-compression - corner.droopStop.gap, -damper.velocity) ==
                    corner.droopStop.force(-oldCompression - corner.droopStop.gap, -damper.velocity));
        }
    }
}

TEST_CASE("the Golf's stop engagement travels have not moved", "[physics][suspension][element]")
{
    // The established characterisation of the authored setup (docs/suspension-geometry-audit.md,
    // step 1) — geometry facts about the current placeholders, not claims about the real car —
    // pinned so a geometry-source migration cannot move them silently.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto engagementTravel = [](const raceengine::CornerSetup& corner, const double target)
    {
        const auto angle = angleForShaftCompression(corner.hardpoints, target);
        const auto solved = raceengine::solveCorner(corner.hardpoints, angle, 0.0);
        REQUIRE(solved.has_value());
        return solved->wheelTravel * 1000.0;
    };

    const auto& front = setup->corners[0];
    const auto& rear = setup->corners[2];

    REQUIRE(engagementTravel(front, front.bumpStop.gap) == Catch::Approx(21.6).margin(0.1));
    REQUIRE(engagementTravel(front, -front.droopStop.gap) == Catch::Approx(-44.5).margin(0.1));
    REQUIRE(engagementTravel(rear, rear.bumpStop.gap) == Catch::Approx(20.3).margin(0.1));
    REQUIRE(engagementTravel(rear, -rear.droopStop.gap) == Catch::Approx(-40.4).margin(0.1));
}

TEST_CASE("stop kinematics follow the damper element, not the spring element", "[physics][suspension][element]")
{
    // The stop lives on the damper shaft, so its geometry must be the damper's: moving the spring
    // seat cannot reach it, moving the damper pickup moves it, and on a split corner the spring
    // axis's own displacement is a visibly different number.
    auto base = CornerSetup{};
    base.hardpoints = splitCorner(CornerSide::Right);
    base.hardpoints.springMount = SpringMount::OnLowerWishbone;
    base.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    base.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    base.springRate = 30000.0;
    base.springFreeLength = 0.5;

    auto movedSpring = base;
    movedSpring.hardpoints.springWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    auto movedDamper = base;
    movedDamper.hardpoints.damperWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    const auto state = solveCornerWithJacobian(base.hardpoints, 0.06, 0.0);
    REQUIRE(state.has_value());

    const auto baseCompression = damperShaftCompression(base, solveDamperForce(base, *state, 0.6));

    // The spring seat cannot reach the stop's geometry.
    const auto springMoved = damperShaftCompression(movedSpring, solveDamperForce(movedSpring, *state, 0.6));
    REQUIRE(springMoved == baseCompression);

    // The damper pickup moves it (its own state solved from its own hardpoints).
    const auto damperState = solveCornerWithJacobian(movedDamper.hardpoints, 0.06, 0.0);
    REQUIRE(damperState.has_value());
    const auto damperMoved = damperShaftCompression(movedDamper, solveDamperForce(movedDamper, *damperState, 0.6));
    REQUIRE(std::abs(damperMoved - baseCompression) > 1e-4);

    // And the spring axis's own displacement is not the stop's number on a split corner.
    const auto springDisplacement =
        solveElement(base.hardpoints, springElementOf(base.hardpoints), 0.0).length -
        solveSpringForce(base, *state).length;
    REQUIRE(std::abs(baseCompression - springDisplacement) > 0.001);
}

TEST_CASE("the stops' generalised contribution rides the damper element", "[physics][suspension][element]")
{
    // Step 13's contract, both halves. Coaxial: on every Golf corner, from full droop through
    // both stop entries to deep bump, the stops' contribution through the element Jacobian is
    // the legacy contribution through the state field bit for bit, the fused product is too, and
    // the fused branch's precondition holds — so the production assembly's total is the legacy
    // total by expression identity.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto& hardpoints = corner.hardpoints;

        const auto droopEntry = angleForShaftCompression(hardpoints, -corner.droopStop.gap);
        const auto bumpEntry = angleForShaftCompression(hardpoints, corner.bumpStop.gap);

        for (const auto angle :
             {hardpoints.droopAngle, droopEntry, 0.5 * (hardpoints.droopAngle + droopEntry), 0.0,
              0.4 * hardpoints.bumpAngle, bumpEntry, 0.95 * hardpoints.bumpAngle})
        {
            const auto state = solveCornerWithJacobian(hardpoints, angle, 0.0);
            REQUIRE(state.has_value());

            const auto damper = solveDamperForce(corner, *state, 0.6);
            const auto spring = solveSpringForce(corner, *state);
            const auto legacyJacobian =
                replicaLengthPerAngle(hardpoints, damperElementOf(hardpoints), state->wishboneAngle);

            // The fused branch's precondition, which routes the Golf through the identity path —
            // asserted through the authored condition and the replica both.
            REQUIRE(coaxialSpring(hardpoints));
            REQUIRE(spring.lengthPerAngle == legacyJacobian);
            REQUIRE(damper.lengthPerAngle == legacyJacobian);

            const auto compression = damperShaftCompression(corner, damper);
            const auto bumpForce = corner.bumpStop.force(compression - corner.bumpStop.gap, damper.velocity);
            const auto droopForce = -corner.droopStop.force(-compression - corner.droopStop.gap, -damper.velocity);
            const auto stopAxisForce = bumpForce + droopForce;

            // The stops' contribution and the fused product, element against legacy, exact.
            REQUIRE(stopAxisForce * damper.lengthPerAngle == stopAxisForce * legacyJacobian);
            const auto axisForce = spring.force + damper.force + bumpForce + droopForce;
            REQUIRE(axisForce * damper.lengthPerAngle == axisForce * legacyJacobian);
        }

        // Both stops actually fire inside the swept positions, so the equalities above are not
        // statements about zero.
        const auto deepBump = solveCornerWithJacobian(hardpoints, 0.95 * hardpoints.bumpAngle, 0.0);
        const auto fullDroop = solveCornerWithJacobian(hardpoints, hardpoints.droopAngle, 0.0);
        REQUIRE(deepBump.has_value());
        REQUIRE(fullDroop.has_value());
        const auto bumpCompression = damperShaftCompression(corner, solveDamperForce(corner, *deepBump, 0.0));
        const auto droopCompression = damperShaftCompression(corner, solveDamperForce(corner, *fullDroop, 0.0));
        REQUIRE(corner.bumpStop.force(bumpCompression - corner.bumpStop.gap) > 0.0);
        REQUIRE(corner.droopStop.force(-droopCompression - corner.droopStop.gap) > 0.0);
    }
}

TEST_CASE("the stops' generalised contribution follows the damper attachment only", "[physics][suspension][element]")
{
    // Split-role isolation for the contribution itself, with the stop genuinely engaged: the
    // spring seat cannot reach it, the damper pickup moves it, and the spring Jacobian wired in
    // by mistake is a visibly different number. The synthetic corner's bump gap is tightened so
    // the stop fires at the probe position — test geometry, not the Golf's.
    auto base = CornerSetup{};
    base.hardpoints = splitCorner(CornerSide::Right);
    base.hardpoints.springMount = SpringMount::OnLowerWishbone;
    base.hardpoints.springChassis = syntheticSpring(CornerSide::Right).chassis;
    base.hardpoints.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    base.springRate = 30000.0;
    base.springFreeLength = 0.5;
    base.bumpStop.gap = 0.005;

    auto movedSpring = base;
    movedSpring.hardpoints.springWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    auto movedDamper = base;
    movedDamper.hardpoints.damperWishbone += glm::dvec3(-0.02, 0.005, 0.0);

    const auto contribution = [](const CornerSetup& corner, const SuspensionState& state)
    {
        const auto damper = solveDamperForce(corner, state, 0.6);
        const auto compression = damperShaftCompression(corner, damper);
        const auto stopAxisForce =
            corner.bumpStop.force(compression - corner.bumpStop.gap, damper.velocity) +
            -corner.droopStop.force(-compression - corner.droopStop.gap, -damper.velocity);

        return std::pair{stopAxisForce, stopAxisForce * damper.lengthPerAngle};
    };

    const auto state = solveCornerWithJacobian(base.hardpoints, 0.06, 0.0);
    REQUIRE(state.has_value());

    const auto [baseForce, baseContribution] = contribution(base, *state);
    REQUIRE(baseForce > 0.0);

    // The spring seat cannot reach it.
    const auto [springForce, springContribution] = contribution(movedSpring, *state);
    REQUIRE(springForce == baseForce);
    REQUIRE(springContribution == baseContribution);

    // The damper pickup moves it.
    const auto damperState = solveCornerWithJacobian(movedDamper.hardpoints, 0.06, 0.0);
    REQUIRE(damperState.has_value());
    const auto [damperForce, damperContribution] = contribution(movedDamper, *damperState);
    REQUIRE(std::abs(damperContribution - baseContribution) > 1.0);

    // The spring Jacobian wired in by mistake is a visibly different number.
    const auto springSide = solveSpringForce(base, *state);
    REQUIRE(std::abs(baseForce * springSide.lengthPerAngle - baseContribution) > 1.0);

    static_cast<void>(damperForce);
}

TEST_CASE("stop engagement through the force law has not moved", "[physics][suspension][element]")
{
    // The step-1 characterisation, read through the whole migrated chain this time: the wheel
    // travel at which the stop FORCE first fires — law over element compression — sits where the
    // geometry-only pins already put it.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto engagementByForce = [](const CornerSetup& corner, const bool bump)
    {
        auto low = 0.0;
        auto high = bump ? corner.hardpoints.bumpAngle : corner.hardpoints.droopAngle;
        for (auto iteration = 0; iteration < 60; iteration++)
        {
            const auto middle = 0.5 * (low + high);
            const auto state = raceengine::solveCorner(corner.hardpoints, middle, 0.0);
            REQUIRE(state.has_value());
            const auto compression = damperShaftCompression(
                corner,
                DamperForceSolution{
                    .length = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), middle).length});
            const auto force = bump ? corner.bumpStop.force(compression - corner.bumpStop.gap)
                                    : corner.droopStop.force(-compression - corner.droopStop.gap);
            (force <= 0.0 ? low : high) = middle;
        }

        const auto solved = raceengine::solveCorner(corner.hardpoints, 0.5 * (low + high), 0.0);
        REQUIRE(solved.has_value());
        return solved->wheelTravel * 1000.0;
    };

    const auto& front = setup->corners[0];
    const auto& rear = setup->corners[2];

    REQUIRE(engagementByForce(front, true) == Catch::Approx(21.6).margin(0.1));
    REQUIRE(engagementByForce(front, false) == Catch::Approx(-44.5).margin(0.1));
    REQUIRE(engagementByForce(rear, true) == Catch::Approx(20.3).margin(0.1));
    REQUIRE(engagementByForce(rear, false) == Catch::Approx(-40.4).margin(0.1));
}

TEST_CASE("the coaxial determination keys the fused branch as before", "[physics][suspension][element]")
{
    // Step 14's guard re-key: the fused branch is chosen by the authored coil-over condition now,
    // not by comparing element Jacobians against the legacy state field — and it must select the
    // same branch for every representable corner. Golf: fused. Split corner: split. The edge case
    // — a separately-mounted spring authored on exactly the damper's points — takes the fused
    // branch, exactly as the old bit-equality guard would have, because identical points make
    // identical Jacobian bits.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        REQUIRE(coaxialSpring(setup->corners[index].hardpoints));
    }

    auto split = splitCorner(CornerSide::Right);
    split.springMount = SpringMount::OnLowerWishbone;
    split.springChassis = syntheticSpring(CornerSide::Right).chassis;
    split.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    REQUIRE_FALSE(coaxialSpring(split));

    auto coaxialByPoints = splitCorner(CornerSide::Right);
    coaxialByPoints.springMount = SpringMount::OnLowerWishbone;
    coaxialByPoints.springChassis = coaxialByPoints.damperChassis;
    coaxialByPoints.springWishbone = coaxialByPoints.damperWishbone;
    REQUIRE(coaxialSpring(coaxialByPoints));
}

TEST_CASE("the setup validator reads the element and refuses identically", "[physics][suspension][element]")
{
    // Step 14's validator migration: the design and end damper lengths come off the element now,
    // and they are the full-solve values bit for bit — so a never-closes refusal fires with the
    // same wording and the same printed number it always carried, and every passing corner still
    // passes.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto corners = std::vector<CornerHardpoints>{};
    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        corners.push_back(setup->corners[index].hardpoints);
    }

    auto split = splitCorner(CornerSide::Right);
    split.springMount = SpringMount::OnLowerWishbone;
    split.springChassis = syntheticSpring(CornerSide::Right).chassis;
    split.springWishbone = syntheticSpring(CornerSide::Right).wishbone;
    corners.push_back(split);

    for (const auto& hardpoints : corners)
    {
        const auto element = damperElementOf(hardpoints);
        for (const auto angle : {0.0, hardpoints.bumpAngle, hardpoints.droopAngle})
        {
            REQUIRE(solveElement(hardpoints, element, angle).length == replicaLength(hardpoints, element, angle));
        }
    }

    // The built Golf corners still validate.
    for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
    {
        REQUIRE(validateCornerSetup(setup->corners[index]).has_value());
    }

    // A gap the linkage cannot close refuses with the legacy value in the legacy wording — the
    // replica supplies that value now that the solver carries no damper channel.
    const auto& front = setup->corners[0];
    const auto frontElement = damperElementOf(front.hardpoints);
    const auto legacyCompression = replicaLength(front.hardpoints, frontElement, 0.0) -
                                   replicaLength(front.hardpoints, frontElement, front.hardpoints.bumpAngle);
    const auto legacyExtension = replicaLength(front.hardpoints, frontElement, front.hardpoints.droopAngle) -
                                 replicaLength(front.hardpoints, frontElement, 0.0);

    auto neverClosesBump = front;
    neverClosesBump.bumpStop.gap = 0.060;
    const auto bumpRefused = validateCornerSetup(neverClosesBump);
    REQUIRE_FALSE(bumpRefused.has_value());
    REQUIRE(bumpRefused.error().find("never closes: the damper only compresses") != std::string::npos);
    REQUIRE(bumpRefused.error().find(std::to_string(legacyCompression)) != std::string::npos);

    auto neverClosesDroop = front;
    neverClosesDroop.droopStop.gap = 0.060;
    const auto droopRefused = validateCornerSetup(neverClosesDroop);
    REQUIRE_FALSE(droopRefused.has_value());
    REQUIRE(droopRefused.error().find("never closes: the damper only extends") != std::string::npos);
    REQUIRE(droopRefused.error().find(std::to_string(legacyExtension)) != std::string::npos);
}
