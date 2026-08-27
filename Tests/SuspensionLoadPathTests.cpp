// The geometric load path: the patch Jacobian, and what switching it into the force path does.
//
// `SuspensionState::patchPerAngle` is the derivative of the point the road pushes on, against the
// corner's own generalised coordinate. It is one vector and it carries four things this model did
// not have — roll centre, jacking, anti-dive and anti-squat — so it is worth pinning from more than
// one direction. Three of these cases do that, and none of them is the production code checking
// itself:
//
//   - against an **independent replica** that differences the patch at a coarse step, which is the
//     discipline `SuspensionElementTests` established for the element evaluator;
//   - against the **roll-centre construction**, which is a completely different piece of geometry
//     arriving at the same claim about where the patch is going;
//   - against a **sign law** on synthetic geometry, so that "which way does a wheel move in bump"
//     is stated two-sidedly rather than argued from a picture.
//
// Full account: docs/suspension-load-path-brief.md.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::computeRollCentre;
using raceengine::cornerCount;
using raceengine::CornerHardpoints;
using raceengine::CornerSide;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeholderCorner;
using raceengine::ProvingGroundDescriptor;
using raceengine::solveCorner;
using raceengine::solveCornerWithJacobian;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

// The replica: the same quantity, computed here, from the exported solve alone. A coarse step on
// purpose — the production Jacobian is differenced at 1e-6, so agreeing with a secant four orders of
// magnitude wider says the derivative is a property of the geometry rather than of the step.
[[nodiscard]] glm::dvec3 patchSecant(const CornerHardpoints& hardpoints, const double angle, const double rackTravel,
                                     const double step)
{
    const auto behind = solveCorner(hardpoints, angle - step, rackTravel);
    const auto ahead = solveCorner(hardpoints, angle + step, rackTravel);
    REQUIRE(behind.has_value());
    REQUIRE(ahead.has_value());

    return (ahead->contactPatch - behind->contactPatch) / (2.0 * step);
}

[[nodiscard]] double sideViewRatio(const CornerHardpoints& hardpoints)
{
    const auto state = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
    REQUIRE(state.has_value());

    return state->patchPerAngle.z / state->patchPerAngle.y;
}

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

} // namespace

TEST_CASE("the patch Jacobian is the derivative of the patch", "[physics][suspension][loadpath]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = built->corners[index].hardpoints;

        for (const auto through : std::vector<double>{0.1, 0.3, 0.5, 0.7, 0.9})
        {
            const auto angle = hardpoints.droopAngle + through * (hardpoints.bumpAngle - hardpoints.droopAngle);

            const auto state = solveCornerWithJacobian(hardpoints, angle, 0.0);
            REQUIRE(state.has_value());

            const auto replica = patchSecant(hardpoints, angle, 0.0, 1e-3);

            // A central difference is second-order, so widening the step by 1e3 widens the
            // truncation error by 1e6 — which is why this is a relative band and not equality.
            CHECK(std::abs(state->patchPerAngle.x - replica.x) < 1e-4 * std::abs(state->travelPerAngle));
            CHECK(std::abs(state->patchPerAngle.y - replica.y) < 1e-4 * std::abs(state->travelPerAngle));
            CHECK(std::abs(state->patchPerAngle.z - replica.z) < 1e-4 * std::abs(state->travelPerAngle));
        }
    }
}

TEST_CASE("the lateral half of the patch Jacobian is the roll centre", "[physics][suspension][loadpath]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    // Two constructions of one geometry. The roll centre is where the line from the patch to the
    // front-view instant centre crosses the centreline; the Jacobian is the patch's own velocity.
    // Both say the same thing about which way the patch scrubs as the wheel rises, so
    // `patchPerAngle.x / patchPerAngle.y` must be `rollCentreHeight / contactPatch.x`.
    //
    // **Pinned at design, and not across the travel, for a stated reason rather than to pass.**
    // `computeRollCentre` is a two-link front-view construction that never looks at the steering, so
    // it misses the tie-rod-imposed steer rotation — step 16 of docs/suspension-geometry-audit.md
    // measured that term and found it worth 2.6% of the camber rate at design. The next case
    // measures what it is worth here and pins its shape; this one pins the agreement where the audit
    // says the neglected term is small.
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = built->corners[index].hardpoints;

        auto state = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
        REQUIRE(state.has_value());
        computeRollCentre(hardpoints, state.value());

        const auto fromJacobian = state->patchPerAngle.x / state->patchPerAngle.y;
        const auto fromConstruction = state->rollCentreHeight / state->contactPatch.x;

        CHECK(std::abs(fromJacobian - fromConstruction) < 0.01);

        // And the same statement in millimetres, because that is the number the audit quotes: the
        // Jacobian's own roll centre is the construction's, a few millimetres lower.
        const auto jacobianRollCentre = fromJacobian * state->contactPatch.x;
        CHECK(std::abs(jacobianRollCentre - state->rollCentreHeight) < 0.008);
        CHECK(jacobianRollCentre < state->rollCentreHeight);
    }
}

TEST_CASE("the roll-centre construction and the Jacobian part company away from design",
          "[physics][suspension][loadpath]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    // The characterisation of the disagreement above, so that nobody reads the design-point
    // agreement as agreement everywhere and then uses `rollCentreHeight` to predict a force.
    //
    // The construction is a diagnostic and the Jacobian is what the linkage does. Away from design
    // the neglected steer term grows until, at the front bump end, the two do not even share a sign
    // — which is the whole reason the force path reads the Jacobian and nothing reads the
    // construction.
    const auto& front = built->corners[0].hardpoints;

    const auto disagreementAt = [&](const double angle)
    {
        auto state = solveCornerWithJacobian(front, angle, 0.0);
        REQUIRE(state.has_value());
        computeRollCentre(front, state.value());

        return std::abs(state->patchPerAngle.x / state->patchPerAngle.y -
                        state->rollCentreHeight / state->contactPatch.x);
    };

    CHECK(disagreementAt(0.0) < disagreementAt(0.05));
    CHECK(disagreementAt(0.0) < disagreementAt(-0.05));
    CHECK(disagreementAt(0.05) > 0.02);

    // At the front bump end the construction says the roll centre is still above ground and the
    // linkage says the patch has started scrubbing the other way. Both are statements about this
    // geometry; only one of them is the geometry.
    auto atBump = solveCornerWithJacobian(front, front.bumpAngle, 0.0);
    REQUIRE(atBump.has_value());
    computeRollCentre(front, atBump.value());

    CHECK(atBump->rollCentreHeight > 0.0);
    CHECK(atBump->patchPerAngle.x / atBump->patchPerAngle.y < 0.0);
}

TEST_CASE("which way a wheel swings in bump is the lower arm's own axis", "[physics][suspension][loadpath]")
{
    // The sign law, on synthetic geometry whose answer is known by symmetry: the placeholder
    // corner's wishbone axes are square across the car and both arms' pivots are level, so the wheel
    // rises very nearly straight.
    //
    // **Very nearly, and not exactly**, and the residual is worth naming rather than tightening
    // away: that corner's tie rod sits 140 mm ahead of the wheel centre, so steering geometry turns
    // the upright a little as the arm swings and carries the patch with it. It is the same neglected
    // term that separates the roll-centre construction from the Jacobian above. So the law below is
    // stated as a *difference* from the level corner, which is the only way it is a law.
    const auto level = placeholderCorner(CornerSide::Left, 1.4);
    const auto square = sideViewRatio(level);
    CHECK(std::abs(square) < 0.01);

    // Raise the rear pivot and the arm's axis slopes up toward the rear, so the wheel swings
    // **forward** as it rises. On a front axle that is anti-dive: a rearward braking force then does
    // negative virtual work in the corner's coordinate and resists compression.
    auto rearHigh = level;
    rearHigh.lower.rearPivot.y += 0.02;
    CHECK(sideViewRatio(rearHigh) - square > 0.01);

    // And the mirror, which is what makes this a law rather than an observation.
    auto rearLow = level;
    rearLow.lower.rearPivot.y -= 0.02;
    CHECK(sideViewRatio(rearLow) - square < -0.01);

    // Equal and opposite about the level corner, to the extent the linkage is linear over 20 mm of
    // bush height.
    const auto raised = sideViewRatio(rearHigh) - square;
    const auto dropped = sideViewRatio(rearLow) - square;
    CHECK(std::abs(raised + dropped) < 0.1 * std::abs(raised));
}

TEST_CASE("the patch's vertical rate is the wheel centre's only while the wheel is upright",
          "[physics][suspension][loadpath]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto& hardpoints = built->corners[0].hardpoints;

    // At design this car's camber is zero by construction, and the patch is then directly below the
    // wheel centre: turning the wheel about its own contact point is a second-order motion, so the
    // two vertical rates are the same number. That is why switching the load path on does not move a
    // parked car.
    const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());
    CHECK(std::abs(design->patchPerAngle.y - design->travelPerAngle) < 1e-9 * std::abs(design->travelPerAngle));

    // Away from design the wheel leans, and the tyre's radius turning with it puts a real difference
    // between them. Small, and not zero — which is the point: `travelPerAngle` is the wheel centre's
    // rate and the road does not push on the wheel centre.
    const auto leaned = solveCornerWithJacobian(hardpoints, 0.12, 0.0);
    REQUIRE(leaned.has_value());
    CHECK(std::abs(leaned->camber) > 0.01);
    CHECK(std::abs(leaned->patchPerAngle.y - leaned->travelPerAngle) > 1e-6);
}

TEST_CASE("the geometric load path changes what the springs carry, not what the tyres carry",
          "[physics][suspension][loadpath]")
{
    const JoltGuard jolt;

    // The model's own default is off, so a car that does not ask for the geometric path gets the
    // arithmetic every figure in docs/ was measured under.
    CHECK_FALSE(VehicleSetup{}.geometricLoadPath);

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 600.0;
    descriptor.width = 600.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto springs = built.value();
    springs.geometricLoadPath = false;

    auto geometric = built.value();
    geometric.geometricLoadPath = true;

    // One car, settled once, then stepped one tick each way from **the same state**. Both see
    // identical geometry, identical tyre deflection and therefore identical road forces; the only
    // thing that can differ is how much of those forces the corner's degree of freedom feels.
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 300.0);
    for (auto step = 0; step < 720; step++)
    {
        REQUIRE(stepVehicle(springs, state, VehicleInput{}, noDriveTorque, world.value(), 1.0 / 360.0).has_value());
    }

    // Cornering: the state is given a lateral velocity so the tyres make a lateral force, which is
    // the term the two models disagree about.
    state.chassis.linearVelocity = glm::dvec3(3.0, 0.0, 25.0);

    auto rolling = state;
    const auto withSprings = stepVehicle(springs, rolling, VehicleInput{}, noDriveTorque, world.value(), 1.0 / 360.0);
    REQUIRE(withSprings.has_value());

    auto same = state;
    const auto withGeometry = stepVehicle(geometric, same, VehicleInput{}, noDriveTorque, world.value(), 1.0 / 360.0);
    REQUIRE(withGeometry.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        // What the road puts on the tyre is the same number. Load transfer is fixed by the whole-car
        // free body whatever path the force takes inside a corner, and this is that statement as an
        // assertion: the switch is an attitude change, not a grip change.
        CHECK(withSprings->corners[index].forces.tireVertical == withGeometry->corners[index].forces.tireVertical);
        CHECK(withSprings->corners[index].contact.tyre.lateral == withGeometry->corners[index].contact.tyre.lateral);

        // And what the corner's degree of freedom feels is not, because the in-plane forces reach it
        // now. A tyre making a lateral force on a linkage with a roll centre above the ground jacks.
        CHECK(withSprings->corners[index].generalisedForce != withGeometry->corners[index].generalisedForce);
    }
}
