// What seal friction is worth to a car that has just been hit, measured because the seat said so.
//
// Dominic's report on 2026-08-27, after driving the build that first carried it: *"the rear end
// settles realistically now after hitting a wall. Before it would bounce like a ball."* That is a
// claim about **ring-down**, and it is the one thing a Coulomb term is supposed to be worth: friction
// dissipates a fixed amount per unit of distance travelled whatever the speed, so it keeps working
// on a small residual oscillation long after a viscous damper — whose force goes to zero with the
// velocity — has stopped mattering.
//
// This probe turns that into a number and an A/B, because a seat report with a named mechanism still
// has to be isolated. It drops the car onto its wheels and counts how long the rear takes to stop
// moving, with the shipped 107 N and with it zeroed and nothing else touched.
//
// Hidden behind a dot tag — `./EngineTests "[.damper-friction]"`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;

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

[[nodiscard]] ProvingGroundDescriptor plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 200.0;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<raceengine::Feature>{};

    return descriptor;
}

struct RingDown
{
    // The classical settling time: the last moment the corner was further than `settledBand` from
    // where it ends up, seconds. **Not** "the last moment anything moved" — that first attempt
    // measured when the last tenth of a micron happened and reported the whole window every time.
    double settlingTime = 0.0;
    // How far the corner still swings over the final half second, millimetres.
    double residual = 0.0;
    // The largest excursion of the whole event, millimetres, so a shorter settle cannot be bought by
    // a softer hit.
    double peak = 0.0;
};

// A tenth of the impact's own scale, which on this drop is about 2 mm of travel. Wide enough that
// tick noise cannot hold it open and narrow enough that a bouncing corner cannot hide inside it.
constexpr auto settledBand = 0.002;

// Settle the car, then hit it: a downward velocity step, which is what a body dropping back onto its
// wheels after a collision does to the suspension. A wall is not needed to ask the question — what
// the seat noticed is the *decay*, and a clean impulse measures that without a contact solver in the
// middle of it.
[[nodiscard]] RingDown ringDown(const VehicleSetup& setup, const PhysicsWorld& world)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 100.0);

    for (auto step = 0; step < 2880; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    // The rear corner's resting travel, which every amplitude below is measured against.
    const auto settled = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick);
    REQUIRE(settled.has_value());
    const auto rest = settled->corners[2].suspension.wheelTravel;

    state.chassis.linearVelocity = glm::dvec3(0.0, -1.5, 0.0);

    constexpr auto steps = 360 * 8;
    constexpr auto tail = 360 / 2;

    auto result = RingDown{};
    auto outsideBand = 0;

    for (auto step = 1; step <= steps; step++)
    {
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick);
        REQUIRE(stepped.has_value());

        const auto excursion = std::abs(stepped->corners[2].suspension.wheelTravel - rest);

        result.peak = std::max(result.peak, excursion * 1000.0);

        if (excursion > settledBand)
        {
            outsideBand = step;
        }

        if (step > steps - tail)
        {
            result.residual = std::max(result.residual, excursion * 1000.0);
        }
    }

    result.settlingTime = static_cast<double>(outsideBand) * tick;

    return result;
}

} // namespace

TEST_CASE("damper friction: what it is worth to a car that has just been hit", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    auto without = built.value();
    for (auto& corner : without.corners)
    {
        corner.damperFriction = 0.0;
    }

    const auto shipped = ringDown(built.value(), world.value());
    const auto viscous = ringDown(without, world.value());

    WARN("=== rear ring-down after a 1.5 m/s downward impulse ===");
    WARN("shipped (" << built->corners[2].damperFriction << " N seal friction): peak " << shipped.peak
                     << " mm, settles to 2 mm in " << shipped.settlingTime << " s, residual " << shipped.residual
                     << " mm");
    WARN("viscous only (0 N):                 peak " << viscous.peak << " mm, settles to 2 mm in "
                                                     << viscous.settlingTime << " s, residual " << viscous.residual
                                                     << " mm");
    WARN("  settling time " << (shipped.settlingTime / viscous.settlingTime - 1.0) * 100.0 << "%, residual "
                            << (shipped.residual / viscous.residual - 1.0) * 100.0 << "%");

    // The claim the seat made, asserted: the corner stops sooner with friction than without it, and
    // it is not bought by a softer hit. The numbers are **not** pinned — what is pinned is the
    // direction, because that is what the report said and it is what a Coulomb term must do. A pinned
    // figure here would be a characterisation of a borrowed 107 N nobody has measured on this car.
    CHECK(shipped.settlingTime < viscous.settlingTime);
    CHECK(shipped.residual < viscous.residual);
    CHECK(shipped.peak > 0.9 * viscous.peak);
}
