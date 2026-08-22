// The stalled-in-gear chatter, measured before it is fixed: `./EngineTests "[.stall-chatter]"`.
//
// Recorded as known and unfixed: "a stalled car in gear with no brakes chatters ±480 N·m at the tick
// rate, because the clutch's slipping torque through first gear exceeds what the wheel's inertia
// absorbs in a tick and there is no arresting clamp on driveline torque the way there is on the
// brake." It is being reclassified as a fix, because a stalled car in gear is a constant state in a
// pursuit game — botched launches, crashes, roadblock impacts — and half a kilonewton-metre arriving
// at the tick rate will be blamed on the tyre model or the force feedback.
//
// The recorded *cause* is a hypothesis and this probe exists to test it rather than repeat it. What
// it prints is where the torque actually alternates and what it is alternating against.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DrivelineState;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeDriveline;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::wheelInertias;

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

} // namespace

TEST_CASE("what a stalled car in gear is actually doing", "[.stall-chatter]")
{
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto vehicle = built.value();
    const auto driveline = golfGtiMk7Driveline();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(vehicle, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    // **A stall, produced rather than assumed.** The engine is started, put in first, and then
    // dragged under with the brake — which is the way a driver reaches this state — and the brake is
    // released afterwards, because "no brakes" is what the recorded fault says.
    auto engine = DrivelineState{};
    startEngine(driveline, engine);

    auto road = std::array<double, cornerCount>{};

    const auto advance = [&](const VehicleInput& input)
    {
        const auto torques =
            stepDriveline(driveline, engine,
                          {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed, state.corners[2].wheelSpeed,
                           state.corners[3].wheelSpeed},
                          wheelInertias(vehicle), road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(vehicle, state, input, torques->wheel, world.value(), tick);
        REQUIRE(stepped.has_value());

        road = roadTorques(stepped.value());

        return torques.value();
    };

    // **Latched stalled, directly.** Dragging it under on the brake no longer works — `antiStallPedal`
    // is exactly the feature that stops it, and it does — so the state is set rather than reached.
    // That is not cheating: nothing in a session calls `startEngine` after spawn, so the latch is a
    // state the car can be left in by a crash or a deliberate clutch foot and cannot get out of.
    engine.engine = raceengine::EngineState::Stalled;
    engine.engineSpeed = 0.0;

    // **And rolling**, which is the case the fault is about: a car that stalls while moving keeps
    // its wheels turning, and they drag a dead engine through the clutch and the first-gear
    // reduction. A stationary stalled car has nothing to drag and reports nothing at all — measured,
    // and it is why the first attempt at this probe found no chatter.
    const auto speed = GENERATE(3.0, 10.0, 25.0);
    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        state.corners[index].wheelSpeed = speed / 0.31;
    }

    // **And the shaft placed with them.** A state written with the wheels turning and the shaft at
    // rest is not a slow car, it is an impossible one: the compliance reads the difference as a
    // twist nobody wound in. Without this the first tick reported -7438 N·m at 25 m/s, which is the
    // fixture's fault and not the model's — and is very likely the same artefact behind the recorded
    // "±480 N·m" this probe was written to reproduce.
    raceengine::placeDriveline(driveline, engine, speed / 0.31);

    std::printf("\n  latched stalled at %.0f m/s: engine %.1f rad/s, shaft %.1f rad/s\n", speed,
                engine.engineSpeed, engine.shaftSpeed);

    auto rolling = VehicleInput{};
    rolling.gear = 1;

    std::printf("\n%6s %11s %11s %11s %11s %11s\n", "tick", "wheel N.m", "engine r/s", "clutch N.m", "wheel r/s",
                "sign flips");

    auto previous = 0.0;
    auto flips = 0;
    auto peak = 0.0;
    auto totalVariation = 0.0;

    for (auto step = 0; step < 720; step++)
    {
        const auto torques = advance(rolling);
        const auto driven = torques.wheel[0];

        if (step > 0 && driven * previous < 0.0)
        {
            flips++;
        }

        totalVariation += std::abs(driven - previous);
        peak = std::max(peak, std::abs(driven));
        previous = driven;

        if (step % 120 == 0 || step < 4)
        {
            std::printf("%6d %11.2f %11.4f %11.2f %11.5f %11d\n", step, driven, engine.engineSpeed,
                        torques.clutch, state.corners[0].wheelSpeed, flips);
        }
    }

    std::printf("  over 720 ticks with no brakes: peak |wheel torque| %.1f N.m, %d sign flips,\n"
                "  total variation %.0f N.m (a smooth torque would be near its own range)\n",
                peak, flips, totalVariation);
}
