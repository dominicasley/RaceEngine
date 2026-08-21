// A full brake application must not stall a car the automation is driving.
//
// The failure this pins was reported from the seat as the wheel dying wholesale: 100% on the
// load-cell brake and throttle, brake and steering all stop answering. What actually happened is a
// race the auto-clutch lost — a full application arrests the wheels in tens of milliseconds, the
// engagement logic watches the road side and so answers late, and the pedal's foot-speed rate limit
// needs a quarter of a second the engine does not have. The locked clutch dragged the engine below
// its stall floor, `settleEngineSpeed` latched it, and nothing in a session calls `startEngine`
// after spawn — so the car was dead until relaunch, which from the seat is indistinguishable from
// the device failing. The Golf is a dual clutch: its automation's emergency disengage is hydraulic,
// which is why `antiStallPedal` is allowed past the rate limit.
//
// The brake demand is a step, deliberately: a ramped foot is the easier case, and the step is what
// a load cell delivers when a leg stamps on it.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DrivelineState;
using raceengine::EngineState;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
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

struct Car
{
    VehicleState state;
    DrivelineState engine;
    std::array<double, cornerCount> road{};
};

void advance(const VehicleSetup& vehicle, const raceengine::DrivelineSetup& driveline, const PhysicsWorld& world,
             Car& car, const VehicleInput& input)
{
    const auto torques = stepDriveline(driveline, car.engine,
                                       {car.state.corners[0].wheelSpeed, car.state.corners[1].wheelSpeed,
                                        car.state.corners[2].wheelSpeed, car.state.corners[3].wheelSpeed},
                                       wheelInertias(vehicle), car.road, input, tick);
    REQUIRE(torques.has_value());

    auto stepped = stepVehicle(vehicle, car.state, input, torques->wheel, world, tick);
    REQUIRE(stepped.has_value());

    car.road = roadTorques(stepped.value());
}

// Stand the car up, start it, and drive it up through the gears for the given time with the
// driver's foot nowhere near the clutch pedal — which is what driving on a wheel and load-cell
// pedals is unless the driver deliberately heel-and-toes.
Car driven(const VehicleSetup& vehicle, const raceengine::DrivelineSetup& driveline, const PhysicsWorld& world,
           const int launchTicks, std::int32_t& gearReached)
{
    auto car = Car{};
    car.state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(vehicle, car.state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    startEngine(driveline, car.engine);

    auto idling = VehicleInput{};
    idling.clutch = 1.0;
    idling.gear = 0;

    for (auto step = 0; step < 720; step++)
    {
        advance(vehicle, driveline, world, car, idling);
    }

    auto launching = VehicleInput{};
    launching.throttle = 1.0;
    launching.gear = 1;

    for (auto step = 0; step < launchTicks; step++)
    {
        // The simplest driver: short-shift at 5000 rpm and hold the gear.
        if (car.engine.engineSpeed > 5000.0 * 0.10472 && launching.gear < 6)
        {
            launching.gear++;
        }

        advance(vehicle, driveline, world, car, launching);
    }

    gearReached = launching.gear;

    return car;
}

} // namespace

TEST_CASE("a full brake application cannot stall the automation", "[physics][clutch][antistall]")
{
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 800.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto& vehicle = built.value();
    const auto driveline = golfGtiMk7Driveline();

    // Both regimes that stalled before the anti-stall existed: braking from speed, where the wheels
    // lock and take the driveline to zero through a still-engaged clutch, and a car lugging a tall
    // gear near walking pace, where the engine has no margin at all.
    for (const auto launchTicks : {720, 2880})
    {
        std::int32_t gear = 0;
        auto car = driven(vehicle, driveline, world.value(), launchTicks, gear);

        auto braking = VehicleInput{};
        braking.brake = 1.0;
        braking.gear = gear;

        auto lowest = car.engine.engineSpeed;

        for (auto step = 0; step < 2160; step++)
        {
            advance(vehicle, driveline, world.value(), car, braking);
            lowest = std::min(lowest, car.engine.engineSpeed);

            REQUIRE(car.engine.engine == EngineState::Running);
        }

        // Not merely alive but kept away from the floor: the band is fully open at 0.55 of idle,
        // and the fight is allowed one excursion into it while the capacity dies, never through it.
        REQUIRE(lowest > driveline.engine.stallSpeed);

        // Stopped, in gear, idling — which is what a dual clutch does at a red light.
        REQUIRE(glm::length(car.state.chassis.linearVelocity) < 0.5);
        REQUIRE(car.engine.engineSpeed == Catch::Approx(driveline.engine.idleSpeed).epsilon(0.05));

        // And the car is still a car: brake off, first gear, and it pulls away.
        auto away = VehicleInput{};
        away.throttle = 1.0;
        away.gear = 1;

        for (auto step = 0; step < 1080; step++)
        {
            advance(vehicle, driveline, world.value(), car, away);
        }

        REQUIRE(car.engine.engine == EngineState::Running);
        REQUIRE(glm::length(car.state.chassis.linearVelocity) > 3.0);
    }
}
