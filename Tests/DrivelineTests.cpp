#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::applyDrivelineTorques;
using raceengine::bringUpJolt;
using raceengine::clutchPackLsd;
using raceengine::cornerCount;
using raceengine::Curve;
using raceengine::Differential;
using raceengine::DrivelineSetup;
using raceengine::DrivenAxle;
using raceengine::EngineModel;
using raceengine::engineTorque;
using raceengine::generateProvingGround;
using raceengine::openDifferential;
using raceengine::PhysicsWorld;
using raceengine::placeholderDriveline;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::spool;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;

std::array<double, cornerCount> sameInertia(const double value)
{
    return {value, value, value, value};
}

} // namespace

TEST_CASE("the engine is a measured curve, not a shape", "[physics][driveline]")
{
    const auto setup = placeholderDriveline();
    const auto& engine = setup.engine;

    SECTION("full throttle follows the curve")
    {
        REQUIRE(engineTorque(engine, 300.0, 1.0) == Catch::Approx(engine.torque.at(300.0)));
        REQUIRE(engineTorque(engine, 470.0, 1.0) == Catch::Approx(engine.torque.at(470.0)));
    }

    SECTION("part throttle is less, and a shut throttle brakes")
    {
        const auto full = engineTorque(engine, 300.0, 1.0);
        const auto half = engineTorque(engine, 300.0, 0.5);
        const auto shut = engineTorque(engine, 300.0, 0.0);

        REQUIRE(half < full);
        REQUIRE(half > 0.0);
        // Engine braking, which is not the torque curve read backwards: the curve is a full-throttle
        // measurement and has no business carrying the closed-throttle behaviour as a negative.
        REQUIRE(shut < 0.0);
    }

    SECTION("the limiter is a cliff")
    {
        const auto below = engineTorque(engine, engine.limiterSpeed - 1.0, 1.0);
        const auto above = engineTorque(engine, engine.limiterSpeed + 1.0, 1.0);

        REQUIRE(below > 100.0);
        REQUIRE(above < 0.0);
    }
}

TEST_CASE("neutral disconnects the chain and nothing else changes", "[physics][driveline]")
{
    const auto setup = placeholderDriveline();

    REQUIRE(setup.gearbox.reduction(0) == 0.0);
    REQUIRE(setup.gearbox.reduction(1) > setup.gearbox.reduction(6));
    REQUIRE(setup.gearbox.reduction(-1) < 0.0);

    auto engineSpeed = 300.0;
    const auto torques = stepDriveline(setup, engineSpeed, {0.0, 0.0, 0.0, 0.0}, sameInertia(1.2), 1.0, 0, tick);

    // The engine revs freely and no torque reaches the wheels.
    REQUIRE(engineSpeed > 300.0);
    for (const auto wheel : torques.wheel)
    {
        REQUIRE(wheel == 0.0);
    }
}

TEST_CASE("the differential is asked a question rather than performing a division",
          "[physics][driveline][differential]")
{
    // The brief's structural requirement, and what makes an LSD a data change later rather than a
    // restructure. All three of these go through one code path.
    SECTION("an open diff splits evenly whatever the wheels are doing")
    {
        const auto open = openDifferential();

        const auto even = open.split(50.0, 50.0, 400.0);
        REQUIRE(even.left == Catch::Approx(200.0));
        REQUIRE(even.right == Catch::Approx(200.0));

        // One wheel spinning changes nothing, which is exactly the open diff's famous failing.
        const auto spinning = open.split(200.0, 50.0, 400.0);
        REQUIRE(spinning.left == Catch::Approx(200.0));
        REQUIRE(spinning.right == Catch::Approx(200.0));
    }

    SECTION("a spool refuses to let the wheels differ")
    {
        const auto locked = spool();
        const auto split = locked.split(60.0, 50.0, 400.0);

        // Enormous transfer away from the fast wheel and into the slow one.
        REQUIRE(split.left < -1000.0);
        REQUIRE(split.right > 1000.0);
        REQUIRE(split.left + split.right == Catch::Approx(400.0));
    }

    SECTION("a clutch pack transfers up to its capacity and no further")
    {
        const auto lsd = clutchPackLsd(60.0, 0.35, 0.15);

        // Under power: preload plus the power ramp.
        const auto driving = lsd.split(60.0, 50.0, 400.0);
        const auto capacity = 60.0 + 0.35 * 400.0;

        REQUIRE(driving.right - driving.left == Catch::Approx(2.0 * capacity));
        REQUIRE(driving.left + driving.right == Catch::Approx(400.0));

        // Off power the coast ramp is shallower, which is most of an LSD's character.
        const auto coasting = lsd.split(60.0, 50.0, -400.0);
        const auto coastCapacity = 60.0 + 0.15 * 400.0;
        REQUIRE(coasting.right - coasting.left == Catch::Approx(2.0 * coastCapacity));

        // And the preload acts with no torque going through at all, which is what holds a car
        // straight with one wheel on ice.
        const auto idle = lsd.split(60.0, 50.0, 0.0);
        REQUIRE(idle.right - idle.left == Catch::Approx(120.0));
    }

    SECTION("a small speed difference is inside the pack rather than at its limit")
    {
        const auto lsd = clutchPackLsd(60.0, 0.35, 0.15);
        const auto barely = lsd.split(50.05, 50.0, 400.0);

        // 0.05 rad/s against a stiffness of 400 is 20 Nm, well under the 200 Nm capacity.
        REQUIRE(barely.right - barely.left == Catch::Approx(40.0));
    }
}

TEST_CASE("a locked clutch locks, and the engine keeps its own speed", "[physics][driveline][clutch]")
{
    // Engine speed is independent state even though a locked clutch makes it derivable. Deriving it
    // would work today and would be a restructure the day the clutch is allowed to slip, which is
    // the whole reason the brief insists on this.
    const auto setup = placeholderDriveline();
    const auto inertias = sameInertia(1.2);

    auto engineSpeed = 100.0;
    const auto wheelSpeed = 30.0;
    const auto reduction = setup.gearbox.reduction(3);

    // Wheels held at a fixed speed, so the clutch has something to pull against.
    for (auto step = 0; step < 3600; step++)
    {
        static_cast<void>(stepDriveline(setup, engineSpeed, sameInertia(wheelSpeed), inertias, 0.0, 3, tick));
    }

    // The engine has been dragged to the speed the gearing demands.
    REQUIRE(engineSpeed == Catch::Approx(wheelSpeed * reduction).epsilon(0.02));

    SECTION("and it does so from either side")
    {
        auto fast = 600.0;
        for (auto step = 0; step < 3600; step++)
        {
            static_cast<void>(stepDriveline(setup, fast, sameInertia(wheelSpeed), inertias, 0.0, 3, tick));
        }

        REQUIRE(fast == Catch::Approx(wheelSpeed * reduction).epsilon(0.02));
    }

    SECTION("without ever going unstable, which an explicit stiff coupling would")
    {
        // A locked clutch is a very stiff spring between two small inertias. Stepped explicitly the
        // two trade energy at the timestep's frequency until one reaches infinity; solved, it simply
        // converges. Started far out of step, with torque, and it must stay finite.
        auto wild = 700.0;
        auto worst = 0.0;

        for (auto step = 0; step < 3600; step++)
        {
            static_cast<void>(stepDriveline(setup, wild, sameInertia(5.0), inertias, 1.0, 1, tick));
            worst = std::max(worst, std::abs(wild));
            REQUIRE(std::isfinite(wild));
        }

        REQUIRE(worst < 5000.0);
    }
}

TEST_CASE("only the driven wheels are driven", "[physics][driveline]")
{
    auto setup = placeholderDriveline();
    const auto inertias = sameInertia(1.2);

    // Engine and wheels consistent with the gear they are in. Picked arbitrarily they are not: at
    // 300 rad/s with the wheels at 40 in a 9.09 reduction the *wheels* are dragging the engine up,
    // the clutch torque is negative, and every driven wheel correctly reports being braked.
    const auto wheelSpeed = 40.0;
    auto engineSpeed = wheelSpeed * setup.gearbox.reduction(2);

    SECTION("front wheel drive")
    {
        setup.driven = DrivenAxle::Front;
        const auto torques = stepDriveline(setup, engineSpeed, sameInertia(wheelSpeed), inertias, 1.0, 2, tick);

        REQUIRE(torques.wheel[0] > 0.0);
        REQUIRE(torques.wheel[1] > 0.0);
        REQUIRE(torques.wheel[2] == 0.0);
        REQUIRE(torques.wheel[3] == 0.0);
    }

    SECTION("rear wheel drive")
    {
        setup.driven = DrivenAxle::Rear;
        const auto torques = stepDriveline(setup, engineSpeed, sameInertia(wheelSpeed), inertias, 1.0, 2, tick);

        REQUIRE(torques.wheel[0] == 0.0);
        REQUIRE(torques.wheel[1] == 0.0);
        REQUIRE(torques.wheel[2] > 0.0);
        REQUIRE(torques.wheel[3] > 0.0);
    }

    SECTION("all wheel drive")
    {
        setup.driven = DrivenAxle::All;
        const auto torques = stepDriveline(setup, engineSpeed, sameInertia(wheelSpeed), inertias, 1.0, 2, tick);

        for (const auto wheel : torques.wheel)
        {
            REQUIRE(wheel > 0.0);
        }
    }
}

TEST_CASE("the car accelerates on the throttle", "[physics][driveline][integration]")
{
    // The chain end to end: throttle, engine, clutch, gearbox, differential, wheels, tires, road.
    struct Guard
    {
        Guard()
        {
            REQUIRE(bringUpJolt().has_value());
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        ~Guard()
        {
            tearDownJolt();
        }
    } guard;

    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());
    const auto driveline = placeholderDriveline();

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 1200.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 4.0;
    descriptor.features = {};

    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(vehicle.value(), state, VehicleInput{}, world.value(), tick).has_value());
    }

    state.engineSpeed = 200.0;

    auto inertias = std::array<double, cornerCount>{};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        inertias[index] = vehicle->corners[index].wheelInertia;
    }

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 2;

    auto peakSpeed = 0.0;
    for (auto step = 0; step < 3600; step++)
    {
        const auto torques = stepDriveline(driveline, state.engineSpeed,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, input.throttle, input.gear, tick);
        applyDrivelineTorques(state, torques, inertias, tick);

        const auto stepped = stepVehicle(vehicle.value(), state, input, world.value(), tick);
        REQUIRE(stepped.has_value());

        peakSpeed = std::max(peakSpeed, state.chassis.linearVelocity.z);
    }

    SECTION("it goes")
    {
        REQUIRE(state.chassis.linearVelocity.z > 10.0);
        REQUIRE(peakSpeed == Catch::Approx(state.chassis.linearVelocity.z).epsilon(0.05));
    }

    SECTION("the driven wheels turn and the undriven ones roll")
    {
        // Front wheel drive: the fronts are pushing and are turning a little faster than the road,
        // the rears are simply rolling with it.
        REQUIRE(state.corners[0].wheelSpeed > state.corners[2].wheelSpeed);
        REQUIRE(state.corners[2].wheelSpeed > 0.0);
    }

    SECTION("the engine ends up where the gearing puts it")
    {
        const auto expected =
            0.5 * (state.corners[0].wheelSpeed + state.corners[1].wheelSpeed) * driveline.gearbox.reduction(input.gear);

        REQUIRE(state.engineSpeed == Catch::Approx(expected).epsilon(0.05));
        REQUIRE(state.engineSpeed < driveline.engine.limiterSpeed * 1.02);
    }
}
