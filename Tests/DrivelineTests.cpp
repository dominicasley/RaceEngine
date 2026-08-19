#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::clutchPackLsd;
using raceengine::cornerCount;
using raceengine::Curve;
using raceengine::Differential;
using raceengine::DifferentialState;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::DrivenAxle;
using raceengine::EngineModel;
using raceengine::engineTorque;
using raceengine::fillDrivelineTelemetry;
using raceengine::generateProvingGround;
using raceengine::noDriveTorque;
using raceengine::openDifferential;
using raceengine::PhysicsWorld;
using raceengine::placeholderDriveline;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::spool;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TelemetryFrame;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;

// No road under the wheels. The cases that use it hold the wheel speeds by hand, so there is no
// tire reaction to feed back and none of what it would change is the question they are asking.
constexpr std::array<double, cornerCount> noRoadTorque{};

std::array<double, cornerCount> sameInertia(const double value)
{
    return {value, value, value, value};
}

// The driver's packet, which is what the driveline tick takes. Every case below wants the clutch out
// and left alone, so it says so once here rather than four times.
VehicleInput driving(const double throttle, const std::int32_t gear)
{
    auto input = VehicleInput{};
    input.throttle = throttle;
    input.gear = gear;

    return input;
}

// An engine that has been started and then set where the case wants it. `DrivelineState{}` is a car
// with the key out and stays at rest however high a speed is written into it, which is the point.
DrivelineState runningAt(const DrivelineSetup& setup, const double engineSpeed)
{
    auto state = DrivelineState{};
    startEngine(setup, state);
    state.engineSpeed = engineSpeed;

    return state;
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

    auto state = runningAt(setup, 300.0);
    const auto torques =
        stepDriveline(setup, state, {0.0, 0.0, 0.0, 0.0}, sameInertia(1.2), noRoadTorque, driving(1.0, 0), tick);
    REQUIRE(torques.has_value());

    // The engine revs freely and no torque reaches the wheels.
    REQUIRE(state.engineSpeed > 300.0);
    for (const auto wheel : torques->wheel)
    {
        REQUIRE(wheel == 0.0);
    }
}

TEST_CASE("the differential is asked a question rather than performing a division",
          "[physics][driveline][differential]")
{
    // The brief's structural requirement, and what makes an LSD a data change later rather than a
    // restructure. All three of these go through one code path.
    // The state is the axle's rather than the differential's, so it arrives as an argument: a
    // `Differential` is setup, handed around by const reference, and all-wheel drive asks the same
    // one twice.
    auto axle = DifferentialState{};

    SECTION("an open diff splits evenly whatever the wheels are doing")
    {
        const auto open = openDifferential();

        const auto even = open.split(axle, 50.0, 50.0, 400.0, tick);
        REQUIRE(even.left == Catch::Approx(200.0));
        REQUIRE(even.right == Catch::Approx(200.0));

        // One wheel spinning changes nothing, which is exactly the open diff's famous failing.
        const auto spinning = open.split(axle, 200.0, 50.0, 400.0, tick);
        REQUIRE(spinning.left == Catch::Approx(200.0));
        REQUIRE(spinning.right == Catch::Approx(200.0));
    }

    SECTION("a spool refuses to let the wheels differ")
    {
        const auto locked = spool();
        const auto split = locked.split(axle, 60.0, 50.0, 400.0, tick);

        // Enormous transfer away from the fast wheel and into the slow one.
        REQUIRE(split.left < -1000.0);
        REQUIRE(split.right > 1000.0);
        REQUIRE(split.left + split.right == Catch::Approx(400.0));
    }

    SECTION("a clutch pack transfers up to its capacity and no further")
    {
        const auto lsd = clutchPackLsd(60.0, 0.35, 0.15);

        // Under power: preload plus the power ramp.
        const auto driving = lsd.split(axle, 60.0, 50.0, 400.0, tick);
        const auto capacity = 60.0 + 0.35 * 400.0;

        REQUIRE(driving.right - driving.left == Catch::Approx(2.0 * capacity));
        REQUIRE(driving.left + driving.right == Catch::Approx(400.0));

        // And it says so: the pack was asked for more than it could hold and reports both numbers,
        // which is what the lock/slip machine will read when it replaces the clamp.
        REQUIRE(axle.capacity == Catch::Approx(capacity));
        REQUIRE(axle.transfer == Catch::Approx(capacity));

        // Off power the coast ramp is shallower, which is most of an LSD's character.
        const auto coasting = lsd.split(axle, 60.0, 50.0, -400.0, tick);
        const auto coastCapacity = 60.0 + 0.15 * 400.0;
        REQUIRE(coasting.right - coasting.left == Catch::Approx(2.0 * coastCapacity));

        // And the preload acts with no torque going through at all, which is what holds a car
        // straight with one wheel on ice.
        const auto idle = lsd.split(axle, 60.0, 50.0, 0.0, tick);
        REQUIRE(idle.right - idle.left == Catch::Approx(120.0));
    }

    SECTION("a small speed difference is inside the pack rather than at its limit")
    {
        const auto lsd = clutchPackLsd(60.0, 0.35, 0.15);
        const auto barely = lsd.split(axle, 50.05, 50.0, 400.0, tick);

        // 0.05 rad/s against a stiffness of 400 is 20 Nm, well under the 200 Nm capacity.
        REQUIRE(barely.right - barely.left == Catch::Approx(40.0));
        REQUIRE(axle.transfer == Catch::Approx(20.0));
    }

    SECTION("two axles on one differential keep their own state")
    {
        // The all-wheel-drive trap: the same `Differential` is asked twice, and once there is any
        // state at all a single object between them would have the front stamping on the rear.
        const auto lsd = clutchPackLsd(60.0, 0.35, 0.15);

        auto front = DifferentialState{};
        auto rear = DifferentialState{};

        static_cast<void>(lsd.split(front, 60.0, 50.0, 400.0, tick));
        static_cast<void>(lsd.split(rear, 50.0, 50.0, 400.0, tick));

        REQUIRE(front.transfer == Catch::Approx(60.0 + 0.35 * 400.0));
        REQUIRE(rear.transfer == Catch::Approx(0.0));
    }
}

TEST_CASE("a locked clutch locks, and the engine keeps its own speed", "[physics][driveline][clutch]")
{
    // Engine speed is independent state even though a locked clutch makes it derivable. Deriving it
    // would work today and would be a restructure the day the clutch is allowed to slip, which is
    // the whole reason the brief insists on this.
    const auto setup = placeholderDriveline();
    const auto inertias = sameInertia(1.2);

    auto state = runningAt(setup, 100.0);
    const auto wheelSpeed = 30.0;
    const auto reduction = setup.gearbox.reduction(3);

    // Wheels held at a fixed speed, so the clutch has something to pull against.
    for (auto step = 0; step < 3600; step++)
    {
        REQUIRE(stepDriveline(setup, state, sameInertia(wheelSpeed), inertias, noRoadTorque, driving(0.0, 3), tick)
                    .has_value());
    }

    // The engine has been dragged to the speed the gearing demands.
    REQUIRE(state.engineSpeed == Catch::Approx(wheelSpeed * reduction).epsilon(0.02));

    SECTION("and it does so from either side")
    {
        auto fast = runningAt(setup, 600.0);

        for (auto step = 0; step < 3600; step++)
        {
            REQUIRE(stepDriveline(setup, fast, sameInertia(wheelSpeed), inertias, noRoadTorque, driving(0.0, 3), tick)
                        .has_value());
        }

        REQUIRE(fast.engineSpeed == Catch::Approx(wheelSpeed * reduction).epsilon(0.02));
    }

    SECTION("without ever going unstable, which an explicit stiff coupling would")
    {
        // A locked clutch is a very stiff spring between two small inertias. Stepped explicitly the
        // two trade energy at the timestep's frequency until one reaches infinity; solved, it simply
        // converges. Started far out of step, with torque, and it must stay finite.
        auto wild = runningAt(setup, 700.0);
        auto worst = 0.0;

        for (auto step = 0; step < 3600; step++)
        {
            REQUIRE(stepDriveline(setup, wild, sameInertia(5.0), inertias, noRoadTorque, driving(1.0, 1), tick)
                        .has_value());
            worst = std::max(worst, std::abs(wild.engineSpeed));
            REQUIRE(std::isfinite(wild.engineSpeed));
            // And never below zero, which is the property the three `std::max(0.0, ...)` floors used
            // to hold by hand and the stall model now holds by construction.
            REQUIRE(wild.engineSpeed >= 0.0);
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
    auto state = runningAt(setup, wheelSpeed * setup.gearbox.reduction(2));

    SECTION("front wheel drive")
    {
        setup.driven = DrivenAxle::Front;
        const auto torques =
            stepDriveline(setup, state, sameInertia(wheelSpeed), inertias, noRoadTorque, driving(1.0, 2), tick);
        REQUIRE(torques.has_value());

        REQUIRE(torques->wheel[0] > 0.0);
        REQUIRE(torques->wheel[1] > 0.0);
        REQUIRE(torques->wheel[2] == 0.0);
        REQUIRE(torques->wheel[3] == 0.0);
    }

    SECTION("rear wheel drive")
    {
        setup.driven = DrivenAxle::Rear;
        const auto torques =
            stepDriveline(setup, state, sameInertia(wheelSpeed), inertias, noRoadTorque, driving(1.0, 2), tick);
        REQUIRE(torques.has_value());

        REQUIRE(torques->wheel[0] == 0.0);
        REQUIRE(torques->wheel[1] == 0.0);
        REQUIRE(torques->wheel[2] > 0.0);
        REQUIRE(torques->wheel[3] > 0.0);
    }

    SECTION("all wheel drive")
    {
        setup.driven = DrivenAxle::All;
        const auto torques =
            stepDriveline(setup, state, sameInertia(wheelSpeed), inertias, noRoadTorque, driving(1.0, 2), tick);
        REQUIRE(torques.has_value());

        for (const auto wheel : torques->wheel)
        {
            REQUIRE(wheel > 0.0);
        }
    }

    SECTION("and only the driven axle's differential is asked anything")
    {
        // An open diff holds nothing and so records nothing; a pack is what makes the two axles'
        // states distinguishable at all.
        setup.driven = DrivenAxle::Rear;
        setup.differential = clutchPackLsd(60.0, 0.35, 0.15);

        REQUIRE(stepDriveline(setup, state, sameInertia(wheelSpeed), inertias, noRoadTorque, driving(1.0, 2), tick)
                    .has_value());

        REQUIRE(state.differentials[0].capacity == 0.0);
        REQUIRE(state.differentials[1].capacity > 0.0);
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
        REQUIRE(stepVehicle(vehicle.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    auto drivelineState = runningAt(driveline, 200.0);

    // One statement of the wheel inertias, from the setup that owns them, rather than an array
    // assembled here that nothing made agree with what the vehicle tick integrates against.
    const auto inertias = wheelInertias(vehicle.value());

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 2;

    auto peakSpeed = 0.0;
    auto lastFrame = TelemetryFrame{};
    // Averaged over the last second rather than sampled at the end, because this car spends that
    // second against the rev limiter and the limiter now has a cycle of its own: with a restore band
    // under it the fuel comes and goes at a rate the engine's inertia sets, tens of milliseconds
    // rather than the two ticks a bare threshold gave. Through a locked clutch the cut phase brakes
    // the driven wheels below the free-rolling ones, so a single tick reports whichever half of that
    // cycle it landed in. The mean is the claim being made.
    auto drivenSum = 0.0;
    auto rollingSum = 0.0;
    auto sampled = 0;
    // The road's answer from the previous tick, which is what the clutch's constraint is solved
    // against. Zero on the first one, before the tire has said anything.
    auto road = noRoadTorque;

    for (auto step = 0; step < 3600; step++)
    {
        const auto torques = stepDriveline(driveline, drivelineState,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, road, input, tick);
        REQUIRE(torques.has_value());

        // The chain's torque is an argument to the vehicle tick, not something applied to the wheels
        // behind its back. Wheel speed is integrated once, inside `stepVehicle`, where the brake
        // clamp can see everything acting on the wheel.
        auto stepped = stepVehicle(vehicle.value(), state, input, torques->wheel, world.value(), tick);
        REQUIRE(stepped.has_value());

        // And the driveline's telemetry channels are filled here, by the caller that owns the
        // driveline state. `:Vehicle` does not import `:Driveline` and cannot fill them itself,
        // which is exactly why `VehicleStep::telemetry` is a by-value member.
        fillDrivelineTelemetry(stepped->telemetry, drivelineState, torques.value());
        road = roadTorques(stepped.value());
        lastFrame = stepped->telemetry;

        peakSpeed = std::max(peakSpeed, state.chassis.linearVelocity.z);

        if (step >= 3240)
        {
            drivenSum += state.corners[0].wheelSpeed;
            rollingSum += state.corners[2].wheelSpeed;
            sampled++;
        }
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
        REQUIRE(sampled == 360);
        REQUIRE(drivenSum / static_cast<double>(sampled) > rollingSum / static_cast<double>(sampled));
        REQUIRE(state.corners[2].wheelSpeed > 0.0);
    }

    SECTION("the engine ends up where the gearing puts it")
    {
        const auto expected =
            0.5 * (state.corners[0].wheelSpeed + state.corners[1].wheelSpeed) * driveline.gearbox.reduction(input.gear);

        REQUIRE(drivelineState.engineSpeed == Catch::Approx(expected).epsilon(0.05));
        REQUIRE(drivelineState.engineSpeed < driveline.engine.limiterSpeed * 1.02);
    }

    SECTION("and the caller is what puts it on the telemetry frame")
    {
        REQUIRE(lastFrame.engineSpeed == Catch::Approx(drivelineState.engineSpeed));
    }
}

TEST_CASE("driveline torque and the brake meet on one wheel", "[physics][driveline][integration]")
{
    // What the single-owner integration is for. Driveline torque used to be put on `wheelSpeed`
    // before the vehicle tick ran, which placed it *upstream* of the tire's slip and upstream of the
    // brake clamp that is supposed to be holding the wheel still — while the road's own torque sat
    // downstream of both. So a braked wheel spun up every tick, the tire read that spin as slip and
    // made a force out of it, and the clamp then quietly took the speed back out again. The wheel
    // never moved and pushed the car anyway.
    //
    // Integrated once, from both torques together, the clamp sees everything on the wheel and the
    // tire sees a wheel that is genuinely stopped. This is the shape of launch control, of creep and
    // of a torque converter stalled against the brakes.
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

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    // Settled and stationary, so anything that moves afterwards moved because of the torque.
    auto settled = VehicleState{};
    settled.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(vehicle.value(), settled, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    // 800 Nm a front wheel, under the 1400 the brake makes, so the brake wins outright.
    const auto drive = std::array<double, cornerCount>{800.0, 800.0, 0.0, 0.0};

    SECTION("on the brake the wheel is held, and a held wheel makes no force")
    {
        auto state = settled;

        for (auto step = 0; step < 360; step++)
        {
            auto input = VehicleInput{};
            input.throttle = 1.0;
            input.brake = 1.0;
            input.gear = 1;

            const auto stepped = stepVehicle(vehicle.value(), state, input, drive, world.value(), tick);
            REQUIRE(stepped.has_value());

            REQUIRE(state.corners[0].wheelSpeed == Catch::Approx(0.0).margin(1e-9));
            // To a newton, not to the bit, and the residue is not the drive torque: a stationary
            // tire's carcass deflection accumulates against whatever the car is creeping at, which
            // is what holds a parked car on a slope. 800 Nm reaching the road through a 0.31 m
            // wheel would be about 2600 N.
            REQUIRE(stepped->corners[0].contact.tyre.longitudinal == Catch::Approx(0.0).margin(1.0));
        }

        REQUIRE(std::abs(state.chassis.linearVelocity.z) < 1e-3);
    }

    SECTION("and off it the same torque spins the wheel up, so the assertion above is not a tautology")
    {
        auto state = settled;

        for (auto step = 0; step < 360; step++)
        {
            auto input = VehicleInput{};
            input.throttle = 1.0;
            input.gear = 1;

            REQUIRE(stepVehicle(vehicle.value(), state, input, drive, world.value(), tick).has_value());
        }

        REQUIRE(state.corners[0].wheelSpeed > 1.0);
        REQUIRE(state.chassis.linearVelocity.z > 1.0);
    }
}
