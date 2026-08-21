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
using raceengine::CouplingMode;
using raceengine::defaultProvingGround;
using raceengine::Differential;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::generateProvingGround;
using raceengine::noDriveTorque;
using raceengine::openDifferential;
using raceengine::PhysicsWorld;
using raceengine::placeDriveline;
using raceengine::placeholderDriveline;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::spool;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::wheelInertias;

namespace
{

constexpr std::array<double, cornerCount> noRoadTorque{};

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

// Tarmac under one driven wheel and grass under the other, which is the only condition in which the
// three differentials are three different things.
//
// On uniform high grip they are not, and that is the model being right rather than the test being
// weak: a clutch pack's capacity on power here is 1768 N.m against the 168 N.m it takes to hold the
// wheels together, so it locks and stays locked, and a locked LSD *is* a spool. Measured on a
// 0.6-steering second-gear exit, the pack came back bit-for-bit identical to the spool. What
// separates them is a wheel that can actually run away — 0.42 of grip against 1.00 — and that is
// also the case a driver would name if asked what a diff is for.
ProvingGroundDescriptor splitGrip()
{
    auto descriptor = defaultProvingGround();
    descriptor.cellSize = 0.5;

    return descriptor;
}

// What the launch looked like, reduced to the channels the acceptance criterion names.
struct Launch
{
    // How far the car got, which is the whole question: a differential that cannot hold the spinning
    // wheel delivers the *other* wheel the same small torque, and the car sits there.
    double speed = 0.0;
    double yawRate = 0.0;
    // The driven pair. One of these is on grass and free to run away; the other is on tarmac and is
    // where the drive has to end up.
    double spinningWheel = 0.0;
    double grippingWheel = 0.0;
    double wheelSplit = 0.0;
    // Transitions between locked and slipping, per second of simulated time.
    double packChatter = 0.0;
    double clutchChatter = 0.0;
};

// A standing start with one driven wheel on grass. The differential is the only thing that varies
// between calls: same car, same road, same place on it, same pedal.
Launch splitGripLaunch(const Differential& differential, const double deltaTime, const double seconds = 3.0)
{
    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(splitGrip()).value());
    REQUIRE(world.has_value());

    auto setup = placeholderDriveline();
    setup.differential = differential;

    auto state = VehicleState{};
    // Astride the boundary, and far enough into the band that the easing at its edge is behind the
    // car rather than under it.
    state.chassis.position = glm::dvec3(0.0, 0.52, 105.0);

    const auto settleSteps = static_cast<int>(4.0 / deltaTime);
    for (auto step = 0; step < settleSteps; step++)
    {
        REQUIRE(
            stepVehicle(vehicle.value(), state, VehicleInput{}, noDriveTorque, world.value(), deltaTime).has_value());
    }

    auto drivelineState = DrivelineState{};
    startEngine(setup, drivelineState);
    placeDriveline(setup, drivelineState, 0.0);

    const auto inertias = wheelInertias(vehicle.value());

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 1;

    auto road = noRoadTorque;
    auto launch = Launch{};

    auto yawSum = 0.0;
    auto yawSamples = 0;
    auto packTransitions = 0;
    auto clutchTransitions = 0;
    auto previousPack = drivelineState.differentials[0].pack.mode;
    auto previousClutch = drivelineState.coupling.coupling.mode;

    const auto driveSteps = static_cast<int>(seconds / deltaTime);
    for (auto step = 0; step < driveSteps; step++)
    {
        const auto torques = stepDriveline(setup, drivelineState,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, road, input, deltaTime);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(vehicle.value(), state, input, torques->wheel, world.value(), deltaTime);
        REQUIRE(stepped.has_value());

        road = roadTorques(stepped.value());

        if (drivelineState.differentials[0].pack.mode != previousPack)
        {
            packTransitions++;
            previousPack = drivelineState.differentials[0].pack.mode;
        }

        if (drivelineState.coupling.coupling.mode != previousClutch)
        {
            clutchTransitions++;
            previousClutch = drivelineState.coupling.coupling.mode;
        }

        yawSum += stepped->telemetry.yawRate;
        yawSamples++;
    }

    launch.speed = state.chassis.linearVelocity.z;
    launch.yawRate = yawSum / static_cast<double>(yawSamples);

    // Read by which one ran away rather than by index, because which side the grass is on is the
    // ground's business and not this test's.
    const auto left = state.corners[0].wheelSpeed;
    const auto right = state.corners[1].wheelSpeed;

    launch.spinningWheel = std::max(left, right);
    launch.grippingWheel = std::min(left, right);
    launch.wheelSplit = launch.spinningWheel - launch.grippingWheel;

    launch.packChatter = static_cast<double>(packTransitions) / seconds;
    launch.clutchChatter = static_cast<double>(clutchTransitions) / seconds;

    return launch;
}

} // namespace

TEST_CASE("the same launch tells you which differential is fitted", "[physics][differential][exit]")
{
    const JoltGuard jolt;

    // The acceptance criterion, and it is a criterion because a differential is the one component
    // that can be entirely absent and still produce a car that drives: an LSD that is not working
    // answers exactly as an open one on every corner where neither wheel is spinning, and nothing
    // about the lap time says so.
    constexpr auto tick = 1.0 / 360.0;

    // A mild road pack, and the strength is part of the experiment rather than a detail. Capacity is
    // `preload + ramp * input`, so a 0.35 ramp on this car's 4879 N.m of first-gear axle torque is
    // 1768 N.m against the ~550 it takes to hold these two wheels together — it locks, and a locked
    // pack *is* a spool. That is the mechanism being right, and it is also why a strong LSD cannot
    // answer this question: the only regime in which three differentials are three different things
    // is the one where the middle is neither open nor locked.
    const auto open = splitGripLaunch(openDifferential(), tick);
    const auto locked = splitGripLaunch(spool(), tick);
    const auto pack = splitGripLaunch(clutchPackLsd(30.0, 0.06, 0.03), tick);

    SECTION("the driven wheels tell them apart, in the order the mechanism says")
    {
        // An open differential holds no torque across itself, so the wheel on grass takes the drive
        // and runs; a spool holds everything and the two can only turn together; a pack holds its
        // preload plus what the ramp gives it, which is between. Anything else fitted here lands in
        // the same order for the same reason.
        REQUIRE(locked.wheelSplit < pack.wheelSplit);
        REQUIRE(pack.wheelSplit < open.wheelSplit);

        // And the spool means it: the two driven wheels are one shaft.
        REQUIRE(locked.wheelSplit < 0.5);
    }

    SECTION("and the car goes, in the same order and for the same reason")
    {
        // What the wheel split costs: torque an open diff cannot hold on the grass is torque the
        // tarmac wheel never receives.
        REQUIRE(open.speed < pack.speed);
        REQUIRE(pack.speed < locked.speed);

        // Measurably, rather than by arithmetic noise between three runs of the same car.
        REQUIRE(locked.speed > 1.2 * open.speed);
    }

    SECTION("and the yaw they produce differs, which is what the driver feels")
    {
        // A front-drive car launched on split grip yaws, because the two driven wheels are pushing
        // by different amounts about the centre of gravity. How much is exactly what the
        // differential decides.
        REQUIRE(std::abs(open.yawRate - locked.yawRate) > 0.01);
        REQUIRE(std::abs(pack.yawRate - locked.yawRate) > 0.005);
    }

    SECTION("and every one of them stayed a number")
    {
        for (const auto& launch : {open, locked, pack})
        {
            REQUIRE(std::isfinite(launch.yawRate));
            REQUIRE(std::isfinite(launch.wheelSplit));
            REQUIRE(std::isfinite(launch.speed));
            REQUIRE(launch.speed > 0.0);
        }
    }
}

TEST_CASE("neither the clutch nor the pack hunts between locked and slipping", "[physics][differential][hunting]")
{
    const JoltGuard jolt;

    // Criterion 3, on the case most likely to produce hunting: a pack right at the edge of its
    // capacity with one wheel spinning and the other gripping, which is where a lock/slip decision
    // is genuinely marginal every tick.
    //
    // The measurement that makes it meaningful is the second one. A transition count that *halves
    // when the tick halves* is the timestep leaking into the answer rather than anything the
    // mechanism is doing, and "fewer transitions" cannot tell the two apart. This is the method the
    // rev limiter's chatter was settled with.
    const auto normal = splitGripLaunch(clutchPackLsd(30.0, 0.06, 0.03), 1.0 / 360.0);
    const auto fast = splitGripLaunch(clutchPackLsd(30.0, 0.06, 0.03), 1.0 / 720.0);

    SECTION("the rate is low in absolute terms")
    {
        // A device that decides a few times a second is responding to the road. One deciding sixty
        // times a second is responding to its own arithmetic, and it is audible.
        REQUIRE(normal.packChatter < 20.0);
        REQUIRE(normal.clutchChatter < 20.0);
    }

    SECTION("and it is a property of the car rather than of the timestep")
    {
        // Doubling the rate must not double the count. Generous on the ratio because the two runs
        // are not the same trajectory to the bit — they are the same car driven the same way — but a
        // timestep artefact fails this by a factor rather than by a margin.
        const auto packRatio = fast.packChatter / std::max(normal.packChatter, 1e-9);
        const auto clutchRatio = fast.clutchChatter / std::max(normal.clutchChatter, 1e-9);

        CAPTURE(normal.packChatter, fast.packChatter, normal.clutchChatter, fast.clutchChatter);
        REQUIRE(packRatio < 2.0);
        REQUIRE(clutchRatio < 2.0);
    }
}
