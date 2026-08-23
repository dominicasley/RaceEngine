#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::advanceCorneringBrake;
using raceengine::advanceTractionControl;
using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::CorneringBrakeSetup;
using raceengine::CorneringBrakeState;
using raceengine::DrivelineState;
using raceengine::Feature;
using raceengine::frontLeft;
using raceengine::frontRight;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::rearLeft;
using raceengine::rearRight;
using raceengine::ReferenceSpeedSetup;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::TractionMode;
using raceengine::TractionSetup;
using raceengine::TractionState;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelCount;
using raceengine::wheelInertias;
using raceengine::WheelSpeedReadings;

// Traction control and XDS. Two systems that share the per-wheel brake path and nothing else, kept
// apart in the telemetry so that a question about which of them did something has an answer.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto fifty = 50.0 / 3.6;

constexpr auto plateLength = 900.0;
constexpr auto plateWidth = 200.0;
constexpr auto startZ = 20.0;

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

// The pedal is off, so every caliper is at atmosphere. Named rather than written as `{}` at the
// call, because a zero pressure array means "no brakes" and that should be legible.
constexpr auto noBrakePressure = std::array<double, cornerCount>{};

// Tarmac under one side, something slippery under the other. **The only surface on which a brake-
// based traction intervention has anything to do**: it is a differential lock, so it needs a
// difference across the axle to act on, and a uniform plate presents none however slippery it is.
[[nodiscard]] SurfaceMesh splitPlate(const double leftGrip, const double rightGrip)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(2);
    mesh->materials[0].gripMultiplier = leftGrip;
    mesh->materials[0].bumpiness = 0.0;
    mesh->materials[1].gripMultiplier = rightGrip;
    mesh->materials[1].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        const auto centroid =
            (mesh->vertices[mesh->indices[triangle * 3 + 0]] + mesh->vertices[mesh->indices[triangle * 3 + 1]] +
             mesh->vertices[mesh->indices[triangle * 3 + 2]]) /
            3.0;

        // **+x is the car's LEFT in this frame**, which is stated once in `outboardSign` and is the
        // trap that has cost this project a day before now. So the *positive* half of the plate is
        // the one the left-hand wheels run on, and a fixture that reads `x < 0` as "left" puts the
        // surfaces on the wrong sides and then blames the controller.
        mesh->surfaces[triangle] = centroid.x > 0.0 ? std::uint32_t{0} : std::uint32_t{1};
    }

    return mesh.value();
}

[[nodiscard]] SurfaceMesh gripPlate(const double grip)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    for (auto& material : mesh->materials)
    {
        material.gripMultiplier = grip;
        material.bumpiness = 0.0;
    }

    return mesh.value();
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
    }
}

struct LaunchResult
{
    double toFifty = -1.0;
    double toHundred = -1.0;
    double peakEngineReduction = 0.0;
    double peakTractionBrake = 0.0;
    // Which corner the peak landed on. A differential lock is only doing its job if it is the
    // *losing* wheel it brakes, and a scalar peak cannot say which one that was.
    std::size_t peakTractionBrakeWheel = cornerCount;
    double firstBrakeIntervention = -1.0;
    double firstEngineIntervention = -1.0;
    double meanDrivenSlip = 0.0;
    double peakDrivenSlip = 0.0;
    bool onPlate = true;
};

// A standing start from an idle in gear on the brakes, which is the state a car in the game is
// always launched from. **Launching from a default-constructed driveline instead measures a cold
// start**: the clutch pedal defaults to fully engaged and the first quarter second is torque dumped
// into a stationary tyre. `LaunchDiagnosticProbe` records that having cost a whole controller build.
[[nodiscard]] LaunchResult launch(const VehicleSetup& setup, const PhysicsWorld& world, AssistSetup assists,
                                  const double seconds)
{
    const auto driveline = golfGtiMk7Driveline();
    const auto inertias = wheelInertias(setup);

    auto state = VehicleState{};
    settle(setup, state, world, 0.0);

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    auto road = std::array<double, cornerCount>{};
    auto result = LaunchResult{};

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

        return sensors;
    };

    const auto speeds = [&]
    {
        return std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                               state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};
    };

    {
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto step = 0; step < 360; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup, 1.0), tick);
            const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, idling, torques->wheel, world, tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
            road = roadTorques(lastStep);
        }
    }

    // --- preconditions: a car idling in gear on its brakes, and nothing else ---
    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.05);
    REQUIRE(drivelineState.gear == 1);
    REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
    REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
    }

    auto gear = 1;
    const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

    auto samples = 0;

    for (auto step = 1; step <= static_cast<int>(seconds * 360.0); step++)
    {
        const auto now = static_cast<double>(step) * tick;

        // Shifted on road speed through the gear, which is what `LaunchDiagnosticProbe` established:
        // any signal taken off the driven wheels upshifts on wheelspin rather than on progress.
        const auto roadSideSpeed = std::abs(state.chassis.linearVelocity.z) / tyreRadius *
                                   driveline.gearbox.finalDrive * driveline.gearbox.ratio(gear);
        if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
        {
            gear++;
        }

        const auto command =
            updateAssists(assists, assistState, sense(), {.brake = 0.0, .throttle = 1.0}, noBrakePressure, tick);

        auto input = VehicleInput{};
        input.throttle = command.throttleScale;
        input.gear = gear;

        const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
        road = roadTorques(lastStep);

        result.peakEngineReduction = std::max(result.peakEngineReduction, command.channels.engineTorqueReduction);

        for (const auto wheel : {frontLeft, frontRight})
        {
            if (command.channels.tractionBrakeTorque[wheel] > result.peakTractionBrake)
            {
                result.peakTractionBrake = command.channels.tractionBrakeTorque[wheel];
                result.peakTractionBrakeWheel = wheel;
            }

            if (result.firstBrakeIntervention < 0.0 && command.channels.tractionBrakeTorque[wheel] > 1.0)
            {
                result.firstBrakeIntervention = now;
            }
        }

        if (result.firstEngineIntervention < 0.0 && command.channels.engineTorqueReduction > 0.05)
        {
            result.firstEngineIntervention = now;
        }

        const auto drivenSlip = 0.5 * (std::abs(lastStep.telemetry.wheels[frontLeft].slipRatio) +
                                       std::abs(lastStep.telemetry.wheels[frontRight].slipRatio));

        result.meanDrivenSlip += drivenSlip;
        result.peakDrivenSlip = std::max(result.peakDrivenSlip, drivenSlip);
        samples++;

        if (state.chassis.position.z > plateLength - 5.0)
        {
            result.onPlate = false;
        }

        const auto speed = state.chassis.linearVelocity.z;
        if (result.toFifty < 0.0 && speed >= fifty)
        {
            result.toFifty = now;
        }
        if (result.toHundred < 0.0 && speed >= hundred)
        {
            result.toHundred = now;
            break;
        }
    }

    result.meanDrivenSlip /= samples > 0 ? static_cast<double>(samples) : 1.0;

    return result;
}

[[nodiscard]] AssistSetup withTraction(const VehicleSetup& setup, const TractionMode mode)
{
    auto assists = golfGtiMk7Assists(setup);
    assists.traction.mode = mode;

    return assists;
}

// A wheel reading built by hand, so the cornering-brake cases can state a turn rather than drive one.
[[nodiscard]] raceengine::WheelSpeedReading reading(const double roadSpeed)
{
    return raceengine::WheelSpeedReading{.speed = roadSpeed / tyreRadius, .age = 0.001, .valid = true, .pulses = 100};
}

} // namespace

TEST_CASE("traction control is off unless something switches it on", "[assists][traction]")
{
    // The default that keeps 433 other tests measuring the car rather than the electronics.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto assists = golfGtiMk7Assists(setup.value());

    REQUIRE(assists.traction.mode == TractionMode::Off);
    REQUIRE_FALSE(assists.antilock.enabled);
    REQUIRE_FALSE(assists.cornering.enabled);
    REQUIRE_FALSE(raceengine::assistsEngaged(assists));

    // And with nothing engaged the layer says so, which is what makes the vehicle model take its own
    // unchanged path rather than a numerically equal one.
    auto state = AssistState{};
    const auto command = updateAssists(assists, state, AssistSensors{}, {.brake = 1.0, .throttle = 0.0},
                                       brakeCircuitPressures(setup.value(), 1.0), tick);

    REQUIRE_FALSE(command.brakes.commanded);
    REQUIRE(command.throttleScale == 1.0);
}

TEST_CASE("traction control is worth a little on dry tarmac and a great deal on a slippery one",
          "[assists][traction][launch]")
{
    // **Criterion 7.** A front-drive hatchback with 350 N.m at the crank spins its wheels off the
    // line on any surface; on dry it costs a few tenths, on a slippery one it costs the launch
    // entirely.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    SECTION("on dry tarmac")
    {
        const auto world = PhysicsWorld::create(gripPlate(1.0));
        REQUIRE(world.has_value());

        const auto plain = launch(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), 20.0);
        const auto assisted =
            launch(setup.value(), world.value(), withTraction(setup.value(), TractionMode::Full), 20.0);

        REQUIRE(plain.onPlate);
        REQUIRE(assisted.onPlate);
        REQUIRE(plain.toHundred > 0.0);
        REQUIRE(assisted.toHundred > 0.0);

        CAPTURE(plain.toHundred, assisted.toHundred, plain.peakDrivenSlip, assisted.peakDrivenSlip);

        // **The unassisted figure is the one already on record**, which is what says this fixture
        // reproduces the launch the rest of the project measures rather than a new one: 6.494 s,
        // against a published 6.4-6.7 s for a Mk7 GTI DSG without launch control.
        //
        // **6.550 until 2026-08-23**, when the load-sensitivity exponent was split and the
        // longitudinal axis took `tyres.ini`'s own `LS_EXP_X` instead of the lateral figure that had
        // been standing in for it. A launch transfers weight *off* the driven axle, so the front tyres
        // work below their nominal load, where a flatter exponent is worth a little less rather than a
        // little more — 0.8% by hand calculation and 0.9% measured, which is about as close as an
        // estimate of a whole-car figure from a single-wheel one has any right to be. Re-recorded
        // rather than widened: the point of the number is that it is the same launch.
        REQUIRE(plain.toHundred == Catch::Approx(6.494).margin(0.02));

        // Comparable or modestly better. Not transformative — on high grip there is not much
        // wheelspin to recover, and a system that found a lot here would be finding it somewhere it
        // is not.
        REQUIRE(assisted.toHundred < plain.toHundred);
        REQUIRE(assisted.toHundred > 0.85 * plain.toHundred);

        // And it got there by taking the wheelspin away rather than by some other route.
        REQUIRE(assisted.peakDrivenSlip < 0.5 * plain.peakDrivenSlip);
    }

    SECTION("on a slippery one")
    {
        const auto world = PhysicsWorld::create(gripPlate(0.35));
        REQUIRE(world.has_value());

        const auto plain = launch(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), 20.0);
        const auto assisted =
            launch(setup.value(), world.value(), withTraction(setup.value(), TractionMode::Full), 20.0);

        REQUIRE(plain.onPlate);
        REQUIRE(assisted.onPlate);

        CAPTURE(plain.toFifty, assisted.toFifty, plain.toHundred, assisted.toHundred, plain.peakDrivenSlip);

        // Both reach 50 km/h inside the window; only the assisted one reaches 100.
        REQUIRE(plain.toFifty > 0.0);
        REQUIRE(assisted.toFifty > 0.0);

        // Clearly better, and the margin is large enough that no tolerance argument is needed.
        REQUIRE(assisted.toFifty < 0.9 * plain.toFifty);
        REQUIRE(assisted.toHundred > 0.0);
        REQUIRE(plain.toHundred < 0.0);

        // Unassisted, the driven wheels are spinning several times faster than the car is going.
        REQUIRE(plain.peakDrivenSlip > 3.0);
    }
}

TEST_CASE("a launch with both wheels spinning together is the engine channel's alone", "[assists][traction][launch]")
{
    // **This case did not exist until 2026-08-23 and its absence is what let the brake channel be
    // the wrong device for as long as it was.** On a uniform surface both driven wheels break away
    // together, and then there is no cross-axle torque to redistribute: braking either one takes
    // drive away from the axle instead of moving it to the other side, and braking both is two discs
    // absorbing engine power to no purpose. The brakes genuinely cannot help here and must stay out.
    //
    // The old brake channel intervened anyway, because it was keyed on each wheel's slip against the
    // reference speed rather than against its partner — the engine channel's signal wired to a valve.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35));
    REQUIRE(world.has_value());

    const auto run = launch(setup.value(), world.value(), withTraction(setup.value(), TractionMode::Full), 20.0);

    CAPTURE(run.firstBrakeIntervention, run.firstEngineIntervention, run.peakTractionBrake, run.peakEngineReduction);

    // The wheels really did spin, or this proves nothing about restraint.
    REQUIRE(run.peakEngineReduction > 0.2);
    REQUIRE(run.firstEngineIntervention > 0.0);

    // And the brakes did not touch it.
    REQUIRE(run.peakTractionBrake == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(run.firstBrakeIntervention < 0.0);
}

TEST_CASE("the brake channel catches the transient and the engine channel sustains it", "[assists][traction][launch]")
{
    // **Criterion 8.** The design is the division of labour, so the telemetry has to be able to
    // show it: brake torque and engine torque reduction on separate channels, with the brake
    // arriving first because it is a valve and the engine channel is a throttle body.
    //
    // **Measured on a split surface since 2026-08-23**, and the change of surface is the criterion
    // becoming honest rather than being weakened. It used to run on a uniform mu 0.35 plate, where a
    // correctly built brake channel has nothing it can do — see the case above. A differential lock
    // needs a difference across the axle, so a split surface is the case it exists for, and it is
    // also the case a driver meets: one wheel on the wet line, one on the dry.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(splitPlate(1.0, 0.35));
    REQUIRE(world.has_value());

    const auto run = launch(setup.value(), world.value(), withTraction(setup.value(), TractionMode::Full), 20.0);

    CAPTURE(run.firstBrakeIntervention, run.firstEngineIntervention, run.peakTractionBrake, run.peakEngineReduction);

    // Both channels did something.
    REQUIRE(run.firstBrakeIntervention > 0.0);
    REQUIRE(run.firstEngineIntervention > 0.0);
    REQUIRE(run.peakTractionBrake > 100.0);
    REQUIRE(run.peakEngineReduction > 0.2);

    // **The ordering assertion that used to be here has moved to a unit case and this is why.** It
    // required the brake channel to intervene before the engine channel, on the grounds that a valve
    // is faster than a throttle body. That was a fair reading while both channels fired on the same
    // signal at the same instant; since the reform they fire on *different* signals — the engine on
    // axle-wide slip, the brake on a cross-axle difference — so which arrives first is a property of
    // which condition the surface presents first, not of how quickly either actuator moves. On a
    // launch from rest the whole axle breaks away before any difference develops across it, so the
    // engine is first here, and that is the surface talking.
    //
    // The actuator claim is real and is still pinned, at the only level it can be stated cleanly:
    // `the brake channel answers a step immediately and the engine channel takes a tenth of a
    // second`, below.

    // It braked the *slippery* side, which is the whole point of a differential lock: torque it
    // cannot use goes back across the axle to the side that can.
    CAPTURE(run.peakTractionBrakeWheel);
    REQUIRE(run.peakTractionBrakeWheel == frontRight);

    // The brake channel is capped well below full braking, because it is a transient device: holding
    // a driven wheel against first gear would put tens of kilowatts into one disc.
    const auto ceiling = setup->corners[frontLeft].brakeTorque * 0.34;

    CAPTURE(ceiling);
    REQUIRE(run.peakTractionBrake <= ceiling);
}

TEST_CASE("the traction modes are different cars to drive", "[assists][traction][launch]")
{
    // **Criterion 9.** Sport allows more slip than full. On dry that makes it *slower*, which is
    // correct rather than a defect: slip past the tyre's peak is force thrown away, and a sport mode
    // is for rotating the car rather than for the stopwatch.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto off = launch(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), 20.0);
    const auto sport = launch(setup.value(), world.value(), withTraction(setup.value(), TractionMode::Sport), 20.0);
    const auto full = launch(setup.value(), world.value(), withTraction(setup.value(), TractionMode::Full), 20.0);

    REQUIRE(off.toHundred > 0.0);
    REQUIRE(sport.toHundred > 0.0);
    REQUIRE(full.toHundred > 0.0);

    CAPTURE(off.toHundred, sport.toHundred, full.toHundred);
    CAPTURE(off.meanDrivenSlip, sport.meanDrivenSlip, full.meanDrivenSlip);

    // Ordered, and measurably: each mode is a tenth apart or more, which is a difference a driver
    // meets rather than one only a stopwatch does.
    REQUIRE(full.toHundred < sport.toHundred);
    REQUIRE(sport.toHundred < off.toHundred);
    REQUIRE(sport.toHundred - full.toHundred > 0.05);

    // And the reason is the slip each one holds, which is what the modes actually state.
    REQUIRE(full.meanDrivenSlip < sport.meanDrivenSlip);
    REQUIRE(sport.meanDrivenSlip < off.meanDrivenSlip);
}

TEST_CASE("the cornering brake acts on the inside front wheel and only while cornering", "[assists][traction][xds]")
{
    // XDS, and it is neither traction control nor the differential: it is triggered by *cornering*
    // and it brakes the inside front to put torque back across an open diff. Driven here off
    // synthetic sensor readings so that each condition can be stated one at a time — the vehicle-level
    // case below shows it changes the car.
    //
    // The turn's kinematics come from the undriven rear pair. A left turn puts the right-hand wheels
    // on the longer path, so they read faster.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto cornering = CorneringBrakeSetup{};
    cornering.enabled = true;

    auto peaks = std::array<double, wheelCount>{};
    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        peaks[index] = setup->corners[index].brakeTorque;
    }

    const auto reference = ReferenceSpeedSetup{};

    // 20 m/s through a turn that makes the right side 0.4 m/s faster than the left.
    const auto meanSpeed = 20.0;
    const auto across = 0.20;

    auto turning = WheelSpeedReadings{};
    turning[rearLeft] = reading(meanSpeed - across);
    turning[rearRight] = reading(meanSpeed + across);

    const auto yaw = 2.0 * across / cornering.rearTrack;
    const auto halfFront = 0.5 * cornering.frontTrack;

    SECTION("a car cornering with both fronts on their kinematics is left alone")
    {
        turning[frontLeft] = reading(meanSpeed - yaw * halfFront);
        turning[frontRight] = reading(meanSpeed + yaw * halfFront);

        auto state = CorneringBrakeState{};
        advanceCorneringBrake(cornering, state, turning, reference, peaks, 5.0, tick);

        CAPTURE(state.brakeFraction[frontLeft], state.brakeFraction[frontRight]);

        REQUIRE_FALSE(state.active);
        REQUIRE(state.brakeFraction[frontLeft] == 0.0);
        REQUIRE(state.brakeFraction[frontRight] == 0.0);
    }

    SECTION("the inside front running away is braked, and only it")
    {
        // Left turn, so the left front is the inside one. Give it a metre a second it has not earned.
        turning[frontLeft] = reading(meanSpeed - yaw * halfFront + 1.0);
        turning[frontRight] = reading(meanSpeed + yaw * halfFront);

        auto state = CorneringBrakeState{};
        advanceCorneringBrake(cornering, state, turning, reference, peaks, 5.0, tick);

        CAPTURE(state.brakeFraction[frontLeft], state.brakeFraction[frontRight]);

        REQUIRE(state.active);
        REQUIRE(state.brakeFraction[frontLeft] > 0.0);
        REQUIRE(state.brakeFraction[frontRight] == 0.0);
        REQUIRE(state.brakeFraction[rearLeft] == 0.0);
        REQUIRE(state.brakeFraction[rearRight] == 0.0);

        // Slight, which is how Volkswagen describe it: a seventh of full braking on one front wheel.
        REQUIRE(state.brakeFraction[frontLeft] <= cornering.ceiling);
    }

    SECTION("the outside front running away is not its business")
    {
        // That is wheelspin under power across the whole axle, which belongs to traction control.
        turning[frontLeft] = reading(meanSpeed - yaw * halfFront);
        turning[frontRight] = reading(meanSpeed + yaw * halfFront + 1.0);

        auto state = CorneringBrakeState{};
        advanceCorneringBrake(cornering, state, turning, reference, peaks, 5.0, tick);

        REQUIRE_FALSE(state.active);
    }

    SECTION("and none of it happens in a straight line")
    {
        // The same inside-wheel excess, with the accelerometer saying the car is not turning.
        turning[frontLeft] = reading(meanSpeed - yaw * halfFront + 1.0);
        turning[frontRight] = reading(meanSpeed + yaw * halfFront);

        auto state = CorneringBrakeState{};
        advanceCorneringBrake(cornering, state, turning, reference, peaks, 0.5, tick);

        CAPTURE(cornering.onsetLateralAcceleration);

        REQUIRE_FALSE(state.active);
        REQUIRE(state.brakeFraction[frontLeft] == 0.0);
    }

    SECTION("a right turn is the mirror of a left one")
    {
        auto rightward = WheelSpeedReadings{};
        rightward[rearLeft] = reading(meanSpeed + across);
        rightward[rearRight] = reading(meanSpeed - across);
        rightward[frontLeft] = reading(meanSpeed + yaw * halfFront);
        rightward[frontRight] = reading(meanSpeed - yaw * halfFront + 1.0);

        auto state = CorneringBrakeState{};
        advanceCorneringBrake(cornering, state, rightward, reference, peaks, -5.0, tick);

        REQUIRE(state.active);
        REQUIRE(state.brakeFraction[frontRight] > 0.0);
        REQUIRE(state.brakeFraction[frontLeft] == 0.0);
    }
}

TEST_CASE("the cornering brake changes what the car does on corner exit", "[assists][traction][xds]")
{
    // The vehicle-level half. A front-drive car accelerating out of a corner on an open differential
    // sends torque to the unloaded inside front; braking it puts that torque back on the outside
    // wheel, where there is load to use it.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    // A damp surface rather than dry: this is the case the system is for — a front-drive car
    // putting power down on corner exit with more torque than the inside wheel can hold — and on full
    // grip the same manoeuvre generates enough lateral acceleration to lift a wheel, which takes the
    // rear pair's kinematics away from the very controller under test.
    const auto world = PhysicsWorld::create(gripPlate(0.6));
    REQUIRE(world.has_value());

    const auto driveline = golfGtiMk7Driveline();
    const auto inertias = wheelInertias(setup.value());

    struct ExitResult
    {
        double insideFrontSlip = 0.0;
        double outsideFrontForce = 0.0;
        double corneringTorque = 0.0;
        double yaw = 0.0;
        double speed = 0.0;
        bool grounded = true;
    };

    const auto exit = [&](const bool enabled)
    {
        auto assists = golfGtiMk7Assists(setup.value());
        assists.cornering.enabled = enabled;

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), 15.0);

        auto drivelineState = DrivelineState{};
        startEngine(driveline, drivelineState);
        drivelineState.gear = 2;

        auto assistState = AssistState{};
        auto lastStep = VehicleStep{};
        auto road = std::array<double, cornerCount>{};
        auto result = ExitResult{};

        auto samples = 0;

        for (auto step = 0; step < 360 * 5; step++)
        {
            auto sensors = AssistSensors{};
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
            }
            sensors.yawRate = lastStep.telemetry.yawRate;
            sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

            const auto command =
                updateAssists(assists, assistState, sensors, {.brake = 0.0, .throttle = 0.7}, noBrakePressure, tick);

            auto input = VehicleInput{};
            input.throttle = command.throttleScale;
            input.steering = 0.30;
            input.gear = 2;

            const auto speeds =
                std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};

            const auto torques = stepDriveline(driveline, drivelineState, speeds, inertias, road, input, tick);
            REQUIRE(torques.has_value());

            const auto stepped =
                stepVehicle(setup.value(), state, input, torques->wheel, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
            road = roadTorques(lastStep);

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                result.grounded = result.grounded && lastStep.telemetry.wheels[index].inContact;
            }

            // Steering to the right on this car's convention turns it one way; the inside wheel is
            // whichever front is on the shorter path, taken from the rear pair as the ECU does.
            const auto inside =
                std::abs(state.corners[rearLeft].wheelSpeed) < std::abs(state.corners[rearRight].wheelSpeed)
                    ? frontLeft
                    : frontRight;
            const auto outside = inside == frontLeft ? frontRight : frontLeft;

            // Sampled over the second half only, once the car has taken a set.
            if (step > 360 * 2)
            {
                result.insideFrontSlip += std::abs(lastStep.telemetry.wheels[inside].slipRatio);
                result.outsideFrontForce += std::abs(lastStep.telemetry.wheels[outside].forceLongitudinal);
                result.corneringTorque += command.channels.corneringBrakeTorque[inside];
                samples++;
            }
        }

        const auto divisor = samples > 0 ? static_cast<double>(samples) : 1.0;
        result.insideFrontSlip /= divisor;
        result.outsideFrontForce /= divisor;
        result.corneringTorque /= divisor;
        result.yaw = lastStep.telemetry.yaw;
        result.speed = glm::length(state.chassis.linearVelocity);

        return result;
    };

    const auto plain = exit(false);
    const auto assisted = exit(true);

    // **All four wheels have to stay down for this to mean anything**, and not merely because a car
    // on three wheels is a different car: the cornering brake reads the turn's kinematics off the
    // undriven rear pair, so a rear wheel in the air is a sensor that has stopped reporting the road.
    // The steering and throttle here are what keeps them down — 0.5 of lock at 0.9 throttle lifts one.
    REQUIRE(plain.grounded);
    REQUIRE(assisted.grounded);

    CAPTURE(plain.insideFrontSlip, assisted.insideFrontSlip);
    CAPTURE(plain.outsideFrontForce, assisted.outsideFrontForce);
    CAPTURE(plain.corneringTorque, assisted.corneringTorque, plain.speed, assisted.speed);

    // It is doing nothing at all when it is switched off — the channel is named separately precisely
    // so this is checkable rather than inferred.
    REQUIRE(plain.corneringTorque == 0.0);
    REQUIRE(assisted.corneringTorque > 0.0);

    // And what it does is take slip off the inside front and put force on the outside one.
    REQUIRE(assisted.insideFrontSlip < plain.insideFrontSlip);
    REQUIRE(assisted.outsideFrontForce > plain.outsideFrontForce);
}

TEST_CASE("the differential lock acts on the axle's own difference and on nothing else", "[assists][traction][eds]")
{
    // **The reformed brake channel, one condition at a time.** It is an electronic differential
    // lock: it brakes whichever driven wheel is running away *from its partner*, so that an open
    // diff passes the torque to the side that can use it. Everything below follows from that being
    // the signal, and none of it was true of the channel this replaced on 2026-08-23, which keyed on
    // each wheel's slip against the estimated road speed.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto traction = TractionSetup{};
    traction.mode = TractionMode::Full;

    auto peaks = std::array<double, wheelCount>{};
    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        peaks[index] = setup->corners[index].brakeTorque;
    }

    const auto reference = ReferenceSpeedSetup{};

    // 15 m/s, which is inside the channel's speed limit so that the limit is not what is being
    // measured in the cases that are not about it.
    const auto meanSpeed = 15.0;

    auto wheels = WheelSpeedReadings{};
    wheels[rearLeft] = reading(meanSpeed);
    wheels[rearRight] = reading(meanSpeed);

    SECTION("both driven wheels equally slow is not wheelspin, whatever the reference speed says")
    {
        // **The lock-up recovery, as the brake channel sees it.** Both fronts crawling at a fifth of
        // road speed after a lock, and a reference speed that has collapsed to nothing — which is
        // exactly the state that produced the defect Dominic found from the seat. The old channel
        // saturated here. This one has no opinion, because left and right agree.
        wheels[frontLeft] = reading(3.0);
        wheels[frontRight] = reading(3.0);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, 0.01, true, 0.0, 0.5, tick);

        CAPTURE(state.brakeFraction[frontLeft], state.brakeFraction[frontRight]);

        REQUIRE_FALSE(state.brakeActive);
        REQUIRE(state.brakeFraction[frontLeft] == 0.0);
        REQUIRE(state.brakeFraction[frontRight] == 0.0);
    }

    SECTION("both driven wheels spinning together is the engine's job and not the brakes'")
    {
        // Nothing to redistribute: braking either wheel takes drive off the axle instead of moving
        // it across, and braking both is two discs absorbing engine power to no purpose.
        wheels[frontLeft] = reading(meanSpeed * 1.5);
        wheels[frontRight] = reading(meanSpeed * 1.5);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, meanSpeed, true, 0.0, 1.0, tick);

        REQUIRE_FALSE(state.brakeActive);

        // But the engine channel has very much noticed.
        REQUIRE(state.engineReduction > 0.0);
    }

    SECTION("one driven wheel running away is braked, and only it")
    {
        wheels[frontLeft] = reading(meanSpeed);
        wheels[frontRight] = reading(meanSpeed + 1.0);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, meanSpeed, true, 0.0, 1.0, tick);

        CAPTURE(state.brakeFraction[frontLeft], state.brakeFraction[frontRight]);

        REQUIRE(state.brakeActive);
        REQUIRE(state.brakeFraction[frontLeft] == 0.0);
        REQUIRE(state.brakeFraction[frontRight] > 0.0);

        // A metre a second of departure, less the deadband, against a gain sized to reach the ceiling
        // at one metre a second: 1221 * 0.9 / 3665.5 = 0.300.
        REQUIRE(state.brakeFraction[frontRight] == Catch::Approx(0.300).margin(0.005));
    }

    SECTION("and the mirror of it, so no side is special")
    {
        wheels[frontLeft] = reading(meanSpeed + 1.0);
        wheels[frontRight] = reading(meanSpeed);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, meanSpeed, true, 0.0, 1.0, tick);

        REQUIRE(state.brakeFraction[frontRight] == 0.0);
        REQUIRE(state.brakeFraction[frontLeft] == Catch::Approx(0.300).margin(0.005));
    }

    SECTION("a corner is not wheelspin, however different the two driven wheels read")
    {
        // A turn tight enough to put 0.4 m/s across the rear axle. The fronts read the difference
        // the kinematics demand of them and not a metre a second more, so nothing is running away —
        // and a lock that did not take the corner out would brake the outside wheel all the way
        // round every bend.
        const auto across = 0.20;
        wheels[rearLeft] = reading(meanSpeed - across);
        wheels[rearRight] = reading(meanSpeed + across);

        const auto yaw = 2.0 * across / traction.rearTrack;
        const auto halfFront = 0.5 * traction.frontTrack;

        wheels[frontLeft] = reading(meanSpeed - yaw * halfFront);
        wheels[frontRight] = reading(meanSpeed + yaw * halfFront);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, meanSpeed, true, 0.0, 1.0, tick);

        CAPTURE(yaw, state.brakeFraction[frontLeft], state.brakeFraction[frontRight]);

        REQUIRE_FALSE(state.brakeActive);
    }

    SECTION("above the speed limit the brakes hand the whole job to the engine")
    {
        // It is a friction device with the engine on the other side of it, so everything it does
        // becomes heat in one disc. 30 m/s is 108 km/h, well past the 80 the fade ends at.
        // 5 m/s over, which is 17% of slip against a 10% target — comfortably enough for the engine
        // channel to want to act, so that what is being measured is the brake channel declining.
        const auto fast = 30.0;
        wheels[rearLeft] = reading(fast);
        wheels[rearRight] = reading(fast);
        wheels[frontLeft] = reading(fast);
        wheels[frontRight] = reading(fast + 5.0);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, fast, true, 0.0, 1.0, tick);

        REQUIRE_FALSE(state.brakeActive);
        REQUIRE(state.engineReduction > 0.0);
    }
}

TEST_CASE("the brake channel answers a step immediately and the engine channel takes a tenth of a second",
          "[assists][traction][eds]")
{
    // **The division of labour, stated where it can be stated cleanly.** This used to be inferred
    // from which channel intervened first on a launch, which stopped being a fair reading once the
    // two channels were given different signals to fire on. What the design actually claims is about
    // the *actuators*: a hydraulic valve moves in a millisecond and a drive-by-wire throttle body
    // takes its full-travel time, so the brake catches a departing wheel and the engine is what
    // sustains the correction.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto traction = TractionSetup{};
    traction.mode = TractionMode::Full;

    auto peaks = std::array<double, wheelCount>{};
    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        peaks[index] = setup->corners[index].brakeTorque;
    }

    const auto reference = ReferenceSpeedSetup{};
    const auto meanSpeed = 15.0;

    // One wheel departs, hard, in a single control step. Both conditions are present at once — the
    // axle is slipping *and* the two sides disagree — so neither channel is waiting on a surface.
    auto wheels = WheelSpeedReadings{};
    wheels[rearLeft] = reading(meanSpeed);
    wheels[rearRight] = reading(meanSpeed);
    wheels[frontLeft] = reading(meanSpeed);
    wheels[frontRight] = reading(meanSpeed + 3.0);

    auto state = TractionState{};
    advanceTractionControl(traction, state, wheels, reference, peaks, meanSpeed, true, 0.0, 1.0, tick);

    CAPTURE(state.brakeFraction[frontRight], state.engineReduction);

    // The valve is already at its ceiling.
    REQUIRE(state.brakeFraction[frontRight] == Catch::Approx(traction.brakeCeiling).margin(1e-6));

    // The throttle body has barely moved: one 1/360 s step against a 100 ms closing time is under
    // 3% of the way, and that is the whole argument for having a brake channel at all.
    REQUIRE(state.engineReduction < 0.05);

    // Held there, it goes where a first-order lag goes. The demand is the slip error times the
    // engine gain: a wheel 3 m/s over a 15 m/s reference is 0.20 of slip, less the 0.10 target, times
    // a gain of 4, which is 0.40. After one time constant a lag is 63% of the way there.
    const auto demand = std::clamp(traction.engineGain * (3.0 / meanSpeed - traction.fullTargetSlip), 0.0, 1.0);

    for (auto step = 0; step < 36; step++)
    {
        advanceTractionControl(traction, state, wheels, reference, peaks, meanSpeed, true, 0.0, 1.0, tick);
    }

    CAPTURE(demand, state.engineReduction);

    REQUIRE(demand == Catch::Approx(0.40).margin(1e-9));
    REQUIRE(state.engineReduction == Catch::Approx(0.632 * demand).epsilon(0.05));
}

TEST_CASE("the differential lock withdraws when the wheels have stopped meaning anything", "[assists][traction][eds]")
{
    // **Found in the seat trace `traces/rack-20260823-noabs-tcsport-eds.csv`**, on the lap that
    // accepted the reform: the car lands from a crest at 126 km/h with all four wheels stopped in
    // the air, every reading is then a wheel spinning back up at its own rate, and the difference
    // across the axle is noise. 3.3% of that lap's diff-lock interventions were in that state.
    //
    // Two things stop it, and they are different things. The estimator's `coasting` channel says the
    // reference is being carried by its own rate limit rather than by a wheel — a plausibility check,
    // which is what its own comment says it is for. And the speed gate reads the *estimate* rather
    // than the wheels, because the estimate is bounded at 1.3 g where a raw reading goes to zero in
    // a tick.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto traction = TractionSetup{};
    traction.mode = TractionMode::Full;

    auto peaks = std::array<double, wheelCount>{};
    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        peaks[index] = setup->corners[index].brakeTorque;
    }

    const auto reference = ReferenceSpeedSetup{};

    // The landing, as the sensors report it: the car is doing 35 m/s and every wheel is somewhere
    // between stopped and half of it, all four at different rates.
    auto wheels = WheelSpeedReadings{};
    wheels[rearLeft] = reading(7.5);
    wheels[rearRight] = reading(10.9);
    wheels[frontLeft] = reading(17.6);
    wheels[frontRight] = reading(20.7);

    SECTION("a reference carried by its own limiter suspends the channel")
    {
        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, 35.0, true, 0.5, 1.0, tick);

        CAPTURE(state.brakeFraction[frontLeft], state.brakeFraction[frontRight]);
        REQUIRE_FALSE(state.brakeActive);
    }

    SECTION("and even believing the wheels, the speed gate is read off the estimate and holds")
    {
        // Coasting under the hold time, so the plausibility check is not what is being measured —
        // the car is still doing 35 m/s and 35 m/s is past where this channel is allowed to act.
        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, 35.0, true, 0.0, 1.0, tick);

        REQUIRE_FALSE(state.brakeActive);
    }

    SECTION("but a genuine low-speed departure with a healthy reference still gets braked")
    {
        // The same shape of numbers at a speed the channel owns, with the reference tracking. It
        // must not have become deaf.
        wheels[rearLeft] = reading(15.0);
        wheels[rearRight] = reading(15.0);
        wheels[frontLeft] = reading(15.0);
        wheels[frontRight] = reading(16.0);

        auto state = TractionState{};
        advanceTractionControl(traction, state, wheels, reference, peaks, 15.0, true, 0.0, 1.0, tick);

        REQUIRE(state.brakeActive);
        REQUIRE(state.brakeFraction[frontRight] > 0.0);
    }
}
