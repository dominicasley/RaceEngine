#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::sampleWheelSensors;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::ToneRing;
using raceengine::TractionMode;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::WheelSensorStates;

// Hidden probes for the ABS and traction-control work. Nothing here asserts a threshold — these
// print, and what they print is what the assertions in `AntilockBrakingTests` were written against.
// Every tag begins with a dot, so Catch leaves them out of --list-tests and ctest never registers
// them: run by hand with `./EngineTests "[.assist-probe]"`.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto gravity = 9.80665;
constexpr auto degrees = 57.29577951308232;

// **The plate runs z from 0 to its length and x from -width/2 to +width/2**, which is not symmetric
// and cost a whole diagnostic pass to notice: started at z = -210 on a 600 m plate the car is not on
// the ground at all, falls 76 m during the settle, and every stop it is then asked for reads as a
// car with no brakes. `stop()` asserts it is standing on something before it quotes a number.
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
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

// A flat plate whose grip is whatever is asked for, and different on each side of the centreline
// when the two arguments differ. The generator emits one material per surface kind and indexes
// triangles into it; both are rewritten here rather than adding a feature to the generator, which is
// a physics file and out of scope for this work.
SurfaceMesh gripPlate(const double leftGrip, const double rightGrip)
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

        mesh->surfaces[triangle] = centroid.x < 0.0 ? std::uint32_t{0} : std::uint32_t{1};
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

struct StopResult
{
    double distance = 0.0;
    double time = 0.0;
    double lateral = 0.0;
    double yaw = 0.0;
    double deceleration = 0.0;
    std::array<double, cornerCount> lowestWheelSpeed{};
    std::array<std::uint32_t, cornerCount> cycles{};
    double lowestLoad = 0.0;
    bool stopped = false;
    bool onPlate = true;
};

// One stop from `entry` to a standstill with the pedal held at `pedal`. Neutral: nothing drives the
// wheels, so what is measured is the brake system and the tyre and nothing else.
StopResult stop(const VehicleSetup& setup, const PhysicsWorld& world, AssistSetup assists, const double entry,
                const double pedal)
{
    auto state = VehicleState{};
    settle(setup, state, world, entry);

    auto assistState = AssistState{};
    auto result = StopResult{};
    result.lowestWheelSpeed.fill(1e9);
    result.lowestLoad = 1e9;

    const auto sense = [&](const raceengine::VehicleStep& previous)
    {
        auto sensors = raceengine::AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = previous.telemetry.yawRate;
        sensors.lateralAcceleration = previous.telemetry.acceleration.x;

        return sensors;
    };

    auto lastStep = raceengine::VehicleStep{};

    // Rolling before the pedal moves, so the tone rings have produced a reading and the reference
    // speed estimator is not asked to start up and brake in the same instant.
    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(lastStep), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    // --- preconditions, before a single number out of this run is worth quoting ---
    REQUIRE(lastStep.telemetry.wheels[0].inContact);
    REQUIRE(lastStep.telemetry.wheels[1].inContact);
    REQUIRE(lastStep.telemetry.wheels[2].inContact);
    REQUIRE(lastStep.telemetry.wheels[3].inContact);
    REQUIRE(std::abs(state.chassis.linearVelocity.z - entry) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - designHeight) < 0.1);

    const auto start = state.chassis.position;
    const auto entrySpeed = state.chassis.linearVelocity.z;

    auto input = VehicleInput{};
    input.brake = pedal;

    for (auto step = 0; step < 360 * 25; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(lastStep), {.brake = pedal, .throttle = 0.0},
                                           brakeCircuitPressures(setup, pedal), tick);

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            result.cycles[index] = command.channels.antilockCycles[index];
            result.lowestWheelSpeed[index] = std::min(result.lowestWheelSpeed[index], state.corners[index].wheelSpeed);
            result.lowestLoad = std::min(result.lowestLoad, lastStep.telemetry.wheels[index].verticalLoad);
        }

        result.time += tick;

        if (std::abs(state.chassis.position.x) > 0.5 * plateWidth - 2.0 || state.chassis.position.z > plateLength - 5.0)
        {
            result.onPlate = false;
        }

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            result.stopped = true;
            break;
        }
    }

    result.distance = state.chassis.position.z - start.z;
    result.lateral = state.chassis.position.x - start.x;
    result.yaw = lastStep.telemetry.yaw;
    result.deceleration = result.time > 0.0 ? entrySpeed / result.time : 0.0;

    return result;
}

void report(const char* label, const StopResult& run)
{
    std::printf("  %-22s %6.2f m  %5.2f s  %5.3f g  lowest omega %7.2f %7.2f %7.2f %7.2f  lateral %+6.3f m  "
                "yaw %+6.2f deg  cycles %u/%u/%u/%u  %s%s\n",
                label, run.distance, run.time, run.deceleration / gravity, run.lowestWheelSpeed[0],
                run.lowestWheelSpeed[1], run.lowestWheelSpeed[2], run.lowestWheelSpeed[3], run.lateral,
                run.yaw * degrees, run.cycles[0], run.cycles[1], run.cycles[2], run.cycles[3],
                run.stopped ? "stopped" : "DID NOT STOP", run.onPlate ? "" : "  OFF THE PLATE");
}

} // namespace

TEST_CASE("what the tone ring reports at speed", "[.assist-probe]")
{
    const auto ring = ToneRing{};
    const auto pitch = 6.283185307179586 / static_cast<double>(ring.teeth);

    std::printf("\n=== tone ring: %u teeth, %.4f rad pitch, %.1f mm of travel per tooth ===\n", ring.teeth, pitch,
                1000.0 * pitch * tyreRadius);

    for (const auto roadSpeed : {100.0, 50.0, 20.0, 10.0, 7.0, 5.0, 3.0, 1.0})
    {
        const auto omega = roadSpeed / 3.6 / tyreRadius;

        auto states = WheelSensorStates{};
        auto speeds = std::array<double, cornerCount>{};
        speeds.fill(omega);

        auto worstAge = 0.0;
        auto reported = 0.0;

        for (auto step = 0; step < 720; step++)
        {
            const auto readings = sampleWheelSensors(ring, states, speeds, tick);
            worstAge = std::max(worstAge, readings[0].age);
            reported = readings[0].speed;
        }

        const auto predicted = static_cast<double>(ring.teeth) * omega / 6.283185307179586 * 2.0;

        std::printf("  %6.1f km/h  omega %7.3f  pulses %5llu (predicted %7.1f)  period %8.5f s  worst age %7.5f s  "
                    "reported %7.3f rad/s (%+.4f%%)\n",
                    roadSpeed, omega, static_cast<unsigned long long>(states[0].pulses), predicted,
                    states[0].measuredPeriod, worstAge, reported, 100.0 * (reported - omega) / omega);
    }
}

TEST_CASE("what this car's brakes can actually do", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = golfGtiMk7Assists(setup.value());

    std::printf("\n=== dry tarmac, brakes only, no electronics ===\n");
    std::printf("  peak brake torque   front %.1f N.m   rear %.1f N.m   total %.1f N.m\n",
                setup->corners[0].brakeTorque, setup->corners[2].brakeTorque,
                2.0 * (setup->corners[0].brakeTorque + setup->corners[2].brakeTorque));

    auto best = 1e9;
    auto bestPedal = 0.0;

    for (const auto pedal : {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
    {
        const auto run = stop(setup.value(), world.value(), assists, hundred, pedal);

        auto label = std::array<char, 32>{};
        std::snprintf(label.data(), label.size(), "pedal %.2f", pedal);
        report(label.data(), run);

        if (run.stopped && run.distance < best)
        {
            best = run.distance;
            bestPedal = pedal;
        }
    }

    std::printf("  best constant-pressure stop: %.2f m at pedal %.2f\n", best, bestPedal);
    std::printf("  published Mk7 GTI Performance 100-0: 34.6-35.1 m (Auto Bild Sportscars)\n");
}

TEST_CASE("what the electronics do to a stop", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    struct Surface
    {
        const char* name;
        double left;
        double right;
    };

    for (const auto surface : {Surface{"dry tarmac 1.00", 1.00, 1.00}, Surface{"low mu 0.35", 0.35, 0.35},
                               Surface{"split mu 1.00 | 0.35", 1.00, 0.35}})
    {
        const auto world = PhysicsWorld::create(gripPlate(surface.left, surface.right));
        REQUIRE(world.has_value());

        std::printf("\n=== %s ===\n", surface.name);

        for (const auto enabled : {false, true})
        {
            auto assists = golfGtiMk7Assists(setup.value());
            assists.antilock.enabled = enabled;

            report(enabled ? "ABS on" : "ABS off", stop(setup.value(), world.value(), assists, hundred, 1.0));
        }
    }
}

TEST_CASE("where the anti-lock system stops being able to help", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    std::printf("\n=== low mu, entry speed swept: where does ABS stop earning its keep ===\n");

    for (const auto entry : {27.78, 20.0, 14.0, 10.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0})
    {
        auto without = golfGtiMk7Assists(setup.value());
        auto with = golfGtiMk7Assists(setup.value());
        with.antilock.enabled = true;

        const auto locked = stop(setup.value(), world.value(), without, entry, 1.0);
        const auto assisted = stop(setup.value(), world.value(), with, entry, 1.0);

        const auto gain = locked.distance > 0.0 ? 100.0 * (locked.distance - assisted.distance) / locked.distance : 0.0;

        std::printf("  entry %5.2f m/s (%5.1f km/h)  locked %7.3f m  ABS %7.3f m  %+6.2f%%  cycles %u/%u/%u  "
                    "lowest omega %6.2f\n",
                    entry, entry * 3.6, locked.distance, assisted.distance, gain, assisted.cycles[0],
                    assisted.cycles[1], assisted.cycles[2], assisted.lowestWheelSpeed[0]);
    }
}

TEST_CASE("what a launch does with traction control", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto driveline = raceengine::golfGtiMk7Driveline();
    const auto inertias = raceengine::wheelInertias(setup.value());

    struct Case
    {
        const char* name;
        double grip;
        TractionMode mode;
    };

    std::printf("\n=== standing start, DSG, launched from an idle in gear on the brakes ===\n");

    for (const auto scenario :
         {Case{"dry, TC off", 1.00, TractionMode::Off}, Case{"dry, TC full", 1.00, TractionMode::Full},
          Case{"dry, TC sport", 1.00, TractionMode::Sport}, Case{"low mu, TC off", 0.35, TractionMode::Off},
          Case{"low mu, TC full", 0.35, TractionMode::Full}, Case{"low mu, TC sport", 0.35, TractionMode::Sport}})
    {
        const auto world = PhysicsWorld::create(gripPlate(scenario.grip, scenario.grip));
        REQUIRE(world.has_value());

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), 0.0);

        auto drivelineState = raceengine::DrivelineState{};
        raceengine::startEngine(driveline, drivelineState);

        auto assists = golfGtiMk7Assists(setup.value());
        assists.traction.mode = scenario.mode;
        auto assistState = AssistState{};

        auto lastStep = raceengine::VehicleStep{};
        auto road = std::array<double, cornerCount>{};

        const auto sense = [&]
        {
            auto sensors = raceengine::AssistSensors{};
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

        // **Idling in gear on the brakes before anyone floors it**, which is the state a car in the
        // game is always launched from. Skipping it launches from a closed clutch and measures a cold
        // start rather than a launch — the fault `LaunchDiagnosticProbe` records having cost a whole
        // controller build.
        {
            auto idling = VehicleInput{};
            idling.brake = 1.0;
            idling.gear = 1;

            for (auto step = 0; step < 360; step++)
            {
                const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                                   brakeCircuitPressures(setup.value(), 1.0), tick);
                const auto torques =
                    raceengine::stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
                REQUIRE(torques.has_value());

                const auto stepped =
                    stepVehicle(setup.value(), state, idling, torques->wheel, world.value(), tick, command.brakes);
                REQUIRE(stepped.has_value());
                lastStep = stepped.value();
                road = raceengine::roadTorques(lastStep);
            }
        }

        // The fixture states what it is launching from before it quotes anything.
        REQUIRE(glm::length(state.chassis.linearVelocity) < 0.05);
        REQUIRE(drivelineState.gear == 1);
        REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
        REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);

        auto gear = 1;
        const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

        auto reachedHundred = -1.0;
        auto reachedFifty = -1.0;
        auto peakReduction = 0.0;
        auto peakTractionBrake = 0.0;
        auto firstBrake = -1.0;
        auto firstEngine = -1.0;
        auto peakSlip = 0.0;

        for (auto step = 1; step <= 360 * 20; step++)
        {
            const auto now = static_cast<double>(step) * tick;

            // `reduction`, not `ratio * finalDrive`: two final drives since 2026-08-24.
            const auto roadSideSpeed =
                std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
            if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
            {
                gear++;
            }

            const auto command =
                updateAssists(assists, assistState, sense(), {.brake = 0.0, .throttle = 1.0}, noBrakePressure, tick);

            auto input = VehicleInput{};
            input.throttle = command.throttleScale;
            input.gear = gear;

            const auto torques =
                raceengine::stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
            REQUIRE(torques.has_value());

            const auto stepped =
                stepVehicle(setup.value(), state, input, torques->wheel, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
            road = raceengine::roadTorques(lastStep);

            peakReduction = std::max(peakReduction, command.channels.engineTorqueReduction);
            peakSlip = std::max(peakSlip, std::abs(lastStep.telemetry.wheels[0].slipRatio));

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                peakTractionBrake = std::max(peakTractionBrake, command.channels.tractionBrakeTorque[index]);
                if (firstBrake < 0.0 && command.channels.tractionBrakeTorque[index] > 1.0)
                {
                    firstBrake = now;
                }
            }

            if (firstEngine < 0.0 && command.channels.engineTorqueReduction > 0.05)
            {
                firstEngine = now;
            }

            const auto speed = state.chassis.linearVelocity.z;
            if (reachedFifty < 0.0 && speed >= 50.0 / 3.6)
            {
                reachedFifty = now;
            }
            if (reachedHundred < 0.0 && speed >= hundred)
            {
                reachedHundred = now;
                break;
            }
        }

        std::printf("  %-18s 0-50 %6.3f s  0-100 %6.3f s  peak cut %5.3f  peak TC brake %7.1f N.m  "
                    "brake at %6.3f s  engine at %6.3f s  peak slip %6.3f\n",
                    scenario.name, reachedFifty, reachedHundred, peakReduction, peakTractionBrake, firstBrake,
                    firstEngine, peakSlip);
    }
}

TEST_CASE("one anti-lock cycle at full resolution", "[.assist-cycle]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), hundred);

    auto assistState = AssistState{};
    auto lastStep = raceengine::VehicleStep{};

    const auto sense = [&]
    {
        auto sensors = raceengine::AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
        return sensors;
    };

    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped =
            stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    std::printf("\n=== dry, full pedal, FRONT channel every tick (0=Passive 1=Hold 2=Dump 3=Recover 4=Reapply)\n");
    std::printf("     t   ph   p.FL   accel.FL   ecuRate   excess   omega.FL   kappa.FL      Fz.FL    trueDecel\n");

    auto input = VehicleInput{};
    input.brake = 1.0;

    for (auto step = 0; step < 1080; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                           brakeCircuitPressures(setup.value(), 1.0), tick);
        const auto stepped =
            stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        if (step % 3 == 0 && step < 180)
        {
            std::printf("  %5.3f  %2u  %5.3f  %+9.2f  %+8.2f  %+8.2f  %9.3f  %+9.4f  %9.1f    %+8.3f\n",
                        static_cast<double>(step) * tick, static_cast<unsigned>(command.channels.antilockPhase[0]),
                        command.channels.pressure[0], command.channels.sensedWheelAcceleration[0],
                        command.channels.referenceAcceleration,
                        command.channels.sensedWheelAcceleration[0] - command.channels.referenceAcceleration,
                        state.corners[0].wheelSpeed, lastStep.telemetry.wheels[0].slipRatio,
                        lastStep.telemetry.wheels[0].verticalLoad, lastStep.telemetry.acceleration.z);
        }
    }
}

TEST_CASE("how much a locked wheel actually gives away on this tyre", "[.assist-probe]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    std::printf("\n=== longitudinal curve: is there anything for ABS to win? ===\n");

    for (const auto grip : {1.00, 0.35})
    {
        for (const auto load : {5700.0, 2900.0, 930.0})
        {
            auto peak = 0.0;
            auto peakSlip = 0.0;

            for (auto step = 1; step <= 400; step++)
            {
                const auto kappa = -static_cast<double>(step) * 0.005;
                const auto force = raceengine::evaluateTyre(setup->corners[0].tyre, load,
                                                            raceengine::TyreSlip{.slipRatio = kappa}, grip);

                if (std::abs(force.longitudinal) > peak)
                {
                    peak = std::abs(force.longitudinal);
                    peakSlip = kappa;
                }
            }

            const auto locked =
                raceengine::evaluateTyre(setup->corners[0].tyre, load, raceengine::TyreSlip{.slipRatio = -1.0}, grip);
            const auto deep =
                raceengine::evaluateTyre(setup->corners[0].tyre, load, raceengine::TyreSlip{.slipRatio = -2.0}, grip);

            std::printf("  grip %.2f  load %6.0f N   peak %7.1f N at kappa %+.3f   locked(-1) %7.1f N (%.1f%% of "
                        "peak)   deep(-2) %7.1f N (%.1f%%)\n",
                        grip, load, peak, peakSlip, std::abs(locked.longitudinal),
                        100.0 * std::abs(locked.longitudinal) / peak, std::abs(deep.longitudinal),
                        100.0 * std::abs(deep.longitudinal) / peak);
        }
    }
}

TEST_CASE("what the low-mu control band is actually worth", "[.assist-probe]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    std::printf("\n=== force against slip, mu 0.35, front load 5000 N ===\n");

    auto peak = 0.0;
    for (auto step = 1; step <= 600; step++)
    {
        const auto force =
            raceengine::evaluateTyre(setup->corners[0].tyre, 5000.0,
                                     raceengine::TyreSlip{.slipRatio = -static_cast<double>(step) * 0.005}, 0.35);
        peak = std::max(peak, std::abs(force.longitudinal));
    }

    for (const auto kappa : {0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.50, 1.00, 2.00})
    {
        const auto force =
            raceengine::evaluateTyre(setup->corners[0].tyre, 5000.0, raceengine::TyreSlip{.slipRatio = -kappa}, 0.35);
        std::printf("  kappa %.2f  Fx %7.1f N  %5.1f%% of peak\n", kappa, std::abs(force.longitudinal),
                    100.0 * std::abs(force.longitudinal) / peak);
    }
}

TEST_CASE("what slip the anti-lock system actually holds on low mu", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    for (const auto enabled : {false, true})
    {
        auto assists = golfGtiMk7Assists(setup.value());
        assists.antilock.enabled = enabled;

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), hundred);

        auto assistState = AssistState{};
        auto lastStep = raceengine::VehicleStep{};

        const auto sense = [&]
        {
            auto sensors = raceengine::AssistSensors{};
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
            }
            sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
            return sensors;
        };

        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        auto input = VehicleInput{};
        input.brake = 1.0;

        auto samples = 0;
        auto meanTrueSlip = 0.0;
        auto meanEstimatedSlip = 0.0;
        auto meanPressure = 0.0;
        auto meanForce = 0.0;
        auto referenceError = 0.0;

        // Sampled from 100 km/h down to 20 km/h only: below that the sensor's own coarseness is the
        // story and it is measured separately.
        for (auto step = 0; step < 360 * 25; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup.value(), 1.0), tick);
            const auto stepped =
                stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            const auto speed = state.chassis.linearVelocity.z;
            if (speed < 20.0 / 3.6)
            {
                break;
            }

            samples++;
            meanTrueSlip += std::abs(lastStep.telemetry.wheels[0].slipRatio);
            meanEstimatedSlip += std::abs(command.channels.estimatedSlip[0]);
            meanPressure += command.channels.pressure[0];
            meanForce += std::abs(lastStep.telemetry.wheels[0].forceLongitudinal);
            referenceError += command.channels.referenceSpeed - speed;
        }

        const auto divisor = samples > 0 ? static_cast<double>(samples) : 1.0;

        std::printf("  ABS %-3s  mean true slip %.4f   mean estimated slip %.4f   mean pressure %.3f   "
                    "mean Fx %7.1f N   mean reference error %+6.3f m/s\n",
                    enabled ? "on" : "off", meanTrueSlip / divisor, meanEstimatedSlip / divisor, meanPressure / divisor,
                    meanForce / divisor, referenceError / divisor);
    }
}

TEST_CASE("does the car still steer under full braking", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    std::printf("\n=== full pedal with a steering step: what the car does with the wheel ===\n");

    for (const auto grip : {1.00, 0.35})
    {
        const auto world = PhysicsWorld::create(gripPlate(grip, grip));
        REQUIRE(world.has_value());

        for (const auto enabled : {false, true})
        {
            auto assists = golfGtiMk7Assists(setup.value());
            assists.antilock.enabled = enabled;

            auto state = VehicleState{};
            settle(setup.value(), state, world.value(), hundred);

            auto assistState = AssistState{};
            auto lastStep = raceengine::VehicleStep{};

            const auto sense = [&]
            {
                auto sensors = raceengine::AssistSensors{};
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
                }
                sensors.yawRate = lastStep.telemetry.yawRate;
                sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
                return sensors;
            };

            for (auto step = 0; step < 180; step++)
            {
                const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
                const auto stepped = stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(),
                                                 tick, command.brakes);
                REQUIRE(stepped.has_value());
                lastStep = stepped.value();
            }

            const auto start = state.chassis.position;

            auto input = VehicleInput{};
            input.brake = 1.0;
            input.steering = 0.35;

            auto peakYawRate = 0.0;
            auto peakLateral = 0.0;
            auto time = 0.0;

            for (auto step = 0; step < 360 * 25; step++)
            {
                const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                                   brakeCircuitPressures(setup.value(), 1.0), tick);
                const auto stepped =
                    stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
                REQUIRE(stepped.has_value());
                lastStep = stepped.value();
                time += tick;

                peakYawRate = std::max(peakYawRate, std::abs(lastStep.telemetry.yawRate));
                peakLateral = std::max(peakLateral, std::abs(lastStep.telemetry.acceleration.x));

                if (state.chassis.linearVelocity.z <= 0.0)
                {
                    break;
                }
            }

            std::printf("  grip %.2f  ABS %-3s  peak yaw rate %6.3f rad/s  peak lateral %5.2f m/s2  "
                        "lateral travel %+7.3f m  distance %6.2f m  %5.2f s\n",
                        grip, enabled ? "on" : "off", peakYawRate, peakLateral, state.chassis.position.x - start.x,
                        state.chassis.position.z - start.z, time);
        }
    }
}

TEST_CASE("what the modulator's own rates are worth", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    struct Surface
    {
        const char* name;
        double grip;
    };

    // **Swept in bar/s since 2026-08-23**, which is the unit the modulator states its gradients in
    // now. They were fractions of full system pressure per second, and deriving this car's brakes
    // from its calipers is what showed that up: the peak moved and every gradient moved with it.
    std::printf("\n=== sensitivity to the hydraulic rates (shipped: dump 1000 bar/s, reapply 300) ===\n");

    for (const auto surface : {Surface{"dry  ", 1.00}, Surface{"lowmu", 0.35}})
    {
        const auto world = PhysicsWorld::create(gripPlate(surface.grip, surface.grip));
        REQUIRE(world.has_value());

        const auto locked = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, 1.0);

        for (const auto dump : {500.0, 1000.0, 2000.0})
        {
            for (const auto reapply : {150.0, 300.0, 600.0, 1000.0})
            {
                auto assists = golfGtiMk7Assists(setup.value());
                assists.antilock.enabled = true;
                assists.antilock.modulator.dumpGradient = dump * 1.0e5;
                assists.antilock.modulator.reapplyGradient = reapply * 1.0e5;

                const auto run = stop(setup.value(), world.value(), assists, hundred, 1.0);
                const auto gain = 100.0 * (locked.distance - run.distance) / locked.distance;

                std::printf("  %s  dump %6.0f bar/s  reapply %6.0f  %7.2f m  (%+6.2f%% vs locked %6.2f m)  "
                            "cycles %u/%u\n",
                            surface.name, dump, reapply, run.distance, gain, locked.distance, run.cycles[0],
                            run.cycles[2]);
            }
        }
    }
}

TEST_CASE("where the low-speed dropout falls on dry tarmac", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    std::printf("\n=== dry, entry speed swept ===\n");

    for (const auto entry : {27.78, 20.0, 14.0, 10.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0})
    {
        auto with = golfGtiMk7Assists(setup.value());
        with.antilock.enabled = true;

        const auto plain = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), entry, 1.0);
        const auto assisted = stop(setup.value(), world.value(), with, entry, 1.0);

        const auto gain = plain.distance > 0.0 ? 100.0 * (plain.distance - assisted.distance) / plain.distance : 0.0;
        const auto cycles = assisted.cycles[0] + assisted.cycles[1] + assisted.cycles[2];
        const auto frequency = assisted.time > 0.0 ? static_cast<double>(assisted.cycles[0]) / assisted.time : 0.0;

        std::printf("  entry %5.2f m/s (%5.1f km/h)  plain %7.3f m  ABS %7.3f m  %+6.2f%%  cycles %u  "
                    "front rate %5.1f Hz\n",
                    entry, entry * 3.6, plain.distance, assisted.distance, gain, cycles, frequency);
    }
}

TEST_CASE("what corrected brake data would be worth", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    // Candidates, as (total N.m, front share). The mod's own is first. Nothing here is committed —
    // this is the measurement that says whether a corrected figure lands on the published distance.
    struct Candidate
    {
        double total;
        double frontShare;
    };

    std::printf("\n=== 100-0 against Auto Bild Sportscars' 34.6 and 35.1 m for a Mk7 GTI Performance ===\n");
    std::printf("  total  front   per-wheel F/R      best pedal stop      ABS stop     lowest omega F / R\n");

    for (const auto candidate : {Candidate{4200.0, 0.75}, Candidate{4800.0, 0.82}, Candidate{5200.0, 0.82},
                                 Candidate{4800.0, 0.75}, Candidate{5600.0, 0.85}})
    {
        auto tuned = setup.value();
        const auto front = candidate.total * candidate.frontShare / 2.0;
        const auto rear = candidate.total * (1.0 - candidate.frontShare) / 2.0;

        tuned.corners[0].brakeTorque = front;
        tuned.corners[1].brakeTorque = front;
        tuned.corners[2].brakeTorque = rear;
        tuned.corners[3].brakeTorque = rear;

        auto assists = golfGtiMk7Assists(tuned);

        // The best a driver can do without cycling the pedal, swept rather than assumed: with more
        // brake than grip the shortest stop is no longer at the floor.
        auto best = 1e9;
        auto bestPedal = 0.0;
        auto bestRun = StopResult{};

        // Finely swept. A coarse sweep flatters whichever candidate happens to optimise at a sampled
        // point — the mod's own optimises exactly at the floor, because it is brake-limited, so it was
        // the only one being measured at its true optimum.
        for (auto step = 10; step <= 20; step++)
        {
            const auto pedal = 0.05 * static_cast<double>(step);
            const auto run = stop(tuned, world.value(), assists, hundred, pedal);
            if (run.stopped && run.distance < best)
            {
                best = run.distance;
                bestPedal = pedal;
                bestRun = run;
            }
        }

        auto withAbs = assists;
        withAbs.antilock.enabled = true;
        const auto assisted = stop(tuned, world.value(), withAbs, hundred, 1.0);

        std::printf("  %6.0f  %.2f  %7.1f/%5.1f  %6.2f m at %.2f  %6.2f m  %8.2f / %6.2f\n", candidate.total,
                    candidate.frontShare, front, rear, best, bestPedal, assisted.distance, bestRun.lowestWheelSpeed[0],
                    bestRun.lowestWheelSpeed[2]);
    }
}

TEST_CASE("does the longitudinal peak have to answer to two references at once", "[.assist-probe]")
{
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());
    const auto driveline = raceengine::golfGtiMk7Driveline();
    const auto inertias = raceengine::wheelInertias(base.value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    std::printf("\n=== mu_x swept against BOTH published references ===\n");
    std::printf("  0-100 published 6.4-6.7 s (DSG, no launch control) | 100-0 published 34.6-35.1 m\n");
    std::printf("  mu_x   DX_REF scale   0-100      best 100-0   (brakes at 9000 N.m so they cannot cap it)\n");

    for (const auto muX : {1.131, 1.170, 1.210, 1.250})
    {
        auto setup = base.value();
        for (auto& corner : setup.corners)
        {
            corner.tyre.longitudinalPeak = muX;
        }

        // **Brakes taken properly out of the way.** 5200 N.m was not enough: at mu_x 1.25 the front
        // axle needs 3996 N.m to lock and 5200 at a 0.75 share gives it 3900, so the top of this
        // sweep was still measuring brakes.ini rather than the tyre and read as a plateau.
        setup.corners[0].brakeTorque = setup.corners[1].brakeTorque = 9000.0 * 0.75 / 2.0;
        setup.corners[2].brakeTorque = setup.corners[3].brakeTorque = 9000.0 * 0.25 / 2.0;

        const auto assists = golfGtiMk7Assists(setup);

        // **Swept from a tenth of the pedal, not from half.** With the brakes deliberately oversized
        // the useful pedal range shrinks — at 9000 N.m the car locks at about half travel, so a sweep
        // starting there is entirely past the optimum and reports a stop several metres long.
        auto best = 1e9;
        auto bestPedal = 0.0;
        for (auto step = 2; step <= 20; step++)
        {
            const auto pedal = 0.05 * static_cast<double>(step);
            const auto run = stop(setup, world.value(), assists, hundred, pedal);
            if (run.stopped && run.distance < best)
            {
                best = run.distance;
                bestPedal = pedal;
            }
        }

        // The launch, from an idle in gear on the brakes.
        auto state = VehicleState{};
        settle(setup, state, world.value(), 0.0);
        auto drivelineState = raceengine::DrivelineState{};
        raceengine::startEngine(driveline, drivelineState);
        auto road = std::array<double, cornerCount>{};
        auto lastStep = raceengine::VehicleStep{};

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
                const auto t =
                    raceengine::stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
                REQUIRE(t.has_value());
                const auto stepped = stepVehicle(setup, state, idling, t->wheel, world.value(), tick);
                REQUIRE(stepped.has_value());
                lastStep = stepped.value();
                road = raceengine::roadTorques(lastStep);
            }
        }

        auto gear = 1;
        auto hundredAt = -1.0;
        const auto upshift = driveline.engine.limiterSpeed * 0.93;

        for (auto step = 1; step <= 360 * 15; step++)
        {
            const auto roadSide =
                std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
            if (roadSide > upshift && gear < driveline.gearbox.topGear())
            {
                gear++;
            }

            auto input = VehicleInput{};
            input.throttle = 1.0;
            input.gear = gear;

            const auto t = raceengine::stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
            REQUIRE(t.has_value());
            const auto stepped = stepVehicle(setup, state, input, t->wheel, world.value(), tick);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
            road = raceengine::roadTorques(lastStep);

            if (state.chassis.linearVelocity.z >= hundred)
            {
                hundredAt = static_cast<double>(step) * tick;
                break;
            }
        }

        std::printf("  %.3f      %.3f      %6.3f s    %6.2f m at pedal %.2f\n", muX, muX / 1.30, hundredAt, best,
                    bestPedal);
    }
}
