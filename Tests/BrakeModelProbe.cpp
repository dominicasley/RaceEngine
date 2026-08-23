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

using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::boosterAssistLimit;
using raceengine::boosterRunoutPedal;
using raceengine::brakeCircuitPressures;
using raceengine::BrakeHardware;
using raceengine::brakeLinePressure;
using raceengine::bringUpJolt;
using raceengine::computeMassProperties;
using raceengine::cornerCount;
using raceengine::effectiveRadius;
using raceengine::Feature;
using raceengine::frontBrakeShare;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::golfMk7FrontBrake;
using raceengine::golfMk7Hydraulics;
using raceengine::golfMk7RearBrake;
using raceengine::golfMk7RearProportioningValve;
using raceengine::masterCylinderArea;
using raceengine::ModulatorPhase;
using raceengine::noDriveTorque;
using raceengine::peakBrakeTorque;
using raceengine::PhysicsWorld;
using raceengine::pistonArea;
using raceengine::ProportioningValve;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::torquePerPressure;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;

// The brake model, printed. `./EngineTests "[.brake-model]"`.
//
// Hidden behind a dotted tag, like every other probe here: what it produces is a table to read rather
// than a bound to hold, and the bounds that *did* come out of it are in `BrakeHardwareTests.cpp`.
//
// Written 2026-08-23 with `docs/brake-model-brief.md`. Its job is to make the derivation arguable —
// every part, what it multiplies out to, and what the car then does — rather than leaving a reader to
// take three functions on trust.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto gravity = 9.80665;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto bar = 1.0e5;

// **The plate runs z from 0 to its length and x from -width/2 to +width/2**, which is the trap
// recorded in the brief: a fixture that assumes it is centred starts the car in mid-air and every
// stop it then reports reads as a car with no brakes.
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

[[nodiscard]] SurfaceMesh gripPlate(const double grip)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(1);
    mesh->materials[0].gripMultiplier = grip;
    mesh->materials[0].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        mesh->surfaces[triangle] = std::uint32_t{0};
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
    double deceleration = 0.0;
    double finalYaw = 0.0;
    double lateralTravel = 0.0;
    std::array<double, cornerCount> lowestWheelSpeed{};
    // The lightest each corner ever got, and when it first read nothing at all. `grounded` is one
    // bit over four wheels and thirty seconds, so a stop that lifts a wheel for a single tick and a
    // stop that lifts one for a second are the same bit — these two say which.
    std::array<double, cornerCount> lowestLoad{};
    std::array<double, cornerCount> firstLiftTime{};
    bool stopped = false;
    bool grounded = true;
    bool onPlate = true;
};

[[nodiscard]] StopResult stop(const VehicleSetup& setup, const PhysicsWorld& world, AssistSetup assists,
                              const double pedal)
{
    auto state = VehicleState{};
    settle(setup, state, world, hundred);

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    auto result = StopResult{};
    result.lowestWheelSpeed.fill(1e9);
    result.lowestLoad.fill(1e9);
    result.firstLiftTime.fill(-1.0);

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

    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    // Preconditions, before any number below is worth quoting: rolling at the entry speed, at ride
    // height, straight, and on all four wheels.
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
    }
    REQUIRE(std::abs(state.chassis.linearVelocity.z - hundred) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - designHeight) < 0.1);
    REQUIRE(std::abs(state.chassis.position.x) < 0.05);

    const auto start = state.chassis.position;
    const auto entrySpeed = state.chassis.linearVelocity.z;

    auto input = VehicleInput{};
    input.brake = pedal;

    for (auto step = 0; step < 360 * 30; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = pedal, .throttle = 0.0},
                                           brakeCircuitPressures(setup, pedal), tick);
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        result.time += tick;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            // **Only while the car is still going somewhere.** Every wheel on a car that has stopped
            // reads zero, so a lowest-wheel-speed taken over the whole stop says "locked" about every
            // pedal position there is — which is what the first version of the lock sweep below
            // reported, uniformly and uselessly, at every candidate valve.
            if (state.chassis.linearVelocity.z > 5.0)
            {
                result.lowestWheelSpeed[index] =
                    std::min(result.lowestWheelSpeed[index], state.corners[index].wheelSpeed);
            }

            result.lowestLoad[index] = std::min(result.lowestLoad[index], lastStep.corners[index].forces.tireVertical);
            if (!lastStep.telemetry.wheels[index].inContact && result.firstLiftTime[index] < 0.0)
            {
                result.firstLiftTime[index] = result.time;
            }

            result.grounded = result.grounded && lastStep.telemetry.wheels[index].inContact;
        }

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
    result.lateralTravel = state.chassis.position.x - start.x;
    result.finalYaw = lastStep.telemetry.yaw;
    result.deceleration = result.time > 0.0 ? entrySpeed / result.time : 0.0;

    return result;
}

} // namespace

TEST_CASE("the brake torque, from the parts that make it", "[.brake-model]")
{
    const auto front = golfMk7FrontBrake();
    const auto rear = golfMk7RearBrake();
    const auto hydraulics = golfMk7Hydraulics();

    std::printf("\n=== the offline half: what each corner makes per pascal ===\n");
    std::printf("        bore   pistons   area      disc    pad    r_eff    mu    faces   N.m/Pa      N.m/bar\n");

    const auto row = [](const char* name, const BrakeHardware& brake)
    {
        std::printf("  %-5s %5.1f mm   %u   %7.1f mm2  %5.0f  %4.1f  %6.1f mm  %.2f    %u    %.6e  %8.2f\n", name,
                    brake.pistonBore * 1000.0, brake.pistons, pistonArea(brake) * 1e6, brake.discDiameter * 1000.0,
                    brake.padRadialHeight * 1000.0, effectiveRadius(brake) * 1000.0, brake.couple.coefficient,
                    brake.frictionFaces, torquePerPressure(brake), torquePerPressure(brake) * bar);
    };

    row("front", front);
    row("rear", rear);

    std::printf("\n  front share falls out at %.4f   (brakes.ini stated FRONT_SHARE=0.75)\n",
                frontBrakeShare(front, rear));

    std::printf("\n=== the online half: what the pedal makes of it ===\n");
    std::printf("  master cylinder %.2f mm bore, %.2f mm2\n", hydraulics.masterCylinderBore * 1000.0,
                masterCylinderArea(hydraulics) * 1e6);
    std::printf("  pedal ratio %.2f, servo gain %.2f, diaphragm %.0f mm at %.0f mbar -> %.0f N of assist\n",
                hydraulics.pedalRatio, hydraulics.boostRatio, hydraulics.boosterDiaphragm * 1000.0,
                hydraulics.boosterVacuum / 100.0, boosterAssistLimit(hydraulics));
    std::printf("  full pedal is %.0f N at the foot; the servo runs out at pedal %.3f (%.1f bar)\n",
                hydraulics.maxPedalForce, boosterRunoutPedal(hydraulics),
                brakeLinePressure(hydraulics, boosterRunoutPedal(hydraulics)) / bar);

    const auto valve = golfMk7RearProportioningValve();
    std::printf("  the rear circuit's proportioning valve: knee %.1f bar, then slope %.2f\n", valve.kneePressure / bar,
                valve.slope);

    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    std::printf("\n   pedal   foot N    master bar   rear bar    front N.m    rear N.m    total N.m   front share\n");
    for (auto step = 0; step <= 10; step++)
    {
        const auto pedal = 0.1 * static_cast<double>(step);
        const auto pressures = brakeCircuitPressures(car.value(), pedal);
        const auto frontTorque = pressures[0] * torquePerPressure(front);
        const auto rearTorque = pressures[2] * torquePerPressure(rear);
        const auto total = frontTorque + rearTorque;

        std::printf("   %5.2f   %6.0f   %10.2f  %9.2f   %10.1f  %10.1f   %10.1f   %11.4f\n", pedal,
                    pedal * hydraulics.maxPedalForce, pressures[0] / bar, pressures[2] / bar, frontTorque, rearTorque,
                    2.0 * total, total > 0.0 ? frontTorque / total : 0.0);
    }

    std::printf("\n  peak: front %.1f N.m a side, rear %.1f, total %.1f N.m\n", car->corners[0].brakeTorque,
                car->corners[2].brakeTorque, 2.0 * (car->corners[0].brakeTorque + car->corners[2].brakeTorque));
    std::printf("  it replaces brakes.ini's 5600 (itself a correction of 4200) and its stated FRONT_SHARE=0.75\n");
    std::printf("  the share is no longer a number: %.4f below the valve's knee, %.4f at a full pedal,\n"
                "  against an ideal that runs 0.647 at 0.3 g to about 0.811 at the limit\n",
                frontBrakeShare(front, rear),
                car->corners[0].brakeTorque / (car->corners[0].brakeTorque + car->corners[2].brakeTorque));

    // The cross-checks. These are what say the derivation is plausible, and none of them is a number
    // anything above was fitted to: a road car locks its fronts somewhere near 50 to 70 bar, and
    // reaches its own limit at 200 to 300 N of pedal effort.
    const auto frontLock = 3468.0 / 2.0 / torquePerPressure(front);

    std::printf("\n=== cross-checks nothing was fitted to ===\n");
    std::printf("  front axle locks at %.1f bar, on a published road-car range of 50-70\n", frontLock / bar);
    // Below the servo's runout, so the map is linear there and the force is a straight ratio. Taking
    // it against the *full pedal* pressure instead reads 237 N, which is wrong by the runout and is
    // the shape of mistake this file exists to make visible.
    const auto runout = boosterRunoutPedal(hydraulics);
    std::printf("  the car reaches its own 0.945 g limit at %.0f N of pedal force, published 200-300 N\n",
                frontLock / brakeLinePressure(hydraulics, runout) * runout * hydraulics.maxPedalForce);
    std::printf("  a third of the rear brake, which is what the handbrake model takes, is %.0f N.m across the\n"
                "  axle -- brakes.ini states HANDBRAKE_TORQUE=1200\n",
                2.0 * car->corners[2].brakeTorque / 3.0);
}

TEST_CASE("what the derived brakes do to a stop", "[.brake-model]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto assists = golfGtiMk7Assists(setup.value());

    const auto sprung = computeMassProperties(setup->sprung);
    REQUIRE(sprung.has_value());
    const auto mass = sprung->mass + setup->unsprungMass();

    std::printf("\n=== dry tarmac, no electronics, 100-0 ===\n");
    std::printf("  peak %.1f N.m total; if grip were unlimited that is %.2f g\n",
                2.0 * (setup->corners[0].brakeTorque + setup->corners[2].brakeTorque),
                2.0 * (setup->corners[0].brakeTorque + setup->corners[2].brakeTorque) / tyreRadius / (mass * gravity));

    // **Swept the whole pedal, finely.** The trap this replaces cost two wrong conclusions on
    // 2026-08-23: a sweep ranged for one brake torque measures every other candidate off its optimum.
    // With the derived brakes the optimum is near a third of the pedal, so a sweep starting at 0.55
    // would never see it.
    std::printf("\n  pedal    stop      time     mean g   lowest omega FL      locked\n");

    auto best = 1e9;
    auto bestPedal = 0.0;

    for (auto step = 2; step <= 20; step++)
    {
        const auto pedal = 0.05 * static_cast<double>(step);
        const auto run = stop(setup.value(), world.value(), assists, pedal);

        const auto locked = run.lowestWheelSpeed[0] < 0.5 || run.lowestWheelSpeed[2] < 0.5;

        std::printf("   %5.2f  %7.2f m  %6.3f s  %6.3f   %10.2f rad/s   %s%s\n", pedal, run.distance, run.time,
                    run.deceleration / gravity, run.lowestWheelSpeed[0], locked ? "yes" : "no",
                    run.grounded ? "" : "   (a wheel left the road)");

        if (run.stopped && run.distance < best)
        {
            best = run.distance;
            bestPedal = pedal;
        }
    }

    std::printf("\n  best constant-pedal stop %.2f m at pedal %.2f\n", best, bestPedal);
    std::printf("  published Mk7 GTI Performance 100-0: 34.6-35.1 m (Auto Bild Sportscars)\n");
}

TEST_CASE("what the anti-lock modulator does with a brake that can lock the wheels", "[.brake-model]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), hundred);

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};

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

    std::printf("\n=== the front left channel through a full-pedal dry stop ===\n");
    std::printf("  every tick for the first 0.6 s, then every tenth\n");
    std::printf("      t      car m/s   wheel m/s    slip   sensed a    phase   pressure   brake N.m   valid\n");

    const auto phaseName = [](const ModulatorPhase phase)
    {
        switch (phase)
        {
        case ModulatorPhase::Passive:
            return "passive";
        case ModulatorPhase::Hold:
            return "hold";
        case ModulatorPhase::Dump:
            return "dump";
        case ModulatorPhase::Recover:
            return "recover";
        case ModulatorPhase::Reapply:
            return "reapply";
        }

        return "?";
    };

    auto elapsed = 0.0;

    for (auto step = 0; step < 360 * 6; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                           brakeCircuitPressures(setup.value(), 1.0), tick);
        const auto stepped =
            stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
        elapsed += tick;

        if (elapsed < 0.6 || step % 10 == 0)
        {
            std::printf("  %6.4f  %8.3f  %9.3f  %6.3f  %9.2f  %8s  %8.4f  %10.1f     %s\n", elapsed,
                        state.chassis.linearVelocity.z, state.corners[0].wheelSpeed * tyreRadius,
                        command.channels.estimatedSlip[0], command.channels.sensedWheelAcceleration[0],
                        phaseName(command.channels.antilockPhase[0]), command.channels.pressure[0],
                        command.brakes.wheels[0], command.channels.referenceValid ? "yes" : "NO");
        }

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            break;
        }
    }

    std::printf("\n  stopped at %.3f s\n", elapsed);
}

TEST_CASE("the two lock pressures the proportioning valve is derived from", "[.brake-model]")
{
    // **This is the measurement `golfMk7RearProportioningValve` is built on**, and the criterion it
    // is built to is one sentence: *the front axle must lock first*. Everything else about a
    // proportioning valve follows from that plus the two pressures below.
    //
    // Swept with the valve removed first — which is the car the calipers alone describe — and then
    // across candidate slopes, reporting the lowest pedal at which each axle stops a wheel. A slope
    // is admissible when the rear's lock pedal is not below the front's.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto tppFront = torquePerPressure(golfMk7FrontBrake());
    const auto tppRear = torquePerPressure(golfMk7RearBrake());

    // Lowest pedal at which the named axle brings a wheel to a stop, and 0 if it never does.
    const auto lockPedal = [&](const VehicleSetup& setup, const std::size_t corner)
    {
        for (auto step = 1; step <= 40; step++)
        {
            const auto pedal = 0.025 * static_cast<double>(step);
            const auto run = stop(setup, world.value(), golfGtiMk7Assists(setup), pedal);

            if (run.lowestWheelSpeed[corner] < 0.5)
            {
                return pedal;
            }
        }

        return 0.0;
    };

    // Fitted with a valve that does nothing, so it is the calipers' own answer. The front is where it
    // is regardless of the rear circuit, so it is measured once.
    auto unvalved = base.value();
    unvalved.rearBrakeValve = ProportioningValve{};
    {
        const auto full = brakeCircuitPressures(unvalved, 1.0);
        unvalved.corners[0].brakeTorque = unvalved.corners[1].brakeTorque = tppFront * full[0];
        unvalved.corners[2].brakeTorque = unvalved.corners[3].brakeTorque = tppRear * full[2];
    }

    const auto frontLock = lockPedal(unvalved, 0);
    const auto rearUnvalved = lockPedal(unvalved, 2);

    std::printf("\n=== dry tarmac, no valve: what the calipers alone do ===\n");
    std::printf("  the front axle locks at pedal %.3f -> %.2f bar\n", frontLock,
                brakeCircuitPressures(unvalved, frontLock)[0] / bar);
    std::printf("  the rear axle locks at  pedal %.3f -> %.2f bar, which is %s\n", rearUnvalved,
                brakeCircuitPressures(unvalved, rearUnvalved)[2] / bar,
                rearUnvalved < frontLock ? "FIRST, and is the whole problem" : "second");

    std::printf("\n=== the valve swept: the most rear brake that still locks the front first ===\n");

    // **On both surfaces, because the criterion is a criterion about the car and not about tarmac.**
    // The ideal distribution moves with grip as well as with deceleration, so a valve fitted against
    // a dry stop alone is a valve that has been checked at one point of the range it has to cover.
    // This is the same complement-testing rule the rest of the suite is written under.
    const auto slippery = PhysicsWorld::create(gripPlate(0.35));
    REQUIRE(slippery.has_value());

    const auto lockPedalOn = [&](const PhysicsWorld& plate, const VehicleSetup& setup, const std::size_t corner)
    {
        for (auto step = 1; step <= 40; step++)
        {
            const auto pedal = 0.025 * static_cast<double>(step);
            const auto run = stop(setup, plate, golfGtiMk7Assists(setup), pedal);

            if (run.lowestWheelSpeed[corner] < 0.5)
            {
                return pedal;
            }
        }

        return 0.0;
    };

    const auto slipperyFront = lockPedalOn(slippery.value(), unvalved, 0);
    std::printf("  on mu 0.35 the front axle locks at pedal %.3f\n", slipperyFront);
    std::printf("\n   knee bar   slope   rear peak N.m    dry: front %.3f  rear    mu 0.35: front %.3f  rear\n",
                frontLock, slipperyFront);

    for (const auto knee : {24.0, 28.0, 32.0})
    {
        for (const auto slope : {0.00, 0.15, 0.30})
        {
            auto setup = base.value();
            setup.rearBrakeValve = ProportioningValve{.kneePressure = knee * bar, .slope = slope};

            // Re-derive the peaks for this valve, exactly as `golfGtiMk7` does, or the rear would keep
            // a peak belonging to a different valve — which is the stale-override trap this project
            // has already paid for twice.
            const auto full = brakeCircuitPressures(setup, 1.0);
            setup.corners[0].brakeTorque = setup.corners[1].brakeTorque = tppFront * full[0];
            setup.corners[2].brakeTorque = setup.corners[3].brakeTorque = tppRear * full[2];

            const auto dryRear = lockPedalOn(world.value(), setup, 2);
            const auto wetRear = lockPedalOn(slippery.value(), setup, 2);

            const auto safe = (dryRear <= 0.0 || dryRear >= frontLock) && (wetRear <= 0.0 || wetRear >= slipperyFront);

            std::printf("   %8.1f   %5.2f   %11.1f    %26.3f    %26.3f   %s\n", knee, slope,
                        setup.corners[2].brakeTorque, dryRear, wetRear, safe ? "ok" : "<- rear first");
        }
    }
}

TEST_CASE("is the derived split or the derived total what moves the car under braking", "[.brake-model]")
{
    // **The decisive experiment for acceptance criterion 3 of the brief.** The derivation changed two
    // things at once — the total went from 5600 to 10688 N.m and the front share from a stated 0.75 to
    // a derived 0.686 — and the two have to be separated before either can be blamed for anything.
    //
    // Held total, swept share; then held share, swept total. Split-mu is where a bad split shows,
    // because the axle that locks first is the axle that decides which way the car ends up pointing.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());
    mesh->materials.resize(2);
    mesh->materials[0].gripMultiplier = 1.0;
    mesh->materials[0].bumpiness = 0.0;
    mesh->materials[1].gripMultiplier = 0.35;
    mesh->materials[1].bumpiness = 0.0;
    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        const auto centroid =
            (mesh->vertices[mesh->indices[triangle * 3 + 0]] + mesh->vertices[mesh->indices[triangle * 3 + 1]] +
             mesh->vertices[mesh->indices[triangle * 3 + 2]]) /
            3.0;
        mesh->surfaces[triangle] = centroid.x < 0.0 ? std::uint32_t{0} : std::uint32_t{1};
    }

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    std::printf("\n=== split-mu 1.00/0.35, full pedal, no electronics ===\n");
    std::printf("  rear/front lock pedals are against the 512 and 1734 N.m a side this car needs\n");
    std::printf("   total    share      stop     final yaw   lateral    rear locks   front locks\n");

    struct Candidate
    {
        double total;
        double share;
    };

    for (const auto candidate : {Candidate{5600.0, 0.75}, Candidate{10688.3, 0.75}, Candidate{10688.3, 0.686},
                                 Candidate{5600.0, 0.686}, Candidate{10688.3, 0.811}})
    {
        auto setup = base.value();
        const auto front = candidate.total * candidate.share / 2.0;
        const auto rear = candidate.total * (1.0 - candidate.share) / 2.0;
        setup.corners[0].brakeTorque = setup.corners[1].brakeTorque = front;
        setup.corners[2].brakeTorque = setup.corners[3].brakeTorque = rear;

        const auto run = stop(setup, world.value(), golfGtiMk7Assists(setup), 1.0);

        std::printf("  %7.0f   %.3f  %8.2f m  %+8.1f deg  %+6.2f m   %7.3f      %7.3f%s\n", candidate.total,
                    candidate.share, run.distance, run.finalYaw * 57.29577951308232, run.lateralTravel, 512.5 / rear,
                    1734.0 / front, run.stopped ? "" : "   (did not stop)");
    }

    std::printf("\n=== the same, with the anti-lock system on ===\n");
    std::printf("   total    share      stop     final yaw   lateral\n");

    // The car exactly as built, first, with nothing overridden. **Split-mu yaw turned out to be
    // sensitive at the tenth of a newton metre** — the rounded 10688.3/0.686 row below lands 0.6 N.m
    // from this one and finishes a hundred degrees away — so a row that is not the shipped car is
    // not evidence about the shipped car.
    {
        auto assists = golfGtiMk7Assists(base.value());
        assists.antilock.enabled = true;

        const auto run = stop(base.value(), world.value(), assists, 1.0);

        std::printf("  as built   0.686  %8.2f m  %+8.1f deg  %+6.2f m%s\n", run.distance,
                    run.finalYaw * 57.29577951308232, run.lateralTravel, run.stopped ? "" : "   (did not stop)");
    }

    for (const auto candidate : {Candidate{5600.0, 0.75}, Candidate{10688.3, 0.75}, Candidate{10688.3, 0.686},
                                 Candidate{5600.0, 0.686}, Candidate{10688.3, 0.811}})
    {
        auto setup = base.value();
        const auto front = candidate.total * candidate.share / 2.0;
        const auto rear = candidate.total * (1.0 - candidate.share) / 2.0;
        setup.corners[0].brakeTorque = setup.corners[1].brakeTorque = front;
        setup.corners[2].brakeTorque = setup.corners[3].brakeTorque = rear;

        auto assists = golfGtiMk7Assists(setup);
        assists.antilock.enabled = true;

        const auto run = stop(setup, world.value(), assists, 1.0);

        std::printf("  %7.0f   %.3f  %8.2f m  %+8.1f deg  %+6.2f m%s\n", candidate.total, candidate.share, run.distance,
                    run.finalYaw * 57.29577951308232, run.lateralTravel, run.stopped ? "" : "   (did not stop)");
    }
}

TEST_CASE("what the two published references demand of this tyre, on the car as it now is", "[.brake-model]")
{
    // **The sweep in `AssistDiagnosticProbe` is stale twice over and this replaces it for braking.**
    // That one overrides the brakes to 9000 N.m at a hand-set 0.75 share to "take them out of the
    // way", which was right when `brakes.ini`'s figure was the constraint. It is not any more: with
    // the derived hardware and the proportioning valve the car's *own* brakes stop it in 39.84 m
    // against that override's 42.22, so the override is now the worse brake system and the sweep it
    // feeds reads the tyre through a handicap.
    //
    // The car below is the shipped car. Nothing is overridden but the longitudinal peak.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    std::printf("\n=== mu_x swept on the shipped brakes, against 100-0 of 34.6-35.1 m ===\n");
    std::printf("  the real car's published figures need 1.121-1.137 g longitudinal against a\n");
    std::printf("  0.90-0.95 g skidpad, which is a long/lat ratio of 1.22; this model ships 1.015\n");
    std::printf("\n   mu_x   mu_x/mu_y   best 100-0   at pedal   mean g   locked\n");

    for (const auto muX : {1.131, 1.250, 1.360, 1.500})
    {
        auto setup = base.value();
        for (auto& corner : setup.corners)
        {
            corner.tyre.longitudinalPeak = muX;
        }

        const auto assists = golfGtiMk7Assists(setup);

        auto best = 1e9;
        auto bestPedal = 0.0;
        auto bestRun = StopResult{};

        // The whole pedal, finely — the optimum moves with grip as well as with brake torque, which
        // is the trap this brief records twice.
        for (auto step = 2; step <= 20; step++)
        {
            const auto pedal = 0.05 * static_cast<double>(step);
            const auto run = stop(setup, world.value(), assists, pedal);

            if (run.stopped && run.distance < best)
            {
                best = run.distance;
                bestPedal = pedal;
                bestRun = run;
            }
        }

        std::printf("  %6.3f  %10.3f  %9.2f m  %9.2f  %7.3f   %s\n", muX, muX / setup.corners[0].tyre.lateralPeak, best,
                    bestPedal, bestRun.deceleration / gravity, bestRun.lowestWheelSpeed[0] < 0.5 ? "yes" : "no");
    }
}

TEST_CASE("what splitting the load-sensitivity exponent did to the stop", "[.brake-model]")
{
    // Stage 1 of `docs/tyre-grip-ratio-brief.md`, measured rather than interpolated. The brief
    // estimated the split worth "about 1.1 m" of the 4.7 m gap by reading across a mu_x sweep; this
    // is the run.
    //
    // The `as shipped` row is the car with `longitudinalLoadSensitivity` forced back to the lateral
    // exponent, which is exactly what this model did until 2026-08-23 — so the two rows differ in one
    // number and nothing else.
    //
    // The load columns are here because the split moved three anti-lock cases from green to red on
    // `inContact`, and one bit over four wheels and thirty seconds cannot say whether that is a wheel
    // lifting or a wheel touching down a millisecond late.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    std::printf("\n=== the two load-sensitivity exponents, on the shipped brakes ===\n");
    std::printf("  lateral 0.1926 (tyres.ini LS_EXP_Y), longitudinal 0.1244 (LS_EXP_X)\n");
    std::printf("\n  car            pedal   100-0      mean g   min rear load   first lift   min rear w   grounded\n");

    const auto sweep = [&](const char* name, const VehicleSetup& setup)
    {
        const auto assists = golfGtiMk7Assists(setup);

        auto best = 1e9;
        auto bestPedal = 0.0;
        auto bestRun = StopResult{};

        for (auto step = 2; step <= 20; step++)
        {
            const auto pedal = 0.05 * static_cast<double>(step);
            const auto run = stop(setup, world.value(), assists, pedal);

            if (run.stopped && run.distance < best)
            {
                best = run.distance;
                bestPedal = pedal;
                bestRun = run;
            }
        }

        const auto rear = std::min(bestRun.lowestLoad[2], bestRun.lowestLoad[3]);
        const auto lift = std::max(bestRun.firstLiftTime[2], bestRun.firstLiftTime[3]);

        std::printf("  %-13s %5.2f  %7.2f m  %7.3f  %11.1f N   %8.3f s   %8.2f     %s\n", name, bestPedal, best,
                    bestRun.deceleration / gravity, rear, lift,
                    std::min(bestRun.lowestWheelSpeed[2], bestRun.lowestWheelSpeed[3]),
                    bestRun.grounded ? "yes" : "no");
    };

    auto asShipped = base.value();
    for (auto& corner : asShipped.corners)
    {
        corner.tyre.longitudinalLoadSensitivity = corner.tyre.lateralLoadSensitivity;
    }

    sweep("one exponent", asShipped);
    sweep("two exponents", base.value());

    // And the whole pedal for the car as it now is, because the optimum moves with grip and a row is
    // not a curve.
    std::printf("\n  pedal   100-0      mean g   min rear load   first lift   min rear w   min front w\n");

    const auto assists = golfGtiMk7Assists(base.value());
    for (auto step = 2; step <= 20; step++)
    {
        const auto pedal = 0.05 * static_cast<double>(step);
        const auto run = stop(base.value(), world.value(), assists, pedal);

        std::printf("  %5.2f  %7.2f m  %7.3f  %11.1f N   %8.3f s   %8.2f   %8.2f%s\n", pedal, run.distance,
                    run.deceleration / gravity, std::min(run.lowestLoad[2], run.lowestLoad[3]),
                    std::max(run.firstLiftTime[2], run.firstLiftTime[3]),
                    std::min(run.lowestWheelSpeed[2], run.lowestWheelSpeed[3]),
                    std::min(run.lowestWheelSpeed[0], run.lowestWheelSpeed[1]), run.stopped ? "" : "  (did not stop)");
    }
}

TEST_CASE("the first half second of a stop, tick by tick", "[.brake-model]")
{
    // **Print the raw samples before theorising.** The exponent split moved the 100-0 the wrong way
    // and three anti-lock cases from green to red, and both come back to one wheel: at the optimum
    // pedal the rear reads *zero load* somewhere. A minimum over thirty seconds does not say whether
    // that is a car whose rear axle is unloaded by braking or a car pitching about its dampers, and
    // the two want completely different answers.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto trace = [&](const char* name, const VehicleSetup& setup, const double pedal)
    {
        auto state = VehicleState{};
        settle(setup, state, world.value(), hundred);

        auto assists = golfGtiMk7Assists(setup);
        auto assistState = AssistState{};

        for (auto step = 0; step < 180; step++)
        {
            const auto sensors = AssistSensors{};
            const auto command = updateAssists(assists, assistState, sensors, {}, noBrakePressure, tick);
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes)
                        .has_value());
        }

        auto input = VehicleInput{};
        input.brake = pedal;

        std::printf("\n  %s, pedal %.2f\n", name, pedal);
        std::printf("    t [s]   speed   front load   rear load   front w   rear w   pitch [deg]\n");

        for (auto step = 1; step <= 360; step++)
        {
            auto sensors = AssistSensors{};
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
            }

            const auto command = updateAssists(assists, assistState, sensors, {.brake = pedal, .throttle = 0.0},
                                               brakeCircuitPressures(setup, pedal), tick);
            const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());

            if (step % 12 != 0)
            {
                continue;
            }

            std::printf("   %6.3f  %6.2f  %10.1f  %10.1f  %8.2f %8.2f  %10.3f\n", static_cast<double>(step) * tick,
                        state.chassis.linearVelocity.z,
                        stepped->corners[0].forces.tireVertical + stepped->corners[1].forces.tireVertical,
                        stepped->corners[2].forces.tireVertical + stepped->corners[3].forces.tireVertical,
                        state.corners[0].wheelSpeed, state.corners[2].wheelSpeed,
                        stepped->telemetry.pitch * 57.29577951308232);
        }
    };

    auto asShipped = base.value();
    for (auto& corner : asShipped.corners)
    {
        corner.tyre.longitudinalLoadSensitivity = corner.tyre.lateralLoadSensitivity;
    }

    std::printf("\n=== the transient the lock order is being read through ===\n");
    trace("one exponent", asShipped, 0.35);
    trace("two exponents", base.value(), 0.35);
}
