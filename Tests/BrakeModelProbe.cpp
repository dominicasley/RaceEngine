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
        // **The two bump-stop columns were added 2026-08-24 and they are the point of this trace now.**
        // The rear-load oscillation measures 3.48 Hz with a damping ratio of 0.10, and neither is what
        // this car's springs and dampers give on their own — computed, the pitch mode is 2.09 Hz at
        // zeta 0.24. Both come into line if the front stiffness is about ten times what its spring
        // says, which is what a bump stop is. Inferring that from a frequency is not the same as
        // seeing the force, so here is the force.
        std::printf("    t [s]   speed   front load   rear load   front w   rear w   pitch [deg]   "
                    "F bumpstop   R bumpstop   F travel\n");

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

            std::printf("   %6.3f  %6.2f  %10.1f  %10.1f  %8.2f %8.2f  %10.3f  %10.1f  %10.1f  %8.4f\n",
                        static_cast<double>(step) * tick, state.chassis.linearVelocity.z,
                        stepped->corners[0].forces.tireVertical + stepped->corners[1].forces.tireVertical,
                        stepped->corners[2].forces.tireVertical + stepped->corners[3].forces.tireVertical,
                        state.corners[0].wheelSpeed, state.corners[2].wheelSpeed,
                        stepped->telemetry.pitch * 57.29577951308232,
                        stepped->corners[0].forces.bumpStop + stepped->corners[1].forces.bumpStop,
                        stepped->corners[2].forces.bumpStop + stepped->corners[3].forces.bumpStop,
                        stepped->corners[0].suspension.wheelTravel);
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

TEST_CASE("what correcting the rear spring rate would do to the pitch transient", "[.brake-model]")
{
    // **Measured, not adopted.** The rear wheel rate is the prime suspect for a pitch mode that runs
    // at 3.48 Hz with a damping ratio of 0.10 and throws the rear axle to 189 N mid-stop.
    //
    // The evidence: `suspensions.ini` states 35000 front and 57000 rear, and this model reads both as
    // **wheel** rates. Against a published-but-unverified 3.5 kg/mm front and 4.5 kg/mm rear on motion
    // ratios of 0.96 and 0.64, the wheel rates should be **31632 and 18076** — so the front is right
    // to 11% and **the rear is 3.15 times too stiff**.
    //
    // That asymmetry is itself the argument. At a front motion ratio of 0.96 a spring rate and a wheel
    // rate are the same number to 8%, so AC's front figure is correct whichever convention it is in.
    // At the rear's 0.64 they differ by 1/0.64^2 = 2.4x, and a spring rate read as a wheel rate lands
    // almost exactly where AC's 57000 is. **One end being right is what makes the other end credible
    // as a defect rather than as our own misreading.**
    //
    // What this case does *not* assume is that fixing it fixes the transient. The front is on its bump
    // stops through the whole stop, and a softer rear pitches the car further nose-down, which would
    // push it *deeper* in. So the two corrections are measured together and apart.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    // Scaling `springRate` scales the wheel rate by the same factor: the conversion the car was built
    // with is `springRate = wheelRate / motionRatio^2`, and the geometry is untouched here.
    const auto withRates = [&base](const double frontScale, const double rearScale)
    {
        auto car = base.value();
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            car.corners[index].springRate *= index < 2 ? frontScale : rearScale;
        }

        return car;
    };

    struct Transient
    {
        double lowestRearAxle = 1e9;
        double peakBumpStop = 0.0;
        double stopDistance = 0.0;
    };

    const auto run = [&world](const VehicleSetup& car)
    {
        auto state = VehicleState{};
        settle(car, state, world.value(), hundred);

        auto result = Transient{};
        const auto start = state.chassis.position.z;

        auto input = VehicleInput{};
        input.brake = 0.35;

        for (auto step = 1; step <= 360 * 12; step++)
        {
            const auto stepped = stepVehicle(car, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto rear = stepped->corners[2].forces.tireVertical + stepped->corners[3].forces.tireVertical;
            const auto stop = stepped->corners[0].forces.bumpStop + stepped->corners[1].forces.bumpStop;

            result.lowestRearAxle = std::min(result.lowestRearAxle, rear);
            result.peakBumpStop = std::max(result.peakBumpStop, stop);

            if (state.chassis.linearVelocity.z <= 0.05)
            {
                result.stopDistance = state.chassis.position.z - start;
                break;
            }
        }

        return result;
    };

    std::printf("\n=== the rear spring rate against the pitch transient ===\n");
    std::printf("  wheel rates the source implies: front 31632, rear 18076 N/m\n");
    std::printf("  the car ships with:             front 35000, rear 57000 N/m\n\n");
    std::printf("  front    rear     lowest rear axle    peak front bumpstop    100-0 m\n");

    const auto cases =
        std::array{std::tuple{"35000", "57000", 1.0, 1.0}, std::tuple{"35000", "18076", 1.0, 18076.0 / 57000.0},
                   std::tuple{"31632", "57000", 31632.0 / 35000.0, 1.0},
                   std::tuple{"31632", "18076", 31632.0 / 35000.0, 18076.0 / 57000.0}};

    for (const auto& [frontName, rearName, frontScale, rearScale] : cases)
    {
        const auto measured = run(withRates(frontScale, rearScale));

        std::printf("  %-7s  %-7s  %14.1f N  %18.1f N  %9.2f%s\n", frontName, rearName, measured.lowestRearAxle,
                    measured.peakBumpStop, measured.stopDistance,
                    frontScale == 1.0 && rearScale == 1.0 ? "   <- as shipped" : "");
    }

    std::printf("\n  A rear axle that stops reaching zero, and a front that stops hitting its stop, are two\n");
    std::printf("  different fixes. This says whether either alone is enough.\n");
}

TEST_CASE("the front bump stop's rate, against the travel a real one has", "[.brake-model]")
{
    // **The gap is right and the rate is not.** Measurements of a 2019 GTI Rabbit Edition DSG — same
    // MQB platform — give a bare shock shaft of 73 mm at ride height with **18 mm to the bump stop**
    // and 73 mm of total bump travel. This model's `bumpStop.gap` is 20 mm, so the *travel before the
    // stop* is right to two millimetres and "the front runs out of suspension" is the wrong diagnosis.
    //
    // What is wrong is what happens after it touches. The stop is a placeholder — AC's own
    // `BUMP_STOP_RATE` of 55000 N/m was not carried across — and at the 12.9 mm the trace reaches it
    // has a tangent stiffness of **1123 kN/m**: twenty times AC's stated rate and **thirty-two times
    // the front spring**. That is what takes the pitch mode to 3.48 Hz and its damping ratio to 0.10.
    //
    // A real car does ride its bump stops under hard braking — at a 31632 N/m wheel rate the front
    // needs about 45 mm of travel for the load a 0.94 g stop puts on it, against 18 mm to the stop —
    // so the stop engaging is correct. Engaging like a wall is not.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto withStop =
        [&base](const double rate, const double damping, const double hysteresis, const double rearScale)
    {
        auto car = base.value();
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            car.corners[index].bumpStop.rate = rate;
            car.corners[index].bumpStop.damping = damping;
            car.corners[index].bumpStop.hysteresis = hysteresis;
            if (index >= 2)
            {
                car.corners[index].springRate *= rearScale;
            }
        }

        return car;
    };

    const auto run = [&world](const VehicleSetup& car)
    {
        auto state = VehicleState{};
        settle(car, state, world.value(), hundred);

        auto lowestRear = 1e9;
        auto peakStop = 0.0;
        auto deepestTravel = 0.0;
        auto deepestDroop = 0.0;
        auto rearExtension = 0.0;
        auto distance = 0.0;
        const auto start = state.chassis.position.z;

        auto input = VehicleInput{};
        input.brake = 0.35;

        for (auto step = 1; step <= 360 * 12; step++)
        {
            const auto stepped = stepVehicle(car, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            lowestRear =
                std::min(lowestRear, stepped->corners[2].forces.tireVertical + stepped->corners[3].forces.tireVertical);
            peakStop = std::max(peakStop, stepped->corners[0].forces.bumpStop);
            deepestTravel = std::max(deepestTravel, stepped->corners[0].suspension.wheelTravel);
            // **The rear's DROOP stop**, which is where this trail ends. A rear tyre carrying 80 N is
            // 0.27 mm from leaving the road, and a wheel can only follow the road as far as its own
            // suspension lets it extend.
            deepestDroop = std::min(deepestDroop, stepped->corners[2].forces.droopStop);
            rearExtension = std::min(rearExtension, stepped->corners[2].suspension.wheelTravel);

            if (state.chassis.linearVelocity.z <= 0.05)
            {
                distance = state.chassis.position.z - start;
                break;
            }
        }

        return std::array{lowestRear, peakStop, deepestTravel, distance, deepestDroop, rearExtension};
    };

    std::printf("\n=== the front bump stop's rate ===\n");
    std::printf("  measured: 18 mm to the stop, 73 mm total bump travel. Model gap: 20 mm.\n");
    std::printf("  AC states BUMP_STOP_RATE = 55000 N/m linear; the model places 900000 at cubic.\n\n");
    // **Swept over the DAMPING and not the rate**, and that is a correction. Sweeping the rate first
    // moved nothing at all — the peak "bump stop force" came back at 7210 N for every stiffness from
    // 55000 to 900000, which is `40000 x 0.18 m/s` and therefore the stop's **damper**, not its
    // spring. At 40000 N.s/m that term is five times the corner's own critical damping, so the stop
    // does not so much stiffen the car as hit it.
    //
    // **Two rows added 2026-08-29 with the hysteresis term**: the corner's own critical damping
    // (derived in the drop-strike case below), and the BASF-sourced alternative — no viscous
    // constant at all, 7 % of the elastic force as rate-independent hysteresis, which is the middle
    // of the band their own Fig. 5 densities convert to.
    std::printf("   rate     damping   hyst   rear spring   lowest rear axle   peak F stop   F travel   R droop "
                "force   R extension\n");

    const auto cases = std::array{std::tuple{900000.0, 40000.0, 0.0, 1.0, "shipped"},
                                  std::tuple{900000.0, 10000.0, 0.0, 1.0, ""},
                                  std::tuple{900000.0, 8000.0, 0.0, 1.0, "~ critical"},
                                  std::tuple{900000.0, 4000.0, 0.0, 1.0, ""},
                                  std::tuple{900000.0, 0.0, 0.0, 1.0, "no stop damping"},
                                  std::tuple{900000.0, 0.0, 0.07, 1.0, "BASF hysteresis only"},
                                  std::tuple{55000.0, 4000.0, 0.0, 1.0, "AC rate, sane damping"},
                                  std::tuple{55000.0, 4000.0, 0.0, 18076.0 / 57000.0, "...and the rear fix"}};

    for (const auto& [rate, damping, hysteresis, rearScale, note] : cases)
    {
        const auto measured = run(withStop(rate, damping, hysteresis, rearScale));

        std::printf("  %8.0f  %8.0f  %5.2f   %9.0f   %14.1f N  %9.1f N  %7.1f mm  %11.1f N  %8.1f mm  %s\n", rate,
                    damping, hysteresis, 57000.0 * rearScale, measured[0], measured[1], 1000.0 * measured[2],
                    measured[4], 1000.0 * measured[5], note);
    }

    std::printf("\n  Watch the travel column: a stop soft enough to stop ringing the car is also a stop\n");
    std::printf("  that lets the suspension run further, and the linkage has a hard limit behind it.\n");
}

TEST_CASE("dropping the car onto its front stops: shipped damping against critical and the material's own loop",
          "[.brake-model]")
{
    // The 40000 N·s/m question, measured on the event it was placed for. The viscous constant
    // exists because a car dropped onto pure-spring stops pogoed; since then the car has grown a
    // kneed damper, 107 N of seal friction and real droop travel, so whether the stop still needs
    // to be the thing that settles the car is a measurement and not a memory.
    //
    // Three stops: shipped (40000), the front corner's own critical damping (derived below from
    // the car's numbers, not typed), and the sourced one — no viscous constant, 7 % of the elastic
    // force as rate-independent hysteresis, the middle of what BASF's Cellasto brochure publishes
    // (a 10-20 % hysteresis-loop share of the deformation work; their Fig. 5 densities give
    // 11.9-13.1 %, and loop share A converts to a force fraction h by A = 2h/(1+h)).
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    // --- the critical-damping arithmetic, from the car rather than from a comment ---
    const auto properties = computeMassProperties(base->sprung);
    REQUIRE(properties.has_value());

    const auto& front = base->corners[0];
    const auto frontAxle = front.hardpoints.wheelCentre.z;
    const auto rearAxle = base->corners[2].hardpoints.wheelCentre.z;
    const auto share = (properties->centreOfMass.z - rearAxle) / (frontAxle - rearAxle);
    const auto cornerMass = properties->mass * share / 2.0;

    const auto kinematics =
        raceengine::solveDamperKinematics(front.hardpoints, raceengine::damperElementOf(front.hardpoints), 0.0, 0.0);
    REQUIRE(kinematics.has_value());
    const auto ratio = std::abs(kinematics->motionRatio);

    const auto wheelRate = front.springRate * ratio * ratio;
    const auto criticalWheel = 2.0 * std::sqrt(wheelRate * cornerMass);
    const auto criticalShaft = criticalWheel / (ratio * ratio);

    // And against the stiffest mode the stop itself reaches: its tangent stiffness at the 12.9 mm
    // the 0.35-pedal braking trace penetrates to.
    const auto braked = 0.0129;
    const auto tangent = front.bumpStop.rate * front.bumpStop.progression *
                         std::pow(braked, front.bumpStop.progression - 1.0) /
                         std::pow(front.bumpStop.gap, front.bumpStop.progression - 1.0);
    const auto criticalOnStop = 2.0 * std::sqrt(tangent * cornerMass / (ratio * ratio));

    std::printf("\n=== the front bump stop's damping against the corner's own critical ===\n");
    std::printf("  front corner sprung mass          %8.1f kg  (sprung %.1f kg, front share %.3f)\n", cornerMass,
                properties->mass, share);
    std::printf("  damper motion ratio at design     %8.3f\n", ratio);
    std::printf("  front wheel rate                  %8.0f N/m\n", wheelRate);
    std::printf("  critical damping at the wheel     %8.0f N.s/m\n", criticalWheel);
    std::printf("  critical damping on the shaft     %8.0f N.s/m\n", criticalShaft);
    std::printf("  shipped stop damping (shaft)      %8.0f N.s/m = %.1fx critical\n", front.bumpStop.damping,
                front.bumpStop.damping / criticalShaft);
    std::printf("  stop tangent stiffness at 12.9 mm %8.0f N/m; critical against IT %8.0f N.s/m (%.2fx)\n", tangent,
                criticalOnStop, front.bumpStop.damping / criticalOnStop);
    std::printf("  a bulk polyurethane's own damping ratio is 0.044-0.133 of critical (PMC11643408)\n");

    // --- the drop ---
    struct Strike
    {
        double peakStop = 0.0;
        double viscousShare = 0.0;
        double deepest = 0.0;
        int touches = 0;
        double firstPeak = 0.0;
        double later = 0.0;
        double settling = 0.0;
    };

    const auto measure = [&](const double damping, const double hysteresis)
    {
        auto car = base.value();
        for (auto& corner : car.corners)
        {
            corner.bumpStop.damping = damping;
            corner.bumpStop.hysteresis = hysteresis;
        }

        auto state = VehicleState{};
        settle(car, state, world.value(), 0.0);

        const auto rested = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(rested.has_value());
        const auto rest = rested->corners[0].suspension.wheelTravel;

        state.chassis.linearVelocity = glm::dvec3(0.0, -3.0, 0.0);

        auto result = Strike{};
        auto outside = 0;
        constexpr auto steps = 360 * 8;

        for (auto step = 1; step <= steps; step++)
        {
            const auto stepped = stepVehicle(car, state, VehicleInput{}, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto stop = stepped->corners[0].forces.bumpStop;
            const auto excursion = std::abs(stepped->corners[0].suspension.wheelTravel - rest);

            if (stop > result.peakStop)
            {
                result.peakStop = stop;
                const auto viscous = damping * stepped->corners[0].damperVelocity;
                result.viscousShare = stop > 0.0 ? std::max(0.0, viscous) / stop : 0.0;
            }

            result.deepest = std::max(result.deepest, stepped->corners[0].suspension.wheelTravel - rest);
            result.touches += stop > 0.0 ? 1 : 0;

            if (step <= 360)
            {
                result.firstPeak = std::max(result.firstPeak, excursion);
            }
            else
            {
                result.later = std::max(result.later, excursion);
            }

            if (excursion > 0.002)
            {
                outside = step;
            }
        }

        result.settling = static_cast<double>(outside) * tick;

        return result;
    };

    std::printf("\n=== a 3.0 m/s drop onto the wheels, front corner ===\n");
    std::printf("   damping   hyst   peak F stop   viscous share   deepest   touches   settle to 2 mm   worst after "
                "1 s\n");

    const auto cases =
        std::array{std::tuple{front.bumpStop.damping, 0.0, "shipped"}, std::tuple{criticalShaft, 0.0, "critical"},
                   std::tuple{0.0, 0.07, "BASF hysteresis only"}, std::tuple{0.0, 0.0, "bare spring"}};

    for (const auto& [damping, hysteresis, note] : cases)
    {
        const auto strike = measure(damping, hysteresis);

        std::printf("  %8.0f  %5.2f  %10.1f N  %13.1f %%  %7.2f mm  %7d  %13.3f s  %11.2f mm  %s\n", damping,
                    hysteresis, strike.peakStop, 100.0 * strike.viscousShare, 1000.0 * strike.deepest, strike.touches,
                    strike.settling, 1000.0 * strike.later, note);

        // The one bound worth holding here: with the sourced dissipation instead of the placed
        // constant the car must still settle inside the window — the main damper and its friction
        // own the settle now, and if that stops being true this table is where it shows first.
        if (hysteresis > 0.0)
        {
            CHECK(strike.settling < 8.0);
            CHECK(strike.touches > 0);
        }
    }

    std::printf("\n  The stop the material describes arrives with the elastic force and cannot spike on entry;\n");
    std::printf("  the shipped constant's peak is mostly its own viscous term. Whether the car should keep the\n");
    std::printf("  placed constant is the seat's call: front.stopdamping / front.stophysteresis on the sheet.\n");
}

TEST_CASE("how much travel the linkage actually has, against the stops placed in it", "[.brake-model]")
{
    // **Droop is set by the damper topping out, not by a separate bumper**, and that reframes the
    // number. On a strut the rebound limit *is* the shock reaching its maximum length, so the travel
    // available is the stroke left over once ride height has used some of it. Measurements of a 2019
    // GTI Rabbit DSG give 73 mm of exposed shaft at ride height — that is the **bump** side — and the
    // droop side is the rest of the stroke.
    //
    // This model expresses both ends as `TravelStop`s with placed 20 mm gaps. Before any of that is
    // worth correcting, the question is whether the **linkage** would even allow more: a stop gap
    // larger than the geometry's own range is a stop that never touches, and `validateCornerSetup`
    // refuses exactly that. So this reports what the geometry has.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    std::printf("\n=== travel the linkage allows, against the stops placed in it ===\n");
    std::printf("  corner   shaft bump   shaft droop   wheel bump   wheel droop   bump gap   droop gap\n");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup->corners[index];

        const auto design = solveCorner(corner.hardpoints, 0.0, 0.0);
        const auto atBump = solveCorner(corner.hardpoints, corner.hardpoints.bumpAngle, 0.0);
        const auto atDroop = solveCorner(corner.hardpoints, corner.hardpoints.droopAngle, 0.0);

        REQUIRE(design.has_value());
        REQUIRE(atBump.has_value());
        REQUIRE(atDroop.has_value());

        // Positive is compression on the shaft; the linkage's own limits either side of design.
        // Shaft lengths off the damper element, which is the sole source of damper geometry since
        // the legacy state fields were retired (step 14) — the same bits the solver used to carry.
        const auto element = raceengine::damperElementOf(corner.hardpoints);
        const auto designLength = raceengine::solveElement(corner.hardpoints, element, 0.0).length;
        const auto shaftBump =
            designLength - raceengine::solveElement(corner.hardpoints, element, corner.hardpoints.bumpAngle).length;
        const auto shaftDroop =
            raceengine::solveElement(corner.hardpoints, element, corner.hardpoints.droopAngle).length - designLength;
        const auto wheelBump = atBump->wheelTravel - design->wheelTravel;
        const auto wheelDroop = design->wheelTravel - atDroop->wheelTravel;

        std::printf("    %zu     %8.1f mm  %9.1f mm  %9.1f mm  %10.1f mm  %6.1f mm  %8.1f mm\n", index,
                    1000.0 * shaftBump, 1000.0 * shaftDroop, 1000.0 * wheelBump, 1000.0 * wheelDroop,
                    1000.0 * corner.bumpStop.gap, 1000.0 * corner.droopStop.gap);
    }

    std::printf("\n  Measured on the real car: 18 mm to the bump stop and 73 mm of total bump travel.\n");
    std::printf("  If the linkage here has far less than 73 mm, the stops are not the whole problem —\n");
    std::printf("  the droop and bump ANGLES are placed too, and they bound everything behind them.\n");
}

TEST_CASE("does giving the rear its droop travel back keep the wheel on the road", "[.brake-model]")
{
    // **The decisive test for the rear lifting.** Droop is the damper topping out, and the linkage
    // here offers 52.6 mm of it at the rear while the placed `droopStop.gap` binds at 20. So the car
    // is being stopped from extending by a placeholder with 32 mm of its own travel unused.
    //
    // Swept up to the linkage's own limit — beyond which `validateCornerSetup` rightly refuses,
    // because a stop with no travel behind it is a clamp and a clamp has no reaction force.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    std::printf("\n=== the rear's droop travel against the wheel staying down ===\n");
    std::printf("  linkage allows 52.6 mm of rear droop; the placed stop binds at 20.\n\n");
    std::printf("  droop gap   lowest rear axle   rear extension   droop force   100-0 m\n");

    for (const auto gap : {0.020, 0.030, 0.040, 0.045})
    {
        auto car = base.value();
        for (auto& corner : car.corners)
        {
            corner.droopStop.gap = gap;
        }

        // The car must still be a legal car at the new gap, or the number below means nothing.
        for (const auto& corner : car.corners)
        {
            REQUIRE(validateCornerSetup(corner).has_value());
        }

        auto state = VehicleState{};
        settle(car, state, world.value(), hundred);

        auto lowestRear = 1e9;
        auto extension = 0.0;
        auto droopForce = 0.0;
        auto distance = 0.0;
        const auto start = state.chassis.position.z;

        auto input = VehicleInput{};
        input.brake = 0.35;

        for (auto step = 1; step <= 360 * 12; step++)
        {
            const auto stepped = stepVehicle(car, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            lowestRear =
                std::min(lowestRear, stepped->corners[2].forces.tireVertical + stepped->corners[3].forces.tireVertical);
            extension = std::min(extension, stepped->corners[2].suspension.wheelTravel);
            droopForce = std::min(droopForce, stepped->corners[2].forces.droopStop);

            if (state.chassis.linearVelocity.z <= 0.05)
            {
                distance = state.chassis.position.z - start;
                break;
            }
        }

        std::printf("   %6.1f mm  %14.1f N  %11.1f mm  %10.1f N  %8.2f%s\n", 1000.0 * gap, lowestRear,
                    1000.0 * extension, droopForce, distance, gap == 0.020 ? "   <- shipped" : "");
    }

    std::printf("\n  If the rear load climbs with the gap, the placeholder was the constraint all along.\n");
    std::printf("  If it does not, the wheel is leaving for a reason the suspension cannot reach.\n");
}
