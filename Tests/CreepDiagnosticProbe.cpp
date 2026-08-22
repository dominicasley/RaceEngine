// T2b: what a DSG does with the brakes released and nobody on the throttle.
//
//   ./EngineTests "[.creep]"        flat creep, against the same code with the rule switched off
//   ./EngineTests "[.creep-grade]"  creep speed against slope, up and down, and where it gives up
//   ./EngineTests "[.creep-brake]"  held on a light brake: still, and quiet
//   ./EngineTests "[.creep-hand]"   the torque trace across throttle-on and throttle-off
//   ./EngineTests "[.creep-hunt]"   sixty seconds of creep, and what the slip energy did
//   ./EngineTests "[.creep-gravity]" whether a car on a hill rolls down it — read this one first
//
// The T-series brief's diagnosis is that the DSG's *plant* is right and its *controller* is missing.
// Anti-stall is one rule of a TCU that needs several, and it is the protective rule. Creep is the
// productive one, and this measures it.
//
// **Launch control is not in scope and this probe must not reach it.** The previous T2 attempt built
// a launch regulator and every metered variant was slower and spinnier than the code it replaced. The
// launch is 6.556 s against a published 6.4-6.7 s for a DSG without launch control; there is no launch
// problem to fix, and `[.launch]` is where it is checked that this work did not touch one.
//
// Every fixture here asserts its own preconditions before a number is taken off it. Five fixture
// faults in one recent session produced three wrong conclusions and an entire wasted TCU build; the
// two that would have caught them are "the state is what it claims" and "the ground is big enough and
// the car stayed on it", and both are asserted below.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::EngineState;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::sampleProvingGround;
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

// Where the car is put down, and it is in the middle of the plate rather than at one end because a
// creep test can go *backwards*: on a grade steep enough to beat the creep torque the car rolls away
// downhill, which is the correct answer and is also a hundred metres of travel in the direction a
// fixture laid out for acceleration has no room in.
constexpr auto startStation = 400.0;

// The ground a creep test needs, and its size is a precondition rather than a convenience. A creep
// test runs for tens of seconds at walking pace and travels further than it looks — sixty seconds at
// 2 m/s is 122 m — and a skidpad fixture in this project has already produced a fictional finding by
// running its car off the end of its plate. This one is 900 m long with the car in the middle, so
// there is room either way, and `runCreep` asserts it rather than trusting it.
//
// The slope, where there is one, covers the whole plate: `Slope` is deliberately un-eased, so a band
// that started partway along would put a crease under the car at the exact moment it was being asked
// to hold station.
[[nodiscard]] ProvingGroundDescriptor creepGround(const double slopeAngle)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 900.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 2.0;
    descriptor.slopeAngle = slopeAngle;
    descriptor.features = {};

    if (std::abs(slopeAngle) > 1e-9)
    {
        descriptor.features = {Feature{.kind = FeatureKind::Slope, .from = 0.0, .to = descriptor.length}};
    }

    return descriptor;
}

[[nodiscard]] double designHeight(const VehicleSetup& setup)
{
    auto highest = 0.0;
    for (const auto& corner : setup.corners)
    {
        highest = std::max(highest, corner.hardpoints.wheelCentre.y + corner.hardpoints.wheelRadius);
    }

    return highest;
}

// One tick as recorded.
struct Sample
{
    double time = 0.0;
    double speed = 0.0;
    double station = 0.0;
    double engineSpeed = 0.0;
    double clutchPedal = 0.0;
    double clutchTorque = 0.0;
    double clutchSlip = 0.0;
    double slipEnergy = 0.0;
    double creepCommand = 0.0;
    double wheelTorque = 0.0;
    double height = 0.0;
    bool locked = false;
};

struct Run
{
    std::vector<Sample> samples;
};

// What the driver does, over the length of one run. A description rather than a callback so that a
// case is one line at the call site and there is nowhere for a lambda's capture to change what a
// previous case measured.
struct Script
{
    double brake = 0.0;
    double brakeFrom = 0.0;

    double throttle = 0.0;
    double throttleFrom = 1e30;
    double throttleTo = 1e30;
};

// Stand the car on the ground, start it, idle it in gear on the brakes, assert that it really is
// standing still in gear at idle with the clutch open, and then run the driver's script.
//
// **The idle-on-the-brakes phase is not decoration.** `DrivelineState::clutchPedal` defaults to 0.0,
// which is a fully *engaged* clutch, and the pedal is rate limited — so a fixture that starts the
// engine and immediately releases the brakes measures a cold start rather than a creep. That exact
// fault cost an entire TCU build once already.
[[nodiscard]] Run runCreep(const PhysicsWorld& world, const ProvingGroundDescriptor& ground, const double slopeAngle,
                           const double seconds, const Script& script, const DrivelineSetup& driveline,
                           const VehicleSetup& setup)
{
    auto state = VehicleState{};

    // Placed on the surface rather than above a nominal plane, and pitched to match it. A level car
    // dropped onto a ten-percent grade puts one axle a hundred and thirty millimetres into the ground
    // and the other the same distance above it, which settles as a bounce nobody asked for and reads
    // as a car that will not hold station.
    const auto surface = sampleProvingGround(ground, 0.0, startStation).height;

    state.chassis.position = glm::dvec3(0.0, surface + designHeight(setup), startStation);
    state.chassis.orientation = glm::angleAxis(-slopeAngle, glm::dvec3(1.0, 0.0, 0.0));

    for (auto step = 0; step < 1440; step++)
    {
        auto holding = VehicleInput{};
        holding.brake = 1.0;

        REQUIRE(stepVehicle(setup, state, holding, noDriveTorque, world, tick).has_value());
    }

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    const auto inertias = wheelInertias(setup);
    auto road = std::array<double, cornerCount>{};

    {
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto step = 0; step < 720; step++)
        {
            const auto torques = stepDriveline(driveline, drivelineState,
                                               {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                               inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, idling, torques->wheel, world, tick);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());
        }
    }

    // --- preconditions, before a single number is taken off this fixture ---
    {
        const auto atRest = glm::length(state.chassis.linearVelocity);
        CAPTURE(atRest, drivelineState.clutchPedal, drivelineState.engineSpeed, drivelineState.gear, slopeAngle);

        // Stationary on the brakes, in first, engine alight at its own idle.
        REQUIRE(atRest < 0.05);
        REQUIRE(drivelineState.gear == 1);
        REQUIRE(drivelineState.engine == EngineState::Running);
        REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
        REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);

        // The clutch is open. Held on the brakes a transmission does not sit slipping against them,
        // so the pedal must be near its stop — and a creep rule that fired through a full brake
        // application would show up here rather than three conclusions downstream.
        REQUIRE(drivelineState.clutchPedal > 0.85);

        // And the wheels are not turning, so the first tick's numbers mean something.
        for (const auto& corner : state.corners)
        {
            REQUIRE(std::abs(corner.wheelSpeed) < 0.5);
        }

        // The car is on the plate, which is what makes a station a distance rather than a coordinate.
        REQUIRE(state.chassis.position.z > 0.0);
        REQUIRE(state.chassis.position.z < ground.length);
    }

    auto run = Run{};

    const auto ticks = static_cast<int>(seconds * 360.0);
    run.samples.reserve(static_cast<std::size_t>(ticks));

    for (auto step = 1; step <= ticks; step++)
    {
        const auto time = static_cast<double>(step) * tick;

        auto input = VehicleInput{};
        input.brake = time >= script.brakeFrom ? script.brake : 0.0;
        input.throttle = time >= script.throttleFrom && time < script.throttleTo ? script.throttle : 0.0;
        input.gear = 1;

        const auto torques = stepDriveline(driveline, drivelineState,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());

        // Along the slope rather than along z, so an uphill and a downhill run report the same
        // quantity. On the flat the two are the same number.
        const auto along = state.chassis.linearVelocity.z * std::cos(slopeAngle) +
                           state.chassis.linearVelocity.y * std::sin(slopeAngle);

        run.samples.push_back(Sample{.time = time,
                                     .speed = along,
                                     .station = state.chassis.position.z,
                                     .engineSpeed = drivelineState.engineSpeed,
                                     .clutchPedal = drivelineState.clutchPedal,
                                     .clutchTorque = torques->clutch,
                                     .clutchSlip = torques->clutchSlip,
                                     .slipEnergy = torques->slipEnergy,
                                     .creepCommand = torques->creepCommand,
                                     .wheelTorque = torques->wheel[0] + torques->wheel[1],
                                     .height = state.chassis.position.y,
                                     .locked = torques->clutchLocked});
    }

    // **The car stayed on the plate**, in whichever direction it went. This is the assertion the
    // coasting-skidpad fault needed and did not have: a car that leaves its ground loses support and
    // every number after that is fiction.
    {
        const auto ended = state.chassis.position.z;
        CAPTURE(ended, ground.length, slopeAngle);
        REQUIRE(ended > 5.0);
        REQUIRE(ended < ground.length - 5.0);
    }

    return run;
}

// The same, on the car every other case here uses.
[[nodiscard]] Run runCreep(const PhysicsWorld& world, const ProvingGroundDescriptor& ground, const double slopeAngle,
                           const double seconds, const Script& script, const DrivelineSetup& driveline)
{
    return runCreep(world, ground, slopeAngle, seconds, script, driveline, golfGtiMk7().value());
}

[[nodiscard]] double settledSpeed(const Run& run, const double from)
{
    auto total = 0.0;
    auto count = 0;

    for (const auto& sample : run.samples)
    {
        if (sample.time >= from)
        {
            total += sample.speed;
            count++;
        }
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

[[nodiscard]] int lockTransitions(const Run& run, const double from)
{
    auto transitions = 0;
    auto previous = false;
    auto started = false;

    for (const auto& sample : run.samples)
    {
        if (sample.time < from)
        {
            continue;
        }

        if (started && sample.locked != previous)
        {
            transitions++;
        }

        previous = sample.locked;
        started = true;
    }

    return transitions;
}

// The largest change in clutch torque between two consecutive ticks over a window. What a "step at
// the transition" would be, stated as a number rather than as a shape.
[[nodiscard]] double worstJump(const Run& run, const double from, const double to)
{
    auto worst = 0.0;
    auto previous = 0.0;
    auto started = false;

    for (const auto& sample : run.samples)
    {
        if (sample.time < from || sample.time > to)
        {
            continue;
        }

        if (started)
        {
            worst = std::max(worst, std::abs(sample.clutchTorque - previous));
        }

        previous = sample.clutchTorque;
        started = true;
    }

    return worst;
}

// The coupling point: idle through first gear on this wheel, which is where a closed clutch puts the
// car and is therefore what creep converges on. Not a chosen number — the three figures it is made of
// are engine.ini's IDLE, drivetrain.ini's GEAR_1 and FINAL, and tyres.ini's RADIUS.
[[nodiscard]] double couplingPointSpeed()
{
    const auto driveline = golfGtiMk7Driveline();
    const auto setup = golfGtiMk7().value();

    return driveline.engine.idleSpeed / (driveline.gearbox.ratios.front() * driveline.gearbox.finalDrive) *
           setup.corners.front().hardpoints.wheelRadius;
}

[[nodiscard]] double gradeAngle(const double percent)
{
    return std::atan(percent / 100.0);
}

} // namespace

TEST_CASE("what a DSG does with the brakes off and nobody on the throttle", "[.creep]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    const auto driveline = golfGtiMk7Driveline();

    // The baseline first, and it is the same code with the rule switched off rather than a different
    // build: what "there is no creep" means has to be a measurement on this fixture too, or the
    // comparison below is against a memory.
    auto without = driveline;
    without.autoClutch.creep = false;

    const auto bare = runCreep(world.value(), flat, 0.0, 20.0, Script{}, without);
    const auto run = runCreep(world.value(), flat, 0.0, 20.0, Script{}, driveline);

    const auto coupling = couplingPointSpeed();

    std::printf("\n=== with the creep rule switched off ===\n");
    std::printf("  travelled in 20 s:     %.4f m\n", bare.samples.back().station - startStation);
    std::printf("  speed at 20 s:         %.4f m/s\n", bare.samples.back().speed);
    std::printf("  clutch pedal at 20 s:  %.4f\n", bare.samples.back().clutchPedal);
    std::printf("  clutch torque at 20 s: %.4f N.m\n", bare.samples.back().clutchTorque);
    std::printf("  slip energy over 20 s: %.4f J\n", bare.samples.back().slipEnergy);

    std::printf("\n=== flat creep, brakes off, no throttle ===\n");
    std::printf("  the coupling point — idle through first, on this wheel — is %.4f m/s = %.3f km/h\n", coupling,
                coupling * 3.6);

    std::printf("\n  %8s %10s %10s %10s %10s %10s %10s %8s\n", "t", "m/s", "km/h", "engine", "pedal", "creepCmd",
                "clutchT", "locked");
    for (const auto& sample : run.samples)
    {
        if (std::abs(sample.time * 4.0 - std::round(sample.time * 4.0)) > 0.5 * tick)
        {
            continue;
        }

        std::printf("  %8.3f %10.4f %10.3f %10.2f %10.4f %10.2f %10.2f %8s\n", sample.time, sample.speed,
                    sample.speed * 3.6, sample.engineSpeed, sample.clutchPedal, sample.creepCommand,
                    sample.clutchTorque, sample.locked ? "yes" : "no");
    }

    const auto settled = settledSpeed(run, 15.0);
    std::printf("\n  settled speed over the last five seconds: %.4f m/s = %.3f km/h\n", settled, settled * 3.6);
    std::printf("  as a fraction of the coupling point:      %.4f\n", coupling > 0.0 ? settled / coupling : 0.0);
    std::printf("  slip energy over the run:                 %.2f J\n", run.samples.back().slipEnergy);
    std::printf("  lock/slip transitions after 2 s:          %d\n", lockTransitions(run, 2.0));
}

// **Before any grade number is believed, the grade has to be there.** The first grade sweep run here
// reported the same creep speed to four figures at every slope from -15% to +15%, which is either a
// finding or a fixture, and the fixture is always checked first.
TEST_CASE("the grade fixture really is on a grade", "[.creep-fixture]")
{
    const JoltGuard jolt;

    std::printf("\n=== is the slope in the mesh, and is the car on it ===\n");
    std::printf("  %8s %12s %12s %12s %12s %12s\n", "grade", "sampled y", "car y start", "car y end", "dz", "dy");

    for (const auto percent : {-10.0, 0.0, 10.0})
    {
        const auto angle = gradeAngle(percent);
        const auto ground = creepGround(angle);

        const auto mesh = generateProvingGround(ground);
        REQUIRE(mesh.has_value());
        const auto world = PhysicsWorld::create(mesh.value());
        REQUIRE(world.has_value());

        // Straight off the sampler, which is what the mesh generator itself calls per vertex.
        const auto sampled = sampleProvingGround(ground, 0.0, startStation).height;

        const auto run = runCreep(world.value(), ground, angle, 8.0, Script{}, golfGtiMk7Driveline());

        std::printf("  %7.1f%% %12.4f %12.4f %12.4f %12.4f %12.4f\n", percent, sampled, run.samples.front().height,
                    run.samples.back().height, run.samples.back().station - run.samples.front().station,
                    run.samples.back().height - run.samples.front().height);
    }

    std::printf("\n  `sampled y` is the ground height the generator wrote at the car's start station.\n"
                "  `dy / dz` over the run must equal tan(grade) if the car is driving on the slope.\n");
}

// **How far the take-up bogs the engine, against how fast the pressure is ramped in.**
//
// Creep loads an engine that is idling on its governor, and the governor is a 2 Hz PI — so a creep
// pressure that arrives faster than half a second arrives faster than the thing that has to answer
// it. Ramped in over three tenths of a second the placeholder car dips to **0.865 of idle**, which is
// inside `antiStallBegin`'s 0.90 band: the protective rule then starts opening the clutch the
// productive one is closing, which is the one interaction between them that must not happen.
//
// Both cars, because a rule that only works on one is not a rule, and the two disagree about this by
// a factor nobody predicted.
TEST_CASE("what the creep take-up costs the idle", "[.creep-idle]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    std::printf("\n=== the engine's worst excursion during a creep take-up ===\n");
    std::printf("  as a fraction of that car's own idle. `antiStallBegin` is 0.90 and the take-up must\n");
    std::printf("  stay above it; `antiStallOpen` is 0.55 and is where the clutch is fully out.\n\n");
    std::printf("  `rise` is the largest tick-to-tick change in *clutch* torque over the first second,\n");
    std::printf("  which is what a ramp on the commanded pressure buys. It is read off the clutch and\n");
    std::printf("  not off the axle because releasing the brake rings the compliant shaft by +/-47 N.m\n");
    std::printf("  through a fully open clutch, which is five times the take-up and none of its doing.\n\n");
    std::printf("  %12s %10s %14s %14s %10s %12s\n", "apply N.m/s", "to full", "Golf GTI", "placeholder", "flips",
                "rise N.m");

    for (const auto rate : {400.0, 100.0, 60.0, 40.0, 25.0, 15.0})
    {
        std::printf("  %12.0f %10.2f", rate, 30.0 / rate);

        auto flips = 0;
        auto jerk = 0.0;

        for (const auto golf : {true, false})
        {
            auto driveline = golf ? golfGtiMk7Driveline() : raceengine::placeholderDriveline();
            driveline.autoClutch.creepApplyRate = rate;

            const auto vehicle = golf ? golfGtiMk7().value() : raceengine::placeholderSedan().value();
            const auto run = runCreep(world.value(), flat, 0.0, 10.0, Script{}, driveline, vehicle);

            auto lowest = driveline.engine.idleSpeed;
            for (const auto& sample : run.samples)
            {
                lowest = std::min(lowest, sample.engineSpeed);
            }

            auto previous = 0.0;
            for (const auto& sample : run.samples)
            {
                if (sample.time <= 1.0)
                {
                    jerk = std::max(jerk, std::abs(sample.clutchTorque - previous));
                }

                previous = sample.clutchTorque;
            }

            std::printf(" %14.4f", lowest / driveline.engine.idleSpeed);
            flips = std::max(flips, lockTransitions(run, 0.0));
        }

        std::printf(" %10d %12.2f\n", flips, jerk);
    }
}

// Where the 92 N.m tick-to-tick step in the first second comes from, since it is the same number at
// every apply rate from 400 N.m/s to 15 and therefore cannot be the ramp.
TEST_CASE("the first second of a creep take-up, at two apply rates", "[.creep-first]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    for (const auto rate : {400.0, 15.0})
    {
        auto driveline = golfGtiMk7Driveline();
        driveline.autoClutch.creepApplyRate = rate;

        const auto run = runCreep(world.value(), flat, 0.0, 1.2, Script{}, driveline);

        std::printf("\n=== apply rate %.0f N.m/s ===\n", rate);
        std::printf("  %8s %10s %10s %10s %10s %10s\n", "t", "creepCmd", "pedal", "clutchT", "wheelT", "d(wheelT)");

        auto previous = 0.0;
        for (const auto& sample : run.samples)
        {
            const auto change = sample.wheelTorque - previous;
            previous = sample.wheelTorque;

            if (sample.time > 0.12 && std::abs(sample.time * 20.0 - std::round(sample.time * 20.0)) > 0.5 * tick)
            {
                continue;
            }

            std::printf("  %8.4f %10.4f %10.4f %10.3f %10.2f %10.3f\n", sample.time, sample.creepCommand,
                        sample.clutchPedal, sample.clutchTorque, sample.wheelTorque, change);
        }
    }
}

// **What the sandbox's scripted launch does in its first third of a second**, which is the only thing
// in this project that the `caught` sign correction moves — the driving parity gate captures at frame
// 120 and it moves by 23,191 of 32,400 blocks.
//
// That launch is not the unit fixture's. It goes to full throttle from a *default-constructed*
// driveline, whose `clutchPedal` is 0.0 — a fully engaged clutch — with no braked idle in front of it.
// So it is the cold-start case, and the question is whether the compliant shaft rings backwards
// through it: `clutchSideSpeed` is `shaftSpeed * gearRatio`, and while it is negative the old
// `std::abs` read a shaft spinning the wrong way as a car that had caught up with its gear.
TEST_CASE("the sandbox launch's first third of a second", "[.creep-coldstart]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), startStation);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    auto drive = DrivelineState{};
    startEngine(driveline, drive);

    const auto inertias = wheelInertias(setup);
    auto road = std::array<double, cornerCount>{};

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 1;

    std::printf("\n=== full throttle from a default-constructed driveline, as the sandbox does it ===\n");
    std::printf("  %7s %11s %13s %10s %11s %11s %10s\n", "tick", "shaftSpeed", "clutchSide", "pedal", "clutchT",
                "wheelSpd", "engine");

    auto negatives = 0;
    auto worst = 0.0;

    for (auto step = 1; step <= 130; step++)
    {
        const auto torques = stepDriveline(driveline, drive,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world.value(), tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());

        const auto clutchSide = drive.shaftSpeed * driveline.gearbox.ratio(drive.gear);

        negatives += clutchSide < 0.0 ? 1 : 0;
        worst = std::min(worst, clutchSide);

        if (step <= 30 || step % 10 == 0)
        {
            std::printf("  %7d %11.4f %13.4f %10.4f %11.2f %11.4f %10.2f\n", step, drive.shaftSpeed, clutchSide,
                        drive.clutchPedal, torques->clutch, state.corners[0].wheelSpeed, drive.engineSpeed);
        }
    }

    std::printf("\n  ticks with a negative clutch-side speed, of 130: %d\n", negatives);
    std::printf("  most negative it got: %.4f rad/s\n", worst);
    std::printf("  `caught` would have read that as %.4f of the way to fully clamped.\n",
                std::clamp((std::abs(worst) - 0.40 * driveline.engine.idleSpeed) /
                               ((1.20 - 0.40) * driveline.engine.idleSpeed),
                           0.0, 1.0));
}

// **Does a car on a slope roll down it?** Asked before any creep number taken on a grade is believed,
// because the 45% take-up below climbs at an acceleration that accounts for the creep torque and the
// rolling resistance and leaves nothing at all for gravity. No driveline, no engine, no brakes: just a
// car put down on a hill in neutral.
TEST_CASE("a car in neutral on a hill", "[.creep-gravity]")
{
    const JoltGuard jolt;

    const auto vehicle = golfGtiMk7().value();

    std::printf("\n=== a car placed in neutral on a slope, nothing driving and nothing braking ===\n");
    std::printf("  %8s %12s %12s %12s %12s\n", "grade", "after 5 s", "speed", "dz", "dy");

    for (const auto percent : {0.0, 10.0, 20.0, 45.0})
    {
        const auto angle = gradeAngle(percent);
        const auto ground = creepGround(angle);
        const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
        REQUIRE(world.has_value());

        auto state = VehicleState{};
        const auto surface = sampleProvingGround(ground, 0.0, startStation).height;
        state.chassis.position = glm::dvec3(0.0, surface + designHeight(vehicle), startStation);
        state.chassis.orientation = glm::angleAxis(-angle, glm::dvec3(1.0, 0.0, 0.0));

        // Settled on the brakes so the starting point is a car standing still, then let go.
        for (auto step = 0; step < 1440; step++)
        {
            auto holding = VehicleInput{};
            holding.brake = 1.0;

            REQUIRE(stepVehicle(vehicle, state, holding, noDriveTorque, world.value(), tick).has_value());
        }

        REQUIRE(glm::length(state.chassis.linearVelocity) < 0.05);

        const auto from = state.chassis.position;

        for (auto step = 0; step < 1800; step++)
        {
            REQUIRE(stepVehicle(vehicle, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        const auto moved = state.chassis.position - from;

        std::printf("  %7.1f%% %12.4f %12.4f %12.4f %12.4f\n", percent, glm::length(glm::dvec3(moved.x, 0.0, moved.z)),
                    glm::length(state.chassis.linearVelocity), moved.z, moved.y);
    }

    std::printf("\n  A car in neutral on a 20%% slope reaches about 9.7 m/s in five seconds if gravity\n"
                "  reaches it along the surface. Anything much less than that and it does not.\n");
}

// The take-up on a grade steep enough that the commanded creep torque cannot possibly climb it: on a
// 45% slope this car needs about 124 N.m at the clutch and creep is commanding 30. If it climbs
// anyway, something other than the creep command is closing the clutch, and this is where that shows.
TEST_CASE("the take-up on a grade creep cannot climb", "[.creep-takeup]")
{
    const JoltGuard jolt;

    const auto angle = gradeAngle(45.0);
    const auto ground = creepGround(angle);
    const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
    REQUIRE(world.has_value());

    const auto run = runCreep(world.value(), ground, angle, 6.0, Script{}, golfGtiMk7Driveline());

    std::printf("\n=== 45%% grade, first six seconds, every fiftieth of a second ===\n");
    std::printf("  %7s %9s %9s %9s %9s %9s %9s %9s %6s\n", "t", "m/s", "travelled", "engine", "pedal", "creepCmd",
                "clutchT", "wheelT", "lock");

    for (const auto& sample : run.samples)
    {
        if (std::abs(sample.time * 50.0 - std::round(sample.time * 50.0)) > 0.5 * tick)
        {
            continue;
        }

        if (sample.time > 1.0 && std::abs(sample.time * 5.0 - std::round(sample.time * 5.0)) > 0.5 * tick)
        {
            continue;
        }

        std::printf("  %7.3f %9.4f %9.4f %9.2f %9.4f %9.2f %9.2f %9.1f %6s\n", sample.time, sample.speed,
                    sample.station - startStation, sample.engineSpeed, sample.clutchPedal, sample.creepCommand,
                    sample.clutchTorque, sample.wheelTorque, sample.locked ? "yes" : "no");
    }
}

TEST_CASE("creep against grade", "[.creep-grade]")
{
    const JoltGuard jolt;

    const auto driveline = golfGtiMk7Driveline();

    // **Read `[.creep-gravity]` before this table.** Every row of it is the same number, and the
    // reason is not the transmission: the vehicle model applies the tyre's load along world up rather
    // than along the contact normal, so no car in this model rolls down a hill at all. What this
    // measures is therefore that creep is *insensitive* to grade, and it cannot be otherwise until
    // that is fixed. A grade-authority sweep against creep torque lived here too and was deleted: it
    // could only ever print one number five times.
    std::printf("\n=== creep speed against grade ===\n");
    std::printf("  positive is uphill. `speed` is along the slope, averaged over the last four seconds\n");
    std::printf("  of a sixteen-second run. `travelled` is signed: negative is a car rolling back.\n\n");
    std::printf("  %8s %10s %10s %10s %10s %10s %8s\n", "grade", "m/s", "km/h", "travelled", "engine", "creepCmd",
                "locked");

    for (const auto percent : {-20.0, -15.0, -10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 45.0})
    {
        const auto angle = gradeAngle(percent);
        const auto ground = creepGround(angle);
        const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
        REQUIRE(world.has_value());

        const auto run = runCreep(world.value(), ground, angle, 16.0, Script{}, driveline);
        const auto settled = settledSpeed(run, 12.0);
        const auto& last = run.samples.back();

        std::printf("  %7.1f%% %10.4f %10.3f %10.3f %10.2f %10.2f %8s\n", percent, settled, settled * 3.6,
                    last.station - startStation, last.engineSpeed, last.creepCommand, last.locked ? "yes" : "no");
    }
}

TEST_CASE("creep is held off by the brake", "[.creep-brake]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    const auto driveline = golfGtiMk7Driveline();

    std::printf("\n=== held on the brake, in gear, for thirty seconds ===\n");
    std::printf("  The row that must not exist is one where the car is *held* and the clutch is still\n");
    std::printf("  slipping: that is a plate being cooked at a set of lights. Slip energy over 30 s is\n");
    std::printf("  the channel that says so — a few hundred joules is the initial take-up and nothing\n");
    std::printf("  after it; tens of thousands is a clutch slipping the whole time.\n\n");
    std::printf("  %8s %12s %12s %12s %12s %10s\n", "brake", "crept (m)", "speed", "creepCmd", "slipEnergy",
                "lock flips");

    for (const auto brake : {0.00, 0.02, 0.04, 0.049, 0.05, 0.06, 0.08, 0.10, 0.20, 0.50, 1.00})
    {
        auto script = Script{};
        script.brake = brake;

        const auto run = runCreep(world.value(), flat, 0.0, 30.0, script, driveline);
        const auto& last = run.samples.back();

        std::printf("  %8.2f %12.5f %12.6f %12.4f %12.3f %10d\n", brake, last.station - startStation, last.speed,
                    last.creepCommand, last.slipEnergy, lockTransitions(run, 1.0));
    }
}

TEST_CASE("the handover between creep and the throttle", "[.creep-hand]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    const auto driveline = golfGtiMk7Driveline();

    // Creep for eight seconds — long enough to be settled — then a quarter throttle for four, then
    // back off. A quarter rather than full: the question is whether the *transition* steps, and a
    // launch would drown it in the answer to a different question.
    auto script = Script{};
    script.throttle = 0.25;
    script.throttleFrom = 8.0;
    script.throttleTo = 12.0;

    const auto run = runCreep(world.value(), flat, 0.0, 18.0, script, driveline);

    std::printf("\n=== creep -> quarter throttle at 8 s -> creep again at 12 s ===\n");
    std::printf("  %8s %10s %10s %10s %10s %10s %8s\n", "t", "m/s", "engine", "pedal", "creepCmd", "clutchT", "locked");

    for (const auto& sample : run.samples)
    {
        const auto near = (sample.time > 7.6 && sample.time < 9.2) || (sample.time > 11.6 && sample.time < 13.2);
        const auto step = near ? 20.0 : 2.0;

        if (std::abs(sample.time * step - std::round(sample.time * step)) > 0.5 * tick)
        {
            continue;
        }

        std::printf("  %8.3f %10.4f %10.2f %10.4f %10.2f %10.2f %8s\n", sample.time, sample.speed, sample.engineSpeed,
                    sample.clutchPedal, sample.creepCommand, sample.clutchTorque, sample.locked ? "yes" : "no");
    }

    std::printf("\n  worst tick-to-tick clutch torque change:\n");
    std::printf("    settled creep,  6.0-8.0 s: %10.4f N.m\n", worstJump(run, 6.0, 8.0));
    std::printf("    throttle on,    8.0-9.0 s: %10.4f N.m\n", worstJump(run, 8.0, 9.0));
    std::printf("    settled drive, 10.5-12.0 s: %9.4f N.m\n", worstJump(run, 10.5, 12.0));
    std::printf("    throttle off,  12.0-13.0 s: %9.4f N.m\n", worstJump(run, 12.0, 13.0));
}

TEST_CASE("sixty seconds of creep", "[.creep-hunt]")
{
    const JoltGuard jolt;

    const auto flat = creepGround(0.0);
    const auto world = PhysicsWorld::create(generateProvingGround(flat).value());
    REQUIRE(world.has_value());

    const auto driveline = golfGtiMk7Driveline();
    const auto run = runCreep(world.value(), flat, 0.0, 60.0, Script{}, driveline);

    // The rate the clutch is turning into heat at, at each end of the run. A running total that has
    // stopped growing is a clutch that has finished engaging; one that keeps growing is a clutch
    // slipping by design, and creep is the operating state where that is expected.
    const auto rateOver = [&run](const double from, const double to)
    {
        auto first = 0.0;
        auto last = 0.0;
        auto seen = false;

        for (const auto& sample : run.samples)
        {
            if (sample.time < from || sample.time > to)
            {
                continue;
            }

            if (!seen)
            {
                first = sample.slipEnergy;
                seen = true;
            }

            last = sample.slipEnergy;
        }

        return (last - first) / (to - from);
    };

    std::printf("\n=== sixty seconds of creep on the flat ===\n");
    std::printf("  lock/slip transitions, whole run:  %d\n", lockTransitions(run, 0.0));
    std::printf("  lock/slip transitions after 3 s:   %d\n", lockTransitions(run, 3.0));
    std::printf("  settled speed, last ten seconds:   %.5f m/s\n", settledSpeed(run, 50.0));
    std::printf("  worst tick-to-tick clutch torque change after 3 s: %.5f N.m\n", worstJump(run, 3.0, 60.0));
    std::printf("\n  slip energy at  2 s: %10.3f J   rate over  0-2 s:  %9.4f W\n", run.samples[719].slipEnergy,
                rateOver(0.0, 2.0));
    std::printf("  slip energy at 60 s: %10.3f J   rate over 50-60 s: %9.4f W\n", run.samples.back().slipEnergy,
                rateOver(50.0, 60.0));
}
