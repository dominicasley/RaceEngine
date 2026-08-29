// What longitudinal recession is worth on a straight-line stop: `./EngineTests "[.recession]"`.
//
// The A/B the mechanism shipped with, printed rather than argued. **No car states a coefficient**
// — the only published figures are Heissing/Ersoy's design targets (front 4-8 mm/kN of braking
// force; rear 8-16 mm per g, a different unit) — so the stated car here is built through the setup
// sheet itself (`front.recession 6`, `rear.recession 10`, the suggested A/B worked out beside the
// Golf's compliance figures), which proves the whole sheet-to-car path in the same run that
// measures the mechanism. Recession never reaches the tyre's force law or the linkage Jacobians.
// It DOES reach the steering weight — corrected 2026-08-29 night, the seat found it first: the
// rack reads the tyre resultant's moment about the kingpin at the patch the solve reports, and
// the patch recedes while the kingpin does not, so trail grows under braking and shrinks on
// power. This probe's straight-line stop measures the position couplings — the hub and sampled
// patch moving fore-aft, the vertical load's pitch lever with them; the steering signature is in
// the seat traces (docs/suspension-fidelity-brief.md, the A/B entry).
//
// The stop distance is printed WITH its caveat: the cold 100-0's recorded noise floor is 1.47 m
// and non-monotonic, so a distance delta here is not a result unless it dwarfs that. The
// per-corner recession peaks are the honest channel.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::applyVehicleTune;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::parseVehicleTune;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto radiansToDegrees = 57.29577951308232;

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

ProvingGroundDescriptor plate(const double size = 1200.0)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

struct StopResult
{
    double distance = 0.0;
    // Peak |pitch| over the stop, degrees — magnitude, so the printout does not depend on which
    // sign this model's pitch convention gives a dive.
    double peakDive = 0.0;
    // Most-negative (rearward) recession over the stop, front-left and rear-left, metres.
    double frontRecession = 0.0;
    double rearRecession = 0.0;
};

// The 0.35-pedal 100-0 the brake probe already uses: hard enough that the front axle carries about
// five kilonewtons a wheel, no assists, a step pedal — the same input to both cars, which is all an
// A/B asks of it.
StopResult stopFromHundred(const VehicleSetup& setup, const PhysicsWorld& world)
{
    // **The plate runs z from 0 to its length and x from -width/2 to +width/2** — the recorded
    // trap: a fixture that assumes it is centred starts the car in mid-air and reports a car with
    // no brakes.
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, 400.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    const auto speed = 100.0 / 3.6;
    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
    }

    auto input = VehicleInput{};
    input.brake = 0.35;

    auto result = StopResult{};
    const auto from = state.chassis.position.z;

    for (auto step = 0; step < 7200; step++)
    {
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick);
        REQUIRE(stepped.has_value());

        result.peakDive = std::max(result.peakDive, std::abs(stepped->telemetry.pitch) * radiansToDegrees);
        result.frontRecession = std::min(result.frontRecession, stepped->telemetry.wheels[0].complianceRecession);
        result.rearRecession = std::min(result.rearRecession, stepped->telemetry.wheels[2].complianceRecession);

        if (glm::length(state.chassis.linearVelocity) < 0.5)
        {
            break;
        }
    }

    result.distance = state.chassis.position.z - from;

    // The fixture asserts its own precondition: a car still moving after twenty seconds did not
    // measure a stop.
    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.5);

    return result;
}

} // namespace

TEST_CASE("what longitudinal recession is worth on a straight-line stop", "[.recession]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto shipped = golfGtiMk7();
    REQUIRE(shipped.has_value());

    // The stated car is built THROUGH the sheet, so this run proves parse -> apply -> corner ->
    // solve -> telemetry in one piece — the "prove a seat knob moves the car before handing it
    // over" rule, applied before there is a seat to hand it to.
    auto stated = shipped.value();
    const auto tune = parseVehicleTune("front.recession 6\nrear.recession 10\n");
    REQUIRE(tune.has_value());
    applyVehicleTune(tune.value(), stated);

    std::printf("\n=== longitudinal recession against a 0.35-pedal 100-0 stop ===\n");

    const auto zeroed = stopFromHundred(shipped.value(), world.value());
    const auto receded = stopFromHundred(stated, world.value());

    std::printf("\n  %-34s %10s %11s %14s %14s\n", "", "stop [m]", "|dive| [deg]", "recession F", "recession R");
    std::printf("  %-34s %10.2f %11.3f %11.1f mm %11.1f mm\n", "no recession - every shipped car", zeroed.distance,
                zeroed.peakDive, zeroed.frontRecession * 1000.0, zeroed.rearRecession * 1000.0);
    std::printf("  %-34s %10.2f %11.3f %11.1f mm %11.1f mm\n", "sheet: front 6 / rear 10 mm/kN", receded.distance,
                receded.peakDive, receded.frontRecession * 1000.0, receded.rearRecession * 1000.0);

    std::printf("\n  The distance delta is %+.2f m against a recorded 1.47 m non-monotonic noise floor —\n"
                "  not a result unless it dwarfs that. The recession columns are the mechanism itself;\n"
                "  the zeroed car must read exactly 0.0 in both.\n",
                receded.distance - zeroed.distance);

    // The zeroed car's channel is the inertness statement in the same units the trace reports.
    REQUIRE(zeroed.frontRecession == 0.0);
    REQUIRE(zeroed.rearRecession == 0.0);

    // And the stated car's front peak is the sanity band the hand arithmetic predicts: ~5 kN a
    // wheel at 6 mm/kN is around 30 mm rearward. Wide on purpose — this pins "the mechanism is
    // live and the right order", not a number.
    REQUIRE(receded.frontRecession < -0.010);
    REQUIRE(receded.frontRecession > -0.060);
}
