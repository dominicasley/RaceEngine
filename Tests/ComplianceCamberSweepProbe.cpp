// What lateral-force compliance camber is worth on the skidpad: `./EngineTests "[.compliance-camber]"`.
//
// The A/B the mechanism shipped with, printed rather than argued. The Golf's front axle states
// 0.17 deg/kN (Kawata, Kouno & Sakuma, Trans. JSME 89(919) 2023, Table 2 — the same K&C campaign as
// the compliance steer); zeroing `lateralForceCamber` is the car before it. Camber never reaches the
// tyre's force law in this model — no camber thrust, no camber-dependent grip — so what this table
// measures is the whole of the coupling that exists: the tilted spin axis moving the sampled contact
// patch and the load path's lever arm.
//
// The steering angles are the skidpad criterion's own peak lattice (0.30 to 0.45 at the 0.02
// spacing, plus its neighbours), because the criterion passes at 0.9009 g against a 0.90 floor —
// a 0.1% margin — and the honest question is whether this mechanism moves that number, not whether
// it moves an easier one.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::CornerSide;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
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
constexpr auto gravity = 9.80665;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;

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

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ)
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

// The skidpad criterion's own driven hold, carried across exactly as `[.bar-sweep]` carried it, so
// this measures the quantity the criterion asserts rather than a near relative of it.
double hold(const VehicleSetup& setup, const PhysicsWorld& world, const double steering, const double speed)
{
    auto state = VehicleState{};
    settle(setup, state, world, speed, 400.0);

    auto input = VehicleInput{};
    input.steering = steering;

    auto integral = 0.0;
    auto lateral = 0.0;
    auto samples = 0;
    auto slowest = std::numeric_limits<double>::max();
    auto fastest = 0.0;

    for (auto step = 0; step < 3600; step++)
    {
        const auto error = speed - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;
        const auto drive = std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0};

        const auto stepped = stepVehicle(setup, state, input, drive, world, tick);
        REQUIRE(stepped.has_value());

        if (step >= 3240)
        {
            constexpr auto toTheRight = outboardSign(CornerSide::Right);
            const auto carried = glm::length(state.chassis.linearVelocity);

            lateral += stepped->telemetry.acceleration.x * toTheRight;
            slowest = std::min(slowest, carried);
            fastest = std::max(fastest, carried);
            samples++;
        }
    }

    // The fixture asserts its own precondition: a hold that did not hold its speed is a spiral, and
    // every number it reports describes a transient rather than a limit.
    CAPTURE(steering, speed, slowest, fastest);
    REQUIRE(slowest > 0.9 * speed);
    REQUIRE(fastest < 1.1 * speed);

    return lateral / (static_cast<double>(samples) * gravity);
}

} // namespace

TEST_CASE("what compliance camber is worth on the skidpad", "[.compliance-camber]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    // The criterion's peak lattice, both cars over the same angles.
    const auto angles = std::vector<double>{0.30, 0.34, 0.36, 0.37, 0.38, 0.40, 0.45};

    const auto sweep = [&world, &angles](const VehicleSetup& setup, const char* label)
    {
        auto peak = 0.0;
        auto at = 0.0;

        std::printf("\n  %s\n", label);
        for (const auto steering : angles)
        {
            const auto measured = hold(setup, world.value(), steering, 20.0);
            std::printf("    steering %.3f -> %.4f g\n", steering, measured);
            std::fflush(stdout);

            if (measured > peak)
            {
                peak = measured;
                at = steering;
            }
        }

        std::printf("    peak %.4f g at %.3f\n", peak, at);

        return peak;
    };

    std::printf("\n=== lateral-force compliance camber against the skidpad ===\n");

    auto stated = golfGtiMk7();
    REQUIRE(stated.has_value());

    auto zeroed = stated.value();
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        zeroed.corners[index].lateralForceCamber = 0.0;
    }

    const auto with = sweep(stated.value(), "front 0.17 deg/kN, as the car states it");
    const auto without = sweep(zeroed, "compliance camber zeroed - the car before it");

    std::printf("\n  stated %.4f g, zeroed %.4f g, difference %+.4f g.\n", with, without, with - without);
    std::printf("  The criterion's band is 0.90-0.95 g and its margin before this mechanism was 0.1%%.\n");
}
