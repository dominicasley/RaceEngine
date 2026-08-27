// How far the sprung mass actually leans, and where that lean comes from.
//
// A probe, not a criterion: it prints numbers and asserts almost nothing. It exists because "the
// body pitches and rolls too much but the car handles fine" is a claim about *attitude* and every
// figure this project has taken so far is about *load*, and the two are only the same number when
// the whole load transfer goes through the springs.
//
// Hidden behind a dot tag — `./EngineTests "[.body-attitude]"` — like every other probe here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
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
constexpr auto degrees = 180.0 / 3.14159265358979323846;

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

ProvingGroundDescriptor plate(const double size)
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
    state.chassis.position = glm::dvec3(0.0, 0.52, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / setup.corners.front().hardpoints.wheelRadius;
    }
}

} // namespace

TEST_CASE("body attitude: what the sprung mass does under load transfer", "[.body-attitude]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto& setup = built.value();

    const auto world = PhysicsWorld::create(generateProvingGround(plate(1200.0)).value());
    REQUIRE(world.has_value());

    // --- the static datum -----------------------------------------------------------------------
    auto state = VehicleState{};
    settle(setup, state, world.value(), 25.0, 600.0);

    auto restTravel = std::array<double, cornerCount>{};
    auto rollCentre = std::array<double, cornerCount>{};
    {
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());

        WARN("=== static ===");
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            restTravel[index] = stepped->corners[index].suspension.wheelTravel;
            rollCentre[index] = stepped->corners[index].suspension.rollCentreHeight;

            WARN("corner " << index << ": travel " << restTravel[index] * 1000.0 << " mm, roll centre "
                           << rollCentre[index] * 1000.0 << " mm, load "
                           << stepped->corners[index].forces.tireVertical << " N");
        }
        WARN("pitch " << stepped->telemetry.pitch * degrees << " deg, roll " << stepped->telemetry.roll * degrees
                      << " deg");
    }

    // --- braking: how far the nose goes down --------------------------------------------------
    //
    // Full pedal from 30 m/s with no electronics, sampled while the car is between 25 and 15 m/s so
    // that the initial pitch transient has passed and the wheels have not yet stopped.
    {
        settle(setup, state, world.value(), 30.0, 900.0);

        auto input = VehicleInput{};
        input.brake = 1.0;

        auto samples = 0;
        auto pitch = 0.0;
        auto longitudinal = 0.0;
        auto frontTravel = 0.0;
        auto rearTravel = 0.0;
        auto worstPitch = 0.0;

        for (auto step = 0; step < 3600; step++)
        {
            const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto speed = state.chassis.linearVelocity.z;
            worstPitch = std::max(worstPitch, std::abs(stepped->telemetry.pitch));

            if (speed < 25.0 && speed > 15.0)
            {
                pitch += stepped->telemetry.pitch;
                longitudinal += stepped->telemetry.acceleration.z;
                frontTravel += 0.5 * (stepped->corners[0].suspension.wheelTravel - restTravel[0] +
                                      stepped->corners[1].suspension.wheelTravel - restTravel[1]);
                rearTravel += 0.5 * (stepped->corners[2].suspension.wheelTravel - restTravel[2] +
                                     stepped->corners[3].suspension.wheelTravel - restTravel[3]);
                samples++;
            }

            if (speed < 15.0)
            {
                break;
            }
        }

        REQUIRE(samples > 0);

        const auto meanPitch = pitch / samples;
        const auto meanLongitudinal = longitudinal / samples;

        WARN("=== braking, full pedal, no assists ===");
        WARN("longitudinal " << meanLongitudinal / gravity << " g");
        WARN("pitch " << meanPitch * degrees << " deg, i.e. "
                      << meanPitch * degrees / std::abs(meanLongitudinal / gravity) << " deg/g");
        WARN("peak pitch over the whole stop " << worstPitch * degrees << " deg");
        WARN("front suspension " << (frontTravel / samples) * 1000.0 << " mm, rear "
                                 << (rearTravel / samples) * 1000.0 << " mm (positive is bump)");
    }

    // --- cornering: how far the body leans ------------------------------------------------------
    //
    // Held steering with a proportional-integral driver on the front axle, the same shape
    // HandlingTests uses, because a coasting car scrubs its speed away and measures a spiral.
    for (const auto steering : std::vector<double>{0.10, 0.20, 0.35, 0.60})
    {
        settle(setup, state, world.value(), 20.0, 600.0);

        auto input = VehicleInput{};
        input.steering = steering;

        auto drive = std::array<double, cornerCount>{};
        auto integral = 0.0;

        auto samples = 0;
        auto roll = 0.0;
        auto lateral = 0.0;
        auto travel = std::array<double, cornerCount>{};
        auto supportedWorst = static_cast<int>(cornerCount);

        for (auto step = 0; step < 3600; step++)
        {
            const auto error = 20.0 - glm::length(state.chassis.linearVelocity);
            integral = std::clamp(integral + error * tick, -4.0, 4.0);
            const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                                  setup.corners.front().hardpoints.wheelRadius / 2.0;
            drive = std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0};

            const auto stepped = stepVehicle(setup, state, input, drive, world.value(), tick);
            REQUIRE(stepped.has_value());

            if (step >= 3240)
            {
                roll += stepped->telemetry.roll;
                lateral += stepped->telemetry.acceleration.x;
                samples++;

                auto supported = 0;
                for (const auto& wheel : stepped->telemetry.wheels)
                {
                    supported += wheel.inContact ? 1 : 0;
                }
                supportedWorst = std::min(supportedWorst, supported);

                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    travel[index] += stepped->corners[index].suspension.wheelTravel - restTravel[index];
                }
            }
        }

        REQUIRE(samples > 0);

        const auto meanRoll = roll / samples;
        const auto meanLateral = lateral / samples;

        WARN("=== cornering, steering " << steering << " at 20 m/s ===");
        WARN("lateral " << std::abs(meanLateral / gravity) << " g, wheels on the ground " << supportedWorst);
        WARN("roll " << std::abs(meanRoll * degrees) << " deg, i.e. "
                     << std::abs(meanRoll * degrees) / std::max(std::abs(meanLateral / gravity), 1e-6) << " deg/g");
        WARN("travel FL " << (travel[0] / samples) * 1000.0 << " FR " << (travel[1] / samples) * 1000.0 << " RL "
                          << (travel[2] / samples) * 1000.0 << " RR " << (travel[3] / samples) * 1000.0 << " mm");
    }
}
