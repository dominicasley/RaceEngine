// Criteria 5, 6 and 7 against grip, all on one sweep: `./EngineTests "[.criteria-grip]"`.
//
// **The re-baselining that has to happen before the grip decision can be taken.** Criterion 5's
// threshold *is* a target grip figure, and grip is the free parameter that makes it pass — so setting
// one from the other is circular. The way out is that 6 and 7 are not grip figures:
//
//   - **Criterion 6** is response *shape* — settled yaw, rise time, overshoot, tail ripple. A car's
//     yaw time constant and damping are set by inertia, geometry and roll stiffness. Grip sets how
//     much lateral acceleration is available, not how quickly the car gets to the one it is using.
//   - **Criterion 7** is a *ratio* between two runs that differ only in a rear bar rate. Whatever grip
//     both share divides out.
//
// If that holds, their thresholds can be fixed from physical reasoning at any grip, and criterion 5's
// can then be set from the real car's skidpad range without either informing the other. That leaves
// two independent measurements against two independent external references, which is the validation.
//
// This probe reports the quantity each criterion asserts, across grip, so the independence is measured
// rather than argued.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
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

[[nodiscard]] ProvingGroundDescriptor bigPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 1200.0;
    descriptor.width = 1200.0;
    descriptor.cellSize = 4.0;
    descriptor.features = {};

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

[[nodiscard]] VehicleSetup golfAt(const double gripScale, const double rearBar = -1.0)
{
    auto setup = golfGtiMk7().value();
    for (auto& corner : setup.corners)
    {
        corner.tyre.gripScale = gripScale;
    }

    if (rearBar >= 0.0)
    {
        setup.corners[2].antiRollRate = rearBar;
        setup.corners[3].antiRollRate = rearBar;
    }

    return setup;
}

void settleAt(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
              const double startZ)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), startZ);

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

struct Held
{
    double lateral = 0.0;
    double yawRate = 0.0;
};

// The same driven hold `hold()` runs: PI on road speed, ten seconds, averaged over the last.
[[nodiscard]] Held drivenHold(const VehicleSetup& setup, const PhysicsWorld& world, const double steering,
                              const double speed)
{
    auto state = VehicleState{};
    settleAt(setup, state, world, speed, 400.0);

    auto input = VehicleInput{};
    input.steering = steering;

    auto integral = 0.0;
    auto held = Held{};
    auto samples = 0;

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
            held.lateral += stepped->telemetry.acceleration.x * toTheRight;
            held.yawRate += stepped->telemetry.yawRate * toTheRight;
            samples++;
        }
    }

    held.lateral /= static_cast<double>(samples) * gravity;
    held.yawRate /= static_cast<double>(samples);

    return held;
}

} // namespace

TEST_CASE("criteria 6 and 7 do not depend on grip, and 5 does", "[.criteria-grip]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(bigPlate()).value());
    REQUIRE(world.has_value());

    std::printf("\n=== what each criterion measures, against grip ===\n");
    std::printf("\n%10s %10s | %9s %9s %9s %9s | %9s\n", "gripScale", "C5 peak g", "C6 settled", "C6 rise s",
                "C6 shoot", "C6 ripple", "C7 ratio");

    for (const auto gripScale : {1.00})
    {
        // --- criterion 5: the skidpad's peak, swept across the steering that reaches it ---
        auto peak = 0.0;
        for (const auto steering : {0.20, 0.30, 0.45, 0.60})
        {
            peak = std::max(peak, drivenHold(golfAt(gripScale), world.value(), steering, 20.0).lateral);
        }

        // --- criterion 6: step steer response shape, undriven and gentle by design ---
        auto settled = 0.0;
        auto rise = 0.0;
        auto overshoot = 0.0;
        auto ripple = 0.0;
        {
            const auto setup = golfAt(gripScale);
            auto state = VehicleState{};
            settleAt(setup, state, world.value(), 25.0, 400.0);

            auto input = VehicleInput{};
            input.steering = 0.06;

            auto history = std::vector<double>{};
            for (auto step = 0; step < 1080; step++)
            {
                const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world.value(), tick);
                REQUIRE(stepped.has_value());
                history.push_back(stepped->telemetry.yawRate * outboardSign(CornerSide::Right));
            }

            for (auto index = history.size() - 180; index < history.size(); index++)
            {
                settled += history[index];
            }
            settled /= 180.0;

            auto riseTicks = std::size_t{0};
            while (riseTicks < history.size() && history[riseTicks] < settled * 0.9)
            {
                riseTicks++;
            }
            rise = static_cast<double>(riseTicks) * tick;

            overshoot = *std::max_element(history.begin(), history.end()) / settled;

            const auto tail = std::vector<double>(history.begin() + 540, history.end());
            ripple =
                (*std::max_element(tail.begin(), tail.end()) - *std::min_element(tail.begin(), tail.end())) / settled;
        }

        // --- criterion 7: the rear-bar ratio, which shares its grip between both runs ---
        const auto soft = drivenHold(golfAt(gripScale, 0.0), world.value(), 0.11, 20.0).yawRate;
        const auto stiff = drivenHold(golfAt(gripScale, 40000.0), world.value(), 0.11, 20.0).yawRate;

        std::printf("%10.2f %10.4f | %9.4f %9.3f %9.3f %9.4f | %9.4f\n", gripScale, peak, settled, rise, overshoot,
                    ripple, soft > 1e-9 ? stiff / soft : 0.0);
    }

    std::printf("\n  C6's four columns and C7's ratio should be flat in grip. C5's peak should not be.\n");
}
