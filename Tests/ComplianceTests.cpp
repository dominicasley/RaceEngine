#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::cornerCount;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::placeDriveline;
using raceengine::placeholderDriveline;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::VehicleInput;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr std::array<double, cornerCount> noRoadTorque{};
constexpr std::array<double, cornerCount> wheelInertia{1.2, 1.2, 1.2, 1.2};

VehicleInput driving(const double throttle, const std::int32_t gear)
{
    auto input = VehicleInput{};
    input.throttle = throttle;
    input.gear = gear;

    return input;
}

// The shaft's own numbers each tick. The wheels are held, which is what isolates the element: a car
// under it would answer with its own mass and the question here is what the shaft does.
struct Shunt
{
    std::vector<double> axleTorque;
    std::vector<double> windUp;
};

Shunt lift(const DrivelineSetup& setup, const double wheelSpeed, const std::int32_t gear, const int settleSteps,
           const int recordSteps)
{
    auto state = DrivelineState{};
    startEngine(setup, state);
    state.engineSpeed = std::max(setup.engine.idleSpeed, wheelSpeed * setup.gearbox.reduction(gear));
    placeDriveline(setup, state, wheelSpeed);

    const auto speeds = std::array<double, cornerCount>{wheelSpeed, wheelSpeed, wheelSpeed, wheelSpeed};

    // Loaded, so there is wind-up in the shaft to be released.
    for (auto step = 0; step < settleSteps; step++)
    {
        REQUIRE(stepDriveline(setup, state, speeds, wheelInertia, noRoadTorque, driving(1.0, gear), tick).has_value());
    }

    auto shunt = Shunt{};
    shunt.axleTorque.reserve(static_cast<std::size_t>(recordSteps));
    shunt.windUp.reserve(static_cast<std::size_t>(recordSteps));

    // And the throttle dropped on it, which is the transient A7 exists to produce.
    for (auto step = 0; step < recordSteps; step++)
    {
        const auto torques = stepDriveline(setup, state, speeds, wheelInertia, noRoadTorque, driving(0.0, gear), tick);
        REQUIRE(torques.has_value());

        shunt.axleTorque.push_back(torques->wheel[0] + torques->wheel[1]);
        shunt.windUp.push_back(torques->windUp);
    }

    return shunt;
}

// The largest excursion either side of the settled value over a window, which is the envelope a
// decay is judged on. A peak-to-peak rather than a peak: an oscillation about a non-zero mean is
// what a driveline settling under load actually does.
[[nodiscard]] double envelope(const std::vector<double>& trace, const std::size_t from, const std::size_t to)
{
    const auto first = trace.begin() + static_cast<std::ptrdiff_t>(from);
    const auto last = trace.begin() + static_cast<std::ptrdiff_t>(std::min(to, trace.size()));

    const auto [low, high] = std::minmax_element(first, last);

    return *high - *low;
}

} // namespace

TEST_CASE("a shaft under load winds up, and what it carries is what the box put in", "[physics][driveline][compliance]")
{
    const auto setup = placeholderDriveline();

    auto state = DrivelineState{};
    startEngine(setup, state);
    state.engineSpeed = 300.0;
    placeDriveline(setup, state, 40.0);

    const auto speeds = std::array<double, cornerCount>{40.0, 40.0, 40.0, 40.0};

    auto last = decltype(stepDriveline(setup, state, speeds, wheelInertia, noRoadTorque, driving(1.0, 3), tick)){};
    for (auto step = 0; step < 1440; step++)
    {
        last = stepDriveline(setup, state, speeds, wheelInertia, noRoadTorque, driving(1.0, 3), tick);
        REQUIRE(last.has_value());
    }

    // A spring at rest transmits exactly what is put into it — the wind-up is where the difference
    // went while it was building, and there is nowhere for it to go once it has stopped moving. This
    // is the property that says the element is a shaft rather than a loss.
    REQUIRE(last->shaftTorque == Catch::Approx(last->gearbox).epsilon(1e-6));
    REQUIRE(state.shaftSpeed == Catch::Approx(40.0 * setup.gearbox.finalDrive).epsilon(1e-6));

    // And it is wound in the direction it is being driven, by the angle the rate says.
    REQUIRE(state.windUp > 0.0);
    REQUIRE(state.windUp == Catch::Approx(last->gearbox / setup.compliance.stiffness).epsilon(1e-6));
}

TEST_CASE("driveline oscillation decays rather than sustaining", "[physics][driveline][compliance]")
{
    const auto setup = placeholderDriveline();

    // Sixty seconds at 360 Hz, which is the window the acceptance criterion names. The lift happens
    // at the start of it, so the first second holds the shunt and the last holds whatever is left.
    constexpr auto seconds = 60;
    constexpr auto steps = seconds * 360;

    const auto shunt = lift(setup, 40.0, 3, 1440, steps);

    const auto opening = envelope(shunt.axleTorque, 0, 360);
    const auto second = envelope(shunt.axleTorque, 360, 720);
    const auto closing =
        envelope(shunt.axleTorque, static_cast<std::size_t>(steps) - 360, static_cast<std::size_t>(steps));

    SECTION("there is a shunt to decay in the first place")
    {
        // A lift that produces nothing is not evidence of a damped driveline, it is evidence of a
        // driveline that is not there. The criterion is only meaningful above this.
        REQUIRE(opening > 100.0);
    }

    SECTION("and it is gone by the end of the minute")
    {
        REQUIRE(second < opening);
        // Three orders of magnitude down on the shunt that started it, which is decay rather than
        // "slow to diverge" — the distinction the brief asks to be verified rather than assumed.
        REQUIRE(closing < opening / 1000.0);
        REQUIRE(closing < 1.0);
    }

    SECTION("and every value in it stayed finite")
    {
        for (const auto torque : shunt.axleTorque)
        {
            REQUIRE(std::isfinite(torque));
        }

        // The guard is a guard, not a working part: a shaft that reaches it in ordinary driving has
        // turned into a torque limiter, which is what a stiffness an order of magnitude too soft did.
        for (const auto twist : shunt.windUp)
        {
            REQUIRE(std::abs(twist) < setup.compliance.maximumTwist);
        }
    }
}

TEST_CASE("the compliance switches off to exactly the driveline that was there before",
          "[physics][driveline][compliance]")
{
    // `enabled = false` is what makes the element provable: it must reproduce the rigid arithmetic
    // tick for tick rather than approximately, which is the same discipline the auto-exposure meter's
    // flat-average case is held to.
    auto rigid = placeholderDriveline();
    rigid.compliance.enabled = false;

    auto state = DrivelineState{};
    startEngine(rigid, state);
    state.engineSpeed = 300.0;

    const auto speeds = std::array<double, cornerCount>{40.0, 40.0, 40.0, 40.0};

    for (auto step = 0; step < 720; step++)
    {
        const auto torques = stepDriveline(rigid, state, speeds, wheelInertia, noRoadTorque, driving(1.0, 3), tick);
        REQUIRE(torques.has_value());

        // The shaft is the wheels and the twist is nothing, every tick, with no second code path
        // keeping them in step by hand.
        REQUIRE(state.shaftSpeed == 40.0 * rigid.gearbox.finalDrive);
        REQUIRE(state.windUp == 0.0);
        REQUIRE(torques->shaftTorque == torques->gearbox);
    }
}
