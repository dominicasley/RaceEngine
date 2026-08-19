#include <algorithm>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::CouplingMode;
using raceengine::CouplingSetup;
using raceengine::CouplingSides;
using raceengine::CouplingState;
using raceengine::stepCoupling;

namespace
{

constexpr auto tick = 1.0 / 360.0;

// Unit inertias on both sides, so the constraint torque is exactly half the driving torque plus
// whatever the slip costs. Nothing here integrates a speed: this is the machine on its own, driven
// by hand, which is the only way to put a torque exactly where a threshold is.
CouplingSides sides(const double drivingTorque, const double capacity, const double slipSpeed = 0.0)
{
    return CouplingSides{.drivingSpeed = slipSpeed,
                         .drivenSpeed = 0.0,
                         .drivingInertia = 1.0,
                         .drivenInertia = 1.0,
                         .drivingTorque = drivingTorque,
                         .drivenTorque = 0.0,
                         .capacity = capacity};
}

// The mode after every tick of a run, so a test can count the changes rather than only look at the
// end — hunting is invisible in a final state and obvious in the sequence.
std::vector<CouplingMode> run(const CouplingSetup& setup, CouplingState& state, const int ticks,
                              const auto& drivingTorqueAt)
{
    auto modes = std::vector<CouplingMode>{};

    for (auto step = 0; step < ticks; step++)
    {
        static_cast<void>(stepCoupling(setup, state, sides(drivingTorqueAt(step), 400.0), tick));
        modes.push_back(state.mode);
    }

    return modes;
}

std::size_t transitionsIn(const std::vector<CouplingMode>& modes)
{
    auto count = std::size_t{0};

    for (auto index = std::size_t{1}; index < modes.size(); index++)
    {
        count += modes[index] != modes[index - 1] ? std::size_t{1} : std::size_t{0};
    }

    return count;
}

// The shortest stretch the mode was held for, ignoring the run's own two ends: a stretch that begins
// before the run or continues past it says nothing about how long it would have been.
std::size_t shortestRun(const std::vector<CouplingMode>& modes)
{
    auto shortest = modes.size();
    auto length = std::size_t{0};
    auto seenTransition = false;

    for (auto index = std::size_t{1}; index < modes.size(); index++)
    {
        if (modes[index] == modes[index - 1])
        {
            length++;
            continue;
        }

        if (seenTransition)
        {
            shortest = std::min(shortest, length + 1);
        }

        seenTransition = true;
        length = 0;
    }

    return shortest;
}

} // namespace

TEST_CASE("a coupling carries what the constraint asks for, up to what it can hold", "[physics][coupling]")
{
    const auto setup = CouplingSetup{};
    auto state = CouplingState{};

    SECTION("slipping, it carries its capacity signed by which way the surfaces are sliding")
    {
        // Far past its capacity, so it cannot lock, and turning fast against its driven side.
        const auto forward = stepCoupling(setup, state, sides(4000.0, 400.0, 20.0), tick);
        REQUIRE_FALSE(forward.locked);
        REQUIRE(forward.torque == Catch::Approx(400.0));

        const auto backward = stepCoupling(setup, state, sides(-4000.0, 400.0, -20.0), tick);
        REQUIRE_FALSE(backward.locked);
        REQUIRE(backward.torque == Catch::Approx(-400.0));
    }

    SECTION("the torque a lock would need is what the two sides moving as one costs")
    {
        // Equal inertias and 200 Nm on one side: half of it has to cross the coupling for the pair
        // to accelerate together, and the slip term is nothing because they are already in step.
        const auto solution = stepCoupling(setup, state, sides(200.0, 400.0), tick);

        REQUIRE(solution.required == Catch::Approx(100.0));
        REQUIRE(solution.slipSpeed == Catch::Approx(0.0));
    }

    SECTION("and the slip it has to remove is part of that cost")
    {
        // The same, plus a rad/s of slip to take out over one tick through the reduced inertia of
        // 0.5 kg.m^2 — 0.5 * 1 * 360 = 180 Nm on top. Without this term a lock holds whatever speed
        // difference it engaged at for ever and is not a lock.
        const auto solution = stepCoupling(setup, state, sides(200.0, 4000.0, 1.0), tick);

        REQUIRE(solution.required == Catch::Approx(100.0 + 180.0));
    }
}

TEST_CASE("a coupling locks when it can and breaks when it cannot", "[physics][coupling]")
{
    const auto setup = CouplingSetup{};

    SECTION("it locks once the torque is inside the lock threshold and the dwell has run")
    {
        auto state = CouplingState{};

        // 200 Nm of driving torque against a capacity of 400 costs the constraint 100, well inside
        // the 320 the lock threshold sits at.
        const auto modes = run(setup, state, 360, [](const int) { return 200.0; });

        REQUIRE(state.mode == CouplingMode::Locked);
        // And not on the first tick: the slip dwell is what it has to sit through first.
        REQUIRE(modes.front() == CouplingMode::Slipping);
        REQUIRE(transitionsIn(modes) == std::size_t{1});

        const auto locked = std::find(modes.begin(), modes.end(), CouplingMode::Locked);
        REQUIRE(static_cast<double>(std::distance(modes.begin(), locked) + 1) * tick >= setup.slipDwell);
    }

    SECTION("and it breaks the moment it is asked for more than it can hold")
    {
        auto state = CouplingState{};

        static_cast<void>(run(setup, state, 360, [](const int) { return 200.0; }));
        REQUIRE(state.mode == CouplingMode::Locked);

        // Its lock dwell is long since served, so 4000 Nm breaks it on the first tick and it stays
        // broken.
        const auto modes = run(setup, state, 360, [](const int) { return 4000.0; });

        REQUIRE(modes.front() == CouplingMode::Slipping);
        REQUIRE(transitionsIn(modes) == std::size_t{0});
        // Broken, it carries its capacity and not the 2000 the constraint wanted.
        REQUIRE(state.torque == Catch::Approx(400.0));
    }

    SECTION("a coupling held locked by its dwell still carries no more than its capacity")
    {
        auto state = CouplingState{};
        static_cast<void>(run(setup, state, 360, [](const int) { return 200.0; }));

        // Freshly locked, so the next tick is inside the lock dwell and it may not break yet. The
        // clamp is what stops that being a licence to transmit whatever was asked for: the dwell
        // buys time against hunting, not friction.
        state.dwell = 0.0;
        const auto solution = stepCoupling(setup, state, sides(4000.0, 400.0), tick);

        REQUIRE(solution.locked);
        REQUIRE(solution.required == Catch::Approx(2000.0));
        REQUIRE(solution.torque == Catch::Approx(400.0));
    }
}

TEST_CASE("a coupling does not hunt", "[physics][coupling][hysteresis]")
{
    // The failure the whole state machine exists to avoid. It shows in telemetry as a buzz on the
    // torque trace at the timestep's frequency rather than as anything a driver could name, which is
    // why it is asserted on the mode sequence and not on where the run ended up.
    const auto setup = CouplingSetup{};

    SECTION("a torque dithering across the lock threshold locks once and stays locked")
    {
        auto state = CouplingState{};

        // The lock threshold is 0.8 * 400 = 320 of constraint torque, which is 640 of driving
        // torque. Straddle it every tick.
        const auto modes = run(setup, state, 720, [](const int step) { return step % 2 == 0 ? 600.0 : 680.0; });

        REQUIRE(transitionsIn(modes) == std::size_t{1});
        REQUIRE(state.mode == CouplingMode::Locked);
    }

    SECTION("and one dithering across the break threshold breaks once and stays broken")
    {
        auto state = CouplingState{};
        static_cast<void>(run(setup, state, 360, [](const int) { return 200.0; }));
        REQUIRE(state.mode == CouplingMode::Locked);

        // The break threshold is the capacity itself, 400, which is 800 of driving torque. Once
        // broken it cannot come back, because coming back needs 320 and this never goes near it —
        // which is the whole of what two thresholds buy.
        const auto modes = run(setup, state, 720, [](const int step) { return step % 2 == 0 ? 760.0 : 840.0; });

        REQUIRE(transitionsIn(modes) == std::size_t{1});
        REQUIRE(state.mode == CouplingMode::Slipping);
    }

    SECTION("a torque swinging past both thresholds is held off by the dwell, in both directions")
    {
        auto state = CouplingState{};

        // Two thresholds do nothing here: this crosses both of them every single tick. Only the
        // dwell is left, and it has to work locking and unlocking alike.
        const auto modes = run(setup, state, 720, [](const int step) { return step % 2 == 0 ? 100.0 : 4000.0; });

        const auto changes = transitionsIn(modes);
        REQUIRE(changes > std::size_t{0});

        // A dwell of 0.02 s at 360 Hz is eight ticks, so 720 of them hold at most ninety stretches
        // rather than the 719 changes an undwelled machine manages.
        REQUIRE(changes <= std::size_t{90});
        REQUIRE(static_cast<double>(shortestRun(modes)) * tick >= setup.lockDwell);
    }

    SECTION("and it is the dwell doing it: taken away, the hunting comes straight back")
    {
        auto state = CouplingState{};
        const auto impatient =
            CouplingSetup{.lockFraction = 0.8, .lockSlipSpeed = 1.0, .slipDwell = 0.0, .lockDwell = 0.0};

        const auto modes = run(impatient, state, 720, [](const int step) { return step % 2 == 0 ? 100.0 : 4000.0; });

        REQUIRE(transitionsIn(modes) > std::size_t{700});
    }
}
