#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::advanceRevLimiter;
using raceengine::clutchCapacity;
using raceengine::cornerCount;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::DriverIntent;
using raceengine::engineTorque;
using raceengine::GearRange;
using raceengine::golfGtiMk7Driveline;
using raceengine::operateTransmission;
using raceengine::placeholderAutomatic;
using raceengine::placeholderDriveline;
using raceengine::revMatchThrottle;
using raceengine::ShiftPhase;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::TransmissionOperation;
using raceengine::TransmissionState;
using raceengine::VehicleInput;

namespace
{

constexpr auto tick = 1.0 / 360.0;

// No road under the wheels: every case here holds its wheel speeds by hand, so there is no tire
// reaction to feed back and none of what it would change is the question being asked.
constexpr std::array<double, cornerCount> noRoadTorque{};

// What one tick of the driveline said, reduced to the channels a shift is judged in.
struct Sample
{
    double axleTorque = 0.0;
    // What left the *box*, which is where a torque interrupt is a fact rather than a consequence:
    // once the shaft below it is compliant, the wheels go on feeling the wind-up unwind through the
    // whole neutral window and never see a zero at all.
    double gearboxTorque = 0.0;
    double clutchTorque = 0.0;
    double engineSpeed = 0.0;
    double referredInertia = 0.0;
    double slipEnergy = 0.0;
    double lockupApply = 0.0;
    std::int32_t gear = 0;
    ShiftPhase phase = ShiftPhase::Engaged;
    bool fuelCut = false;
};

// A driveline with its wheels held at a speed of the caller's choosing, which is the rig every shift
// question wants: what the box does is a property of the box, and putting a car under it only adds a
// second thing moving.
struct Bench
{
    DrivelineSetup setup = placeholderDriveline();
    DrivelineState state{};
    std::array<double, cornerCount> speeds{};
    std::array<double, cornerCount> inertias{1.2, 1.2, 1.2, 1.2};
    double deltaTime = tick;

    void start(const std::int32_t gear, const double wheelSpeed)
    {
        speeds = {wheelSpeed, wheelSpeed, wheelSpeed, wheelSpeed};

        startEngine(setup, state);
        state.engineSpeed = std::max(setup.engine.idleSpeed, wheelSpeed * setup.gearbox.reduction(gear));
    }

    Sample step(const VehicleInput& input)
    {
        const auto torques = stepDriveline(setup, state, speeds, inertias, noRoadTorque, input, deltaTime);
        REQUIRE(torques.has_value());

        return Sample{.axleTorque = torques->wheel[0] + torques->wheel[1],
                      .gearboxTorque = torques->gearbox,
                      .clutchTorque = torques->clutch,
                      .engineSpeed = state.engineSpeed,
                      .referredInertia = torques->referredInertia,
                      .slipEnergy = torques->slipEnergy,
                      .lockupApply = state.coupling.lockupApply,
                      .gear = torques->gear,
                      .phase = torques->shiftPhase,
                      .fuelCut = torques->fuelCut};
    }

    std::vector<Sample> run(const VehicleInput& input, const int steps)
    {
        auto samples = std::vector<Sample>{};
        samples.reserve(static_cast<std::size_t>(steps));

        for (auto step = 0; step < steps; step++)
        {
            samples.push_back(this->step(input));
        }

        return samples;
    }
};

VehicleInput driving(const double throttle, const std::int32_t gear)
{
    auto input = VehicleInput{};
    input.throttle = throttle;
    input.gear = gear;

    return input;
}

// How long the gearbox passed nothing at all, in ticks, counting the first run of exact zeros.
//
// Measured at the box and not at the wheels, and that is the whole of what a compliant driveline
// changed about this question. An open gearbox is an exact zero and always was; what the *wheels*
// see through that window is the shaft giving back what it stored, which decays over some tens of
// milliseconds and is never once equal to zero. Counting zeros at the wheels therefore returns 0 on
// a car whose interrupt is working perfectly — which is exactly how it read.
[[nodiscard]] int interruptTicks(const std::vector<Sample>& samples)
{
    auto first = std::size_t{0};
    while (first < samples.size() && samples[first].gearboxTorque != 0.0)
    {
        first++;
    }

    auto last = first;
    while (last < samples.size() && samples[last].gearboxTorque == 0.0)
    {
        last++;
    }

    return static_cast<int>(last - first);
}

// Every field, compared exactly. `std::memcmp` would read the padding between them, which nothing
// initialises and nothing is entitled to compare.
[[nodiscard]] bool identical(const DrivelineState& a, const DrivelineState& b)
{
    return a.engineSpeed == b.engineSpeed && a.gear == b.gear && a.targetGear == b.targetGear &&
           a.shiftFrom == b.shiftFrom && a.shiftPhase == b.shiftPhase && a.shiftTimer == b.shiftTimer &&
           a.fuelCut == b.fuelCut && a.engine == b.engine && a.clutchPedal == b.clutchPedal &&
           a.idleIntegral == b.idleIntegral && a.slipEnergy == b.slipEnergy &&
           a.coupling.lockupApply == b.coupling.lockupApply && a.coupling.coupling.mode == b.coupling.coupling.mode &&
           a.coupling.coupling.dwell == b.coupling.coupling.dwell &&
           a.coupling.coupling.torque == b.coupling.coupling.torque &&
           a.differentials[0].transfer == b.differentials[0].transfer &&
           a.differentials[0].capacity == b.differentials[0].capacity &&
           a.differentials[1].transfer == b.differentials[1].transfer &&
           a.differentials[1].capacity == b.differentials[1].capacity;
}

// Transitions of the fuel cut per second of simulated time, which is the only honest way to compare
// a limiter against itself across two timesteps.
[[nodiscard]] double chatterRate(const std::vector<Sample>& samples, const double deltaTime, const int skip)
{
    auto transitions = 0;
    for (auto index = static_cast<std::size_t>(skip) + 1; index < samples.size(); index++)
    {
        if (samples[index].fuelCut != samples[index - 1].fuelCut)
        {
            transitions++;
        }
    }

    return static_cast<double>(transitions) /
           (static_cast<double>(samples.size() - static_cast<std::size_t>(skip)) * deltaTime);
}

} // namespace

TEST_CASE("an upshift is a real torque interrupt, and it lasts as long as the car says", "[physics][shift]")
{
    auto bench = Bench{};
    bench.start(3, 60.0);

    const auto settled = bench.run(driving(1.0, 3), 720);
    const auto before = settled.back().axleTorque;

    // Pulling against held wheels through a locked clutch in third: the engine's own torque times the
    // reduction, and nothing else.
    REQUIRE(before > 2000.0);

    const auto shifted = bench.run(driving(1.0, 4), 720);

    SECTION("nothing at all leaves the box while the ratio is being changed")
    {
        const auto ticks = interruptTicks(shifted);

        // The box states 8 ms and a 360 Hz tick is 2.778 ms, so three ticks is the nearest it can be
        // held for: 8.33 ms against 8, which is the quantisation and not a modelling choice.
        REQUIRE(ticks == 3);
        REQUIRE(static_cast<double>(ticks) * tick == Catch::Approx(0.008333).epsilon(1e-3));

        // Zero, and exactly zero — an open gearbox is not a small torque.
        for (auto index = 0; index < ticks; index++)
        {
            REQUIRE(shifted[static_cast<std::size_t>(index)].gearboxTorque == 0.0);
            REQUIRE(shifted[static_cast<std::size_t>(index)].phase != ShiftPhase::Engaged);
        }

        // And the wheels do *not* see that zero, which is the compliant shaft doing its job rather
        // than the interrupt failing: the wind-up the box left in it comes back out over the window.
        // It rings while it does — freed of the box the shaft is alone on its spring at about 44 Hz,
        // so a quarter cycle is under three ticks and the torque crosses zero inside the interrupt.
        // That crossing is the element working rather than a sign error, and it is why this asks for
        // magnitude rather than for a monotonic fall.
        for (auto index = 0; index < ticks; index++)
        {
            const auto& sample = shifted[static_cast<std::size_t>(index)];

            REQUIRE(sample.axleTorque != 0.0);
            REQUIRE(std::abs(sample.axleTorque) < before);
        }
    }

    SECTION("and it comes back, in the gear that was asked for")
    {
        REQUIRE(bench.state.gear == 4);
        REQUIRE(shifted.back().phase == ShiftPhase::Engaged);

        // Settled in fourth: the same engine torque through a taller gear.
        const auto after = shifted.back().axleTorque;
        CAPTURE(before, after, settled.back().engineSpeed, shifted.back().engineSpeed, bench.setup.gearbox.reduction(3),
                bench.setup.gearbox.reduction(4));
        REQUIRE(after > 1500.0);
        REQUIRE(after < before);
        REQUIRE(
            after ==
            Catch::Approx(before * bench.setup.gearbox.reduction(4) / bench.setup.gearbox.reduction(3)).epsilon(0.05));
    }

    SECTION("the machine walks the phases in order and only once")
    {
        auto order = std::vector<ShiftPhase>{};
        for (const auto& sample : shifted)
        {
            if (order.empty() || order.back() != sample.phase)
            {
                order.push_back(sample.phase);
            }
        }

        REQUIRE(order == std::vector<ShiftPhase>{ShiftPhase::Disengaging, ShiftPhase::Neutral, ShiftPhase::Engaged});
    }
}

TEST_CASE("the referred inertia never passes through the values a ratio ramp would visit", "[physics][shift]")
{
    // The trap this shift machine exists to avoid. `referredInertia` goes as 1/reduction^2, so a
    // torque interrupt expressed as a ratio walked toward zero sends it to infinity on the way — and
    // the early return that catches a ratio of *exactly* zero catches nothing on that walk. Expressed
    // as an open gearbox instead, the ratio only ever holds one of the two gears' values.
    auto bench = Bench{};
    bench.start(3, 60.0);

    const auto axleInertia = bench.inertias[0] + bench.inertias[1];
    const auto third = axleInertia / (bench.setup.gearbox.reduction(3) * bench.setup.gearbox.reduction(3));
    const auto fourth = axleInertia / (bench.setup.gearbox.reduction(4) * bench.setup.gearbox.reduction(4));

    static_cast<void>(bench.run(driving(1.0, 3), 360));
    const auto shifted = bench.run(driving(1.0, 4), 360);

    auto peak = 0.0;
    for (const auto& sample : shifted)
    {
        REQUIRE(std::isfinite(sample.referredInertia));
        REQUIRE((sample.referredInertia == third || sample.referredInertia == fourth));
        peak = std::max(peak, sample.referredInertia);
    }

    REQUIRE(peak == Catch::Approx(fourth));

    SECTION("and the tallest gear this box has is still a small number")
    {
        // The whole ladder, which is where the peak actually lives: the taller the gear the less the
        // axle is divided down by, and sixth is the worst case a six-speed can offer.
        auto worst = 0.0;
        for (auto gear = 1; gear <= bench.setup.gearbox.topGear(); gear++)
        {
            for (const auto& sample : bench.run(driving(1.0, gear), 180))
            {
                REQUIRE(std::isfinite(sample.referredInertia));
                worst = std::max(worst, sample.referredInertia);
            }
        }

        REQUIRE(worst ==
                Catch::Approx(axleInertia / (bench.setup.gearbox.reduction(6) * bench.setup.gearbox.reduction(6))));
        REQUIRE(worst < 1.0);
    }
}

TEST_CASE("a rev match blips the engine with the throttle and nothing else", "[physics][shift]")
{
    struct Result
    {
        double entrySpeed = 0.0;
        double closeSpeed = 0.0;
        double closeSlip = 0.0;
        double reengagementHeat = 0.0;
        double settledSpeed = 0.0;
    };

    const auto target = 60.0 * placeholderDriveline().gearbox.reduction(3);

    const auto downshift = [target](const bool assist)
    {
        auto bench = Bench{};
        bench.setup.shiftAssist.revMatch = assist;
        bench.start(4, 60.0);

        // Off the throttle, which is where a downshift happens.
        const auto settled = bench.run(driving(0.0, 4), 720);

        const auto before = bench.state.slipEnergy;
        const auto samples = bench.run(driving(0.0, 3), 180);

        // Where the engine was on the tick the box took hold, which is the number the assist exists
        // to move. Everything after it is the plate paying for whatever is left.
        auto close = 0.0;
        for (auto index = std::size_t{1}; index < samples.size(); index++)
        {
            if (samples[index].phase == ShiftPhase::Engaged && samples[index - 1].phase != ShiftPhase::Engaged)
            {
                close = samples[index - 1].engineSpeed;
                break;
            }
        }

        return Result{.entrySpeed = settled.back().engineSpeed,
                      .closeSpeed = close,
                      .closeSlip = std::abs(close - target),
                      // Half a second from the request, which is the re-engagement and nothing else:
                      // a locked plate keeps a fraction of a rad/s of slip for ever, so a longer
                      // window measures the residual rather than the shift.
                      .reengagementHeat = bench.state.slipEnergy - before,
                      .settledSpeed = samples.back().engineSpeed};
    };

    const auto matched = downshift(true);
    const auto bare = downshift(false);

    // Both start and finish in the same two places: the gearing decides that, not the assist.
    REQUIRE(matched.entrySpeed == Catch::Approx(bare.entrySpeed));
    REQUIRE(matched.settledSpeed == Catch::Approx(bare.settledSpeed).epsilon(0.01));
    REQUIRE(matched.settledSpeed == Catch::Approx(target).epsilon(0.02));

    SECTION("the assist raises the engine to the gear before the gear raises the engine")
    {
        // With it, the engine has been taken to the new gear's speed under its own torque and closes
        // within a few rad/s. Without it, it *falls* while the box is open — the throttle is shut,
        // the driver is braking — and the plate finds nearly seventy rad/s to take out.
        REQUIRE(matched.closeSpeed > matched.entrySpeed + 50.0);
        REQUIRE(bare.closeSpeed < bare.entrySpeed);

        REQUIRE(matched.closeSlip < 10.0);
        REQUIRE(bare.closeSlip > 60.0);
        REQUIRE(matched.closeSlip < 0.2 * bare.closeSlip);
    }

    SECTION("and that mismatch is what the plate does not have to pay for")
    {
        REQUIRE(matched.reengagementHeat < bare.reengagementHeat);
        REQUIRE(matched.reengagementHeat < 0.25 * bare.reengagementHeat);
        // 363 J measured. It was over 400 with rigid shafts and the drop is the compliance rather
        // than a softer shift: the plate is re-syncing the gearbox input and the shaft above the
        // spring, not the car through it, so there is less inertia to bring back into step. What the
        // floor is for is unchanged — an unmatched downshift must cost real heat, and 60 rad/s of
        // mismatch through this plate does.
        REQUIRE(bare.reengagementHeat > 300.0);
    }
}

TEST_CASE("the rev matcher is a controller over the real engine, not a speed being written", "[physics][shift][assist]")
{
    const auto setup = placeholderDriveline();
    const auto assist = setup.shiftAssist;

    SECTION("below the target it opens the throttle, above it shuts it")
    {
        REQUIRE(revMatchThrottle(setup.engine, assist, 400.0, 300.0) == Catch::Approx(1.0));
        REQUIRE(revMatchThrottle(setup.engine, assist, 400.0, 380.0) == Catch::Approx(0.4));
        REQUIRE(revMatchThrottle(setup.engine, assist, 300.0, 400.0) == 0.0);
    }

    SECTION("and a money shift is blipped to the limiter and no further")
    {
        // What the *game* does about a request the engine cannot survive is a gameplay decision and
        // is deliberately not taken here — but the assist will not be the thing that carries it out.
        REQUIRE(revMatchThrottle(setup.engine, assist, 2.0 * setup.engine.limiterSpeed, setup.engine.limiterSpeed) ==
                0.0);
    }
}

TEST_CASE("the rev limiter chatters at the engine's rate rather than the timestep's", "[physics][shift][limiter]")
{
    // The measurement that decides it. A bare threshold re-arms the tick after the cut has slowed the
    // engine past it, so the transition count is a function of the *timestep*: halve the tick and it
    // doubles. That is the signature of an artefact. With a restore band the engine has to cross a
    // real speed interval both ways, and the rate is set by its own inertia against its own torque —
    // so it does not move when the timestep does.
    const auto rate = [](const double band, const double deltaTime)
    {
        auto bench = Bench{};
        bench.setup.engine.limiterRestoreBand = band;
        bench.deltaTime = deltaTime;
        bench.start(0, 0.0);
        bench.state.engineSpeed = 600.0;

        const auto seconds = 4.0;
        const auto steps = static_cast<int>(seconds / deltaTime);
        const auto skip = steps / 4;

        return chatterRate(bench.run(driving(1.0, 0), steps), deltaTime, skip);
    };

    const auto bare360 = rate(0.0, tick);
    const auto bare720 = rate(0.0, 0.5 * tick);
    const auto band360 = rate(16.0, tick);
    const auto band720 = rate(16.0, 0.5 * tick);

    SECTION("the bare threshold's rate is the timestep's")
    {
        REQUIRE(bare360 > 100.0);
        REQUIRE(bare720 == Catch::Approx(2.0 * bare360).epsilon(0.15));
    }

    SECTION("the band's rate is the engine's")
    {
        REQUIRE(band360 < 0.25 * bare360);
        REQUIRE(band720 == Catch::Approx(band360).epsilon(0.15));
    }

    SECTION("and the decision itself is a pure function of where the speed is and where it was")
    {
        const auto engine = placeholderDriveline().engine;

        REQUIRE_FALSE(advanceRevLimiter(engine, false, engine.limiterSpeed - 1.0));
        REQUIRE(advanceRevLimiter(engine, false, engine.limiterSpeed));
        // Inside the band, the answer depends on which way it was crossed, which is the whole point.
        REQUIRE(advanceRevLimiter(engine, true, engine.limiterSpeed - 8.0));
        REQUIRE_FALSE(advanceRevLimiter(engine, false, engine.limiterSpeed - 8.0));
        REQUIRE_FALSE(advanceRevLimiter(engine, true, engine.limiterSpeed - 20.0));
    }

    SECTION("and the instantaneous form of the torque is unchanged, because sweeps still want it")
    {
        const auto engine = placeholderDriveline().engine;

        REQUIRE(engineTorque(engine, engine.limiterSpeed + 1.0, 1.0) < 0.0);
        REQUIRE(engineTorque(engine, engine.limiterSpeed + 1.0, 1.0, false) > 100.0);
    }
}

TEST_CASE("paddles are counted, not held, and the gear demanded is a level", "[physics][shift][operation]")
{
    const auto setup = placeholderDriveline();
    const auto operation = TransmissionOperation{};
    const auto driveline = DrivelineState{};

    auto state = TransmissionState{};
    auto intent = DriverIntent{};

    const auto operate = [&]()
    {
        const auto input = operateTransmission(setup, operation, state, driveline, intent, tick);
        REQUIRE(input.has_value());

        return input.value();
    };

    SECTION("a held paddle asks once")
    {
        intent.upshifts = 1;
        REQUIRE(operate().gear == 2);

        // Seventy more ticks with the paddle still down, which is what a fifth of a second of holding
        // it looks like at 360 Hz.
        for (auto step = 0; step < 70; step++)
        {
            REQUIRE(operate().gear == 2);
        }
    }

    SECTION("a replayed tick asks for nothing new, which is the whole reason it is a count")
    {
        intent.upshifts = 3;
        intent.downshifts = 0;

        const auto first = operate();
        const auto restored = state;

        REQUIRE(first.gear == 4);

        // The same packet against the same restored state gives the same answer, every time.
        for (auto replay = 0; replay < 8; replay++)
        {
            state = restored;
            REQUIRE(operate().gear == first.gear);
        }
    }

    SECTION("a count that went backwards is a resynchronisation, not four billion requests")
    {
        intent.upshifts = 5;
        REQUIRE(operate().gear == 6);

        intent.upshifts = 1;
        REQUIRE(operate().gear == 6);
    }

    SECTION("the lever owns neutral and reverse, so a downshift under braking cannot find either")
    {
        intent.range = GearRange::Drive;
        REQUIRE(operate().gear == 1);

        intent.downshifts = 6;
        REQUIRE(operate().gear == 1);

        intent.range = GearRange::Neutral;
        REQUIRE(operate().gear == 0);

        intent.range = GearRange::Reverse;
        REQUIRE(operate().gear == -1);

        intent.range = GearRange::Drive;
        REQUIRE(operate().gear == 1);
    }

    SECTION("and the driver's other controls are passed through untouched")
    {
        intent.steering = -0.25;
        intent.throttle = 0.75;
        intent.brake = 0.5;
        intent.clutch = 0.55;

        const auto input = operate();

        REQUIRE(input.steering == -0.25);
        REQUIRE(input.throttle == 0.75);
        REQUIRE(input.brake == 0.5);
        REQUIRE(input.clutch == 0.55);
    }
}

TEST_CASE("an impossible gear cannot be requested, however hard it is asked for", "[physics][shift]")
{
    const auto setup = placeholderDriveline();

    SECTION("Gearbox::reduction still answers one, which is why nobody may reach it")
    {
        // Recorded rather than fixed: an eighth-gear request in a six-speed comes back as sixth, and
        // it is a plausible number, which is what makes it expensive. Everything that *makes* a gear
        // number clamps first, so this clamp is never the thing that answers.
        REQUIRE(setup.gearbox.reduction(8) == setup.gearbox.reduction(6));
        REQUIRE(setup.gearbox.reduction(600) == setup.gearbox.reduction(6));
        REQUIRE(setup.gearbox.topGear() == 6);
        REQUIRE(setup.gearbox.clampGear(8) == 6);
        REQUIRE(setup.gearbox.clampGear(-9) == -1);
    }

    SECTION("the operation mode cannot walk past the top gear")
    {
        auto state = TransmissionState{};
        auto intent = DriverIntent{};
        intent.upshifts = 200;

        const auto input = operateTransmission(setup, TransmissionOperation{}, state, DrivelineState{}, intent, tick);
        REQUIRE(input.has_value());
        REQUIRE(input->gear == 6);
    }

    SECTION("and the shift machine refuses one handed to it directly")
    {
        auto bench = Bench{};
        bench.start(6, 60.0);

        static_cast<void>(bench.run(driving(1.0, 99), 360));

        // Sixth, and it *says* sixth, which is the difference: the state reports the gear that is in
        // mesh rather than leaving the number to be guessed from the torque.
        REQUIRE(bench.state.gear == 6);
        REQUIRE(bench.state.shiftPhase == ShiftPhase::Engaged);
    }
}

TEST_CASE("a request during a shift is neither queued nor dropped, because it never stops being made",
          "[physics][shift]")
{
    // The queue-or-drop decision, and the answer is that there is nothing to queue. The demand is a
    // level, so a paddle pulled while the box is busy simply leaves a different number standing, and
    // the machine reads it the moment it is entitled to.
    auto bench = Bench{};
    bench.start(4, 60.0);

    static_cast<void>(bench.run(driving(0.0, 4), 720));

    const auto shifts = [](const std::vector<Sample>& samples)
    {
        auto count = 0;
        for (auto index = std::size_t{1}; index < samples.size(); index++)
        {
            if (samples[index].phase == ShiftPhase::Disengaging && samples[index - 1].phase == ShiftPhase::Engaged)
            {
                count++;
            }
        }

        return count;
    };

    SECTION("inside the neutral window it retargets, and the second request costs no second shift")
    {
        auto samples = bench.run(driving(0.0, 3), 12);
        REQUIRE(samples.back().phase == ShiftPhase::Neutral);

        for (const auto& sample : bench.run(driving(0.0, 2), 200))
        {
            samples.push_back(sample);
        }

        REQUIRE(bench.state.gear == 2);
        REQUIRE(shifts(samples) == 0);
        REQUIRE(interruptTicks(samples) < 40);
    }

    SECTION("past it the box is committed, and the standing demand starts the next shift immediately")
    {
        auto samples = bench.run(driving(0.0, 3), 30);
        REQUIRE(samples.back().phase == ShiftPhase::Engaging);

        for (const auto& sample : bench.run(driving(0.0, 2), 200))
        {
            samples.push_back(sample);
        }

        // Third went in, and second followed it without the request having to be made again.
        REQUIRE(bench.state.gear == 2);
        REQUIRE(shifts(samples) == 1);

        auto reachedThird = false;
        for (const auto& sample : samples)
        {
            reachedThird = reachedThird || (sample.gear == 3 && sample.phase == ShiftPhase::Engaged);
        }

        REQUIRE(reachedThird);
    }
}

TEST_CASE("both transmissions take the same paddle, and the converter's lockup behaves through it",
          "[physics][shift][operation]")
{
    // The point of the operation mode. A friction clutch with a manual gearbox and a torque converter
    // with a planetary one present the driver the same two paddles, and nothing below this line
    // learns which is fitted.
    const auto sequence = [](DrivelineSetup setup)
    {
        auto bench = Bench{};
        bench.setup = std::move(setup);
        bench.start(3, 60.0);

        auto mode = TransmissionState{};
        mode.gearDemand = 3;

        auto intent = DriverIntent{};
        intent.throttle = 1.0;

        const auto operate = [&]()
        {
            const auto input =
                operateTransmission(bench.setup, TransmissionOperation{}, mode, bench.state, intent, tick);
            REQUIRE(input.has_value());

            return input.value();
        };

        auto samples = std::vector<Sample>{};
        for (auto step = 0; step < 720; step++)
        {
            samples.push_back(bench.step(operate()));
        }

        const auto before = samples.back();

        // One pull of the upshift paddle, and then nothing at all for two seconds.
        intent.upshifts = 1;
        for (auto step = 0; step < 720; step++)
        {
            samples.push_back(bench.step(operate()));
        }

        struct Result
        {
            std::int32_t gear = 0;
            int interrupt = 0;
            double lockupBefore = 0.0;
            double lockupLowest = 0.0;
            double lockupAfter = 0.0;
            double torqueAfter = 0.0;
        };

        auto lowest = 1.0;
        for (auto index = std::size_t{720}; index < samples.size(); index++)
        {
            lowest = std::min(lowest, samples[index].lockupApply);
        }

        return Result{.gear = bench.state.gear,
                      .interrupt = interruptTicks(std::vector<Sample>(samples.begin() + 720, samples.end())),
                      .lockupBefore = before.lockupApply,
                      .lockupLowest = lowest,
                      .lockupAfter = samples.back().lockupApply,
                      .torqueAfter = samples.back().axleTorque};
    };

    const auto plate = sequence(placeholderDriveline());
    const auto fluid = sequence(placeholderAutomatic());

    SECTION("both change gear, and both interrupt for the same three ticks")
    {
        REQUIRE(plate.gear == 4);
        REQUIRE(fluid.gear == 4);
        REQUIRE(plate.interrupt == 3);
        REQUIRE(fluid.interrupt == 3);
        REQUIRE(plate.torqueAfter > 1000.0);
        REQUIRE(fluid.torqueAfter > 500.0);
    }

    SECTION("the lockup bleeds through the shift and takes hold again, rather than being cleared")
    {
        REQUIRE(fluid.lockupBefore == Catch::Approx(1.0));
        // Three ticks at the release rate is a sliver off the top, which is what a transmission
        // controller does with it. Cleared instead — which is what happened when the neutral path
        // reset the whole slot — it would have taken 400 ms to come back on every gear change.
        REQUIRE(fluid.lockupLowest < 1.0);
        REQUIRE(fluid.lockupLowest > 0.9);
        REQUIRE(fluid.lockupAfter == Catch::Approx(1.0));
    }
}

TEST_CASE("the clutch pedal stays the driver's, shift or no shift", "[physics][shift][clutch]")
{
    // The combination the rig exists to test: a human slipping a real pedal from rest while the
    // automation is operating the same clutch. `advanceClutchPedal`'s rule decides it and this states
    // nothing new — past the pedal's free play the foot wins outright.
    auto bench = Bench{};
    bench.start(1, 0.0);
    // Revs held for a launch. Idle will not do: 218 N.m of capacity against the 145 this engine makes
    // at 850 rpm stalls it in a sixth of a second, which is both correct and a different experiment.
    bench.state.engineSpeed = 350.0;

    auto input = driving(1.0, 1);
    input.clutch = 0.55;

    const auto launching = bench.run(input, 10);

    SECTION("a pedal held at the bite point is held there, and the clutch carries what it carries")
    {
        REQUIRE(bench.state.clutchPedal == 0.55);

        const auto capacity = clutchCapacity(bench.setup.coupling.clutch, 0.55);
        REQUIRE(capacity > 0.0);
        REQUIRE(capacity < 480.0);
        // The pedal is what the plate can hold, and it holds exactly that for as long as it is
        // sliding — the engine climbs away from it rather than being held down to the gearing, which
        // is what a launch is and what says the pedal reached the friction model rather than a number
        // that happened to multiply out.
        //
        // Asserted at the plate. The *wheels* cannot see it yet on tick one and it would be wrong if
        // they could: what stands between them is a spring, and a spring delivers by winding up. This
        // read as a clutch carrying a quarter of its capacity and was a shaft that had not finished
        // taking hold.
        for (const auto& sample : launching)
        {
            REQUIRE(sample.clutchTorque == Catch::Approx(capacity).epsilon(0.02));
        }

        // And the wheels settle on exactly that, once the shaft has stopped ringing about it. Getting
        // there is not a gentle build: a plate dumped onto a light gearbox input winds the spring
        // past where it is going and comes back, so the first tenth of a second *overshoots* the
        // steady figure rather than approaching it. That is the shunt a clutch dump produces, and the
        // bound is what says it is a shunt rather than a divergence.
        const auto held = bench.run(input, 350);
        REQUIRE(held.back().axleTorque == Catch::Approx(capacity * bench.setup.gearbox.reduction(1)).epsilon(0.02));

        const auto peak = std::max_element(launching.begin(), launching.end(),
                                           [](const Sample& a, const Sample& b) { return a.axleTorque < b.axleTorque; })
                              ->axleTorque;

        REQUIRE(peak > held.back().axleTorque);
        REQUIRE(peak < 3.0 * held.back().axleTorque);

        REQUIRE(bench.state.engineSpeed > 350.0);
    }

    SECTION("and it is a pedal rather than a switch: less of it carries less")
    {
        auto lighter = Bench{};
        lighter.start(1, 0.0);
        lighter.state.engineSpeed = 350.0;

        auto slipping = driving(1.0, 1);
        slipping.clutch = 0.62;

        const auto carried = lighter.run(slipping, 10).front().axleTorque;

        REQUIRE(carried > 0.0);
        REQUIRE(carried < 0.6 * launching.front().axleTorque);
    }

    SECTION("and a shift happening around it does not take it away")
    {
        bench.speeds = {60.0, 60.0, 60.0, 60.0};
        bench.state.engineSpeed = 60.0 * bench.setup.gearbox.reduction(3);

        auto held = driving(1.0, 3);
        held.clutch = 0.55;
        static_cast<void>(bench.run(held, 360));

        held.gear = 4;
        for (const auto& sample : bench.run(held, 120))
        {
            static_cast<void>(sample);
            REQUIRE(bench.state.clutchPedal == 0.55);
        }

        REQUIRE(bench.state.gear == 4);
    }

    SECTION("a converter car ignores it, once, and still shifts")
    {
        auto automatic = Bench{};
        automatic.setup = placeholderAutomatic();
        automatic.start(3, 60.0);

        auto pressed = driving(1.0, 3);
        pressed.clutch = 1.0;
        static_cast<void>(automatic.run(pressed, 360));

        pressed.gear = 4;
        const auto shifted = automatic.run(pressed, 360);

        REQUIRE(automatic.state.gear == 4);
        REQUIRE(shifted.back().axleTorque > 500.0);
    }
}

TEST_CASE("a shift is deterministic and survives being copied mid-flight", "[physics][shift][determinism]")
{
    static_assert(std::is_trivially_copyable_v<DrivelineState>, "a shift may not be what stops this being a memcpy");
    static_assert(std::is_standard_layout_v<DrivelineState>, "nor this");
    static_assert(std::is_trivially_copyable_v<TransmissionState>, "and the mode's state is saved the same way");

    const auto sequence = [](Bench& bench)
    {
        auto trace = std::vector<Sample>{};

        for (const auto& sample : bench.run(driving(1.0, 3), 360))
        {
            trace.push_back(sample);
        }
        for (const auto& sample : bench.run(driving(1.0, 4), 180))
        {
            trace.push_back(sample);
        }
        for (const auto& sample : bench.run(driving(0.0, 2), 360))
        {
            trace.push_back(sample);
        }

        return trace;
    };

    SECTION("the same sequence twice is the same run to the bit")
    {
        auto first = Bench{};
        first.start(3, 60.0);
        auto second = Bench{};
        second.start(3, 60.0);

        const auto left = sequence(first);
        const auto right = sequence(second);

        REQUIRE(left.size() == right.size());
        for (auto index = std::size_t{0}; index < left.size(); index++)
        {
            REQUIRE(left[index].axleTorque == right[index].axleTorque);
            REQUIRE(left[index].engineSpeed == right[index].engineSpeed);
            REQUIRE(left[index].referredInertia == right[index].referredInertia);
            REQUIRE(left[index].slipEnergy == right[index].slipEnergy);
            REQUIRE(left[index].gear == right[index].gear);
            REQUIRE(left[index].phase == right[index].phase);
        }

        REQUIRE(identical(first.state, second.state));
    }

    SECTION("and a state copied out of the middle of a shift continues identically")
    {
        auto bench = Bench{};
        bench.start(4, 60.0);
        static_cast<void>(bench.run(driving(0.0, 4), 360));

        // Into the long one, and stopped where the box has let go of fourth and not yet taken third.
        static_cast<void>(bench.run(driving(0.0, 3), 12));
        REQUIRE(bench.state.shiftPhase == ShiftPhase::Neutral);

        auto saved = DrivelineState{};
        std::memcpy(&saved, &bench.state, sizeof(DrivelineState));

        auto restored = bench;
        restored.state = saved;

        const auto left = bench.run(driving(0.0, 3), 720);
        const auto right = restored.run(driving(0.0, 3), 720);

        for (auto index = std::size_t{0}; index < left.size(); index++)
        {
            REQUIRE(left[index].axleTorque == right[index].axleTorque);
            REQUIRE(left[index].engineSpeed == right[index].engineSpeed);
            REQUIRE(left[index].gear == right[index].gear);
            REQUIRE(left[index].phase == right[index].phase);
        }

        REQUIRE(identical(bench.state, restored.state));
        REQUIRE(bench.state.gear == 3);
    }
}

TEST_CASE("the published car shifts on its own numbers", "[physics][shift][golf]")
{
    // The Golf's data states 8 ms up and 100 ms down for its dual clutch, and those are the defaults
    // every box here carries, so this car gets them without `PublishedCars` restating them.
    auto bench = Bench{};
    bench.setup = golfGtiMk7Driveline();

    REQUIRE(bench.setup.gearbox.topGear() == 7);
    REQUIRE(bench.setup.gearbox.shift.upshiftTime == Catch::Approx(0.008));
    REQUIRE(bench.setup.gearbox.shift.downshiftTime == Catch::Approx(0.100));

    bench.start(5, 60.0);
    static_cast<void>(bench.run(driving(1.0, 5), 720));

    const auto up = bench.run(driving(1.0, 6), 360);
    REQUIRE(interruptTicks(up) == 3);
    REQUIRE(bench.state.gear == 6);

    // A downshift is twelve times longer, because the engine has to be *raised* into the lower gear
    // and 100 ms is very nearly what 0.15 kg.m^2 takes to gain 200 rad/s on this engine's torque.
    const auto down = bench.run(driving(0.0, 5), 360);
    REQUIRE(interruptTicks(down) == 36);
    REQUIRE(static_cast<double>(interruptTicks(down)) * tick == Catch::Approx(0.1).epsilon(0.02));
    REQUIRE(bench.state.gear == 5);

    // And it cannot be asked for the eighth gear it does not have.
    static_cast<void>(bench.run(driving(1.0, 9), 360));
    REQUIRE(bench.state.gear == 7);
}
