#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;
import raceengine.tests.input;
import raceengine.tests.log;

using raceengine::ClipHistogram;
using raceengine::DeviceForceProfile;
using raceengine::ForceFeedbackOptions;
using raceengine::ForceFeedbackService;
using raceengine::ForceMapping;
using raceengine::InputOptions;
using raceengine::InputService;
using raceengine::mapRackTorque;
using raceengine::quantisationStep;
using raceengine::quietForce;
using raceengine::RackFeedback;
using raceengine::releaseForce;
using raceengine::tests::CapturedLog;
using raceengine::tests::RecordingInputBackend;
using raceengine::tests::SilentWindow;

namespace
{

constexpr auto tick = 1.0 / 500.0;

// Enough slew that a single call reaches its target, for the cases that are not about the slew.
[[nodiscard]] ForceMapping unlimited()
{
    auto mapping = ForceMapping{};
    mapping.slewRate = 1e9;

    return mapping;
}

// Straight to the target: previous output at the target already, and the ramp fully in.
[[nodiscard]] auto settled(const DeviceForceProfile& device, const ForceMapping& mapping, const double torque)
{
    auto command = mapRackTorque(device, mapping, torque, 0.0, 0.0, tick, 1.0);
    for (auto step = 0; step < 8; step++)
    {
        command = mapRackTorque(device, mapping, torque, 0.0, command.commandedTorque, tick, 1.0);
    }

    return command;
}

// The service will not take a wheel it has not been told about, so every case below needs the input
// service to have actually found one. That is a thread and a rediscovery interval, so it is waited
// for with a bounded poll whose expiry is the failure rather than asserted on immediately.
struct Rig
{
    CapturedLog log{};
    RecordingInputBackend backend{};
    SilentWindow window{};
    InputService input;
    ForceFeedbackService service;

    explicit Rig(ForceFeedbackOptions options) :
        input(log.sink(), backend, window,
              []
              {
                  auto brisk = InputOptions{};
                  brisk.rediscoveryInterval = std::chrono::milliseconds{1};
                  brisk.readTimeout = std::chrono::milliseconds{1};
                  brisk.vehicleRotationDegrees = 756.0;

                  return brisk;
              }()),
        service(log.sink(), input, std::move(options))
    {
        backend.attach();

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (std::chrono::steady_clock::now() < deadline && !input.deviceLink().takesTorque)
        {
            std::ignore = input.sample();
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }

        REQUIRE(input.deviceLink().connected);
        REQUIRE(input.deviceLink().takesTorque);
    }
};

} // namespace

TEST_CASE("the mapping is linear with a hard clip, and the ceiling is not the gain's to raise", "[input][ffb][mapping]")
{
    const auto device = DeviceForceProfile{};

    SECTION("gain scales what is asked for, and only that")
    {
        auto mapping = unlimited();
        mapping.gain = 0.5;

        const auto half = settled(device, mapping, 6.0);
        REQUIRE(half.requestedTorque == Catch::Approx(3.0));
        REQUIRE_FALSE(half.clipped);

        mapping.gain = 1.0;
        const auto full = settled(device, mapping, 6.0);
        REQUIRE(full.requestedTorque == Catch::Approx(6.0));

        // Linear, which is the settled decision: twice the gain is twice the torque everywhere in
        // the range, with no knee bending the middle of it.
        REQUIRE(full.requestedTorque == Catch::Approx(2.0 * half.requestedTorque));
    }

    SECTION("the ceiling holds however high the gain goes")
    {
        auto mapping = unlimited();
        mapping.ceilingTorque = 4.0;

        for (const auto gain : {1.0, 4.0, 40.0, 400.0})
        {
            mapping.gain = gain;

            const auto command = settled(device, mapping, 6.0);

            // A wrist is on the other end of this. A ceiling a dial can raise is not a ceiling.
            REQUIRE(std::abs(command.commandedTorque) <= 4.0 + quantisationStep(device));
            REQUIRE(command.clipped);
            REQUIRE(command.excessTorque > 0.0);
        }
    }

    SECTION("and the device's own peak is a second ceiling under it")
    {
        auto mapping = unlimited();
        // Asking for more than the base can make is not a way to get more than the base can make.
        mapping.ceilingTorque = 100.0;

        const auto command = settled(device, mapping, 50.0);
        REQUIRE(std::abs(command.commandedTorque) <= device.peakTorque);
    }

    SECTION("what is clipped is counted, because a gain is set from that measurement")
    {
        auto histogram = ClipHistogram{};
        auto mapping = unlimited();
        mapping.ceilingTorque = 4.0;

        for (auto step = 0; step < 100; step++)
        {
            // Half of them over the ceiling, half under it.
            const auto torque = step < 50 ? 8.0 : 1.0;
            histogram.add(settled(device, mapping, torque).requestedTorque, mapping.ceilingTorque);
        }

        REQUIRE(histogram.samples() == 100);
        REQUIRE(histogram.clippedPercentage() == Catch::Approx(50.0).epsilon(0.02));
        // A multiple of the ceiling rather than a torque: 2.0 says the gain could come down by half
        // and nothing would be lost that the ceiling was not already refusing.
        REQUIRE(histogram.peakDemand() == Catch::Approx(2.0));
    }
}

TEST_CASE("the slew limit bounds how fast the wheel may change and filters nothing", "[input][ffb][mapping]")
{
    const auto device = DeviceForceProfile{};

    auto mapping = ForceMapping{};
    mapping.slewRate = 400.0;

    // One tick at 500 Hz is 0.8 N.m of allowed change at this rate.
    const auto step = mapping.slewRate * tick;

    const auto first = mapRackTorque(device, mapping, 8.0, 0.0, 0.0, tick, 1.0);
    REQUIRE(first.commandedTorque == Catch::Approx(step).epsilon(1e-9));

    const auto second = mapRackTorque(device, mapping, 8.0, 0.0, first.commandedTorque, tick, 1.0);
    REQUIRE(second.commandedTorque == Catch::Approx(2.0 * step).epsilon(1e-9));

    // And it is a rate limit rather than a low pass: anything moving slower than the rate passes
    // through completely untouched, which is what keeps the texture a belt drive has not already
    // taken out.
    const auto gentle = mapRackTorque(device, mapping, 0.5, 0.0, 0.4, tick, 1.0);
    REQUIRE(gentle.commandedTorque == Catch::Approx(0.5).epsilon(1e-9));
}

TEST_CASE("the device's grid is applied here rather than left to the driver", "[input][ffb][mapping]")
{
    const auto device = DeviceForceProfile{};
    const auto mapping = unlimited();

    SECTION("a code is worth what the profile says it is")
    {
        // Eight bits over sixteen: 256 codes across the full range, one every 62.5 mN.m at 8 N.m.
        REQUIRE(quantisationStep(device) == Catch::Approx(2.0 * device.peakTorque / 256.0));
    }

    SECTION("and what is delivered lands on it")
    {
        for (const auto asked : {0.3, 1.0, 2.5, 5.0, 7.9})
        {
            const auto command = settled(device, mapping, asked);

            // Never further from the request than one code, which is the whole of what the device
            // can do about it.
            REQUIRE(std::abs(command.deliveredTorque - command.commandedTorque) <= quantisationStep(device));
        }
    }

    SECTION("and zero torque is not level zero")
    {
        // This driver reads `level == 0` as "disable the slot" rather than "make no force", so a
        // torque hovering either side of nothing would switch the effect off and on again hundreds
        // of times a second. Emitting the *centre* of the code keeps the slot armed while the
        // device makes nothing.
        const auto command = settled(device, mapping, 0.0);

        REQUIRE(command.fraction != 0.0);
        REQUIRE(std::abs(command.deliveredTorque) <= quantisationStep(device));

        // Letting go is the other thing, and it is exactly zero on purpose.
        REQUIRE(releaseForce().fraction == 0.0);
    }
}

TEST_CASE("a mapping asked for something that is not a number refuses it", "[input][ffb][mapping]")
{
    const auto device = DeviceForceProfile{};
    const auto mapping = unlimited();
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    const auto rejected = mapRackTorque(device, mapping, nan, 0.0, 0.0, tick, 1.0);
    REQUIRE(rejected.rejected);
    REQUIRE(rejected.fraction == 0.0);
    REQUIRE(rejected.commandedTorque == 0.0);
}

TEST_CASE("the wheel is let go within one output frame of anything going wrong", "[input][ffb][safety]")
{
    // Criterion 9, and all three of its cases. `step` is public precisely so these are properties of
    // a function rather than of a thread: a safety rule that could only be tested by starting a
    // thread and waiting is a rule nobody tests.
    //
    // Stepped on the real clock rather than a simulated one, because `publish` stamps what it stores
    // against `steady_clock` — a `now` running ahead of that makes every published state look older
    // than the watchdog allows and releases the wheel for the wrong reason, which is how this test
    // first read.
    auto options = ForceFeedbackOptions{};
    // No thread: this test drives the output frames itself.
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};
    options.mapping.slewRate = 1e9;

    auto rig = Rig(options);
    auto& service = rig.service;

    // Publish first, then take the instant to step at. The other order steps with a timestamp older
    // than the state it just stored, which the service correctly treats as stale and releases on —
    // so the whole test passes while proving nothing, because everything is zero anyway.
    const auto drive = [&service](const double torque)
    {
        const auto sampled = std::chrono::steady_clock::now();
        service.publish(RackFeedback{
            .steeringTorque = torque,
            .inputTimestampNanos = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(sampled.time_since_epoch()).count())});

        return service.step(std::chrono::steady_clock::now(), tick);
    };

    const auto engage = [&drive]
    {
        for (auto step = 0; step < 20; step++)
        {
            std::ignore = drive(5.0);
        }

        REQUIRE(std::abs(drive(5.0).commandedTorque) > 0.0);
    };

    SECTION("a value that is not a number")
    {
        engage();

        const auto poisoned = drive(std::numeric_limits<double>::quiet_NaN());

        // One frame. Not the next one, and not once the ramp has wound down.
        REQUIRE(poisoned.fraction == 0.0);
        REQUIRE(poisoned.commandedTorque == 0.0);
    }

    SECTION("focus lost to another window")
    {
        engage();

        service.setFocus(false);

        const auto released = drive(5.0);
        REQUIRE(released.fraction == 0.0);
        REQUIRE(released.commandedTorque == 0.0);
    }

    SECTION("the simulation stopping, which nothing tells this service about")
    {
        engage();

        // Nothing publishes again: the physics thread is gone. The watchdog is the only thing that
        // can notice, and what it must not do is hold the last torque for ever.
        std::this_thread::sleep_for(options.staleTimeout + std::chrono::milliseconds{10});

        const auto stalled = service.step(std::chrono::steady_clock::now(), tick);
        REQUIRE(stalled.fraction == 0.0);
        REQUIRE(stalled.commandedTorque == 0.0);
    }
}

TEST_CASE("torque comes up over a ramp rather than arriving", "[input][ffb][safety]")
{
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{100};
    options.mapping.slewRate = 1e9;

    auto rig = Rig(options);
    auto& service = rig.service;

    auto history = std::vector<double>{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{200};

    while (std::chrono::steady_clock::now() < deadline)
    {
        const auto sampled = std::chrono::steady_clock::now();
        service.publish(RackFeedback{
            .steeringTorque = 6.0,
            .inputTimestampNanos = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(sampled.time_since_epoch()).count())});

        history.push_back(std::abs(service.step(std::chrono::steady_clock::now(), tick).commandedTorque));
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }

    REQUIRE(history.size() > 10);

    // It starts at nothing and it gets there. A base that engages at full force the instant a device
    // is seen is a base that can take a wrist with it.
    const auto peak = *std::max_element(history.begin(), history.end());
    auto zeros = 0;
    for (const auto value : history)
    {
        zeros += value == 0.0 ? 1 : 0;
    }
    CAPTURE(history.size(), history.front(), history.back(), peak, zeros);

    REQUIRE(history.front() < 0.5);
    REQUIRE(history.back() > 5.0);

    // And it only ever goes up on the way, which is what makes it a ramp rather than a fade-in that
    // happens to end in the right place.
    for (auto index = std::size_t{1}; index < history.size(); index++)
    {
        REQUIRE(history[index] >= history[index - 1] - 1e-9);
    }
}

TEST_CASE("the damper has authority where the loop can serve it and not above", "[input][ffb][mapping]")
{
    // A damper closes a loop through the hardware, so the base's own delay bounds the band it can
    // damp in: below about `1/(4τ)` it opposes motion, above it leads, and further still it is a
    // negative spring. τ measured 12.7 ms on this rig, so the boundary is 20 Hz — and a damper's
    // torque is `c·ω`, which *rises* with frequency, so an unfiltered estimator hands the loop its
    // strongest term exactly where the loop cannot serve it. That is what shook: 3.1 N·m of damper
    // out of a 0.58 mm rack wiggle at 20 Hz, and a limit cycle that got faster as `c` went up.
    //
    // Two poles at eight hertz is what separates the two bands. The numbers below are the filter's
    // own, so they are arithmetic rather than a rig measurement — what the rig contributes is which
    // side of them the answer has to fall on.
    // Driven through the real estimator with a real device thread, because the filter is in the
    // writer and an arithmetic restatement of it here would pin the restatement.
    const auto peakDamperTorque = [](const double hertz, const double bandwidth)
    {
        auto options = ForceFeedbackOptions{};
        options.unattended = true;
        options.engageRamp = std::chrono::milliseconds{0};
        options.mapping.slewRate = 1e9;
        options.mapping.damping = 0.4;
        options.mapping.damperBandwidth = bandwidth;
        options.mapping.ceilingTorque = 8.0;

        auto rig = Rig(options);

        // Small enough that neither frequency reaches the ceiling: 200 counts of 65535 over 900
        // degrees is 2.75 degrees, so 20 Hz asks about 2.4 N·m of a 0.4 damper.
        constexpr auto amplitude = 200.0;
        constexpr auto centre = 32273.0;

        const auto cycles = 6.0;
        const auto seconds = cycles / hertz;
        const auto start = std::chrono::steady_clock::now();

        auto peak = 0.0;
        auto elapsed = 0.0;

        while (elapsed < seconds)
        {
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            rig.backend.turn(
                static_cast<std::int32_t>(centre + amplitude * std::sin(2.0 * 3.141592653589793 * hertz * elapsed)));

            // The car asks for nothing, so whatever is commanded is the damper and only the damper.
            rig.service.publish(RackFeedback{.steeringTorque = 0.0, .publishInterval = 1.0 / 120.0});

            // Read the clock *after* publishing: `publish` stamps itself, and a step time taken
            // before it is a step time older than the newest sample, which the watchdog correctly
            // reads as the simulation having stopped.
            const auto command = rig.service.step(std::chrono::steady_clock::now(), tick);

            // The last two thirds only: the estimator's own poles have to settle first.
            if (elapsed > seconds / 3.0)
            {
                peak = std::max(peak, std::abs(command.commandedTorque));
            }

            std::this_thread::sleep_for(std::chrono::milliseconds{2});
        }

        return peak;
    };

    // What a damper of this gain *should* make of a rim swinging this far at this rate: the
    // analytic `c·ω`, which is the only honest reference. Comparing a filtered peak against an
    // unfiltered one instead would be comparing against differentiation noise, which the filter
    // also removes and which is not what is being measured.
    constexpr auto rimAmplitude = 200.0 / 65535.0 * 900.0 * 0.017453292519943295;

    const auto ideal = [](const double hertz)
    {
        return 0.4 * 2.0 * 3.141592653589793 * hertz * rimAmplitude;
    };

    // Wide open, the damper's torque goes as `c·ω`: it has full authority at twenty hertz, which is
    // exactly where the loop's twelve and a half milliseconds of delay cannot deliver it in phase.
    const auto openFast = peakDamperTorque(20.0, 0.0);

    CAPTURE(openFast, ideal(20.0));
    REQUIRE(openFast > 0.8 * ideal(20.0));

    // Filtered, the frequency a released rim returns to centre at keeps its damper...
    const auto slow = peakDamperTorque(2.0, 8.0);

    CAPTURE(slow, ideal(2.0));
    REQUIRE(slow > 0.85 * ideal(2.0));

    // ...and the frequency the rig oscillated at loses it. Two poles at eight hertz predict 13.8% of
    // the unfiltered torque at twenty; the filter is being pinned against its own arithmetic here,
    // and what the rig contributes is which side of the boundary each number has to fall on.
    const auto fast = peakDamperTorque(20.0, 8.0);

    CAPTURE(fast, ideal(20.0));
    REQUIRE(fast < 0.25 * ideal(20.0));

    // The whole point, in one line: unfiltered the damper's authority climbs steeply across the
    // decade between the band it is for and the band that shook. Filtered, it barely climbs at all.
    REQUIRE(openFast / slow > 4.0);
    REQUIRE(fast / slow < 2.0);
}

TEST_CASE("the watchdog is measured against the cadence the publisher is managing", "[input][ffb][safety]")
{
    // The timeout was a flat thirty milliseconds, reasoned as "three ticks of a 120 Hz loop and a
    // little". That reasoning assumes the ticks arrive one at a time, and they do not: the engine
    // publishes a whole frame's worth in a burst, so the interval this is measured against is the
    // **frame** time. Thirty milliseconds is one frame at 33 fps — so at 4K/30 the wheel went stale
    // every single frame, released, and re-engaged into a fresh ramp it never got through. It read
    // from the seat as "the shaking is much milder at 4K/30", which was true and was the force
    // feedback switching itself off. Measured on the 37 fps exit trace, 1.2% of frames already
    // exceeded it.
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};
    options.mapping.slewRate = 1e9;

    auto rig = Rig(options);
    auto& service = rig.service;

    // Ten frames at 30 fps, which every one of them is past the flat floor.
    constexpr auto frame = std::chrono::milliseconds{33};
    auto last = raceengine::ForceCommand{};

    for (auto index = 0; index < 10; index++)
    {
        service.publish(RackFeedback{.steeringTorque = 6.0, .publishInterval = 1.0 / 120.0});
        last = service.step(std::chrono::steady_clock::now(), tick);
        std::this_thread::sleep_for(frame);
    }

    // Still holding the wheel: a simulation running slowly is not a simulation that has stopped.
    REQUIRE(std::abs(last.commandedTorque) > 5.0);

    // And a simulation that genuinely stops is still caught, within a few of its own frames.
    std::this_thread::sleep_for(frame * 5);

    const auto stalled = service.step(std::chrono::steady_clock::now(), tick);

    REQUIRE(stalled.commandedTorque == 0.0);
}

TEST_CASE("a momentary release costs its own duration of ramp and not the whole of it", "[input][ffb][safety]")
{
    // The ramp used to be a stopwatch restarted from zero on every engagement, so a two-millisecond
    // hitch cost the full second. Paired with a watchdog shorter than a frame that was the whole of
    // the force: released and re-engaged every frame, the envelope never got three percent up. The
    // safety rule is about engaging from *nothing*, and a wheel that was holding four newton metres
    // two milliseconds ago is not engaging from nothing — so the envelope falls at the rate it
    // rises rather than being reset.
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{200};
    options.mapping.slewRate = 1e9;

    auto rig = Rig(options);
    auto& service = rig.service;

    const auto publish = [&service]
    {
        service.publish(RackFeedback{.steeringTorque = 6.0, .publishInterval = 1.0 / 120.0});
    };

    // Up to full force. Real time throughout, because `publish` stamps itself off the same clock
    // the watchdog reads and a synthetic step time would simply read as stale.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
    while (std::chrono::steady_clock::now() < deadline)
    {
        publish();
        std::ignore = service.step(std::chrono::steady_clock::now(), tick);
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }

    publish();
    const auto full = service.step(std::chrono::steady_clock::now(), tick);
    REQUIRE(std::abs(full.commandedTorque) > 5.0);

    // Focus lost for one output frame, then back.
    service.setFocus(false);
    std::ignore = service.step(std::chrono::steady_clock::now(), tick);
    service.setFocus(true);

    publish();
    const auto resumed = service.step(std::chrono::steady_clock::now(), tick);

    // One frame of a two-hundred-millisecond ramp is one percent of it, so the force comes back at
    // very nearly what it left at rather than at nothing.
    REQUIRE(std::abs(resumed.commandedTorque) > 5.0);

    // A cold start still takes the whole ramp, which is the rule this is not allowed to weaken.
    auto cold = Rig(options);
    cold.service.publish(RackFeedback{.steeringTorque = 6.0, .publishInterval = 1.0 / 120.0});

    const auto first = cold.service.step(std::chrono::steady_clock::now(), tick);

    REQUIRE(std::abs(first.commandedTorque) < 0.5);
}

TEST_CASE("input to output is measured rather than estimated", "[input][ffb][latency]")
{
    // Criterion 7. The histogram is fed from the input sample's own timestamp, carried on the
    // published state, and stamped against the clock at the instant the torque is written — so what
    // it reports is the whole path from the axis being read to the device being told, and not the
    // part of it this service happens to own.
    //
    // The budget it has to fit in: an axis read on the input thread, a physics tick that publishes
    // what it computed from it, and an output frame that picks the newest published state up. At
    // 120 Hz and 500 Hz those are 8.3 ms and 2 ms, and the two do not add — the writer takes the
    // newest state whenever it runs rather than waiting for a fresh one. What is asserted here is
    // the criterion's own bound on the whole thing.
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};

    auto rig = Rig(options);
    auto& service = rig.service;

    // Real timestamps, because the histogram stamps its far end against the real clock: a simulated
    // `now` running ahead of it would record no samples at all, which is how this test first read.
    const auto nanosOf = [](const std::chrono::steady_clock::time_point when)
    {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(when.time_since_epoch()).count());
    };

    // How old the axis reading already is by the time the physics tick that used it publishes. One
    // 120 Hz period is the honest worst case and is most of the budget on its own.
    constexpr auto sampleAge = std::chrono::microseconds{8333};

    for (auto step = 0; step < 500; step++)
    {
        const auto now = std::chrono::steady_clock::now();

        if (step % 4 == 0)
        {
            service.publish(RackFeedback{.steeringTorque = 3.0, .inputTimestampNanos = nanosOf(now - sampleAge)});
        }

        std::ignore = service.step(std::chrono::steady_clock::now(), tick);
    }

    const auto& latency = service.latencyHistogram();

    REQUIRE(latency.samples() > 0);
    CAPTURE(latency.mean(), latency.percentile(0.99), latency.peak());

    // The criterion's figure, and it is a bound on the whole path rather than on the mean of it.
    REQUIRE(latency.percentile(0.99) < 10.0);
    REQUIRE(latency.peak() < 10.0);
}

TEST_CASE("one input sample is measured once, however many frames it survives into", "[input][ffb][latency]")
{
    // The defect this pins read as criterion 7 failing on hardware: a median of 16.75 ms and a worst
    // of 243 ms against a 10 ms budget. The pipeline was fine — the *instrument* was measuring the
    // age of the device's last report once per output frame, and a wheel nobody is touching sends no
    // reports at all. At 43 Hz idle against a 500 Hz writer that is a histogram of the device's idle
    // interval with no latency in it anywhere.
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};

    auto rig = Rig(options);
    auto& service = rig.service;

    const auto sampled = std::chrono::steady_clock::now();
    const auto stamp = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(sampled.time_since_epoch()).count());

    // One report from the device, and then two hundred output frames with nothing new arriving —
    // which is what a straight is.
    service.publish(RackFeedback{.steeringTorque = 3.0, .inputTimestampNanos = stamp});

    for (auto step = 0; step < 200; step++)
    {
        std::ignore = service.step(std::chrono::steady_clock::now(), tick);
    }

    REQUIRE(service.latencyHistogram().samples() == 1);

    // And a second report is a second measurement, so nothing has been lost by not counting the
    // repeats.
    service.publish(RackFeedback{
        .steeringTorque = 3.0,
        .inputTimestampNanos = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count())});

    std::ignore = service.step(std::chrono::steady_clock::now(), tick);

    REQUIRE(service.latencyHistogram().samples() == 2);
}

TEST_CASE("letting go of the wheel does not put the effect away", "[input][ffb][safety]")
{
    // On this driver level zero means "disable the slot" rather than "make no force". A game stalls
    // for longer than the watchdog allows every time it loads something, so the service engages,
    // starves, releases and re-engages several times a second — and if releasing wrote level zero
    // that is an effect being torn down and re-uploaded at that rate. Measured on the rig, the wheel
    // left the USB bus every few seconds with force feedback running and held the same port for
    // minutes under a read-only probe.
    const auto device = DeviceForceProfile{};

    const auto quiet = quietForce(device);

    // No torque, which is the whole of what the safety rules ask for.
    REQUIRE(std::abs(quiet.deliveredTorque) <= quantisationStep(device));
    REQUIRE(quiet.commandedTorque == 0.0);

    // And not level zero, so the slot stays armed.
    REQUIRE(quiet.fraction != 0.0);

    // `releaseForce` is still the one that puts it away, for shutting down.
    REQUIRE(releaseForce().fraction == 0.0);
}

TEST_CASE("a released wheel is written the quiet code rather than level zero", "[input][ffb][safety]")
{
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};

    auto rig = Rig(options);

    // Engage, then let the watchdog take it away, which is what every asset load does.
    const auto sampled = std::chrono::steady_clock::now();
    rig.service.publish(
        RackFeedback{.steeringTorque = 4.0,
                     .inputTimestampNanos = static_cast<std::uint64_t>(
                         std::chrono::duration_cast<std::chrono::nanoseconds>(sampled.time_since_epoch()).count())});

    std::ignore = rig.service.step(std::chrono::steady_clock::now(), tick);

    std::this_thread::sleep_for(options.staleTimeout + std::chrono::milliseconds{10});
    std::ignore = rig.service.step(std::chrono::steady_clock::now(), tick);

    // What reached the device is the zero code, not the disable.
    REQUIRE(rig.backend.torque() != 0.0);
    REQUIRE(std::abs(rig.backend.torque()) < 0.01);
}

TEST_CASE("the damper opposes the measured rim and cannot pierce the ceiling", "[input][ffb][mapping]")
{
    // The damper is the tuning-menu damper this base cannot reach under hid-fanatecff, restated as
    // a mapping term: a torque against the rim's *measured* rate, which is the one dissipation a
    // hands-off rim has. `SteeringShimmyProbe` is where its necessity was measured; what is pinned
    // here is its arithmetic.
    const auto device = DeviceForceProfile{};
    auto mapping = unlimited();
    mapping.damping = 0.5;

    SECTION("a moving rim is opposed, and the clip accounting does not see it")
    {
        const auto still = mapRackTorque(device, mapping, 2.0, 0.0, 2.0, tick, 1.0);
        const auto moving = mapRackTorque(device, mapping, 2.0, 3.0, 2.0, tick, 1.0);

        // The requested torque is the car's ask, which is what the histogram measures a gain
        // against — the damper must not launder it.
        REQUIRE(moving.requestedTorque == Catch::Approx(still.requestedTorque));
        REQUIRE(moving.commandedTorque == Catch::Approx(still.commandedTorque - 0.5 * 3.0));
    }

    SECTION("the invert flips the car's torque and not the damper")
    {
        // The damper opposes measured motion in the device's own frame, so it is the one term the
        // invert must not touch: flipped with it, an inverted car would have the damper *driving*
        // the rim.
        auto inverted = mapping;
        inverted.invert = true;

        const auto command = mapRackTorque(device, inverted, 2.0, 3.0, -2.0, tick, 1.0);

        REQUIRE(command.requestedTorque == Catch::Approx(-2.0));
        REQUIRE(command.commandedTorque == Catch::Approx(-2.0 - 0.5 * 3.0));
    }

    SECTION("the ceiling holds against the damper too")
    {
        auto guarded = mapping;
        guarded.ceilingTorque = 4.0;

        const auto command = mapRackTorque(device, guarded, 2.0, -20.0, 4.0, tick, 1.0);

        // 2 + 10 wants 12; the wrist was promised 4. And the clip accounting still answers for the
        // car's own 2, because that is what a gain is set from.
        REQUIRE(command.commandedTorque == Catch::Approx(4.0));
        REQUIRE_FALSE(command.clipped);
    }

    SECTION("and it keeps its authority over a demand the ceiling already clipped")
    {
        // The rig's standstill shake, in one line: the carcass spring asks for twenty times the
        // ceiling, so a damper applied before the clip is subtracted from a number the clip then
        // throws away — re-clipped to the same bang-bang square wave, the one regime the damper
        // exists for. Post-clip it can take a saturated demand all the way to the opposite rail,
        // which is what draining a limit cycle requires, and the second clamp keeps the wrist's
        // ceiling honest.
        auto guarded = mapping;
        guarded.ceilingTorque = 4.0;

        const auto still = mapRackTorque(device, guarded, 94.0, 0.0, 4.0, tick, 1.0);
        const auto shaking = mapRackTorque(device, guarded, 94.0, 28.0, 4.0, tick, 1.0);

        REQUIRE(still.commandedTorque == Catch::Approx(4.0));
        REQUIRE(shaking.commandedTorque == Catch::Approx(-4.0));
    }

    SECTION("a negative damping is inert rather than an exciter")
    {
        auto backwards = mapping;
        backwards.damping = -1.0;

        const auto command = mapRackTorque(device, backwards, 2.0, 3.0, 2.0, tick, 1.0);

        REQUIRE(command.commandedTorque == Catch::Approx(2.0));
    }
}

TEST_CASE("a catch-up burst does not steepen the reconstruction", "[input][ffb][safety]")
{
    // The engine's fixed-step loop catches up, so on a 60 Hz display two ticks publish
    // microseconds apart and then nothing for a frame. The slope used to be taken over that wall
    // interval — hundreds of times too steep, saturating its clamp instantly — so the writer sat at
    // newest-plus-one-whole-change for the rest of the frame: every ripple doubled, and the rack's
    // friction term delivered inverted on half the frames, which is what shook the wheel at speed.
    // The publisher states the simulated interval now, and this pins that a burst is reconstructed
    // on that interval rather than on the wall.
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};
    options.mapping.slewRate = 1e9;

    auto rig = Rig(options);
    auto& service = rig.service;

    constexpr auto interval = 1.0 / 120.0;

    const auto publish = [&service](const double torque, const double stated)
    {
        const auto sampled = std::chrono::steady_clock::now();
        service.publish(RackFeedback{
            .steeringTorque = torque,
            .publishInterval = stated,
            .inputTimestampNanos = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(sampled.time_since_epoch()).count())});
    };

    for (auto step = 0; step < 20; step++)
    {
        publish(1.0, interval);
        std::ignore = service.step(std::chrono::steady_clock::now(), tick);
    }

    SECTION("the default is to hold the newest sample rather than run past it")
    {
        // The burst: two consecutive ticks, back to back, one tick's slope apart.
        publish(1.0, interval);
        publish(1.2, interval);

        std::this_thread::sleep_for(std::chrono::milliseconds{1});

        const auto held = service.step(std::chrono::steady_clock::now(), tick);

        // Replayed against the rig's own 300-second trace, extrapolating carried 8128 N·m of total
        // variation and 2.4x the 5-40 Hz energy of the physics it was reconstructing, against 3441
        // and 1.0x for holding. Prediction in a loop that closes through a driver's hands is
        // positive feedback, so nothing runs forward unless it is asked to.
        REQUIRE(held.commandedTorque == Catch::Approx(1.2));
    }

    SECTION("no stated interval means hold, not extrapolate")
    {
        publish(1.0, 0.0);
        publish(1.2, 0.0);

        std::this_thread::sleep_for(std::chrono::milliseconds{1});

        const auto held = service.step(std::chrono::steady_clock::now(), tick);

        REQUIRE(held.commandedTorque == Catch::Approx(1.2));
    }
}

TEST_CASE("the reconstruction is paced by the samples this thread actually holds", "[input][ffb][safety]")
{
    // **The writer does not see consecutive ticks.** `publish` overwrites one slot and the engine
    // delivers a whole frame's ticks inside a fraction of a millisecond, so a writer sampling at
    // 500 Hz takes the last tick of each burst and never sees the others: `newest` and `older` are
    // one *frame* apart, not one tick. Pacing the slope by the publisher's stated tick therefore
    // overstated it by the burst length — 3.4x on the rig's 37 fps trace — which saturated the
    // excursion clamp inside the first third of every frame and left the output sitting a whole
    // frame's change past the newest sample for the other two thirds. This pins that the span is
    // measured rather than stated.
    auto options = ForceFeedbackOptions{};
    options.unattended = true;
    options.engageRamp = std::chrono::milliseconds{0};
    options.mapping.slewRate = 1e9;
    options.extrapolation = 1.0;

    auto rig = Rig(options);
    auto& service = rig.service;

    constexpr auto interval = 1.0 / 120.0;

    const auto publish = [&service](const double torque)
    {
        service.publish(RackFeedback{.steeringTorque = torque, .publishInterval = interval});
    };

    // Two samples the writer genuinely takes one from another, ten milliseconds apart on the wall.
    publish(1.0);
    std::ignore = service.step(std::chrono::steady_clock::now(), tick);

    std::this_thread::sleep_for(std::chrono::milliseconds{10});

    publish(1.2);
    const auto taken = std::chrono::steady_clock::now();
    std::ignore = service.step(taken, tick);

    // Five milliseconds on: half of the ten-millisecond span the writer measured, so half of the
    // 0.2 change. Paced by the stated 8.33 ms tick it would already be clamped at the full 1.4.
    const auto midway = service.step(taken + std::chrono::milliseconds{5}, tick);

    REQUIRE(midway.commandedTorque > 1.25);
    REQUIRE(midway.commandedTorque < 1.35);

    // And the excursion is still bounded by the change itself however far past the sample it runs —
    // measured inside the watchdog's window, since past that the honest answer is to let go rather
    // than to keep predicting.
    const auto far = service.step(taken + std::chrono::milliseconds{25}, tick);

    REQUIRE(far.commandedTorque == Catch::Approx(1.4));
}

TEST_CASE("the damper is outside the rate limit, because a rate-limited damper is an oscillator",
          "[input][ffb][mapping]")
{
    // A rate limit is a statement about how fast the *car's* demand may change. A damper is not a
    // demand: it is a torque computed from a measurement of the rim taken microseconds ago, and its
    // whole value is being in phase with the rim now. Delivered through a 400 N·m/s limiter — 0.8
    // N·m per output frame — a damper asked for several newton metres arrives as a triangle wave
    // lagging its own demand by up to a quarter period, and a velocity term ninety degrees late is
    // a spring. That is why the rig shook *harder* as `ffb.damping` went up.
    const auto device = DeviceForceProfile{};

    auto mapping = ForceMapping{};
    mapping.slewRate = 400.0;
    mapping.damping = 0.5;

    // The car asks for nothing and the rim is turning at eight radians per second: four newton
    // metres of pure damping, which is five output frames' worth of slew.
    const auto command = mapRackTorque(device, mapping, 0.0, 8.0, 0.0, tick, 1.0);

    REQUIRE(command.commandedTorque == Catch::Approx(-4.0));

    // The car's own torque is still rate limited — the hammer-blow guarantee is untouched — and it
    // is that, not the damped output, that the next call is handed back.
    REQUIRE(command.slewedTorque == Catch::Approx(0.0));

    const auto pulling = mapRackTorque(device, mapping, 8.0, 0.0, 0.0, tick, 1.0);

    REQUIRE(pulling.slewedTorque == Catch::Approx(mapping.slewRate * tick));
    REQUIRE(pulling.commandedTorque == Catch::Approx(mapping.slewRate * tick));

    // And the ceiling still holds over the pair of them.
    auto tight = mapping;
    tight.ceilingTorque = 2.0;

    const auto capped = mapRackTorque(device, tight, 0.0, 40.0, 0.0, tick, 1.0);

    REQUIRE(capped.commandedTorque == Catch::Approx(-2.0));
}
