module;

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

export module raceengine.input:ForceFeedbackService;

import :ForceMapping;
import :PedalFeedback;
import :PedalMotors;
import :InputBackend;
import :InputService;
import :RackTorque;

namespace raceengine
{

// Stage three of three: a thread, a clock, four safety rules and one syscall.
//
// Nothing in this file decides how hard the wheel pushes. Stage one said what the road is doing in
// newton metres and stage two said what this base can make of that; what is left is when to write
// it, how often, and every circumstance in which the honest answer is to let go.

// What the simulation hands over each tick. Stage one's answer plus the two things needed to make
// sense of it: where the rack is, and which device report the whole chain was derived from.
export struct RackFeedback
{
    // N·m at the rim, and **what the driver's hands are meant to feel** — this car's own power
    // steering included, because that is between the road and the rim in the real car too. Always
    // in newton metres, never a fraction of anything.
    double steeringTorque = 0.0;
    // The same instant with no power steering in it: the road and the geometry alone. Nothing
    // downstream acts on this and that is the point — it is the column of the trace that stays
    // comparable across devices, sessions and future hardware, and it was carrying an assisted
    // number until 2026-08-21 because the two were one field.
    //
    // A publisher is expected to state both. One that does not writes a flat zero into that column,
    // which reads as obviously absent rather than as quietly wrong — the failure this whole split
    // exists to make visible.
    double unassistedTorque = 0.0;
    // Newtons at the rack: the whole of it, and the tyres' share of it. Carried so the trace can
    // show where a torque came from without the reader having to divide by a pinion radius they
    // would have to go and look up.
    double rackForce = 0.0;
    double tyreRackForce = 0.0;
    double rackTravel = 0.0;
    double rackVelocity = 0.0;
    // The *simulated* time between this publish and the next, in seconds, stated by the publisher
    // because the wall clock cannot state it: a fixed-step game publishes from inside a catch-up
    // loop, so two consecutive publishes can be microseconds apart on the wall and one whole tick
    // apart in the simulation — and a slope taken over the wall interval is then hundreds of times
    // too steep. Zero means no slope is continued and the writer holds the newest value.
    double publishInterval = 0.0;
    // `steady_clock` nanoseconds, from `DeviceSample::timestampNanos`. Zero when the demand did not
    // come from a device — a keyboard, or the gate's scripted launch — in which case no end-to-end
    // latency is measurable and none is reported.
    std::uint64_t inputTimestampNanos = 0;

    // What the car was doing on this tick, for the trace. Nothing downstream of here acts on it —
    // no safety rule reads it and no torque is derived from it — it rides along so that the assist
    // curve's *aim* is measurable and not only its shape. A publisher that leaves it default writes
    // zeros, which reads as obviously absent rather than as quietly wrong.
    VehicleTrace vehicle{};
};

// **There was a `damping` field on `RackFeedback` and it is gone**, and its going is the point rather
// than a tidy-up. The simulation used to ask for 0.4 N·m per rad/s of rim speed at a standstill,
// sized as 2·√(K·J) — but the `J` in it was fitted from *one wheel base's* limit cycle, so a device
// inertia was reaching the pipeline dressed as a car's request. Reckoned honestly for the car it is
// thirty times larger, and the reason the two disagree is that the simulation has no steering degree
// of freedom at all: the rack angle is commanded from the demand, so the only inertia in the loop a
// damper closes belongs to the motor, the belt and the rim.
//
// So the damper is stage two's, in `ForceMapping::deviceDamping`, which is where a number about a
// wheel base is allowed to live. A car with a rack that has a mass may want that channel back; a car
// without one has nothing to say through it, and an inert field is a feature that reads as
// implemented and behaves as a comment.

// Where a torque went, in tenths of a millisecond up to twenty. A fixed set of counters rather than
// a list of samples because it is written on the output thread, where nothing may allocate.
export inline constexpr std::size_t latencyBins = 80;
export inline constexpr double latencyBinMilliseconds = 0.25;

export class LatencyHistogram
{
    std::array<std::uint64_t, latencyBins + 1> counts{};
    std::uint64_t total = 0;
    double worst = 0.0;
    double sum = 0.0;

public:
    void add(const double milliseconds)
    {
        if (!std::isfinite(milliseconds) || milliseconds < 0.0)
        {
            return;
        }

        total++;
        sum += milliseconds;
        worst = std::max(worst, milliseconds);

        const auto scaled = milliseconds / latencyBinMilliseconds;
        const auto bin = scaled >= static_cast<double>(latencyBins) ? latencyBins : static_cast<std::size_t>(scaled);

        counts[bin]++;
    }

    [[nodiscard]] std::uint64_t samples() const
    {
        return total;
    }

    [[nodiscard]] double mean() const
    {
        return total == 0 ? 0.0 : sum / static_cast<double>(total);
    }

    [[nodiscard]] double peak() const
    {
        return worst;
    }

    // The upper edge of the bin the given fraction of samples falls at or below. Quantised to the
    // bin width, which is what makes this a histogram and not a sorted list — and a quarter of a
    // millisecond is finer than anything downstream can act on anyway.
    [[nodiscard]] double percentile(const double fraction) const
    {
        if (total == 0)
        {
            return 0.0;
        }

        const auto wanted = static_cast<std::uint64_t>(std::clamp(fraction, 0.0, 1.0) * static_cast<double>(total));
        auto running = std::uint64_t{0};

        for (auto index = std::size_t{0}; index <= latencyBins; index++)
        {
            running += counts[index];
            if (running >= wanted)
            {
                return static_cast<double>(index + 1) * latencyBinMilliseconds;
            }
        }

        return static_cast<double>(latencyBins + 1) * latencyBinMilliseconds;
    }

    void clear()
    {
        counts = {};
        total = 0;
        worst = 0.0;
        sum = 0.0;
    }
};

export struct ForceFeedbackOptions
{
    // No thread is started and no torque is ever written. The same rule every other input path here
    // keeps: an unattended run owns no hands at the controls, and it is what makes a capture
    // byte-identical with and without any of this.
    bool unattended = false;

    // The pedals' own motors, and the driver's dials for them. Defaulted to a set with none, which
    // is both the safe answer and the common one — CSL Elite and CSL LC pedals have no motors, and a
    // ClubSport V3 set plugged into the base's RJ12 rather than its own USB cable has motors that
    // nothing on this platform can reach.
    PedalMotorProfile pedals{};
    PedalMotorMapping pedalMapping{};

    // What the writer thread paces itself to. **Measured rather than assumed**: this base's driver
    // consumes its effect slots on a two millisecond `hrtimer` that is compile-time fixed, so five
    // hundred is the ceiling whatever the transport allows — and the transport does allow a
    // thousand, which is exactly the trap. Writing at twice the rate the driver reads is half the
    // writes going nowhere and a rate estimate that says everything is fine.
    double outputHz = 500.0;

    // How long the newest published state may go unrefreshed before the wheel is let go — a
    // **floor**, which the writer raises to a multiple of the publish cadence it is actually
    // seeing. It was a flat thirty milliseconds, reasoned as "three ticks of a 120 Hz loop and a
    // little", and that reasoning quietly assumed the ticks arrive one at a time. They do not: the
    // engine publishes a whole frame's worth in a burst, so the interval this timeout is measured
    // against is the **frame** time and not the tick. Thirty milliseconds is one frame at 33 fps —
    // so at 4K/30 the wheel went stale every single frame, released, and re-engaged into a fresh
    // one-second ramp it never got more than three percent through. That is a force feedback
    // system that switches itself off below 33 fps, and it presented as "the shaking is much
    // milder at 4K/30", which is true and is not good news. Measured on the 37 fps exit trace,
    // 1.2% of frames already exceeded it.
    std::chrono::milliseconds staleTimeout{30};

    // What the floor above is raised to, as a multiple of the longest publish gap recently seen. A
    // simulation that has genuinely stopped is still caught within a few frames, which is what the
    // watchdog is for; a simulation running slowly is not a simulation that has stopped.
    double staleFrames = 3.0;

    // How much of the last sample-to-sample change the writer may run *past* the newest sample
    // while waiting for the next one, 0 to 1. Zero holds. See `reconstruct` for why zero is the
    // default and why this is a dial: it trades latency against phase margin in a loop that closes
    // through the driver's hands, and that is a seat judgement rather than an arithmetic one.
    double extrapolation = 0.0;

    // Torque comes up over this, from nothing, every time the wheel is taken and every time it has
    // been let go for any reason. A base that engages at full force the instant a device is seen is
    // a base that can take a wrist with it.
    std::chrono::milliseconds engageRamp{1000};

    ForceMapping mapping;
    DeviceForceProfile device;

    // Frames of stage-one trace kept. Zero records nothing; it is a ring, so this is a memory
    // budget and not a run length. 360 Hz for a minute is about 22000.
    std::size_t traceCapacity = 0;
};

// The device layer's write side: one thread, one wheel, and a torque that is a function of what the
// simulation last said and of the clock.
//
// The thread exists for the same reason the reader's does. A wheel wants five hundred torques a
// second and a simulation produces one every eight milliseconds, and neither may be able to hold up
// the other: the writer takes the published state if it can have it without waiting and works from
// what it already had if it cannot, and the simulation publishes the same way.
export class ForceFeedbackService
{
    spdlog::logger& logger;
    InputService& input;
    ForceFeedbackOptions options;

    // The handoff. Written by whoever is stepping the car, read by the writer thread, and **neither
    // side ever waits**: both use `try_lock` and both count what that cost them. A dropped publish
    // is one physics tick the writer interpolates through; a contended take is one output frame the
    // writer spends on the state it already had.
    mutable std::mutex publication;
    RackFeedback publishedState{};

    // **The pedals ride on the rack's publish and are otherwise nothing to do with it.** One publish
    // because they come off the same tick and a driver's feet and hands are being told about one
    // instant; two structs because they are two devices, and the day the pedals are on their own USB
    // cable — which is the only way their motors are reachable at all — that is literally true.
    //
    // Stage one only. Turning this into motor codes is the writer thread's job, because the profile
    // that knows what a motor is belongs on the device's side of the line and not the car's.
    PedalFeedback publishedPedals{};
    PedalFeedback newestPedals{};
    // What was last written, so the motors are only told when the answer changes and at a rate an
    // eccentric mass can render. See `PedalMotorProfile::updateHz`.
    PedalMotorCommand lastPedalCommand{};
    std::chrono::steady_clock::time_point lastPedalWrite{};
    bool pedalsSilenced = true;
    std::uint64_t publishedNanos = 0;
    std::uint64_t publishedSequence = 0;

    std::atomic<std::uint64_t> droppedPublishes{0};
    std::atomic<std::uint64_t> contendedTakes{0};
    std::atomic<std::uint64_t> nanRejections{0};
    std::atomic<std::uint64_t> writeRefusals{0};

    std::atomic<bool> hasFocus{true};
    std::atomic<bool> engagedNow{false};
    std::atomic<double> writeRate{0.0};
    // What the writer last committed, so the trace can carry stage two beside stage one without the
    // physics thread ever touching the writer's state.
    std::atomic<double> lastRequested{0.0};
    std::atomic<double> lastCommandedTorque{0.0};
    std::atomic<double> lastDelivered{0.0};
    std::atomic<double> lastLatency{0.0};
    std::atomic<bool> lastClipped{false};

    // Writer thread only, past construction.
    RackFeedback newest{};
    RackFeedback older{};
    std::uint64_t newestNanos = 0;
    std::uint64_t olderNanos = 0;
    std::uint64_t takenSequence = 0;
    bool haveNewest = false;
    bool haveOlder = false;
    // How many rejections this thread has already answered. A count rather than a flag, so a value
    // that is not a number always costs at least one released output frame even when the very next
    // publish is finite again: a latch some other thread could clear before this one looked at it
    // would make "zero within one frame" a race.
    std::uint64_t nanHandled = 0;
    bool engaged = false;
    double commanded = 0.0;
    // The rim's measured rate, radians per second, for the mapping's damper. Differenced over the
    // *device's own report timestamps* and never over this thread's frames: a wheel nobody touches
    // reports at 43 Hz, so an output frame that differenced a stale-then-fresh angle over its own
    // two milliseconds would read one report's worth of motion as eleven times the speed — the
    // same instrument fault the latency histogram had, from the same idle interval.
    double rimRadians = 0.0;
    std::uint64_t rimStampNanos = 0;
    // The first of the estimator's two poles; `rimSpeed` is the second and is what the damper uses.
    double rimSpeedRaw = 0.0;
    double rimSpeed = 0.0;
    bool haveRimAngle = false;
    // The input report the latency histogram has already accounted for, so a sample is measured once
    // rather than once per output frame it survives into.
    std::uint64_t measuredInputNanos = 0;
    // The longest publish gap seen lately, decaying towards the newest one so that a single hitch
    // relaxes out of it. The watchdog is measured against this rather than against an assumed
    // cadence: what the engine actually delivers is one burst per rendered frame, and how long a
    // frame is is not this service's to assume.
    double publishGapNanos = 0.0;
    // The engage envelope as a **level** that rises while engaged and falls while released, rather
    // than a stopwatch restarted from zero on every re-engagement. A hitch then costs its own
    // duration of ramp instead of the whole second, while a cold start still takes the full ramp —
    // the safety rule is about engaging from nothing, and a wheel that was holding four newton
    // metres two milliseconds ago is not engaging from nothing.
    double rampLevel = 0.0;
    std::uint64_t frames = 0;
    std::uint64_t firstFrameNanos = 0;
    ClipHistogram clipping;
    LatencyHistogram latencies;

    // Physics thread and `takeTrace` only. Deliberately not the writer's: a recorder the output
    // thread also touched would be a lock the simulation could find held.
    mutable std::mutex traceLock;
    RackTorqueRecorder trace;
    std::uint64_t traceEpochNanos = 0;
    // **The trace's own count, and it is deliberately not `publishedSequence`.** That one is the
    // freshness handshake with the writer thread and only advances when the `try_lock` below is
    // taken; a tick whose handoff missed used to be written into the trace carrying a literal zero,
    // because the local was initialised to zero and only filled inside the branch that got the lock.
    // In a 200 s session that produced exactly one such row — a real, correctly computed tick, on
    // cadence, interpolating cleanly between its neighbours in all 61 physics columns, labelled 0 —
    // and it reads to any gap analysis as *two* discontinuities in the middle of the run. That is
    // the one column whose whole job is to say nothing was lost. This counter is incremented under
    // `traceLock`, which is held unconditionally, so a gap in it means a lost trace record and
    // nothing else. Whether a publish reached the writer is a different fact and is counted by
    // `droppedPublishes`, which the shutdown report already prints.
    std::uint64_t traceSequence = 0;

    std::mutex waiting;
    std::condition_variable_any wake;

    void pump(const std::stop_token& stopToken);
    void take();
    [[nodiscard]] double reconstruct(std::uint64_t nowNanos) const;
    void release();
    void writePedals(std::chrono::steady_clock::time_point now);

    // Last, so it stops and joins before anything it touches on the way down is destroyed.
    std::jthread writer;

public:
    ForceFeedbackService(spdlog::logger& logger, InputService& input, ForceFeedbackOptions options);
    ForceFeedbackService(const ForceFeedbackService&) = delete;
    ForceFeedbackService(ForceFeedbackService&&) = delete;
    ForceFeedbackService& operator=(const ForceFeedbackService&) = delete;
    ForceFeedbackService& operator=(ForceFeedbackService&&) = delete;
    ~ForceFeedbackService();

    // Once per physics tick, by whoever computed stage one. Never blocks.
    void publish(const RackFeedback& feedback);

    // Stage one for the pedals, from the same tick. Separate from the rack's publish because a car
    // that has no opinion about its pedals — or a game that has not wired them — should not have to
    // say so every tick, and because the two are different devices.
    void publishPedals(const PedalFeedback& feedback);

    // What the pedal motors are set to. Stated in the codes the device takes, because that is what
    // was written; a trace of a cue nobody felt is answered by this rather than by the severity that
    // asked for it.
    [[nodiscard]] PedalMotorCommand pedalMotors() const
    {
        return lastPedalCommand;
    }

    // The application lost the keyboard, so it has no business holding somebody's hands either. The
    // next output frame lets go, and the ramp starts again when focus returns.
    void setFocus(bool focused);

    // One output frame, evaluated at the given instant. Public because the thread is not the
    // interesting part: every safety rule in here is a property of this function, and a rule that
    // could only be tested by starting a thread and waiting is a rule nobody tests.
    ForceCommand step(std::chrono::steady_clock::time_point now, double deltaTime);

    [[nodiscard]] bool active() const
    {
        return engagedNow.load();
    }

    // What this service is writing at, measured over the run. It is the **write** rate and says so:
    // what the device consumes is set by the driver's own timer and cannot be observed from here at
    // all, which is exactly why `outputHz` is paced to a figure taken from the driver rather than
    // from the transport.
    [[nodiscard]] double writeRateHz() const
    {
        return writeRate.load();
    }

    [[nodiscard]] const ClipHistogram& clipHistogram() const
    {
        return clipping;
    }

    [[nodiscard]] const LatencyHistogram& latencyHistogram() const
    {
        return latencies;
    }

    void setMapping(const ForceMapping& mapping)
    {
        options.mapping = mapping;
    }

    [[nodiscard]] const ForceMapping& mapping() const
    {
        return options.mapping;
    }

    // Oldest first. Ask for it when the car is not ticking: it copies the ring under the lock the
    // simulation writes through.
    [[nodiscard]] std::vector<RackTorqueFrame> takeTrace() const;

    // One line, for the shutdown log. The clipping histogram is in it because a gain nobody
    // measured is a gain somebody guessed.
    [[nodiscard]] std::string report() const;
};

} // namespace raceengine

namespace raceengine
{

// Between two physics samples, and past the newest one for as long as one sample's worth of change.
//
// **This is a first-order hold with a bounded excursion, and it is not what the obvious reading of
// the brief asks for.** The brief offers interpolating between physics ticks or deriving the force
// from interpolated state, both of which lag by one sample; that is the right answer when the
// output is slower than the physics, which is the usual case and is what it assumes. It is not the
// case here. The vehicle substeps at 360 Hz but it does so *inside* a 120 Hz engine tick, so what
// reaches this thread is a burst every 8.33 ms however fast the model is integrated — and a lagging
// interpolation would spend the whole of an 8.33 ms budget before the wheel was written at all,
// against a ten millisecond end-to-end target that already has the tick's own input sampling in it.
//
// So the reconstruction runs forward instead, and is bounded so that it can never invent anything:
// the excursion is clamped to the change the physics itself last made over one sample interval.
// Between publishes the output continues the last slope and then holds; it cannot overshoot a step
// by more than the step, and if the simulation stops it saturates at one tick's worth of change and
// the watchdog lets go a few milliseconds later.
//
// **The slope's denominator is the interval between the two samples this thread actually holds**
// (2026-08-21, second correction). It was the wall clock, which on a catch-up burst is microseconds
// and gave a slope hundreds of times too steep. It was then the publisher's stated tick, which is
// nearer and still wrong, and the reason is worth stating because it is not obvious from either
// side alone: **this thread does not see consecutive ticks.** The engine runs its ticks in a burst
// — measured on the rig, three or four of them inside a quarter of a millisecond, once per rendered
// frame — and `publish` overwrites a single slot, so a writer sampling at 500 Hz takes the *last*
// tick of each burst and never sees the others. `newest` and `older` are therefore one **frame**
// apart, not one tick: 27 ms of simulated time against a stated 8.33, so the slope came out 3.4×
// too steep, saturated the excursion clamp within the first third of every frame, and sat at
// newest-plus-one-whole-frame's-change for the remaining two thirds. A 100% overshoot held for
// most of every frame is a lead term with a gain of two at the frame rate, which is how a limit
// cycle is fed rather than damped.
//
// So the span is measured, floored at one stated tick so that a take landing inside a burst cannot
// divide by a microsecond, and the excursion is a **fraction** of the last change rather than the
// whole of it. Replayed against the rig's own 300-second trace, the shipped form carried 8128 N·m
// of total variation and 2.4× the 5–40 Hz energy of the physics it was reconstructing; holding
// carries 3441 and 1.0×. Prediction in a loop closed through a human's hands is positive feedback,
// and the honest answer to a stale sample is to hold it — so `extrapolation` defaults to zero and
// is a dial rather than a constant, because what it buys is latency and what it costs is phase
// margin, and only the seat can weigh those.

// The pedals' motors, on the same thread and deliberately not at the same rate.
//
// **An eccentric mass has to spin up and down through its own inertia**, which takes tens of
// milliseconds, so writing at the wheel's five hundred a second would be several hundred USB control
// transfers a second the hardware cannot render — on the same pipe the force feedback is using, for
// a cue that would look identical at a tenth of the rate. It is written when the answer changes and
// no more often than `PedalMotorProfile::updateHz`.

} // namespace raceengine
