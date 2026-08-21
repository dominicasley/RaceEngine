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
    // N·m at the rim. **Always in newton metres**, never a fraction of anything: a trace of this is
    // the physics artefact and has to stay comparable across devices, sessions and future hardware.
    double steeringTorque = 0.0;
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
    // N·m per radian per second of measured rim speed the *simulation* asks for on top of the
    // driver's `ForceMapping::damping` dial. An electric rack schedules its damping by road speed
    // — heavy at parking speeds, where the carcass spring needs ~0.5 to return the released rim
    // without ringing, fading to nothing as the car rolls — and which speed the car is doing is
    // the simulation's fact, not this service's. The dial stays the driver's; this rides on it.
    double damping = 0.0;
    // `steady_clock` nanoseconds, from `DeviceSample::timestampNanos`. Zero when the demand did not
    // come from a device — a keyboard, or the gate's scripted launch — in which case no end-to-end
    // latency is measurable and none is reported.
    std::uint64_t inputTimestampNanos = 0;
};

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

    std::mutex waiting;
    std::condition_variable_any wake;

    void pump(const std::stop_token& stopToken);
    void take();
    [[nodiscard]] double reconstruct(std::uint64_t nowNanos) const;
    void release();

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

namespace
{

[[nodiscard]] std::uint64_t nanosOf(const std::chrono::steady_clock::time_point when)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(when.time_since_epoch()).count());
}

[[nodiscard]] std::string fixed(const double value, const int precision)
{
    auto buffer = std::array<char, 48>{};
    const auto written =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, precision);

    return std::string(buffer.data(), written.ptr);
}

} // namespace

ForceFeedbackService::ForceFeedbackService(spdlog::logger& logger, InputService& input, ForceFeedbackOptions options) :
    logger(logger),
    input(input),
    options(std::move(options)),
    trace(this->options.traceCapacity)
{
    if (this->options.unattended)
    {
        logger.info("Force feedback is not driven on an unattended run");

        return;
    }

    writer = std::jthread([this](const std::stop_token& stopToken) { pump(stopToken); });
}

ForceFeedbackService::~ForceFeedbackService()
{
    writer.request_stop();
    wake.notify_all();
}

void ForceFeedbackService::publish(const RackFeedback& feedback)
{
    const auto now = std::chrono::steady_clock::now();
    const auto stamp = nanosOf(now);

    // Rejected here rather than carried one stage further. A value that is not a number cannot be
    // clamped, cannot be compared and cannot be written, and the one thing it must not do is reach
    // a device — so it latches the wheel free instead, and the ramp brings the force back when
    // finite values resume.
    const auto sane = std::isfinite(feedback.steeringTorque) && std::isfinite(feedback.rackTravel) &&
                      std::isfinite(feedback.rackVelocity) && std::isfinite(feedback.publishInterval) &&
                      std::isfinite(feedback.damping);

    if (!sane)
    {
        nanRejections.fetch_add(1);
    }

    auto sequence = std::uint64_t{0};

    {
        // try_lock, so the simulation can never be held up by the output thread — not even for the
        // hundred nanoseconds a copy takes, because a thread preempted while holding a lock holds it
        // for a scheduler quantum and not for the length of its critical section.
        auto held = std::unique_lock<std::mutex>(publication, std::try_to_lock);
        if (held.owns_lock())
        {
            if (sane)
            {
                publishedState = feedback;
                publishedNanos = stamp;
                publishedSequence++;
            }

            sequence = publishedSequence;
        }
        else
        {
            droppedPublishes.fetch_add(1);
        }
    }

    if (trace.capacity() == 0)
    {
        return;
    }

    // Stage one is recorded whatever happened downstream — no device attached, no thread running,
    // clipping or not. That trace is the artefact; the stage-two columns beside it are this
    // session's hardware answering, and are only meaningful against the stage-one column they sit
    // next to.
    const auto guard = std::lock_guard<std::mutex>(traceLock);

    if (traceEpochNanos == 0)
    {
        traceEpochNanos = stamp;
    }

    trace.record(RackTorqueFrame{.time = static_cast<double>(stamp - traceEpochNanos) * 1e-9,
                                 .sequence = sequence,
                                 .steeringTorque = feedback.steeringTorque,
                                 .rackForce = feedback.rackForce,
                                 .tyreRackForce = feedback.tyreRackForce,
                                 .rackTravel = feedback.rackTravel,
                                 .rackVelocity = feedback.rackVelocity,
                                 .requestedTorque = lastRequested.load(),
                                 .commandedTorque = lastCommandedTorque.load(),
                                 .deliveredTorque = lastDelivered.load(),
                                 .clipped = lastClipped.load(),
                                 .latencyMilliseconds = lastLatency.load()});
}

void ForceFeedbackService::setFocus(const bool focused)
{
    hasFocus.store(focused);
}

void ForceFeedbackService::take()
{
    auto held = std::unique_lock<std::mutex>(publication, std::try_to_lock);
    if (!held.owns_lock())
    {
        contendedTakes.fetch_add(1);

        return;
    }

    if (publishedSequence == takenSequence)
    {
        return;
    }

    older = newest;
    olderNanos = newestNanos;
    haveOlder = haveNewest;

    newest = publishedState;
    newestNanos = publishedNanos;
    takenSequence = publishedSequence;
    haveNewest = true;

    // A decaying peak, so the watchdog sizes itself to the slowest frame in recent memory rather
    // than to the fastest: rising instantly and falling slowly is what makes it a bound and not an
    // average, and an average would let go on exactly the long frame the bound exists for.
    if (haveOlder && newestNanos > olderNanos)
    {
        const auto gap = static_cast<double>(newestNanos - olderNanos);

        publishGapNanos = gap > publishGapNanos ? gap : publishGapNanos + (gap - publishGapNanos) * 0.01;
    }
}

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
double ForceFeedbackService::reconstruct(const std::uint64_t nowNanos) const
{
    if (!haveNewest)
    {
        return 0.0;
    }

    const auto reach = std::clamp(options.extrapolation, 0.0, 1.0);

    if (!haveOlder || reach <= 0.0 || newest.publishInterval <= 0.0 || nowNanos <= newestNanos ||
        olderNanos >= newestNanos)
    {
        return newest.steeringTorque;
    }

    const auto ahead = static_cast<double>(nowNanos - newestNanos) * 1e-9;
    const auto span = std::max(static_cast<double>(newestNanos - olderNanos) * 1e-9, newest.publishInterval);
    const auto change = newest.steeringTorque - older.steeringTorque;
    const auto excursion = std::abs(change) * reach;

    return newest.steeringTorque + std::clamp(change * (ahead / span), -excursion, excursion);
}

void ForceFeedbackService::release()
{
    // The device's zero *code* rather than level zero. What the safety rules require is that the
    // wheel makes no torque, and a device sitting on its zero code makes none — where level zero
    // means "disable the slot" on this driver, and a game that engages and releases several times a
    // second while it loads is then uploading and tearing down an effect at that rate. See
    // `quietForce`.
    engaged = false;
    commanded = 0.0;
    engagedNow.store(false);
    lastRequested.store(0.0);
    lastCommandedTorque.store(0.0);
    lastDelivered.store(0.0);
    lastClipped.store(false);

    if (const auto written = input.writeTorque(quietForce(options.device).fraction); !written)
    {
        writeRefusals.fetch_add(1);
    }
}

ForceCommand ForceFeedbackService::step(const std::chrono::steady_clock::time_point now, const double deltaTime)
{
    const auto stamp = nanosOf(now);

    frames++;
    if (firstFrameNanos == 0)
    {
        firstFrameNanos = stamp;
    }
    else if (stamp > firstFrameNanos)
    {
        writeRate.store(static_cast<double>(frames - 1) * 1e9 / static_cast<double>(stamp - firstFrameNanos));
    }

    take();

    const auto seen = nanRejections.load();
    const auto rejected = seen != nanHandled;
    nanHandled = seen;

    const auto link = input.deviceLink();

    // The floor the caller stated, raised to a few of whatever the publisher is actually managing.
    // A flat floor is a frame rate limit in disguise: thirty milliseconds is one frame at 33 fps.
    const auto staleNanos = std::max(static_cast<double>(options.staleTimeout.count()) * 1e6,
                                     std::max(options.staleFrames, 1.0) * publishGapNanos);
    const auto stale = !haveNewest || stamp < newestNanos || static_cast<double>(stamp - newestNanos) > staleNanos;

    // The four ways this ends with a free wheel, checked before anything is computed rather than
    // after: no device or a device that takes no torque, an application nobody is looking at, a
    // simulation that has stopped talking, and a number that is not one.
    if (!link.connected || !link.takesTorque || !hasFocus.load() || rejected || stale)
    {
        if (engaged)
        {
            // Named, because "the wheel went light" has five causes and four of them are this
            // service doing its job. Only ever on a transition, so it cannot become a log per
            // output frame.
            logger.info("Force feedback let go of the wheel: {}",
                        !link.connected     ? "the device is no longer there"
                        : !link.takesTorque ? "the device takes no constant force"
                        : !hasFocus.load()  ? "the window lost focus"
                        : rejected          ? "something upstream produced a value that is not a number"
                                            : "the simulation stopped publishing");

            release();
        }

        // Falls at the rate it rises. A wheel let go for two milliseconds comes back at very nearly
        // the force it left at; one let go for the length of the ramp comes back from nothing.
        const auto rampSeconds = static_cast<double>(options.engageRamp.count()) * 1e-3;
        rampLevel = rampSeconds <= 0.0 ? 0.0 : std::max(0.0, rampLevel - std::max(deltaTime, 0.0) / rampSeconds);

        return releaseForce();
    }

    if (!engaged)
    {
        engaged = true;
        engagedNow.store(true);
        commanded = 0.0;

        logger.info("Force feedback engaged on {:04x}:{:04x}, {:.1f} N.m peak, coming up over {} ms",
                    link.identity.vendor, link.identity.product, link.peakTorque, options.engageRamp.count());
    }

    const auto rampSeconds = static_cast<double>(options.engageRamp.count()) * 1e-3;
    rampLevel = rampSeconds <= 0.0 ? 1.0 : std::min(1.0, rampLevel + std::max(deltaTime, 0.0) / rampSeconds);

    const auto ramp = rampLevel;

    const auto torque = reconstruct(stamp);

    if (!std::isfinite(torque))
    {
        nanHandled = nanRejections.fetch_add(1) + 1;
        release();

        return releaseForce();
    }

    // The device's own peak, from its profile rather than from this service's defaults, so a base
    // rated differently is a file and not a rebuild.
    auto profile = options.device;
    if (link.peakTorque > 0.0)
    {
        profile.peakTorque = link.peakTorque;
    }

    // The rim's rate, for the damper: fresh report against fresh report on the device's own stamps,
    // then **two poles at `ForceMapping::damperBandwidth`**. Between reports the estimate decays
    // rather than holds — a wheel that stops reporting is a wheel that stopped moving, and a held
    // estimate would have the damper leaning on a rim that is standing still.
    //
    // The filter is the damper's stability limit and not smoothing; `damperBandwidth` carries the
    // whole derivation, including why the previous two-millisecond blend was the wrong direction.
    // Briefly: the damper's torque rises with frequency and the loop's 12.7 ms of delay can only
    // serve it below about 20 Hz, so a wide-open estimator hands the loop its own exciter and the
    // oscillation parks on the boundary. The corner has to be well under that, and the decay
    // between reports uses the same time constant so the two halves of the estimator agree.
    const auto damperTau =
        options.mapping.damperBandwidth > 0.0 ? 1.0 / (6.283185307179586 * options.mapping.damperBandwidth) : 0.0;

    if (link.hasRim && link.sampleTimestampNanos != 0)
    {
        const auto angle = link.rimDegrees * 0.017453292519943295;

        if (haveRimAngle && link.sampleTimestampNanos > rimStampNanos)
        {
            const auto interval = static_cast<double>(link.sampleTimestampNanos - rimStampNanos) * 1e-9;
            const auto instant = (angle - rimRadians) / interval;
            const auto blend = damperTau > 0.0 ? interval / (interval + damperTau) : 1.0;

            rimSpeedRaw += (instant - rimSpeedRaw) * blend;
            rimSpeed += (rimSpeedRaw - rimSpeed) * blend;
        }
        else if (haveRimAngle)
        {
            const auto decay = deltaTime / (deltaTime + std::max(damperTau, 0.020));

            rimSpeedRaw -= rimSpeedRaw * decay;
            rimSpeed -= rimSpeed * decay;
        }

        if (link.sampleTimestampNanos >= rimStampNanos)
        {
            rimRadians = angle;
            rimStampNanos = link.sampleTimestampNanos;
            haveRimAngle = true;
        }
    }
    else
    {
        rimSpeed = 0.0;
        rimSpeedRaw = 0.0;
        haveRimAngle = false;
    }

    // The car's scheduled damping joins the driver's dial here, clamped so a publisher cannot
    // subtract the driver's own setting away.
    auto mapping = options.mapping;
    mapping.damping += std::max(newest.damping, 0.0);

    const auto command = mapRackTorque(profile, mapping, torque, rimSpeed, commanded, deltaTime, ramp);

    if (command.rejected)
    {
        nanHandled = nanRejections.fetch_add(1) + 1;
        release();

        return releaseForce();
    }

    // The **slewed car torque**, not the damped output: the rate limiter is the car's and the
    // damper sits outside it, so feeding the damped value back would put the damper inside the
    // limiter it was deliberately moved out of. See `mapRackTorque`.
    commanded = command.slewedTorque;
    clipping.add(command.requestedTorque, std::clamp(options.mapping.ceilingTorque, 0.0, profile.peakTorque));

    if (const auto written = input.writeTorque(command.fraction); !written)
    {
        writeRefusals.fetch_add(1);
    }

    const auto done = nanosOf(std::chrono::steady_clock::now());

    // **Once per input sample, not once per output frame**, and that distinction is the whole
    // measurement. The far end of this interval is the timestamp evdev put on the device's own
    // report, so what it measures for a *fresh* sample is exactly what the criterion asks: the axis
    // being read to the torque being written.
    //
    // Recorded every frame it measured something else entirely. A wheel nobody is touching sends no
    // reports at all — 43 Hz idle against 1000 Hz while it is moving — so the writer spends hundreds
    // of frames re-measuring the *same* report as it ages, and the histogram fills with the device's
    // idle interval rather than with any latency. Measured on the rig that read as a median of 16.75
    // ms and a worst of 243 ms against a 10 ms budget, which is a failure of the instrument.
    if (newest.inputTimestampNanos != 0 && newest.inputTimestampNanos != measuredInputNanos &&
        done > newest.inputTimestampNanos)
    {
        measuredInputNanos = newest.inputTimestampNanos;

        const auto milliseconds = static_cast<double>(done - newest.inputTimestampNanos) * 1e-6;
        latencies.add(milliseconds);
        lastLatency.store(milliseconds);
    }

    lastRequested.store(command.requestedTorque);
    lastCommandedTorque.store(command.commandedTorque);
    lastDelivered.store(command.deliveredTorque);
    lastClipped.store(command.clipped);

    return command;
}

void ForceFeedbackService::pump(const std::stop_token& stopToken)
{
    // Paced to what the driver consumes and not to what the wire allows. Clamped into the band the
    // brief names because outside it the thread is either audible as a stepped force or is spending
    // a core on writes nothing reads.
    const auto rate = std::clamp(options.outputHz, 300.0, 1000.0);
    const auto period = std::chrono::nanoseconds(static_cast<std::int64_t>(1e9 / rate));

    auto next = std::chrono::steady_clock::now();
    auto previous = next;

    while (!stopToken.stop_requested())
    {
        const auto now = std::chrono::steady_clock::now();
        const auto deltaTime = std::chrono::duration<double>(now - previous).count();
        previous = now;

        static_cast<void>(step(now, deltaTime > 0.0 ? deltaTime : 1.0 / rate));

        next += period;

        // Behind by more than a period: resync rather than sprint. Catching up would write a burst
        // the driver's timer cannot consume anyway, and the burst is what a stall would turn into a
        // jolt.
        if (next < now)
        {
            next = now + period;
        }

        auto held = std::unique_lock<std::mutex>(waiting);
        static_cast<void>(wake.wait_until(held, stopToken, next, [] { return false; }));
    }

    // The wheel is let go on the way out, and it is the last thing this thread does. A process that
    // exited holding a torque leaves the base holding it until something else claims the node.
    release();
}

std::vector<RackTorqueFrame> ForceFeedbackService::takeTrace() const
{
    const auto guard = std::lock_guard<std::mutex>(traceLock);

    return trace.inOrder();
}

std::string ForceFeedbackService::report() const
{
    auto text = std::string("force feedback: ");

    text += fixed(writeRateHz(), 1) + " Hz written";

    if (latencies.samples() > 0)
    {
        text += ", latency median " + fixed(latencies.percentile(0.5), 2) + " ms, 99th " +
                fixed(latencies.percentile(0.99), 2) + " ms, worst " + fixed(latencies.peak(), 2) + " ms";
    }

    text += ", " + std::to_string(droppedPublishes.load()) + " publishes dropped, " +
            std::to_string(contendedTakes.load()) + " takes contended, " + std::to_string(nanRejections.load()) +
            " values rejected as not a number, " + std::to_string(writeRefusals.load()) + " writes the device declined";

    return text + "; " + clipping.report();
}

} // namespace raceengine
