// ForceFeedbackService bodies. Declarations are in Input/Services/ForceFeedbackService.cppm.
//
// A **module implementation unit** — `module raceengine.input;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
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

module raceengine.input;

import :ForceFeedbackService;
import :ForceMapping;
import :PedalFeedback;
import :PedalMotors;
import :InputBackend;
import :InputService;
import :RackTorque;

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
                      std::isfinite(feedback.unassistedTorque);

    if (!sane)
    {
        nanRejections.fetch_add(1);
    }

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

    traceSequence++;

    trace.record(RackTorqueFrame{.time = static_cast<double>(stamp - traceEpochNanos) * 1e-9,
                                 .sequence = traceSequence,
                                 .steeringTorque = feedback.unassistedTorque,
                                 .assistedTorque = feedback.steeringTorque,
                                 .rackForce = feedback.rackForce,
                                 .tyreRackForce = feedback.tyreRackForce,
                                 .rackTravel = feedback.rackTravel,
                                 .rackVelocity = feedback.rackVelocity,
                                 .requestedTorque = lastRequested.load(),
                                 .commandedTorque = lastCommandedTorque.load(),
                                 .deliveredTorque = lastDelivered.load(),
                                 .clipped = lastClipped.load(),
                                 .latencyMilliseconds = lastLatency.load(),
                                 .vehicle = feedback.vehicle});
}

void ForceFeedbackService::publishPedals(const PedalFeedback& feedback)
{
    // Same asymmetry as the rack's: the tick never waits, and a dropped publish costs the motors one
    // update of freshness. They are rendered at 60 Hz against a tick at 360, so a miss is invisible.
    auto held = std::unique_lock<std::mutex>(publication, std::try_to_lock);
    if (held.owns_lock())
    {
        publishedPedals = feedback;
    }
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
    newestPedals = publishedPedals;
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

void ForceFeedbackService::writePedals(const std::chrono::steady_clock::time_point now)
{
    // **Two gates, and they say different things.** The profile says this pedal set has motors at
    // all — CSL Elite and CSL LC have none, and no amount of cabling changes that. The device link
    // says they are attached *and reachable*, which on Linux means the pedals are on their own USB
    // cable rather than the base's RJ12, because that is the only wiring the driver exposes a
    // control for. Either alone would be a guess.
    if (!options.pedals.hasMotors || !input.deviceLink().hasPedalMotors)
    {
        return;
    }

    const auto command = mapPedalFeedback(options.pedals, options.pedalMapping, newestPedals);

    // Silence goes out immediately and unconditionally. Everything else waits its turn: a cue that
    // arrives a sixtieth of a second late is a cue, and a motor that stops a sixtieth of a second
    // late against a driver who has already caught the slide is a motor buzzing about nothing.
    const auto stopping = command.silent() && !pedalsSilenced;
    const auto period = std::chrono::duration<double>(1.0 / std::max(options.pedals.updateHz, 1.0));

    if (!stopping)
    {
        if (command.word() == lastPedalCommand.word())
        {
            return;
        }

        if (now - lastPedalWrite < std::chrono::duration_cast<std::chrono::steady_clock::duration>(period))
        {
            return;
        }
    }

    if (const auto written = input.writePedalMotors(command.throttle, command.brake); !written)
    {
        // The pedals are a separate device and may be unplugged while the wheel keeps working. The
        // backend drops its own capability when that happens, so this is counted and not repeated.
        writeRefusals.fetch_add(1);

        return;
    }

    lastPedalCommand = command;
    lastPedalWrite = now;
    pedalsSilenced = command.silent();
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

    // **And stop the motors, which is not the same statement as writing no torque.** A torque
    // settles the moment the effect is removed; an eccentric mass keeps turning until it is told
    // nought, so every path that lets go of the wheel has to say so separately. Unconditional
    // rather than filtered by `lastPedalCommand`, because the whole reason this is being called is
    // that something has gone wrong and what the service believes it last wrote is exactly what is
    // now in doubt.
    lastPedalCommand = PedalMotorCommand{};
    pedalsSilenced = true;

    if (options.pedals.hasMotors)
    {
        // Not gated on the link: the whole reason this is being called may be that the device has
        // gone, and a write that refuses costs nothing where a motor left turning costs the driver
        // a rig they have to unplug.
        static_cast<void>(input.writePedalMotors(0, 0));
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

    // The base's own damping joins the driver's dial here, clamped so a configuration cannot
    // subtract the driver's setting away. It used to be the *car's* request arriving through
    // `RackFeedback` — see there for why a number fitted from a wheel base's limit cycle had no
    // business travelling that way.
    auto mapping = options.mapping;
    mapping.damping += std::max(mapping.deviceDamping, 0.0);

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
    // The output period goes in with it, so the histogram can say how *long* a clip lasted rather
    // than only how often one happened — which is the discriminator the gain is actually set by.
    clipping.add(command.requestedTorque, std::clamp(options.mapping.ceilingTorque, 0.0, profile.peakTorque),
                 deltaTime);

    if (const auto written = input.writeTorque(command.fraction); !written)
    {
        writeRefusals.fetch_add(1);
    }

    writePedals(now);

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
