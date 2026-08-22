module;

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

export module raceengine.input:ForceMapping;

namespace raceengine
{

// Stage two of three: newton metres at the rim into what one particular base can make of them.
//
// **Every hardware compensation in this pipeline lives here and only here.** Stage one is the car
// and stage three is a syscall; this is the only place that knows the base is eight newton metres,
// that it takes two hundred and fifty six codes, or that its driver reads a signed sixteen-bit level
// and throws eight of those bits away. Anything that starts by saying "the wheel feels..." belongs
// in this file or it belongs nowhere.
//
// The compression is **linear with hard clipping**, and that is settled rather than provisional. A
// soft knee spends codes making the approach to the limit gradual, and this device has 256 of them
// across its whole range — a quarter of a newton metre wide each side of a knee is four codes. What
// a knee would buy is a smoother arrival at a limit; what it costs is that no torque anywhere in the
// range means what it says any more, which is the property the whole three-stage split exists to
// keep. Clip, and *count the clipping*, so the gain can be set from a measurement.

// What this device can do, as the device's own profile states it. Nothing here is discovered at
// runtime: evdev reports no torque rating and DirectInput reports a per-effect magnitude, so the
// number comes from the file keyed on the device's identity.
export struct DeviceForceProfile
{
    // N·m at the rim for a torque fraction of one. `DeviceProfile::peakTorque`.
    double peakTorque = 8.0;

    // The signed level a constant-force effect carries. This is evdev's constant, not the device's:
    // `ff_effect::u.constant.level` is an `__s16`, and the backend scales a fraction onto it.
    double levelFullScale = 32767.0;

    // How many of those sixteen bits the device actually receives. Eight on this base: the driver's
    // high-resolution path is gated behind a quirk this product id does not carry, so what reaches
    // the wire is `(level + 0x8000) >> 8` — 256 codes over the full range, one code every 0.39% of
    // peak, which at 8 N·m is 62.5 mN·m. Quantising here deliberately rather than letting the driver
    // do it silently is what stops the step being met later as apparent stiction on centre.
    std::uint32_t quantisationBits = 8;
};

// The level step one device code spans, in the units `ff_effect` is written in.
export [[nodiscard]] inline double levelStep(const DeviceForceProfile& profile)
{
    const auto bits = std::min<std::uint32_t>(profile.quantisationBits, 16u);

    return static_cast<double>(std::uint32_t{1} << (16u - bits));
}

// One code, in newton metres. The smallest change in torque this device can be asked for.
export [[nodiscard]] inline double quantisationStep(const DeviceForceProfile& profile)
{
    return 2.0 * profile.peakTorque * levelStep(profile) / 65536.0;
}

export struct ForceMapping
{
    // The player's dial, and the only thing here they are expected to move.
    double gain = 1.0;

    // N·m, and **independent of the gain** by construction: it is applied after the gain and is
    // clamped against the device's own peak whatever it says. A wrist is on the other end of this,
    // and a ceiling a gain dial can raise is not a ceiling.
    double ceilingTorque = 8.0;

    // N·m per second at the output. A rate limit and not a filter — it bounds how fast the wheel may
    // change what it is doing and leaves every frequency below that rate completely untouched. There
    // is deliberately **no low-pass anywhere in this pipeline**: a belt drive already attenuates the
    // high frequencies mechanically, and software filtering on top of that removes the little
    // texture that survives.
    //
    // 400 N·m/s is one full-scale swing in twenty milliseconds, which is well above anything the
    // tyre model produces and low enough that a single bad tick cannot arrive as a hammer blow.
    double slewRate = 400.0;

    // N·m per radian per second of the rim's own measured motion, opposing it. Not a filter either:
    // a damper is a torque computed from a measured state, and it is the one torque a wheel base
    // needs that the car cannot supply. Hands off, the rim is an inertia on the end of the tyre's
    // spring delivered through a sampled pipeline, and the only dissipation in that loop is
    // whatever the base's own bearings give — this base's tuning menu (its NDP damper included) is
    // unreachable under hid-fanatecff, so if a damper exists at all it is this one. The driver's
    // hands mask the need; letting go of the wheel is what exposes it. Zero by default: the dial is
    // the player's, stated per car in the setup sheet.
    double damping = 0.0;

    // **The base's own oscillator, damped here because here is where a number about a wheel base is
    // allowed to live.** Same units as the dial above and summed with it, so the driver's setting
    // rides on top rather than replacing it.
    //
    // Until 2026-08-21 this was 0.4, scheduled down over 5 m/s of *road* speed, and it was published
    // by the car as part of its rack. It was sized as 2·√(K·J) against a `J` of 3.1e-3 kg·m² fitted
    // from the limit cycle **this base** was doing — so a device inertia was reaching the pipeline
    // dressed as a car parameter, which is the one thing the three-stage split exists to prevent.
    //
    // What settled where it belongs is that the simulation has no steering degree of freedom: the
    // rack angle is commanded straight from the driver's demand, so the only inertia physically in
    // the loop this damper closes is the motor, the belt and the rim. The car's own steering inertia
    // is 0.093 kg·m² and would ask for 0.83 — correct about an oscillator that is not being
    // simulated, and a damper sized for a mass that is not there is wrong in whichever direction the
    // numerics happen to point.
    //
    // **It no longer fades with road speed, and that is a change a driver will feel**: this layer
    // does not know how fast the car is going, and telling it would be the same inversion by a
    // different door. If it is too much at speed, the honest fix is a rack with a mass and a
    // compliance in the vehicle model rather than a schedule here.
    double deviceDamping = 0.4;

    // Hz. **The damper's own bandwidth, and it is a stability limit rather than a smoothing taste.**
    //
    // A damper is the one term here that closes a loop through the hardware: it measures the rim,
    // pushes on the rim, and measures the result. Every millisecond between the measurement and the
    // push is phase lag in that loop, and the total — the writer's 2 ms period, the estimator, the
    // driver's 2 ms `hrtimer`, USB, the base's firmware, current rise, belt — came out at **12.7 ms
    // measured on this rig**. A velocity term delivered a quarter period late is a spring, and later
    // than that it is a *negative* spring, so a damper has authority only below about `1/(4τ)` —
    // 20 Hz here — and above it is an exciter.
    //
    // What makes that lethal rather than merely useless is that a damper's torque is `c·ω` and so
    // **rises with frequency**: unfiltered, the highest frequency the loop can sustain always wins,
    // and the oscillation parks itself exactly on the stability boundary. Measured on the 12:51 exit
    // trace, a **0.58 mm** rack wiggle produced **3.1 N·m** of damper against 0.33 N·m of carcass
    // spring — nine tenths of the output at 20 Hz was the damper. Fitting the standard
    // delayed-velocity-feedback result (`c·τ/J = π/2` at onset, `ω = π/(2τ)`) gives τ = 12.7 ms and
    // an effective rim inertia J = 3.1e-3 kg·m², so **the most damping this loop can carry
    // unfiltered is 0.39 N·m per rad/s** — and the car was asking for 0.40. That is why the rig sat
    // exactly on the edge, and why every increase made the shake worse *and faster*: a limit cycle's
    // frequency here goes as `c/J`, and 23.75 Hz at c ≈ 0.52 against 19.75 Hz at c ≈ 0.39 is that
    // relation read straight off the trace.
    //
    // Two poles rather than one, because one leaves the damper's torque asymptotically flat with
    // frequency where two make it fall. At eight hertz the same fit puts the critical damping at
    // **6.6** instead of 0.39 — a seventeenfold margin — for 6% of the damper's magnitude and 28
    // degrees of its phase at the two hertz a released rim actually returns to centre at.
    //
    // This is **not** the low pass the pipeline refuses. That rule is about the car's torque, which
    // is road feel and is still untouched end to end. This filters a *measurement of the device*
    // feeding a synthetic term, and its corner is set by the loop's delay rather than by taste. Zero
    // or less disables the filtering, which is the old behaviour and is kept only so the difference
    // can be demonstrated.
    double damperBandwidth = 8.0;

    // Some wheels' positive axis and some cars' positive rack disagree. Data, not a code path.
    bool invert = false;
};

export struct ForceCommand
{
    // N·m after the gain and before any limit — what the mapping was asked for. This is what the
    // clipping histogram is fed, because what matters is how much torque the car wanted and could
    // not have.
    double requestedTorque = 0.0;
    // N·m after the ceiling, the slew limit and the engage ramp.
    double commandedTorque = 0.0;
    // The **car's** torque after the ceiling, the ramp and the slew limit, and *before* the damper.
    // This is what the next call must be handed as `previousCommanded`: the rate limit belongs to
    // the car's demand alone, so feeding it back the damped output would put the damper inside the
    // limiter it is deliberately outside of. See the slew/damper ordering below.
    double slewedTorque = 0.0;
    // N·m the device will actually make, after its own 256-code grid has been applied.
    double deliveredTorque = 0.0;
    // What `IInputBackend::writeTorque` takes. Never exactly zero while engaged — see below.
    double fraction = 0.0;
    // The device code, 0 to 255 on this base. Kept because it is the one number a quantisation
    // argument can be settled with.
    std::uint32_t code = 0;

    bool clipped = false;
    double excessTorque = 0.0;
    // The input was not a number. The caller lets go of the wheel; it does not pass this on.
    bool rejected = false;
};

// Letting go *and putting the effect away*. Exactly zero, which on this driver is not "zero force"
// but "disable the slot" — and that difference is the whole reason this is a separate function
// rather than `map(..., 0.0)`. For shutting down, not for going quiet.
export [[nodiscard]] inline ForceCommand releaseForce()
{
    return ForceCommand{};
}

// Rack newton metres to a device code, with the four limits in the order a wrist wants them.
//
// `rimSpeed` is the physical rim's own measured rate in radians per second, positive in the same
// sense as positive demand — the damper's input, and it is a *measurement* of the device rather
// than anything the simulation said, which is why it arrives as its own argument instead of riding
// in the torque. `previousCommanded` is what went out last time, which is what the slew limit is
// measured from; `ramp` is the engage envelope, 0 to 1. Pure in all its arguments, so every one of
// the limits can be pinned without a device: that is the point of the mapper being a function and
// the thread being somewhere else.
//
// **The zero code is not zero.** This driver treats `level == 0` as "disable this effect slot"
// rather than as "make no force", so a torque hovering either side of nothing would switch the slot
// off and on again several hundred times a second. The fix is not hysteresis: it is to emit the
// *centre* of the device code rather than its edge. The driver floors a level into a code, so any
// level in [code*step, code*step + step) produces the same code — and the centre of the code that
// means no force is level 128, which is not zero, so the slot stays armed and the device makes
// nothing. Emitting the code's centre also removes the half-code bias the driver's flooring would
// otherwise put on every value, which on a symmetric input is a steady pull to one side.
export [[nodiscard]] inline ForceCommand mapRackTorque(const DeviceForceProfile& profile, const ForceMapping& mapping,
                                                       const double rackTorque, const double rimSpeed,
                                                       const double previousCommanded, const double deltaTime,
                                                       const double ramp)
{
    auto command = ForceCommand{};

    if (!std::isfinite(rackTorque) || !std::isfinite(rimSpeed) || !std::isfinite(previousCommanded) ||
        !std::isfinite(deltaTime) || !std::isfinite(ramp) || !std::isfinite(mapping.gain) ||
        !std::isfinite(mapping.ceilingTorque) || !std::isfinite(mapping.damping))
    {
        command.rejected = true;

        return command;
    }

    const auto peak = std::max(profile.peakTorque, 1e-9);
    // Two ceilings, and the lower always wins: what the player asked to be protected by, and what
    // the hardware can do at all.
    const auto ceiling = std::clamp(mapping.ceilingTorque, 0.0, peak);

    command.requestedTorque = rackTorque * mapping.gain * (mapping.invert ? -1.0 : 1.0);

    const auto magnitude = std::abs(command.requestedTorque);
    command.clipped = magnitude > ceiling;
    command.excessTorque = std::max(0.0, magnitude - ceiling);

    // The damper joins after the gain and the invert, because it lives entirely in the device's own
    // frame — it opposes the measured motion whichever way the car's rack is signed — and **after
    // the ceiling**, then clamped against it again. Applied before the clip it is subtracted from a
    // demand the clip then discards: a standstill's carcass spring asks for twenty times the
    // ceiling, so the sum re-clips to the same bang-bang square wave and the damper never reaches
    // the motor — measured on the rig as a 10–17 Hz full-amplitude shake whose only damping was
    // the driver's grip. Post-clip it always has its authority; the second clamp keeps the wrist's
    // guarantee, and a still rim leaves the path bit-identical. The clip accounting above
    // deliberately excludes it: what the histogram measures is how much torque the *car* wanted
    // and could not have.
    //
    // **And after the slew limit too, which is the same argument one step further** (2026-08-21).
    // A rate limit is a statement about how fast the *car's* demand may change — one bad tick must
    // not arrive as a hammer blow. A damper is not a demand: it is a torque computed from a
    // measurement of the rim taken microseconds ago, and it is the one term in the pipeline whose
    // whole value is that it is in phase with the rim *now*. Rate-limiting it is the classic way to
    // turn a damper into an oscillator — 400 N·m/s is 0.8 N·m per output frame, so a damper asked
    // for c·ω of a shaking rim (c = 0.5 against ω = 40 rad/s is twenty newton metres) is delivered
    // as a triangle wave lagging its own demand by up to a quarter period, and a velocity term
    // ninety degrees late is a spring rather than a damper. That is why the rig got *more violent*
    // as `ffb.damping` went up, which is the opposite of what a damper does and was the clue that
    // found this: the limiter, not the estimator, is what bounded how much damping the loop could
    // carry. The car's torque is still rate limited — the hammer-blow guarantee is untouched,
    // because that guarantee was only ever about the demand — and the damper is added to the
    // limited result and clamped against the ceiling, so the wrist's ceiling still holds.
    const auto limited = std::clamp(command.requestedTorque, -ceiling, ceiling);
    const auto envelope = std::clamp(ramp, 0.0, 1.0);

    const auto target = limited * envelope;
    const auto step = std::max(mapping.slewRate, 0.0) * std::max(deltaTime, 0.0);

    command.slewedTorque = previousCommanded + std::clamp(target - previousCommanded, -step, step);

    // The damper rides the same engage envelope: a base that took a wrist on engagement would take
    // it just as surely through a damper as through a spring.
    const auto damped = command.slewedTorque - std::max(mapping.damping, 0.0) * rimSpeed * envelope;

    command.commandedTorque = std::clamp(damped, -ceiling, ceiling);

    // --- the device's own grid ---
    const auto span = levelStep(profile);
    const auto scale = std::max(profile.levelFullScale, 1.0);
    const auto level = std::clamp(command.commandedTorque / peak, -1.0, 1.0) * scale;
    const auto codes = 65536.0 / span;
    const auto raw = std::floor((level + 32768.0) / span);
    const auto chosen = std::clamp(raw, 0.0, codes - 1.0);

    command.code = static_cast<std::uint32_t>(chosen);

    const auto edge = chosen * span - 32768.0;
    command.deliveredTorque = std::clamp(edge / scale, -1.0, 1.0) * peak;
    command.fraction = std::clamp((edge + span * 0.5) / scale, -1.0, 1.0);

    return command;
}

export inline constexpr std::size_t clipHistogramBins = 16;

// How much of the time the car asked for more torque than the device has, and by how much.
//
// **A primary instrument, not a diagnostic.** Every newton metre past the ceiling is detail the
// driver does not receive, and the limit is exactly where the detail matters — a gain set by feel
// with no measurement of this is a gain set to whatever felt strongest on the corner somebody
// happened to try. The bins run to twice the ceiling because what a gain is being chosen against is
// not whether it clips but how far past it goes and how often.
export class ClipHistogram
{
    std::array<std::uint64_t, clipHistogramBins + 1> counts{};
    std::uint64_t total = 0;
    std::uint64_t clipped = 0;
    double worst = 0.0;

    // How long the ceiling has been refusing something without a break, and what that has amounted
    // to. See `sustainedEpisodes` for why duration and not percentage.
    double currentRun = 0.0;
    double longestRun = 0.0;
    std::uint64_t sustained = 0;

public:
    // The top of the range the bins cover, as a multiple of the ceiling. Everything above it lands
    // in the overflow bin, which is `clipHistogramBins`.
    static constexpr double span = 2.0;

    // How long a clip has to last before it counts as one the driver was denied something by.
    //
    // **This is the number the gain is set from, and it is a duration rather than a percentage.** A
    // kerb strike *should* hit the rail — that is a real event and the wheel saying so is correct. A
    // corner should not, because a corner is where the pneumatic trail collapses and that cue is the
    // most useful thing the tyre says; a quarter of a second of rail is long enough that it can only
    // be a corner. Sweeping the gain and reading this to zero is what set 0.70, twice, by hand.
    static constexpr double sustainedSeconds = 0.25;

    // `deltaSeconds` is how long this sample stands for — the output period. Zero, the default,
    // records the demand and nothing about its duration, which is what every caller that does not
    // care about episodes gets.
    void add(const double requestedTorque, const double ceiling, const double deltaSeconds = 0.0)
    {
        if (!std::isfinite(requestedTorque) || ceiling <= 0.0)
        {
            return;
        }

        const auto fraction = std::abs(requestedTorque) / ceiling;

        total++;
        if (fraction > 1.0)
        {
            clipped++;
        }

        if (std::isfinite(deltaSeconds) && deltaSeconds > 0.0)
        {
            if (fraction > 1.0)
            {
                currentRun += deltaSeconds;
                longestRun = std::max(longestRun, currentRun);

                // Counted once, on the sample that crosses the threshold, rather than at the end of
                // the run — so a session that is still clipping when it is asked has already
                // reported the episode it is in.
                if (currentRun >= sustainedSeconds && currentRun - deltaSeconds < sustainedSeconds)
                {
                    sustained++;
                }
            }
            else
            {
                currentRun = 0.0;
            }
        }

        worst = std::max(worst, fraction);

        const auto scaled = fraction * static_cast<double>(clipHistogramBins) / span;
        const auto bin =
            scaled >= static_cast<double>(clipHistogramBins) ? clipHistogramBins : static_cast<std::size_t>(scaled);

        counts[bin]++;
    }

    [[nodiscard]] std::uint64_t samples() const
    {
        return total;
    }

    // Percentage of time the ceiling refused something, which is the headline number.
    [[nodiscard]] double clippedPercentage() const
    {
        return total == 0 ? 0.0 : 100.0 * static_cast<double>(clipped) / static_cast<double>(total);
    }

    // The largest demand seen, as a multiple of the ceiling. 3.0 says the gain could come down by
    // two thirds and nothing would have been lost that is not already lost.
    [[nodiscard]] double peakDemand() const
    {
        return worst;
    }

    [[nodiscard]] std::uint64_t bin(const std::size_t index) const
    {
        return index < counts.size() ? counts[index] : 0;
    }

    // **The number the gain is chosen by.** How many separate times the ceiling refused something
    // for longer than `sustainedSeconds` without a break. Zero is the target; a kerb strike does not
    // reach it and a corner held against the rail does.
    [[nodiscard]] std::uint64_t sustainedEpisodes() const
    {
        return sustained;
    }

    // The longest unbroken clip, in seconds. Reported beside the count because "one episode" reads
    // very differently at 0.3 s and at 4 s.
    [[nodiscard]] double longestClip() const
    {
        return longestRun;
    }

    void clear()
    {
        counts = {};
        total = 0;
        clipped = 0;
        worst = 0.0;
        currentRun = 0.0;
        longestRun = 0.0;
        sustained = 0;
    }

    // One line, because it is printed at shutdown beside everything else and a table nobody reads is
    // a measurement nobody makes. Bins are named by their upper edge as a multiple of the ceiling.
    [[nodiscard]] std::string report() const
    {
        auto text = std::string("torque demand against the ceiling: ");

        if (total == 0)
        {
            return text + "nothing was asked for";
        }

        auto number = [](const double value, const int precision)
        {
            auto buffer = std::array<char, 32>{};
            const auto written =
                std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, precision);

            return std::string(buffer.data(), written.ptr);
        };

        text += number(clippedPercentage(), 2) + "% of " + std::to_string(total) + " frames clipped, peak demand " +
                number(worst, 2) + "x, " + std::to_string(sustained) + " sustained (longest " +
                number(longestRun, 2) + " s), histogram [";

        for (auto index = std::size_t{0}; index <= clipHistogramBins; index++)
        {
            if (index > 0)
            {
                text += " ";
            }

            text += number(100.0 * static_cast<double>(counts[index]) / static_cast<double>(total), 1);
        }

        return text + "] % per 0.125x band, last is above " + number(span, 1) + "x";
    }
};

// Going quiet without putting the effect away: the device's own code for no force at all, which on
// this grid is the centre of the range and is *not* level zero.
//
// This is what letting go of the wheel should write, and the distinction is not pedantry. A game
// loading its assets stalls the simulation for longer than the watchdog allows, so the service
// engages, starves, releases and re-engages several times a second — and if releasing means level
// zero, that is the effect slot being disabled and re-uploaded several times a second on a driver
// whose documented behaviour is to treat zero as a disable. Measured on the rig: the wheel dropped
// off the USB bus every few seconds with force feedback running, while a read-only probe and the
// calibration tool, which write nothing, held the same port for minutes.
//
// It satisfies every safety rule the brief names, because what those rules require is *no torque*,
// and a device sitting on its zero code makes none.
export [[nodiscard]] inline ForceCommand quietForce(const DeviceForceProfile& profile)
{
    return mapRackTorque(profile, ForceMapping{}, 0.0, 0.0, 0.0, 0.0, 0.0);
}

} // namespace raceengine
