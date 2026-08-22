module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

export module raceengine.input:PedalFeedback;

namespace raceengine
{

// Stage one of three for the pedals, and it is the same three stages the rack goes through.
//
// A ClubSport V3 pedal set has an eccentric-mass motor under the throttle and another under the
// brake. What they are *for* is the one cue a steering wheel physically cannot give: a wheel that
// has stopped rotating with the road under the foot that caused it. The wheel tells you about the
// front axle's grip through the trail; it tells you nothing about a locked rear, and nothing at all
// about wheelspin, because neither reaches the steering.
//
// **Nothing in this file knows what a motor is.** No duty cycles, no eight-bit levels, no sysfs. It
// answers "how badly is the tyre under this pedal past its limit, and is the driver's foot on that
// pedal", between nought and one, and stage two decides what a particular pedal set can make of it.
// The rack's history is why that line is drawn this hard: an assist sized against a wheel base's
// ceiling lived one layer outside stage one for a whole milestone and made the car wrong on every
// other base.
//
// It imports no physics, for the same reason `:RackTorque` does not: what it takes is what a tick
// already produced, as plain numbers, so the whole derivation can be pinned by tests that stand up
// no vehicle at all. The caller does the extraction, because the caller is holding the tick.

// One wheel, as the tick left it.
export struct SlippingWheel
{
    // **Signed, and the sign is the whole of how the two pedals are told apart.** Negative is a wheel
    // turning slower than the road — braking, and at the limit, locking. Positive is a wheel turning
    // faster — driving, and at the limit, spinning.
    double slipRatio = 0.0;

    // Where *this* tyre's longitudinal curve peaks, from `TyreForces::longitudinalPeakSlip`. The
    // thresholds below are multiples of it rather than absolute slips, so a change of compound moves
    // the cue with the grip instead of leaving it where the old tyre's limit used to be.
    double peakSlipRatio = 0.1;

    // Newtons at the patch, and whether there is a patch at all. A wheel in the air spins freely at
    // enormous slip and must be silent: it is not telling the driver anything they can act on.
    double load = 0.0;
    bool inContact = false;

    // Whether the driveline puts torque through this wheel. Only a driven wheel can spin under
    // power; every wheel can lock under braking, which is why one of these is a flag and the other
    // is not.
    bool driven = false;
};

export inline constexpr std::size_t pedalWheelLimit = 4;

// The car's own thresholds, as multiples of the tyre's peak slip.
export struct PedalFeedbackSetup
{
    // **Where the cue starts, and it is deliberately past the peak rather than at it.** At the peak
    // the tyre is making its most force — that is threshold braking, the thing a good driver is
    // trying to sit on, and a pedal that buzzed there would buzz on every good stop and teach the
    // driver to brake less hard. What is worth telling them is that they have gone *past* it, where
    // more pedal buys less braking.
    double onsetPeaks = 1.15;

    // Where each cue reaches full strength, and **the two differ because the quantity is not
    // symmetric**. Slip ratio is bounded below at −1: a locked wheel has stopped, and there is
    // nothing worse than stopped, so braking has about six peaks of range in total and 2.75 of them
    // is a wheel well past saving. Above, it is unbounded — a wheel spinning at four times road
    // speed reads 3.0 — so the same number saturates almost at once.
    //
    // Measured on a standing start (`./EngineTests "[.pedal-cue]"`): a front-drive hatchback at full
    // throttle in first sits between **eight and twenty-four times its peak slip** for two solid
    // seconds. With both cues sharing 2.75 the accelerator's motor pinned at full for the whole of
    // it — that is not a cue, it is a constant, and it is the same failure as a force feedback gain
    // whose top half never means anything.
    //
    // Twelve is chosen so the range covers what a foot can *change*: a scrabble off a damp junction
    // reads two or three peaks and lands gently, a lit-up launch reads twenty and saturates, and
    // backing off moves the motor. Between onset and full it is linear, because what a motor
    // renders is amplitude.
    //
    // **This is the one number here that no measurement settles**, and it is worth saying why rather
    // than leaving it looking derived. Whether a two-second full-spin launch is even *correct* is
    // itself unverified — a real Mk7 does 0–100 in 6.2 s, which is hard to reconcile with the fronts
    // spinning at four times road speed for a third of it — so the range cannot be fitted to that
    // launch without fitting it to a possible fault. It is a feel number and it belongs to whoever
    // is driving, exactly as `ffb.gain` does.
    double brakeFullPeaks = 2.75;
    double throttleFullPeaks = 12.0;

    // A wheel carrying less than this share of the quarter of the car's weight it holds standing
    // still contributes nothing. An unloaded inside wheel on a kerb will show any slip you like and
    // is not the reason the car is not stopping.
    double minimumLoadShare = 0.20;

    // How far the pedal has to be pressed before its motor says anything. **The cue belongs to the
    // foot that caused it**: a rear wheel locked on a trailing throttle is real, and buzzing a brake
    // pedal nobody is standing on is a message to an empty room. It also stops a downshift's engine
    // braking rattling the brake pedal, which is the same fault by a different route.
    double pedalThreshold = 0.05;
};

// What the two motors are being asked for, before anything device-shaped has touched it.
export struct PedalFeedback
{
    // Nought to one. `brake` is the worst lockup under braking across all four wheels; `throttle` is
    // the worst wheelspin across the driven ones.
    double brake = 0.0;
    double throttle = 0.0;

    // Which wheel each came from, for a trace to name. `pedalWheelLimit` when nothing is slipping.
    std::size_t brakeWheel = pedalWheelLimit;
    std::size_t throttleWheel = pedalWheelLimit;

    // False when anything upstream handed this a value that is not a number. The caller must treat
    // that as a reason to stop the motors rather than as a number to pass on — a stuck-on vibration
    // motor is the worst failure this feature has, because unlike a torque it does not self-centre
    // and unlike a sound it cannot be turned down from the desk.
    bool finite = true;
};

namespace
{

[[nodiscard]] inline double rampBetween(const double value, const double low, const double high)
{
    if (high - low < 1e-9)
    {
        return value >= high ? 1.0 : 0.0;
    }

    return std::clamp((value - low) / (high - low), 0.0, 1.0);
}

} // namespace

// The whole derivation, and it is one ratio and a ramp per wheel.
//
// `staticShare` is a quarter of the car's weight in newtons — the load a wheel carries standing
// still — which is what `minimumLoadShare` is a share *of*. Passed rather than derived because this
// partition does not know what a vehicle is.
export [[nodiscard]] PedalFeedback derivePedalFeedback(const PedalFeedbackSetup& setup,
                                                       const std::span<const SlippingWheel> wheels,
                                                       const double brakePedal, const double throttlePedal,
                                                       const double staticShare)
{
    auto result = PedalFeedback{};

    if (!std::isfinite(brakePedal) || !std::isfinite(throttlePedal) || !std::isfinite(staticShare))
    {
        result.finite = false;

        return result;
    }

    const auto braking = brakePedal >= setup.pedalThreshold;
    const auto driving = throttlePedal >= setup.pedalThreshold;

    for (auto index = std::size_t{0}; index < wheels.size() && index < pedalWheelLimit; index++)
    {
        const auto& wheel = wheels[index];

        if (!std::isfinite(wheel.slipRatio) || !std::isfinite(wheel.peakSlipRatio) || !std::isfinite(wheel.load))
        {
            result.finite = false;

            return result;
        }

        if (!wheel.inContact)
        {
            continue;
        }

        // The load gate, and it is a gate rather than a weight. A wheel that is barely touching is
        // not a quieter version of the same message, it is a different situation: the car is not
        // being slowed by that corner at all, and saying so softly would be saying it wrongly.
        if (staticShare > 1e-6 && wheel.load < setup.minimumLoadShare * staticShare)
        {
            continue;
        }

        const auto peak = std::max(wheel.peakSlipRatio, 1e-9);
        const auto beyond = std::abs(wheel.slipRatio) / peak;

        // **The sign picks the pedal**, and with it which of the two ranges applies. A wheel cannot
        // be locking and spinning at once, so there is no case where one slip feeds both.
        const auto locking = wheel.slipRatio < 0.0;
        const auto severity =
            rampBetween(beyond, setup.onsetPeaks, locking ? setup.brakeFullPeaks : setup.throttleFullPeaks);

        if (severity <= 0.0)
        {
            continue;
        }

        if (locking)
        {
            // Every wheel can lock, not only the braked axle a driver is thinking about — and a
            // locked *rear* is the one that spins the car, which is exactly the one the steering
            // will not tell them about. Worst wheel wins: what a foot needs to know is that
            // something has let go, not the average of four things.
            if (braking && severity > result.brake)
            {
                result.brake = severity;
                result.brakeWheel = index;
            }
        }
        else if (driving && wheel.driven && severity > result.throttle)
        {
            result.throttle = severity;
            result.throttleWheel = index;
        }
    }

    return result;
}

} // namespace raceengine
