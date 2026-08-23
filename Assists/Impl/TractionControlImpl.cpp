module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

module raceengine.assists;

namespace raceengine
{

namespace
{

// A first-order lag stated as the time it takes to close, rather than as a per-tick coefficient.
// Exponential rather than linear because that is what a throttle body following a demand does, and
// because a linear ramp's rate depends on the step size it was calibrated at.
[[nodiscard]] double approach(const double value, const double target, const double timeConstant,
                              const double deltaTime)
{
    if (timeConstant <= 0.0)
    {
        return target;
    }

    return value + (target - value) * (1.0 - std::exp(-deltaTime / timeConstant));
}

// Drive slip: how much faster the wheel is turning than the road is going, as a fraction. The
// negative of the braking slip `estimatedSlip` reports, taken through that function rather than
// written again so the two cannot end up with different divisors.
[[nodiscard]] double driveSlip(const double referenceSpeed, const double wheelSpeed)
{
    return -estimatedSlip(referenceSpeed, wheelSpeed);
}

// Which wheel on each side the engine does not drive. Those two are a road-speed measurement that
// owes the engine nothing, and their difference is the yaw rate the road is imposing.
[[nodiscard]] std::size_t undrivenOn(const TractionSetup& setup, const WheelSpeedReadings& wheels,
                                     const std::size_t near, const std::size_t far)
{
    if (!setup.driven[near] && wheels[near].valid)
    {
        return near;
    }

    if (!setup.driven[far] && wheels[far].valid)
    {
        return far;
    }

    return wheelCount;
}

// The brake channel, as an electronic differential lock.
//
// **The signal is the driven pair's own speed difference, corrected for the turn the car is in, and
// nothing else** — no reference speed appears anywhere in it. Writing it out shows why the
// correction is only one term: with `expected = mean -/+ yaw * halfFront` on each side, the *mean*
// cancels in the difference and what is left is the yaw term over the front track. So this cannot
// be fooled by both undriven wheels being wrong together, only by them being wrong differently,
// which is the same limitation `advanceCorneringBrake` has and is a property of the sensor set.
void advanceDifferentialLock(const TractionSetup& setup, TractionState& state, const WheelSpeedReadings& wheels,
                             const ReferenceSpeedSetup& reference,
                             const std::array<double, wheelCount>& brakePeakTorque, const double referenceSpeed,
                             const double referenceCoasting)
{
    if (!wheels[frontLeft].valid || !wheels[frontRight].valid || setup.rearTrack <= 0.0)
    {
        return;
    }

    // **Plausibility, and it is the estimator's own opinion rather than a second guess at it.** When
    // the reference is being carried by its rate limit the wheels have stopped meaning anything —
    // all four in the air over a crest is the case that showed up in the seat — and a differential
    // lock reading the difference between four numbers that are all wrong is braking at random.
    if (referenceCoasting > setup.brakePlausibilityHold)
    {
        return;
    }

    const auto leftUndriven = undrivenOn(setup, wheels, rearLeft, frontLeft);
    const auto rightUndriven = undrivenOn(setup, wheels, rearRight, frontRight);

    // **No undriven pair, no differential lock.** On four-wheel drive there is no free road-speed
    // measurement and therefore no way to tell wheelspin from a corner, so the channel withdraws and
    // the engine keeps the whole job. That is a real limitation of a real sensor set rather than a
    // simplification, and it is why traction control on a front-drive car is the easy case.
    if (leftUndriven >= wheelCount || rightUndriven >= wheelCount)
    {
        return;
    }

    const auto drivenLeft = setup.driven[frontLeft] ? frontLeft : rearLeft;
    const auto drivenRight = setup.driven[frontRight] ? frontRight : rearRight;

    if (!setup.driven[drivenLeft] || !setup.driven[drivenRight])
    {
        return;
    }

    const auto yaw = (std::abs(sensedRoadSpeed(reference, wheels[rightUndriven])) -
                      std::abs(sensedRoadSpeed(reference, wheels[leftUndriven]))) /
                     setup.rearTrack;

    const auto measured = std::abs(sensedRoadSpeed(reference, wheels[drivenRight])) -
                          std::abs(sensedRoadSpeed(reference, wheels[drivenLeft]));

    // Positive means the right-hand driven wheel is the one running away.
    const auto departure = measured - yaw * setup.frontTrack;

    const auto magnitude = std::abs(departure) - setup.brakeDeadband;
    if (magnitude <= 0.0)
    {
        return;
    }

    // Speed limited and faded, because everything this channel does becomes heat in one disc.
    //
    // **Off the reference estimate rather than off the wheels**, and that is the one place in this
    // channel where the reference belongs. The *signal* must not touch it — that is the whole design
    // — but a gate wants the quantity that survives the wheels being wrong, and the estimate is
    // bounded at 1.3 g where a raw wheel reading can go to zero in a tick. Taken off the wheels, a
    // car landing at 126 km/h with all four stopped reads as walking pace and the limit lets the
    // channel act at twice the speed it is allowed to.
    const auto roadSpeed = std::abs(referenceSpeed);

    const auto fade = setup.brakeSpeedFade > 0.0
                          ? std::clamp((setup.brakeSpeedLimit - roadSpeed) / setup.brakeSpeedFade, 0.0, 1.0)
                          : (roadSpeed < setup.brakeSpeedLimit ? 1.0 : 0.0);

    if (fade <= 0.0)
    {
        return;
    }

    const auto spinning = departure > 0.0 ? drivenRight : drivenLeft;
    const auto peak = brakePeakTorque[spinning];
    if (peak <= 0.0)
    {
        return;
    }

    state.brakeFraction[spinning] = std::clamp(fade * setup.brakeGain * magnitude / peak, 0.0, setup.brakeCeiling);
    state.brakeActive = state.brakeFraction[spinning] > 0.0;
}

} // namespace

[[nodiscard]] double tractionTargetSlip(const TractionSetup& setup)
{
    switch (setup.mode)
    {
    case TractionMode::Full:
        return setup.fullTargetSlip;
    case TractionMode::Sport:
        return setup.sportTargetSlip;
    case TractionMode::Off:
        break;
    }

    return 0.0;
}

[[nodiscard]] double tractionThrottleScale(const TractionState& state)
{
    return std::clamp(1.0 - state.engineReduction, 0.0, 1.0);
}

void advanceTractionControl(const TractionSetup& setup, TractionState& state, const WheelSpeedReadings& wheels,
                            const ReferenceSpeedSetup& reference, const std::array<double, wheelCount>& brakePeakTorque,
                            const double referenceSpeed, const bool referenceValid, const double referenceCoasting,
                            const double throttle, const double deltaTime)
{
    state.brakeFraction = {};
    state.brakeActive = false;
    state.engineActive = false;

    if (setup.mode == TractionMode::Off || !referenceValid)
    {
        state.engineReduction = 0.0;

        return;
    }

    const auto target = tractionTargetSlip(setup);

    // --- the engine channel: the whole axle, because one throttle serves both wheels -------------
    //
    // Sized by the worse of the two driven wheels, and this is the channel that owns the case where
    // both of them are spinning together — which is most of a launch.
    auto worstError = 0.0;

    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        if (!setup.driven[index] || !wheels[index].valid)
        {
            continue;
        }

        const auto wheelSpeed = std::abs(sensedRoadSpeed(reference, wheels[index]));
        worstError = std::max(worstError, driveSlip(referenceSpeed, wheelSpeed) - target);
    }

    worstError = std::max(worstError, 0.0);

    // --- the brake channel: an electronic differential lock -------------------------------------
    //
    // Left against right, never against the reference speed. See the note on `TractionSetup` for
    // why that distinction is the whole design and not a detail.
    advanceDifferentialLock(setup, state, wheels, reference, brakePeakTorque, referenceSpeed, referenceCoasting);

    // A driver off the throttle is not asking for torque and there is nothing for this channel to
    // take away; the reduction is released rather than held, so lifting restores the pedal map
    // instead of leaving a car that will not pull away.
    const auto demanded = throttle > 0.0 ? std::clamp(setup.engineGain * worstError, 0.0, 1.0) : 0.0;

    state.engineReduction =
        approach(state.engineReduction, demanded,
                 demanded > state.engineReduction ? setup.engineCloseTime : setup.engineOpenTime, deltaTime);
    state.engineActive = state.engineReduction > 1e-3;
}

void advanceCorneringBrake(const CorneringBrakeSetup& setup, CorneringBrakeState& state,
                           const WheelSpeedReadings& wheels, const ReferenceSpeedSetup& reference,
                           const std::array<double, wheelCount>& brakePeakTorque, const double lateralAcceleration,
                           const double deltaTime)
{
    static_cast<void>(deltaTime);

    state.brakeFraction = {};
    state.active = false;

    if (!setup.enabled || std::abs(lateralAcceleration) < setup.onsetLateralAcceleration)
    {
        return;
    }

    for (const auto wheel : {frontLeft, frontRight, rearLeft, rearRight})
    {
        if (!wheels[wheel].valid)
        {
            return;
        }
    }

    // The turn's kinematics, taken off the undriven axle. Two rear wheels rolling freely on
    // different radii are a yaw rate measurement that owes nothing to the engine, which is why a
    // front-drive car is the easy case for this and a four-wheel-drive one is not.
    const auto rearLeftSpeed = std::abs(sensedRoadSpeed(reference, wheels[rearLeft]));
    const auto rearRightSpeed = std::abs(sensedRoadSpeed(reference, wheels[rearRight]));

    if (setup.rearTrack <= 0.0)
    {
        return;
    }

    const auto mean = 0.5 * (rearLeftSpeed + rearRightSpeed);
    const auto yaw = (rearRightSpeed - rearLeftSpeed) / setup.rearTrack;
    const auto halfFront = 0.5 * setup.frontTrack;

    const auto expectedLeft = mean - yaw * halfFront;
    const auto expectedRight = mean + yaw * halfFront;

    const auto excessLeft = std::abs(sensedRoadSpeed(reference, wheels[frontLeft])) - expectedLeft;
    const auto excessRight = std::abs(sensedRoadSpeed(reference, wheels[frontRight])) - expectedRight;

    // Which front is on the inside of the turn: the rear wheels say which side is travelling the
    // shorter distance and that is the answer, without the accelerometer's sign convention or the
    // steering wheel's ever entering it.
    const auto inside = yaw > 0.0 ? frontLeft : frontRight;

    // **Differenced across the axle rather than taken absolutely**, and that removes a real error
    // rather than smoothing one. The front wheels are steered, so they travel a longer path than the
    // rears and both read above the kinematics predicted from an unsteered axle. That bias is common
    // to the pair; the difference between the two is not, and the difference is the thing XDS exists
    // to act on. Correcting the bias properly needs the steering angle sensor, which the assist unit
    // already carries and nothing here reads yet.
    const auto departure = inside == frontLeft ? excessLeft - excessRight : excessRight - excessLeft;

    if (departure <= 0.0)
    {
        return;
    }

    const auto peak = brakePeakTorque[inside];
    if (peak <= 0.0)
    {
        return;
    }

    state.brakeFraction[inside] = std::clamp(setup.gain * departure / peak, 0.0, setup.ceiling);
    state.active = state.brakeFraction[inside] > 0.0;
}

} // namespace raceengine
