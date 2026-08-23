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
                            const double referenceSpeed, const bool referenceValid, const double throttle,
                            const double deltaTime)
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

    // The engine channel acts on the whole axle because it has no choice — one throttle serves both
    // driven wheels — so it is sized by the worst of them. The brake channel is per wheel, which is
    // the entire reason a system with both is better than a system with either.
    auto worstError = 0.0;

    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        if (!setup.driven[index] || !wheels[index].valid)
        {
            continue;
        }

        const auto wheelSpeed = std::abs(sensedRoadSpeed(reference, wheels[index]));
        const auto error = driveSlip(referenceSpeed, wheelSpeed) - target;

        if (error <= 0.0)
        {
            continue;
        }

        worstError = std::max(worstError, error);

        const auto peak = brakePeakTorque[index];
        if (peak <= 0.0)
        {
            continue;
        }

        state.brakeFraction[index] = std::clamp(setup.brakeGain * error / peak, 0.0, setup.brakeCeiling);
        state.brakeActive = state.brakeActive || state.brakeFraction[index] > 0.0;
    }

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
