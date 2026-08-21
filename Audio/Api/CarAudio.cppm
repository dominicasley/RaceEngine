module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

export module raceengine.audio:CarAudio;

import raceengine.physics;

namespace raceengine
{

// Stage one of the sound, and it is the same split the force feedback keeps: what the *car* is doing,
// in the units a sound bank names, with nothing in it that knows what will play it.
//
// A car's engine sound is not a sample, it is a crossfade. A bank carries the same engine recorded at
// a handful of speeds, on and off the throttle, inside and out; what makes it sound like an engine is
// choosing between them continuously from the state the driveline is already computing. So the whole
// of this file is a conversion, and the conversion is where every unit mistake would live.
//
// **The turbo event cannot be driven yet and is deliberately absent.** The Golf's bank has one, and
// the importer computes boost to reconstruct the torque curve — but boost is not *state* the engine
// model carries at runtime, so there is no honest number to give it. Deriving one from load and speed
// would be inventing a signal and dressing it as a measurement, which is how `peakSlipScale` became a
// feature that read as implemented and behaved as a comment. It arrives when the engine model does.
//
// **Engine speed leaves here in rpm.** The physics works in rad/s and every bank in existence names
// its parameter `rpm`, so the factor belongs at this boundary rather than in whoever writes the
// parameter — which is exactly the mistake `Engine RPM [rpm]` made in the telemetry, where a channel
// labelled rpm carried rad/s and every engine number this project produced read 9.55x low for a
// milestone. One conversion, stated once, at the edge.

export inline constexpr double radiansPerSecondToRpm = 60.0 / (2.0 * 3.14159265358979323846);

// What a car event needs to know, and nothing else. Every field is a quantity a sound designer would
// recognise rather than a quantity this engine happens to have: `load` rather than clutch torque,
// `slip` rather than a Pacejka deflection.
export struct CarAudioState
{
    // The two an engine event always takes.
    double engineRpm = 0.0;
    double throttle = 0.0;

    // The engine's own range, carried because the mix needs it and the thing that does the mixing
    // must not know what a `DrivelineSetup` is. Constant per car; it rides here rather than being
    // passed separately so there is one struct to keep in step instead of two.
    double idleRpm = 0.0;
    double limiterRpm = 0.0;

    // How hard the engine is working at that speed, 0 to 1. A bank uses it to pick between the "on"
    // and "off" layers — an engine at 4000 rpm coasting downhill and one at 4000 rpm pulling are the
    // same speed and completely different sounds, and speed alone cannot tell them apart.
    double load = 0.0;

    std::int32_t gear = 0;

    // The limiter is a *state* and not an event: it comes and goes at the engine's own rate, so a
    // bank crossfades a layer rather than triggering a one-shot. Triggering would stutter at exactly
    // the rate the limiter cycles.
    bool onLimiter = false;
    // True through the neutral window of a shift, which is what a bank plays its transmission layer
    // against.
    bool shifting = false;
    // The clutch's own slip, rad/s. A launch is audible and this is what makes it so.
    double clutchSlip = 0.0;

    // The car rather than the engine.
    double roadSpeed = 0.0;
    // The tread's own surface speed, m/s: mean over the wheels of |wheel speed| times rolling
    // radius. It is not the road speed and the difference is the whole point of carrying both — a
    // locked wheel slides at road speed with its tread stopped, so its rolling noise is gone and
    // its skid is everything; a burnout is the reverse. Driving the rolling loop from road speed
    // would play tread noise from a tyre that is not turning.
    double rollingSpeed = 0.0;
    // The worst corner's, 0 to 1: 0 is gripping and 1 is fully sliding. One number because a bank has
    // one skid event, and the worst wheel is the one you can hear.
    double wheelSlip = 0.0;
    // Load on the worst-slipping wheel as a fraction of its static share, so a skid under braking is
    // louder than one from a wheel in the air.
    double slipLoad = 0.0;
};

namespace
{

[[nodiscard]] double saturate(const double value)
{
    return std::isfinite(value) ? std::clamp(value, 0.0, 1.0) : 0.0;
}

} // namespace

// The car's state, as the bank wants it.
//
// `torques` and `step` are the same tick's results rather than the previous one's: a sound derived
// from a tick older than the picture is a sound that arrives late, and at 120 Hz that is audible on
// a gear change even though it is inaudible on an engine note.
export [[nodiscard]] CarAudioState deriveCarAudio(const DrivelineSetup& setup, const DrivelineState& driveline,
                                                  const DrivelineTorques& torques, const VehicleState& vehicle,
                                                  const VehicleStep& step, const VehicleInput& input)
{
    auto state = CarAudioState{};

    state.engineRpm = std::max(0.0, driveline.engineSpeed) * radiansPerSecondToRpm;
    state.idleRpm = setup.engine.idleSpeed * radiansPerSecondToRpm;
    state.limiterRpm = setup.engine.limiterSpeed * radiansPerSecondToRpm;
    state.throttle = saturate(input.throttle);
    state.gear = driveline.gear;
    state.onLimiter = driveline.fuelCut;
    state.shifting = driveline.shiftPhase != ShiftPhase::Engaged;

    // What the engine is delivering against what it could deliver at this speed, which is what a
    // designer means by load. Taken from the torque the coupling actually passed rather than from the
    // pedal: a wide-open throttle at 1000 rpm against a locked torque converter is full pedal and
    // very little load, and it sounds like it.
    const auto available = setup.engine.torque.at(std::max(0.0, driveline.engineSpeed));
    state.load = available > 1e-6 ? saturate(torques.clutch / available) : 0.0;

    state.clutchSlip = std::isfinite(torques.clutchSlip) ? std::abs(torques.clutchSlip) : 0.0;

    state.roadSpeed = glm::length(vehicle.chassis.linearVelocity);

    // The worst wheel, and the load under it. A bank has one skid event and what a driver hears is
    // whichever tyre is losing — averaging the four would make a locked front under braking quieter
    // than the same car sliding gently on all four, which is the opposite of true.
    auto worst = 0.0;
    auto worstLoad = 0.0;
    auto treadSpeed = 0.0;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& contact = step.corners[index].contact;
        const auto slip = saturate(std::hypot(contact.slip.slipRatio, std::tan(contact.slip.slipAngle)));

        if (slip > worst)
        {
            worst = slip;
            worstLoad = step.corners[index].forces.tireVertical;
        }

        treadSpeed += std::abs(vehicle.corners[index].wheelSpeed) * contact.effectiveRadius;
    }

    // The mean rather than the worst, because rolling noise is four tyres' hiss together and one
    // locked wheel should quieten it by a quarter, not silence it.
    state.rollingSpeed = std::isfinite(treadSpeed) ? treadSpeed / static_cast<double>(cornerCount) : 0.0;

    state.wheelSlip = worst;

    // Against a quarter of the car's weight, which is the share a wheel carries standing still. Over
    // one under load transfer, under one on an unloaded inside wheel, zero in the air.
    const auto share = 0.25 * vehicle.chassis.mass * 9.80665;
    state.slipLoad = share > 1e-6 ? saturate(worstLoad / share) : 0.0;

    return state;
}

} // namespace raceengine
