// CarAudio bodies. Declarations are in Audio/Api/CarAudio.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

module raceengine.audio;

import :CarAudio;
import raceengine.physics;

namespace raceengine
{

namespace
{

[[nodiscard]] double saturate(const double value)
{
    return std::isfinite(value) ? std::clamp(value, 0.0, 1.0) : 0.0;
}

} // namespace

[[nodiscard]] CarAudioState deriveCarAudio(const DrivelineSetup& setup, const DrivelineState& driveline,
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
