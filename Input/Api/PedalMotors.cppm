module;

#include <algorithm>
#include <cmath>
#include <cstdint>

export module raceengine.input:PedalMotors;

import :PedalFeedback;

namespace raceengine
{

// Stage two of three for the pedals: what one particular pedal set can make of stage one's answer.
//
// **Every hardware fact about the pedals lives here and only here**, exactly as `ForceMapping` holds
// every hardware fact about the wheel base. Stage one is the car and stage three is a write; this is
// the only place that knows a ClubSport V3's motors take eight bits each, that an eccentric mass does
// not turn at all below about a third of full duty, and that the two of them share one 24-bit report.

// What a pedal set's motors can do. Nothing here is discovered at runtime — the driver reports no
// motor characteristics at all, so it comes from the profile keyed on the device's identity.
export struct PedalMotorProfile
{
    // Whether there are motors at all. **False by default, and that is the honest default**: CSL
    // Elite and CSL LC pedals are recognised by the same driver, are the same brand, and have no
    // vibration motors in them whatsoever. A profile that assumed motors would silently ask a device
    // that cannot answer.
    bool hasMotors = false;

    // Codes per motor. Eight bits: the report carries one byte each for throttle and brake.
    std::uint32_t levels = 256;

    // **The duty below which an eccentric-mass motor does not turn**, as a fraction of full.
    //
    // This is the one number that decides whether the feature works at all, and it is a property of
    // the physics of a weight on a shaft rather than of anything the driver chose: below its
    // stiction the motor draws current and sits still. Mapping a severity of 0.05 onto 5% duty
    // therefore produces *nothing*, and the cue appears to have a threshold far higher than the one
    // stage one states.
    //
    // So the useful range is compressed into what actually spins: anything above zero severity
    // starts at `minimumDuty` and runs to full. **Placeholder** in the sense every device figure
    // here is until somebody feels it — a small ERM is typically a quarter to a third.
    double minimumDuty = 0.30;

    // How often the motors are worth being told anything. An eccentric mass has to spin up and down
    // through its own inertia, which takes tens of milliseconds, so commanding at the simulation's
    // rate would be writing hundreds of reports a second the hardware cannot render — and every one
    // of them is a USB control transfer on the same pipe the force feedback is using.
    double updateHz = 60.0;
};

// The two motor levels, as the codes the report carries.
export struct PedalMotorCommand
{
    std::uint8_t throttle = 0;
    std::uint8_t brake = 0;

    [[nodiscard]] bool silent() const
    {
        return throttle == 0 && brake == 0;
    }

    // The 24-bit word the driver's `rumble` attribute takes: throttle in the high byte, brake in the
    // middle one, the low byte unused. The driver's own documentation states it as `0xFF0000` for
    // the throttle and `0xFF00` for the brake, and this is that sentence written once.
    [[nodiscard]] std::uint32_t word() const
    {
        return (static_cast<std::uint32_t>(throttle) << 16) | (static_cast<std::uint32_t>(brake) << 8);
    }
};

// The driver's dial, per pedal, so a driver who wants one cue and not the other can have that. Both
// default to on: a feature nobody can find is a feature nobody has.
export struct PedalMotorMapping
{
    double brakeGain = 1.0;
    double throttleGain = 1.0;
};

namespace
{

[[nodiscard]] inline std::uint8_t codeFor(const PedalMotorProfile& profile, const double severity, const double gain)
{
    const auto asked = std::clamp(severity, 0.0, 1.0) * std::max(gain, 0.0);

    // Nought is nought, and it has to be exactly nought rather than the minimum duty: the whole
    // point of a floor is that it applies to a motor that is meant to be turning.
    if (asked <= 0.0)
    {
        return 0;
    }

    const auto floor = std::clamp(profile.minimumDuty, 0.0, 1.0);
    const auto duty = floor + (1.0 - floor) * std::min(asked, 1.0);

    const auto top = static_cast<double>(std::max(profile.levels, 2u) - 1);

    return static_cast<std::uint8_t>(std::lround(std::clamp(duty, 0.0, 1.0) * top));
}

} // namespace

// Stage one's answer, as the codes this pedal set takes.
//
// A pedal set with no motors answers silent for everything, which is what makes it safe to run this
// on any hardware: the cue simply is not there, rather than the game asking a device to do something
// it will not do and then wondering.
export [[nodiscard]] PedalMotorCommand mapPedalFeedback(const PedalMotorProfile& profile,
                                                        const PedalMotorMapping& mapping,
                                                        const PedalFeedback& feedback)
{
    if (!profile.hasMotors || !feedback.finite)
    {
        return PedalMotorCommand{};
    }

    return PedalMotorCommand{.throttle = codeFor(profile, feedback.throttle, mapping.throttleGain),
                             .brake = codeFor(profile, feedback.brake, mapping.brakeGain)};
}

} // namespace raceengine
