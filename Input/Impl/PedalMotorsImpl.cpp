// PedalMotors bodies. Declarations are in Input/Api/PedalMotors.cppm.
//
// A **module implementation unit** — `module raceengine.input;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>

module raceengine.input;

import :PedalMotors;
import :PedalFeedback;

namespace raceengine
{

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

[[nodiscard]] PedalMotorCommand mapPedalFeedback(const PedalMotorProfile& profile, const PedalMotorMapping& mapping,
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
