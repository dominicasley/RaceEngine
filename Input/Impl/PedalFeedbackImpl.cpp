// PedalFeedback bodies. Declarations are in Input/Api/PedalFeedback.cppm.
//
// A **module implementation unit** — `module raceengine.input;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

module raceengine.input;

import :PedalFeedback;

namespace raceengine
{

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

[[nodiscard]] PedalFeedback derivePedalFeedback(const PedalFeedbackSetup& setup,
                                                const std::span<const SlippingWheel> wheels, const double brakePedal,
                                                const double throttlePedal, const double staticShare)
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
