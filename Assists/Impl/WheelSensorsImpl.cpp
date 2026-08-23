module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

// The bodies for `raceengine.assists:WheelSensors`. No `export`, no partition name: this unit
// produces an object and no BMI, so editing a rate limit here rebuilds one file rather than every
// importer of the module. Same rule as `Physics/Impl/*Impl.cpp` — see docs/build-times.md.
module raceengine.assists;

namespace raceengine
{

namespace
{

// The fraction of the way a first-order lag closes on its target in one step. Written once because
// three places need it and a per-step coefficient calibrated at one timestep is wrong at every other.
[[nodiscard]] double blendOver(const double timeConstant, const double deltaTime)
{
    return timeConstant > 0.0 ? 1.0 - std::exp(-deltaTime / timeConstant) : 1.0;
}

// One tooth, in radians. Guarded against a ring with no teeth so a default-constructed setup cannot
// divide by zero on the first tick of a test that forgot to fill it in.
[[nodiscard]] double toothPitch(const ToneRing& ring)
{
    return ring.teeth > 0 ? 6.283185307179586 / static_cast<double>(ring.teeth) : 6.283185307179586;
}

// The capture timer, which measures whole counts and not real numbers. Rounding down rather than to
// nearest because a counter latched on an edge has already discarded the fraction.
[[nodiscard]] double quantisePeriod(const ToneRing& ring, const double period)
{
    if (ring.timerResolution <= 0.0)
    {
        return period;
    }

    return std::floor(period / ring.timerResolution) * ring.timerResolution;
}

} // namespace

[[nodiscard]] WheelSpeedReadings sampleWheelSensors(const ToneRing& ring, WheelSensorStates& states,
                                                    const std::array<double, wheelCount>& wheelSpeeds,
                                                    const double deltaTime)
{
    auto readings = WheelSpeedReadings{};

    const auto pitch = toothPitch(ring);

    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        auto& state = states[index];
        const auto speed = wheelSpeeds[index];
        const auto rate = std::abs(speed);

        // How far the wheel turns this tick. The vehicle integrates its wheel speed once per tick,
        // so the rotation across the tick is taken as uniform at the speed it ends at — which is
        // what the tick's own arithmetic says happened.
        auto remaining = rate * deltaTime;
        auto elapsed = 0.0;

        // Every tooth that goes past inside the tick, each with its own instant. At 100 km/h this
        // runs about twice per tick; sampling once and calling it a pulse would alias two thirds of
        // the signal away and would put the whole low-speed story out of reach.
        while (remaining > 0.0 && state.angleSincePulse + remaining >= pitch)
        {
            const auto needed = pitch - state.angleSincePulse;
            const auto toPulse = needed / rate;

            elapsed += toPulse;

            state.measuredPeriod = quantisePeriod(ring, state.timeSincePulse + toPulse);
            state.direction = speed >= 0.0 ? 1.0 : -1.0;
            state.timeSincePulse = 0.0;
            state.angleSincePulse = 0.0;
            state.pulses++;

            remaining -= needed;
        }

        state.angleSincePulse += remaining;
        state.timeSincePulse += deltaTime - elapsed;

        auto& reading = readings[index];
        reading.pulses = state.pulses;
        reading.valid = state.pulses > 0;
        reading.age = state.timeSincePulse;

        if (!reading.valid || state.measuredPeriod <= 0.0)
        {
            // Nothing has gone past yet, or the last two crossings fell inside one timer count.
            // Either way there is no measurement, and a controller reading zero here is reading the
            // truth about what its sensor knows.
            reading.speed = 0.0;
            continue;
        }

        const auto measured = pitch / state.measuredPeriod;

        // The bound the missing pulse implies. Held rather than decayed while the reading is fresh —
        // `timeSincePulse` under one period cannot contradict the measurement — and biting the
        // moment the wheel is overdue, which is the whole of the low-speed behaviour.
        const auto bound = state.timeSincePulse > 0.0 ? pitch / state.timeSincePulse : measured;

        reading.speed = std::min(measured, bound) * state.direction;
    }

    return readings;
}

[[nodiscard]] double sensedRoadSpeed(const ReferenceSpeedSetup& setup, const WheelSpeedReading& wheel)
{
    return wheel.speed * setup.nominalRadius;
}

[[nodiscard]] double estimatedSlip(const double referenceSpeed, const double wheelSpeed)
{
    // Against the reference rather than against the wheel, because the reference is the quantity
    // that stays finite: a locked wheel divides by zero the other way round. The floor is a
    // millimetre a second, which is below anything either controller can act on.
    const auto divisor = std::max(std::abs(referenceSpeed), 1e-3);

    return (referenceSpeed - wheelSpeed) / divisor;
}

void advanceReferenceSpeed(const ReferenceSpeedSetup& setup, ReferenceSpeedState& state,
                           const WheelSpeedReadings& wheels, const bool braking, const double deltaTime)
{
    // Which wheels are worth believing, and it flips with the brake light switch. Under braking
    // every wheel reads at or below road speed, so the fastest is the closest to the truth. Under
    // power the driven wheels read high, so an undriven one is believed outright and, with none
    // available, the slowest driven wheel is the least wrong.
    auto candidate = 0.0;
    auto supported = false;

    const auto consider = [&](const double speed)
    {
        if (!supported)
        {
            candidate = speed;
            supported = true;
            return;
        }

        candidate = braking ? std::max(candidate, speed) : std::min(candidate, speed);
    };

    auto anyUndriven = false;
    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        if (!setup.driven[index] && wheels[index].valid)
        {
            anyUndriven = true;
        }
    }

    for (auto index = std::size_t{0}; index < wheelCount; index++)
    {
        const auto& wheel = wheels[index];
        if (!wheel.valid)
        {
            continue;
        }

        // Under power an undriven wheel is a free road-speed measurement and the driven ones are
        // noise. Under braking all four are braked, so the distinction buys nothing and the whole
        // population is used.
        if (!braking && anyUndriven && setup.driven[index])
        {
            continue;
        }

        consider(std::abs(sensedRoadSpeed(setup, wheel)));
    }

    if (!state.valid)
    {
        if (!supported)
        {
            return;
        }

        state.speed = candidate;
        state.lagged = candidate;
        state.valid = true;
        state.coasting = 0.0;
        state.rate = 0.0;

        return;
    }

    // **The limits are one-sided, and which side depends on what the car is doing.** A braked wheel
    // cannot turn faster than the road is going, so under braking the fastest wheel is a *lower
    // bound on the truth* and there is nothing to be gained by rationing how fast the estimate is
    // allowed to climb back to it — only the fall needs bounding, because that is the direction a
    // locked wheel lies in. Under power it is the other way about: a driven wheel can turn faster
    // than the road, so the rise is what needs bounding.
    //
    // Rationed both ways, the estimate could not recover between anti-lock cycles: on a uniform
    // mu 0.35 surface a front wheel came back to 27.5 m/s against a true 26.7 while the estimate,
    // allowed to climb at 0.6 g, was still down at 24.9 and being dragged lower. It never caught up
    // for the whole stop, and the anti-lock system it feeds bought 1% over locked wheels.
    const auto ceiling = braking ? std::max(candidate, state.speed) : state.speed + setup.riseLimit * deltaTime;
    const auto floor = braking ? std::max(0.0, state.speed - setup.fallLimit * deltaTime) : 0.0;

    if (!supported)
    {
        // No wheel is reporting at all. The estimate carries on at the rate it was last seen to be
        // changing, capped by what a car can physically do — which is the failure this component is
        // allowed to have, and it degrades rather than collapsing. The rate itself is frozen: it is
        // derived from the estimate below, and letting it learn from an estimate it is itself driving
        // is a loop that runs away from the road rather than following it.
        state.speed = std::clamp(state.speed + state.rate * deltaTime, floor, ceiling);
        state.coasting += deltaTime;
        state.lagged += (state.speed - state.lagged) * blendOver(setup.rateSmoothing, deltaTime);

        return;
    }

    const auto bounded = std::clamp(candidate, floor, ceiling);

    // Whether a wheel actually set the answer or the limiter did. A system running on its own bound
    // has lost its reference, and this is the channel that says so rather than a quantity anybody
    // has to infer from the numbers.
    state.coasting = bounded != candidate ? state.coasting + deltaTime : 0.0;
    state.speed = bounded;

    // --- how fast the car is changing speed, as the ECU sees it --------------------------------
    //
    // **Against a lagged copy of the estimate rather than against the previous step**, because the
    // estimate is quantised in *time*: a wheel speed reading only moves when a tooth goes past, so
    // at the instant it moves its apparent rate is the true rate multiplied by the ratio of the
    // pulse interval to the control period. At 100 km/h that is 1.6x, which puts a genuine 8 m/s^2
    // deceleration over the 12.749 m/s^2 physical bound — so the step gets clamped, marked as
    // limiter-carried, and thrown away, while every step between pulses teaches zero. Measured, the
    // ECU believed a 0.9 g stop was 0.39 m/s^2 and every threshold that reads this fired on nothing.
    //
    // Differencing against a first-order lag of the same signal cannot be fooled that way: the lag
    // integrates the steps, so what comes out is the average rate over `rateSmoothing` however
    // coarsely the input arrives.
    state.lagged += (state.speed - state.lagged) * blendOver(setup.rateSmoothing, deltaTime);

    const auto measured = setup.rateSmoothing > 0.0 ? (state.speed - state.lagged) / setup.rateSmoothing : 0.0;

    state.rate = std::clamp(measured, -setup.fallLimit, braking ? 0.0 : setup.riseLimit);
}

} // namespace raceengine
