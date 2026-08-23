#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine.physics;

using raceengine::advanceReferenceSpeed;
using raceengine::estimatedSlip;
using raceengine::frontLeft;
using raceengine::rearLeft;
using raceengine::rearRight;
using raceengine::ReferenceSpeedSetup;
using raceengine::ReferenceSpeedState;
using raceengine::sampleWheelSensors;
using raceengine::sensedRoadSpeed;
using raceengine::ToneRing;
using raceengine::wheelCount;
using raceengine::WheelSensorStates;

// The sensor half of the ABS work: whether the controller is being fed something a real ECU could
// have, and whether it fails the way a real one does.
//
// **Nothing here asserts against what the code produced.** Every expected value is derived from the
// tone ring's tooth count and the wheel's radius, so changing the ring from 48 poles to 44 moves the
// numbers and breaks no assertion — which is the difference between a test that catches an error and
// a test that only catches a change.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto controlPeriod = 1.0 / 1000.0;
constexpr auto tyreRadius = 0.3186;
constexpr auto twoPi = 6.283185307179586;

// Road speed to wheel rotation, and back. Written out rather than reused from the module so that a
// sign or a radius going wrong in there cannot be cancelled by the same mistake here.
[[nodiscard]] double omegaFor(const double roadSpeed)
{
    return roadSpeed / tyreRadius;
}

[[nodiscard]] std::array<double, wheelCount> allAt(const double omega)
{
    auto speeds = std::array<double, wheelCount>{};
    speeds.fill(omega);

    return speeds;
}

} // namespace

TEST_CASE("a tone ring reports one pulse per tooth and nothing in between", "[assists][sensor]")
{
    const auto ring = ToneRing{};
    REQUIRE(ring.teeth > 0);

    auto states = WheelSensorStates{};

    SECTION("the pulse rate is the tooth count times the wheel's revolutions")
    {
        // 20 m/s: 9.99 revolutions a second on a 0.3186 m wheel, so 48 teeth give 479.6 pulses.
        const auto omega = omegaFor(20.0);
        const auto speeds = allAt(omega);

        for (auto step = 0; step < 360; step++)
        {
            static_cast<void>(sampleWheelSensors(ring, states, speeds, tick));
        }

        const auto predicted = static_cast<double>(ring.teeth) * omega / twoPi;

        CAPTURE(states[frontLeft].pulses, predicted);

        // One second of rotation, so the count is the rate. Within a tooth, because the second
        // ends wherever it ends.
        REQUIRE(static_cast<double>(states[frontLeft].pulses) >= predicted - 1.0);
        REQUIRE(static_cast<double>(states[frontLeft].pulses) <= predicted + 1.0);
    }

    SECTION("and the interval between them is the tooth's own arc, whatever the speed")
    {
        // A tooth is 2*pi*r/N of travel — 41.7 mm for 48 teeth on this wheel — so the interval is
        // that divided by road speed, and it is a *property of the ring* rather than of the code.
        const auto toothTravel = twoPi * tyreRadius / static_cast<double>(ring.teeth);

        for (const auto roadSpeed : {27.78, 13.89, 5.56, 1.39})
        {
            auto swept = WheelSensorStates{};
            const auto speeds = allAt(omegaFor(roadSpeed));

            for (auto step = 0; step < 3600; step++)
            {
                static_cast<void>(sampleWheelSensors(ring, swept, speeds, controlPeriod));
            }

            const auto predicted = toothTravel / roadSpeed;

            CAPTURE(roadSpeed, toothTravel, predicted, swept[frontLeft].measuredPeriod);

            // Within the capture timer's own resolution, which is the only thing between the
            // measurement and the arithmetic.
            REQUIRE(std::abs(swept[frontLeft].measuredPeriod - predicted) <= 2.0 * ring.timerResolution);
        }
    }
}

TEST_CASE("what the sensor reports is quantised by the ring and stale by a tooth", "[assists][sensor]")
{
    const auto ring = ToneRing{};
    const auto toothTravel = twoPi * tyreRadius / static_cast<double>(ring.teeth);

    SECTION("the reading is never older than one tooth's worth of travel")
    {
        for (const auto roadSpeed : {27.78, 5.56})
        {
            auto states = WheelSensorStates{};
            const auto speeds = allAt(omegaFor(roadSpeed));

            auto worstAge = 0.0;

            for (auto step = 0; step < 3600; step++)
            {
                const auto readings = sampleWheelSensors(ring, states, speeds, controlPeriod);

                if (readings[frontLeft].valid)
                {
                    worstAge = std::max(worstAge, readings[frontLeft].age);
                }
            }

            const auto interval = toothTravel / roadSpeed;

            CAPTURE(roadSpeed, interval, worstAge);

            // Never older than the interval, and — because the wheel is turning steadily — it does
            // get that old, which is what makes latency a function of speed rather than a constant.
            REQUIRE(worstAge <= interval + controlPeriod);
            REQUIRE(worstAge > 0.5 * interval);
        }
    }

    SECTION("a decelerating wheel is reported as a staircase with one step per tooth")
    {
        // The sensor has nothing new to say between crossings, so a wheel whose speed is falling
        // continuously is reported as a sequence of held values. Counting the distinct values over a
        // known rotation says how many teeth went past, which is the ring's arithmetic again.
        auto states = WheelSensorStates{};

        auto omega = omegaFor(20.0);
        const auto deceleration = 30.0; // rad/s^2 at the wheel
        const auto seconds = 0.5;

        auto changes = 0;
        auto previous = 0.0;
        auto turned = 0.0;

        for (auto step = 0; step < static_cast<int>(seconds / controlPeriod); step++)
        {
            const auto readings = sampleWheelSensors(ring, states, allAt(omega), controlPeriod);

            if (readings[frontLeft].valid && readings[frontLeft].speed != previous)
            {
                changes++;
                previous = readings[frontLeft].speed;
            }

            turned += omega * controlPeriod;
            omega -= deceleration * controlPeriod;
        }

        // A held reading is bounded above by the pulse that has not arrived, so between crossings it
        // decays rather than sitting still — the reading changes on more steps than there are teeth.
        // What the tooth count bounds is the number of *crossings*, and that is what is asserted.
        const auto predicted = turned / twoPi * static_cast<double>(ring.teeth);

        CAPTURE(turned, predicted, states[frontLeft].pulses, changes);

        REQUIRE(static_cast<double>(states[frontLeft].pulses) >= predicted - 1.0);
        REQUIRE(static_cast<double>(states[frontLeft].pulses) <= predicted + 1.0);
        REQUIRE(changes > 0);
    }
}

TEST_CASE("a wheel that stops has its reading decay rather than being held", "[assists][sensor]")
{
    // The one inference that makes every low-speed behaviour emergent instead of thresholded: if the
    // wheel were still turning at the last measured speed, the next tooth would already have gone
    // past. So the reading falls as one tooth pitch over the time since the last crossing.
    const auto ring = ToneRing{};
    const auto pitch = twoPi / static_cast<double>(ring.teeth);

    auto states = WheelSensorStates{};
    const auto rolling = allAt(omegaFor(10.0));

    for (auto step = 0; step < 1000; step++)
    {
        static_cast<void>(sampleWheelSensors(ring, states, rolling, controlPeriod));
    }

    const auto moving = sampleWheelSensors(ring, states, rolling, 0.0);
    REQUIRE(moving[frontLeft].speed == Catch::Approx(omegaFor(10.0)).epsilon(0.001));

    // Now stopped dead. Nothing else changes.
    const auto stopped = allAt(0.0);

    for (const auto after : {0.05, 0.20, 1.00})
    {
        auto held = states;
        auto elapsed = 0.0;

        while (elapsed < after)
        {
            static_cast<void>(sampleWheelSensors(ring, held, stopped, controlPeriod));
            elapsed += controlPeriod;
        }

        const auto readings = sampleWheelSensors(ring, held, stopped, 0.0);
        const auto bound = pitch / readings[frontLeft].age;

        CAPTURE(after, readings[frontLeft].speed, readings[frontLeft].age, bound);

        REQUIRE(readings[frontLeft].speed == Catch::Approx(bound).epsilon(0.01));
        REQUIRE(readings[frontLeft].speed < omegaFor(10.0));
    }
}

TEST_CASE("the reference speed estimate degrades when every wheel slips together", "[assists][sensor][reference]")
{
    // **Criterion 10.** The controller can only see wheels. Slip all four equally and there is
    // nothing left to take a reference from, so the estimate must come away from the truth. If it
    // still tracked, something would have leaked.
    const auto ring = ToneRing{};
    const auto setup = ReferenceSpeedSetup{};

    const auto trueSpeed = 25.0;

    auto states = WheelSensorStates{};
    auto reference = ReferenceSpeedState{};

    // Rolling honestly first, so the estimate is established and correct before anything goes wrong.
    for (auto step = 0; step < 1000; step++)
    {
        const auto readings = sampleWheelSensors(ring, states, allAt(omegaFor(trueSpeed)), controlPeriod);
        advanceReferenceSpeed(setup, reference, readings, false, controlPeriod);
    }

    REQUIRE(reference.valid);
    REQUIRE(reference.speed == Catch::Approx(trueSpeed).epsilon(0.01));

    SECTION("one wheel locked is caught by the other three")
    {
        auto speeds = allAt(omegaFor(trueSpeed));
        speeds[frontLeft] = 0.0;

        for (auto step = 0; step < 500; step++)
        {
            const auto readings = sampleWheelSensors(ring, states, speeds, controlPeriod);
            advanceReferenceSpeed(setup, reference, readings, true, controlPeriod);
        }

        CAPTURE(reference.speed, reference.coasting);

        // Still the truth, because three wheels are still telling it.
        REQUIRE(reference.speed == Catch::Approx(trueSpeed).epsilon(0.01));
        REQUIRE(reference.coasting == Catch::Approx(0.0).margin(0.005));
    }

    SECTION("all four slipping equally is the case it cannot see")
    {
        // Every wheel at 70% of road speed — a car sliding on all four with the road still moving
        // under it. Nothing in the wheel population says so.
        const auto slipping = allAt(omegaFor(0.7 * trueSpeed));

        for (auto step = 0; step < 500; step++)
        {
            const auto readings = sampleWheelSensors(ring, states, slipping, controlPeriod);
            advanceReferenceSpeed(setup, reference, readings, true, controlPeriod);
        }

        CAPTURE(reference.speed, trueSpeed);

        // The estimate must have followed the wheels down rather than holding the truth. A car whose
        // wheels all read 17.5 m/s while it is doing 25 has no way to know.
        REQUIRE(reference.speed < 0.9 * trueSpeed);

        // And the slip it computes is therefore far smaller than the slip that is really there:
        // 30% of true against what the estimate can see.
        const auto believed = estimatedSlip(reference.speed, 0.7 * trueSpeed);

        CAPTURE(believed);
        REQUIRE(std::abs(believed) < 0.15);
    }
}

TEST_CASE("the reference speed reads slow because the ECU uses a nominal radius", "[assists][sensor][reference]")
{
    // A systematic error a real system lives with and this one is not allowed to correct: the tyre's
    // effective rolling radius shrinks under load, so converting wheel speed with the unloaded radius
    // reads every wheel — and therefore the car — slightly slow.
    const auto setup = ReferenceSpeedSetup{};

    REQUIRE(setup.nominalRadius == Catch::Approx(tyreRadius));

    // A wheel rolling without slip on a tyre squashed to a 0.31 m effective radius is doing
    // 25 m/s over the ground; through the nominal 0.3186 it reports 25.7.
    const auto loadedRadius = 0.310;
    const auto roadSpeed = 25.0;
    const auto omega = roadSpeed / loadedRadius;

    const auto reading = raceengine::WheelSpeedReading{.speed = omega, .age = 0.0, .valid = true, .pulses = 1};
    const auto reported = sensedRoadSpeed(setup, reading);

    CAPTURE(loadedRadius, roadSpeed, reported);

    // Reads high by exactly the radius ratio, which under braking — where the loaded radius is
    // smallest — makes the wheel look faster than it is and the slip look smaller.
    REQUIRE(reported == Catch::Approx(roadSpeed * setup.nominalRadius / loadedRadius));
    REQUIRE(reported > roadSpeed);
}

TEST_CASE("slip is signed the same way for both controllers", "[assists][sensor]")
{
    // One expression, two consumers. A braking wheel turns slower than the road and a driven one
    // turns faster; the sign convention is stated once so the two cannot disagree about it.
    REQUIRE(estimatedSlip(20.0, 16.0) == Catch::Approx(0.20));
    REQUIRE(estimatedSlip(20.0, 24.0) == Catch::Approx(-0.20));
    REQUIRE(estimatedSlip(20.0, 20.0) == Catch::Approx(0.0));

    // A locked wheel against a moving car is total slip, and a stopped car is not a division by zero.
    REQUIRE(estimatedSlip(20.0, 0.0) == Catch::Approx(1.0));
    REQUIRE(std::isfinite(estimatedSlip(0.0, 0.0)));
}
