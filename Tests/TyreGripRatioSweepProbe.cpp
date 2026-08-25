// The longitudinal/lateral grip ratio, swept. `./EngineTests "[.grip-ratio]"`.
//
// **Stage 4 of `docs/tyre-grip-ratio-brief.md`, run as an experiment rather than as a change.** Two
// published references jointly demand a mu_x/mu_y of **1.22** where this car ships **1.015**, and
// after the evidence pass in `docs/engine-curve-validation-brief.md` the tyre is the only candidate
// left for the 0-100 shortfall. That is exactly the moment to be careful: a residual with one
// suspect remaining is not the same thing as a suspect that has been convicted.
//
// **So 1.22 is treated here as a hypothesis to falsify, not as the answer.** Dominic's framing, and
// it is the right one: the question is not "does 1.22 produce 6.5 s" but
//
//   1. does the model's *sensitivity* to mu_x make the required increase physically plausible, and
//   2. does the tyre still behave like a tyre at that value, judged on evidence that has nothing to
//      do with acceleration?
//
// If 1.22 lands the acceleration but produces absurd slip, wheelspin or force behaviour, then what
// has been discovered is that the residual was never simply mu_x.
//
// **Three things this probe is built around.**
//
// **The shape of the error across speeds, not one number.** The car is traction-limited, so grip
// matters most where the tyre is nearest its limit — off the line — and least where drag and gearing
// dominate. Splitting the run into 0-30, 0-50, 0-80, 0-100, 30-100, 60-100 and 80-120 therefore turns
// one scalar into a profile. If raising mu_x fixes 0-30 and 0-50 and leaves 80-120 wrong, the missing
// quantity is not longitudinal grip. If the whole low-speed trace moves into agreement together, it
// very likely is.
//
// **A control that must NOT move.** The in-gear 80-120 pull at a fixed ratio has almost no traction
// content — measured, driven-wheel slip runs 0.013 to 0.031 — so it should be very nearly insensitive
// to mu_x. It is run at every point of the sweep for exactly that reason: if the control moves with
// grip, the sweep is changing something other than what it claims to, and no conclusion from the
// accelerating rows is safe.
//
// **Braking is a check and never an input.** `100-0` is reported at every ratio and is deliberately
// *not* used to choose one. The whole value of having two independent longitudinal references is
// lost the moment a number is fitted to sit between them: if the mu_x that acceleration wants also
// brings braking into its published region without being asked to, that is strong evidence about the
// tyre; if the two want substantially different numbers, that is a finding about the tyre model and
// an invitation to look at it rather than to average them.
//
// Nothing here modifies the car. Every run takes a copy of the setup and scales `longitudinalPeak`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TractionMode;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto tyreRadius = 0.3186;
constexpr auto startZ = 20.0;

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

[[nodiscard]] ProvingGroundDescriptor straightGround(const double length)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = 60.0;
    descriptor.cellSize = 4.0;
    descriptor.features = std::vector<Feature>{};

    return descriptor;
}

[[nodiscard]] double designHeight(const VehicleSetup& setup)
{
    auto highest = 0.0;
    for (const auto& corner : setup.corners)
    {
        highest = std::max(highest, corner.hardpoints.wheelCentre.y + corner.hardpoints.wheelRadius);
    }

    return highest;
}

// The car with its longitudinal peak set to a stated multiple of its lateral one. **`lateralPeak` is
// not touched**: it was derived against the skidpad on 2026-08-22 and is not this experiment's to
// move, so every point of the sweep changes exactly one number.
[[nodiscard]] VehicleSetup withGripRatio(VehicleSetup setup, const double ratio)
{
    for (auto& corner : setup.corners)
    {
        corner.tyre.longitudinalPeak = ratio * corner.tyre.lateralPeak;
    }

    return setup;
}

struct Acceleration
{
    // Times through the standing-start run, seconds. -1 where the speed was never reached.
    double toThirty = -1.0;
    double toFifty = -1.0;
    double toEighty = -1.0;
    double toHundred = -1.0;
    double thirtyToHundred = -1.0;
    double sixtyToHundred = -1.0;
    double eightyToOneTwenty = -1.0;

    double peakDrivenSlip = 0.0;
    // The largest longitudinal force any driven wheel produced as a fraction of its own vertical
    // load. **The number that says whether the tyre is still behaving like a tyre**: it is bounded by
    // the friction available, so a sweep that raises it in step with mu_x is doing what it says, and
    // one that does not has hit some other limit.
    double peakForceRatio = 0.0;
    // Fraction of the run spent past the tyre's own longitudinal peak — the tyre reports where its
    // curve turns over, so this needs no threshold invented for it.
    double tractionLimitedFraction = 0.0;
    // Which driven wheel took the deepest slip, and how far apart the two were at that moment. A
    // straight-line launch has no inside or outside, so what an asymmetry here means is torque steer
    // and the differential rather than cornering.
    std::size_t deepestWheel = 0;
    double worstAcrossAxleSlip = 0.0;

    double peakEngineRpm = 0.0;
    double peakWheelTorque = 0.0;
    std::array<double, 8> shiftRpm{};
    std::uint32_t shifts = 0;
    // Road speed every half second, for the trace.
    std::vector<double> speedTrace;

    bool valid = false;
};

// A standing start at full throttle, shifted on road speed through the gear.
//
// The shift signal is deliberately *not* taken off the driven wheels: `LaunchDiagnosticProbe` records
// that doing so upshifts on wheelspin rather than on progress, which on a low-grip run sends the box
// to third by one second. It is also `reduction(gear)` and not `ratio * finalDrive` — this car has two
// final drives and **2nd and 3rd are on the tall one**, so the product over-states them by 33% and
// shifts out of the two gears a 0-100 lives in far too early. That cost 0.255 s before it was caught.
[[nodiscard]] Acceleration accelerate(const VehicleSetup& setup, const PhysicsWorld& world, const double seconds,
                                      const TractionMode traction = TractionMode::Off, const bool launchControl = false)
{
    auto driveline = golfGtiMk7Driveline();
    driveline.autoClutch.launch.enabled = launchControl;
    auto assists = golfGtiMk7Assists(setup);
    assists.traction.mode = traction;
    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    const auto inertias = wheelInertias(setup);

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    auto road = std::array<double, cornerCount>{};
    const auto speeds = [&]
    {
        return std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                               state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};
    };

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

        return sensors;
    };

    {
        // With launch control arming, the pre-phase is both pedals and long enough for the regulator
        // to settle the engine; without it, the brake alone at an idle. Same fixture either way.
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.throttle = launchControl ? 1.0 : 0.0;
        idling.gear = 1;

        for (auto held = 0; held < (launchControl ? 360 * 3 : 360); held++)
        {
            const auto command =
                updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = idling.throttle},
                              brakeCircuitPressures(setup, 1.0), tick);
            const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, idling, torques->wheel, world, tick, command.brakes);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());
            lastStep = stepped.value();
        }
    }

    // --- preconditions. Nothing this returns is worth quoting until these hold. ---
    //
    // **The engine check depends on which launch is being measured**, and that is the point rather
    // than an inconvenience: from an idle the engine must be *at* idle, and under launch control it
    // must be at the regulator's target with the clutch loaded and the car still stationary. Asserting
    // the idle condition under launch control would fail on a fixture that was working perfectly,
    // which is exactly what it did the first time.
    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.2);
    REQUIRE(drivelineState.gear == 1);

    if (launchControl)
    {
        const auto target = driveline.autoClutch.launch.targetSpeed;

        REQUIRE(drivelineState.launchArmed);
        REQUIRE(drivelineState.engineSpeed > 0.75 * target);
        REQUIRE(drivelineState.engineSpeed < 1.25 * target);
    }
    else
    {
        REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
        REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);
    }

    auto result = Acceleration{};
    auto gear = 1;
    const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;
    auto engaged = 0;
    auto limited = 0;

    const auto mark = [](double& slot, const double now)
    {
        if (slot < 0.0)
        {
            slot = now;
        }
    };

    for (auto step = 1; step <= static_cast<int>(seconds * 360.0); step++)
    {
        const auto now = static_cast<double>(step) * tick;

        const auto roadSideSpeed =
            std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
        if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
        {
            if (result.shifts < result.shiftRpm.size())
            {
                result.shiftRpm[result.shifts] = drivelineState.engineSpeed * 9.549296585513721;
            }
            result.shifts++;
            gear++;
        }

        const auto command = updateAssists(assists, assistState, sense(), {.brake = 0.0, .throttle = 1.0},
                                           brakeCircuitPressures(setup, 0.0), tick);

        auto input = VehicleInput{};
        input.throttle = command.throttleScale;
        input.gear = gear;

        const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());
        lastStep = stepped.value();

        const auto speed = std::abs(state.chassis.linearVelocity.z) * 3.6;

        if (speed >= 30.0)
        {
            mark(result.toThirty, now);
        }
        if (speed >= 50.0)
        {
            mark(result.toFifty, now);
        }
        if (speed >= 80.0)
        {
            mark(result.toEighty, now);
        }
        if (speed >= 60.0)
        {
            mark(result.sixtyToHundred, now);
        }
        if (speed >= 100.0)
        {
            mark(result.toHundred, now);
        }
        if (speed >= 120.0 && result.eightyToOneTwenty < 0.0 && result.toEighty > 0.0)
        {
            result.eightyToOneTwenty = now - result.toEighty;
        }

        if (step % 180 == 0)
        {
            result.speedTrace.push_back(speed);
        }

        result.peakEngineRpm = std::max(result.peakEngineRpm, drivelineState.engineSpeed * 9.549296585513721);
        result.peakWheelTorque =
            std::max(result.peakWheelTorque, std::abs(torques->wheel[0]) + std::abs(torques->wheel[1]));

        engaged++;

        auto pastPeak = false;
        auto slips = std::array<double, 2>{};

        for (const auto wheel : {std::size_t{0}, std::size_t{1}})
        {
            const auto& contact = stepped->corners[wheel].contact;
            const auto slip = std::abs(contact.slip.slipRatio);
            slips[wheel] = slip;

            if (slip > result.peakDrivenSlip)
            {
                result.peakDrivenSlip = slip;
                result.deepestWheel = wheel;
            }

            const auto load = stepped->corners[wheel].forces.tireVertical;
            if (load > 1.0)
            {
                result.peakForceRatio = std::max(result.peakForceRatio, std::abs(contact.tyre.longitudinal) / load);
            }

            // The tyre reports where its own curve peaks, so "past the peak" needs no threshold
            // invented for it and moves with the compound rather than against a fixed number.
            if (contact.tyre.longitudinalPeakSlip > 1e-9 && slip > contact.tyre.longitudinalPeakSlip)
            {
                pastPeak = true;
            }
        }

        result.worstAcrossAxleSlip = std::max(result.worstAcrossAxleSlip, std::abs(slips[0] - slips[1]));

        if (pastPeak)
        {
            limited++;
        }

        if (result.toHundred > 0.0 && result.eightyToOneTwenty > 0.0)
        {
            break;
        }
    }

    if (result.toHundred > 0.0)
    {
        if (result.toThirty > 0.0)
        {
            result.thirtyToHundred = result.toHundred - result.toThirty;
        }
        // `sixtyToHundred` has been holding the *time at sixty* up to here; it becomes the interval
        // once the far end is known.
        if (result.sixtyToHundred > 0.0)
        {
            result.sixtyToHundred = result.toHundred - result.sixtyToHundred;
        }
    }

    result.tractionLimitedFraction = engaged > 0 ? static_cast<double>(limited) / static_cast<double>(engaged) : 0.0;
    result.valid = result.toHundred > 0.0;

    return result;
}

// One constant-pedal stop from 100 km/h, in neutral. Nothing drives the wheels, so what is measured
// is the brake system and the tyre.
struct Stop
{
    double distance = 0.0;
    double pedal = 0.0;
    bool stopped = false;
};

[[nodiscard]] Stop stopFrom(const VehicleSetup& setup, const PhysicsWorld& world, const double entry,
                            const double pedal)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, entry);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = entry / tyreRadius;
    }

    for (auto step = 0; step < 180; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    const auto start = state.chassis.position.z;
    auto braking = VehicleInput{};
    braking.brake = pedal;

    auto result = Stop{.distance = 0.0, .pedal = pedal, .stopped = false};

    for (auto step = 0; step < 360 * 12; step++)
    {
        REQUIRE(stepVehicle(setup, state, braking, noDriveTorque, world, tick).has_value());

        if (state.chassis.linearVelocity.z <= 0.05)
        {
            result.distance = state.chassis.position.z - start;
            result.stopped = true;
            break;
        }
    }

    return result;
}

// The shortest constant-pedal stop, swept finely enough to mean something. The optimum moves with
// grip, which is the whole point here, so the sweep has to cover the range rather than bracket one
// candidate's answer.
[[nodiscard]] Stop bestStop(const VehicleSetup& setup, const PhysicsWorld& world, const double entry)
{
    auto best = Stop{.distance = 1e9, .pedal = 0.0, .stopped = false};

    for (auto pedal = 0.24; pedal <= 0.62; pedal += 0.02)
    {
        const auto run = stopFrom(setup, world, entry, pedal);
        if (run.stopped && run.distance < best.distance)
        {
            best = run;
        }
    }

    return best;
}

// A stop with the anti-lock system working and the pedal on the floor, which is what a published
// 100-0 figure actually is: a driver stamps on it and the electronics do the rest. **Not the same
// measurement as `bestStop`**, which is the shortest stop a driver holding one constant pressure could
// theoretically achieve with the electronics off — a hypothetical, and the thing this project has been
// comparing against Auto Bild all along.
[[nodiscard]] Stop stopWithAntilock(const VehicleSetup& setup, const PhysicsWorld& world, const double entry)
{
    auto assists = golfGtiMk7Assists(setup);
    assists.antilock.enabled = true;

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, entry);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = entry / tyreRadius;
    }

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

        return sensors;
    };

    // Rolling first, so the tone rings have produced readings and the reference speed estimator is not
    // asked to start up and brake in the same instant.
    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, brakeCircuitPressures(setup, 0.0), tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    const auto start = state.chassis.position.z;
    auto braking = VehicleInput{};
    braking.brake = 1.0;

    auto result = Stop{.distance = 0.0, .pedal = 1.0, .stopped = false};

    for (auto step = 0; step < 360 * 12; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                           brakeCircuitPressures(setup, 1.0), tick);
        const auto stepped = stepVehicle(setup, state, braking, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        if (state.chassis.linearVelocity.z <= 0.05)
        {
            result.distance = state.chassis.position.z - start;
            result.stopped = true;
            break;
        }
    }

    return result;
}

// The sweep. The shipped ratio is included so every row has something to be read against.
constexpr auto shippedRatio = 1.131 / 1.114;

const auto ratios = std::vector<double>{0.95, 1.00, 1.05, shippedRatio, 1.10, 1.15, 1.20, 1.22, 1.25, 1.30};

} // namespace

TEST_CASE("what the longitudinal grip ratio is worth, across the whole acceleration trace", "[.grip-ratio]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(world.has_value());

    std::printf("\n================ longitudinal grip ratio sweep ================\n");
    std::printf("mu_y is held at %.3f throughout — it was derived against the skidpad and is not this\n",
                setup->corners.front().tyre.lateralPeak);
    std::printf("experiment's to move. Only mu_x changes, so every row differs by exactly one number.\n");
    std::printf("\nreferences: 0-100 in 6.5-6.6 s (MEASURED, n=2), 100-0 in 34.6-35.1 m\n");

    std::printf("\n--- acceleration, from a standing start at full throttle ---\n");
    std::printf(" mu_x/mu_y   mu_x    0-30   0-50   0-80  0-100  30-100  60-100  80-120\n");

    auto results = std::vector<Acceleration>{};

    for (const auto ratio : ratios)
    {
        const auto car = withGripRatio(setup.value(), ratio);
        const auto run = accelerate(car, world.value(), 30.0);
        results.push_back(run);

        std::printf("   %5.3f%s  %5.3f  %6.3f %6.3f %6.3f %6.3f  %6.3f  %6.3f  %6.3f\n", ratio,
                    std::abs(ratio - shippedRatio) < 1e-9 ? "*" : " ", ratio * car.corners.front().tyre.lateralPeak,
                    run.toThirty, run.toFifty, run.toEighty, run.toHundred, run.thirtyToHundred, run.sixtyToHundred,
                    run.eightyToOneTwenty);
    }

    std::printf("   * = as shipped\n");

    std::printf("\n--- does the tyre still behave like a tyre? ---\n");
    std::printf(" mu_x/mu_y   peak slip   peak Fx/Fz   past-peak   across-axle   peak rpm   shift rpm\n");

    for (auto index = std::size_t{0}; index < ratios.size(); index++)
    {
        const auto& run = results[index];

        std::printf("   %5.3f      %7.3f      %6.3f      %5.1f%%       %6.3f     %6.0f    ", ratios[index],
                    run.peakDrivenSlip, run.peakForceRatio, 100.0 * run.tractionLimitedFraction,
                    run.worstAcrossAxleSlip, run.peakEngineRpm);

        for (auto shift = std::uint32_t{0}; shift < std::min(run.shifts, std::uint32_t{3}); shift++)
        {
            std::printf("%.0f ", run.shiftRpm[shift]);
        }
        std::printf("\n");
    }

    std::printf("\n--- the velocity trace, km/h every half second ---\n");
    for (auto index = std::size_t{0}; index < ratios.size(); index++)
    {
        std::printf("   %5.3f  ", ratios[index]);
        for (auto sample = std::size_t{0}; sample < std::min(results[index].speedTrace.size(), std::size_t{14});
             sample++)
        {
            std::printf("%5.1f", results[index].speedTrace[sample]);
        }
        std::printf("\n");
    }
}

TEST_CASE("the control: the high-speed interval must barely move with grip", "[.grip-ratio]")
{
    // **The row that decides whether any of the sweep above means what it says.** By 80-120 km/h the
    // car is nowhere near its traction limit — drag and gearing dominate — so that interval is the
    // engine, the mass and the road load, and very nearly not the tyre at all. If it moves with mu_x
    // then the sweep is changing something besides longitudinal grip and every accelerating row is
    // uninterpretable.
    //
    // **What this is and is not.** It is the 80-120 *interval of the same standing-start run*, not a
    // fixed-gear pull: the stronger version of this control is `InGearProbe`'s, which holds one ratio
    // and measured 1.3% to 3.1% driven slip. This one shares the run with the rows above, which is
    // what makes it a control *for those rows* rather than a separate measurement. The peak slip
    // printed beside it is the **whole run's** and not the interval's — it is there to show how far
    // the launch end of the same run travels while this end does not.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround(6000.0)).value());
    REQUIRE(world.has_value());

    std::printf("\n--- control: 80-120 as an interval of the same run ---\n");
    std::printf(" mu_x/mu_y     0-30    80-120    whole-run peak slip\n");

    auto slowest = 0.0;
    auto quickest = 1e9;
    auto lowSpeedSpread = 0.0;
    auto firstToThirty = 0.0;

    for (const auto ratio : {0.95, 1.15, 1.30})
    {
        const auto car = withGripRatio(setup.value(), ratio);
        const auto run = accelerate(car, world.value(), 30.0);

        slowest = std::max(slowest, run.eightyToOneTwenty);
        quickest = std::min(quickest, run.eightyToOneTwenty);
        firstToThirty = firstToThirty > 0.0 ? firstToThirty : run.toThirty;
        lowSpeedSpread = std::max(lowSpeedSpread, firstToThirty / run.toThirty);

        std::printf("   %5.3f      %6.3f    %6.3f         %7.4f\n", ratio, run.toThirty, run.eightyToOneTwenty,
                    run.peakDrivenSlip);
    }

    std::printf("\n  across a 37%% change in mu_x:  0-30 moves %.1f%%,  80-120 moves %.2f%%\n",
                100.0 * (lowSpeedSpread - 1.0), 100.0 * (slowest / quickest - 1.0));
    std::printf("  That separation is the whole argument: the sweep acts where traction limits and\n");
    std::printf("  nowhere else. If 80-120 ever moves appreciably, stop and read the sweep again.\n");

    // Asserted rather than printed and admired.
    REQUIRE(slowest / quickest < 1.02);
}

TEST_CASE("braking is a check on the grip ratio and never an input to it", "[.grip-ratio]")
{
    // **Deliberately reported and deliberately not used to choose.** The value of two independent
    // longitudinal references is destroyed the moment a number is fitted to sit between them. So: if
    // the mu_x that acceleration wants also brings 100-0 into its published region without being asked
    // to, that is strong evidence about the tyre. If the two want substantially different numbers,
    // that is a finding about the tyre model — most likely its load sensitivity, since braking and
    // accelerating load the driven axle in opposite directions — and not an invitation to average.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround(400.0)).value());
    REQUIRE(world.has_value());

    std::printf("\n--- braking, 100-0 (Auto Bild Sportscars: 34.6 and 35.1 m, two tests) ---\n");
    std::printf("  **A published 100-0 is an ABS-on, pedal-on-the-floor stop.** The constant-pedal\n");
    std::printf("  column is the hypothetical this project has been comparing against all along.\n\n");
    std::printf(" mu_x/mu_y   mu_x    ABS on    best constant pedal   at pedal   ABS penalty\n");

    for (const auto ratio : ratios)
    {
        const auto car = withGripRatio(setup.value(), ratio);
        const auto best = bestStop(car, world.value(), 100.0 / 3.6);
        const auto abs = stopWithAntilock(car, world.value(), 100.0 / 3.6);

        REQUIRE(best.stopped);
        REQUIRE(abs.stopped);

        std::printf("   %5.3f%s  %5.3f   %7.2f          %7.2f          %4.2f      %+5.1f%%\n", ratio,
                    std::abs(ratio - shippedRatio) < 1e-9 ? "*" : " ", ratio * car.corners.front().tyre.lateralPeak,
                    abs.distance, best.distance, best.pedal, 100.0 * (abs.distance / best.distance - 1.0));
    }
}

TEST_CASE("the two references cannot be reconciled by load sensitivity", "[.grip-ratio]")
{
    // **Where the sweep leaves the argument.** Acceleration is satisfied at a ratio near 1.08 and
    // braking near 1.22 — the two published references over-determine the tyre and disagree by about
    // 13% in nominal mu_x. The obvious suspect is load sensitivity, because the two tests load the
    // driven axle in *opposite* directions: braking transfers weight onto the front, a launch takes it
    // off, and the same nominal peak therefore produces different effective friction in each.
    //
    // This case tests that suspect and clears it, using the tyre model's own `tyreFriction` rather
    // than a hand calculation beside it.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto& tyre = setup->corners.front().tyre;
    constexpr auto mass = 1452.0;
    constexpr auto gravity = 9.80665;
    constexpr auto centreHeight = 0.572;
    constexpr auto wheelbase = 2.638;
    constexpr auto frontFraction = 0.614;

    const auto staticFront = mass * gravity * frontFraction / 2.0;
    const auto transferPerWheel = [](const double g)
    {
        return mass * g * gravity * centreHeight / wheelbase / 2.0;
    };

    // Braking at about 1 g puts load on; launching at about 0.5 g takes it off.
    const auto brakingLoad = staticFront + transferPerWheel(1.0);
    const auto launchLoad = staticFront - transferPerWheel(0.5);

    REQUIRE(brakingLoad > launchLoad);

    // What each reference demands of the *nominal* peak, read off the sweep above.
    constexpr auto brakingRatio = 1.22;
    constexpr auto launchRatio = 1.08;

    const auto frictionAt = [&tyre](const double load, const double exponent)
    {
        auto probe = tyre;
        probe.longitudinalLoadSensitivity = exponent;

        return raceengine::tyreFriction(probe, raceengine::TyreAxis::Longitudinal, load, 1.0);
    };

    const auto shipped = tyre.longitudinalLoadSensitivity;

    // The *effective* friction each reference needs is fixed by the physics of its own test, so it is
    // computed once at the shipped exponent and then held while the exponent is varied.
    const auto brakingNeeds =
        brakingRatio * tyre.lateralPeak * frictionAt(brakingLoad, shipped) / tyre.longitudinalPeak;
    const auto launchNeeds = launchRatio * tyre.lateralPeak * frictionAt(launchLoad, shipped) / tyre.longitudinalPeak;

    std::printf("\n--- can load sensitivity reconcile the two references? ---\n");
    std::printf("  front wheel load: braking %.0f N, launch %.0f N, tyre nominal %.0f N\n", brakingLoad, launchLoad,
                tyre.nominalLoad);
    std::printf("\n   exponent   braking wants   launch wants   disagreement\n");

    auto atShipped = 0.0;
    auto atZero = 0.0;

    for (const auto exponent : {0.0, 0.05, shipped, 0.20, 0.30})
    {
        const auto braking =
            brakingNeeds * tyre.longitudinalPeak / (tyre.lateralPeak * frictionAt(brakingLoad, exponent));
        const auto launch = launchNeeds * tyre.longitudinalPeak / (tyre.lateralPeak * frictionAt(launchLoad, exponent));
        const auto gap = braking / launch - 1.0;

        if (std::abs(exponent - shipped) < 1e-12)
        {
            atShipped = gap;
        }
        if (exponent == 0.0)
        {
            atZero = gap;
        }

        std::printf("    %6.4f       %6.3f         %6.3f        %+6.1f%%%s\n", exponent, braking, launch, 100.0 * gap,
                    std::abs(exponent - shipped) < 1e-12 ? "   <- shipped" : "");
    }

    std::printf("\n  The disagreement GROWS with the exponent, so closing it needs a NEGATIVE one —\n");
    std::printf("  grip rising with load, which is not a tyre. And at zero load sensitivity, which is\n");
    std::printf("  itself contradicted by tyres.ini, %.1f%% still remains.\n", 100.0 * atZero);
    std::printf("\n  So load sensitivity is a CONTRIBUTOR and not the explanation. What is left points at\n");
    std::printf("  the curve's shape rather than its peak: braking sits at the peak (best constant pedal\n");
    std::printf("  0.34-0.48), while a launch spends 39-59%% of its run PAST the peak, out on the tail\n");
    std::printf("  where the asymptote D*sin(C*pi/2) governs. C, E and K are recorded in PublishedCarsImpl\n");
    std::printf("  as fitted to target behaviour rather than sourced. That is the next thing to test.\n");

    // The finding, asserted so it cannot quietly stop being true.
    REQUIRE(atShipped > 0.10);
    REQUIRE(atZero > 0.0);
    REQUIRE(atZero < atShipped);
}

TEST_CASE("how much of the 0-100 gap is launch technique rather than grip", "[.grip-ratio]")
{
    // **The control question nobody had asked, and it may invalidate the target.** This model has no
    // launch control: `AutoClutch::launchSpeed` is a dead field, the rev-based regulator that used to
    // stand in for one was deliberately removed on 2026-08-22 as an assist no dual clutch has, and the
    // launch fixture goes from an idle in gear on the brakes to full throttle in the same instant. No
    // brake-torquing, no engine pre-load, ~850 rpm on the crank when the clutch starts to close.
    //
    // **A manufacturer's DSG figure is not measured that way.** Launch control holds the clutches
    // against a built engine — typically three to three and a half thousand rpm — and meters the
    // take-up. So comparing this fixture against VW's 6.2 s is comparing two different manoeuvres, and
    // the project's old "6.4-6.7 s for a DSG without launch control" band was **asserted in eight places
    // and sourced in none**: no citation for it exists anywhere in this tree.
    //
    // This case measures the sensitivity rather than arguing about it. The engine is placed at a
    // stated speed with the clutch open — which is what a launch control release looks like from the
    // driveline's point of view — and everything else is held.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(world.has_value());

    const auto driveline = golfGtiMk7Driveline();
    const auto inertias = wheelInertias(setup.value());

    std::printf("\n--- 0-100 against the engine speed the clutch is released at ---\n");
    std::printf("  the model cannot perform a launch control launch; this places the engine where one\n");
    std::printf("  would have put it, which measures what the technique is worth.\n\n");
    std::printf("  release rpm    0-100 s    peak driven slip\n");

    auto atIdle = 0.0;
    auto best = 1e9;

    for (const auto releaseRpm : {850.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0})
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, designHeight(setup.value()), startZ);

        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        auto drivelineState = DrivelineState{};
        startEngine(driveline, drivelineState);

        auto road = std::array<double, cornerCount>{};
        const auto speeds = [&]
        {
            return std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                   state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};
        };

        {
            auto idling = VehicleInput{};
            idling.brake = 1.0;
            idling.gear = 1;

            for (auto held = 0; held < 360; held++)
            {
                const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
                REQUIRE(torques.has_value());

                const auto stepped = stepVehicle(setup.value(), state, idling, torques->wheel, world.value(), tick);
                REQUIRE(stepped.has_value());
                road = roadTorques(stepped.value());
            }
        }

        // The release: the engine where launch control would have built it to, and the clutch open.
        // **Stated rather than driven to**, because the automation closes the clutch the moment the
        // driver asks for torque and would drag the engine down instead of letting it build — which is
        // itself the finding, and is why this is placed rather than performed.
        drivelineState.engineSpeed = releaseRpm * 0.10471975511965977;
        drivelineState.clutchPedal = 1.0;

        auto toHundred = -1.0;
        auto peakSlip = 0.0;
        auto gear = 1;
        const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

        for (auto step = 1; step <= 360 * 20 && toHundred < 0.0; step++)
        {
            const auto roadSideSpeed =
                std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
            if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
            {
                gear++;
            }

            auto input = VehicleInput{};
            input.throttle = 1.0;
            input.gear = gear;

            const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup.value(), state, input, torques->wheel, world.value(), tick);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());

            for (const auto wheel : {std::size_t{0}, std::size_t{1}})
            {
                peakSlip = std::max(peakSlip, std::abs(stepped->corners[wheel].contact.slip.slipRatio));
            }

            if (std::abs(state.chassis.linearVelocity.z) * 3.6 >= 100.0)
            {
                toHundred = static_cast<double>(step) * tick;
            }
        }

        if (releaseRpm == 850.0)
        {
            atIdle = toHundred;
        }
        best = std::min(best, toHundred);

        std::printf("     %6.0f       %6.3f       %7.3f\n", releaseRpm, toHundred, peakSlip);
    }

    std::printf("\n  launch technique is worth %.3f s on this car (%.1f%%), against a %.3f s gap to the\n",
                atIdle - best, 100.0 * (atIdle - best) / atIdle, atIdle - 6.55);
    std::printf("  middle of the measured 6.5-6.6 s.\n");
    std::printf("\n  **Before any tyre parameter is changed, the target has to say which manoeuvre it is.**\n");

    REQUIRE(atIdle > 0.0);
    REQUIRE(best > 0.0);
    REQUIRE(best <= atIdle);
}

TEST_CASE("the post-peak tail, which is what a spinning wheel actually rides on", "[.grip-ratio]")
{
    // **Where the whole investigation converges.** Three separate findings point at the same place:
    //
    //   - the ratio sweep: acceleration wants mu_x/mu_y ~1.08, braking wants ~1.22, and load
    //     sensitivity cannot reconcile them — it makes the gap *worse*;
    //   - the launch-technique sweep: releasing at 3500 rpm is *slower* than at idle, so the launch is
    //     completely grip-saturated;
    //   - the arithmetic: in first gear this car feeds **1.89 times** what a front tyre can take, so it
    //     must spin, and 39-59% of the run is spent past the tyre's own peak.
    //
    // **Braking sits AT the peak; a launch rides the tail.** So the two references are reading two
    // different parts of one curve, and the parameter that sets the ratio between them is the shape
    // factor `C` — `D·sin(C·pi/2)` is the sliding asymptote. At the shipped 1.50 that is **0.707 of
    // peak**, which is the bottom of the 0.70-0.85 a real tyre shows. Raising the tail should speed
    // acceleration and leave braking untouched, which is the exact shape of the disagreement.
    //
    // **C is one of only three numbers in this tyre marked fitted rather than sourced**, so it is a
    // legitimate thing to question — unlike the mass, the gearing or the efficiency.
    //
    // **The trap this case is built to avoid:** changing `C` moves the peak's *height ratio* and the
    // peak's *location* together, because `B = K/(C·mu)`. Sweeping it naively would change two things
    // and prove nothing. So each row is run twice — once holding `K` (the peak slides) and once
    // holding the peak slip (K is rescaled by `tan(pi/2C)·C`) — and **K is printed**, because the
    // record already notes a road tyre runs 10-30 and a curve needing 80 is not this tyre.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto accelWorld = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(accelWorld.has_value());
    const auto brakeWorld = PhysicsWorld::create(generateProvingGround(straightGround(400.0)).value());
    REQUIRE(brakeWorld.has_value());

    constexpr auto shippedShape = 1.50;
    constexpr auto shippedStiffness = 28.0;
    const auto anchor = std::tan(3.14159265358979323846 / (2.0 * shippedShape)) * shippedShape;

    std::printf("\n================ the longitudinal curve's tail ================\n");
    std::printf("mu_x is held at the shipped %.3f throughout. Only the SHAPE changes.\n",
                setup->corners.front().tyre.longitudinalPeak);
    std::printf("references: 0-100 6.5-6.6 s measured (n=2), 100-0 34.6-35.1 m\n");
    std::printf("\n   C     tail    K      peak slip    0-100 s    100-0 m    past-peak\n");

    for (const auto holdPeakSlip : {false, true})
    {
        std::printf("  %s\n", holdPeakSlip ? "-- peak slip held (K rescaled with C) --"
                                           : "-- K held at 28 (the peak slides with C) --");

        for (const auto shape : {1.25, 1.35, 1.45, shippedShape, 1.60})
        {
            auto car = setup.value();
            const auto stiffness =
                holdPeakSlip ? shippedStiffness * (std::tan(3.14159265358979323846 / (2.0 * shape)) * shape) / anchor
                             : shippedStiffness;

            for (auto& corner : car.corners)
            {
                corner.tyre.longitudinalShape = shape;
                corner.tyre.longitudinalStiffness = stiffness;
            }

            const auto run = accelerate(car, accelWorld.value(), 30.0);
            const auto best = bestStop(car, brakeWorld.value(), 100.0 / 3.6);
            REQUIRE(best.stopped);

            // Where this curve's own peak sits, read off the tyre rather than recomputed beside it.
            const auto probe = raceengine::evaluateTyre(car.corners.front().tyre, 3600.0,
                                                        raceengine::TyreSlip{.slipRatio = 0.05, .slipAngle = 0.0}, 1.0);

            std::printf("  %.2f   %.3f  %5.2f     %6.4f      %6.3f     %6.2f      %4.1f%%%s\n", shape,
                        std::sin(shape * 3.14159265358979323846 / 2.0), stiffness, probe.longitudinalPeakSlip,
                        run.toHundred, best.distance, 100.0 * run.tractionLimitedFraction,
                        shape == shippedShape && !holdPeakSlip ? "   <- shipped" : "");
        }
    }

    std::printf("\n  **Read the two columns against each other.** If 0-100 moves with the tail while\n");
    std::printf("  100-0 stays put, the two references are reading one curve in two places and the\n");
    std::printf("  disagreement is a SHAPE defect, not a grip-level one — and mu_x should not be touched.\n");
    std::printf("  If braking moves too, the tail is not the mechanism and this is back open.\n");
}

TEST_CASE("the candidate the two sweeps jointly point at, measured rather than extrapolated", "[.grip-ratio]")
{
    // **Extrapolating from two one-dimensional sweeps is not a measurement.** The mu_x sweep and the
    // tail sweep are differentially sensitive — braking responds more to the peak, acceleration more
    // to the tail — so a *combination* can in principle satisfy both references where neither alone
    // can. Reading the two sets of sensitivities against each other says the demands converge at a
    // tail near 0.59 (C ~1.60) with mu_x/mu_y ~1.25. That is a linear extrapolation across a 37% and
    // a 31% change, which is exactly the kind of thing that turns out to be wrong, so it is measured
    // here rather than believed.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto accelWorld = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(accelWorld.has_value());
    const auto brakeWorld = PhysicsWorld::create(generateProvingGround(straightGround(400.0)).value());
    REQUIRE(brakeWorld.has_value());

    std::printf("\n--- the joint candidate, measured ---\n");
    std::printf("  targets: 0-100 6.5-6.6 s (measured, n=2)   100-0 34.6-35.1 m\n\n");
    std::printf("   C     tail    mu_x/mu_y   mu_x     0-100 s    100-0 m    both?\n");

    for (const auto shape : {1.50, 1.60, 1.65})
    {
        for (const auto ratio : {1.20, 1.25, 1.30})
        {
            auto car = withGripRatio(setup.value(), ratio);
            for (auto& corner : car.corners)
            {
                corner.tyre.longitudinalShape = shape;
            }

            const auto run = accelerate(car, accelWorld.value(), 30.0);
            const auto best = bestStop(car, brakeWorld.value(), 100.0 / 3.6);
            REQUIRE(best.stopped);

            const auto accelOk = run.toHundred >= 6.4 && run.toHundred <= 6.7;
            const auto brakeOk = best.distance >= 34.4 && best.distance <= 35.3;

            std::printf("  %.2f   %.3f     %5.3f    %5.3f    %6.3f     %6.2f     %s\n", shape,
                        std::sin(shape * 3.14159265358979323846 / 2.0), ratio,
                        ratio * car.corners.front().tyre.lateralPeak, run.toHundred, best.distance,
                        accelOk && brakeOk ? "**BOTH**" : (accelOk ? "accel" : (brakeOk ? "brake" : "-")));
        }
    }

    std::printf("\n  A tail below ~0.70 is below what real tyres show (0.70-0.85), so a row that only\n");
    std::printf("  works down there is a finding about the model rather than a setting to adopt.\n");
}

TEST_CASE("launch control, as the car does it", "[.grip-ratio]")
{
    // **The procedure from the seat, reproduced**: traction control in sport, brake and throttle held
    // together until the engine settles near 3000 rpm on a slipping clutch, then the brake released.
    //
    // This exists because the earlier launch-technique experiment *placed* the engine at a speed with
    // the clutch open, which is not what launch control does — the clutch is loaded the whole time, so
    // the driveline is already wound and the tyre already at the limit when the brake comes off.
    // Placing a speed measured the engine; this measures the programme.
    //
    // Run as a 2x2 so the two halves of the procedure are separable: a car cannot tell you which of
    // its electronics did something if they are only ever switched on together.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(world.has_value());

    const auto inertias = wheelInertias(setup.value());

    std::printf("\n--- launch control, as the car does it ---\n");
    std::printf("  references: 0-100 6.5-6.6 s measured (n=2)\n\n");
    std::printf("   launch   traction    settled rpm   clutch    0-100 s   peak driven slip\n");

    for (const auto useLaunch : {false, true})
    {
        for (const auto mode : {TractionMode::Off, TractionMode::Sport})
        {
            auto driveline = golfGtiMk7Driveline();
            driveline.autoClutch.launch.enabled = useLaunch;

            auto assists = golfGtiMk7Assists(setup.value());
            assists.traction.mode = mode;

            auto state = VehicleState{};
            state.chassis.position = glm::dvec3(0.0, designHeight(setup.value()), startZ);
            for (auto step = 0; step < 1440; step++)
            {
                REQUIRE(
                    stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
            }

            auto drivelineState = DrivelineState{};
            startEngine(driveline, drivelineState);

            auto assistState = AssistState{};
            auto road = std::array<double, cornerCount>{};
            auto lastStep = VehicleStep{};

            const auto speeds = [&]
            {
                return std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                       state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};
            };

            const auto sense = [&]
            {
                auto sensors = AssistSensors{};
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
                }
                sensors.yawRate = lastStep.telemetry.yawRate;
                sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

                return sensors;
            };

            const auto drive = [&](VehicleInput input)
            {
                const auto command =
                    updateAssists(assists, assistState, sense(), {.brake = input.brake, .throttle = input.throttle},
                                  brakeCircuitPressures(setup.value(), input.brake), tick);
                input.throttle *= command.throttleScale;

                const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
                REQUIRE(torques.has_value());

                const auto stepped =
                    stepVehicle(setup.value(), state, input, torques->wheel, world.value(), tick, command.brakes);
                REQUIRE(stepped.has_value());
                road = roadTorques(stepped.value());
                lastStep = stepped.value();

                return stepped.value();
            };

            // --- phase one: both pedals down, and let the regulator settle the engine ---
            auto held = VehicleInput{};
            held.brake = 1.0;
            held.throttle = useLaunch ? 1.0 : 0.0;
            held.gear = 1;

            for (auto step = 0; step < 360 * 3; step++)
            {
                static_cast<void>(drive(held));
            }

            const auto settled = drivelineState.engineSpeed * 9.549296585513721;
            const auto holdingPedal = drivelineState.clutchPedal;

            // The preconditions the measurement rests on: the car really has not crept away, and where
            // launch control is on it really is armed and really is holding the engine near target.
            REQUIRE(std::abs(state.chassis.linearVelocity.z) < 0.2);
            REQUIRE(drivelineState.launchArmed == useLaunch);
            if (useLaunch)
            {
                const auto target = driveline.autoClutch.launch.targetSpeed * 9.549296585513721;
                REQUIRE(settled > 0.75 * target);
                REQUIRE(settled < 1.25 * target);
            }

            // --- phase two: brake off, throttle flat, nothing else changed ---
            auto gear = 1;
            const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;
            auto toHundred = -1.0;
            auto peakSlip = 0.0;

            for (auto step = 1; step <= 360 * 25 && toHundred < 0.0; step++)
            {
                const auto roadSide =
                    std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
                if (roadSide > upshiftSpeed && gear < driveline.gearbox.topGear())
                {
                    gear++;
                }

                auto input = VehicleInput{};
                input.throttle = 1.0;
                input.gear = gear;

                const auto stepped = drive(input);
                for (const auto wheel : {std::size_t{0}, std::size_t{1}})
                {
                    peakSlip = std::max(peakSlip, std::abs(stepped.corners[wheel].contact.slip.slipRatio));
                }

                if (std::abs(state.chassis.linearVelocity.z) * 3.6 >= 100.0)
                {
                    toHundred = static_cast<double>(step) * tick;
                }
            }

            REQUIRE(toHundred > 0.0);

            std::printf("     %-3s     %-8s     %6.0f      %5.3f    %6.3f       %6.3f\n", useLaunch ? "on" : "off",
                        mode == TractionMode::Sport ? "sport" : "off", settled, holdingPedal, toHundred, peakSlip);
        }
    }

    std::printf("\n  If launch control is worth only a few hundredths, the launch is grip-limited rather\n");
    std::printf("  than technique-limited, and no launch programme will close the gap to the reference.\n");
}

TEST_CASE("both references under the controls the real measurements were taken with", "[.grip-ratio]")
{
    // **The sweep, re-run with the control variable fixed.** Everything in this file before this case
    // was measured on a car with its electronics switched off, and both published references are
    // measurements of cars with theirs on: a 0-100 is a launch with traction control working, and a
    // 100-0 is a stop with ABS working and the pedal on the floor.
    //
    // Getting that wrong on the acceleration side made the car look 4% slow and pointed the whole
    // grip-ratio investigation at the tyre. This case is what the sweep should have been.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto accelWorld = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(accelWorld.has_value());
    const auto brakeWorld = PhysicsWorld::create(generateProvingGround(straightGround(400.0)).value());
    REQUIRE(brakeWorld.has_value());

    std::printf("\n================ both references, correct controls ================\n");
    std::printf("  0-100 with TRACTION CONTROL in sport, against 6.5-6.6 s measured (n=2)\n");
    std::printf("  100-0 with ABS on and the pedal floored, against 34.6-35.1 m (Auto Bild, 2 tests)\n\n");
    std::printf(" mu_x/mu_y   mu_x     0-100 s   verdict    100-0 m   verdict\n");

    for (const auto ratio : ratios)
    {
        const auto car = withGripRatio(setup.value(), ratio);

        const auto run = accelerate(car, accelWorld.value(), 30.0, TractionMode::Sport);
        const auto stop = stopWithAntilock(car, brakeWorld.value(), 100.0 / 3.6);
        REQUIRE(stop.stopped);

        const auto accelOk = run.toHundred >= 6.4 && run.toHundred <= 6.7;
        const auto brakeOk = stop.distance >= 34.4 && stop.distance <= 35.3;

        std::printf("   %5.3f%s  %5.3f    %6.3f    %-7s    %6.2f    %-7s\n", ratio,
                    std::abs(ratio - shippedRatio) < 1e-9 ? "*" : " ", ratio * car.corners.front().tyre.lateralPeak,
                    run.toHundred, accelOk ? "**ok**" : (run.toHundred < 6.4 ? "quick" : "slow"), stop.distance,
                    brakeOk ? "**ok**" : (stop.distance < 34.4 ? "short" : "long"));
    }

    std::printf("\n  If acceleration is satisfied at the shipped ratio and braking is not, the tyre's\n");
    std::printf("  longitudinal PEAK is not the defect — one of the two tests would have to be wrong\n");
    std::printf("  about the same tyre. The braking chain is then what to look at: the ABS penalty\n");
    std::printf("  (measured at 1.5-7%% here, against ~0 on a real car), the bias, or the reference.\n");
}

TEST_CASE("what the braking-satisfying grip does to acceleration, with everything switched on", "[.grip-ratio]")
{
    // **Humouring the obvious question**: if `mu_x` is set to whatever braking demands, what does the
    // car then do to 100 km/h with the electronics it actually has — traction control in sport, and
    // launch control?
    //
    // It is worth asking properly because the two references have to be judged on the *same* car, and
    // a ratio chosen to satisfy one of them is only interesting if the other is then reported under
    // the controls its own measurement was taken with.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto accelWorld = PhysicsWorld::create(generateProvingGround(straightGround(3000.0)).value());
    REQUIRE(accelWorld.has_value());
    const auto brakeWorld = PhysicsWorld::create(generateProvingGround(straightGround(400.0)).value());
    REQUIRE(brakeWorld.has_value());

    std::printf("\n=========== mu_x set to what BRAKING wants, acceleration reported ===========\n");
    std::printf("  0-100 measured reference 6.5-6.6 s (n=2)   |   100-0 reference 34.6-35.1 m\n\n");
    std::printf(" mu_x/mu_y   mu_x    100-0 (ABS)   0-100 TC     0-100 TC+LC    vs 6.55 s\n");

    for (const auto ratio : {shippedRatio, 1.20, 1.25, 1.27, 1.30})
    {
        const auto car = withGripRatio(setup.value(), ratio);

        const auto stop = stopWithAntilock(car, brakeWorld.value(), 100.0 / 3.6);
        REQUIRE(stop.stopped);

        const auto tc = accelerate(car, accelWorld.value(), 30.0, TractionMode::Sport, false);
        const auto both = accelerate(car, accelWorld.value(), 30.0, TractionMode::Sport, true);

        std::printf("   %5.3f%s  %5.3f     %6.2f       %6.3f       %6.3f       %+6.1f%%\n", ratio,
                    std::abs(ratio - shippedRatio) < 1e-9 ? "*" : " ", ratio * car.corners.front().tyre.lateralPeak,
                    stop.distance, tc.toHundred, both.toHundred, 100.0 * (both.toHundred / 6.55 - 1.0));
    }

    std::printf("\n  The last column is the price of satisfying the braking reference, paid in the\n");
    std::printf("  currency of the acceleration one — on the same car, with the same electronics.\n");
}
