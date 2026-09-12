#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::BrakeCommand;
using raceengine::bringUpJolt;
using raceengine::computeMassProperties;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::evaluateTyre;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::ModulatorPhase;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::seedTyreGasPressures;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::TyreAxis;
using raceengine::tyreFriction;
using raceengine::TyreModel;
using raceengine::TyreSlip;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;

// The grip-utilisation instrument. `./EngineTests "[.brake-utilisation]"`.
//
// Step 1 of `docs/braking-chain-brief.md`, and it is deliberately **an instrument and not a gate**.
// Dominic's rule for that brief, verbatim: *"don't assume the 95% utilisation target itself is the
// truth. Use it as a diagnostic, not a calibration target."* So there is no `REQUIRE` on a
// utilisation figure anywhere in this file, no band, and nothing here is pinned by a
// characterisation case. Its job is to say **which wheel, in which phase, is leaving grip on the
// table**, so that the next question is about a mechanism rather than about a mean.
//
// The `REQUIRE`s that *are* here are all preconditions — the car is rolling at the entry speed, at
// ride height, straight, on all four wheels, on the surface it thinks it is on. Every one of them is
// the [fixtures-must-assert-preconditions] rule, and five faults in one session on this project were
// fixtures measuring something other than what they claimed.
//
// **What it measures.** Per wheel, per tick, through a floored ABS-on 100-0: `Fz`, `Fx`, `mu_x(Fz)`,
// `|Fx| / (mu_x(Fz) * Fz)`, slip ratio, anti-lock phase and activity, caliper pressure, and the
// phase of the stop.
//
// **And it settles the denominator question the brief could not.** There are two figures on record
// for "what the tyre has" under braking — 1.037, `mu_x` at one front wheel's load, and 1.078, the
// whole car with load transfer — and the gap between them is the
// [one-wheel-calc-cannot-answer-a-two-axle-question] trap that already reversed the sign on the
// load-sensitivity prediction. Neither is quoted here as established. What is measured instead is
// the whole ladder, from the car's own loads tick by tick, plus an **oracle** run whose brake torque
// is commanded per wheel to hold each tyre at its own peak. The oracle is what turns the arithmetic
// into an arbiter: whichever candidate denominator it lands on is the one a perfect chain can reach.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto gravity = 9.80665;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto bar = 1.0e5;
constexpr auto degrees = 57.29577951308232;

// The plate runs z from 0 to its length and x from -width/2 to +width/2 — the trap
// `docs/brake-model-brief.md` records, where a fixture that assumes it is centred starts the car in
// mid-air and every stop it reports is a car with no brakes.
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;

constexpr auto noBrakePressure = std::array<double, cornerCount>{};

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

[[nodiscard]] SurfaceMesh gripPlate(const double grip)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(1);
    mesh->materials[0].gripMultiplier = grip;
    mesh->materials[0].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        mesh->surfaces[triangle] = std::uint32_t{0};
    }

    return mesh.value();
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
    }
}

// --- what one wheel was doing on one tick ---

struct WheelSample
{
    double load = 0.0;
    // Signed, in the tyre's own frame. Under braking it is negative, and the utilisation below takes
    // its magnitude — but the sign is kept because a wheel making *positive* Fx in the middle of a
    // stop is a finding and an absolute value would hide it.
    double forceLongitudinal = 0.0;
    double forceLateral = 0.0;
    // `mu_x(Fz)` at this wheel's own load and this wheel's own patch grip. Not a constant: it is the
    // whole reason the two candidate denominators disagree.
    double friction = 0.0;
    double slipRatio = 0.0;
    double slipAngle = 0.0;
    double peakSlip = 0.0;
    double surfaceGrip = 0.0;
    double pressure = 0.0;
    double brakeTorque = 0.0;
    // **What the ECU believed the slip was**, against `slipRatio` which is what it actually was.
    // Printed beside the truth rather than instead of it, because every anti-lock threshold in
    // `raceengine.assists` is measured against this number and a controller acting correctly on a
    // wrong belief looks exactly like a controller acting incorrectly. [print-the-raw-inputs-first],
    // and [wheel-thresholds-are-relative-to-the-car] — four anti-lock faults on this project were one
    // mistake, which was reading the wheel without reading what the ECU made of it.
    double estimatedSlip = 0.0;
    ModulatorPhase phase = ModulatorPhase::Passive;
    bool antilockActive = false;
    bool inContact = false;

    // Captured for the `[.brake-gap]` decomposition at the foot of this file, which has to be able to
    // say whether a thermal or a pressure arm actually moved the tyre it is named after. Read-only
    // telemetry: nothing above or below consumes them, so every figure this file printed before they
    // existed is unchanged to the bit.
    double coreTemperature = 0.0;
    double gasPressurePsi = 0.0;
    double discTemperature = 0.0;

    // How far past its own peak the tyre is working. 1.0 is exactly on the peak; below is
    // under-braked and above is on the falling side of the curve, where more pressure buys less
    // force. It is the channel utilisation cannot show: a Magic Formula's falling side is shallow, so
    // a wheel at three times its peak slip can still read 0.88 utilised.
    [[nodiscard]] double slipAgainstPeak() const
    {
        return peakSlip > 1e-9 ? std::abs(slipRatio) / peakSlip : 0.0;
    }

    // What this wheel could have made, and what it made of it. Both denominators are `mu*Fz`; the
    // achievable-peak check in the first case below is what says that is the right denominator for
    // *one wheel* rather than an upper bound it can never touch.
    [[nodiscard]] double capacity() const
    {
        return friction * load;
    }

    [[nodiscard]] double utilisation() const
    {
        const auto available = capacity();

        return available > 1.0 ? std::abs(forceLongitudinal) / available : 0.0;
    }
};

struct Sample
{
    double time = 0.0;
    double speed = 0.0;
    double deceleration = 0.0;
    double pitch = 0.0;
    // The ECU's own road speed, and whether it thinks it has one. On a four-wheel stop every wheel is
    // in control at once and the fastest of them is not at road speed, which is the honest failure of
    // a wheel-speed-only estimator — measured at 19% low on mu 0.35 in `docs/known-red.md`. Whether
    // it is biased on *dry* tarmac has never been printed.
    double referenceSpeed = 0.0;
    bool referenceValid = false;
    // The estimator's own confession channels: how long it has been carried by its rate limiter
    // rather than by a wheel, and what it believes the car's deceleration is.
    double referenceCoasting = 0.0;
    double referenceAcceleration = 0.0;
    std::array<WheelSample, cornerCount> wheels{};

    [[nodiscard]] double totalLoad() const
    {
        auto total = 0.0;
        for (const auto& wheel : wheels)
        {
            total += wheel.load;
        }

        return total;
    }

    [[nodiscard]] double totalForce() const
    {
        auto total = 0.0;
        for (const auto& wheel : wheels)
        {
            total += std::abs(wheel.forceLongitudinal);
        }

        return total;
    }

    [[nodiscard]] double totalCapacity() const
    {
        auto total = 0.0;
        for (const auto& wheel : wheels)
        {
            total += wheel.capacity();
        }

        return total;
    }
};

struct Run
{
    std::vector<Sample> samples;
    double distance = 0.0;
    double time = 0.0;
    double entrySpeed = 0.0;
    // The most any wheel was ever asked for, N.m. **The oracle's credibility turns on this.** A
    // per-wheel controller with an unbounded actuator is not a measurement of what this car can do,
    // it is a measurement of what a different car could — so the peak command is carried out of the
    // run and printed against the brake each corner actually has, rather than assumed to fit.
    std::array<double, cornerCount> peakCommand{};
    bool stopped = false;
    bool grounded = true;
    // How many ticks had at least one wheel off the road, and how many had a REAR wheel off. Read
    // only by `[.brake-gap]`; `grounded` above is the same fact as a boolean and is what every figure
    // printed before 2026-09-06 used.
    std::size_t airborneTicks = 0;
    std::size_t rearAirborneTicks = 0;

    [[nodiscard]] double meanDeceleration() const
    {
        return time > 0.0 ? entrySpeed / time / gravity : 0.0;
    }
};

// Which brake system is driving the stop.
//
// `Driver` is the car: a constant pedal through the hydraulics the car states, with whatever the
// assist layer is set to on top. `Oracle` is **not a proposal** — it commands each wheel's brake
// torque directly from the tyre's own peak slip, which is a quantity no wheel-speed sensor can
// observe. It exists to measure the ceiling the chassis allows, and it is the arbiter for the
// denominator question. Nothing in this file suggests shipping it.
// `BoundedOracle` is the same controller clamped at each corner's own peak brake torque — what a
// fully applied pedal makes there, after the servo and after the rear circuit's proportioning valve.
// It exists because the unbounded run turned out to ask the rear for **1.39 times the brake it has**,
// which the headroom check below caught: an unbounded actuator measures a car that does not exist.
// The pair is what separates *the controller* from *the hardware it is driving*.
//
// `SlewOracle` is the bounded oracle forced through the **modulator's own valve rates** from the
// **same initial condition a stamped pedal creates** — full pressure at every caliper on tick one,
// commands then slewing no faster than the dump and re-apply gradients allow, converted to torque
// through each corner's own N.m-per-pascal. It exists for the transient question alone: the first
// 150 ms of a floored stop tangles three owners — the valve (a real 1000 bar/s dump), the controller
// (its decisions during and after the dive), and the step-pedal fixture convention — and this is the
// **best any controller could possibly do through this valve from this start**. The gap between the
// real controller and this is the controller's share of the transient; between this and the free
// bounded oracle is the actuator's share, which no control law can touch.
enum class Actuator : std::uint8_t
{
    Driver,
    Oracle,
    BoundedOracle,
    SlewOracle
};

[[nodiscard]] constexpr bool isOracle(const Actuator actuator)
{
    return actuator != Actuator::Driver;
}

// The oracle's trim gain, dimensionless: how much of `Fz * r` it will add or remove per unit of
// normalised slip error. Feedforward carries the bulk — at steady state the torque that holds a
// wheel at constant speed is exactly `|Fx| * r_eff` — so this only has to close the error, and a
// proportional term is enough because the feedforward makes the steady-state error the wheel's own
// angular deceleration and nothing else. Measured, that residual is about 37 N.m against a 1600 N.m
// command, which is 4.6% of peak slip on a curve that is flat there.
constexpr auto oracleTrim = 0.5;

// `rampSeconds` spreads the pedal application over that long instead of a step. **Diagnostic only,
// and never the fixture convention**: every recorded brake figure in this project is a step
// application, `docs/known-red.md` explicitly rules the ramp out as a fix because it changes the
// measurement rather than the car — but the transient decomposition needs to *quantify* what the
// step convention itself is worth, and an informational row is the honest way to carry that number.
//
// `seedGasAtIdeal` puts the cavity air on the temperature that makes each tyre sit at its own ideal
// pressure, after the settle and before the roll. **Every figure this file printed before 2026-09-06
// was taken with it false**, which is the default here and is what keeps them reproducible — a
// default-constructed `TyreState` leaves the gas at `tyreDefaultTemperature` (65 °C) rather than at
// the 61.2 °C this car's cold-and-ideal pair implies, so the shipped fixture runs about 0.6 psi hard.
// The `[.brake-gap]` pressure arm measures both, because comparing a seeded model against an unseeded
// one is a false plant difference [a-shipped-model-creates-fixture-initial-conditions].
//
// `entrySpeed` exists for the `[.brake-gap]` ensemble alone and defaults to the fixture's own
// 100 km/h, so every figure this file printed before it existed is unchanged to the bit. A
// full-pedal anti-lock stop is chaotic in the same way the split-mu criteria are, and this project
// already settled how to measure one: a deterministic entry-speed ensemble inside the fixture's own
// settling tolerance, asserted on medians (`AntilockBrakingTests.cpp`, `steeringEnsemble`).
[[nodiscard]] Run record(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                         const double pedal, const Actuator actuator, const double rampSeconds = 0.0,
                         const bool seedGasAtIdeal = false, const double entrySpeed = hundred)
{
    auto state = VehicleState{};
    settle(setup, state, world, entrySpeed);

    if (seedGasAtIdeal)
    {
        seedTyreGasPressures(setup, state);
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

    // Half a second of rolling at the entry speed before anything is touched, so the suspension is
    // where a car arriving at a braking point actually is rather than where `settle` left it.
    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    // Preconditions. Nothing below is worth quoting without them.
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
        REQUIRE(lastStep.telemetry.wheels[index].gripMultiplier > 0.0);
    }
    REQUIRE(std::abs(state.chassis.linearVelocity.z - entrySpeed) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - designHeight) < 0.1);
    REQUIRE(std::abs(state.chassis.position.x) < 0.05);

    auto run = Run{};
    run.entrySpeed = state.chassis.linearVelocity.z;
    run.samples.reserve(360 * 8);

    const auto start = state.chassis.position;
    auto previousSpeed = run.entrySpeed;

    auto input = VehicleInput{};
    input.brake = pedal;

    // The oracle's per-wheel command, carried across ticks so the feedforward has something to be a
    // correction to.
    auto oracle = BrakeCommand{};
    oracle.commanded = true;

    // The slew oracle's actuator model: each corner's torque per pascal (exactly as
    // `golfGtiMk7Assists` derives it — divided out of the car, so a tuned brake carries through),
    // its command rate-limited by the modulator's own gradients, and **the stamp's initial
    // condition**: every caliper starts the stop at its full-pedal torque, because that is what a
    // step pedal has already done by the time any controller can act.
    const auto fullPressure = brakeCircuitPressures(setup, 1.0);
    auto slewCommand = std::array<double, cornerCount>{};
    if (actuator == Actuator::SlewOracle)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            slewCommand[index] = setup.corners[index].brakeTorque;
        }
    }

    for (auto step = 0; step < 360 * 30; step++)
    {
        auto brakes = BrakeCommand{};
        // **One call to `updateAssists` per tick and no more.** It advances the controller's own
        // state — sensor ages, modulator phases, the reference speed's filter — so calling it a
        // second time to read the channels off would run the anti-lock unit at twice the car's rate
        // and report a system that does not exist.
        auto assistChannels = raceengine::AssistChannels{};

        if (isOracle(actuator))
        {
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& solution = lastStep.corners[index];
                const auto& corner = setup.corners[index];

                // Where this tyre's own longitudinal curve peaks, at the load it is carrying now.
                // Read off the tyre rather than chosen, which is the whole difference between this
                // and a slip limiter with a number in it.
                const auto peakSlip = std::max(solution.contact.tyre.longitudinalPeakSlip, 1e-3);
                const auto slip = std::abs(solution.contact.slip.slipRatio);
                const auto radius = std::max(solution.contact.effectiveRadius, 1e-3);

                const auto feedforward = std::abs(solution.contact.tyre.longitudinal) * radius;
                const auto error = std::clamp((peakSlip - slip) / peakSlip, -1.0, 1.0);
                const auto trim = oracleTrim * solution.forces.tireVertical * radius * error;

                oracle.wheels[index] = std::max(0.0, feedforward + trim);

                // Rolling resistance is applied on top of whatever is commanded, so a wheel already
                // at its peak would be pushed past it by the tyre's own drag. Take it back out.
                oracle.wheels[index] = std::max(0.0, oracle.wheels[index] - corner.rollingResistance *
                                                                                solution.forces.tireVertical * radius);

                if (actuator != Actuator::Oracle)
                {
                    oracle.wheels[index] = std::min(oracle.wheels[index], corner.brakeTorque);
                }

                if (actuator == Actuator::SlewOracle)
                {
                    // The valve between the wish and the wheel: the command may fall no faster than
                    // the dump gradient and rise no faster than the re-apply gradient, each in this
                    // corner's own torque units. What the tyre law above computes is the *target*;
                    // what the wheel gets is the actuator chasing it.
                    const auto perPascal = fullPressure[index] > 0.0 ? corner.brakeTorque / fullPressure[index] : 0.0;
                    const auto down = assists.antilock.modulator.dumpGradient * perPascal * tick;
                    const auto up = assists.antilock.modulator.reapplyGradient * perPascal * tick;

                    slewCommand[index] =
                        std::clamp(oracle.wheels[index], slewCommand[index] - down, slewCommand[index] + up);
                    oracle.wheels[index] = slewCommand[index];
                }
            }

            brakes = oracle;
        }
        else
        {
            const auto applied = rampSeconds > 0.0 ? pedal * std::min(1.0, (run.time + tick) / rampSeconds) : pedal;
            input.brake = applied;

            const auto command = updateAssists(assists, assistState, sense(), {.brake = applied, .throttle = 0.0},
                                               brakeCircuitPressures(setup, applied), tick);
            brakes = command.brakes;
            assistChannels = command.channels;
        }

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        run.time += tick;

        auto sample = Sample{};
        sample.time = run.time;
        sample.speed = state.chassis.linearVelocity.z;
        sample.deceleration = (previousSpeed - sample.speed) / tick;
        sample.pitch = lastStep.telemetry.pitch;
        sample.referenceSpeed = assistChannels.referenceSpeed;
        sample.referenceValid = assistChannels.referenceValid;
        sample.referenceCoasting = assistChannels.referenceCoasting;
        sample.referenceAcceleration = assistChannels.referenceAcceleration;
        previousSpeed = sample.speed;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];
            auto& wheel = sample.wheels[index];

            wheel.load = solution.forces.tireVertical;
            wheel.forceLongitudinal = solution.contact.tyre.longitudinal;
            wheel.forceLateral = solution.contact.tyre.lateral;
            wheel.surfaceGrip = solution.patch.gripMultiplier;
            wheel.friction =
                tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, wheel.load, wheel.surfaceGrip);
            wheel.slipRatio = solution.contact.slip.slipRatio;
            wheel.slipAngle = solution.contact.slip.slipAngle;
            wheel.peakSlip = solution.contact.tyre.longitudinalPeakSlip;
            wheel.pressure = assistChannels.pressure[index];
            wheel.brakeTorque = brakes.commanded ? brakes.wheels[index] : 0.0;
            wheel.estimatedSlip = assistChannels.estimatedSlip[index];
            wheel.phase = assistChannels.antilockPhase[index];
            wheel.antilockActive = assistChannels.antilockActive[index];
            wheel.inContact = lastStep.telemetry.wheels[index].inContact;
            wheel.coreTemperature = lastStep.telemetry.wheels[index].tyreCoreTemperature;
            wheel.gasPressurePsi = lastStep.telemetry.wheels[index].tyrePressurePsi;
            wheel.discTemperature = lastStep.telemetry.wheels[index].discTemperature;

            run.grounded = run.grounded && wheel.inContact;
            run.peakCommand[index] = std::max(run.peakCommand[index], wheel.brakeTorque);
        }

        {
            auto anyOff = false;
            auto rearOff = false;
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                anyOff = anyOff || !sample.wheels[index].inContact;
                rearOff = rearOff || (index >= 2 && !sample.wheels[index].inContact);
            }

            run.airborneTicks += anyOff ? 1 : 0;
            run.rearAirborneTicks += rearOff ? 1 : 0;
        }

        run.samples.push_back(sample);

        if (sample.speed <= 0.0)
        {
            run.stopped = true;
            break;
        }
    }

    run.distance = state.chassis.position.z - start.z;

    return run;
}

// --- phases ---
//
// **Named by what the car is doing rather than by thirds of a clock.** A stop has a transient at the
// top of it — the pressure building, the pitch mode ringing, the load arriving on the front axle —
// and a runout at the bottom where the anti-lock unit drops out and the wheel speed estimator stops
// meaning anything. Between them is the part every published figure is really about. Splitting by
// time alone puts the transient and the first part of the steady phase in the same bucket, which is
// how a 0.19 s rear-wheel lift got averaged into invisibility once already.

enum class Phase : std::uint8_t
{
    Transient,
    High,
    Middle,
    Runout,
    Count
};

constexpr auto phaseCount = static_cast<std::size_t>(Phase::Count);

[[nodiscard]] Phase phaseOf(const Sample& sample)
{
    if (sample.time < 0.30)
    {
        return Phase::Transient;
    }

    if (sample.speed < 5.0)
    {
        return Phase::Runout;
    }

    return sample.speed >= 15.0 ? Phase::High : Phase::Middle;
}

[[nodiscard]] const char* phaseName(const Phase phase)
{
    switch (phase)
    {
    case Phase::Transient:
        return "transient <0.3s";
    case Phase::High:
        return "high  >54 km/h";
    case Phase::Middle:
        return "mid 18-54 km/h";
    case Phase::Runout:
        return "runout <18km/h";
    case Phase::Count:
        break;
    }

    return "?";
}

[[nodiscard]] const char* modulatorName(const ModulatorPhase phase)
{
    switch (phase)
    {
    case ModulatorPhase::Passive:
        return "passive";
    case ModulatorPhase::Hold:
        return "hold";
    case ModulatorPhase::Dump:
        return "dump";
    case ModulatorPhase::Recover:
        return "recover";
    case ModulatorPhase::Reapply:
        return "reapply";
    }

    return "?";
}

// What a wheel averaged over a phase. Time-weighted, and every tick weighs the same because the
// integrator's is fixed.
struct Accumulated
{
    double force = 0.0;
    double capacity = 0.0;
    double load = 0.0;
    double slip = 0.0;
    double againstPeak = 0.0;
    double estimatedSlip = 0.0;
    double pressure = 0.0;
    std::size_t ticks = 0;
    std::size_t antilockTicks = 0;
    std::size_t pastPeakTicks = 0;

    void add(const WheelSample& wheel)
    {
        force += std::abs(wheel.forceLongitudinal);
        capacity += wheel.capacity();
        load += wheel.load;
        slip += std::abs(wheel.slipRatio);
        againstPeak += wheel.slipAgainstPeak();
        estimatedSlip += std::abs(wheel.estimatedSlip);
        pressure += wheel.pressure;
        ticks++;
        antilockTicks += wheel.antilockActive ? 1 : 0;
        pastPeakTicks += wheel.slipAgainstPeak() > 1.0 ? 1 : 0;
    }

    [[nodiscard]] double utilisation() const
    {
        return capacity > 1.0 ? force / capacity : 0.0;
    }

    [[nodiscard]] double mean(const double total) const
    {
        return ticks > 0 ? total / static_cast<double>(ticks) : 0.0;
    }
};

// The best constant pedal, which is the "perfect threshold-braking driver" every earlier figure in
// this project is quoted against. Swept over the whole pedal because the optimum moves with
// everything — grip, brake torque, the valve — and a sweep ranged for one car measures every other
// off its optimum, which cost two wrong conclusions on 2026-08-23.
struct BestPedal
{
    double pedal = 0.0;
    Run run;
};

[[nodiscard]] BestPedal bestConstantPedal(const VehicleSetup& setup, const PhysicsWorld& world,
                                          const AssistSetup& assists)
{
    auto best = BestPedal{};
    auto shortest = 1e9;

    for (auto step = 2; step <= 20; step++)
    {
        const auto pedal = 0.05 * static_cast<double>(step);
        auto run = record(setup, world, assists, pedal, Actuator::Driver);

        if (run.stopped && run.distance < shortest)
        {
            shortest = run.distance;
            best.pedal = pedal;
            best.run = std::move(run);
        }
    }

    return best;
}

} // namespace

TEST_CASE("what one wheel can actually make, against the mu.Fz everything is divided by", "[.brake-utilisation]")
{
    // **The denominator has to be checked before it is used**, and this is the check. `mu(Fz)*Fz` is
    // the Magic Formula's *amplitude*, not necessarily a force the curve reaches: the shape and
    // curvature terms decide whether `sin(C*atan(...))` ever gets to one. If it does not, every
    // utilisation figure in this file is divided by a number the tyre cannot touch and the whole
    // instrument reads low by a constant nobody would ever see.
    //
    // Swept in pure longitudinal slip at the loads a stop actually puts on this car's wheels.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto& tyre = setup->corners[0].tyre;

    std::printf("\n=== is mu.Fz reachable, or is it an amplitude the curve never gets to? ===\n");
    std::printf("  swept in pure longitudinal slip, %s\n", "0 to 0.60 slip ratio in 6000 steps, at grip 1.0");
    std::printf("\n      Fz N     mu_x(Fz)      mu.Fz N    peak |Fx| N   peak/mu.Fz   at slip\n");

    for (const auto load : {1000.0, 2000.0, 2939.0, 4000.0, 5000.0, 5915.0, 7000.0})
    {
        const auto friction = tyreFriction(tyre, TyreAxis::Longitudinal, load, 1.0);

        auto peak = 0.0;
        auto atSlip = 0.0;

        for (auto step = 0; step <= 6000; step++)
        {
            const auto slip = 0.0001 * static_cast<double>(step);
            const auto forces = evaluateTyre(tyre, load, TyreSlip{.slipRatio = slip, .slipAngle = 0.0}, 1.0);

            if (std::abs(forces.longitudinal) > peak)
            {
                peak = std::abs(forces.longitudinal);
                atSlip = slip;
            }
        }

        std::printf("  %8.0f  %11.4f  %11.0f  %13.0f  %11.4f  %8.4f\n", load, friction, friction * load, peak,
                    peak / (friction * load), atSlip);
    }

    std::printf("\n  If the last column is 1.000 then mu.Fz IS the tyre's peak and the denominator is\n"
                "  sound. If it is materially below 1, every utilisation figure below is understated\n"
                "  by that factor and the shortfall this brief is chasing is partly an artefact.\n");
    std::printf("\n  Note what does NOT change with load: the slip the peak sits at. The tyre's\n"
                "  stiffness scales with Fz and its limit scales with mu(Fz)*Fz, so the curve's shape\n"
                "  parameter is a function of mu alone -- which is why load sensitivity moves the\n"
                "  peak's HEIGHT and barely moves its POSITION.\n");
}

TEST_CASE("the grip every wheel had and the grip it used, tick by tick through a floored ABS stop",
          "[.brake-utilisation]")
{
    // **Print the raw inputs before theorising.** [print-the-raw-inputs-first] — two "fully
    // diagnosed" causes on this project were both wrong this month, and nine per-sample numbers
    // settled it in ten minutes. So this case prints and does not conclude.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    const auto run = record(setup.value(), world.value(), assists, 1.0, Actuator::Driver);
    REQUIRE(run.stopped);

    std::printf("\n=== dry tarmac, pedal 1.00, anti-lock on: %.2f m in %.3f s, mean %.3f g%s ===\n", run.distance,
                run.time, run.meanDeceleration(), run.grounded ? "" : "  (a wheel left the road)");

    std::printf("\n--- the whole car, per wheel: load and what fraction of mu.Fz it is using ---\n");
    std::printf("  every 4th tick to 0.5 s, then every 24th\n");
    std::printf("\n    t [s]   speed   a [g]  pitch    %sFz N   %sFz N   %sFz N   %sFz N"
                "     %s     %s     %s     %s    car\n",
                cornerAbbreviation(raceengine::Corner::FrontLeft), cornerAbbreviation(raceengine::Corner::FrontRight),
                cornerAbbreviation(raceengine::Corner::RearLeft), cornerAbbreviation(raceengine::Corner::RearRight),
                cornerAbbreviation(raceengine::Corner::FrontLeft), cornerAbbreviation(raceengine::Corner::FrontRight),
                cornerAbbreviation(raceengine::Corner::RearLeft), cornerAbbreviation(raceengine::Corner::RearRight));

    for (auto index = std::size_t{0}; index < run.samples.size(); index++)
    {
        const auto& sample = run.samples[index];
        const auto stride = sample.time < 0.5 ? std::size_t{4} : std::size_t{24};

        if (index % stride != 0)
        {
            continue;
        }

        std::printf("  %7.4f  %6.2f  %6.3f %6.2f", sample.time, sample.speed, sample.deceleration / gravity,
                    sample.pitch * degrees);
        for (const auto& wheel : sample.wheels)
        {
            std::printf("  %7.0f", wheel.load);
        }
        for (const auto& wheel : sample.wheels)
        {
            std::printf("  %5.3f", wheel.utilisation());
        }
        std::printf("  %5.3f\n", sample.totalCapacity() > 1.0 ? sample.totalForce() / sample.totalCapacity() : 0.0);
    }

    // The two channels a mean cannot show: what the anti-lock unit is doing to the pressure, and
    // where the wheel is sitting relative to its own peak. Printed every tick over one window so the
    // cycle is legible rather than aliased -- the rear channel runs at 24.2 Hz on dry tarmac
    // (docs/known-red.md), which is 15 ticks a cycle, and a 24-tick stride cannot see it.
    const auto detail = [&run](const std::size_t corner, const double from, const double to)
    {
        std::printf("\n--- %s, every tick from %.2f s to %.2f s ---\n",
                    cornerAbbreviation(static_cast<raceengine::Corner>(corner)), from, to);
        std::printf("    t [s]      Fz N      Fx N    mu_x   mu.Fz N    util     slip   peak slip  slip/peak"
                    "   ECU slip   ref m/s    bar    phase    ABS\n");

        for (const auto& sample : run.samples)
        {
            if (sample.time < from || sample.time > to)
            {
                continue;
            }

            const auto& wheel = sample.wheels[corner];

            std::printf("  %7.4f  %8.0f  %8.0f  %6.4f  %8.0f  %6.3f  %7.4f  %9.4f  %9.3f  %9.4f  %8.3f  %5.1f"
                        "  %8s  %s\n",
                        sample.time, wheel.load, wheel.forceLongitudinal, wheel.friction, wheel.capacity(),
                        wheel.utilisation(), wheel.slipRatio, wheel.peakSlip, wheel.slipAgainstPeak(),
                        wheel.estimatedSlip, sample.referenceSpeed, wheel.pressure / bar, modulatorName(wheel.phase),
                        wheel.antilockActive ? "yes" : "no");
        }
    };

    // **Two windows, and the first one is the one nobody has looked at.** The phase means below put
    // the front wheels at several times their own peak slip inside the first 0.3 s, which is the
    // pedal arriving against a pitch transient that has not settled. A mean cannot show whether that
    // is one excursion or a sustained drag, and those want different answers.
    detail(0, 0.00, 0.20);
    detail(2, 0.00, 0.20);
    detail(0, 0.60, 0.80);
    detail(2, 0.60, 0.80);

    // The runout, anchored where the car actually enters it rather than at a guessed time — below
    // 5 m/s the phase means read the rear at half its capacity under both recovery laws, and a mean
    // cannot say whether that is the sensor going quiet, the modulator over-dumping, or the load
    // still settling.
    auto runoutStart = run.time;
    for (const auto& sample : run.samples)
    {
        if (sample.speed < 5.0)
        {
            runoutStart = sample.time;
            break;
        }
    }

    detail(0, runoutStart, runoutStart + 0.30);
    detail(2, runoutStart, runoutStart + 0.30);

    std::printf("\n  `slip/peak` is the channel to read: 1.000 is a tyre sitting exactly on its own\n"
                "  longitudinal peak. Below 1 the wheel is under-braked; above it the wheel is past\n"
                "  the peak and on the falling side of the curve, where more pedal buys less force.\n");
}

TEST_CASE("which wheel, in which phase, is leaving grip on the table", "[.brake-utilisation]")
{
    // The decomposition the brief asks for, and the one number that is *not* reported is a verdict.
    // Every figure here is a mean with its ticks stated beside it, and the ranking at the bottom is a
    // share of the unused force rather than a score against a target.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    const auto run = record(setup.value(), world.value(), assists, 1.0, Actuator::Driver);
    REQUIRE(run.stopped);

    auto byPhase = std::array<std::array<Accumulated, cornerCount>, phaseCount>{};
    auto whole = std::array<Accumulated, cornerCount>{};
    auto phaseTicks = std::array<std::size_t, phaseCount>{};

    for (const auto& sample : run.samples)
    {
        const auto phase = static_cast<std::size_t>(phaseOf(sample));
        phaseTicks[phase]++;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            byPhase[phase][index].add(sample.wheels[index]);
            whole[index].add(sample.wheels[index]);
        }
    }

    std::printf("\n=== %.2f m in %.3f s at %.3f g, pedal 1.00, anti-lock on ===\n", run.distance, run.time,
                run.meanDeceleration());

    std::printf("\n--- mean utilisation, by wheel and by phase ---\n");
    std::printf("\n  phase              ticks       %s      %s      %s      %s      car\n",
                cornerAbbreviation(raceengine::Corner::FrontLeft), cornerAbbreviation(raceengine::Corner::FrontRight),
                cornerAbbreviation(raceengine::Corner::RearLeft), cornerAbbreviation(raceengine::Corner::RearRight));

    for (auto phase = std::size_t{0}; phase < phaseCount; phase++)
    {
        if (phaseTicks[phase] == 0)
        {
            continue;
        }

        auto force = 0.0;
        auto capacity = 0.0;

        std::printf("  %-18s %6zu", phaseName(static_cast<Phase>(phase)), phaseTicks[phase]);
        for (const auto& wheel : byPhase[phase])
        {
            std::printf("  %6.3f", wheel.utilisation());
            force += wheel.force;
            capacity += wheel.capacity;
        }
        std::printf("  %7.3f\n", capacity > 1.0 ? force / capacity : 0.0);
    }

    {
        auto force = 0.0;
        auto capacity = 0.0;
        std::printf("  %-18s %6zu", "whole stop", run.samples.size());
        for (const auto& wheel : whole)
        {
            std::printf("  %6.3f", wheel.utilisation());
            force += wheel.force;
            capacity += wheel.capacity;
        }
        std::printf("  %7.3f\n", capacity > 1.0 ? force / capacity : 0.0);
    }

    std::printf("\n--- mean slip / peak slip, by wheel and by phase: 1.000 is exactly on the peak ---\n");
    std::printf("\n  phase              ticks       %s      %s      %s      %s\n",
                cornerAbbreviation(raceengine::Corner::FrontLeft), cornerAbbreviation(raceengine::Corner::FrontRight),
                cornerAbbreviation(raceengine::Corner::RearLeft), cornerAbbreviation(raceengine::Corner::RearRight));

    for (auto phase = std::size_t{0}; phase < phaseCount; phase++)
    {
        if (phaseTicks[phase] == 0)
        {
            continue;
        }

        std::printf("  %-18s %6zu", phaseName(static_cast<Phase>(phase)), phaseTicks[phase]);
        for (const auto& wheel : byPhase[phase])
        {
            std::printf("  %6.3f", wheel.mean(wheel.againstPeak));
        }
        std::printf("\n");
    }

    std::printf("\n--- what each wheel is carrying, and how hard it is being asked to work ---\n");
    std::printf("\n  wheel   mean Fz N   mean mu_x   mean |Fx| N   mean mu.Fz N   mean |slip|   ABS ticks\n");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& wheel = whole[index];
        const auto load = wheel.mean(wheel.load);

        std::printf("  %-6s  %9.0f  %10.4f  %12.0f  %13.0f  %11.4f  %6zu / %zu\n",
                    cornerAbbreviation(static_cast<raceengine::Corner>(index)), load,
                    load > 1.0 ? wheel.mean(wheel.capacity) / load : 0.0, wheel.mean(wheel.force),
                    wheel.mean(wheel.capacity), wheel.mean(wheel.slip), wheel.antilockTicks, wheel.ticks);
    }

    // **The channel utilisation cannot show, and the reason this table is here at all.** A Magic
    // Formula's falling side is shallow, so a tyre dragged to three times its peak slip still reports
    // a high utilisation — it is making most of `mu.Fz` while being nowhere near where it should be
    // working. Utilisation says how much force came out; this says whether the operating point is on
    // the right side of the curve, and the two can disagree completely.
    std::printf("\n--- where each tyre is sitting on its own curve ---\n");
    std::printf("\n  wheel   mean slip/peak   ticks past the peak   mean true slip   mean ECU slip   ECU error\n");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& wheel = whole[index];
        const auto trueSlip = wheel.mean(wheel.slip);
        const auto believed = wheel.mean(wheel.estimatedSlip);

        std::printf("  %-6s  %14.3f   %9zu / %-9zu  %14.4f  %14.4f  %+8.1f%%\n",
                    cornerAbbreviation(static_cast<raceengine::Corner>(index)), wheel.mean(wheel.againstPeak),
                    wheel.pastPeakTicks, wheel.ticks, trueSlip, believed,
                    trueSlip > 1e-6 ? 100.0 * (believed / trueSlip - 1.0) : 0.0);
    }

    // **The ranking, and the only aggregation in this file that is an argument rather than a table.**
    // A wheel's shortfall is `mu*Fz - |Fx|` summed over the stop; its share of the total says which
    // wheel to look at first. It is deliberately a *share* and not a distance: converting unused
    // force into metres needs an assumption about what the car would have done with it, and that
    // assumption is exactly the thing the next stage has to measure rather than assert.
    std::printf("\n--- the unused force, ranked ---\n");

    auto totalShortfall = 0.0;
    auto shortfall = std::array<double, cornerCount>{};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        shortfall[index] = std::max(0.0, whole[index].capacity - whole[index].force);
        totalShortfall += shortfall[index];
    }

    std::printf("\n  wheel   unused N.ticks   share of the total unused\n");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("  %-6s  %14.0f   %8.1f%%\n", cornerAbbreviation(static_cast<raceengine::Corner>(index)),
                    shortfall[index], totalShortfall > 0.0 ? 100.0 * shortfall[index] / totalShortfall : 0.0);
    }

    std::printf("\n  Read this as a pointer and not as a budget. A wheel at 0.6 with a small load is\n"
                "  worth less than a wheel at 0.9 with a large one, which is why the share is taken\n"
                "  in newtons rather than in utilisation points.\n");
}

TEST_CASE("which denominator is right: one front wheel's mu, or the whole car's", "[.brake-utilisation]")
{
    // **The question `docs/braking-chain-brief.md` refused to answer by arithmetic**, and it refused
    // for a good reason: the same trap reversed the sign on the load-sensitivity prediction, where a
    // hand calculation on one front wheel at 5577 N said the stop would get 1.1 m shorter and it got
    // half a metre longer, because braking unloads the *rear* below the tyre's nominal load and below
    // nominal a flatter exponent gives less grip, not more.
    //
    // So nothing here is computed at a representative load. Every figure is taken from the loads the
    // car actually had, tick by tick, and then an oracle is run to say which of them a perfect chain
    // can reach.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto sprung = computeMassProperties(setup->sprung);
    REQUIRE(sprung.has_value());
    const auto mass = sprung->mass + setup->unsprungMass();

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    const auto assisted = record(setup.value(), world.value(), assists, 1.0, Actuator::Driver);
    REQUIRE(assisted.stopped);

    const auto oracle = record(setup.value(), world.value(), assists, 1.0, Actuator::Oracle);
    REQUIRE(oracle.stopped);

    const auto bounded = record(setup.value(), world.value(), assists, 1.0, Actuator::BoundedOracle);
    REQUIRE(bounded.stopped);

    std::printf("\n=== the ladder, measured from the car's own loads ===\n");
    std::printf("  car mass %.1f kg, so 1 g is %.0f N of longitudinal force\n", mass, mass * gravity);

    // Both candidate denominators, tick by tick.
    //
    // `front-wheel mu` is the 1.037 construction: `mu_x` at ONE front wheel's load, used as though it
    // were the car's. That is only the car's figure if every wheel were at that load, and under
    // braking none of the rears is.
    //
    // `load-weighted mu` is the 1.078 construction done properly: `sum(mu_i * Fz_i) / sum(Fz_i)`,
    // which needs no mass and no assumption about distribution because both come out of the tick.
    const auto ladder = [&mass](const char* name, const Run& run)
    {
        auto frontMu = 0.0;
        auto weightedMu = 0.0;
        auto achievedMu = 0.0;
        auto achievedG = 0.0;
        auto againstPeak = std::array<double, cornerCount>{};
        auto ticks = std::size_t{0};

        for (const auto& sample : run.samples)
        {
            // The runout is excluded from the means and said so: below 5 m/s the anti-lock unit
            // drops out by design and the loads are still settling, so a mean taken across it is a
            // mean of two different experiments.
            if (sample.speed < 5.0)
            {
                continue;
            }

            const auto load = sample.totalLoad();
            if (load < 1.0)
            {
                continue;
            }

            frontMu += sample.wheels[0].friction;
            weightedMu += sample.totalCapacity() / load;
            achievedMu += sample.totalForce() / load;
            achievedG += sample.totalForce() / (mass * gravity);
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                againstPeak[index] += sample.wheels[index].slipAgainstPeak();
            }
            ticks++;
        }

        const auto n = ticks > 0 ? static_cast<double>(ticks) : 1.0;

        std::printf("\n  %s -- %.2f m, %.3f s, %.3f g mean\n", name, run.distance, run.time, run.meanDeceleration());
        std::printf("    front-wheel mu_x at its own load     %.4f   <- the 1.037 construction\n", frontMu / n);
        std::printf("    load-weighted mu_x over four wheels  %.4f   <- the 1.078 construction\n", weightedMu / n);
        std::printf("    achieved  sum|Fx| / sum Fz           %.4f\n", achievedMu / n);
        std::printf("    achieved  sum|Fx| / (m g)            %.4f g\n", achievedG / n);
        std::printf("    utilisation against front-wheel mu   %.3f\n", frontMu > 0.0 ? achievedMu / frontMu : 0.0);
        std::printf("    utilisation against load-weighted mu %.3f\n",
                    weightedMu > 0.0 ? achievedMu / weightedMu : 0.0);
        std::printf("    mean slip / peak slip, FL FR RL RR    ");
        for (const auto total : againstPeak)
        {
            std::printf("%6.3f", total / n);
        }
        std::printf("\n");

        return std::array{frontMu / n, weightedMu / n, achievedMu / n};
    };

    const auto assistedLadder = ladder("as the car brakes it: pedal 1.00, anti-lock on", assisted);
    const auto boundedLadder = ladder("the oracle, clamped at each corner's own peak brake torque", bounded);
    const auto oracleLadder = ladder("the oracle, unbounded: what the tyres and the chassis have", oracle);

    std::printf("\n=== what the oracle settles ===\n");
    std::printf("  The oracle holds every wheel at its own longitudinal peak using true slip, which no\n");
    std::printf("  wheel-speed sensor can observe. It is not a proposal. What it measures is the\n");
    std::printf("  ceiling THIS CHASSIS allows, with the real load transfer, the real suspension and\n");
    std::printf("  the real pitch transient in it -- and therefore which candidate denominator is a\n");
    std::printf("  number the car can reach rather than a number on a page.\n");
    // **Whether the oracle is a measurement of THIS car**, which turns entirely on the actuator. A
    // per-wheel controller commanding torque the calipers cannot make would be describing a different
    // brake system, and every metre it saved would be unreachable by any controller whatever.
    std::printf("\n  --- and does the oracle stay inside the brakes this car has? ---\n");
    std::printf("    wheel   peak commanded N.m   the corner's own peak N.m   headroom\n");

    auto insideHardware = true;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto available = setup->corners[index].brakeTorque;
        insideHardware = insideHardware && oracle.peakCommand[index] <= available;

        std::printf("    %-6s  %17.1f   %25.1f   %7.2fx\n", cornerAbbreviation(static_cast<raceengine::Corner>(index)),
                    oracle.peakCommand[index], available,
                    oracle.peakCommand[index] > 0.0 ? available / oracle.peakCommand[index] : 0.0);
    }
    std::printf("    %s\n",
                insideHardware
                    ? "Inside on every corner: the unbounded oracle's stop is reachable with the car's own brakes."
                    : "OUTSIDE on at least one corner -- so the UNBOUNDED row below is the tyre's ceiling and\n"
                      "    not this car's, and the clamped row is the one to compare a controller against.");

    std::printf("\n  unbounded oracle achieved  %.4f\n", oracleLadder[2]);
    std::printf("  clamped oracle achieved    %.4f\n", boundedLadder[2]);
    std::printf("  front-wheel candidate      %.4f   (unbounded oracle is %+.1f%% of it)\n", oracleLadder[0],
                100.0 * (oracleLadder[2] / oracleLadder[0] - 1.0));
    std::printf("  load-weighted candidate    %.4f   (unbounded oracle is %+.1f%% of it)\n", oracleLadder[1],
                100.0 * (oracleLadder[2] / oracleLadder[1] - 1.0));

    std::printf("\n  And what the car actually leaves on the table against each:\n");
    std::printf("    car / front-wheel candidate    %.3f\n",
                assistedLadder[0] > 0.0 ? assistedLadder[2] / assistedLadder[0] : 0.0);
    std::printf("    car / load-weighted candidate  %.3f\n",
                assistedLadder[1] > 0.0 ? assistedLadder[2] / assistedLadder[1] : 0.0);
    std::printf("    car / clamped oracle           %.3f   <- what the CONTROLLER costs\n",
                boundedLadder[2] > 0.0 ? assistedLadder[2] / boundedLadder[2] : 0.0);
    std::printf("    clamped / unbounded oracle     %.3f   <- what the BRAKE HARDWARE costs\n",
                oracleLadder[2] > 0.0 ? boundedLadder[2] / oracleLadder[2] : 0.0);
    std::printf("    distance: %.2f m, clamped oracle %.2f m (%+.1f%%), unbounded %.2f m (%+.1f%%)\n",
                assisted.distance, bounded.distance, 100.0 * (assisted.distance / bounded.distance - 1.0),
                oracle.distance, 100.0 * (assisted.distance / oracle.distance - 1.0));
}

TEST_CASE("the first 150 milliseconds, decomposed: the valve's share, the controller's, and the step's",
          "[.brake-utilisation]")
{
    // **The front transient, given the same treatment the rear equilibrium got: decomposed before
    // anything is proposed.** The event on record: a stamped pedal puts 124.7 bar at the caliper on
    // tick one, the modulator can shed 1000 bar/s, and the front wheels dive to 8 times their peak
    // slip before pressure can get below what pins them. Three owners are tangled in that sentence —
    // the valve (a real hardware rate), the controller (everything it decides during and after the
    // dive), and the step itself (this project's measurement convention, which `docs/known-red.md`
    // rules out changing because it would move every recorded brake figure).
    //
    // The separation is the slew oracle: perfect per-wheel knowledge, forced through the modulator's
    // own gradients from the same full-pressure start. Better than it is impossible through this
    // valve from this initial condition — so the controller's true headroom in the transient is the
    // gap to *it*, not the gap to the free oracle.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    struct Transient
    {
        double frontUtilisation = 0.0;
        double rearUtilisation = 0.0;
        double carUtilisation = 0.0;
        double frontSlipOverPeak = 0.0;
        double settled = -1.0;
        double distance = 0.0;
    };

    const auto transientOf = [](const Run& run)
    {
        auto result = Transient{};
        auto force = std::array<double, 2>{};
        auto capacity = std::array<double, 2>{};
        auto slipOverPeak = 0.0;
        auto ticks = std::size_t{0};
        auto settledTicks = 0;

        for (const auto& sample : run.samples)
        {
            const auto util = sample.totalCapacity() > 1.0 ? sample.totalForce() / sample.totalCapacity() : 0.0;

            // The first moment the whole car holds nine tenths of its grip for 50 ms **with every
            // wheel inside twice its own peak slip** — how long the transient effectively lasts.
            // The slip condition is not decoration: a wheel diving through the peak reads util 1.0
            // all the way down (the amplitude is still being delivered while the operating point
            // runs away), so utilisation alone declares a car "settled" in the middle of the dive.
            if (result.settled < 0.0)
            {
                auto healthy = util >= 0.9;
                for (const auto& wheel : sample.wheels)
                {
                    healthy = healthy && wheel.slipAgainstPeak() < 2.0;
                }

                settledTicks = healthy ? settledTicks + 1 : 0;
                if (settledTicks >= 18)
                {
                    result.settled = sample.time - 18.0 * tick;
                }
            }

            if (sample.time >= 0.30)
            {
                continue;
            }

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                force[index / 2] += std::abs(sample.wheels[index].forceLongitudinal);
                capacity[index / 2] += sample.wheels[index].capacity();
            }

            slipOverPeak += 0.5 * (sample.wheels[0].slipAgainstPeak() + sample.wheels[1].slipAgainstPeak());
            ticks++;
        }

        result.frontUtilisation = capacity[0] > 1.0 ? force[0] / capacity[0] : 0.0;
        result.rearUtilisation = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0;
        result.carUtilisation =
            capacity[0] + capacity[1] > 1.0 ? (force[0] + force[1]) / (capacity[0] + capacity[1]) : 0.0;
        result.frontSlipOverPeak = ticks > 0 ? slipOverPeak / static_cast<double>(ticks) : 0.0;
        result.distance = run.distance;

        return result;
    };

    std::printf("\n=== the transient (t < 0.3 s), four ways from the same stamp ===\n");
    std::printf("  system                        F util   R util   car util   F slip/peak   settled   100-0\n");

    const auto row = [&transientOf](const char* name, const Run& run)
    {
        const auto measured = transientOf(run);

        std::printf("  %-28s  %6.3f   %6.3f   %8.3f   %11.2f   %6.3f s  %6.2f m\n", name, measured.frontUtilisation,
                    measured.rearUtilisation, measured.carUtilisation, measured.frontSlipOverPeak, measured.settled,
                    measured.distance);
    };

    row("the car: ABS, step pedal", record(setup.value(), world.value(), assists, 1.0, Actuator::Driver));
    row("slew oracle (this valve)", record(setup.value(), world.value(), assists, 1.0, Actuator::SlewOracle));
    row("free oracle (no valve)", record(setup.value(), world.value(), assists, 1.0, Actuator::BoundedOracle));
    row("ABS, 150 ms pedal (info)", record(setup.value(), world.value(), assists, 1.0, Actuator::Driver, 0.15));

    std::printf("\n  car -> slew oracle is the CONTROLLER's share of the transient: what better\n");
    std::printf("  decisions through the same valve from the same stamp could still buy.\n");
    std::printf("  slew -> free oracle is the VALVE's share: unreachable by any control law.\n");
    std::printf("  The 150 ms row is what the step convention itself costs, and it is informational\n");
    std::printf("  only -- the step stays the fixture convention, docs/known-red.md says why.\n");
}

TEST_CASE("how the reference speed estimate tracks the car through a floored ABS stop", "[.brake-utilisation]")
{
    // **The estimator, printed against the truth it estimates — before anything about it is
    // changed.** The runout's phantom dumps were attributed to "staleness" from the phase means, and
    // that word turned out to be a narrative rather than a measurement: the tick trace shows the
    // reference 14-23% HIGH at low speed while falling slower than the car, which none of the
    // obvious mechanisms — floor-riding, radius bias, aged readings — cleanly produces. Four wrong
    // causes have been written into this project's records by exactly this kind of confidence.
    // [print-the-raw-inputs-first].
    //
    // Every column is either the estimator's own channel or reconstructed exactly from one: the
    // fastest sensed wheel is `ref x (1 - min estimated slip)`, since `estimatedSlip` is defined as
    // `(ref - sensed) / ref`.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;

    const auto run = record(setup.value(), world.value(), assists, 1.0, Actuator::Driver);
    REQUIRE(run.stopped);

    std::printf("\n=== reference vs truth, pedal 1.00, anti-lock on: %.2f m in %.3f s ===\n", run.distance, run.time);
    std::printf("  every 12th tick above 8 m/s, every 4th below\n");
    std::printf("\n    t [s]    true v    ref v    error%%   fastest wheel   ref decel   true decel   coasting\n");

    auto previous = run.entrySpeed;
    auto index = std::size_t{0};

    for (const auto& sample : run.samples)
    {
        const auto stride = sample.speed > 8.0 ? std::size_t{12} : std::size_t{4};
        const auto show = index % stride == 0;
        index++;

        const auto trueDecel = (previous - sample.speed) / tick;
        previous = sample.speed;

        if (!show || sample.speed <= 0.0)
        {
            continue;
        }

        auto minSlip = 1.0;
        for (const auto& wheel : sample.wheels)
        {
            minSlip = std::min(minSlip, wheel.estimatedSlip);
        }

        std::printf("  %7.4f  %7.3f  %7.3f  %+7.2f  %13.3f  %10.2f  %11.2f  %9.4f\n", sample.time, sample.speed,
                    sample.referenceSpeed, 100.0 * (sample.referenceSpeed / std::max(sample.speed, 1e-6) - 1.0),
                    sample.referenceSpeed * (1.0 - minSlip), -sample.referenceAcceleration, trueDecel,
                    sample.referenceCoasting);
    }
}

TEST_CASE("the three brake systems on one plate: driver, anti-lock, and a perfect one", "[.brake-utilisation]")
{
    // Same car, same surface, same entry speed. The only difference is what is deciding the torque.
    //
    // **The constant-pedal row is the reference every earlier figure in this project is quoted
    // against**, and `docs/known-red.md` is explicit that nobody is that driver: it holds one pedal
    // position exactly for the whole stop. It is in the table because it is what the 8.7% and then
    // 4.92% "ABS penalty" was measured against, and a comparison whose reference is off the page is
    // a comparison nobody can check.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto plain = golfGtiMk7Assists(setup.value());
    auto antilock = plain;
    antilock.antilock.enabled = true;

    const auto best = bestConstantPedal(setup.value(), world.value(), plain);
    REQUIRE(best.run.stopped);

    const auto floored = record(setup.value(), world.value(), plain, 1.0, Actuator::Driver);
    const auto assisted = record(setup.value(), world.value(), antilock, 1.0, Actuator::Driver);
    const auto bounded = record(setup.value(), world.value(), antilock, 1.0, Actuator::BoundedOracle);
    const auto oracle = record(setup.value(), world.value(), antilock, 1.0, Actuator::Oracle);

    std::printf("\n=== 100-0 on dry tarmac, five ways ===\n");
    std::printf("  published Mk7 GTI Performance: 34.6-35.1 m (Auto Bild Sportscars), 35.5 m kalt (amS)\n");
    std::printf("\n  system                          stop      time    mean g   car util   grounded\n");

    const auto row = [](const char* name, const Run& run)
    {
        auto force = 0.0;
        auto capacity = 0.0;

        for (const auto& sample : run.samples)
        {
            if (sample.speed < 5.0)
            {
                continue;
            }

            force += sample.totalForce();
            capacity += sample.totalCapacity();
        }

        std::printf("  %-28s  %7.2f m  %6.3f s  %7.3f   %8.3f   %s\n", name, run.distance, run.time,
                    run.meanDeceleration(), capacity > 1.0 ? force / capacity : 0.0, run.grounded ? "yes" : "NO");
    };

    row("best constant pedal", best.run);
    row("pedal 1.00, no electronics", floored);
    row("pedal 1.00, anti-lock on", assisted);
    row("oracle, clamped to the brakes", bounded);
    row("oracle, unbounded", oracle);

    std::printf("\n  best constant pedal was %.2f\n", best.pedal);
    std::printf("\n  anti-lock -> clamped oracle is what the CONTROLLER costs, on hardware the car has.\n");
    std::printf("  clamped -> unbounded oracle is what the BRAKE HARDWARE costs. The rear circuit does\n");
    std::printf("  run short of what the rear tyre would take -- the headroom table in the case above\n");
    std::printf("  shows it -- but only at a transient peak, and the two rows here say what that is\n");
    std::printf("  worth rather than leaving it to be argued about.\n");
    std::printf("  unbounded oracle -> the published figure is all that is left for the tyre and the\n");
    std::printf("  chassis, and it is the only part of the gap `docs/tyre-grip-ratio-brief.md` could\n");
    std::printf("  still be about.\n");
}

// =====================================================================================
//  THE CONTROLLER-GAP DECOMPOSITION -- `./EngineTests "[.brake-gap]"`, 2026-09-06
// =====================================================================================
//
// **The question, stated so it cannot drift.** The braking ledger above says the production
// anti-lock controller costs 6.04 m against a clamped oracle driving the same chassis and the same
// brakes. `docs/physics-and-tyre-model-report.md` records that this figure "has doubled" from a
// historical 2.99 m, and names four plant changes that entered production between the two
// measurements -- the geometric load path, the compliance trio, the thermal tyre and the cavity air
// -- without isolating them. This section runs that isolation.
//
// **It is diagnostic only.** Nothing here changes production. Every arm is a probe-local copy of
// `golfGtiMk7()` with one or more switches put back to what they were before a dated commit, and
// every arm is driven by **today's controller**, unmodified. What that buys and what it costs is
// stated in the first case below.
//
// **The two analyses are kept apart on purpose.** The chronological chain answers "what did each
// change cost the controller when it landed, on the plant that existed then"; the leave-one-out
// answers "what is each change worth in the plant that exists now". They are different questions and
// they do not have to agree -- the difference between them IS the interaction, which is section 7's
// residual and is measured rather than assumed away.

namespace
{

// Which of the changes that entered production between the historical ledger and today are present.
//
// **The four named changes are not the whole window and the chronology is not what the report
// assumed.** Traced with `git log -S` on `PublishedCarsImpl.cpp`:
//
//   7ff4c1a  2026-08-25  the droop stop's gap 20 -> 40 mm   (and the ABS improvements, see below)
//   9c0f409  2026-08-27  geometric load path + drivelineReaction + compliance STEER  (one commit)
//   c0c23c7  2026-08-28  the thermal tyre + the thermal brake                        (one commit)
//   acb3312  2026-08-29  compliance CAMBER
//   eb9bf51  2026-08-29  the rear damper's seal friction 107 -> 25 N
//   7551bfc  2026-08-29  the cavity air / pressure model switched on
//   2469b4f  2026-08-29  longitudinal RECESSION
//
// So the "compliance trio" is **not one chronological change**: steer landed with the load path and
// camber and recession landed four days later around the pressure switch-on. Four more changes in
// the same window are not among the named four at all, and three of them (the droop gap, the rear
// seal friction, the driveline reaction) act directly on the pitch transient and on how a dumping
// wheel settles, which is the controller's own loop. They are carried here so that the residual has
// somewhere to go rather than being declared unexplained.
struct Plant
{
    const char* name = "P4 (today)";

    // The four named changes.
    bool loadPath = true;
    bool complianceSteer = true;
    bool complianceCamber = true;
    bool recession = true;
    bool thermal = true;
    bool pressure = true;

    // The same window, not among the named four.
    bool drivelineReaction = true;
    bool brakeThermal = true;
    // The rear axle's 25 N seal friction; false restores the 107 N the front's Passat figure was
    // standing in with. It closed criterion 6 by changing how a dumping wheel settles.
    bool rearSealFriction25 = true;
    // The 40 mm droop gap; false restores the 20 mm it replaced, which is what was on the car when
    // the historical ledger was taken.
    bool droop40 = true;

    // Section 15's control, and it is not a plant change -- it is a fixture initial condition. See
    // `record`'s comment: the shipped fixture leaves the cavity air at 65 C, which is about 0.6 psi
    // over this tyre's own ideal. Both are run wherever the pressure model is on.
    bool seedGas = false;
};

[[nodiscard]] VehicleSetup plantOf(const VehicleSetup& base, const Plant& plant)
{
    auto setup = base;

    setup.geometricLoadPath = plant.loadPath;
    setup.drivelineReaction = plant.drivelineReaction;
    setup.tyreThermal = plant.thermal;
    setup.tyrePressure = plant.pressure;
    setup.brakeThermal = plant.brakeThermal;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& corner = setup.corners[index];

        if (!plant.complianceSteer)
        {
            corner.lateralForceSteer = 0.0;
        }

        if (!plant.complianceCamber)
        {
            corner.lateralForceCamber = 0.0;
        }

        if (!plant.recession)
        {
            corner.longitudinalForceRecession = 0.0;
        }

        // Rear corners only: the front's 107 N is its own measurement and never moved.
        if (!plant.rearSealFriction25 && index >= 2)
        {
            corner.damperFriction = 107.0;
        }

        if (!plant.droop40)
        {
            corner.droopStop.gap = 0.020;
        }
    }

    return setup;
}

// Every arm's five brake systems, in the order the ledger above prints them.
struct Ledger
{
    Run floored;
    Run assisted;
    Run bounded;
    Run unbounded;
    Run slew;
};

// The whole car's utilisation over the part of the stop a published figure is about -- the same
// construction the `row` lambda in the ledger case uses, so the numbers are comparable to it.
[[nodiscard]] double utilisationOf(const Run& run)
{
    auto force = 0.0;
    auto capacity = 0.0;

    for (const auto& sample : run.samples)
    {
        if (sample.speed < 5.0)
        {
            continue;
        }

        force += sample.totalForce();
        capacity += sample.totalCapacity();
    }

    return capacity > 1.0 ? force / capacity : 0.0;
}

// Taken past the first 50 ms and above the runout, because tick one carries the stamped pedal's own
// step and the last tick carries the car arriving at zero -- neither is a deceleration the car held.
[[nodiscard]] double peakDeceleration(const Run& run)
{
    auto peak = 0.0;

    for (const auto& sample : run.samples)
    {
        if (sample.time < 0.05 || sample.speed < 5.0)
        {
            continue;
        }

        peak = std::max(peak, sample.deceleration / gravity);
    }

    return peak;
}

[[nodiscard]] Ledger ledgerOf(const VehicleSetup& setup, const PhysicsWorld& world, const bool seedGas)
{
    const auto plain = golfGtiMk7Assists(setup);
    auto antilock = plain;
    antilock.antilock.enabled = true;

    return Ledger{
        .floored = record(setup, world, plain, 1.0, Actuator::Driver, 0.0, seedGas),
        .assisted = record(setup, world, antilock, 1.0, Actuator::Driver, 0.0, seedGas),
        .bounded = record(setup, world, antilock, 1.0, Actuator::BoundedOracle, 0.0, seedGas),
        .unbounded = record(setup, world, antilock, 1.0, Actuator::Oracle, 0.0, seedGas),
        .slew = record(setup, world, antilock, 1.0, Actuator::SlewOracle, 0.0, seedGas),
    };
}

// --- what the controller was doing, section 9 ---

struct ControllerStats
{
    std::array<std::size_t, cornerCount> dumps{};
    std::array<double, cornerCount> engaged{};
    std::array<double, cornerCount> frequency{};
    std::array<double, cornerCount> duty{};
    // Time the whole car spent below nine tenths of its own available grip while it still mattered.
    double underUtilised = 0.0;
    double firstIntervention = -1.0;
    double slipAtFirstIntervention = 0.0;
    // The reference-speed estimator against the chassis's own velocity. Signed, so a low estimate is
    // negative -- which is the direction that makes a controller believe a wheel is slipping less
    // than it is.
    double estimatorPeak = 0.0;
    double estimatorRms = 0.0;
    double estimatorMean = 0.0;
    double estimatorAtFirst = 0.0;
    double coastingFraction = 0.0;
};

[[nodiscard]] ControllerStats controllerStatsOf(const Run& run)
{
    auto stats = ControllerStats{};

    auto previous = std::array<ModulatorPhase, cornerCount>{};
    previous.fill(ModulatorPhase::Passive);

    auto errorSquared = 0.0;
    auto errorSum = 0.0;
    auto errorTicks = std::size_t{0};
    auto coastingTicks = std::size_t{0};
    auto ticks = std::size_t{0};

    for (const auto& sample : run.samples)
    {
        ticks++;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& wheel = sample.wheels[index];

            if (wheel.phase == ModulatorPhase::Dump && previous[index] != ModulatorPhase::Dump)
            {
                stats.dumps[index]++;
            }

            if (wheel.phase != ModulatorPhase::Passive)
            {
                stats.engaged[index] += tick;
            }

            if (wheel.antilockActive)
            {
                stats.duty[index] += tick;
            }

            previous[index] = wheel.phase;

            if (stats.firstIntervention < 0.0 && wheel.antilockActive)
            {
                stats.firstIntervention = sample.time;
                stats.slipAtFirstIntervention = std::abs(wheel.slipRatio);
                stats.estimatorAtFirst = sample.referenceValid ? sample.referenceSpeed - sample.speed : 0.0;
            }
        }

        if (sample.speed >= 5.0)
        {
            const auto capacity = sample.totalCapacity();
            if (capacity > 1.0 && sample.totalForce() / capacity < 0.9)
            {
                stats.underUtilised += tick;
            }
        }

        if (sample.referenceValid && sample.speed >= 5.0)
        {
            const auto error = sample.referenceSpeed - sample.speed;

            stats.estimatorPeak = std::abs(error) > std::abs(stats.estimatorPeak) ? error : stats.estimatorPeak;
            errorSquared += error * error;
            errorSum += error;
            errorTicks++;
        }

        coastingTicks += sample.referenceCoasting > 0.0 ? 1 : 0;
    }

    if (errorTicks > 0)
    {
        stats.estimatorRms = std::sqrt(errorSquared / static_cast<double>(errorTicks));
        stats.estimatorMean = errorSum / static_cast<double>(errorTicks);
    }

    if (ticks > 0)
    {
        stats.coastingFraction = static_cast<double>(coastingTicks) / static_cast<double>(ticks);
    }

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        stats.frequency[index] =
            stats.engaged[index] > 0.05 ? static_cast<double>(stats.dumps[index]) / stats.engaged[index] : 0.0;
    }

    return stats;
}

// --- the load and slip picture, section 11 ---

struct AxleWindow
{
    double frontLoad = 0.0;
    double rearLoad = 0.0;
    double minimumRearLoad = 1e9;
    double frontSlip = 0.0;
    double rearSlip = 0.0;
    double frontUtilisation = 0.0;
    double rearUtilisation = 0.0;
};

[[nodiscard]] AxleWindow axleWindow(const Run& run, const double from, const double to)
{
    auto window = AxleWindow{};
    auto force = std::array<double, 2>{};
    auto capacity = std::array<double, 2>{};
    auto ticks = std::size_t{0};

    for (const auto& sample : run.samples)
    {
        if (sample.time < from || sample.time > to)
        {
            continue;
        }

        ticks++;

        auto rearLoad = 0.0;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& wheel = sample.wheels[index];
            const auto axle = index < 2 ? std::size_t{0} : std::size_t{1};

            force[axle] += std::abs(wheel.forceLongitudinal);
            capacity[axle] += wheel.capacity();

            if (axle == 0)
            {
                window.frontLoad += wheel.load;
                window.frontSlip += std::abs(wheel.slipRatio);
            }
            else
            {
                window.rearLoad += wheel.load;
                window.rearSlip += std::abs(wheel.slipRatio);
                rearLoad += wheel.load;
            }
        }

        window.minimumRearLoad = std::min(window.minimumRearLoad, rearLoad);
    }

    if (ticks > 0)
    {
        const auto count = 2.0 * static_cast<double>(ticks);
        window.frontLoad /= count;
        window.rearLoad /= count;
        window.frontSlip /= count;
        window.rearSlip /= count;
    }

    window.frontUtilisation = capacity[0] > 1.0 ? force[0] / capacity[0] : 0.0;
    window.rearUtilisation = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0;

    return window;
}

// The gap this whole section is about: what the production controller costs against a clamped oracle
// driving the same chassis through the same brakes.
[[nodiscard]] double gapOf(const Ledger& ledger)
{
    return ledger.assisted.distance - ledger.bounded.distance;
}

void printLedgerRow(const char* name, const Ledger& ledger)
{
    std::printf("  %-34s %7.2f  %7.2f  %7.2f  %7.2f  %7.2f   %6.2f  %6.2f\n", name, ledger.floored.distance,
                ledger.assisted.distance, ledger.slew.distance, ledger.bounded.distance, ledger.unbounded.distance,
                gapOf(ledger), ledger.assisted.distance - ledger.slew.distance);
}

// The arms, in the order they entered production. `P4` is today's car and is asserted to be so.
[[nodiscard]] Plant stageP0()
{
    return Plant{.name = "P0  2026-08-24 plant",
                 .loadPath = false,
                 .complianceSteer = false,
                 .complianceCamber = false,
                 .recession = false,
                 .thermal = false,
                 .pressure = false,
                 .drivelineReaction = false,
                 .brakeThermal = false,
                 .rearSealFriction25 = false,
                 .droop40 = false};
}

} // namespace

TEST_CASE("the fixture the braking ledger is measured on, frozen and printed", "[.brake-gap]")
{
    // **Section 1. Nothing below this line is worth reading without it.** Every figure in this
    // section comes out of one deterministic fixture and the decomposition is only as good as the
    // statement of what is being held constant.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto assists = golfGtiMk7Assists(setup.value());

    std::printf("\n=== the frozen braking experiment ===\n");
    std::printf("  probe          RaceEngine/Tests/BrakingUtilisationProbe.cpp, `record()`\n");
    std::printf("  authoritative  [.brake-utilisation], `the three brake systems on one plate`\n");
    std::printf("  entry speed    %.4f m/s (100.0 km/h), stamped after a 1440-tick settle\n", hundred);
    std::printf("  pre-roll       180 ticks (0.500 s) at the entry speed before the pedal moves\n");
    std::printf("  pedal          1.00, STEP (rampSeconds 0.0) -- the project's fixture convention\n");
    std::printf("  surface        flat proving-ground plate, %.0f x %.0f m, cell %.1f m, bumpiness 0\n", plateLength,
                plateWidth, 2.0);
    std::printf("  grip           gripMultiplier 1.00 on every triangle\n");
    std::printf("  timestep       %.9f s (%.0f Hz), fixed\n", tick, 1.0 / tick);
    std::printf("  road geometry  straight, level; car starts at x 0, z %.1f, y %.3f\n", startZ, designHeight);
    std::printf("  steering       0.0 throughout; no drive torque\n");
    std::printf("  ambient        AmbientConditions{} default: air %.1f C, track %.1f C\n", 20.0, 20.0);

    std::printf("\n  --- the car's own state at the fixture's start ---\n");
    std::printf("  tyre seed      surface/core/carcass %.1f C (tyreDefaultTemperature, plateau centre)\n",
                raceengine::tyreDefaultTemperature);
    std::printf("  cavity seed    gas %.1f C -- the DEFAULT-CONSTRUCTED value, NOT the ideal-pressure seed\n",
                raceengine::tyreDefaultTemperature);
    const auto fresh = VehicleState{};
    std::printf("  disc seed      %.1f C (VehicleState{} default, and a pad is flat cold)\n",
                fresh.corners[0].discTemperature);

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& pressure = setup->corners[index].tyre.pressure;
        auto seeded = raceengine::TyreState{};
        raceengine::seedTyreGasAtIdealPressure(pressure, seeded);

        const auto unseeded = raceengine::TyreState{};
        const auto psi = 1.0 / 6894.757293168361;

        std::printf("  %s  cold %.1f psi at %.1f C, ideal %.1f psi; fixture runs %.2f psi (gas %.1f C), "
                    "ideal seed would be %.2f psi (gas %.1f C)\n",
                    cornerAbbreviation(static_cast<raceengine::Corner>(index)), pressure.coldPressure * psi,
                    pressure.coldReferenceTemperature, pressure.idealPressure * psi,
                    raceengine::tyrePressureAt(pressure, unseeded) * psi, unseeded.gasTemperature,
                    raceengine::tyrePressureAt(pressure, seeded) * psi, seeded.gasTemperature);
    }

    std::printf("\n  --- the electronics ---\n");
    std::printf("  ABS            OFF for the `no electronics` and oracle-plant rows; ON for `anti-lock on`\n");
    std::printf("  traction ctrl  mode %d (0 = off)\n", static_cast<int>(assists.traction.mode));
    std::printf("  cornering brake %s\n", assists.cornering.enabled ? "ON" : "off");
    std::printf("  yaw delay      %s\n", assists.antilock.yawMomentDelay ? "ON" : "off");
    std::printf("  recovery       authority %d (0 unconditional, 1 supervised, 2 disabled)\n",
                static_cast<int>(assists.antilock.recoveryAuthority));
    std::printf("  rear metering  reapply %.3e vs rear %.3e bar/s-equivalent (bit-inert)\n",
                assists.antilock.modulator.reapplyGradient, assists.antilock.modulator.rearReapplyGradient);

    std::printf("\n  --- the definitions, stated once ---\n");
    std::printf("  stopping distance  chassis z travelled from the tick the pedal is applied to the\n");
    std::printf("                     first tick with speed <= 0. Not a 100-0 corrected to any datum.\n");
    std::printf("  clamped oracle     per-wheel torque from the tyre's own peak slip, clamped at each\n");
    std::printf("                     corner's full-pedal brake torque. Uses TRUE slip: an instrument.\n");
    std::printf("  unbounded oracle   the same with the clamp removed -- a car whose brakes are bigger.\n");
    std::printf("  slew oracle        the clamped oracle forced through the modulator's own gradients\n");
    std::printf("                     from the stamped pedal's initial condition.\n");
    std::printf("  tyre utilisation   sum|Fx| / sum(mu_x(Fz)*Fz) over ticks with speed >= 5 m/s.\n");
    std::printf("  G                  D_ABS - D_clamped-oracle. THE quantity this section decomposes.\n");

    REQUIRE(setup->geometricLoadPath);
    REQUIRE(setup->tyreThermal);
    REQUIRE(setup->tyrePressure);
    REQUIRE(setup->brakeThermal);
    REQUIRE(setup->drivelineReaction);
    REQUIRE(!assists.antilock.enabled);
    REQUIRE(!assists.antilock.yawMomentDelay);
}

TEST_CASE("the current ledger, and the historical one restated like-for-like", "[.brake-gap]")
{
    // **Sections 2 and the historical reproduction, and the first finding is a bookkeeping one.**
    //
    // `docs/physics-and-tyre-model-report.md` states the historical controller share as
    // `41.74 - 38.75 = 2.99 m`. **38.75 m is the SLEW oracle**, not the clamped one -- it is the
    // bounded oracle forced through the modulator's own valve rates from a stamped-pedal start
    // (`docs/braking-chain-brief.md`, 2026-08-24 late, the transient decomposition table). The
    // clamped oracle on that same day was **36.92 m** (the same brief's 2026-08-24 table). So the
    // like-for-like historical clamped gap is `41.74 - 36.92 = 4.82 m`, and the "doubling" compares
    // two different denominators.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto ledger = ledgerOf(setup.value(), world.value(), false);

    std::printf("\n=== today's ledger, measured on the present tree ===\n");
    std::printf("\n  system                          stop      time    mean g   peak g   car util  grounded\n");

    const auto row = [](const char* name, const Run& run)
    {
        std::printf("  %-28s  %7.2f m  %6.3f s  %7.3f  %7.3f  %8.3f  %s\n", name, run.distance, run.time,
                    run.meanDeceleration(), peakDeceleration(run), utilisationOf(run), run.grounded ? "yes" : "NO");
    };

    row("pedal 1.00, no electronics", ledger.floored);
    row("pedal 1.00, anti-lock on", ledger.assisted);
    row("slew oracle (this valve)", ledger.slew);
    row("oracle, clamped to the brakes", ledger.bounded);
    row("oracle, unbounded", ledger.unbounded);

    std::printf("\n  published, verified at source (amS Supertest, kalt): 35.50 m, 1.108 g\n");

    std::printf("\n  --- the ledger's steps, today ---\n");
    std::printf("  ABS -> clamped oracle    %6.2f m   the CONTROLLER\n", gapOf(ledger));
    std::printf("  clamped -> unbounded     %6.2f m   the BRAKE HARDWARE\n",
                ledger.bounded.distance - ledger.unbounded.distance);
    std::printf("  unbounded -> published   %6.2f m   the TYRE and the CHASSIS\n", ledger.unbounded.distance - 35.5);
    std::printf("  ABS -> slew oracle       %6.2f m   the controller's share THROUGH THIS VALVE\n",
                ledger.assisted.distance - ledger.slew.distance);

    std::printf("\n  --- the historical figures, from the briefs, with their denominators named ---\n");
    std::printf("  2026-08-24 evening   car (ABS)                41.74 m\n");
    std::printf("  2026-08-24           oracle, clamped          36.92 m\n");
    std::printf("  2026-08-24           oracle, unbounded        36.79 m\n");
    std::printf("  2026-08-24 late      slew oracle              38.75 m\n");
    std::printf("\n  so the historical gaps were:\n");
    std::printf("    G(clamped) = 41.74 - 36.92 = 4.82 m     <-- the like-for-like predecessor of today's\n");
    std::printf("    G(slew)    = 41.74 - 38.75 = 2.99 m     <-- what the report quotes, a DIFFERENT arm\n");
    std::printf("\n  today:\n");
    std::printf("    G(clamped) = %.2f - %.2f = %.2f m\n", ledger.assisted.distance, ledger.bounded.distance,
                gapOf(ledger));
    std::printf("    G(slew)    = %.2f - %.2f = %.2f m\n", ledger.assisted.distance, ledger.slew.distance,
                ledger.assisted.distance - ledger.slew.distance);
    std::printf("\n  Like-for-like, G(clamped) moved 4.82 -> %.2f m, which is %+.2f m and not %+.2f m.\n",
                gapOf(ledger), gapOf(ledger) - 4.82, gapOf(ledger) - 2.99);
    std::printf("  The report's `2.99 -> 6.04, doubled` is %.2f m of denominator swap plus %.2f m of\n", 4.82 - 2.99,
                gapOf(ledger) - 4.82);
    std::printf("  real movement. Correct it forward; do not restate the old numbers.\n");
}

TEST_CASE("the chronological chain: what each plant change cost the controller when it landed", "[.brake-gap]")
{
    // **Sections 4 and 5.** Real historical revisions were NOT built. Every stage below is today's
    // tree with probe-local overrides putting the named switches back, and **today's controller is in
    // every arm**. That is stated rather than hidden because it is the one thing this decomposition
    // cannot separate: `7ff4c1a` (2026-08-25) and `1006b11` (2026-08-30) both changed
    // `Assists/Impl/BrakeControlImpl.cpp` inside the same window, and neither is behind a switch.
    // What P0 therefore measures is **the 2026-08-24 plant driven by the 2026-09-06 controller**.
    // The gap between P0's measured G and the historical 4.82 m is what the controller changes and
    // the un-revertable rest of the window are worth, together, and it is printed at the foot.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto stages = std::vector<Plant>{};

    stages.push_back(stageP0());

    // 7ff4c1a, 2026-08-25 -- not one of the named four.
    auto p0b = stages.back();
    p0b.name = "P0b +droop 40mm      (08-25)";
    p0b.droop40 = true;
    stages.push_back(p0b);

    // 9c0f409, 2026-08-27 -- the geometric load path, and two riders in the same commit.
    auto p1 = stages.back();
    p1.name = "P1  +load path       (08-27)";
    p1.loadPath = true;
    p1.drivelineReaction = true;
    p1.complianceSteer = true;
    stages.push_back(p1);

    // c0c23c7, 2026-08-28.
    auto p2 = stages.back();
    p2.name = "P2  +thermal tyre    (08-28)";
    p2.thermal = true;
    p2.brakeThermal = true;
    stages.push_back(p2);

    // acb3312 + eb9bf51 + 7551bfc, 2026-08-29 day.
    auto p3 = stages.back();
    p3.name = "P3  +camber,seal,gas (08-29)";
    p3.complianceCamber = true;
    p3.rearSealFriction25 = true;
    p3.pressure = true;
    stages.push_back(p3);

    // 2469b4f, 2026-08-29 night.
    auto p4 = stages.back();
    p4.name = "P4  +recession       (08-29n)";
    p4.recession = true;
    stages.push_back(p4);

    std::printf("\n=== the chronological chain, dry tarmac, pedal 1.00 ===\n");
    std::printf("  every stage is today's tree with switches put back; today's CONTROLLER throughout.\n");
    std::printf("\n  stage                                 no-ABS     ABS    slew  clamped  unbnd        "
                "G  A-slew\n");

    auto gaps = std::vector<double>{};
    auto ledgers = std::vector<Ledger>{};

    for (const auto& stage : stages)
    {
        const auto plant = plantOf(base.value(), stage);
        auto ledger = ledgerOf(plant, world.value(), false);

        REQUIRE(ledger.assisted.stopped);
        REQUIRE(ledger.bounded.stopped);

        printLedgerRow(stage.name, ledger);
        gaps.push_back(gapOf(ledger));
        ledgers.push_back(std::move(ledger));
    }

    std::printf("\n  --- dG, and the oracle/controller separation of every step (section 8) ---\n");
    std::printf("\n  step                                  d ABS   d clamped        dG   plant share  ctrl share\n");

    for (auto index = std::size_t{1}; index < stages.size(); index++)
    {
        const auto deltaAbs = ledgers[index].assisted.distance - ledgers[index - 1].assisted.distance;
        const auto deltaOracle = ledgers[index].bounded.distance - ledgers[index - 1].bounded.distance;

        std::printf("  %-34s %7.3f    %8.3f  %8.3f      %7.3f     %7.3f\n", stages[index].name, deltaAbs, deltaOracle,
                    gaps[index] - gaps[index - 1], deltaOracle, deltaAbs - deltaOracle);
    }

    std::printf("\n  P0 G = %.3f m against the historical clamped gap of 4.82 m: the difference, %+.3f m,\n", gaps[0],
                gaps[0] - 4.82);
    std::printf("  is everything this decomposition CANNOT revert -- the two controller commits, and\n");
    std::printf("  whatever else moved in the window that is not one of the switches above.\n");
    std::printf("\n  P0 -> P4 total dG = %+.3f m; the chain's increments sum to %+.3f m by construction.\n",
                gaps.back() - gaps.front(), gaps.back() - gaps.front());

    // P4 must be today's car exactly, or the chain is measuring something else.
    const auto today = ledgerOf(base.value(), world.value(), false);
    REQUIRE(std::abs(ledgers.back().assisted.distance - today.assisted.distance) < 1e-9);
    REQUIRE(std::abs(ledgers.back().bounded.distance - today.bounded.distance) < 1e-9);
}

TEST_CASE("leave-one-out from today's plant, and the interaction residual", "[.brake-gap]")
{
    // **Sections 6 and 7.** A different question from the chain above: what is each change worth in
    // the plant that exists now. The two do not have to agree and the difference is the interaction.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto today = ledgerOf(base.value(), world.value(), false);
    const auto todayGap = gapOf(today);

    struct Arm
    {
        const char* name;
        Plant plant;
    };

    auto arms = std::vector<Arm>{};

    const auto without = [](auto&& mutate)
    {
        auto plant = Plant{};
        mutate(plant);

        return plant;
    };

    arms.push_back({"minus geometric load path", without([](Plant& p) { p.loadPath = false; })});
    arms.push_back({"minus compliance (all three)", without(
                                                        [](Plant& p)
                                                        {
                                                            p.complianceSteer = false;
                                                            p.complianceCamber = false;
                                                            p.recession = false;
                                                        })});
    arms.push_back({"minus thermal tyre", without([](Plant& p) { p.thermal = false; })});
    arms.push_back({"minus cavity air / pressure", without([](Plant& p) { p.pressure = false; })});

    arms.push_back({"  .. minus compliance steer", without([](Plant& p) { p.complianceSteer = false; })});
    arms.push_back({"  .. minus compliance camber", without([](Plant& p) { p.complianceCamber = false; })});
    arms.push_back({"  .. minus recession", without([](Plant& p) { p.recession = false; })});

    arms.push_back({"[x] minus driveline reaction", without([](Plant& p) { p.drivelineReaction = false; })});
    arms.push_back({"[x] minus thermal brake", without([](Plant& p) { p.brakeThermal = false; })});
    arms.push_back({"[x] minus 25 N rear seal (107)", without([](Plant& p) { p.rearSealFriction25 = false; })});
    arms.push_back({"[x] minus 40 mm droop (20 mm)", without([](Plant& p) { p.droop40 = false; })});

    std::printf("\n=== leave-one-out from today's plant ===\n");
    std::printf("  [x] marks a change in the same window that is NOT one of the four named ones.\n");
    std::printf("\n  arm                                   no-ABS     ABS    slew  clamped  unbnd        "
                "G  A-slew\n");

    printLedgerRow("P4 (today)", today);

    auto sum = 0.0;
    auto namedSum = 0.0;
    auto measured = std::vector<Ledger>{};

    for (const auto& arm : arms)
    {
        auto ledger = ledgerOf(plantOf(base.value(), arm.plant), world.value(), false);
        REQUIRE(ledger.assisted.stopped);

        printLedgerRow(arm.name, ledger);

        const auto contribution = todayGap - gapOf(ledger);
        sum += contribution;

        if (arm.name[0] != ' ' && arm.name[0] != '[')
        {
            namedSum += contribution;
        }

        measured.push_back(std::move(ledger));
    }

    std::printf("\n  --- what removing each is worth, in G, and the plant/controller split ---\n");
    std::printf("  dG_LOO > 0 means the change ENLARGES today's controller gap.\n");
    std::printf("\n  arm                                  G   dG_LOO   d clamped     d ABS   plant    ctrl\n");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& ledger = measured[index];
        const auto deltaOracle = today.bounded.distance - ledger.bounded.distance;
        const auto deltaAbs = today.assisted.distance - ledger.assisted.distance;

        std::printf("  %-30s %6.3f  %7.3f   %8.3f  %8.3f  %6.3f  %6.3f\n", arms[index].name, gapOf(ledger),
                    todayGap - gapOf(ledger), deltaOracle, deltaAbs, deltaOracle, deltaAbs - deltaOracle);
    }

    std::printf("\n  today's G = %.3f m.  The four NAMED leave-one-out contributions sum to %+.3f m.\n", todayGap,
                namedSum);
    std::printf("  Every contribution above (named and unnamed) sums to %+.3f m -- and the sub-arms of\n", sum);
    std::printf("  compliance are counted twice in that figure, so read the named sum, not this one.\n");

    // The pressure arm's own control, section 15: seeded against unseeded, both plants otherwise
    // identical to today's.
    const auto seeded = ledgerOf(base.value(), world.value(), true);
    std::printf("\n  --- the cavity's initial condition, controlled (section 15) ---\n");
    printLedgerRow("P4, gas seeded at ideal", seeded);
    std::printf("  seeding the gas moves G by %+.3f m, the clamped oracle by %+.3f m and ABS by %+.3f m.\n",
                gapOf(seeded) - todayGap, seeded.bounded.distance - today.bounded.distance,
                seeded.assisted.distance - today.assisted.distance);
}

TEST_CASE("what the controller and its estimator were doing at every stage", "[.brake-gap]")
{
    // **Sections 9, 10 and 11 on one fixture.** No new telemetry: every channel below is one the
    // assist layer already publishes, read off the same runs the ledger above is built from.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    struct Arm
    {
        const char* name;
        Plant plant;
    };

    auto p1 = stageP0();
    p1.name = "P1  +load path";
    p1.droop40 = true;
    p1.loadPath = true;
    p1.drivelineReaction = true;
    p1.complianceSteer = true;

    auto p2 = p1;
    p2.name = "P2  +thermal";
    p2.thermal = true;
    p2.brakeThermal = true;

    auto p3 = p2;
    p3.name = "P3  +camber,seal,gas";
    p3.complianceCamber = true;
    p3.rearSealFriction25 = true;
    p3.pressure = true;

    auto p0b = stageP0();
    p0b.name = "P0b +droop 40mm";
    p0b.droop40 = true;

    const auto arms = std::vector<Arm>{
        {"P0  2026-08-24 plant", stageP0()}, {"P0b +droop 40mm", p0b}, {"P1  +load path", p1}, {"P2  +thermal", p2},
        {"P3  +camber,seal,gas", p3},        {"P4  (today)", Plant{}}};

    std::printf("\n=== the controller, per stage: duty, cycling, and where the grip went ===\n");
    std::printf("\n  stage                   first ABS   slip@1st   duty FL/RL     dumps FL/RL   "
                "Hz FL/RL   under-util s\n");

    auto ledgers = std::vector<Ledger>{};
    auto stats = std::vector<ControllerStats>{};

    for (const auto& arm : arms)
    {
        auto ledger = ledgerOf(plantOf(base.value(), arm.plant), world.value(), false);
        const auto stat = controllerStatsOf(ledger.assisted);

        std::printf("  %-22s  %8.4f   %8.4f   %5.3f/%5.3f   %5zu/%5zu   %5.1f/%5.1f   %8.3f\n", arm.name,
                    stat.firstIntervention, stat.slipAtFirstIntervention, stat.duty[0], stat.duty[2], stat.dumps[0],
                    stat.dumps[2], stat.frequency[0], stat.frequency[2], stat.underUtilised);

        ledgers.push_back(std::move(ledger));
        stats.push_back(stat);
    }

    std::printf("\n=== the reference-speed estimator against the chassis's own velocity (section 10) ===\n");
    std::printf("  signed, m/s. NEGATIVE means the ECU believes the car is SLOWER than it is, which\n");
    std::printf("  makes every wheel look less slipped than it is and under-brakes the car.\n");
    std::printf("\n  stage                    peak err    RMS err   mean err   err@1st ABS   coasting frac\n");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& stat = stats[index];
        std::printf("  %-22s  %+9.4f  %9.4f  %+9.4f     %+9.4f        %8.4f\n", arms[index].name, stat.estimatorPeak,
                    stat.estimatorRms, stat.estimatorMean, stat.estimatorAtFirst, stat.coastingFraction);
    }

    std::printf("\n=== load, slip and utilisation by window, ABS arm (section 11) ===\n");
    std::printf("\n  stage                   window        Fz f/r N      slip f/r      util f/r   min rear Fz N\n");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& run = ledgers[index].assisted;

        for (const auto window : {std::pair{0.0, 0.30}, std::pair{0.30, 1.00}, std::pair{1.00, 9.00}})
        {
            const auto measured = axleWindow(run, window.first, window.second);

            std::printf("  %-22s  %4.2f-%4.2f s  %5.0f/%5.0f  %6.4f/%6.4f  %5.3f/%5.3f   %13.1f\n",
                        window.first > 0.0 ? "" : arms[index].name, window.first, window.second, measured.frontLoad,
                        measured.rearLoad, measured.frontSlip, measured.rearSlip, measured.frontUtilisation,
                        measured.rearUtilisation, measured.minimumRearLoad);
        }
    }

    std::printf("\n=== the same windows for the CLAMPED ORACLE, so plant and controller separate ===\n");
    std::printf("\n  stage                   window        Fz f/r N      slip f/r      util f/r   min rear Fz N\n");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& run = ledgers[index].bounded;

        for (const auto window : {std::pair{0.0, 0.30}, std::pair{0.30, 1.00}})
        {
            const auto measured = axleWindow(run, window.first, window.second);

            std::printf("  %-22s  %4.2f-%4.2f s  %5.0f/%5.0f  %6.4f/%6.4f  %5.3f/%5.3f   %13.1f\n",
                        window.first > 0.0 ? "" : arms[index].name, window.first, window.second, measured.frontLoad,
                        measured.rearLoad, measured.frontSlip, measured.rearSlip, measured.frontUtilisation,
                        measured.rearUtilisation, measured.minimumRearLoad);
        }
    }

    std::printf("\n=== the tyre state each arm actually ran on (sections 14 and 15) ===\n");
    std::printf("\n  stage                   core start/end C    gas psi start/end    disc start/end C\n");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& run = ledgers[index].assisted;
        REQUIRE(!run.samples.empty());

        const auto& first = run.samples.front().wheels[0];
        const auto& last = run.samples.back().wheels[0];

        std::printf("  %-22s  %7.3f/%7.3f      %7.3f/%7.3f      %7.2f/%7.2f\n", arms[index].name, first.coreTemperature,
                    last.coreTemperature, first.gasPressurePsi, last.gasPressurePsi, first.discTemperature,
                    last.discTemperature);
    }
}

TEST_CASE("the two anti-lock reds at every stage, and the low-mu one", "[.brake-gap]")
{
    // **Sections 16 and 17.** Neither criterion is altered. What is reported is the *quantity* each
    // one asserts, measured through this probe's own fixture at every stage, so that a plant change
    // that enlarged G can be checked against a plant change that opened a red.
    //
    // The two are near-replicas rather than the criteria themselves -- `AntilockBrakingTests.cpp`'s
    // `stop()` helper is private to that translation unit -- so read the DIRECTION and the ordering,
    // not the absolute value against the criterion's bound.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto dry = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(dry.has_value());

    auto p1 = stageP0();
    p1.droop40 = true;
    p1.loadPath = true;
    p1.drivelineReaction = true;
    p1.complianceSteer = true;

    auto p2 = p1;
    p2.thermal = true;
    p2.brakeThermal = true;

    auto p3 = p2;
    p3.complianceCamber = true;
    p3.rearSealFriction25 = true;
    p3.pressure = true;

    auto p0b = stageP0();
    p0b.droop40 = true;

    struct Arm
    {
        const char* name;
        Plant plant;
    };

    const auto arms = std::vector<Arm>{{"P0  2026-08-24 plant", stageP0()},
                                       {"P0b +droop 40mm", p0b},
                                       {"P1  +load path", p1},
                                       {"P2  +thermal", p2},
                                       {"P3  +camber,seal,gas", p3},
                                       {"P4  (today)", Plant{}},
                                       {"P4 minus load path",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.loadPath = false;
                                            return p;
                                        }()},
                                       {"P4 minus compliance",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.complianceSteer = false;
                                            p.complianceCamber = false;
                                            p.recession = false;
                                            return p;
                                        }()},
                                       {"P4 minus thermal",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.thermal = false;
                                            return p;
                                        }()},
                                       {"P4 minus pressure",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.pressure = false;
                                            return p;
                                        }()},
                                       {"P4 minus droop 40mm", []
                                        {
                                            auto p = Plant{};
                                            p.droop40 = false;
                                            return p;
                                        }()}};

    std::printf("\n=== red 387, `all four wheels on the ground through a hard stop` ===\n");
    std::printf("  its fixture: half pedal, electronics OFF, dry. Reported here as the minimum REAR\n");
    std::printf("  AXLE load through the stop -- the continuous quantity behind the criterion's boolean.\n");
    std::printf("\n  stage                    grounded   min rear axle Fz N   min single-wheel Fz N\n");

    for (const auto& arm : arms)
    {
        const auto setup = plantOf(base.value(), arm.plant);
        const auto run = record(setup, dry.value(), golfGtiMk7Assists(setup), 0.50, Actuator::Driver);
        REQUIRE(run.stopped);

        auto minimumAxle = 1e9;
        auto minimumWheel = 1e9;

        for (const auto& sample : run.samples)
        {
            const auto axle = sample.wheels[2].load + sample.wheels[3].load;
            minimumAxle = std::min(minimumAxle, axle);
            minimumWheel = std::min({minimumWheel, sample.wheels[2].load, sample.wheels[3].load});
        }

        std::printf("  %-22s   %8s   %18.1f   %21.1f\n", arm.name, run.grounded ? "yes" : "NO", minimumAxle,
                    minimumWheel);
    }

    std::printf("\n=== criterion 6, the cycling frequency, dry full-pedal ABS ===\n");
    std::printf("  band 4 to 20 Hz. Dumps per second of ENGAGED time, front-left and rear-left.\n");
    std::printf("\n  stage                    FL Hz   FL engaged s      RL Hz   RL engaged s\n");

    for (const auto& arm : arms)
    {
        const auto setup = plantOf(base.value(), arm.plant);
        auto assists = golfGtiMk7Assists(setup);
        assists.antilock.enabled = true;

        const auto run = record(setup, dry.value(), assists, 1.0, Actuator::Driver);
        REQUIRE(run.stopped);

        const auto stat = controllerStatsOf(run);

        std::printf("  %-22s  %6.2f   %12.3f   %8.2f   %12.3f\n", arm.name, stat.frequency[0], stat.engaged[0],
                    stat.frequency[2], stat.engaged[2]);
    }

    std::printf("\n=== the low-mu red, `anti-lock is worth something on a slippery surface` ===\n");
    std::printf("  mu 0.35, pedal 1.00. Run ONLY at today's plant and at today's plant minus the\n");
    std::printf("  dominant high-mu contributor, per section 17 -- this is a check, not a second study.\n");
    std::printf("\n  arm                       locked m     ABS m     ABS worth        G\n");

    const auto slippery = PhysicsWorld::create(gripPlate(0.35));
    REQUIRE(slippery.has_value());

    const auto lowMuArms = std::vector<Arm>{{"P4 (today)", Plant{}},
                                            {"P4 minus compliance",
                                             []
                                             {
                                                 auto p = Plant{};
                                                 p.complianceSteer = false;
                                                 p.complianceCamber = false;
                                                 p.recession = false;
                                                 return p;
                                             }()},
                                            {"P4 minus recession only",
                                             []
                                             {
                                                 auto p = Plant{};
                                                 p.recession = false;
                                                 return p;
                                             }()},
                                            {"P4 minus driveline reaction",
                                             []
                                             {
                                                 auto p = Plant{};
                                                 p.drivelineReaction = false;
                                                 return p;
                                             }()},
                                            {"P4 minus load path",
                                             []
                                             {
                                                 auto p = Plant{};
                                                 p.loadPath = false;
                                                 return p;
                                             }()},
                                            {"P4 minus thermal",
                                             []
                                             {
                                                 auto p = Plant{};
                                                 p.thermal = false;
                                                 return p;
                                             }()},
                                            {"P0 (2026-08-24 plant)", stageP0()}};

    for (const auto& arm : lowMuArms)
    {
        const auto setup = plantOf(base.value(), arm.plant);
        const auto ledger = ledgerOf(setup, slippery.value(), false);

        REQUIRE(ledger.floored.stopped);
        REQUIRE(ledger.assisted.stopped);

        std::printf(
            "  %-22s  %9.2f  %9.2f   %+9.2f%%  %7.2f\n", arm.name, ledger.floored.distance, ledger.assisted.distance,
            100.0 * (ledger.floored.distance - ledger.assisted.distance) / ledger.floored.distance, gapOf(ledger));
    }
}

namespace
{

// A deterministic fifteen-member entry-speed ensemble, the same construction and the same +/-0.7%
// band `AntilockBrakingTests.cpp`'s `steeringEnsemble` uses. Every member is inside the fixture's own
// settling tolerance and is the same stop the ledger measures; what the ensemble buys is a MEDIAN,
// which is the only statistic this project has found that survives a chaotic assisted trajectory.
constexpr auto ensembleMembers = std::size_t{15};

struct Ensemble
{
    std::array<double, ensembleMembers> assisted{};
    std::array<double, ensembleMembers> bounded{};
    std::array<double, ensembleMembers> gap{};

    double medianAssisted = 0.0;
    double medianBounded = 0.0;
    double medianGap = 0.0;
    double lowestGap = 0.0;
    double highestGap = 0.0;
    double lowestAssisted = 0.0;
    double highestAssisted = 0.0;
    // The mean is carried beside the median because this distribution turned out to be BIMODAL: a
    // median on a two-cluster sample flips clusters when the mode fraction crosses a half, and the
    // mean does not. Neither is right on its own; disagreement between them is the warning.
    double meanAssisted = 0.0;
    double meanGap = 0.0;
    std::array<std::size_t, ensembleMembers> airborne{};
    // Per member, so the two clusters can be told apart by what the controller was doing rather than
    // only by how far the car went.
    std::array<double, ensembleMembers> underUtilised{};
    std::array<double, ensembleMembers> frontUtilisation{};
    std::array<double, ensembleMembers> rearUtilisation{};
    std::array<std::size_t, ensembleMembers> frontDumps{};
    std::array<std::size_t, ensembleMembers> rearDumps{};
    // Members whose rear axle ended the stop stranded near a third of its own capacity rather than
    // recovering past a half. Nothing between 0.42 and 0.57 was ever observed, so the threshold is a
    // gap in the data rather than a chosen number. This is the physical statistic the distance
    // distribution is made of.
    std::size_t strandedMembers = 0;
};

[[nodiscard]] double medianOfCopy(std::array<double, ensembleMembers> values)
{
    std::sort(values.begin(), values.end());

    return values[ensembleMembers / 2];
}

[[nodiscard]] Ensemble ensembleOf(const VehicleSetup& setup, const PhysicsWorld& world)
{
    const auto plain = golfGtiMk7Assists(setup);
    auto antilock = plain;
    antilock.antilock.enabled = true;

    auto ensemble = Ensemble{};

    for (auto k = -7; k <= 7; k++)
    {
        const auto index = static_cast<std::size_t>(k + 7);
        const auto entry = hundred * (1.0 + 0.001 * static_cast<double>(k));

        const auto assisted = record(setup, world, antilock, 1.0, Actuator::Driver, 0.0, false, entry);
        const auto bounded = record(setup, world, antilock, 1.0, Actuator::BoundedOracle, 0.0, false, entry);

        ensemble.assisted[index] = assisted.distance;
        ensemble.bounded[index] = bounded.distance;
        ensemble.gap[index] = assisted.distance - bounded.distance;
        ensemble.airborne[index] = assisted.rearAirborneTicks;

        const auto stat = controllerStatsOf(assisted);
        const auto whole = axleWindow(assisted, 0.0, 9.0);

        ensemble.underUtilised[index] = stat.underUtilised;
        ensemble.frontDumps[index] = stat.dumps[0];
        ensemble.rearDumps[index] = stat.dumps[2];
        ensemble.frontUtilisation[index] = whole.frontUtilisation;
        ensemble.rearUtilisation[index] = whole.rearUtilisation;
    }

    ensemble.medianAssisted = medianOfCopy(ensemble.assisted);
    ensemble.medianBounded = medianOfCopy(ensemble.bounded);
    ensemble.medianGap = medianOfCopy(ensemble.gap);
    ensemble.lowestGap = *std::min_element(ensemble.gap.begin(), ensemble.gap.end());
    ensemble.highestGap = *std::max_element(ensemble.gap.begin(), ensemble.gap.end());
    ensemble.lowestAssisted = *std::min_element(ensemble.assisted.begin(), ensemble.assisted.end());
    ensemble.highestAssisted = *std::max_element(ensemble.assisted.begin(), ensemble.assisted.end());

    auto assistedSum = 0.0;
    auto gapSum = 0.0;

    for (auto index = std::size_t{0}; index < ensembleMembers; index++)
    {
        assistedSum += ensemble.assisted[index];
        gapSum += ensemble.gap[index];
        ensemble.strandedMembers += ensemble.rearUtilisation[index] < 0.50 ? 1 : 0;
    }

    ensemble.meanAssisted = assistedSum / static_cast<double>(ensembleMembers);
    ensemble.meanGap = gapSum / static_cast<double>(ensembleMembers);

    return ensemble;
}

} // namespace

TEST_CASE("is a single floored anti-lock stop a measurement at all", "[.brake-gap]")
{
    // **The question the single-run decomposition raised and could not answer.** In the
    // leave-one-out above, removing the thermal tyre moved the ABS distance 2.09 m while moving the
    // clamped oracle by **exactly zero** -- the plant was bit-identical and the controller's stop was
    // two metres longer. Removing the thermal brake moved it 3.60 m the same way. A plant change that
    // does not move a perfect controller by a millimetre cannot legitimately move a real one by
    // metres through a mechanism; what it can do is land on the other side of a bifurcation.
    //
    // So before any attribution: hold the plant exactly and move only the entry speed, inside the
    // fixture's own settling tolerance. If the ABS distance spans metres across that band, no single
    // stop is a measurement of anything and the whole decomposition has to be read on medians.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto ensemble = ensembleOf(setup.value(), world.value());

    std::printf("\n=== one plant, fifteen entry speeds inside +/-0.7%% ===\n");
    std::printf("  the plant is byte-identical in every member. Only the entry speed moves.\n");
    std::printf("\n    entry km/h       ABS m   clamped m       G m  rear off  under-util  util f/r    dumps f/r\n");

    for (auto k = -7; k <= 7; k++)
    {
        const auto index = static_cast<std::size_t>(k + 7);
        std::printf("     %9.3f   %9.3f   %9.3f  %8.3f  %8zu  %8.3f  %5.3f/%5.3f  %4zu/%4zu\n",
                    3.6 * hundred * (1.0 + 0.001 * static_cast<double>(k)), ensemble.assisted[index],
                    ensemble.bounded[index], ensemble.gap[index], ensemble.airborne[index],
                    ensemble.underUtilised[index], ensemble.frontUtilisation[index], ensemble.rearUtilisation[index],
                    ensemble.frontDumps[index], ensemble.rearDumps[index]);
    }

    std::printf("\n  ABS      median %.3f, span %.3f to %.3f  (%.3f m wide)\n", ensemble.medianAssisted,
                ensemble.lowestAssisted, ensemble.highestAssisted, ensemble.highestAssisted - ensemble.lowestAssisted);
    std::printf("  clamped  median %.3f\n", ensemble.medianBounded);
    std::printf("  G        median %.3f, span %.3f to %.3f  (%.3f m wide)\n", ensemble.medianGap, ensemble.lowestGap,
                ensemble.highestGap, ensemble.highestGap - ensemble.lowestGap);
    std::printf("  G        mean   %.3f;   ABS mean %.3f\n", ensemble.meanGap, ensemble.meanAssisted);
    std::printf("  members with a STRANDED rear axle: %zu of %zu\n", ensemble.strandedMembers, ensembleMembers);
    std::printf("\n  The entry-speed band is 0.7%% -- smaller than every plant change measured above.\n");
    std::printf("  Whatever this span is, it is the noise floor any single-stop attribution sits on.\n");
    std::printf("\n  The clamped oracle is smooth and monotone across the same band: %.3f -> %.3f m,\n",
                ensemble.bounded.front(), ensemble.bounded.back());
    std::printf("  which is %.3f m over a 1.4%% speed change and is exactly the v^2 the fixture implies.\n",
                ensemble.bounded.back() - ensemble.bounded.front());
    std::printf("  So the plant is deterministic and well behaved; the ANTI-LOCK LOOP is what is not.\n");
}

TEST_CASE("the decomposition again, on ensemble medians", "[.brake-gap]")
{
    // **The whole of sections 5 and 6, re-run on the statistic that survives the chaos.** Same arms,
    // same switches, same controller; fifteen members each, medians reported. Read this table and not
    // the single-stop ones above -- those are kept because the difference between them is the finding.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto p0b = stageP0();
    p0b.name = "P0b +droop 40mm      (08-25)";
    p0b.droop40 = true;

    auto p1 = p0b;
    p1.name = "P1  +load path       (08-27)";
    p1.loadPath = true;
    p1.drivelineReaction = true;
    p1.complianceSteer = true;

    auto p2 = p1;
    p2.name = "P2  +thermal tyre    (08-28)";
    p2.thermal = true;
    p2.brakeThermal = true;

    auto p3 = p2;
    p3.name = "P3  +camber,seal,gas (08-29)";
    p3.complianceCamber = true;
    p3.rearSealFriction25 = true;
    p3.pressure = true;

    auto p4 = p3;
    p4.name = "P4  +recession       (08-29n)";
    p4.recession = true;

    const auto chain = std::vector<Plant>{stageP0(), p0b, p1, p2, p3, p4};

    std::printf("\n=== the chronological chain, ENSEMBLE MEDIANS ===\n");
    std::printf("\n  stage                                 ABS med  ABS mean   G med  G mean  G span  long/15\n");

    auto gaps = std::vector<double>{};
    auto means = std::vector<double>{};

    for (const auto& stage : chain)
    {
        const auto ensemble = ensembleOf(plantOf(base.value(), stage), world.value());

        std::printf("  %-34s %8.3f  %8.3f  %6.3f  %6.3f  %6.3f    %2zu\n", stage.name, ensemble.medianAssisted,
                    ensemble.meanAssisted, ensemble.medianGap, ensemble.meanGap,
                    ensemble.highestGap - ensemble.lowestGap, ensemble.strandedMembers);

        gaps.push_back(ensemble.medianGap);
        means.push_back(ensemble.meanGap);
    }

    std::printf("\n  step                                        dG(median)   dG(mean)\n");
    for (auto index = std::size_t{1}; index < chain.size(); index++)
    {
        std::printf("  %-34s      %+8.3f   %+8.3f\n", chain[index].name, gaps[index] - gaps[index - 1],
                    means[index] - means[index - 1]);
    }

    std::printf("\n  P0 -> P4 total dG(mean) = %+.3f m\n", means.back() - means.front());

    std::printf("\n  P0 -> P4 total dG(median) = %+.3f m\n", gaps.back() - gaps.front());
    std::printf("  P0 median G = %.3f m against the historical clamped gap of 4.82 m: %+.3f m.\n", gaps.front(),
                gaps.front() - 4.82);

    struct Arm
    {
        const char* name;
        Plant plant;
    };

    const auto arms = std::vector<Arm>{{"P4 (today)", Plant{}},
                                       {"minus geometric load path",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.loadPath = false;
                                            return p;
                                        }()},
                                       {"minus compliance (all three)",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.complianceSteer = false;
                                            p.complianceCamber = false;
                                            p.recession = false;
                                            return p;
                                        }()},
                                       {"minus thermal tyre",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.thermal = false;
                                            return p;
                                        }()},
                                       {"minus cavity air / pressure",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.pressure = false;
                                            return p;
                                        }()},
                                       {"[x] minus driveline reaction",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.drivelineReaction = false;
                                            return p;
                                        }()},
                                       {"[x] minus thermal brake",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.brakeThermal = false;
                                            return p;
                                        }()},
                                       {"[x] minus 25 N rear seal (107)",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.rearSealFriction25 = false;
                                            return p;
                                        }()},
                                       {"[x] minus 40 mm droop (20 mm)", []
                                        {
                                            auto p = Plant{};
                                            p.droop40 = false;
                                            return p;
                                        }()}};

    std::printf("\n=== leave-one-out from today's plant, ENSEMBLE MEDIANS ===\n");
    std::printf("\n  arm                              G med   G mean  dG_LOO med  dG_LOO mean  G span stranded\n");

    auto todayGap = 0.0;
    auto todayMean = 0.0;
    auto namedSum = 0.0;
    auto namedMeanSum = 0.0;

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto ensemble = ensembleOf(plantOf(base.value(), arms[index].plant), world.value());

        if (index == 0)
        {
            todayGap = ensemble.medianGap;
            todayMean = ensemble.meanGap;
        }

        const auto contribution = todayGap - ensemble.medianGap;
        const auto meanContribution = todayMean - ensemble.meanGap;

        std::printf("  %-30s %6.3f  %6.3f     %+7.3f      %+7.3f  %6.3f     %2zu\n", arms[index].name,
                    ensemble.medianGap, ensemble.meanGap, index == 0 ? 0.0 : contribution,
                    index == 0 ? 0.0 : meanContribution, ensemble.highestGap - ensemble.lowestGap,
                    ensemble.strandedMembers);

        if (index >= 1 && index <= 4)
        {
            namedSum += contribution;
            namedMeanSum += meanContribution;
        }
    }

    std::printf("\n  the four NAMED leave-one-out contributions sum to %+.3f m (median) / %+.3f m (mean)\n", namedSum,
                namedMeanSum);
    std::printf("  against a chronological total dG of %+.3f m (median) / %+.3f m (mean).\n",
                gaps.back() - gaps.front(), means.back() - means.front());
    std::printf("  Read every one of these against the G span in the last-but-one column: a dG_LOO\n");
    std::printf("  smaller than the span is not resolved by fifteen members.\n");
}

namespace
{

// The same ensemble at an arbitrary member count over the same +/-0.7% band, so the fifteen-member
// median can be checked for stability rather than trusted. Returns median, mean and the fraction of
// members in the stranded-rear cluster.
struct Resolution
{
    double medianGap = 0.0;
    double meanGap = 0.0;
    double strandedFraction = 0.0;
    std::size_t members = 0;
};

[[nodiscard]] Resolution resolutionOf(const VehicleSetup& setup, const PhysicsWorld& world, const std::size_t half)
{
    const auto plain = golfGtiMk7Assists(setup);
    auto antilock = plain;
    antilock.antilock.enabled = true;

    auto gaps = std::vector<double>{};
    auto stranded = std::size_t{0};

    const auto span = 0.007;
    const auto count = 2 * half + 1;

    for (auto k = std::size_t{0}; k < count; k++)
    {
        const auto offset = -span + 2.0 * span * static_cast<double>(k) / static_cast<double>(count - 1);
        const auto entry = hundred * (1.0 + offset);

        const auto assisted = record(setup, world, antilock, 1.0, Actuator::Driver, 0.0, false, entry);
        const auto bounded = record(setup, world, antilock, 1.0, Actuator::BoundedOracle, 0.0, false, entry);
        const auto whole = axleWindow(assisted, 0.0, 9.0);

        gaps.push_back(assisted.distance - bounded.distance);
        // The separator the member table found: the rear axle either recovers past a half of its own
        // capacity or is left stranded near a third. Nothing between 0.42 and 0.57 was observed.
        stranded += whole.rearUtilisation < 0.50 ? 1 : 0;
    }

    auto sorted = gaps;
    std::sort(sorted.begin(), sorted.end());

    auto sum = 0.0;
    for (const auto gap : gaps)
    {
        sum += gap;
    }

    return Resolution{.medianGap = sorted[count / 2],
                      .meanGap = sum / static_cast<double>(count),
                      .strandedFraction = static_cast<double>(stranded) / static_cast<double>(count),
                      .members = count};
}

} // namespace

TEST_CASE("how much of the decomposition survives a finer ensemble", "[.brake-gap]")
{
    // **The honesty check on section 5's own statistic.** A median over a bimodal sample flips when
    // the mode fraction crosses a half, so a fifteen-member median can move a whole cluster width on
    // a plant change that only nudged the fraction. Re-run at 15, 29 and 57 members over the SAME
    // band: whatever is stable across those is resolved, and whatever is not, is not.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    struct Arm
    {
        const char* name;
        Plant plant;
    };

    const auto arms = std::vector<Arm>{{"P4 (today)", Plant{}},
                                       {"P0 (2026-08-24 plant)", stageP0()},
                                       {"P4 minus compliance",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.complianceSteer = false;
                                            p.complianceCamber = false;
                                            p.recession = false;
                                            return p;
                                        }()},
                                       {"P4 minus driveline reaction", []
                                        {
                                            auto p = Plant{};
                                            p.drivelineReaction = false;
                                            return p;
                                        }()}};

    std::printf("\n=== is the median resolved? the same arms at three ensemble sizes ===\n");
    std::printf("  band is +/-0.7%% of entry speed in every case; only the sampling density changes.\n");
    std::printf("\n  arm                            n    G median    G mean   stranded-rear fraction\n");

    for (const auto& arm : arms)
    {
        const auto setup = plantOf(base.value(), arm.plant);

        for (const auto half : {std::size_t{7}, std::size_t{14}, std::size_t{28}})
        {
            const auto resolution = resolutionOf(setup, world.value(), half);

            std::printf("  %-28s %3zu    %8.3f  %8.3f                    %6.3f\n", half == 7 ? arm.name : "",
                        resolution.members, resolution.medianGap, resolution.meanGap, resolution.strandedFraction);
        }
    }

    std::printf("\n  The stranded-rear fraction is the physical statistic: what proportion of otherwise\n");
    std::printf("  identical stops end with the rear axle parked near a third of its own capacity\n");
    std::printf("  instead of recovering past a half. It is what the distance distribution is made of.\n");
}

TEST_CASE("the resolved summary: the stranded-rear fraction at every arm", "[.brake-gap]")
{
    // **The one table to read.** Twenty-nine members over the fixture's own +/-0.7% settling
    // tolerance, which the resolution check above showed reproduces the fifteen-member medians to
    // about 0.3 m and the stranded fraction to about 0.1. Every arm is the same plant switch set as
    // the tables above, and the statistic is the physical one: what proportion of otherwise identical
    // stops end with the rear axle parked near a third of its own capacity.
    const auto guard = JoltGuard{};

    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto p0b = stageP0();
    p0b.droop40 = true;

    auto p1 = p0b;
    p1.loadPath = true;
    p1.drivelineReaction = true;
    p1.complianceSteer = true;

    // P1 with its three riders separated, because they arrived in one commit and are not one thing.
    auto p1PathOnly = p0b;
    p1PathOnly.loadPath = true;

    auto p1ReactionOnly = p0b;
    p1ReactionOnly.drivelineReaction = true;

    auto p2 = p1;
    p2.thermal = true;
    p2.brakeThermal = true;

    // P3's three, separated the same way.
    auto p2Camber = p2;
    p2Camber.complianceCamber = true;

    auto p2Seal = p2;
    p2Seal.rearSealFriction25 = true;

    auto p2Gas = p2;
    p2Gas.pressure = true;

    auto p3 = p2;
    p3.complianceCamber = true;
    p3.rearSealFriction25 = true;
    p3.pressure = true;

    struct Arm
    {
        const char* name;
        Plant plant;
    };

    const auto arms = std::vector<Arm>{{"P0   2026-08-24 plant", stageP0()},
                                       {"P0b  +droop 40 mm", p0b},
                                       {"P0b  +load path only", p1PathOnly},
                                       {"P0b  +driveline reaction only", p1ReactionOnly},
                                       {"P1   +load path (whole commit)", p1},
                                       {"P2   +thermal tyre and brake", p2},
                                       {"P2   +compliance camber only", p2Camber},
                                       {"P2   +25 N rear seal only", p2Seal},
                                       {"P2   +cavity air only", p2Gas},
                                       {"P3   +camber, seal, gas", p3},
                                       {"P4   +recession  (TODAY)", Plant{}},
                                       {"P4   minus load path",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.loadPath = false;
                                            return p;
                                        }()},
                                       {"P4   minus compliance (all 3)",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.complianceSteer = false;
                                            p.complianceCamber = false;
                                            p.recession = false;
                                            return p;
                                        }()},
                                       {"P4   minus thermal tyre",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.thermal = false;
                                            return p;
                                        }()},
                                       {"P4   minus cavity air",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.pressure = false;
                                            return p;
                                        }()},
                                       {"P4   minus driveline reaction",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.drivelineReaction = false;
                                            return p;
                                        }()},
                                       {"P4   minus 25 N rear seal",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.rearSealFriction25 = false;
                                            return p;
                                        }()},
                                       {"P4   minus 40 mm droop",
                                        []
                                        {
                                            auto p = Plant{};
                                            p.droop40 = false;
                                            return p;
                                        }()},
                                       {"P4   minus recession", []
                                        {
                                            auto p = Plant{};
                                            p.recession = false;
                                            return p;
                                        }()}};

    std::printf("\n=== the resolved summary: 29 members, +/-0.7%% entry band ===\n");
    std::printf("\n  arm                                G median   G mean   stranded rear fraction\n");

    for (const auto& arm : arms)
    {
        const auto resolution = resolutionOf(plantOf(base.value(), arm.plant), world.value(), 14);

        std::printf("  %-32s %8.3f %8.3f                    %6.3f\n", arm.name, resolution.medianGap,
                    resolution.meanGap, resolution.strandedFraction);
    }

    std::printf("\n  Historical, for the record: the 2026-08-24 ledger's single stop gave G(clamped)\n");
    std::printf("  = 41.74 - 36.92 = 4.82 m, and today's single stop gives 43.53 - 37.49 = 6.05 m.\n");
    std::printf("  Both are ONE member of a distribution this table shows is metres wide.\n");
}

TEST_CASE("what actually strands the rear axle: two members of the same ensemble", "[.brake-gap]")
{
    // **The mechanism, traced.** Two members of today's plant's own ensemble, 0.1% of entry speed
    // apart: 99.500 km/h ends stranded and 99.600 km/h recovers. Same car, same surface, same
    // controller, same pedal. Whatever separates them is the whole of what this section is about.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto antilock = golfGtiMk7Assists(setup.value());
    antilock.antilock.enabled = true;

    const auto stranded =
        record(setup.value(), world.value(), antilock, 1.0, Actuator::Driver, 0.0, false, hundred * (1.0 - 0.005));
    const auto recovered =
        record(setup.value(), world.value(), antilock, 1.0, Actuator::Driver, 0.0, false, hundred * (1.0 - 0.004));

    REQUIRE(stranded.stopped);
    REQUIRE(recovered.stopped);

    std::printf("\n=== two members of one ensemble, 0.1%% of entry speed apart ===\n");
    std::printf("  stranded  entry %.3f km/h  ->  %.3f m, rear off %zu ticks\n", 3.6 * hundred * 0.995,
                stranded.distance, stranded.rearAirborneTicks);
    std::printf("  recovered entry %.3f km/h  ->  %.3f m, rear off %zu ticks\n", 3.6 * hundred * 0.996,
                recovered.distance, recovered.rearAirborneTicks);

    const auto lastAirborne = [](const Run& run)
    {
        auto last = 0.0;
        for (const auto& sample : run.samples)
        {
            if (!sample.wheels[2].inContact || !sample.wheels[3].inContact)
            {
                last = sample.time;
            }
        }

        return last;
    };

    std::printf("\n  last tick with a rear wheel off the road: stranded %.4f s, recovered %.4f s\n",
                lastAirborne(stranded), lastAirborne(recovered));

    std::printf("\n  --- the REAR LEFT channel, every 36th tick ---\n");
    std::printf("\n     t [s]      stranded: Fz    slip  phase     bar    util   |   recovered: Fz    slip"
                "  phase     bar    util\n");

    const auto rows = std::min(stranded.samples.size(), recovered.samples.size());

    for (auto index = std::size_t{0}; index < rows; index += 36)
    {
        const auto& a = stranded.samples[index];
        const auto& b = recovered.samples[index];
        const auto& wa = a.wheels[2];
        const auto& wb = b.wheels[2];

        std::printf("  %8.4f   %14.0f  %6.3f  %-7s %6.1f  %6.3f   |   %14.0f  %6.3f  %-7s %6.1f  %6.3f\n", a.time,
                    wa.load, wa.slipRatio, modulatorName(wa.phase), wa.pressure / bar, wa.utilisation(), wb.load,
                    wb.slipRatio, modulatorName(wb.phase), wb.pressure / bar, wb.utilisation());
    }

    std::printf("\n  --- and the same two through the first 400 ms, every 6th tick ---\n");
    std::printf("\n     t [s]      stranded: Fz    slip  phase     bar    util   |   recovered: Fz    slip"
                "  phase     bar    util\n");

    for (auto index = std::size_t{0}; index < rows && stranded.samples[index].time < 0.40; index += 6)
    {
        const auto& a = stranded.samples[index];
        const auto& b = recovered.samples[index];
        const auto& wa = a.wheels[2];
        const auto& wb = b.wheels[2];

        std::printf("  %8.4f   %14.0f  %6.3f  %-7s %6.1f  %6.3f   |   %14.0f  %6.3f  %-7s %6.1f  %6.3f\n", a.time,
                    wa.load, wa.slipRatio, modulatorName(wa.phase), wa.pressure / bar, wa.utilisation(), wb.load,
                    wb.slipRatio, modulatorName(wb.phase), wb.pressure / bar, wb.utilisation());
    }
}

// ---------------------------------------------------------------------------------------------
// The standard dry braking ensemble, and the ledger built on it. `./EngineTests "[.brake-ledger]"`.
//
// **This is the successor to every single-stop braking figure in this project**, and it exists
// because `[.brake-gap]` above established that a floored anti-lock stop on this car is not a
// measurement: hold the plant byte-identical, move only the entry speed inside the fixture's own
// settling tolerance, and the stop spans four metres in two clusters. Read this section's tables and
// not the single-stop ones above; those are kept because the difference between them is the finding.
//
// **The protocol is a protocol and not a fixture detail.** RaceEngine has to compare a stopping
// distance across vehicles, tyres, brake systems and controllers, so the unit of measurement is an
// ensemble and never a stop. Nothing here is random: the perturbation is a deterministic sweep of
// entry speed, every member reproduces to the bit, and each member is the same stop the ledger
// always measured.
// ---------------------------------------------------------------------------------------------

namespace
{

// The band, stated once: +/-0.7% of the fixture's own 100 km/h. It is the band
// `AntilockBrakingTests.cpp`'s `steeringEnsemble` already uses for the two steering criteria, it is
// inside the fixture's own settling tolerance (`record` asserts the entry speed to 0.5 m/s, which is
// 1.8%), and it is smaller than every plant change `[.brake-gap]` measured.
constexpr auto ensembleBand = 0.007;

// Member `index` of `count`, uniformly spaced across the band. Counts are odd everywhere below so
// the median is a member rather than an average of two; at count 15 this reproduces `ensembleOf`'s
// own -0.007 + 0.001k to the bit, which is what makes the two sections comparable.
[[nodiscard]] double memberEntry(const std::size_t index, const std::size_t count)
{
    const auto offset =
        count <= 1 ? 0.0
                   : -ensembleBand + 2.0 * ensembleBand * static_cast<double>(index) / static_cast<double>(count - 1);

    return hundred * (1.0 + offset);
}

// The last moment a rear wheel was off the road, seconds. The clusters `[.brake-gap]` traced are
// separated by this and by nothing the distance itself shows: 0.21 s recovers, 2.88 s strands.
[[nodiscard]] double lastRearAirborne(const Run& run)
{
    auto last = 0.0;

    for (const auto& sample : run.samples)
    {
        if (!sample.wheels[2].inContact || !sample.wheels[3].inContact)
        {
            last = sample.time;
        }
    }

    return last;
}

// One member's worth of every observable the canonical ledger reports.
struct Member
{
    double entry = 0.0;
    double distance = 0.0;
    double meanG = 0.0;
    double peakG = 0.0;
    double utilisation = 0.0;
    double frontUtilisation = 0.0;
    double rearUtilisation = 0.0;
    double lastRearAirborne = 0.0;
    std::size_t rearAirborne = 0;
    bool stopped = false;
};

[[nodiscard]] std::vector<Member> armEnsemble(const VehicleSetup& setup, const PhysicsWorld& world,
                                              const AssistSetup& assists, const double pedal, const Actuator actuator,
                                              const std::size_t count)
{
    auto members = std::vector<Member>{};
    members.reserve(count);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto entry = memberEntry(index, count);
        const auto run = record(setup, world, assists, pedal, actuator, 0.0, false, entry);
        const auto whole = axleWindow(run, 0.0, 9.0);

        members.push_back(Member{.entry = entry,
                                 .distance = run.distance,
                                 .meanG = run.meanDeceleration(),
                                 .peakG = peakDeceleration(run),
                                 .utilisation = utilisationOf(run),
                                 .frontUtilisation = whole.frontUtilisation,
                                 .rearUtilisation = whole.rearUtilisation,
                                 .lastRearAirborne = lastRearAirborne(run),
                                 .rearAirborne = run.rearAirborneTicks,
                                 .stopped = run.stopped});
    }

    return members;
}

// The statistics the ledger reports. A median alone cannot describe a bimodal arm, which is the
// whole reason this struct has nine fields instead of one.
struct Distribution
{
    std::size_t n = 0;
    double minimum = 0.0;
    double p10 = 0.0;
    double p25 = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double p75 = 0.0;
    double p90 = 0.0;
    double maximum = 0.0;
    // Sample standard deviation, n-1. It is a spread descriptor for a deterministic set rather than
    // an estimate of any population parameter, and it is here to be read against the P10-P90 span:
    // on a two-cluster arm the two disagree, and the disagreement is the warning.
    double deviation = 0.0;
};

// Linear interpolation between order statistics -- R's type 7, numpy's default. Stated because on
// fifteen members a percentile is a definition and not a fact. On an odd count it puts the median
// exactly on the middle member, so every median below agrees to the bit with `[.brake-gap]`'s.
[[nodiscard]] double percentile(const std::vector<double>& sorted, const double fraction)
{
    const auto position = fraction * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);

    return sorted[lower] + (position - static_cast<double>(lower)) * (sorted[upper] - sorted[lower]);
}

[[nodiscard]] Distribution distributionOf(std::vector<double> values)
{
    if (values.empty())
    {
        return Distribution{};
    }

    std::sort(values.begin(), values.end());

    auto sum = 0.0;
    for (const auto value : values)
    {
        sum += value;
    }

    const auto count = static_cast<double>(values.size());
    const auto mean = sum / count;

    auto squared = 0.0;
    for (const auto value : values)
    {
        squared += (value - mean) * (value - mean);
    }

    return Distribution{.n = values.size(),
                        .minimum = values.front(),
                        .p10 = percentile(values, 0.10),
                        .p25 = percentile(values, 0.25),
                        .median = percentile(values, 0.50),
                        .mean = mean,
                        .p75 = percentile(values, 0.75),
                        .p90 = percentile(values, 0.90),
                        .maximum = values.back(),
                        .deviation = values.size() > 1 ? std::sqrt(squared / (count - 1.0)) : 0.0};
}

template <typename Projection>
[[nodiscard]] std::vector<double> project(const std::vector<Member>& members, Projection projection)
{
    auto values = std::vector<double>{};
    values.reserve(members.size());

    for (const auto& member : members)
    {
        values.push_back(projection(member));
    }

    return values;
}

// The separator `[.brake-gap]` found in the data rather than chose: the rear axle either recovers
// past a half of its own capacity over the stop or is left stranded near a third, and nothing
// between 0.42 and 0.57 was ever observed. It is re-checked below on every arm before it is used.
constexpr auto strandedRear = 0.50;

// Whether 0.50 still falls in an empty region of this arm's observed rear utilisation. A separator
// that has stopped separating must not be quoted as a fraction [check-the-control-variable].
struct Separation
{
    bool clean = false;
    double highestStranded = 0.0;
    double lowestRecovered = 0.0;
    std::size_t stranded = 0;
};

[[nodiscard]] Separation separationOf(const std::vector<Member>& members)
{
    auto separation = Separation{.clean = true, .highestStranded = 0.0, .lowestRecovered = 1e9, .stranded = 0};

    for (const auto& member : members)
    {
        if (member.rearUtilisation < strandedRear)
        {
            separation.stranded++;
            separation.highestStranded = std::max(separation.highestStranded, member.rearUtilisation);
        }
        else
        {
            separation.lowestRecovered = std::min(separation.lowestRecovered, member.rearUtilisation);
        }
    }

    if (separation.stranded == 0 || separation.stranded == members.size())
    {
        separation.clean = false;
    }

    return separation;
}

void printDistributionRow(const char* name, const Distribution& value)
{
    std::printf("  %-30s %3zu  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %6.3f\n", name, value.n,
                value.minimum, value.p10, value.p25, value.median, value.mean, value.p75, value.p90, value.maximum,
                value.deviation);
}

void printDistributionHeader(const char* what)
{
    std::printf("\n  %-30s   n      min      P10      P25   MEDIAN     mean      P75      P90      max      sd\n",
                what);
}

// The five brake systems of the canonical ledger, in the order it prints them.
struct LedgerArm
{
    const char* name;
    double pedal;
    Actuator actuator;
    bool antilock;
    // Whether a rear-utilisation cluster fraction means anything for this arm. It is a statement
    // about a closed-loop rear anti-lock channel; an oracle holds each wheel at its own peak by
    // construction and a locked wheel has no channel at all, so on those arms the number would be
    // arithmetic without a referent.
    bool clustered;
};

[[nodiscard]] std::vector<LedgerArm> ledgerArms()
{
    return std::vector<LedgerArm>{{"A pedal 1.00, no electronics", 1.0, Actuator::Driver, false, false},
                                  {"B production ABS", 1.0, Actuator::Driver, true, true},
                                  {"C slew oracle (this valve)", 1.0, Actuator::SlewOracle, true, false},
                                  {"D clamped oracle", 1.0, Actuator::BoundedOracle, true, false},
                                  {"E unbounded oracle", 1.0, Actuator::Oracle, true, false}};
}

// The published reference, and it is a SCALAR. auto motor und sport's Supertest, kalt, verified at
// source 2026-08-25. There is no distribution around it and none is invented below: one number
// measured on one car on one day, and every comparison against it is a distribution against a point.
constexpr auto publishedStop = 35.50;

} // namespace

TEST_CASE("how many members the standard braking ensemble needs, and what they cost", "[.brake-ledger]")
{
    // **Section 2 of the ledger rebuild: choose the instrument before using it.** The same band at
    // six densities, the plant byte-identical in every member, wall clock recorded for each. What is
    // being chosen is not a statistic that is *right* -- a bimodal arm has no such statistic -- but
    // the smallest count whose median, gap median and stranded fraction reproduce well enough to
    // compare two cars with.
    //
    // The clamped oracle is run alongside at every count for a reason: it is the arm that is
    // deterministic and monotone in entry speed, so whatever spread IT shows at a given count is
    // pure `v^2` and is the floor any assisted spread has to be read against.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto plain = golfGtiMk7Assists(setup.value());
    auto antilock = plain;
    antilock.antilock.enabled = true;

    std::printf("\n=== the standard dry braking ensemble, at six densities ===\n");
    std::printf("  band +/-%.1f%% of entry speed, uniform, deterministic. Only the density changes.\n",
                100.0 * ensembleBand);
    std::printf("\n    n   ABS med   ABS mean   ABS sd  ABS P10-P90   clamp med  clamp sd    G med   G mean"
                "  stranded   seconds\n");

    for (const auto count :
         {std::size_t{9}, std::size_t{15}, std::size_t{21}, std::size_t{29}, std::size_t{43}, std::size_t{57}})
    {
        const auto began = std::chrono::steady_clock::now();

        const auto assisted = armEnsemble(setup.value(), world.value(), antilock, 1.0, Actuator::Driver, count);
        const auto clamped = armEnsemble(setup.value(), world.value(), antilock, 1.0, Actuator::BoundedOracle, count);

        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();

        auto gaps = std::vector<double>{};
        gaps.reserve(count);
        for (auto index = std::size_t{0}; index < count; index++)
        {
            REQUIRE(assisted[index].stopped);
            REQUIRE(clamped[index].stopped);
            gaps.push_back(assisted[index].distance - clamped[index].distance);
        }

        const auto stop = distributionOf(project(assisted, [](const Member& m) { return m.distance; }));
        const auto oracle = distributionOf(project(clamped, [](const Member& m) { return m.distance; }));
        const auto gap = distributionOf(gaps);
        const auto separation = separationOf(assisted);

        std::printf("  %3zu  %8.3f  %9.3f  %7.3f  %5.3f-%5.3f    %8.3f  %8.3f  %7.3f  %7.3f    %6.3f  %8.2f\n", count,
                    stop.median, stop.mean, stop.deviation, stop.p10, stop.p90, oracle.median, oracle.deviation,
                    gap.median, gap.mean, static_cast<double>(separation.stranded) / static_cast<double>(count),
                    elapsed);
    }

    std::printf("\n  The clamped oracle's sd is the pure v^2 spread of the band and is the floor. Any\n");
    std::printf("  assisted sd far above it is the controller's loop and not the fixture.\n");
    std::printf("  Runtime is one arm PAIR at that count, on this machine, including the Jolt bring-up.\n");
}

TEST_CASE("the current dry braking ledger, its gaps and its two clusters", "[.brake-ledger]")
{
    // **Sections 4, 5, 6 and 7 of the ledger rebuild, and the one table to quote.** Five brake
    // systems on the same fixture, the same ensemble under every one of them, full distributions
    // rather than medians, and the gaps taken PAIRED -- member `i` of the anti-lock arm against
    // member `i` of the oracle arm, which is the same entry speed and the same plant.
    //
    // A difference of medians is not the median of the differences, and on a bimodal arm they are
    // not close. Both are printed and the paired one is the one that is labelled as the gap.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    const auto plain = golfGtiMk7Assists(setup.value());
    auto antilock = plain;
    antilock.antilock.enabled = true;

    const auto count = std::size_t{29};
    const auto arms = ledgerArms();

    auto ensembles = std::vector<std::vector<Member>>{};
    ensembles.reserve(arms.size());

    for (const auto& arm : arms)
    {
        ensembles.push_back(
            armEnsemble(setup.value(), world.value(), arm.antilock ? antilock : plain, arm.pedal, arm.actuator, count));

        for (const auto& member : ensembles.back())
        {
            REQUIRE(member.stopped);
        }
    }

    std::printf("\n=== the canonical dry braking ledger, %zu members, +/-%.1f%% entry band ===\n", count,
                100.0 * ensembleBand);
    std::printf("  Golf GTI Mk7, flat plate, grip 1.00, step pedal, 360 Hz. Initial condition below.\n");

    printDistributionHeader("stopping distance [m]");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printDistributionRow(arms[index].name,
                             distributionOf(project(ensembles[index], [](const Member& m) { return m.distance; })));
    }

    std::printf("\n  --- the same five arms, the rest of the ledger's columns, as MEDIANS ---\n");
    std::printf("\n  system                          stop med   mean g   peak g   car util   front ut   rear ut"
                "  stranded rear\n");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& members = ensembles[index];
        const auto separation = separationOf(members);
        const auto stop = distributionOf(project(members, [](const Member& m) { return m.distance; }));
        const auto meanG = distributionOf(project(members, [](const Member& m) { return m.meanG; }));
        const auto peakG = distributionOf(project(members, [](const Member& m) { return m.peakG; }));
        const auto util = distributionOf(project(members, [](const Member& m) { return m.utilisation; }));
        const auto front = distributionOf(project(members, [](const Member& m) { return m.frontUtilisation; }));
        const auto rear = distributionOf(project(members, [](const Member& m) { return m.rearUtilisation; }));

        std::printf("  %-30s %8.3f  %7.3f  %7.3f   %8.3f   %8.3f  %8.3f  ", arms[index].name, stop.median, meanG.median,
                    peakG.median, util.median, front.median, rear.median);

        if (arms[index].clustered)
        {
            std::printf("%2zu/%zu = %.3f\n", separation.stranded, count,
                        static_cast<double>(separation.stranded) / static_cast<double>(count));
        }
        else
        {
            std::printf("not meaningful\n");
        }
    }

    std::printf("\n  `stranded rear` is a statement about a closed-loop rear anti-lock channel. An oracle\n");
    std::printf("  holds each wheel at its own peak by construction and a locked wheel has no channel at\n");
    std::printf("  all, so the fraction is not computed on those arms rather than being printed as zero.\n");

    // --- the gap ledger, paired ---

    const auto& floored = ensembles[0];
    const auto& assisted = ensembles[1];
    const auto& slew = ensembles[2];
    const auto& clamped = ensembles[3];
    const auto& unbounded = ensembles[4];

    auto gapSlew = std::vector<double>{};
    auto gapClamped = std::vector<double>{};
    auto hardware = std::vector<double>{};
    auto residual = std::vector<double>{};
    auto worth = std::vector<double>{};

    for (auto index = std::size_t{0}; index < count; index++)
    {
        gapSlew.push_back(assisted[index].distance - slew[index].distance);
        gapClamped.push_back(assisted[index].distance - clamped[index].distance);
        hardware.push_back(clamped[index].distance - unbounded[index].distance);
        residual.push_back(unbounded[index].distance - publishedStop);
        worth.push_back(100.0 * (floored[index].distance - assisted[index].distance) / floored[index].distance);
    }

    std::printf("\n=== the gap ledger, PAIRED member by member ===\n");
    printDistributionHeader("gap [m]");
    printDistributionRow("ABS - slew oracle", distributionOf(gapSlew));
    printDistributionRow("ABS - clamped oracle", distributionOf(gapClamped));
    printDistributionRow("clamped - unbounded", distributionOf(hardware));
    printDistributionRow("unbounded - published 35.50", distributionOf(residual));
    printDistributionRow("ABS worth vs locked [%]", distributionOf(worth));

    const auto stopOf = [&](const std::size_t arm)
    {
        return distributionOf(project(ensembles[arm], [](const Member& m) { return m.distance; }));
    };

    std::printf("\n  --- and the same four as DIFFERENCES OF MEDIANS, which is a different quantity ---\n");
    std::printf("  ABS - slew            paired median %7.3f    difference of medians %7.3f\n",
                distributionOf(gapSlew).median, stopOf(1).median - stopOf(2).median);
    std::printf("  ABS - clamped         paired median %7.3f    difference of medians %7.3f\n",
                distributionOf(gapClamped).median, stopOf(1).median - stopOf(3).median);
    std::printf("  clamped - unbounded   paired median %7.3f    difference of medians %7.3f\n",
                distributionOf(hardware).median, stopOf(3).median - stopOf(4).median);
    std::printf("  unbounded - published paired median %7.3f    difference of medians %7.3f\n",
                distributionOf(residual).median, stopOf(4).median - publishedStop);
    std::printf("\n  The published 35.50 m is a SCALAR -- one measured car on one day. The spread in the\n");
    std::printf("  last row is entirely the model's, and no distribution is invented around the reference.\n");

    // --- the two clusters ---

    const auto separation = separationOf(assisted);

    std::printf("\n=== the production ABS arm, separated into its two clusters ===\n");

    auto recoveredStop = std::vector<double>{};
    auto strandedStop = std::vector<double>{};
    auto recoveredRear = std::vector<double>{};
    auto strandedRearUtil = std::vector<double>{};
    auto recoveredTicks = std::vector<double>{};
    auto strandedTicks = std::vector<double>{};
    auto recoveredLast = std::vector<double>{};
    auto strandedLast = std::vector<double>{};

    for (const auto& member : assisted)
    {
        const auto isStranded = member.rearUtilisation < strandedRear;

        (isStranded ? strandedStop : recoveredStop).push_back(member.distance);
        (isStranded ? strandedRearUtil : recoveredRear).push_back(member.rearUtilisation);
        (isStranded ? strandedTicks : recoveredTicks).push_back(static_cast<double>(member.rearAirborne));
        (isStranded ? strandedLast : recoveredLast).push_back(member.lastRearAirborne);
    }

    const auto cluster = [&](const char* name, const std::vector<double>& stops, const std::vector<double>& rear,
                             const std::vector<double>& ticks, const std::vector<double>& last)
    {
        if (stops.empty())
        {
            std::printf("  %-12s  EMPTY -- every member of this ensemble is in the other cluster\n", name);
            return;
        }

        const auto s = distributionOf(stops);
        const auto r = distributionOf(rear);
        const auto t = distributionOf(ticks);
        const auto l = distributionOf(last);

        std::printf("  %-12s %3zu  %6.3f   %7.3f %7.3f %7.3f  %5.3f %5.3f %5.3f  %5.0f %5.0f %5.0f"
                    "  %6.3f %6.3f %6.3f\n",
                    name, s.n, static_cast<double>(s.n) / static_cast<double>(count), s.minimum, s.median, s.maximum,
                    r.minimum, r.median, r.maximum, t.minimum, t.median, t.maximum, l.minimum, l.median, l.maximum);
    };

    std::printf("\n  cluster        n    frac    stop: min  median     max   rear util: lo   md   hi"
                "   rear-off ticks     last airborne [s]\n");
    cluster("RECOVERED", recoveredStop, recoveredRear, recoveredTicks, recoveredLast);
    cluster("STRANDED", strandedStop, strandedRearUtil, strandedTicks, strandedLast);

    std::printf("\n  separator      rear utilisation %.2f\n", strandedRear);
    if (separation.clean)
    {
        std::printf("  observed gap   highest stranded %.3f, lowest recovered %.3f -- the separator sits in an\n",
                    separation.highestStranded, separation.lowestRecovered);
        std::printf("                 EMPTY region of the observed data, so it is a gap and not a chosen number.\n");
    }
    else
    {
        std::printf("  observed gap   every member is on ONE side of the separator at this density; the\n");
        std::printf("                 separator cannot be validated from this arm alone.\n");
    }

    std::printf("\n  --- the initial condition, stated so nothing is compared across it ---\n");
    std::printf("  tyre gas       %.1f C default-constructed (tyreDefaultTemperature), NOT the ideal-pressure\n",
                raceengine::tyreDefaultTemperature);
    std::printf("                 seed of about 61.2 C, so the fixture starts near 34.55 psi against an ideal\n");
    std::printf("                 of 34.00. This is SHIPPED behaviour -- the game does the same -- and it is\n");
    std::printf("                 characterised here rather than fixed. Do not mix seeded and unseeded runs.\n");
    std::printf("  electronics    ABS as stated per arm; traction control off, cornering brake off, yaw delay off\n");
    std::printf("  published ref  %.2f m, a SCALAR (amS Supertest, kalt, verified at source 2026-08-25)\n",
                publishedStop);
}
