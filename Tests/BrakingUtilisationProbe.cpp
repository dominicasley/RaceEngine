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
[[nodiscard]] Run record(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                         const double pedal, const Actuator actuator, const double rampSeconds = 0.0)
{
    auto state = VehicleState{};
    settle(setup, state, world, hundred);

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
    REQUIRE(std::abs(state.chassis.linearVelocity.z - hundred) < 0.5);
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

            run.grounded = run.grounded && wheel.inContact;
            run.peakCommand[index] = std::max(run.peakCommand[index], wheel.brakeTorque);
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
