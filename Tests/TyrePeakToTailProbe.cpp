// The tyre's peak-to-tail ratio, measured. `./EngineTests "[.peak-to-tail]"`.
//
// Steps 1 and 2 of `docs/tyre-peak-to-tail-brief.md`, and like the utilisation instrument this is
// **a probe and not a gate**: no acceptance threshold on any finding, every `REQUIRE` a fixture
// precondition. The question it exists to answer: can the shape of the longitudinal curve past its
// peak — `longitudinalShape`, `longitudinalCurvature`, `longitudinalStiffness`, the only three
// numbers in this tyre marked *fitted rather than sourced* — reconcile the two published references
// once the peak's height and position are held?
//
// **The two jaws, and why these two measurements.**
//
//   - Braking: the **clamped oracle's** 100-0, never the car's own. The car's 41.74 m owes a
//     measured ~3.5 m to the controller and the valve (`docs/braking-chain-brief.md`), so fitting a
//     tyre to make the car hit the published 34.6-35.1 m would fit it to the modulator's hunting.
//     The oracle holds each wheel at its own peak through the brakes the car has: 36.92 m shipped,
//     and that 5-6% over the published band is the tyre's (and chassis's) own number.
//   - Acceleration: **0-100 with traction control in sport**, against the measured 6.5-6.6 s — the
//     validated electronics-on control, and the launch is what rides the tail (39-59% of the run is
//     spent past the peak).
//
// **The trap the sweep is built to avoid**: C, E and K trade against each other. `B = K/(C·mu)`,
// so changing C or E alone moves the peak's position along the slip axis and the sweep would be
// changing two things. Each candidate therefore has K solved to hold the *measured* peak at the
// shipped position — and the peak's height is `mu·Fz` for any C > 1, verified by scan rather than
// trusted. `peakSlipOf` ignores E entirely (`TyreShapeProbe` measured it 25% adrift at E = -1), so
// the solve is done on the full formula and the oracle is handed a per-candidate correction — the
// true-peak-to-reported ratio is load-independent, since both scale with `mu(Fz)` — so that it
// measures each candidate's tyre rather than the E-blindness of one helper.
//
// **The locked-wheel stop is the direct tail read.** A locked wheel sits at slip 1.0, so the ratio
// of the oracle's stop to the locked stop is the tyre's own sliding-to-peak ratio through the real
// chassis — the one part of the fitted tail that published evidence exists for (printed with the
// shipped curve in the first case).

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
using raceengine::cornerCount;
using raceengine::DrivelineState;
using raceengine::evaluateTyre;
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
using raceengine::TyreAxis;
using raceengine::tyreFriction;
using raceengine::TyreModel;
using raceengine::TyreSlip;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto rideHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto gravity = 9.80665;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto startZ = 20.0;
constexpr auto pi = 3.14159265358979323846;

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

// The braking fixture's plate, exactly as `BrakingUtilisationProbe` builds it, so the shipped
// candidate reproduces that instrument's 36.92 m and the two files cross-check each other.
[[nodiscard]] raceengine::SurfaceMesh gripPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 600.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(1);
    mesh->materials[0].gripMultiplier = 1.0;
    mesh->materials[0].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        mesh->surfaces[triangle] = std::uint32_t{0};
    }

    return mesh.value();
}

[[nodiscard]] ProvingGroundDescriptor straightGround(const double length)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = 60.0;
    descriptor.cellSize = 4.0;
    descriptor.features = std::vector<Feature>{};

    return descriptor;
}

[[nodiscard]] double spawnHeight(const VehicleSetup& setup)
{
    auto highest = 0.0;
    for (const auto& corner : setup.corners)
    {
        highest = std::max(highest, corner.hardpoints.wheelCentre.y + corner.hardpoints.wheelRadius);
    }

    return highest;
}

// --- the curve's own arithmetic ---

// Where the full Magic Formula peaks, E included. The peak is where `C·atan(shaped) = pi/2`, i.e.
// `shaped = tan(pi/2C)`, and `shaped(x) = x - E(x - atan x)` is monotone for E <= 1, so a bisection
// finds the inner argument `x = B·slip` exactly. This is the solve `peakSlipOf` does not do.
[[nodiscard]] double peakInnerArgument(const double shape, const double curvature)
{
    const auto target = std::tan(pi / (2.0 * shape));
    const auto shaped = [curvature](const double x)
    {
        return x - curvature * (x - std::atan(x));
    };

    auto low = 0.0;
    auto high = 1.0;
    while (shaped(high) < target)
    {
        high *= 2.0;
        REQUIRE(high < 1.0e9);
    }

    for (auto iteration = 0; iteration < 200; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        (shaped(middle) < target ? low : high) = middle;
    }

    return 0.5 * (low + high);
}

// The slip stiffness that puts a candidate's true peak at `targetSlip`. `B = K / (C·mu·scale)` and
// the peak sits at `slip = x* / B`, so K falls straight out — and because B carries `mu(Fz)`, the
// peak's position is the same at every load once it is held at one.
[[nodiscard]] double stiffnessHoldingPeak(const TyreModel& model, const double shape, const double curvature,
                                          const double targetSlip)
{
    const auto friction = tyreFriction(model, TyreAxis::Longitudinal, model.nominalLoad, 1.0);
    const auto inner = peakInnerArgument(shape, curvature);

    return (inner / targetSlip) * shape * friction * std::max(model.peakSlipScale, 1e-6);
}

// What the oracle must multiply the tyre's reported peak slip by to stand on the *true* peak:
// `peakSlipOf` reports `tan(pi/2C)/B` where the truth is `x*/B`, and the ratio is a constant of the
// candidate. 1.0 exactly at E = 0.
[[nodiscard]] double oracleCorrectionOf(const double shape, const double curvature)
{
    return peakInnerArgument(shape, curvature) / std::tan(pi / (2.0 * shape));
}

// What the curve actually does at one load, by scanning it. Fractions are of `mu(Fz)·Fz`, which the
// utilisation instrument verified the shipped curve reaches exactly — re-verified per candidate
// here, because a candidate whose amplitude is unreachable has silently lost peak height and every
// comparison against it is void.
struct TailReading
{
    double peakSlip = 0.0;
    double peakFraction = 0.0;
    double atOneAndHalf = 0.0;
    double atTwice = 0.0;
    double atThrice = 0.0;
    double atSlipOne = 0.0;
    double atSlipThree = 0.0;
    double reportedPeakSlip = 0.0;
};

[[nodiscard]] TailReading readTail(const TyreModel& model, const double load)
{
    const auto available = tyreFriction(model, TyreAxis::Longitudinal, load, 1.0) * load;
    const auto fractionAt = [&](const double slip)
    {
        return evaluateTyre(model, load, TyreSlip{.slipRatio = slip, .slipAngle = 0.0}, 1.0).longitudinal / available;
    };

    auto reading = TailReading{};

    for (auto step = 1; step <= 4000; step++)
    {
        const auto slip = 0.0001 * static_cast<double>(step);
        const auto fraction = fractionAt(slip);

        if (fraction > reading.peakFraction)
        {
            reading.peakFraction = fraction;
            reading.peakSlip = slip;
        }
    }

    reading.atOneAndHalf = fractionAt(1.5 * reading.peakSlip);
    reading.atTwice = fractionAt(2.0 * reading.peakSlip);
    reading.atThrice = fractionAt(3.0 * reading.peakSlip);
    reading.atSlipOne = fractionAt(1.0);
    reading.atSlipThree = fractionAt(3.0);
    reading.reportedPeakSlip =
        evaluateTyre(model, load, TyreSlip{.slipRatio = 0.05, .slipAngle = 0.0}, 1.0).longitudinalPeakSlip;

    return reading;
}

// --- the stopping harness, from `BrakingUtilisationProbe` ---
//
// Trimmed to the two actuators this sweep needs: the driver (a pedal through the hydraulics, with
// whatever the assist layer is set to) and the clamped oracle (per-wheel torque holding each tyre
// at its own peak, bounded by the brakes the car has). `oracleCorrection` is the E-blindness fix
// above; it is 1.0 for every E = 0 candidate.

enum class Actuator : std::uint8_t
{
    Driver,
    BoundedOracle
};

struct WheelSample
{
    double load = 0.0;
    double forceLongitudinal = 0.0;
    double friction = 0.0;
    double slipRatio = 0.0;
    double peakSlip = 0.0;
    bool inContact = false;

    [[nodiscard]] double slipAgainstPeak() const
    {
        return peakSlip > 1e-9 ? std::abs(slipRatio) / peakSlip : 0.0;
    }

    [[nodiscard]] double capacity() const
    {
        return friction * load;
    }
};

struct Sample
{
    double time = 0.0;
    double speed = 0.0;
    std::array<WheelSample, cornerCount> wheels{};
};

struct Run
{
    std::vector<Sample> samples;
    double distance = 0.0;
    double time = 0.0;
    double entrySpeed = 0.0;
    bool stopped = false;

    [[nodiscard]] double meanDeceleration() const
    {
        return time > 0.0 ? entrySpeed / time / gravity : 0.0;
    }
};

constexpr auto oracleTrim = 0.5;

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, rideHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, hundred);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = hundred / tyreRadius;
    }
}

[[nodiscard]] Run record(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                         const double pedal, const Actuator actuator, const double oracleCorrection)
{
    auto state = VehicleState{};
    settle(setup, state, world);

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

    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
    }
    REQUIRE(std::abs(state.chassis.linearVelocity.z - hundred) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - rideHeight) < 0.1);

    auto run = Run{};
    run.entrySpeed = state.chassis.linearVelocity.z;
    run.samples.reserve(360 * 8);

    const auto start = state.chassis.position;

    auto input = VehicleInput{};
    input.brake = pedal;

    auto oracle = BrakeCommand{};
    oracle.commanded = true;

    for (auto step = 0; step < 360 * 30; step++)
    {
        auto brakes = BrakeCommand{};

        if (actuator == Actuator::BoundedOracle)
        {
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& solution = lastStep.corners[index];
                const auto& corner = setup.corners[index];

                const auto peakSlip = std::max(solution.contact.tyre.longitudinalPeakSlip * oracleCorrection, 1e-3);
                const auto slip = std::abs(solution.contact.slip.slipRatio);
                const auto radius = std::max(solution.contact.effectiveRadius, 1e-3);

                const auto feedforward = std::abs(solution.contact.tyre.longitudinal) * radius;
                const auto error = std::clamp((peakSlip - slip) / peakSlip, -1.0, 1.0);
                const auto trim = oracleTrim * solution.forces.tireVertical * radius * error;

                oracle.wheels[index] = std::max(0.0, feedforward + trim);
                oracle.wheels[index] = std::max(0.0, oracle.wheels[index] - corner.rollingResistance *
                                                                                solution.forces.tireVertical * radius);
                oracle.wheels[index] = std::min(oracle.wheels[index], corner.brakeTorque);
            }

            brakes = oracle;
        }
        else
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = pedal, .throttle = 0.0},
                                               brakeCircuitPressures(setup, pedal), tick);
            brakes = command.brakes;
        }

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        run.time += tick;

        auto sample = Sample{};
        sample.time = run.time;
        sample.speed = state.chassis.linearVelocity.z;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];
            auto& wheel = sample.wheels[index];

            wheel.load = solution.forces.tireVertical;
            wheel.forceLongitudinal = solution.contact.tyre.longitudinal;
            wheel.friction = tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, wheel.load,
                                          solution.patch.gripMultiplier);
            wheel.slipRatio = solution.contact.slip.slipRatio;
            wheel.peakSlip = solution.contact.tyre.longitudinalPeakSlip * oracleCorrection;
            wheel.inContact = lastStep.telemetry.wheels[index].inContact;
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

// --- the stop's phase decomposition, from the utilisation instrument ---

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

// Per-axle means over one phase of one run: slip against the tyre's own peak, and utilisation.
struct AxlePhase
{
    double slipOverPeak = 0.0;
    double force = 0.0;
    double capacity = 0.0;
    std::size_t ticks = 0;

    [[nodiscard]] double utilisation() const
    {
        return capacity > 1.0 ? force / capacity : 0.0;
    }

    [[nodiscard]] double meanSlipOverPeak() const
    {
        return ticks > 0 ? slipOverPeak / static_cast<double>(ticks) : 0.0;
    }
};

using PhaseTable = std::array<std::array<AxlePhase, 2>, phaseCount>;

[[nodiscard]] PhaseTable phaseTableOf(const Run& run)
{
    auto table = PhaseTable{};

    for (const auto& sample : run.samples)
    {
        const auto phase = static_cast<std::size_t>(phaseOf(sample));

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            auto& axle = table[phase][index / 2];
            const auto& wheel = sample.wheels[index];

            axle.slipOverPeak += wheel.slipAgainstPeak();
            axle.force += std::abs(wheel.forceLongitudinal);
            axle.capacity += wheel.capacity();
            axle.ticks++;
        }
    }

    return table;
}

// Whole-car utilisation over the moving window, the figure the braking-chain ledger is kept in.
[[nodiscard]] double movingUtilisation(const Run& run)
{
    auto force = 0.0;
    auto capacity = 0.0;

    for (const auto& sample : run.samples)
    {
        if (sample.speed < 5.0)
        {
            continue;
        }

        for (const auto& wheel : sample.wheels)
        {
            force += std::abs(wheel.forceLongitudinal);
            capacity += wheel.capacity();
        }
    }

    return capacity > 1.0 ? force / capacity : 0.0;
}

// How much of the moving window the front axle spends effectively locked. The direct-tail claim
// rests on the locked run actually locking, so it is printed rather than assumed.
[[nodiscard]] double lockedFraction(const Run& run)
{
    auto locked = std::size_t{0};
    auto ticks = std::size_t{0};

    for (const auto& sample : run.samples)
    {
        if (sample.speed < 5.0)
        {
            continue;
        }

        ticks++;
        if (std::abs(sample.wheels[0].slipRatio) > 0.9 && std::abs(sample.wheels[1].slipRatio) > 0.9)
        {
            locked++;
        }
    }

    return ticks > 0 ? static_cast<double>(locked) / static_cast<double>(ticks) : 0.0;
}

// --- the launch harness, from `TyreGripRatioSweepProbe` ---
//
// A standing start at full throttle with traction control in sport, shifted on road speed through
// the gear. Returns the two numbers the sweep reads: 0-100, and 80-120 as an interval of the same
// run — the control that must not move.

struct Launch
{
    double toHundred = -1.0;
    double toEighty = -1.0;
    double eightyToOneTwenty = -1.0;
    double peakDrivenSlip = 0.0;
};

[[nodiscard]] Launch accelerate(const VehicleSetup& setup, const PhysicsWorld& world)
{
    auto driveline = golfGtiMk7Driveline();
    auto assists = golfGtiMk7Assists(setup);
    assists.traction.mode = TractionMode::Sport;
    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    const auto inertias = wheelInertias(setup);

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, spawnHeight(setup), startZ);

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
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto held = 0; held < 360; held++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup, 1.0), tick);
            const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, idling, torques->wheel, world, tick, command.brakes);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());
            lastStep = stepped.value();
        }
    }

    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.2);
    REQUIRE(drivelineState.gear == 1);
    REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
    REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);

    auto result = Launch{};
    auto gear = 1;
    const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

    for (auto step = 1; step <= 360 * 30; step++)
    {
        const auto now = static_cast<double>(step) * tick;

        const auto roadSideSpeed =
            std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
        if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
        {
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

        for (const auto wheel : {std::size_t{0}, std::size_t{1}})
        {
            result.peakDrivenSlip =
                std::max(result.peakDrivenSlip, std::abs(stepped->corners[wheel].contact.slip.slipRatio));
        }

        const auto speed = std::abs(state.chassis.linearVelocity.z) * 3.6;

        if (speed >= 80.0 && result.toEighty < 0.0)
        {
            result.toEighty = now;
        }
        if (speed >= 100.0 && result.toHundred < 0.0)
        {
            result.toHundred = now;
        }
        if (speed >= 120.0 && result.eightyToOneTwenty < 0.0 && result.toEighty > 0.0)
        {
            result.eightyToOneTwenty = now - result.toEighty;
        }

        if (result.toHundred > 0.0 && result.eightyToOneTwenty > 0.0)
        {
            break;
        }
    }

    REQUIRE(result.toHundred > 0.0);
    REQUIRE(result.eightyToOneTwenty > 0.0);

    return result;
}

// --- the candidates ---

struct Candidate
{
    const char* name;
    double shape;
    double curvature;
};

// One axis at a time around the shipped curve, then two combined corners. C sets the sliding
// asymptote outright (`sin(C·pi/2)`); E sets how fast the curve falls toward it, so at slip 1.0 —
// where the locked-wheel evidence lives — the two together span tails from about 0.46 to 0.91.
constexpr auto candidates = std::array<Candidate, 12>{{{"shipped", 1.50, 0.0},
                                                       {"C 1.30", 1.30, 0.0},
                                                       {"C 1.40", 1.40, 0.0},
                                                       {"C 1.60", 1.60, 0.0},
                                                       {"C 1.70", 1.70, 0.0},
                                                       {"C 1.80", 1.80, 0.0},
                                                       {"E -1.0", 1.50, -1.0},
                                                       {"E -0.5", 1.50, -0.5},
                                                       {"E +0.5", 1.50, 0.5},
                                                       {"E +0.9", 1.50, 0.9},
                                                       {"C 1.65 E +0.5", 1.65, 0.5},
                                                       {"C 1.35 E -1.0", 1.35, -1.0}}};

// A candidate car: shape and curvature stated, stiffness solved to hold the peak, nothing else
// touched. The solve is verified by scan before the car is driven anywhere.
[[nodiscard]] VehicleSetup withTail(VehicleSetup setup, const Candidate& candidate, const double targetSlip)
{
    const auto stiffness =
        stiffnessHoldingPeak(setup.corners.front().tyre, candidate.shape, candidate.curvature, targetSlip);

    for (auto& corner : setup.corners)
    {
        corner.tyre.longitudinalShape = candidate.shape;
        corner.tyre.longitudinalCurvature = candidate.curvature;
        corner.tyre.longitudinalStiffness = stiffness;
    }

    const auto held = readTail(setup.corners.front().tyre, setup.corners.front().tyre.nominalLoad);

    // The two preconditions the whole sweep stands on: the peak's position is where the shipped
    // one's is, and its height is still the full amplitude. A candidate failing either is changing
    // two things, and nothing measured on it would be about the tail.
    REQUIRE(std::abs(held.peakSlip - targetSlip) < 0.003);
    REQUIRE(held.peakFraction > 0.999);

    return setup;
}

} // namespace

TEST_CASE("the shipped tail, printed beside the published sliding-to-peak evidence", "[.peak-to-tail]")
{
    // Step 1 of the brief: the fitted tail has never been compared to anything, so print what it
    // actually is before proposing anything. No thresholds anywhere in this case.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto& tyre = setup->corners.front().tyre;

    std::printf("\n=== the shipped longitudinal curve past its peak ===\n");
    std::printf("  C = %.2f, E = %.2f, K = %.1f, mu_x = %.3f at %.0f N — the three shape numbers are\n",
                tyre.longitudinalShape, tyre.longitudinalCurvature, tyre.longitudinalStiffness, tyre.longitudinalPeak,
                tyre.nominalLoad);
    std::printf("  the only ones in this tyre fitted rather than sourced (PublishedCarsImpl.cpp).\n");
    std::printf("  asymptote sin(C·90°) = %.3f of peak, whatever B and E are\n",
                std::sin(tyre.longitudinalShape * 0.5 * pi));

    std::printf("\n  fractions of mu(Fz)·Fz at each load; 'peak' is the scanned maximum\n");
    std::printf("\n  %-22s %8s %9s %9s %9s %9s %9s %9s %9s\n", "load", "mu_x(Fz)", "peak slip", "peak", "1.5x peak",
                "2x peak", "3x peak", "slip 1.0", "slip 3.0");

    struct Station
    {
        const char* name;
        double load;
    };

    for (const auto& station : std::array<Station, 4>{{{"braking, front (5915 N)", 5915.0},
                                                       {"launch, front (3600 N)", 3600.0},
                                                       {"nominal (2939 N)", 2939.0},
                                                       {"braking, rear (1300 N)", 1300.0}}})
    {
        const auto reading = readTail(tyre, station.load);

        std::printf("  %-22s %8.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n", station.name,
                    tyreFriction(tyre, TyreAxis::Longitudinal, station.load, 1.0), reading.peakSlip,
                    reading.peakFraction, reading.atOneAndHalf, reading.atTwice, reading.atThrice, reading.atSlipOne,
                    reading.atSlipThree);
    }

    std::printf("\n  `peakSlipOf` reports the peak at %.4f against the measured value above — exact at\n",
                readTail(tyre, tyre.nominalLoad).reportedPeakSlip);
    std::printf("  E = 0, and the sweep corrects the oracle for candidates where it is not.\n");

    std::printf("\n--- the published sliding-to-peak evidence this tail has never been read against ---\n");
    std::printf("\n  What a locked wheel reads is the curve at slip 1.0, so the external evidence is\n");
    std::printf("  locked-wheel-versus-peak braking on dry surfaces:\n");
    std::printf("\n  - NHTSA light-vehicle ABS research: locked-wheel stops on dry pavement run about\n");
    std::printf("    15-20%% longer than ABS stops — a sliding-to-peak ratio of ~0.83-0.87.\n");
    std::printf("  - MDPI Sustainability 15/8/6945 (2023), dry asphalt, 60 km/h: mean mu 0.83 with ABS\n");
    std::printf("    (0.768-0.894) against 0.73 locked (0.681-0.780) — a mean ratio of ~0.88.\n");
    std::printf("  - The vehicle-dynamics texts (Gillespie ch. 10, Wong): sliding friction is typically\n");
    std::printf("    70-85%% of peak on dry surfaces, the band already recorded when C was fitted.\n");
    std::printf("\n  Together: slip-1.0 force at ~0.75-0.90 of peak on dry tarmac. The shipped curve reads\n");
    std::printf("  %.3f at braking load — inside the band, toward its steeper edge.\n",
                readTail(tyre, 5915.0).atSlipOne);
    std::printf("\n  The car's own pair says the same thing from the chassis: clamped-oracle 36.92 m\n");
    std::printf("  against locked 45.95 m is a distance ratio of 0.803. The sweep below re-measures\n");
    std::printf("  both per candidate.\n");
}

TEST_CASE("the sweep: shape and curvature with the peak held, both jaws measured", "[.peak-to-tail]")
{
    // Step 2. Per candidate: the clamped oracle's 100-0 (the capability jaw — never the car's own,
    // which owes ~3.5 m to the controller and valve), the locked-wheel 100-0 (the direct tail
    // read), the ABS-on floored stop (reported so the anti-lock interactions are visible, never a
    // target), 0-100 with traction control in sport (the other jaw), and 80-120 from the same run
    // (the control that must not move).
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto brakeWorld = PhysicsWorld::create(gripPlate());
    REQUIRE(brakeWorld.has_value());

    const auto accelWorld = PhysicsWorld::create(generateProvingGround(straightGround(6000.0)).value());
    REQUIRE(accelWorld.has_value());

    // The shipped peak's measured position is the anchor every candidate holds.
    const auto anchor = readTail(setup->corners.front().tyre, setup->corners.front().tyre.nominalLoad);

    std::printf("\n================ the peak-to-tail sweep ================\n");
    std::printf("mu_x %.3f, both load-sensitivity exponents, and the measured peak slip %.4f are held\n",
                setup->corners.front().tyre.longitudinalPeak, anchor.peakSlip);
    std::printf("at every point; only the curve past the peak changes. Jaws: clamped oracle against\n");
    std::printf("100-0 in 34.6-35.1 m, and 0-100 with TC sport against the measured 6.5-6.6 s.\n");

    std::printf("\n%-15s %5s %6s %6s | %9s | %8s %8s %7s | %8s | %8s %8s %7s\n", "", "C", "E", "K", "tail@1.0",
                "oracle m", "locked m", "lk/orc", "ABS m", "0-100 s", "80-120", "verdict");

    struct Outcome
    {
        Run oracle;
        Run locked;
        Run assisted;
        Launch launch;
    };

    auto outcomes = std::vector<Outcome>{};
    auto slowestControl = 0.0;
    auto quickestControl = 1e9;
    auto slowestPlausible = 0.0;
    auto quickestPlausible = 1e9;

    for (const auto& candidate : candidates)
    {
        const auto car = withTail(setup.value(), candidate, anchor.peakSlip);
        const auto& tyre = car.corners.front().tyre;
        const auto correction = oracleCorrectionOf(candidate.shape, candidate.curvature);
        const auto tail = readTail(tyre, 5915.0);

        const auto plain = golfGtiMk7Assists(car);
        auto antilock = plain;
        antilock.antilock.enabled = true;

        auto outcome = Outcome{};
        outcome.oracle = record(car, brakeWorld.value(), plain, 1.0, Actuator::BoundedOracle, correction);
        outcome.locked = record(car, brakeWorld.value(), plain, 1.0, Actuator::Driver, correction);
        outcome.assisted = record(car, brakeWorld.value(), antilock, 1.0, Actuator::Driver, correction);
        outcome.launch = accelerate(car, accelWorld.value());

        REQUIRE(outcome.oracle.stopped);
        REQUIRE(outcome.locked.stopped);
        REQUIRE(outcome.assisted.stopped);

        slowestControl = std::max(slowestControl, outcome.launch.eightyToOneTwenty);
        quickestControl = std::min(quickestControl, outcome.launch.eightyToOneTwenty);

        // The compensation itself moves slip stiffness, and slip stiffness has a real (small)
        // high-speed cost: less K means more slip for the same force, and slip dissipates. So the
        // control is asserted over the candidates a road tyre could be — K inside the recorded
        // 10-to-low-forties range — and the whole-sweep span is printed with that mechanism named,
        // rather than the assertion quietly excluding what it was inconvenient to include.
        if (tyre.longitudinalStiffness <= 45.0)
        {
            slowestPlausible = std::max(slowestPlausible, outcome.launch.eightyToOneTwenty);
            quickestPlausible = std::min(quickestPlausible, outcome.launch.eightyToOneTwenty);
        }

        const auto brakeOk = outcome.oracle.distance >= 34.4 && outcome.oracle.distance <= 35.3;
        const auto accelOk = outcome.launch.toHundred >= 6.5 && outcome.launch.toHundred <= 6.6;

        std::printf("%-15s %5.2f %6.2f %6.1f | %9.3f | %8.2f %8.2f %7.3f | %8.2f | %8.3f %8.3f %7s\n", candidate.name,
                    candidate.shape, candidate.curvature, tyre.longitudinalStiffness, tail.atSlipOne,
                    outcome.oracle.distance, outcome.locked.distance, outcome.oracle.distance / outcome.locked.distance,
                    outcome.assisted.distance, outcome.launch.toHundred, outcome.launch.eightyToOneTwenty,
                    brakeOk && accelOk ? "BOTH" : (brakeOk ? "brake" : (accelOk ? "accel" : "-")));

        outcomes.push_back(std::move(outcome));
    }

    std::printf("\n  K outside 10-30 is not a road tyre (race constructions reach the low forties), so a\n");
    std::printf("  row that only works with one is a finding about the model rather than a candidate.\n");
    std::printf("  lk/orc is the direct tail read through the chassis; published dry-surface evidence\n");
    std::printf("  puts it at ~0.75-0.90 (the first case has the sources).\n");
    std::printf("\n  80-120 control: %.2f%% across road-plausible K, %.2f%% across the whole sweep —\n",
                100.0 * (slowestPlausible / quickestPlausible - 1.0), 100.0 * (slowestControl / quickestControl - 1.0));
    std::printf("  the residual movement tracks K, not the tail, and is the slip-dissipation cost of\n");
    std::printf("  the operating slip the compensation moves (less stiffness, more slip at one force).\n");

    // The anti-lock interactions the brief says to expect: the recovery law and the criterion-2
    // band were calibrated against the shipped tail's shallowness, so where each candidate leaves
    // each axle relative to its own peak is printed per phase — the channel the brief's rejected
    // aggressive-dump experiments were judged on.
    std::printf("\n--- the ABS-on stop's phase tables: mean slip/peak and utilisation, per axle ---\n");
    std::printf("\n%-15s %-17s %9s %9s %9s %9s %9s\n", "", "phase", "F s/pk", "R s/pk", "F util", "R util", "car util");

    for (auto index = std::size_t{0}; index < candidates.size(); index++)
    {
        const auto table = phaseTableOf(outcomes[index].assisted);

        for (auto phase = std::size_t{0}; phase < phaseCount; phase++)
        {
            const auto& front = table[phase][0];
            const auto& rear = table[phase][1];
            const auto car = front.capacity + rear.capacity > 1.0
                                 ? (front.force + rear.force) / (front.capacity + rear.capacity)
                                 : 0.0;

            std::printf("%-15s %-17s %9.2f %9.2f %9.3f %9.3f %9.3f\n", phase == 0 ? candidates[index].name : "",
                        phaseName(static_cast<Phase>(phase)), front.meanSlipOverPeak(), rear.meanSlipOverPeak(),
                        front.utilisation(), rear.utilisation(), car);
        }
    }

    std::printf("\n--- the instrument's own health, per candidate ---\n");
    std::printf("\n%-15s %12s %12s %13s\n", "", "oracle util", "ABS util", "locked front");

    for (auto index = std::size_t{0}; index < candidates.size(); index++)
    {
        std::printf("%-15s %12.3f %12.3f %12.0f%%\n", candidates[index].name, movingUtilisation(outcomes[index].oracle),
                    movingUtilisation(outcomes[index].assisted), 100.0 * lockedFraction(outcomes[index].locked));
    }

    std::printf("\n  oracle util near 1.0 says the capability jaw really was measured at the peak for\n");
    std::printf("  that candidate; locked front near 100%% says the tail read really was slip 1.0.\n");

    // The control, asserted rather than admired, exactly as the grip-ratio sweep does — over the
    // candidates a road tyre could be. If the high-speed interval moves appreciably with the tail
    // itself, the sweep is changing something else and none of the rows above is interpretable.
    REQUIRE(slowestPlausible / quickestPlausible < 1.02);
}
