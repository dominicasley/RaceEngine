#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::advanceAntilockChannel;
using raceengine::advanceReferenceSpeed;
using raceengine::advanceYawMomentDelay;
using raceengine::AntilockChannelInputs;
using raceengine::antilockControlWheel;
using raceengine::antilockDrivesWheel;
using raceengine::AntilockSetup;
using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::BrakeChannel;
using raceengine::brakeChannelCount;
using raceengine::brakeCircuitPressures;
using raceengine::BrakeCommand;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::estimatedSlip;
using raceengine::Feature;
using raceengine::frictionAtTemperature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::ModulatorPhase;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::RecoveryAuthority;
using raceengine::sampleWheelSensors;
using raceengine::sensedRoadSpeed;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::TyreAxis;
using raceengine::tyreFriction;
using raceengine::tyrePressureRollingResistanceScale;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::WheelSpeedReadings;

// =============================================================================================
// A REALISTIC ROAD-TORQUE ACCEPTANCE OBSERVABLE FOR ANTI-LOCK RECOVERY.
// `./EngineTests "[.road-torque]"` — hidden, registers no ctest entry, asserts nothing about the
// car, and changes no production file.
//
// **What this file is for.** `docs/braking-chain-brief.md`'s 2026-09-06 entry closed the rear-axle
// hop causally: the sustained failure is a REFUSAL TO REAPPLY, in which the reference speed is
// contaminated by an airborne rear wheel, the wheel lands and rolls at essentially zero true slip,
// the guard slip nevertheless stays around 0.33, `pastBand` therefore stays asserted, `Recover`
// returns to `Dump`, and the rear caliper stays empty for hundreds of milliseconds. That entry
// nominated exactly one next task: test whether the wheel's own rotational dynamics carry a
// REALISTIC observable — no privileged load, contact, road speed, tyre force or true slip — that
// can say "the tyre is accepting road torque" when the guard says otherwise.
//
// **Nothing here is a proposal.** Every threshold that appears is swept and reported; none is
// chosen, and no production default is written anywhere.
//
// The three levels of observable are kept apart everywhere in this file:
//
//   A. TRUTH             — the simulator's own tyre longitudinal force. Diagnostic ground truth.
//   B. IDEAL KINEMATIC   — continuous wheel angular acceleration, ECU-knowable constants only.
//   C. SENSOR-REALISTIC  — the tone ring's own pulse-derived acceleration, which is what the
//                          production controller has. This is the candidate.
// =============================================================================================

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto bar = 1.0e5;
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;
constexpr auto degrees = 57.29577951308232;

// The ensemble protocol, identical to `[.abs-ledger]` and `[.rear-hop]`: a deterministic uniform
// sweep of entry speed across +/-0.7%, odd count, no random number anywhere in it.
constexpr auto ensembleBand = 0.007;
constexpr auto ensembleCount = std::size_t{29};

// The rear axle is "stranded" below half its own capacity. The same definition `[.rear-hop]` uses,
// restated here so this file stands alone; the separator it was chosen on is 0.4722 / 0.5656.
constexpr auto strandedRear = 0.50;

constexpr auto noBrakePressure = std::array<double, cornerCount>{};

[[nodiscard]] double memberEntry(const std::size_t index, const std::size_t count)
{
    const auto offset =
        count <= 1 ? 0.0
                   : -ensembleBand + 2.0 * ensembleBand * static_cast<double>(index) / static_cast<double>(count - 1);

    return hundred * (1.0 + offset);
}

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

[[nodiscard]] SurfaceMesh gripPlate(const double leftGrip, const double rightGrip)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(2);
    mesh->materials[0].gripMultiplier = leftGrip;
    mesh->materials[0].bumpiness = 0.0;
    mesh->materials[1].gripMultiplier = rightGrip;
    mesh->materials[1].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        const auto centroid =
            (mesh->vertices[mesh->indices[triangle * 3 + 0]] + mesh->vertices[mesh->indices[triangle * 3 + 1]] +
             mesh->vertices[mesh->indices[triangle * 3 + 2]]) /
            3.0;

        mesh->surfaces[triangle] = centroid.x < 0.0 ? std::uint32_t{0} : std::uint32_t{1};
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

// ---------------------------------------------------------------------------------------------
// THE CANDIDATE OBSERVABLE, and every constant in it is ECU calibration of exactly the kind
// `AssistSetup::brakeTorquePerPressure` already is.
//
// **Section 1 of the deliverable — the wheel's equation of motion as RaceEngine actually writes
// it.** `VehicleImpl.cpp`'s pass three, per corner, per physics tick:
//
//     roadTorque   = -tyre.longitudinal * effectiveRadius            // spin-UP under braking
//     w_mid        = w_0 + (roadTorque + driveTorque) / I * dt
//     rolling      = rollingResistance * pressureScale * Fz * effectiveRadius
//     braking      = (assist ? max(0, brakes.wheels[i]) : brakeTorque * pedalResponse) * fade
//     commanded    = braking + rolling
//     arresting    = |w_mid| * I / dt
//     applied      = min(commanded, arresting)
//     w_1          = w_mid - copysign(applied / I * dt, w_mid)
//
// so, exactly, and this is the equation everything below is derived from:
//
//     I * (w_1 - w_0) / dt = roadTorque + driveTorque - copysign(min(commanded, |w_mid| I/dt), w_mid)
//
// There is no bearing-friction term and no other wheel-axis torque; rolling resistance is a TORQUE
// carried inside the same arrest clamp as the brake, which matters below because it does not vanish
// when the caliper does.
//
// Sign convention, resolved from `TyreImpl.cpp`: `forces.longitudinal` carries the sign of the slip
// ratio, so a braked wheel (slower than the road, slipRatio < 0) makes a negative longitudinal force
// and `roadTorque = -longitudinal * r` is POSITIVE — the road drags the braked wheel back up. A
// free-rolling wheel on a decelerating car sits at slightly POSITIVE slip and its road torque is
// negative: the road is de-spinning it.
//
// Inverting for what the road is doing, with w > 0 and the arrest not clamping:
//
//     T_road = I * dw/dt + T_brake + T_rolling - T_drive
//
// The ECU can form the first two terms and not the last two. What it can compute is therefore
//
//     T_road_hat = T_brake_cmd + I * dw/dt                      [N.m, spin-up positive]
//     eta        = T_road_hat / T_brake_cmd = 1 + I*(dw/dt)/T_brake_cmd     [dimensionless]
//
// which is the brief's proposed form, and it is confirmed rather than assumed: the derivation above
// reaches it, and `the wheel's equation of motion, reconstructed from what an ECU can know` below
// measures the residual the two omitted terms leave.
// ---------------------------------------------------------------------------------------------

struct RoadTorqueCalibration
{
    // Wheel rotational inertia, kg.m^2. The Golf's own, and the ONE new per-vehicle number.
    double inertia = 1.45;
    // The radius the ECU already multiplies wheel speed by — `ReferenceSpeedSetup::nominalRadius`,
    // not the effective rolling radius, because the effective one is not observable.
    double radius = tyreRadius;
    // Already on `AssistSetup`.
    std::array<double, cornerCount> torquePerPressure{};
    // Already derivable on `AssistSetup`: torquePerPressure * maximumWheelPressure.
    std::array<double, cornerCount> peakTorque{};
};

// Why a sample cannot be used. **Measured, never hidden**: the failing state has an almost empty
// caliper, so a quantity divided by brake torque is ill-conditioned exactly where it is needed, and
// the fractions below are the whole answer to whether normalised eta survives.
enum class Validity : std::uint8_t
{
    Valid,
    // The caliper is too empty to normalise by. The floor is a stated fraction of THIS corner's own
    // peak brake torque, so it means the same thing on a different car.
    LowTorqueInvalid,
    // No new tooth has gone past for longer than the stated age, so the acceleration term is a
    // difference of two numbers the ECU has not refreshed.
    SensorStale,
    // No reference, or no wheel speed measurement at all.
    OtherInvalid
};

[[nodiscard]] const char* validityName(const Validity validity)
{
    switch (validity)
    {
    case Validity::Valid:
        return "VALID";
    case Validity::LowTorqueInvalid:
        return "LOW_TORQUE_INVALID";
    case Validity::SensorStale:
        return "SENSOR_STALE";
    case Validity::OtherInvalid:
        return "OTHER_INVALID";
    }

    return "?";
}

// The floor eta is declared ill-conditioned below, as a fraction of the corner's peak brake torque.
// **A reporting parameter and not a calibration**: section 3's table sweeps it.
constexpr auto lowTorqueFraction = 0.02;
// And the reading age past which the acceleration term is stale, seconds. One controller period is
// 1 ms; ten is the age a 5 km/h wheel's tooth pitch already exceeds.
constexpr auto staleAge = 0.010;

struct Observable
{
    // The three levels, all in N.m of SPIN-UP road torque at the wheel.
    double truth = 0.0;
    double ideal = 0.0;
    double sensed = 0.0;

    // The brake torque the estimate was formed against, N.m, and the inertial term each level used.
    double brakeTorque = 0.0;
    double inertialIdeal = 0.0;
    double inertialSensed = 0.0;

    // Normalised eta at each level. Meaningless unless `validity == Valid`.
    double etaTruth = 0.0;
    double etaIdeal = 0.0;
    double etaSensed = 0.0;

    Validity validity = Validity::Valid;
};

} // namespace

namespace
{

// What the world was actually doing on the last physics tick, per wheel. **Diagnostic only.** It
// labels samples offline and it drives the TRUTH level; it never enters a candidate decision, and
// the candidate struct below cannot see it.
struct Truth
{
    std::array<bool, cornerCount> contact{true, true, true, true};
    std::array<double, cornerCount> load{};
    std::array<double, cornerCount> longitudinal{};
    std::array<double, cornerCount> effectiveRadius{};
    std::array<double, cornerCount> slipRatio{};
    std::array<double, cornerCount> peakSlip{};
    std::array<double, cornerCount> capacity{};
    // Continuous wheel angular acceleration over the last physics tick, rad/s^2. The IDEAL level's
    // one input, and the only thing that separates it from the sensor-realistic level.
    std::array<double, cornerCount> angularAcceleration{};
    std::array<double, cornerCount> wheelSpeed{};
    std::array<double, cornerCount> fade{};
    std::array<double, cornerCount> rollingTorque{};
    double staticRearLoad = 1.0;
};

// The probe-only alternative law. **`pastBand` is QUALIFIED, never replaced**: the production
// `advanceAntilockChannel` is called with a per-step local copy of `AntilockSetup` whose
// `slipAwareRecovery` is cleared when the road-torque evidence says this wheel has recovered.
// Clearing it for one step makes `pastBand` false for that step, which is exactly
//
//     pastBand AND road-torque evidence says the wheel has NOT recovered
//
// **The polarity is derived from the data, in `how well each observable separates ...` below, and
// not assumed here.** No production file is touched: `AntilockSetup` is a value type and the copy
// is local to the call.
struct Candidate
{
    bool enabled = false;
    // The road-torque estimate above which the wheel is judged to be STILL resisting — that is,
    // still slipping — so `pastBand` is left alone. Below it the wheel is judged recovered and
    // `pastBand` is suppressed. Expressed as a fraction of the corner's own peak brake torque when
    // `normalised`, and in N.m when not.
    double accept = 0.0;
    bool normalised = true;
    // Which channels the qualification applies to. The failure is on the rear axle; the front arm
    // exists so the answer is not "it happened to be the rear".
    bool rearOnly = true;
    // Use the ideal continuous acceleration instead of the tone ring's. The sensor-realistic arm is
    // the one that matters; this exists to say how much the ring costs.
    bool useIdeal = false;
    // Normalised eta rather than the unnormalised road torque. Section 7's arm.
    bool useEta = false;
    double etaAccept = 0.0;
};

// One controller step of one channel, with every decision term the production law evaluated and
// every level of the candidate observable beside it.
struct Step
{
    double time = 0.0;
    std::size_t channel = 0;
    std::size_t controlWheel = 0;

    ModulatorPhase before = ModulatorPhase::Passive;
    ModulatorPhase after = ModulatorPhase::Passive;
    double pressureBefore = 0.0;
    double pressureAfter = 0.0;
    double request = 0.0;

    // The production decision terms, evaluated on the state the production call left behind.
    double sensedSpeed = 0.0;
    double acceleration = 0.0;
    double excess = 0.0;
    double slip = 0.0;
    double guardSlip = 0.0;
    double referenceSpeed = 0.0;
    double referenceRate = 0.0;
    double sensorAge = 0.0;
    bool referenceValid = false;
    bool losing = false;
    bool surging = false;
    bool pastBand = false;
    // What `pastBand` would have been WITHOUT the candidate's qualification, so a changed decision
    // can be counted.
    bool pastBandUnqualified = false;
    bool qualified = false;

    Observable observable;

    // Ground truth at the control wheel, for labelling only.
    double trueSlip = 0.0;
    double trueLoad = 0.0;
    double trueLongitudinal = 0.0;
    double trueCapacity = 0.0;
    bool trueContact = true;
};

struct ShadowState
{
    AssistState assists{};
    std::array<double, cornerCount> sensedSpeed{};
    std::array<double, cornerCount> estimatedSlip{};
    std::array<ModulatorPhase, cornerCount> phase{};
    std::size_t qualifiedSteps = 0;
    std::size_t changedDecisions = 0;
};

// The observable, formed for one wheel at one controller step from the channel state as it stands
// BEFORE the state machine runs — which is when the decision is taken.
[[nodiscard]] Observable observeRoadTorque(const RoadTorqueCalibration& calibration, const std::size_t wheel,
                                           const double pressure, const double sensedAcceleration,
                                           const double sensorAge, const bool readingValid, const bool referenceValid,
                                           const Truth& truth)
{
    auto observable = Observable{};

    // What the ECU asked the caliper for. It is `pressure * brakeTorquePerPressure`, which is what
    // `updateAssists` already computes to fill `BrakeCommand::wheels`, and it is the PRE-FADE
    // figure: a hot pad delivers less than this and the ECU cannot know it.
    observable.brakeTorque = std::max(0.0, pressure) * calibration.torquePerPressure[wheel];

    observable.inertialSensed = calibration.inertia * sensedAcceleration / calibration.radius;
    observable.inertialIdeal = calibration.inertia * truth.angularAcceleration[wheel];

    observable.sensed = observable.brakeTorque + observable.inertialSensed;
    observable.ideal = observable.brakeTorque + observable.inertialIdeal;

    // TRUTH: the road's own spin-up torque, `-Fx * r_eff`, straight out of the plant.
    observable.truth = -truth.longitudinal[wheel] * truth.effectiveRadius[wheel];

    const auto floor = lowTorqueFraction * calibration.peakTorque[wheel];

    if (!readingValid || !referenceValid)
    {
        observable.validity = Validity::OtherInvalid;
    }
    else if (sensorAge > staleAge)
    {
        observable.validity = Validity::SensorStale;
    }
    else if (observable.brakeTorque <= floor)
    {
        observable.validity = Validity::LowTorqueInvalid;
    }

    if (observable.brakeTorque > 0.0)
    {
        observable.etaTruth = observable.truth / observable.brakeTorque;
        observable.etaIdeal = observable.ideal / observable.brakeTorque;
        observable.etaSensed = observable.sensed / observable.brakeTorque;
    }

    return observable;
}

// The probe-local copy of `updateAssists`'s outer loop, following `[.rear-hop]`'s `shadowUpdate`.
// **The control law is never copied**: `advanceAntilockChannel` is called, with a local copy of the
// setup as the only intervention. Traction control and the cornering brake are not stepped, and the
// bit-identity case at the foot of this file is what proves the omission costs nothing.
[[nodiscard]] BrakeCommand shadowUpdate(const AssistSetup& setup, ShadowState& shadow, const AssistSensors& sensors,
                                        const double brakeDemand, const std::array<double, cornerCount>& wheelPressure,
                                        const double deltaTime, const RoadTorqueCalibration& calibration,
                                        const Candidate& candidate, const Truth& truth, std::vector<Step>* trace,
                                        const double runTime)
{
    auto& state = shadow.assists;
    const auto braking = brakeDemand > setup.brakeSwitch;

    auto driverPressure = std::array<double, cornerCount>{};
    for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
    {
        driverPressure[wheel] = std::max(wheelPressure[wheel], 0.0);
    }

    const auto period = setup.controlRate > 0.0 ? 1.0 / setup.controlRate : deltaTime;

    state.clockRemainder += deltaTime;

    auto readings = WheelSpeedReadings{};
    auto stepped = false;
    auto elapsed = 0.0;

    while (state.clockRemainder >= period)
    {
        state.clockRemainder -= period;
        stepped = true;
        elapsed += period;

        readings = sampleWheelSensors(setup.toneRing, state.sensors, sensors.wheelSpeeds, period);

        advanceReferenceSpeed(setup.reference, state.reference, readings, braking, period);

        const auto requested = driverPressure;

        auto frontRequests = std::array<double, 2>{requested[0], requested[1]};
        const auto ceiling = advanceYawMomentDelay(setup.antilock, state.antilock.yawDelay, state.antilock.channels[0],
                                                   state.antilock.channels[1], frontRequests,
                                                   sensors.lateralAcceleration, braking, period);

        for (auto index = std::size_t{0}; index < brakeChannelCount; index++)
        {
            const auto channel = index == 0   ? BrakeChannel::FrontLeft
                                 : index == 1 ? BrakeChannel::FrontRight
                                              : BrakeChannel::Rear;
            const auto controlWheel = antilockControlWheel(channel, readings);

            auto channelRequest = 0.0;
            for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
            {
                if (antilockDrivesWheel(channel, wheel))
                {
                    channelRequest = std::max(channelRequest, requested[wheel]);
                }
            }

            if (state.antilock.yawDelay.engaged && index == state.antilock.yawDelay.highChannel)
            {
                channelRequest = std::min(channelRequest, ceiling);
            }

            const auto wheelSpeed = std::abs(sensedRoadSpeed(setup.reference, readings[controlWheel]));

            auto& channelState = state.antilock.channels[index];
            const auto snapshot = channelState;

            // --- the acceleration the production call is ABOUT to use --------------------------
            //
            // A replication of `advanceAntilockChannel`'s sensor bookkeeping and of nothing else,
            // needed because the observable has to be formed at the instant of the decision and the
            // production call updates this field inside itself. It is self-checking: the assertion
            // below compares it against what the production call actually left.
            auto predictedAcceleration = channelState.acceleration;
            auto predictedAge = channelState.sinceUpdate + period;
            auto predictedLastSpeed = channelState.lastSpeed;

            if (readings[controlWheel].pulses != channelState.lastPulses)
            {
                const auto span = channelState.sinceUpdate + period;
                if (channelState.lastPulses > 0 && span > 0.0)
                {
                    predictedAcceleration = (wheelSpeed - channelState.lastSpeed) / span;
                }
                predictedAge = 0.0;
                predictedLastSpeed = wheelSpeed;
            }

            // And the guard the production law is about to evaluate, from the same replicated
            // bookkeeping. Self-checking in the same way: the assertion after the call compares it
            // against the state the production call left.
            const auto projected = std::max(wheelSpeed, predictedLastSpeed + state.reference.rate * predictedAge);
            const auto guardSlip = state.reference.valid ? estimatedSlip(state.reference.speed, projected) : 0.0;
            const auto pastBandUnqualified = (setup.antilock.recoveryAuthority != RecoveryAuthority::Disabled) &&
                                             state.reference.valid && guardSlip > setup.antilock.slipEnter;

            const auto observable =
                observeRoadTorque(calibration, controlWheel, snapshot.pressure, predictedAcceleration, predictedAge,
                                  readings[controlWheel].valid, state.reference.valid, truth);

            // --- the qualification ------------------------------------------------------------
            auto local = setup.antilock;
            auto qualified = false;

            // **Only where `pastBand` would actually fire.** Clearing `slipAwareRecovery` for a step
            // also removes the re-apply proximity taper, so qualifying a step whose guard slip is
            // already inside the band would change behaviour the qualification is not about.
            if (candidate.enabled && (!candidate.rearOnly || index == 2) && pastBandUnqualified)
            {
                const auto estimate = candidate.useIdeal ? observable.ideal : observable.sensed;

                if (candidate.useEta)
                {
                    // Section 7's arm. An ill-conditioned sample cannot qualify anything, which is
                    // the honest behaviour and is why this arm is expected to do less.
                    const auto eta = candidate.useIdeal ? observable.etaIdeal : observable.etaSensed;
                    qualified = observable.validity == Validity::Valid && eta < candidate.etaAccept;
                }
                else
                {
                    const auto accept = candidate.normalised ? candidate.accept * calibration.peakTorque[controlWheel]
                                                             : candidate.accept;
                    qualified = estimate < accept;
                }

                if (qualified)
                {
                    local.recoveryAuthority = RecoveryAuthority::Disabled;
                }
            }

            const auto pressure = advanceAntilockChannel(
                local, channel, channelState, readings[controlWheel], wheelSpeed, state.reference.speed,
                state.reference.rate, state.reference.valid, channelRequest, AntilockChannelInputs{}, period);

            // Cheap in the common path on purpose: a `REQUIRE` per controller step per channel is
            // a quarter of a million Catch2 assertions per ensemble arm.
            if (channelState.acceleration != predictedAcceleration)
            {
                REQUIRE(channelState.acceleration == predictedAcceleration);
            }

            if (qualified)
            {
                shadow.qualifiedSteps++;
            }

            {
                const auto after =
                    std::max(wheelSpeed, channelState.lastSpeed + state.reference.rate * channelState.sinceUpdate);
                if (after != projected)
                {
                    REQUIRE(after == projected);
                }
            }

            if (trace != nullptr)
            {
                auto record = Step{};
                record.time = runTime + elapsed;
                record.channel = index;
                record.controlWheel = controlWheel;
                record.before = snapshot.phase;
                record.after = channelState.phase;
                record.pressureBefore = snapshot.pressure;
                record.pressureAfter = pressure;
                record.request = channelRequest;
                record.sensedSpeed = wheelSpeed;
                record.acceleration = channelState.acceleration;
                record.excess = channelState.acceleration - state.reference.rate;
                record.referenceSpeed = state.reference.speed;
                record.referenceRate = state.reference.rate;
                record.referenceValid = state.reference.valid;
                record.sensorAge = predictedAge;
                record.slip = state.reference.valid ? estimatedSlip(state.reference.speed, wheelSpeed) : 0.0;

                record.guardSlip = guardSlip;
                record.losing = channelState.acceleration < setup.antilock.lockDeceleration;
                record.surging = record.excess > setup.antilock.recoverySurge;
                record.pastBandUnqualified = pastBandUnqualified;
                record.pastBand = record.pastBandUnqualified && !qualified;
                record.qualified = qualified;
                record.observable = observable;
                record.trueSlip = truth.slipRatio[controlWheel];
                record.trueLoad = truth.load[controlWheel];
                record.trueLongitudinal = truth.longitudinal[controlWheel];
                record.trueCapacity = truth.capacity[controlWheel];
                record.trueContact = truth.contact[controlWheel];

                trace->push_back(record);
            }

            for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
            {
                if (antilockDrivesWheel(channel, wheel))
                {
                    state.pressure[wheel] = pressure;
                }
            }
        }
    }

    if (!stepped)
    {
        readings = sampleWheelSensors(setup.toneRing, state.sensors, sensors.wheelSpeeds, 0.0);
    }

    auto command = BrakeCommand{};
    command.commanded = true;

    for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
    {
        command.wheels[wheel] = state.pressure[wheel] * setup.brakeTorquePerPressure[wheel];
        shadow.sensedSpeed[wheel] = sensedRoadSpeed(setup.reference, readings[wheel]);
        shadow.estimatedSlip[wheel] = estimatedSlip(state.reference.speed, std::abs(shadow.sensedSpeed[wheel]));
    }

    for (auto index = std::size_t{0}; index < brakeChannelCount; index++)
    {
        const auto channel = index == 0   ? BrakeChannel::FrontLeft
                             : index == 1 ? BrakeChannel::FrontRight
                                          : BrakeChannel::Rear;
        for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
        {
            if (antilockDrivesWheel(channel, wheel))
            {
                shadow.phase[wheel] = state.antilock.channels[index].phase;
            }
        }
    }

    return command;
}

} // namespace

namespace
{

// --- one physics tick, for the equation check and for the run statistics --------------------

struct WheelTick
{
    double load = 0.0;
    double longitudinal = 0.0;
    double capacity = 0.0;
    double slipRatio = 0.0;
    double effectiveRadius = 0.0;
    double wheelSpeedBefore = 0.0;
    double wheelSpeedAfter = 0.0;
    double roadTorque = 0.0;
    double brakeTorqueCommanded = 0.0;
    double fade = 0.0;
    double rollingTorque = 0.0;
    double driveTorque = 0.0;
    double pressure = 0.0;
    ModulatorPhase phase = ModulatorPhase::Passive;
    bool inContact = false;
};

struct Tick
{
    double time = 0.0;
    double speed = 0.0;
    double rearAxleLoad = 0.0;
    bool rearContact = true;
    std::array<WheelTick, cornerCount> wheels{};
};

struct Run
{
    std::vector<Tick> ticks;
    std::vector<Step> steps;
    double distance = 0.0;
    double time = 0.0;
    double entrySpeed = 0.0;
    double lateralTravel = 0.0;
    double finalYaw = 0.0;
    double peakYawRate = 0.0;
    double maximumDiscTemperature = 0.0;
    double minimumFade = 1.0;
    // One rear wheel's settled load, so a fraction-of-static label means the same thing on a
    // different car [borrowed-numbers-need-a-platform-match].
    double staticRearLoad = 1.0;
    bool stopped = false;
    bool grounded = true;
    std::size_t rearAirborneTicks = 0;
    std::size_t qualifiedSteps = 0;
    std::size_t landings = 0;
};

[[nodiscard]] Run record(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                         const RoadTorqueCalibration& calibration, const Candidate& candidate, const double pedal,
                         const double entrySpeed, const double steering = 0.0, const bool traceSteps = false,
                         const std::array<double, cornerCount>& driveTorque = noDriveTorque)
{
    auto state = VehicleState{};
    settle(setup, state, world, entrySpeed);

    auto shadow = ShadowState{};
    auto lastStep = VehicleStep{};
    auto truth = Truth{};
    const auto passive = Candidate{};

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
        const auto command = shadowUpdate(assists, shadow, sense(), 0.0, noBrakePressure, tick, calibration, passive,
                                          truth, nullptr, 0.0);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
    }
    REQUIRE(std::abs(state.chassis.linearVelocity.z - entrySpeed) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - designHeight) < 0.1);
    REQUIRE(std::abs(state.chassis.position.x) < 0.05);

    truth.staticRearLoad =
        std::max(1.0, 0.5 * (lastStep.corners[2].forces.tireVertical + lastStep.corners[3].forces.tireVertical));

    auto run = Run{};
    run.staticRearLoad = truth.staticRearLoad;
    run.entrySpeed = state.chassis.linearVelocity.z;
    run.ticks.reserve(360 * 8);

    const auto start = state.chassis.position;
    auto previousWheelSpeed = std::array<double, cornerCount>{};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        previousWheelSpeed[index] = state.corners[index].wheelSpeed;
    }

    auto input = VehicleInput{};
    input.brake = pedal;
    input.steering = steering;

    const auto pressures = brakeCircuitPressures(setup, pedal);
    auto rearWasOff = false;
    // The tyre-pressure multiplier on rolling resistance, read BEFORE the tick that uses it — which
    // is where `VehicleImpl` reads it. The Golf states `tyrePressure = true`, so it is not 1.0.
    auto rollingScale = std::array<double, cornerCount>{1.0, 1.0, 1.0, 1.0};

    for (auto step = 0; step < 360 * 30; step++)
    {
        // Ground truth, one physics tick old — what a probe standing outside the tick has, and the
        // honest lag to give it.
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];

            truth.contact[index] = lastStep.telemetry.wheels[index].inContact;
            truth.load[index] = solution.forces.tireVertical;
            truth.longitudinal[index] = solution.contact.tyre.longitudinal;
            truth.effectiveRadius[index] = solution.contact.effectiveRadius;
            truth.slipRatio[index] = solution.contact.slip.slipRatio;
            truth.peakSlip[index] = solution.contact.tyre.longitudinalPeakSlip;
            truth.capacity[index] = tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, truth.load[index],
                                                 solution.patch.gripMultiplier) *
                                    truth.load[index];
            truth.angularAcceleration[index] = (state.corners[index].wheelSpeed - previousWheelSpeed[index]) / tick;
            truth.wheelSpeed[index] = state.corners[index].wheelSpeed;
            truth.fade[index] = setup.brakeThermal ? frictionAtTemperature(setup.corners[index].disc.couple,
                                                                           state.corners[index].discTemperature) /
                                                         std::max(setup.corners[index].disc.couple.coefficient, 1e-9)
                                                   : 1.0;
            rollingScale[index] =
                setup.tyrePressure
                    ? tyrePressureRollingResistanceScale(setup.corners[index].tyre.pressure, state.corners[index].tyre)
                    : 1.0;
            truth.rollingTorque[index] = setup.corners[index].rollingResistance * rollingScale[index] *
                                         truth.load[index] * truth.effectiveRadius[index];
        }

        const auto nowOff = !truth.contact[2] || !truth.contact[3];
        run.landings += (rearWasOff && !nowOff) ? 1 : 0;
        rearWasOff = nowOff;

        const auto speedBefore =
            std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            previousWheelSpeed[index] = state.corners[index].wheelSpeed;
        }

        const auto brakes = shadowUpdate(assists, shadow, sense(), pedal, pressures, tick, calibration, candidate,
                                         truth, traceSteps ? &run.steps : nullptr, run.time);

        const auto stepped = stepVehicle(setup, state, input, driveTorque, world, tick, brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        run.time += tick;
        run.peakYawRate = std::max(run.peakYawRate, std::abs(lastStep.telemetry.yawRate));

        auto sample = Tick{};
        sample.time = run.time;
        sample.speed = state.chassis.linearVelocity.z;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];
            auto& wheel = sample.wheels[index];

            wheel.load = solution.forces.tireVertical;
            wheel.longitudinal = solution.contact.tyre.longitudinal;
            wheel.capacity = tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, wheel.load,
                                          solution.patch.gripMultiplier) *
                             wheel.load;
            wheel.slipRatio = solution.contact.slip.slipRatio;
            wheel.effectiveRadius = solution.contact.effectiveRadius;
            wheel.wheelSpeedBefore = speedBefore[index];
            wheel.wheelSpeedAfter = state.corners[index].wheelSpeed;
            wheel.roadTorque = -wheel.longitudinal * wheel.effectiveRadius;
            wheel.fade = setup.brakeThermal ? frictionAtTemperature(setup.corners[index].disc.couple,
                                                                    state.corners[index].discTemperature) /
                                                  std::max(setup.corners[index].disc.couple.coefficient, 1e-9)
                                            : 1.0;
            wheel.brakeTorqueCommanded = std::max(0.0, brakes.wheels[index]);
            wheel.rollingTorque =
                setup.corners[index].rollingResistance * rollingScale[index] * wheel.load * wheel.effectiveRadius;
            wheel.driveTorque = driveTorque[index];
            wheel.pressure = shadow.assists.pressure[index];
            wheel.phase = shadow.phase[index];
            wheel.inContact = lastStep.telemetry.wheels[index].inContact;

            run.grounded = run.grounded && wheel.inContact;
            run.maximumDiscTemperature = std::max(run.maximumDiscTemperature, state.corners[index].discTemperature);
            run.minimumFade = std::min(run.minimumFade, wheel.fade);
        }

        sample.rearContact = sample.wheels[2].inContact && sample.wheels[3].inContact;
        sample.rearAxleLoad = sample.wheels[2].load + sample.wheels[3].load;
        run.rearAirborneTicks += sample.rearContact ? 0 : 1;

        run.ticks.push_back(sample);

        if (sample.speed <= 0.0)
        {
            run.stopped = true;
            break;
        }
    }

    run.distance = state.chassis.position.z - start.z;
    run.lateralTravel = state.chassis.position.x - start.x;
    run.finalYaw = lastStep.telemetry.yaw;
    run.qualifiedSteps = shadow.qualifiedSteps;

    return run;
}

// --- statistics ------------------------------------------------------------------------------

struct Distribution
{
    std::size_t n = 0;
    double minimum = 0.0;
    double p10 = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double p90 = 0.0;
    double maximum = 0.0;
    double deviation = 0.0;
};

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
                        .median = percentile(values, 0.50),
                        .mean = mean,
                        .p90 = percentile(values, 0.90),
                        .maximum = values.back(),
                        .deviation = values.size() > 1 ? std::sqrt(squared / (count - 1.0)) : 0.0};
}

void distributionHeader(const char* what)
{
    std::printf("\n  %-34s     n        min        P10     MEDIAN       mean        P90        max         sd\n", what);
}

void distributionRow(const char* name, const Distribution& value)
{
    std::printf("  %-34s %5zu  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f\n", name, value.n, value.minimum,
                value.p10, value.median, value.mean, value.p90, value.maximum, value.deviation);
}

// Threshold-independent separation. The Mann-Whitney statistic: the probability that a randomly
// drawn `positive` exceeds a randomly drawn `negative`, ties counted as a half. 0.5 is no
// information at all and 1.0 is perfect separation; below 0.5 the separation is real and the
// polarity is the other way round, which is why this is reported rather than an accuracy.
[[nodiscard]] double areaUnderCurve(std::vector<double> positive, std::vector<double> negative)
{
    if (positive.empty() || negative.empty())
    {
        return 0.5;
    }

    std::sort(positive.begin(), positive.end());
    std::sort(negative.begin(), negative.end());

    auto above = 0.0;

    for (const auto value : positive)
    {
        const auto lower = static_cast<double>(
            std::distance(negative.begin(), std::lower_bound(negative.begin(), negative.end(), value)));
        const auto upper = static_cast<double>(
            std::distance(negative.begin(), std::upper_bound(negative.begin(), negative.end(), value)));

        above += lower + 0.5 * (upper - lower);
    }

    return above / (static_cast<double>(positive.size()) * static_cast<double>(negative.size()));
}

// The best single-threshold operating point on the candidate, reported as evidence about the
// SEPARATION and never as a proposal. `positive` is the population a threshold should keep.
struct Operating
{
    double threshold = 0.0;
    double truePositive = 0.0;
    double falsePositive = 0.0;
    double youden = 0.0;
};

[[nodiscard]] Operating bestOperating(const std::vector<double>& positive, const std::vector<double>& negative)
{
    auto candidates = positive;
    candidates.insert(candidates.end(), negative.begin(), negative.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    auto best = Operating{};
    best.youden = -1.0;

    for (const auto threshold : candidates)
    {
        auto hit = std::size_t{0};
        auto miss = std::size_t{0};

        for (const auto value : positive)
        {
            hit += value >= threshold ? 1 : 0;
        }
        for (const auto value : negative)
        {
            miss += value >= threshold ? 1 : 0;
        }

        const auto truePositive =
            static_cast<double>(hit) / static_cast<double>(std::max(positive.size(), std::size_t{1}));
        const auto falsePositive =
            static_cast<double>(miss) / static_cast<double>(std::max(negative.size(), std::size_t{1}));

        if (truePositive - falsePositive > best.youden)
        {
            best = Operating{.threshold = threshold,
                             .truePositive = truePositive,
                             .falsePositive = falsePositive,
                             .youden = truePositive - falsePositive};
        }
    }

    return best;
}

} // namespace

namespace
{

[[nodiscard]] RoadTorqueCalibration calibrationOf(const VehicleSetup& setup, const AssistSetup& assists)
{
    auto calibration = RoadTorqueCalibration{};

    calibration.inertia = setup.corners[0].wheelInertia;
    calibration.radius = assists.reference.nominalRadius;
    calibration.torquePerPressure = assists.brakeTorquePerPressure;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        // Every corner on this car states the same wheel, and a per-corner inertia would be read
        // here if one did not.
        REQUIRE(setup.corners[index].wheelInertia == calibration.inertia);
        calibration.peakTorque[index] =
            assists.brakeTorquePerPressure[index] * std::max(assists.maximumWheelPressure[index], 0.0);
    }

    return calibration;
}

[[nodiscard]] AssistSetup withAntilock(const VehicleSetup& setup)
{
    auto assists = golfGtiMk7Assists(setup);
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    return assists;
}

struct AxleUse
{
    double front = 0.0;
    double rear = 0.0;
};

[[nodiscard]] AxleUse axleUse(const Run& run)
{
    auto force = std::array<double, 2>{};
    auto capacity = std::array<double, 2>{};

    for (const auto& sample : run.ticks)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto axle = index < 2 ? std::size_t{0} : std::size_t{1};

            force[axle] += std::abs(sample.wheels[index].longitudinal);
            capacity[axle] += sample.wheels[index].capacity;
        }
    }

    return AxleUse{.front = capacity[0] > 1.0 ? force[0] / capacity[0] : 0.0,
                   .rear = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0};
}

// Seconds a wheel's channel spent anywhere other than passive. A cycling RATE has to be divided by
// this and not by the whole stop: on this car the front channel spends most of a dry stop passive.
[[nodiscard]] double engagedOf(const Run& run, const std::size_t wheel)
{
    auto ticks = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        ticks += sample.wheels[wheel].phase != ModulatorPhase::Passive ? 1 : 0;
    }

    return static_cast<double>(ticks) * tick;
}

[[nodiscard]] std::size_t cyclesOf(const Run& run, const std::size_t wheel)
{
    auto cycles = std::size_t{0};
    auto previous = ModulatorPhase::Passive;

    for (const auto& sample : run.ticks)
    {
        const auto phase = sample.wheels[wheel].phase;
        cycles += (phase == ModulatorPhase::Dump && previous != ModulatorPhase::Dump) ? 1 : 0;
        previous = phase;
    }

    return cycles;
}

struct Member
{
    std::size_t index = 0;
    double entry = 0.0;
    double distance = 0.0;
    double frontUtilisation = 0.0;
    double rearUtilisation = 0.0;
    double lateralTravel = 0.0;
    double finalYaw = 0.0;
    double peakYawRate = 0.0;
    std::size_t rearAirborne = 0;
    std::size_t frontCycles = 0;
    std::size_t rearCycles = 0;
    std::size_t qualifiedSteps = 0;
    double frontEngaged = 0.0;
    double rearEngaged = 0.0;
    double time = 0.0;
    bool grounded = false;
    bool stopped = false;

    [[nodiscard]] bool stranded() const
    {
        return rearUtilisation < strandedRear;
    }
};

[[nodiscard]] std::vector<Member> ensembleOf(const VehicleSetup& setup, const PhysicsWorld& world,
                                             const AssistSetup& assists, const RoadTorqueCalibration& calibration,
                                             const Candidate& candidate, const double pedal,
                                             const double steering = 0.0, const std::size_t count = ensembleCount)
{
    auto members = std::vector<Member>{};
    members.reserve(count);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto entry = memberEntry(index, count);
        const auto run = record(setup, world, assists, calibration, candidate, pedal, entry, steering);
        const auto use = axleUse(run);

        members.push_back(Member{.index = index,
                                 .entry = entry,
                                 .distance = run.distance,
                                 .frontUtilisation = use.front,
                                 .rearUtilisation = use.rear,
                                 .lateralTravel = run.lateralTravel,
                                 .finalYaw = run.finalYaw,
                                 .peakYawRate = run.peakYawRate,
                                 .rearAirborne = run.rearAirborneTicks,
                                 .frontCycles = cyclesOf(run, 0),
                                 .rearCycles = cyclesOf(run, 2),
                                 .qualifiedSteps = run.qualifiedSteps,
                                 .frontEngaged = engagedOf(run, 0),
                                 .rearEngaged = engagedOf(run, 2),
                                 .time = run.time,
                                 .grounded = run.grounded,
                                 .stopped = run.stopped});
    }

    return members;
}

struct ArmSummary
{
    Distribution distance;
    Distribution rearUtilisation;
    Distribution frontUtilisation;
    std::size_t stranded = 0;
    std::size_t n = 0;
    double meanRearAirborne = 0.0;
    double meanRearCycles = 0.0;
    double meanFrontCycles = 0.0;
    double meanQualified = 0.0;
    double highestStranded = 0.0;
    double lowestRecovered = 1e9;
};

[[nodiscard]] ArmSummary summarise(const std::vector<Member>& members)
{
    auto summary = ArmSummary{};
    auto distances = std::vector<double>{};
    auto rear = std::vector<double>{};
    auto front = std::vector<double>{};

    for (const auto& member : members)
    {
        REQUIRE(member.stopped);
        distances.push_back(member.distance);
        rear.push_back(member.rearUtilisation);
        front.push_back(member.frontUtilisation);

        summary.stranded += member.stranded() ? 1 : 0;
        summary.meanRearAirborne += static_cast<double>(member.rearAirborne);
        summary.meanRearCycles += static_cast<double>(member.rearCycles);
        summary.meanFrontCycles += static_cast<double>(member.frontCycles);
        summary.meanQualified += static_cast<double>(member.qualifiedSteps);

        if (member.stranded())
        {
            summary.highestStranded = std::max(summary.highestStranded, member.rearUtilisation);
        }
        else
        {
            summary.lowestRecovered = std::min(summary.lowestRecovered, member.rearUtilisation);
        }
    }

    const auto count = static_cast<double>(std::max(members.size(), std::size_t{1}));
    summary.n = members.size();
    summary.meanRearAirborne /= count;
    summary.meanRearCycles /= count;
    summary.meanFrontCycles /= count;
    summary.meanQualified /= count;
    summary.distance = distributionOf(distances);
    summary.rearUtilisation = distributionOf(rear);
    summary.frontUtilisation = distributionOf(front);

    return summary;
}

void printArmHeader()
{
    std::printf("\n  %-38s  stranded      P10   MEDIAN      P90       sd    rear util  front util   rearOff"
                "  rearCyc  qualified\n",
                "arm");
}

void printArm(const char* name, const ArmSummary& summary)
{
    std::printf("  %-38s   %2zu/%2zu   %7.3f  %7.3f  %7.3f  %7.3f      %6.3f      %6.3f   %6.1f   %6.1f   %8.0f\n",
                name, summary.stranded, summary.n, summary.distance.p10, summary.distance.median, summary.distance.p90,
                summary.distance.deviation, summary.rearUtilisation.median, summary.frontUtilisation.median,
                summary.meanRearAirborne, summary.meanRearCycles, summary.meanQualified);
}

} // namespace

// =============================================================================================
// 0. THE SHADOW CONTROLLER IS THE PRODUCTION CONTROLLER
// =============================================================================================

TEST_CASE("the shadow controller with the candidate off is the production controller, to the bit", "[.road-torque]")
{
    // **Nothing below this case means anything without it.** The probe carries its own copy of
    // `updateAssists`'s outer loop so that the qualification can act between the channel's decision
    // and the caliper; this is what says that copy is the same loop. Traction control and the
    // cornering brake are not stepped in the shadow and this is what proves the omission costs
    // nothing on this fixture.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto off = Candidate{};

    std::printf("\n=== the shadow controller against production `updateAssists`, %zu members ===\n", ensembleCount);
    std::printf("  member    shadow m   production m       delta m   rear util delta   rearOff delta\n");

    auto identical = std::size_t{0};

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);
        const auto shadow = record(setup.value(), world.value(), assists, calibration, off, 1.0, entry);

        // The same stop through production `updateAssists`.
        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), entry);

        auto assistState = AssistState{};
        auto lastStep = VehicleStep{};
        auto force = std::array<double, 2>{};
        auto capacity = std::array<double, 2>{};
        auto rearOff = std::size_t{0};

        const auto sense = [&]
        {
            auto sensors = AssistSensors{};
            for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
            {
                sensors.wheelSpeeds[corner] = state.corners[corner].wheelSpeed;
            }
            sensors.yawRate = lastStep.telemetry.yawRate;
            sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

            return sensors;
        };

        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        const auto start = state.chassis.position;
        auto input = VehicleInput{};
        input.brake = 1.0;
        const auto pressures = brakeCircuitPressures(setup.value(), 1.0);
        auto stopped = false;

        for (auto step = 0; step < 360 * 30; step++)
        {
            const auto command =
                updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0}, pressures, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            auto off2 = false;
            for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
            {
                const auto& solution = lastStep.corners[corner];
                const auto axle = corner < 2 ? std::size_t{0} : std::size_t{1};

                force[axle] += std::abs(solution.contact.tyre.longitudinal);
                capacity[axle] += tyreFriction(setup->corners[corner].tyre, TyreAxis::Longitudinal,
                                               solution.forces.tireVertical, solution.patch.gripMultiplier) *
                                  solution.forces.tireVertical;

                off2 = off2 || (corner >= 2 && !lastStep.telemetry.wheels[corner].inContact);
            }

            rearOff += off2 ? 1 : 0;

            if (state.chassis.linearVelocity.z <= 0.0)
            {
                stopped = true;
                break;
            }
        }

        REQUIRE(stopped);

        const auto distance = state.chassis.position.z - start.z;
        const auto rearUse = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0;
        const auto shadowUse = axleUse(shadow);

        const auto same =
            distance == shadow.distance && rearUse == shadowUse.rear && rearOff == shadow.rearAirborneTicks;
        identical += same ? 1 : 0;

        std::printf("  %6zu  %10.6f  %13.6f  %12.3e  %16.3e  %14zd\n", index, shadow.distance, distance,
                    shadow.distance - distance, shadowUse.rear - rearUse,
                    static_cast<std::ptrdiff_t>(shadow.rearAirborneTicks) - static_cast<std::ptrdiff_t>(rearOff));
    }

    std::printf("\n  identical on every compared channel: %zu of %zu\n", identical, ensembleCount);

    REQUIRE(identical == ensembleCount);
}

// =============================================================================================
// 1. THE WHEEL TORQUE EQUATION
// =============================================================================================

TEST_CASE("the wheel's equation of motion, reconstructed tick by tick and then from what an ECU can know",
          "[.road-torque]")
{
    // Two reconstructions of the same integration, on one full-pedal anti-lock stop.
    //
    //   EXACT   — every term `VehicleImpl.cpp` uses, including the two the ECU cannot have. If this
    //             does not close to rounding, the equation written at the head of this file is wrong
    //             and nothing downstream means anything.
    //   ECU     — `T_road_hat = T_brake_cmd + I * dw/dt`, which drops rolling resistance, pad fade
    //             and the arrest clamp. The residual IS the observable's bias, and it is measured
    //             here rather than assumed away.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto off = Candidate{};

    const auto run = record(setup.value(), world.value(), assists, calibration, off, 1.0,
                            memberEntry(ensembleCount / 2, ensembleCount));
    REQUIRE(run.stopped);

    std::printf("\n=== the wheel's rotational dynamics, as RaceEngine writes them ===\n");
    std::printf("  source   RaceEngine/Physics/Impl/VehicleImpl.cpp, pass three, `--- wheel spin ---`\n\n");
    std::printf("    roadTorque = -tyre.longitudinal * effectiveRadius          [spin-UP, N.m]\n");
    std::printf("    w_mid      = w_0 + (roadTorque + driveTorque) / I * dt\n");
    std::printf("    rolling    = rollingResistance * pressureScale * Fz * effectiveRadius\n");
    std::printf("    braking    = commandedTorque * fade(discTemperature)\n");
    std::printf("    commanded  = braking + rolling\n");
    std::printf("    applied    = min(commanded, |w_mid| * I / dt)\n");
    std::printf("    w_1        = w_mid - copysign(applied / I * dt, w_mid)\n\n");
    std::printf("  so, exactly:\n");
    std::printf("    I * (w_1 - w_0)/dt = roadTorque + driveTorque"
                " - copysign(min(commanded, |w_mid| I/dt), w_mid)\n\n");
    std::printf("  There is no bearing-friction term and no other wheel-axis torque. Rolling\n");
    std::printf("  resistance is a TORQUE inside the same arrest clamp as the brake, which is why it\n");
    std::printf("  does not vanish when the caliper does.\n");
    std::printf("\n  car: wheel inertia %.4f kg.m^2, rolling resistance %.4f, nominal radius %.4f m\n",
                calibration.inertia, setup->corners[0].rollingResistance, calibration.radius);
    std::printf("  peak brake torque: front %.1f N.m, rear %.1f N.m\n", calibration.peakTorque[0],
                calibration.peakTorque[2]);
    std::printf("  brake thermal %s; over this stop the hottest disc reached %.1f C and the LOWEST\n",
                setup->brakeThermal ? "ON" : "off", run.maximumDiscTemperature);
    std::printf("  fade multiplier anywhere was %.6f -- so `T_brake_cmd` is %s the delivered torque.\n",
                run.minimumFade, run.minimumFade == 1.0 ? "EXACTLY" : "NOT exactly");

    auto exactResidual = std::vector<double>{};
    auto ecuResidual = std::vector<double>{};
    auto ecuResidualRear = std::vector<double>{};
    auto clamped = std::size_t{0};
    auto samples = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& wheel = sample.wheels[index];
            const auto inertia = calibration.inertia;

            const auto mid = wheel.wheelSpeedBefore + (wheel.roadTorque + wheel.driveTorque) / inertia * tick;
            const auto braking = wheel.brakeTorqueCommanded * wheel.fade;
            const auto commanded = braking + wheel.rollingTorque;
            auto predicted = mid;

            if (commanded > 0.0 && std::abs(mid) > 0.0)
            {
                const auto arresting = std::abs(mid) * inertia / tick;
                const auto applied = std::min(commanded, arresting);
                predicted = mid - std::copysign(applied / inertia * tick, mid);
                clamped += commanded > arresting ? 1 : 0;
            }

            exactResidual.push_back(inertia * (predicted - wheel.wheelSpeedAfter) / tick);

            // The ECU's own form, run forward as an estimate of the road's spin-up torque.
            const auto measured = inertia * (wheel.wheelSpeedAfter - wheel.wheelSpeedBefore) / tick;
            const auto estimate = wheel.brakeTorqueCommanded + measured;

            ecuResidual.push_back(estimate - wheel.roadTorque);
            if (index >= 2)
            {
                ecuResidualRear.push_back(estimate - wheel.roadTorque);
            }

            samples++;
        }
    }

    std::printf("\n=== residuals over %zu wheel-ticks of one full-pedal ABS stop ===\n", samples);
    distributionHeader("residual [N.m]");
    distributionRow("EXACT reconstruction", distributionOf(exactResidual));
    distributionRow("ECU form, all four wheels", distributionOf(ecuResidual));
    distributionRow("ECU form, rear axle only", distributionOf(ecuResidualRear));

    std::printf("\n  ticks where the arrest clamp fired (wheel held rather than slid): %zu of %zu\n", clamped, samples);
    std::printf("\n  The EXACT row is the identity above; it closes to floating-point rounding, which is\n");
    std::printf("  what says the equation at the head of this file is the one the plant integrates.\n");
    std::printf("  The ECU row is the observable's BIAS: it is the sum of the rolling-resistance\n");
    std::printf("  torque (%.1f N.m at 4000 N here) it cannot know, the fade it cannot know, and the\n",
                setup->corners[0].rollingResistance * 4000.0 * tyreRadius);
    std::printf("  arrest clamp on a held wheel. Sign convention: POSITIVE is a road torque that\n");
    std::printf("  spins the wheel UP, which is what a braking tyre does.\n");

    // **Not to the bit, and the reason is stated rather than absorbed into the bound.** The
    // reconstruction reads the tyre-pressure multiplier on rolling resistance and the disc
    // temperature that feeds fade from the state as it stands AFTER the tick, because that is what
    // a caller outside `stepVehicle` can see; the plant read them from the state as it stood
    // BEFORE. Both advance by one tick of a slow thermal state, which is worth about a hundredth of
    // a per cent of a 15 N.m rolling torque. A milli-newton-metre against wheel torques of
    // thousands is that and nothing else.
    const auto exact = distributionOf(exactResidual);
    REQUIRE(std::abs(exact.maximum) < 1e-3);
    REQUIRE(std::abs(exact.minimum) < 1e-3);
}

namespace
{

// The physically labelled populations of section 4 of the task. **Ground truth labels samples
// offline and never enters a candidate decision.**
struct Populations
{
    // A: the guard says the wheel is out there and the tyre is rolling.
    std::vector<double> falseTruth, falseIdeal, falseSensed, falseEta;
    // B: the guard says the wheel is out there and it genuinely is.
    std::vector<double> realTruth, realIdeal, realSensed, realEta;
    // C / D: loaded wheel, low and high true slip, whatever the guard says.
    std::vector<double> loadedLowTruth, loadedLowIdeal, loadedLowSensed;
    std::vector<double> loadedHighTruth, loadedHighIdeal, loadedHighSensed;
    // E: airborne.
    std::vector<double> airTruth, airIdeal, airSensed, airEta;
    std::vector<double> airBrakeTorque;
    // F / G: the two Recover exits.
    std::vector<double> falseExitTruth, falseExitIdeal, falseExitSensed;
    std::vector<double> goodExitTruth, goodExitIdeal, goodExitSensed;

    // Validity bookkeeping, by phase and overall.
    std::array<std::array<std::size_t, 4>, 5> byPhase{};
    std::array<std::size_t, 4> onRecoverToDump{};
    std::array<std::size_t, 4> onRecoverToReapply{};
    std::array<std::size_t, 4> inDarkTime{};
    std::size_t darkSteps = 0;
    std::size_t steps = 0;

    // The empty-caliper case, section 7.
    std::vector<double> darkTruth, darkIdeal, darkSensed, darkBrakeTorque, darkTrueSlip;

    // Sensor realism, section 9.
    std::vector<double> ageAll, ageAtFalseExit, sensedMinusIdeal, sensedMinusIdealAtFalseExit;
    std::size_t staleAtFalseExit = 0;
    std::size_t falseExits = 0;
    std::size_t airborneExits = 0;
    std::size_t slippingExits = 0;
    std::size_t goodExits = 0;
};

constexpr auto lowSlip = 0.05;
constexpr auto highSlip = 0.20;
constexpr auto loadedFraction = 0.25;
constexpr auto darkPressure = 1.0 * bar;

[[nodiscard]] std::size_t phaseIndex(const ModulatorPhase phase)
{
    switch (phase)
    {
    case ModulatorPhase::Passive:
        return 0;
    case ModulatorPhase::Hold:
        return 1;
    case ModulatorPhase::Dump:
        return 2;
    case ModulatorPhase::Recover:
        return 3;
    case ModulatorPhase::Reapply:
        return 4;
    }

    return 0;
}

void accumulate(Populations& out, const Run& run, const std::size_t channel)
{
    for (const auto& step : run.steps)
    {
        if (step.channel != channel)
        {
            continue;
        }

        out.steps++;

        const auto validity = static_cast<std::size_t>(step.observable.validity);
        out.byPhase[phaseIndex(step.before)][validity]++;
        out.ageAll.push_back(step.sensorAge);
        out.sensedMinusIdeal.push_back(step.observable.sensed - step.observable.ideal);

        const auto slip = std::abs(step.trueSlip);
        const auto loaded = step.trueContact && step.trueLoad > loadedFraction * run.staticRearLoad;
        const auto dark =
            step.pressureBefore < darkPressure && step.trueContact && step.trueLoad > 0.5 * run.staticRearLoad;

        if (step.pastBandUnqualified && slip < lowSlip && step.trueContact)
        {
            out.falseTruth.push_back(step.observable.truth);
            out.falseIdeal.push_back(step.observable.ideal);
            out.falseSensed.push_back(step.observable.sensed);
            if (step.observable.validity == Validity::Valid)
            {
                out.falseEta.push_back(step.observable.etaSensed);
            }
        }

        if (step.pastBandUnqualified && slip > highSlip)
        {
            out.realTruth.push_back(step.observable.truth);
            out.realIdeal.push_back(step.observable.ideal);
            out.realSensed.push_back(step.observable.sensed);
            if (step.observable.validity == Validity::Valid)
            {
                out.realEta.push_back(step.observable.etaSensed);
            }
        }

        if (loaded && slip < lowSlip)
        {
            out.loadedLowTruth.push_back(step.observable.truth);
            out.loadedLowIdeal.push_back(step.observable.ideal);
            out.loadedLowSensed.push_back(step.observable.sensed);
        }

        if (loaded && slip > highSlip)
        {
            out.loadedHighTruth.push_back(step.observable.truth);
            out.loadedHighIdeal.push_back(step.observable.ideal);
            out.loadedHighSensed.push_back(step.observable.sensed);
        }

        if (!step.trueContact)
        {
            out.airTruth.push_back(step.observable.truth);
            out.airIdeal.push_back(step.observable.ideal);
            out.airSensed.push_back(step.observable.sensed);
            out.airBrakeTorque.push_back(step.observable.brakeTorque);
            if (step.observable.validity == Validity::Valid)
            {
                out.airEta.push_back(step.observable.etaSensed);
            }
        }

        if (dark)
        {
            out.darkSteps++;
            out.inDarkTime[validity]++;
            out.darkTruth.push_back(step.observable.truth);
            out.darkIdeal.push_back(step.observable.ideal);
            out.darkSensed.push_back(step.observable.sensed);
            out.darkBrakeTorque.push_back(step.observable.brakeTorque);
            out.darkTrueSlip.push_back(slip);
        }

        if (step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Dump)
        {
            out.onRecoverToDump[validity]++;

            out.airborneExits += step.trueContact ? 0 : 1;
            out.slippingExits += (step.trueContact && slip > highSlip) ? 1 : 0;

            if (slip < lowSlip && step.trueContact)
            {
                out.falseExits++;
                out.staleAtFalseExit += step.observable.validity == Validity::SensorStale ? 1 : 0;
                out.falseExitTruth.push_back(step.observable.truth);
                out.falseExitIdeal.push_back(step.observable.ideal);
                out.falseExitSensed.push_back(step.observable.sensed);
                out.ageAtFalseExit.push_back(step.sensorAge);
                out.sensedMinusIdealAtFalseExit.push_back(step.observable.sensed - step.observable.ideal);
            }
        }

        if (step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Reapply)
        {
            out.onRecoverToReapply[validity]++;
            out.goodExits++;
            out.goodExitTruth.push_back(step.observable.truth);
            out.goodExitIdeal.push_back(step.observable.ideal);
            out.goodExitSensed.push_back(step.observable.sensed);
        }
    }
}

[[nodiscard]] Populations gather(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                                 const RoadTorqueCalibration& calibration, const std::size_t channel,
                                 const double pedal = 1.0)
{
    auto out = Populations{};
    const auto off = Candidate{};

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto run =
            record(setup, world, assists, calibration, off, pedal, memberEntry(index, ensembleCount), 0.0, true);
        REQUIRE(run.stopped);
        accumulate(out, run, channel);
    }

    return out;
}

void printValidityRow(const char* name, const std::array<std::size_t, 4>& counts)
{
    const auto total = counts[0] + counts[1] + counts[2] + counts[3];
    const auto share = [total](const std::size_t value)
    {
        return total > 0 ? 100.0 * static_cast<double>(value) / static_cast<double>(total) : 0.0;
    };

    std::printf("  %-24s %8zu   %7.2f%%   %7.2f%%   %7.2f%%   %7.2f%%\n", name, total, share(counts[0]),
                share(counts[1]), share(counts[2]), share(counts[3]));
}

} // namespace

// =============================================================================================
// 3. LOW BRAKE TORQUE, AND WHETHER NORMALISED ETA IS DEFINED WHERE IT IS NEEDED
// =============================================================================================

TEST_CASE("how often the normalised ratio is defined, phase by phase and through the dark time", "[.road-torque]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    const auto rear = gather(setup.value(), world.value(), assists, calibration, 2);

    std::printf("\n=== validity of eta = T_road_hat / T_brake_cmd, REAR channel, %zu members ===\n", ensembleCount);
    std::printf("  LOW_TORQUE_INVALID  brake torque <= %.0f%% of this corner's peak (%.1f N.m)\n",
                100.0 * lowTorqueFraction, lowTorqueFraction * calibration.peakTorque[2]);
    std::printf("  SENSOR_STALE        no new tooth for more than %.0f ms\n", 1000.0 * staleAge);
    std::printf("  OTHER_INVALID       no reference, or no wheel-speed measurement at all\n");
    std::printf("  Precedence is OTHER > STALE > LOW_TORQUE, so the shares below do not double count.\n");

    std::printf("\n  %-24s %8s   %8s   %8s   %8s   %8s\n", "population", "steps", validityName(Validity::Valid),
                "LOW_TORQ", "STALE", "OTHER");
    std::printf("  (the four columns are %s, %s, %s and %s, as shares of the row)\n", validityName(Validity::Valid),
                validityName(Validity::LowTorqueInvalid), validityName(Validity::SensorStale),
                validityName(Validity::OtherInvalid));

    printValidityRow("phase Passive", rear.byPhase[0]);
    printValidityRow("phase Hold", rear.byPhase[1]);
    printValidityRow("phase Dump", rear.byPhase[2]);
    printValidityRow("phase Recover", rear.byPhase[3]);
    printValidityRow("phase Reapply", rear.byPhase[4]);
    printValidityRow("on Recover -> Dump", rear.onRecoverToDump);
    printValidityRow("on Recover -> Reapply", rear.onRecoverToReapply);
    printValidityRow("in the DARK TIME", rear.inDarkTime);

    std::printf("\n  DARK TIME is the state the mechanism is about: rear pressure below %.0f bar with the\n",
                darkPressure / bar);
    std::printf("  control wheel in contact and carrying more than half its static load. %zu steps of\n",
                rear.darkSteps);
    std::printf("  %zu (%.2f%%) on the rear channel.\n", rear.steps,
                100.0 * static_cast<double>(rear.darkSteps) /
                    static_cast<double>(std::max(rear.steps, std::size_t{1})));

    distributionHeader("in the dark time");
    distributionRow("commanded brake torque [N.m]", distributionOf(rear.darkBrakeTorque));
    distributionRow("true slip []", distributionOf(rear.darkTrueSlip));
    distributionRow("TRUTH  road torque [N.m]", distributionOf(rear.darkTruth));
    distributionRow("IDEAL  road torque [N.m]", distributionOf(rear.darkIdeal));
    distributionRow("SENSED road torque [N.m]", distributionOf(rear.darkSensed));

    const auto darkValid = rear.inDarkTime[0];
    const auto darkTotal = rear.inDarkTime[0] + rear.inDarkTime[1] + rear.inDarkTime[2] + rear.inDarkTime[3];

    std::printf("\n  VERDICT ON NORMALISED ETA: it is defined on %zu of %zu dark-time steps (%.2f%%).\n", darkValid,
                darkTotal,
                darkTotal > 0 ? 100.0 * static_cast<double>(darkValid) / static_cast<double>(darkTotal) : 0.0);
    std::printf("  Nothing is rescued with an epsilon anywhere in this file.\n");

    // The floor is a reporting parameter, so its sweep has to be printed rather than argued.
    std::printf("\n  --- and the same count against the floor it is declared invalid below ---\n");
    std::printf("  floor as %% of peak   floor N.m    dark-time steps with T_brake above it\n");

    for (const auto fraction : {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25})
    {
        const auto floor = fraction * calibration.peakTorque[2];
        auto above = std::size_t{0};

        for (const auto torque : rear.darkBrakeTorque)
        {
            above += torque > floor ? 1 : 0;
        }

        std::printf("  %14.3f%%   %9.2f    %zu of %zu  (%.2f%%)\n", 100.0 * fraction, floor, above,
                    rear.darkBrakeTorque.size(),
                    rear.darkBrakeTorque.empty()
                        ? 0.0
                        : 100.0 * static_cast<double>(above) / static_cast<double>(rear.darkBrakeTorque.size()));
    }
}

// =============================================================================================
// 4. THE INFORMATION, NOT A THRESHOLD
// =============================================================================================

TEST_CASE("how well each observable separates a false pastBand from a real one", "[.road-torque]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    const auto rear = gather(setup.value(), world.value(), assists, calibration, 2);

    std::printf("\n=== the labelled populations, REAR channel, %zu members ===\n", ensembleCount);
    std::printf("  Ground truth LABELS these samples. It enters no candidate decision anywhere.\n");
    std::printf("  Sign convention: POSITIVE road torque spins the wheel UP, which is what a braking\n");
    std::printf("  tyre does. A free-rolling wheel on a decelerating car sits at slightly POSITIVE\n");
    std::printf("  slip and its road torque is NEGATIVE -- the road is de-spinning it.\n");

    std::printf("\n  A  guard past band, true slip < %.2f, in contact      %8zu steps\n", lowSlip,
                rear.falseSensed.size());
    std::printf("  B  guard past band, true slip > %.2f                  %8zu steps\n", highSlip,
                rear.realSensed.size());
    std::printf("  C  loaded wheel, true slip < %.2f                     %8zu steps\n", lowSlip,
                rear.loadedLowSensed.size());
    std::printf("  D  loaded wheel, true slip > %.2f                     %8zu steps\n", highSlip,
                rear.loadedHighSensed.size());
    std::printf("  E  airborne wheel                                     %8zu steps\n", rear.airSensed.size());
    std::printf("  F  Recover -> Dump, IN CONTACT, true slip < %.2f      %8zu steps\n", lowSlip, rear.falseExits);
    std::printf("     ...of the same transitions, AIRBORNE                %8zu steps\n", rear.airborneExits);
    std::printf("     ...and genuinely slipping past %.2f                 %8zu steps\n", highSlip, rear.slippingExits);
    std::printf("  G  Recover -> Reapply                                 %8zu steps\n", rear.goodExits);

    distributionHeader("A: false pastBand [N.m]");
    distributionRow("TRUTH", distributionOf(rear.falseTruth));
    distributionRow("IDEAL kinematic", distributionOf(rear.falseIdeal));
    distributionRow("SENSOR realistic", distributionOf(rear.falseSensed));

    distributionHeader("B: real excessive slip [N.m]");
    distributionRow("TRUTH", distributionOf(rear.realTruth));
    distributionRow("IDEAL kinematic", distributionOf(rear.realIdeal));
    distributionRow("SENSOR realistic", distributionOf(rear.realSensed));

    distributionHeader("C: loaded, low true slip [N.m]");
    distributionRow("TRUTH", distributionOf(rear.loadedLowTruth));
    distributionRow("IDEAL kinematic", distributionOf(rear.loadedLowIdeal));
    distributionRow("SENSOR realistic", distributionOf(rear.loadedLowSensed));

    distributionHeader("D: loaded, high true slip [N.m]");
    distributionRow("TRUTH", distributionOf(rear.loadedHighTruth));
    distributionRow("IDEAL kinematic", distributionOf(rear.loadedHighIdeal));
    distributionRow("SENSOR realistic", distributionOf(rear.loadedHighSensed));

    distributionHeader("F: false Recover->Dump [N.m]");
    distributionRow("TRUTH", distributionOf(rear.falseExitTruth));
    distributionRow("IDEAL kinematic", distributionOf(rear.falseExitIdeal));
    distributionRow("SENSOR realistic", distributionOf(rear.falseExitSensed));

    distributionHeader("G: Recover->Reapply [N.m]");
    distributionRow("TRUTH", distributionOf(rear.goodExitTruth));
    distributionRow("IDEAL kinematic", distributionOf(rear.goodExitIdeal));
    distributionRow("SENSOR realistic", distributionOf(rear.goodExitSensed));

    distributionHeader("normalised eta, where DEFINED");
    distributionRow("A: false pastBand", distributionOf(rear.falseEta));
    distributionRow("B: real excessive slip", distributionOf(rear.realEta));
    distributionRow("E: airborne", distributionOf(rear.airEta));

    // --- threshold-independent separation ---
    std::printf("\n=== threshold-independent separation, AUC of `B is above A` ===\n");
    std::printf("  0.500 is no information. 1.000 is perfect. Below 0.500 the separation is real and\n");
    std::printf("  the polarity is the other way round.\n\n");
    std::printf("  %-34s %8s   %8s   %8s\n", "discrimination", "TRUTH", "IDEAL", "SENSED");

    const auto row = [](const char* name, const double truth, const double ideal, const double sensed)
    {
        std::printf("  %-34s %8.4f   %8.4f   %8.4f\n", name, truth, ideal, sensed);
    };

    row("B over A (the important one)", areaUnderCurve(rear.realTruth, rear.falseTruth),
        areaUnderCurve(rear.realIdeal, rear.falseIdeal), areaUnderCurve(rear.realSensed, rear.falseSensed));
    row("D over C (loaded, slip only)", areaUnderCurve(rear.loadedHighTruth, rear.loadedLowTruth),
        areaUnderCurve(rear.loadedHighIdeal, rear.loadedLowIdeal),
        areaUnderCurve(rear.loadedHighSensed, rear.loadedLowSensed));
    row("D over E (loaded slip over air)", areaUnderCurve(rear.loadedHighTruth, rear.airTruth),
        areaUnderCurve(rear.loadedHighIdeal, rear.airIdeal), areaUnderCurve(rear.loadedHighSensed, rear.airSensed));
    row("C over E (rolling over air)", areaUnderCurve(rear.loadedLowTruth, rear.airTruth),
        areaUnderCurve(rear.loadedLowIdeal, rear.airIdeal), areaUnderCurve(rear.loadedLowSensed, rear.airSensed));
    row("G over F (the two Recover exits)", areaUnderCurve(rear.goodExitTruth, rear.falseExitTruth),
        areaUnderCurve(rear.goodExitIdeal, rear.falseExitIdeal),
        areaUnderCurve(rear.goodExitSensed, rear.falseExitSensed));

    std::printf("\n  and the same for normalised eta, on the steps where it is DEFINED:\n");
    std::printf("  B over A, eta (SENSED)             %8.4f   on %zu / %zu samples\n",
                areaUnderCurve(rear.realEta, rear.falseEta), rear.realEta.size() + rear.falseEta.size(),
                rear.realSensed.size() + rear.falseSensed.size());

    // --- the best single operating point, as evidence about the separation ---
    const auto best = bestOperating(rear.realSensed, rear.falseSensed);

    std::printf("\n=== the best single threshold on the SENSOR-REALISTIC road torque, B against A ===\n");
    std::printf("  **Reported as evidence about the separation. It is not a proposal and section 11\n");
    std::printf("  sweeps the region rather than taking it.**\n");
    std::printf("  threshold %.2f N.m (%.4f of the rear corner's peak brake torque)\n", best.threshold,
                best.threshold / std::max(calibration.peakTorque[2], 1e-9));
    std::printf("  keeps %.2f%% of B (real slip) and %.2f%% of A (false pastBand); Youden %.4f\n",
                100.0 * best.truePositive, 100.0 * best.falsePositive, best.youden);
    std::printf("\n  POLARITY, read off the medians above rather than assumed: a wheel in a REAL\n");
    std::printf("  excessive-slip state carries a LARGE positive road torque, and a wheel whose\n");
    std::printf("  pastBand is FALSE carries a small or negative one. So the qualification that\n");
    std::printf("  keeps pastBand asserted is `road torque estimate ABOVE the threshold`.\n");
}

// =============================================================================================
// 6. THE AIRBORNE CLAIM
// =============================================================================================

TEST_CASE("whether the observable really tends to zero on an airborne wheel", "[.road-torque]")
{
    // The brief's section 17 claims eta "is zero exactly when the wheel is off the road". That is a
    // statement about the equation and not about the implementation, and this is the measurement.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    const auto rear = gather(setup.value(), world.value(), assists, calibration, 2);

    std::printf("\n=== the airborne wheel, REAR channel, %zu airborne controller steps ===\n", rear.airSensed.size());

    distributionHeader("airborne [N.m]");
    distributionRow("TRUTH  road torque", distributionOf(rear.airTruth));
    distributionRow("IDEAL  road torque", distributionOf(rear.airIdeal));
    distributionRow("SENSED road torque", distributionOf(rear.airSensed));
    distributionRow("commanded brake torque", distributionOf(rear.airBrakeTorque));

    distributionHeader("airborne eta, where DEFINED");
    distributionRow("SENSED eta", distributionOf(rear.airEta));

    const auto truth = distributionOf(rear.airTruth);
    const auto ideal = distributionOf(rear.airIdeal);
    const auto sensed = distributionOf(rear.airSensed);

    std::printf("\n  TRUTH is EXACTLY zero on an airborne wheel by construction: `evaluateTyre` returns\n");
    std::printf("  a zeroed force at `verticalLoad <= 0`, so `-Fx * r` is identically 0. Measured\n");
    std::printf("  min %.6e, max %.6e.\n", truth.minimum, truth.maximum);
    std::printf("\n  IDEAL is NOT zero: median %.3f, P10-P90 %.3f to %.3f, min-max %.3f to %.3f N.m.\n", ideal.median,
                ideal.p10, ideal.p90, ideal.minimum, ideal.maximum);
    std::printf("  SENSED is NOT zero: median %.3f, P10-P90 %.3f to %.3f, min-max %.3f to %.3f N.m.\n", sensed.median,
                sensed.p10, sensed.p90, sensed.minimum, sensed.maximum);
    std::printf("\n  The deviations are the terms the ECU form drops, and each is nameable:\n");
    std::printf("   - ROLLING RESISTANCE rides in the same arrest clamp as the brake and is\n");
    std::printf("     proportional to Fz, so it vanishes with the load and contributes nothing here.\n");
    std::printf("   - RESIDUAL BRAKE TORQUE does not: the caliper carries whatever pressure the\n");
    std::printf("     channel left in it, median %.2f N.m over these steps, and the wheel then spins\n",
                distributionOf(rear.airBrakeTorque).median);
    std::printf("     DOWN at exactly `-T_brake / I`, which the estimate reconstructs as zero only if\n");
    std::printf("     the acceleration term is exact.\n");
    std::printf("   - SENSOR QUANTISATION and STALE PULSES are what make it inexact: the tone ring's\n");
    std::printf("     derived acceleration is a difference across one tooth interval, and a wheel\n");
    std::printf("     turning slowly between teeth carries that error into `I * dw/dt` directly.\n");
    std::printf("   - THE ARREST CLAMP holds a stopped wheel: the plant applies only as much torque\n");
    std::printf("     as brings it to rest, and the ECU form still subtracts the whole command.\n");
    std::printf("   - DRIVELINE TORQUE is zero on this fixture (neutral, `noDriveTorque`).\n");
    std::printf("\n  So `eta = 0 airborne` holds for TRUTH and is an approximation at both realistic\n");
    std::printf("  levels. It is also the WRONG QUESTION for this mechanism: the failure is a loaded\n");
    std::printf("  rolling wheel, not an airborne one, and section 11 of `docs/braking-chain-brief.md`\n");
    std::printf("  already measured every contact-gated oracle doing nothing.\n");
}

// =============================================================================================
// 7. THE EMPTY-CALIPER CASE
// =============================================================================================

TEST_CASE("what the observable can say when the caliper is empty", "[.road-torque]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    const auto rear = gather(setup.value(), world.value(), assists, calibration, 2);

    // The dark time split by what the tyre is actually doing, which is the question: with the
    // caliper empty, can anything separate "rolling, recovered" from "still slipping"?
    auto darkLow = std::vector<double>{};
    auto darkHigh = std::vector<double>{};
    auto darkLowEta = std::vector<double>{};
    auto darkHighEta = std::vector<double>{};

    for (auto index = std::size_t{0}; index < rear.darkSensed.size(); index++)
    {
        if (rear.darkTrueSlip[index] < lowSlip)
        {
            darkLow.push_back(rear.darkSensed[index]);
        }
        else if (rear.darkTrueSlip[index] > highSlip)
        {
            darkHigh.push_back(rear.darkSensed[index]);
        }

        if (rear.darkBrakeTorque[index] > 0.0)
        {
            const auto eta = rear.darkSensed[index] / rear.darkBrakeTorque[index];
            if (rear.darkTrueSlip[index] < lowSlip)
            {
                darkLowEta.push_back(eta);
            }
            else if (rear.darkTrueSlip[index] > highSlip)
            {
                darkHighEta.push_back(eta);
            }
        }
    }

    std::printf("\n=== the empty-caliper case: %zu dark-time steps on the rear channel ===\n", rear.darkSteps);
    std::printf("  DARK TIME: rear pressure below %.0f bar, control wheel in contact, carrying more\n",
                darkPressure / bar);
    std::printf("  than half its static load. This is the state the diagnosed failure lives in.\n");

    distributionHeader("dark time, split by TRUE slip");
    distributionRow("SENSED road torque, slip < 0.05", distributionOf(darkLow));
    distributionRow("SENSED road torque, slip > 0.20", distributionOf(darkHigh));
    distributionRow("eta, slip < 0.05 (where defined)", distributionOf(darkLowEta));
    distributionRow("eta, slip > 0.20 (where defined)", distributionOf(darkHighEta));

    std::printf("\n  separation in the dark time, AUC of `high slip is above low slip`:\n");
    std::printf("    UNNORMALISED road torque   %8.4f   on %zu / %zu samples\n", areaUnderCurve(darkHigh, darkLow),
                darkHigh.size() + darkLow.size(), rear.darkSteps);
    std::printf("    NORMALISED eta             %8.4f   on %zu / %zu samples\n",
                areaUnderCurve(darkHighEta, darkLowEta), darkHighEta.size() + darkLowEta.size(), rear.darkSteps);

    std::printf("\n  ANSWER: with the brake torque at or near zero the ratio has no denominator, so\n");
    std::printf("  the normalised form carries information on a %.2f%% subset of the state it is\n",
                100.0 * static_cast<double>(darkHighEta.size() + darkLowEta.size()) /
                    static_cast<double>(std::max(rear.darkSteps, std::size_t{1})));
    std::printf("  needed in, while the UNNORMALISED road-torque estimate is defined on all of it --\n");
    std::printf("  it degenerates to `I * dw/dt`, which is a direct measurement of what the road is\n");
    std::printf("  doing to a wheel nothing else is touching.\n");
}

// =============================================================================================
// 5. TIMING
// =============================================================================================

namespace
{

struct Timing
{
    bool found = false;
    std::size_t decision = 0;
    double time = 0.0;
    // How many consecutive controller steps BEFORE the decision already read below the threshold.
    // The controller runs at 1 kHz, so a step is a millisecond.
    std::size_t lead = 0;
    bool atDecision = false;
    std::size_t firstAfter = 0;
};

[[nodiscard]] Timing leadOf(const std::vector<Step>& steps, const double threshold, const bool ideal)
{
    auto timing = Timing{};
    const auto value = [ideal](const Step& step)
    {
        return ideal ? step.observable.ideal : step.observable.sensed;
    };

    for (auto index = std::size_t{0}; index < steps.size(); index++)
    {
        const auto& step = steps[index];

        if (!(step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Dump &&
              std::abs(step.trueSlip) < lowSlip && step.trueContact))
        {
            continue;
        }

        timing.found = true;
        timing.decision = index;
        timing.time = step.time;
        timing.atDecision = value(step) < threshold;

        auto back = index;
        while (back > 0 && value(steps[back - 1]) < threshold)
        {
            timing.lead++;
            back--;
        }

        for (auto forward = index; forward < steps.size(); forward++)
        {
            if (value(steps[forward]) < threshold)
            {
                timing.firstAfter = forward - index;
                break;
            }
        }

        break;
    }

    return timing;
}

[[nodiscard]] std::vector<Step> rearSteps(const Run& run)
{
    auto steps = std::vector<Step>{};

    for (const auto& step : run.steps)
    {
        if (step.channel == 2)
        {
            steps.push_back(step);
        }
    }

    return steps;
}

} // namespace

TEST_CASE("when the information arrives, relative to the first erroneous Recover -> Dump", "[.road-torque]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto off = Candidate{};

    // The representative members `[.rear-hop]` established: 27 clearly recovered, 4 clearly
    // stranded, and 0/1 the nearest cluster-changing adjacent pair.
    const auto chosen = std::array<std::size_t, 4>{27, 4, 0, 1};
    const auto label =
        std::array<const char*, 4>{"RECOVERED (27)", "STRANDED (4)", "boundary lo (0)", "boundary hi (1)"};

    std::printf("\n=== the observable around the first FALSE Recover -> Dump ===\n");
    std::printf("  FALSE means the wheel is IN CONTACT and its true slip at the transition is below\n");
    std::printf("  %.2f -- an airborne wheel reads zero slip too, and dumping an already empty\n", lowSlip);
    std::printf("  caliper on one costs nothing. The controller clock is 1 kHz, so one\n");
    std::printf("  controller step is one millisecond.\n");

    for (auto slot = std::size_t{0}; slot < chosen.size(); slot++)
    {
        const auto run = record(setup.value(), world.value(), assists, calibration, off, 1.0,
                                memberEntry(chosen[slot], ensembleCount), 0.0, true);
        REQUIRE(run.stopped);

        const auto steps = rearSteps(run);
        const auto use = axleUse(run);
        const auto timing = leadOf(steps, 100.0, false);

        std::printf("\n  --- member %zu, %s: %.3f m, rear utilisation %.4f ---\n", chosen[slot], label[slot],
                    run.distance, use.rear);

        if (!timing.found)
        {
            std::printf("    no FALSE Recover -> Dump in this member.\n");
            continue;
        }

        std::printf("    first false Recover -> Dump at t = %.4f s (rear-channel step %zu of %zu)\n", timing.time,
                    timing.decision, steps.size());
        std::printf("\n    %8s %8s %9s %8s %8s %9s %9s %9s %9s %9s %8s\n", "t [s]", "phase", "bar", "guard", "trueSlip",
                    "acc m/s2", "excess", "T_brake", "SENSED", "IDEAL", "Fz N");

        const auto first = timing.decision > 24 ? timing.decision - 24 : std::size_t{0};
        const auto last = std::min(timing.decision + 8, steps.size() - 1);

        for (auto index = first; index <= last; index++)
        {
            const auto& step = steps[index];
            std::printf("    %8.4f %8s %9.2f %8.4f %8.4f %9.2f %9.2f %9.2f %9.2f %9.2f %8.0f%s\n", step.time,
                        modulatorName(step.before), step.pressureBefore / bar, step.guardSlip, step.trueSlip,
                        step.acceleration, step.excess, step.observable.brakeTorque, step.observable.sensed,
                        step.observable.ideal, step.trueLoad, index == timing.decision ? "  <== DECISION" : "");
        }

        std::printf("\n    lead, by threshold on the SENSOR-REALISTIC road torque:\n");
        std::printf("      threshold N.m   consecutive steps already below, before the decision   at decision\n");

        for (const auto threshold : {25.0, 50.0, 100.0, 200.0, 400.0})
        {
            const auto measured = leadOf(steps, threshold, false);
            std::printf("      %13.1f   %8zu steps = %8.1f ms                        %s\n", threshold, measured.lead,
                        1000.0 * static_cast<double>(measured.lead) / assists.controlRate,
                        measured.atDecision ? "YES" : "no");
        }

        std::printf("\n    and on the IDEAL kinematic form:\n");
        for (const auto threshold : {25.0, 50.0, 100.0, 200.0, 400.0})
        {
            const auto measured = leadOf(steps, threshold, true);
            std::printf("      %13.1f   %8zu steps = %8.1f ms                        %s\n", threshold, measured.lead,
                        1000.0 * static_cast<double>(measured.lead) / assists.controlRate,
                        measured.atDecision ? "YES" : "no");
        }
    }

    // And the same over the whole ensemble, so the classification is not read off four members.
    std::printf("\n=== every FALSE Recover -> Dump in the ensemble, classified ===\n");
    std::printf("  %10s   %12s   %12s   %12s   %10s\n", "threshold", "BEFORE", "AT ONLY", "AFTER ONLY", "never");

    auto traces = std::vector<std::vector<Step>>{};
    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        traces.push_back(rearSteps(record(setup.value(), world.value(), assists, calibration, off, 1.0,
                                          memberEntry(index, ensembleCount), 0.0, true)));
    }

    for (const auto threshold : {25.0, 50.0, 100.0, 200.0, 400.0})
    {
        auto before = std::size_t{0};
        auto at = std::size_t{0};
        auto after = std::size_t{0};
        auto never = std::size_t{0};

        for (const auto& steps : traces)
        {
            for (auto position = std::size_t{0}; position < steps.size(); position++)
            {
                const auto& step = steps[position];

                if (!(step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Dump &&
                      std::abs(step.trueSlip) < lowSlip && step.trueContact))
                {
                    continue;
                }

                const auto belowNow = step.observable.sensed < threshold;
                const auto belowBefore = position > 0 && steps[position - 1].observable.sensed < threshold;

                if (belowBefore)
                {
                    before++;
                }
                else if (belowNow)
                {
                    at++;
                }
                else
                {
                    auto later = false;
                    for (auto forward = position + 1; forward < std::min(position + 50, steps.size()); forward++)
                    {
                        later = later || steps[forward].observable.sensed < threshold;
                    }
                    after += later ? 1 : 0;
                    never += later ? 0 : 1;
                }
            }
        }

        std::printf("  %10.1f   %12zu   %12zu   %12zu   %10zu\n", threshold, before, at, after, never);
    }

    std::printf("\n  BEFORE   the observable was already below the threshold on the PREVIOUS step, so\n");
    std::printf("           the qualification would have fired before the erroneous decision.\n");
    std::printf("  AT ONLY  it crosses on the decision step itself -- still usable, because the\n");
    std::printf("           qualification is evaluated before the state machine runs.\n");
    std::printf("  AFTER    it only becomes informative later, which cannot solve the decision.\n");
}

// =============================================================================================
// 9. SENSOR REALISM
// =============================================================================================

TEST_CASE("what the tone ring does to the observable around the false decisions", "[.road-torque]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    const auto rear = gather(setup.value(), world.value(), assists, calibration, 2);

    std::printf("\n=== the tone ring, REAR channel, %zu controller steps ===\n", rear.steps);
    std::printf("  ring: %u poles, %.0e s capture timer, no edge jitter and no damaged teeth modelled\n",
                assists.toneRing.teeth, assists.toneRing.timerResolution);
    std::printf("  The controller updates its speed and acceleration ONLY on a tooth crossing; between\n");
    std::printf("  crossings it holds both, which is what `WheelSpeedReading::age` reports.\n");

    distributionHeader("pulse age [s]");
    distributionRow("every rear-channel step", distributionOf(rear.ageAll));
    distributionRow("at a FALSE Recover -> Dump", distributionOf(rear.ageAtFalseExit));

    distributionHeader("SENSED - IDEAL [N.m]");
    distributionRow("every rear-channel step", distributionOf(rear.sensedMinusIdeal));
    distributionRow("at a FALSE Recover -> Dump", distributionOf(rear.sensedMinusIdealAtFalseExit));

    auto fresh = std::size_t{0};
    auto spikes = std::size_t{0};
    for (auto index = std::size_t{0}; index < rear.ageAll.size(); index++)
    {
        fresh += rear.ageAll[index] == 0.0 ? 1 : 0;
        spikes += std::abs(rear.sensedMinusIdeal[index]) > 200.0 ? 1 : 0;
    }

    std::printf("\n  steps carrying a FRESH tooth crossing        %zu of %zu  (%.2f%%)\n", fresh, rear.ageAll.size(),
                100.0 * static_cast<double>(fresh) / static_cast<double>(std::max(rear.ageAll.size(), std::size_t{1})));
    std::printf("  steps where SENSED and IDEAL differ by > 200 N.m  %zu  (%.2f%%)\n", spikes,
                100.0 * static_cast<double>(spikes) /
                    static_cast<double>(std::max(rear.ageAll.size(), std::size_t{1})));
    std::printf("  FALSE Recover -> Dump transitions classified SENSOR_STALE  %zu of %zu\n", rear.staleAtFalseExit,
                rear.falseExits);

    const auto sensedGap = distributionOf(rear.sensedMinusIdeal);
    std::printf("\n  CONSEQUENCE. The acceleration term is `I / r_nominal` times a difference of two\n");
    std::printf("  speeds across one tooth interval, and %.4f / %.4f = %.2f N.m per (m/s^2) of\n", calibration.inertia,
                calibration.radius, calibration.inertia / calibration.radius);
    std::printf("  peripheral acceleration error. One microsecond of timer at 100 km/h is about\n");
    std::printf("  5.6 m/s^2 by the calibration comment on `AntilockSetup::lockDeceleration`, so the\n");
    std::printf("  quantisation floor alone is of the order of %.0f N.m -- which is the SAME ORDER as\n",
                5.6 * calibration.inertia / calibration.radius);
    std::printf("  the separation section 4 measures. The spread above is the measurement of it:\n");
    std::printf("  P10-P90 %.2f to %.2f N.m, sd %.2f.\n", sensedGap.p10, sensedGap.p90, sensedGap.deviation);
    std::printf("\n  FILTER REQUIRED: yes at the tail. No filter constant is selected here.\n");
}

// =============================================================================================
// 8. THE DRIVEN WHEEL
// =============================================================================================

TEST_CASE("the driveline term, and whether the observable generalises off an undriven axle", "[.road-torque]")
{
    // The Golf's failure is on its UNDRIVEN rear axle and the whole fixture is in neutral, so this
    // case makes the driveline term real by putting one there. `stepVehicle` takes the per-wheel
    // drive torques as an argument, which is the same seam the driveline uses.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto off = Candidate{};
    const auto entry = memberEntry(ensembleCount / 2, ensembleCount);

    std::printf("\n=== the driveline term ===\n");
    std::printf("  From the equation at the head of this file, with the arrest clamp not firing:\n\n");
    std::printf("    T_road = I * dw/dt + T_brake + T_rolling - T_drive\n\n");
    std::printf("  so a driven wheel needs the drive torque SUBTRACTED from the estimate. It enters\n");
    std::printf("  with the opposite sign to the brake, which is what makes an undriven axle the easy\n");
    std::printf("  case: `T_drive` is identically zero there and drops out.\n");

    struct Arm
    {
        const char* name;
        std::array<double, cornerCount> drive;
    };

    const auto arms = std::array<Arm, 3>{Arm{"neutral (the shipped fixture)", {0.0, 0.0, 0.0, 0.0}},
                                         Arm{"front axle, -200 N.m each (engine braking)", {-200.0, -200.0, 0.0, 0.0}},
                                         Arm{"front axle, +200 N.m each (drive)", {200.0, 200.0, 0.0, 0.0}}};

    std::printf("\n  %-44s %10s %10s %10s %10s\n", "arm", "stop m", "front bias", "rear bias", "corrected");
    std::printf("  %-44s %10s %10s %10s %10s\n", "", "", "N.m med", "N.m med", "front N.m");

    for (const auto& arm : arms)
    {
        const auto run =
            record(setup.value(), world.value(), assists, calibration, off, 1.0, entry, 0.0, false, arm.drive);
        REQUIRE(run.stopped);

        auto front = std::vector<double>{};
        auto rear = std::vector<double>{};
        auto corrected = std::vector<double>{};

        for (const auto& sample : run.ticks)
        {
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& wheel = sample.wheels[index];
                const auto measured = calibration.inertia * (wheel.wheelSpeedAfter - wheel.wheelSpeedBefore) / tick;
                const auto estimate = wheel.brakeTorqueCommanded + measured;
                const auto bias = estimate - wheel.roadTorque;

                if (index < 2)
                {
                    front.push_back(bias);
                    corrected.push_back(estimate - wheel.driveTorque - wheel.roadTorque);
                }
                else
                {
                    rear.push_back(bias);
                }
            }
        }

        std::printf("  %-44s %10.3f %10.3f %10.3f %10.3f\n", arm.name, run.distance, distributionOf(front).median,
                    distributionOf(rear).median, distributionOf(corrected).median);
    }

    std::printf("\n  The `corrected` column subtracts the KNOWN drive torque and recovers the undriven\n");
    std::printf("  bias, which is what says the term enters linearly and nothing else changes.\n");

    // Front against rear on the standard neutral ensemble, which is the comparison the task asks
    // for: same car, same stop, two axles that differ in load, brake share and cycling.
    const auto frontPopulation = gather(setup.value(), world.value(), assists, calibration, 0);
    const auto rearPopulation = gather(setup.value(), world.value(), assists, calibration, 2);

    std::printf("\n=== front channel against rear, same %zu stops, both UNDRIVEN (neutral) ===\n", ensembleCount);

    distributionHeader("SENSED road torque [N.m]");
    distributionRow("FRONT: guard past band, slip<0.05", distributionOf(frontPopulation.falseSensed));
    distributionRow("FRONT: guard past band, slip>0.20", distributionOf(frontPopulation.realSensed));
    distributionRow("REAR:  guard past band, slip<0.05", distributionOf(rearPopulation.falseSensed));
    distributionRow("REAR:  guard past band, slip>0.20", distributionOf(rearPopulation.realSensed));

    std::printf("\n  AUC of `real slip is above false pastBand`, SENSOR-realistic:\n");
    std::printf("    front channel   %8.4f   (%zu / %zu samples)\n",
                areaUnderCurve(frontPopulation.realSensed, frontPopulation.falseSensed),
                frontPopulation.realSensed.size(), frontPopulation.falseSensed.size());
    std::printf("    rear channel    %8.4f   (%zu / %zu samples)\n",
                areaUnderCurve(rearPopulation.realSensed, rearPopulation.falseSensed), rearPopulation.realSensed.size(),
                rearPopulation.falseSensed.size());
    std::printf("\n  The front axle carries roughly five times the rear's brake torque on this car, so a\n");
    std::printf("  threshold in N.m does NOT carry across the two axles of one vehicle, let alone\n");
    std::printf("  across vehicles. The peak-brake-torque normalisation used everywhere below is what\n");
    std::printf("  that says is needed. front peak %.1f N.m, rear peak %.1f N.m, ratio %.2f.\n",
                calibration.peakTorque[0], calibration.peakTorque[2],
                calibration.peakTorque[0] / std::max(calibration.peakTorque[2], 1e-9));
}

// =============================================================================================
// 10-12. THE PROBE-LOCAL QUALIFIED ARM, SWEPT, AND AGAINST LAW-OFF
// =============================================================================================

namespace
{

[[nodiscard]] AssistSetup withoutSlipAwareRecovery(const AssistSetup& assists)
{
    auto copy = assists;
    copy.antilock.recoveryAuthority = RecoveryAuthority::Disabled;

    return copy;
}

[[nodiscard]] Candidate acceptingBelow(const double fraction, const bool ideal = false)
{
    auto candidate = Candidate{};
    candidate.enabled = true;
    candidate.accept = fraction;
    candidate.normalised = true;
    candidate.rearOnly = true;
    candidate.useIdeal = ideal;

    return candidate;
}

} // namespace

TEST_CASE("the qualified law on the dry ensemble, swept broadly", "[.road-torque]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    std::printf("\n=== the candidate: `pastBand AND the road torque estimate is still large` ===\n");
    std::printf("  The production `advanceAntilockChannel` is CALLED, with a per-step local copy of\n");
    std::printf("  `AntilockSetup` whose `slipAwareRecovery` is cleared on the steps where the\n");
    std::printf("  evidence says the wheel has recovered. Only on steps where `pastBand` would have\n");
    std::printf("  fired, so the re-apply taper is not disturbed anywhere else. No production file is\n");
    std::printf("  touched and no default is written.\n");
    std::printf("\n  The threshold is a fraction of THIS corner's own peak brake torque (rear %.1f N.m),\n",
                calibration.peakTorque[2]);
    std::printf("  which is ECU calibration the layer already holds. **Nothing is chosen here.**\n");
    std::printf("  Stopping distance is NOT optimised; the search is for a broad physical plateau.\n");

    printArmHeader();

    const auto production = summarise(ensembleOf(setup.value(), world.value(), assists, calibration, Candidate{}, 1.0));
    printArm("production (control)", production);

    const auto lawOff = summarise(
        ensembleOf(setup.value(), world.value(), withoutSlipAwareRecovery(assists), calibration, Candidate{}, 1.0));
    printArm("slipAwareRecovery OFF (control)", lawOff);

    struct SweepPoint
    {
        double fraction;
        ArmSummary summary;
    };

    auto sweep = std::vector<SweepPoint>{};

    for (const auto fraction :
         {-0.20, -0.10, -0.05, -0.02, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00})
    {
        const auto summary =
            summarise(ensembleOf(setup.value(), world.value(), assists, calibration, acceptingBelow(fraction), 1.0));

        auto name = std::string{"candidate, accept below "};
        name += std::to_string(fraction).substr(0, 6);
        name += " x peak";

        printArm(name.c_str(), summary);
        sweep.push_back(SweepPoint{.fraction = fraction, .summary = summary});
    }

    std::printf("\n  --- the same sweep in N.m at the rear corner, and what each arm changed ---\n");
    std::printf("  %10s  %10s   %8s   %9s   %9s   %9s   %10s   %8s\n", "x peak", "N.m", "stranded", "stop med",
                "stop P10", "stop P90", "rear util", "changed");

    for (const auto& point : sweep)
    {
        std::printf("  %10.3f  %10.2f   %5zu/%2zu   %9.3f   %9.3f   %9.3f   %10.4f   %8.0f\n", point.fraction,
                    point.fraction * calibration.peakTorque[2], point.summary.stranded, point.summary.n,
                    point.summary.distance.median, point.summary.distance.p10, point.summary.distance.p90,
                    point.summary.rearUtilisation.median, point.summary.meanQualified);
    }

    std::printf("\n  `changed` is the mean number of controller steps per stop on which the\n");
    std::printf("  qualification suppressed a `pastBand` that would otherwise have been asserted.\n");
    std::printf("  A zero there and a production-identical row is the arm doing nothing at all.\n");

    // The ideal-kinematic arm, to say what the tone ring costs.
    std::printf("\n  --- the same qualification on the IDEAL continuous acceleration ---\n");
    printArmHeader();

    for (const auto fraction : {0.05, 0.15, 0.30})
    {
        const auto summary = summarise(
            ensembleOf(setup.value(), world.value(), assists, calibration, acceptingBelow(fraction, true), 1.0));

        auto name = std::string{"IDEAL kinematic, accept below "};
        name += std::to_string(fraction).substr(0, 5);

        printArm(name.c_str(), summary);
    }

    // And the normalised-eta arm section 7 exists to test.
    std::printf("\n  --- the NORMALISED eta arm: qualify on eta < threshold, VALID samples only ---\n");
    printArmHeader();

    for (const auto threshold : {0.25, 0.50, 0.75, 1.00})
    {
        auto candidate = Candidate{};
        candidate.enabled = true;
        candidate.useEta = true;
        candidate.etaAccept = threshold;
        candidate.rearOnly = true;

        const auto summary = summarise(ensembleOf(setup.value(), world.value(), assists, calibration, candidate, 1.0));

        auto name = std::string{"eta arm, accept below "};
        name += std::to_string(threshold).substr(0, 4);

        printArm(name.c_str(), summary);
    }

    // The whole-car arm, so the answer is not "it happened to be the rear channel".
    std::printf("\n  --- the qualification on ALL THREE channels rather than the rear alone ---\n");
    printArmHeader();

    for (const auto fraction : {0.05, 0.15, 0.30})
    {
        auto candidate = acceptingBelow(fraction);
        candidate.rearOnly = false;

        const auto summary = summarise(ensembleOf(setup.value(), world.value(), assists, calibration, candidate, 1.0));

        auto name = std::string{"all channels, accept below "};
        name += std::to_string(fraction).substr(0, 5);

        printArm(name.c_str(), summary);
    }

    std::printf("\n  CONTROLS, restated so the sweep is read against them:\n");
    std::printf("    production            stranded %2zu/%zu, median %.3f m, rear util %.4f\n", production.stranded,
                production.n, production.distance.median, production.rearUtilisation.median);
    std::printf("    slipAwareRecovery OFF stranded %2zu/%zu, median %.3f m, rear util %.4f\n", lawOff.stranded,
                lawOff.n, lawOff.distance.median, lawOff.rearUtilisation.median);
    std::printf("\n  The candidate has architectural value ONLY if it keeps something the law-off arm\n");
    std::printf("  gives away. That question is answered on low mu, not here.\n");
}

// =============================================================================================
// 13. LOW MU
// =============================================================================================

namespace
{

struct SurfaceArm
{
    std::string name;
    std::vector<Member> members;
};

void printSurfaceHeader(const char* what)
{
    std::printf("\n  %-40s   %8s %8s %8s   %8s %8s   %8s %8s   %8s\n", what, "stop P10", "MEDIAN", "P90", "front u",
                "rear u", "f cyc/s", "r cyc/s", "engaged");
}

void printSurfaceArm(const SurfaceArm& arm)
{
    auto distances = std::vector<double>{};
    auto front = std::vector<double>{};
    auto rear = std::vector<double>{};
    auto frontRate = std::vector<double>{};
    auto rearRate = std::vector<double>{};
    auto engagedShare = std::vector<double>{};

    for (const auto& member : arm.members)
    {
        distances.push_back(member.distance);
        front.push_back(member.frontUtilisation);
        rear.push_back(member.rearUtilisation);
        frontRate.push_back(member.frontEngaged > 0.0 ? static_cast<double>(member.frontCycles) / member.frontEngaged
                                                      : 0.0);
        rearRate.push_back(member.rearEngaged > 0.0 ? static_cast<double>(member.rearCycles) / member.rearEngaged
                                                    : 0.0);
        engagedShare.push_back(member.time > 0.0 ? member.rearEngaged / member.time : 0.0);
    }

    const auto stop = distributionOf(distances);

    std::printf("  %-40s   %8.3f %8.3f %8.3f   %8.4f %8.4f   %8.2f %8.2f   %8.3f\n", arm.name.c_str(), stop.p10,
                stop.median, stop.p90, distributionOf(front).median, distributionOf(rear).median,
                distributionOf(frontRate).median, distributionOf(rearRate).median, distributionOf(engagedShare).median);
}

} // namespace

TEST_CASE("the candidate on the low-mu ensemble, against the law on and off", "[.road-torque]")
{
    // **The low-mu red is not touched here.** `anti-lock braking is worth something on a uniformly
    // slippery surface` keeps its bound and its `[!shouldfail]`. What this case answers is the one
    // question the architecture turns on: the slip-aware recovery law is documented as degrading to
    // the previous law when the estimator under-reads, which is what it does on a uniformly slippery
    // surface — so does the law buy anything measurable there, and does the candidate keep it?
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto slippery = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(slippery.has_value());

    const auto assists = withAntilock(setup.value());
    const auto plain = golfGtiMk7Assists(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    std::printf("\n=== low mu (0.35 everywhere), %zu members, full pedal ===\n", ensembleCount);

    auto arms = std::vector<SurfaceArm>{};
    arms.push_back(SurfaceArm{"no electronics (locked wheels)",
                              ensembleOf(setup.value(), slippery.value(), plain, calibration, Candidate{}, 1.0)});
    arms.push_back(SurfaceArm{"production ABS, law ON",
                              ensembleOf(setup.value(), slippery.value(), assists, calibration, Candidate{}, 1.0)});
    arms.push_back(SurfaceArm{"law OFF", ensembleOf(setup.value(), slippery.value(), withoutSlipAwareRecovery(assists),
                                                    calibration, Candidate{}, 1.0)});

    for (const auto fraction : {0.05, 0.15, 0.30})
    {
        auto name = std::string{"candidate, accept below "};
        name += std::to_string(fraction).substr(0, 5);
        arms.push_back(SurfaceArm{
            name, ensembleOf(setup.value(), slippery.value(), assists, calibration, acceptingBelow(fraction), 1.0)});
    }

    printSurfaceHeader("arm");
    for (const auto& arm : arms)
    {
        printSurfaceArm(arm);
    }

    std::printf("\n  --- what the anti-lock system is worth on this surface, PAIRED member by member ---\n");

    const auto& locked = arms[0].members;

    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        auto worth = std::vector<double>{};
        auto qualified = 0.0;

        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            worth.push_back(100.0 * (locked[member].distance - arms[index].members[member].distance) /
                            locked[member].distance);
            qualified += static_cast<double>(arms[index].members[member].qualifiedSteps);
        }

        const auto distribution = distributionOf(worth);
        auto positive = std::size_t{0};
        for (const auto value : worth)
        {
            positive += value > 0.0 ? 1 : 0;
        }

        std::printf("  %-40s  worth %+7.3f%% median (%+7.3f to %+7.3f), %2zu/%zu positive, %6.0f changed steps\n",
                    arms[index].name.c_str(), distribution.median, distribution.minimum, distribution.maximum, positive,
                    ensembleCount, qualified / static_cast<double>(ensembleCount));
    }

    std::printf("\n  READ THIS AGAINST THE LAW'S OWN STATED SCOPE. `AntilockSetup::slipAwareRecovery`\n");
    std::printf("  says every leg is gated on the reference reading PAST the band, and that on a\n");
    std::printf("  uniformly slippery surface the estimate reads far BELOW the truth, so the guards\n");
    std::printf("  never fire and the channel behaves exactly as the previous law did. If the law-ON\n");
    std::printf("  and law-OFF rows above are the same to the metre, that is that statement measured,\n");
    std::printf("  and it means the law has no demonstrated low-mu benefit to preserve.\n");
}

// =============================================================================================
// 14. SPLIT MU
// =============================================================================================

TEST_CASE("the candidate on the split-mu ensemble", "[.road-torque]")
{
    // No yaw control is redesigned and `yawMomentDelay` is not touched: it is off on every arm.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto plain = golfGtiMk7Assists(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    REQUIRE_FALSE(assists.antilock.yawMomentDelay);

    std::printf("\n=== split mu (1.00 left, 0.35 right), %zu members, full pedal ===\n", ensembleCount);

    auto arms = std::vector<SurfaceArm>{};
    arms.push_back(
        SurfaceArm{"no electronics", ensembleOf(setup.value(), split.value(), plain, calibration, Candidate{}, 1.0)});
    arms.push_back(SurfaceArm{"production ABS, law ON",
                              ensembleOf(setup.value(), split.value(), assists, calibration, Candidate{}, 1.0)});
    arms.push_back(SurfaceArm{"law OFF", ensembleOf(setup.value(), split.value(), withoutSlipAwareRecovery(assists),
                                                    calibration, Candidate{}, 1.0)});

    for (const auto fraction : {0.05, 0.15, 0.30})
    {
        auto name = std::string{"candidate, accept below "};
        name += std::to_string(fraction).substr(0, 5);
        arms.push_back(SurfaceArm{
            name, ensembleOf(setup.value(), split.value(), assists, calibration, acceptingBelow(fraction), 1.0)});
    }

    printSurfaceHeader("arm");
    for (const auto& arm : arms)
    {
        printSurfaceArm(arm);
    }

    std::printf("\n  --- the directional half ---\n");
    std::printf("  %-40s   %9s %9s %9s   %9s %9s   %9s\n", "arm", "yaw P10", "yaw MED", "yaw P90", "lateral",
                "peak rate", "in 45 deg");

    for (const auto& arm : arms)
    {
        auto yaw = std::vector<double>{};
        auto lateral = std::vector<double>{};
        auto rate = std::vector<double>{};
        auto inside = std::size_t{0};

        for (const auto& member : arm.members)
        {
            yaw.push_back(std::abs(member.finalYaw) * degrees);
            lateral.push_back(std::abs(member.lateralTravel));
            rate.push_back(member.peakYawRate);
            inside += std::abs(member.finalYaw) < 45.0 / degrees ? 1 : 0;
        }

        const auto yawDistribution = distributionOf(yaw);

        std::printf("  %-40s   %9.3f %9.3f %9.3f   %9.3f %9.4f   %6zu/%zu\n", arm.name.c_str(), yawDistribution.p10,
                    yawDistribution.median, yawDistribution.p90, distributionOf(lateral).median,
                    distributionOf(rate).median, inside, ensembleCount);
    }

    std::printf("\n  Yaw in degrees at the end of the stop, lateral travel in metres, peak yaw RATE in\n");
    std::printf("  rad/s. The 45-degree column is the criterion's own quarter-turn bound, printed\n");
    std::printf("  where each arm sits inside it. **No criterion is read for a value to change.**\n");
}

// =============================================================================================
// 15. PHYSICAL EVENT AUDIT
// =============================================================================================

TEST_CASE("the decisions the candidate changes, event by event", "[.road-torque]")
{
    // A shorter stop proves nothing on its own. This is the evidence that the qualification fires
    // for the physical reason it is supposed to: a loaded wheel at essentially zero true slip whose
    // guard slip is nevertheless past the band.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    for (const auto member : {std::size_t{4}, std::size_t{27}})
    {
        const auto entry = memberEntry(member, ensembleCount);
        const auto candidate = acceptingBelow(0.15);

        const auto production =
            record(setup.value(), world.value(), assists, calibration, Candidate{}, 1.0, entry, 0.0, true);
        const auto qualifiedRun =
            record(setup.value(), world.value(), assists, calibration, candidate, 1.0, entry, 0.0, true);

        const auto productionUse = axleUse(production);
        const auto qualifiedUse = axleUse(qualifiedRun);

        std::printf("\n=== member %zu at %.4f km/h ===\n", member, 3.6 * entry);
        std::printf("  production  %.3f m, rear utilisation %.4f, rear-off ticks %zu\n", production.distance,
                    productionUse.rear, production.rearAirborneTicks);
        std::printf("  candidate   %.3f m, rear utilisation %.4f, rear-off ticks %zu, %zu changed decisions\n",
                    qualifiedRun.distance, qualifiedUse.rear, qualifiedRun.rearAirborneTicks,
                    qualifiedRun.qualifiedSteps);

        const auto steps = rearSteps(qualifiedRun);

        std::printf("\n  the first 30 changed decisions on the rear channel:\n");
        std::printf("  %8s %8s %8s %8s %8s %9s %9s %9s %9s %9s %9s\n", "t [s]", "phase", "->", "guard", "sensed m/s",
                    "acc m/s2", "T_brake", "OBSERVED", "trueSlip", "true Fz", "true Fx");

        auto shown = std::size_t{0};

        for (const auto& step : steps)
        {
            if (!step.qualified)
            {
                continue;
            }

            std::printf("  %8.4f %8s %8s %8.4f %10.3f %9.2f %9.2f %9.2f %9.4f %9.0f %9.1f\n", step.time,
                        modulatorName(step.before), modulatorName(step.after), step.guardSlip, step.sensedSpeed,
                        step.acceleration, step.observable.brakeTorque, step.observable.sensed, step.trueSlip,
                        step.trueLoad, step.trueLongitudinal);

            if (++shown >= 30)
            {
                break;
            }
        }

        // And the population every changed decision belongs to, which is the audit's real answer.
        auto rolling = std::size_t{0};
        auto slipping = std::size_t{0};
        auto airborne = std::size_t{0};
        auto middling = std::size_t{0};
        auto loads = std::vector<double>{};
        auto slips = std::vector<double>{};

        for (const auto& step : steps)
        {
            if (!step.qualified)
            {
                continue;
            }

            loads.push_back(step.trueLoad);
            slips.push_back(std::abs(step.trueSlip));

            if (!step.trueContact)
            {
                airborne++;
            }
            else if (std::abs(step.trueSlip) < lowSlip)
            {
                rolling++;
            }
            else if (std::abs(step.trueSlip) > highSlip)
            {
                slipping++;
            }
            else
            {
                middling++;
            }
        }

        std::printf("\n  every changed decision in this member, by what the tyre was ACTUALLY doing:\n");
        std::printf("    loaded and rolling (true slip < %.2f)   %6zu\n", lowSlip, rolling);
        std::printf("    in the band                             %6zu\n", middling);
        std::printf("    genuinely slipping (true slip > %.2f)   %6zu  <-- these are the mistakes\n", highSlip,
                    slipping);
        std::printf("    airborne                                %6zu\n", airborne);

        distributionHeader("at the changed decisions");
        distributionRow("true vertical load [N]", distributionOf(loads));
        distributionRow("true slip []", distributionOf(slips));

        // **The audit's real answer.** The observable is a measurement of |Fx| * r, and that is
        // small either because the tyre is barely slipping OR because it is barely loaded. Splitting
        // the changed decisions on true slip and then reporting the LOAD in each half is what says
        // which of the two the candidate is actually detecting.
        auto slippingLoads = std::vector<double>{};
        auto slippingForce = std::vector<double>{};
        auto slippingShare = std::vector<double>{};
        auto rollingLoads = std::vector<double>{};

        for (const auto& step : steps)
        {
            if (!step.qualified || !step.trueContact)
            {
                continue;
            }

            if (std::abs(step.trueSlip) > highSlip)
            {
                slippingLoads.push_back(step.trueLoad);
                slippingForce.push_back(std::abs(step.trueLongitudinal));
                slippingShare.push_back(step.trueLoad / qualifiedRun.staticRearLoad);
            }
            else if (std::abs(step.trueSlip) < lowSlip)
            {
                rollingLoads.push_back(step.trueLoad);
            }
        }

        distributionHeader("the `genuinely slipping` half");
        distributionRow("true vertical load [N]", distributionOf(slippingLoads));
        distributionRow("...as a fraction of static", distributionOf(slippingShare));
        distributionRow("|true longitudinal force| [N]", distributionOf(slippingForce));
        distributionHeader("the `loaded and rolling` half");
        distributionRow("true vertical load [N]", distributionOf(rollingLoads));
        std::printf("\n  static rear wheel load on this car: %.1f N\n", qualifiedRun.staticRearLoad);
    }
}

// =============================================================================================
// 18. ACCEPTANCE-TEST AUDIT
// =============================================================================================

TEST_CASE("what the single-stop recovery-law acceptance test actually asserts", "[.road-torque]")
{
    // **The test is not changed, loosened, tightened or moved.** What this case does is print what
    // its own single member reads under each arm, beside what the ensemble reads.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    // Ensemble member 14 of 29 is exactly 100.000 km/h, which is the entry speed the acceptance
    // test uses.
    const auto entry = memberEntry(ensembleCount / 2, ensembleCount);
    REQUIRE(std::abs(3.6 * entry - 100.0) < 1e-9);

    std::printf("\n=== `what the slip-aware recovery law is worth, measured against itself switched off` ===\n");
    std::printf("  file      RaceEngine/Tests/AntilockBrakingTests.cpp\n");
    std::printf("  asserts   on.distance < off.distance + 0.05, on ONE stop at exactly 100.000 km/h\n");
    std::printf("  status    GREEN, and UNCHANGED by this probe.\n");

    const auto on = record(setup.value(), world.value(), assists, calibration, Candidate{}, 1.0, entry);
    const auto offRun =
        record(setup.value(), world.value(), withoutSlipAwareRecovery(assists), calibration, Candidate{}, 1.0, entry);

    std::printf("\n  the single member the test reads:\n");
    std::printf("    law ON    %.3f m, rear utilisation %.4f\n", on.distance, axleUse(on).rear);
    std::printf("    law OFF   %.3f m, rear utilisation %.4f\n", offRun.distance, axleUse(offRun).rear);
    std::printf("    the assertion `%.3f < %.3f + 0.05` is %s\n", on.distance, offRun.distance,
                on.distance < offRun.distance + 0.05 ? "TRUE" : "FALSE");

    for (const auto fraction : {0.05, 0.15, 0.30})
    {
        const auto arm =
            record(setup.value(), world.value(), assists, calibration, acceptingBelow(fraction), 1.0, entry);
        std::printf("    candidate at %.3f x peak   %.3f m, rear utilisation %.4f, %zu changed decisions\n", fraction,
                    arm.distance, axleUse(arm).rear, arm.qualifiedSteps);
    }

    std::printf("\n  WHAT THE TEST ASSERTS, stated plainly: that on ONE entry speed the law does not\n");
    std::printf("  make the stop more than 5 cm longer. It is a NON-REGRESSION guard on one member of\n");
    std::printf("  a bimodal distribution, and it is not evidence that the law shortens a stop, that\n");
    std::printf("  it improves rear utilisation, or that it does anything at all on low grip.\n");
    std::printf("  `docs/braking-chain-brief.md` section 19 already recorded that this member is the\n");
    std::printf("  second shortest of the twenty-nine and that on the ensemble the comparison runs the\n");
    std::printf("  other way. Nothing here changes it.\n");
}

// =============================================================================================
// 17. DIMENSIONLESS IS NOT UNIVERSAL
// =============================================================================================

TEST_CASE("what the candidate is sensitive to: the ring, the inertia, the radius", "[.road-torque]")
{
    // **The observable is dimensionless only after it is divided by a peak brake torque, and even
    // then it is not automatically vehicle-invariant.** Three of its inputs are per-vehicle numbers
    // and one of them — the tone ring — is not even a property of the vehicle's mechanics. Each is
    // varied here on the SAME plant, with the production arm re-measured beside it so a moved
    // baseline cannot be read as a moved candidate.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto base = withAntilock(setup.value());
    const auto baseCalibration = calibrationOf(setup.value(), base);

    std::printf("\n=== tone-ring resolution: the sensor the observable is read through ===\n");
    std::printf("  `ToneRing::teeth` is marked in `WheelSensors.cppm` as a SOFT source -- 48 is the\n");
    std::printf("  common VAG count and the exact Mk7 figure is not confirmed. The plant's sensor and\n");
    std::printf("  the ECU's estimate both change with it, so the production arm moves too.\n");

    printArmHeader();

    for (const auto teeth :
         {std::uint32_t{24}, std::uint32_t{44}, std::uint32_t{48}, std::uint32_t{60}, std::uint32_t{96}})
    {
        auto assists = base;
        assists.toneRing.teeth = teeth;
        const auto calibration = calibrationOf(setup.value(), assists);

        auto name = std::string{"ring "};
        name += std::to_string(teeth);
        name += " poles, production";
        printArm(name.c_str(),
                 summarise(ensembleOf(setup.value(), world.value(), assists, calibration, Candidate{}, 1.0)));

        for (const auto fraction : {0.05, 0.15})
        {
            auto arm = std::string{"ring "};
            arm += std::to_string(teeth);
            arm += " poles, candidate ";
            arm += std::to_string(fraction).substr(0, 5);
            printArm(arm.c_str(), summarise(ensembleOf(setup.value(), world.value(), assists, calibration,
                                                       acceptingBelow(fraction), 1.0)));
        }
    }

    std::printf("\n=== wheel inertia: the ONE new per-vehicle number the observable needs ===\n");
    std::printf("  The plant keeps the car's own %.4f kg.m^2; only the ECU's calibration is wrong by\n",
                setup->corners[0].wheelInertia);
    std::printf("  the stated factor, which is what a fleet estimate off an archetype would be.\n");

    printArmHeader();

    for (const auto scale : {0.5, 0.7, 1.0, 1.4, 2.0})
    {
        auto calibration = baseCalibration;
        calibration.inertia *= scale;

        for (const auto fraction : {0.05, 0.15})
        {
            auto name = std::string{"inertia x"};
            name += std::to_string(scale).substr(0, 4);
            name += ", candidate ";
            name += std::to_string(fraction).substr(0, 5);
            printArm(name.c_str(), summarise(ensembleOf(setup.value(), world.value(), base, calibration,
                                                        acceptingBelow(fraction), 1.0)));
        }
    }

    std::printf("\n=== rolling radius, and the brake-torque scale the threshold is a fraction of ===\n");

    printArmHeader();

    for (const auto scale : {0.90, 1.00, 1.10})
    {
        auto calibration = baseCalibration;
        calibration.radius *= scale;

        auto name = std::string{"ECU radius x"};
        name += std::to_string(scale).substr(0, 4);
        name += ", candidate 0.15";
        printArm(name.c_str(),
                 summarise(ensembleOf(setup.value(), world.value(), base, calibration, acceptingBelow(0.15), 1.0)));
    }

    for (const auto scale : {0.5, 1.0, 2.0})
    {
        auto calibration = baseCalibration;
        for (auto& value : calibration.peakTorque)
        {
            value *= scale;
        }

        auto name = std::string{"ECU peak torque x"};
        name += std::to_string(scale).substr(0, 4);
        name += ", candidate 0.15";
        printArm(name.c_str(),
                 summarise(ensembleOf(setup.value(), world.value(), base, calibration, acceptingBelow(0.15), 1.0)));
    }

    std::printf("\n  The peak-torque row is the same experiment as moving the threshold, which is what\n");
    std::printf("  says the normalisation and the threshold are one parameter and not two.\n");
}
