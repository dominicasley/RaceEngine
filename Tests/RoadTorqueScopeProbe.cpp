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
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::advanceAntilockChannel;
using raceengine::advanceReferenceSpeed;
using raceengine::advanceYawMomentDelay;
using raceengine::AntilockChannelInputs;
using raceengine::AntilockChannelState;
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
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::WheelSpeedReading;
using raceengine::WheelSpeedReadings;

// =============================================================================================
// CAN THE ROAD-TORQUE QUALIFICATION BE SCOPED TO THE EMPTY-CALIPER / RECOVERY CONDITION?
// `./EngineTests "[.road-torque-scope]"` — hidden, registers no ctest entry, asserts nothing about
// the car, and changes no production file.
//
// **What this file is for.** `[.road-torque]` established that the road-torque observable
//
//     T_road_hat = T_brake_cmd + I * dw/dt          [N.m, spin-up positive]
//
// is a ROAD-FORCE detector and not a SLIP detector, so a low reading is ambiguous between
// `recovered and rolling` and `slipping hard while carrying almost no load`. On the representative
// stranded dry member it changed 412 decisions, of which 64 were the intended loaded/rolling case,
// 216 were a genuinely slipping wheel at 3-7% of static load, and 101 were airborne. Broad
// qualification also worsened split-mu yaw monotonically with the threshold.
//
// **The hypothesis under test here is one of SCOPE and not of threshold.** The road-torque
// evidence may only need authority in the state the dry failure actually lives in: the channel is
// in recovery, its own caliper is already empty, and `pastBand` would otherwise refuse to
// re-apply. In that state a further dump removes nothing, so an override cannot give away brake
// pressure that was doing work.
//
// The road-torque threshold is held FIXED at 0.150 x the corner's peak brake torque for the whole
// scoping comparison, which is the arm `[.road-torque]` already characterised. It is not optimised
// anywhere in this file; a three-point robustness check at 0.05 / 0.15 / 0.30 is the only other
// value that appears.
//
// **Truth is used for offline labelling only.** No candidate decision anywhere in this file reads
// load, contact, true slip, tyre force, road speed or surface grip.
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

// The ensemble protocol, identical to `[.abs-ledger]`, `[.rear-hop]` and `[.road-torque]`.
constexpr auto ensembleBand = 0.007;
constexpr auto ensembleCount = std::size_t{29};

constexpr auto strandedRear = 0.50;

// The one road-torque threshold this file uses, as a fraction of the corner's own peak brake
// torque. **Held fixed on purpose** — the experiment is about scope.
constexpr auto fixedAccept = 0.150;

// The offline labels, identical to `[.road-torque]`'s so the two files' numbers are comparable.
constexpr auto lowSlip = 0.05;
constexpr auto highSlip = 0.20;
constexpr auto loadedFraction = 0.25;

// The reading age past which the acceleration term is a difference of two numbers the ECU has not
// refreshed. A reporting parameter; nothing gates on it.
constexpr auto staleAge = 0.010;

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

struct RoadTorqueCalibration
{
    double inertia = 1.45;
    double radius = tyreRadius;
    std::array<double, cornerCount> torquePerPressure{};
    std::array<double, cornerCount> peakTorque{};
};

// ---------------------------------------------------------------------------------------------
// SECTION 2 OF THE DELIVERABLE — "CALIPER EMPTY" FROM EXISTING STATE.
//
// **The production state machine already contains the discrete condition.** `BrakeControlImpl`'s
// dump branch is
//
//     state.pressure = std::max(0.0, state.pressure - dumpGradient * deltaTime);
//
// so the actuator has a hard lower BOUND at zero pascals and reaches it exactly. A channel sitting
// on that bound has, in the controller's own terms, already removed every pascal it commanded:
// another dump step subtracts from a number the clamp puts straight back. That is the state
// "no further meaningful dump authority remains", stated by the implementation rather than by a
// threshold this file invents.
//
// It is read off `AntilockChannelState::pressure` as it stands BEFORE the state machine runs, which
// is the value the decision is taken against. No new scalar, no bar figure, no per-vehicle number.
// ---------------------------------------------------------------------------------------------
[[nodiscard]] bool caliperEmpty(const double pressureBefore)
{
    return pressureBefore <= 0.0;
}

// The architecture arms of section 5. Every one of them uses the SAME road-torque threshold; the
// only thing that differs is where the qualification is allowed to act.
enum class Scope : std::uint8_t
{
    // A — wherever `pastBand` fires. The arm `[.road-torque]` characterised.
    Everywhere,
    // B — the Recover phase only.
    RecoverOnly,
    // C — a caliper already at its lower bound, whatever the phase.
    EmptyOnly,
    // D — both.
    RecoverAndEmpty,
    // E — the Recover -> Dump decision itself, with the caliper already empty.
    StuckDecisionAndEmpty
};

[[nodiscard]] const char* scopeName(const Scope scope)
{
    switch (scope)
    {
    case Scope::Everywhere:
        return "A everywhere pastBand fires";
    case Scope::RecoverOnly:
        return "B Recover only";
    case Scope::EmptyOnly:
        return "C caliper empty only";
    case Scope::RecoverAndEmpty:
        return "D Recover AND caliper empty";
    case Scope::StuckDecisionAndEmpty:
        return "E Recover->Dump AND caliper empty";
    }

    return "?";
}

struct Candidate
{
    bool enabled = false;
    double accept = fixedAccept;
    bool rearOnly = true;
    Scope scope = Scope::Everywhere;

    // --- section 16's leg ablation (2026-09-07) -----------------------------------------------
    //
    // **Probe-local, and defaulted to the production path.** `replica` routes the channel through
    // this file's own verbatim copy of `advanceAntilockChannel` instead of through the production
    // function; the leg flags are the only thing that copy can do differently, and with every leg
    // on the copy is required to reproduce production bit for bit (sections 16.1, 17.1 and 18.1).
    // Every arm in sections 1-15 leaves `replica` false, so nothing above this line changes.
    bool replica = false;
    bool legA = true;
    bool legB = true;
    bool legC = true;

    // **LEG D, added 2026-09-07 for section 18.** The fourth `pastBand` consumer — the
    // Recover -> Reapply hold gate — which sections 16 and 17 held ON in every arm and named in
    // their own comments as an unablated asymmetry. It is a switch here for exactly one reason:
    // section 18 is asked which of B, C and D carries the remaining split-mu yaw cost with leg A
    // held in its production state, and that question cannot be put without it. Defaulted true, so
    // every arm above this line is unchanged by a bit.
    bool legD = true;
};

// ---------------------------------------------------------------------------------------------
// SECTION 16 — THE PROBE-LOCAL COPY OF THE CONTROL LAW, WITH THREE SWITCHES.
//
// **A verbatim copy of `advanceAntilockChannel` and nothing else.** Not a re-derivation, not a
// simplification and not a re-ordering: the sensor bookkeeping, the term definitions, the phase
// order and every branch are the production body character for character, with exactly four
// boolean guards spliced into the four named expressions. Nothing else in this file, and nothing
// at all in `RaceEngine/Assists`, is touched.
//
// It exists because the legs of `AntilockSetup::slipAwareRecovery` share ONE production switch,
// so the shipped API cannot ablate them individually. Giving the shipped struct four more switches
// would be a production change for a probe's convenience, which is the thing this task forbids; a
// copy that must prove itself bit-identical to production before its results are read costs the
// same and asserts more.
//
// **The copy is proved rather than asserted.** Section 16.1 runs the whole 15-member steered
// ensemble through this function with all three legs on and requires every member to match the
// production ensemble to the bit. If that check ever fails the copy has drifted and every ablation
// number below it is void.
//
// THE LEGS, AS THE TASK NAMES THEM AND AS THE CODE WRITES THEM:
//
//   LEG A — the Dump-exit guard.        `(!losing && !pastBand)`      -> `(!losing)`
//   LEG B — the Recover stuck branch.   `pastBand && excess < recoveryAcceleration` -> removed
//   LEG C — the Reapply control, BOTH halves: the proximity taper (with the `!pastBand` term in
//           `fast`, which is the same leg's other end) AND the Reapply -> Dump stuck protection.
//
//   LEG D — the Recover -> Reapply hold gate, `if (!pastBand)` inside the `!(surged && surging)`
//           branch. A separate `pastBand` consumer in a separate phase, added after the three legs
//           the setup field's own comment lists. **It is held ON in every arm of sections 16 and
//           17**, which is the asymmetry those sections record; section 18 is the first to ablate
//           it, and does so with leg A held in its production state.
// ---------------------------------------------------------------------------------------------
[[nodiscard]] double ablatedAntilockChannel(const AntilockSetup& setup, const BrakeChannel channel,
                                            AntilockChannelState& state, const WheelSpeedReading& wheel,
                                            const double wheelRoadSpeed, const double referenceSpeed,
                                            const double referenceAcceleration, const bool referenceValid,
                                            const double requestedPressure, const double deltaTime, const bool legA,
                                            const bool legB, const bool legC, const bool legD)
{
    const auto reapplyGradient =
        channel == BrakeChannel::Rear ? setup.modulator.rearReapplyGradient : setup.modulator.reapplyGradient;

    if (wheel.pulses != state.lastPulses)
    {
        const auto elapsed = state.sinceUpdate + deltaTime;

        if (state.lastPulses > 0 && elapsed > 0.0)
        {
            state.acceleration = (wheelRoadSpeed - state.lastSpeed) / elapsed;
        }

        state.lastPulses = wheel.pulses;
        state.lastSpeed = wheelRoadSpeed;
        state.sinceUpdate = 0.0;
    }
    else
    {
        state.sinceUpdate += deltaTime;
    }

    const auto request = std::max(requestedPressure, 0.0);

    if (!setup.enabled || !wheel.valid || request <= 0.0)
    {
        state.phase = ModulatorPhase::Passive;
        state.pressure = request;
        state.surged = false;

        return request;
    }

    const auto slip = referenceValid ? estimatedSlip(referenceSpeed, wheelRoadSpeed) : 0.0;

    const auto excess = state.acceleration - referenceAcceleration;
    const auto losing = state.acceleration < setup.lockDeceleration;
    const auto surging = excess > setup.recoverySurge;

    const auto projectedWheel = std::max(wheelRoadSpeed, state.lastSpeed + referenceAcceleration * state.sinceUpdate);
    const auto guardSlip = referenceValid ? estimatedSlip(referenceSpeed, projectedWheel) : 0.0;

    const auto pastBand =
        (setup.recoveryAuthority != RecoveryAuthority::Disabled) && referenceValid && guardSlip > setup.slipEnter;

    switch (state.phase)
    {
    case ModulatorPhase::Passive:
        state.pressure = request;

        if (losing)
        {
            state.phase = ModulatorPhase::Hold;
        }

        break;

    case ModulatorPhase::Hold:
        if (slip > setup.slipEnter)
        {
            state.phase = ModulatorPhase::Dump;
            state.surged = false;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        else if (!losing)
        {
            state.phase = ModulatorPhase::Reapply;
        }

        break;

    case ModulatorPhase::Dump:
        state.pressure = std::max(0.0, state.pressure - setup.modulator.dumpGradient * deltaTime);

        // **LEG A.** Production is `(!losing && !pastBand)`; with the leg off the disjunct is
        // `!losing`, which is the acceleration-only law this file's dump exit had before 2026-08-24.
        if (excess > setup.recoveryAcceleration || (!losing && !(legA && pastBand)))
        {
            state.phase = ModulatorPhase::Recover;
            state.surged = false;
        }

        break;

    case ModulatorPhase::Recover:
        state.surged = state.surged || surging;

        if (slip > setup.slipEnter && losing)
        {
            state.phase = ModulatorPhase::Dump;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        // **LEG B.** The stuck branch. With the leg off this `else if` cannot be taken and control
        // falls through to the surge test below it.
        else if (legB && pastBand && excess < setup.recoveryAcceleration)
        {
            state.phase = ModulatorPhase::Dump;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        else if (!(state.surged && surging))
        {
            // **LEG D.** The Recover -> Reapply hold gate — a fourth `pastBand` consumer that the
            // original three-leg decomposition does not name. Held ON in every arm of sections 16
            // and 17; switchable from section 18 onwards, where the question is which of B, C and
            // D carries the remaining split-mu yaw cost with leg A in its production state. With
            // the leg off the gate is gone and a channel that is neither departing nor surging
            // re-applies whatever its estimated slip says, which is the pre-law transition.
            if (!(legD && pastBand))
            {
                state.phase = ModulatorPhase::Reapply;
            }
        }

        break;

    case ModulatorPhase::Reapply:
    {
        // **LEG C, first half — the proximity taper.** With the leg off `proximity` is 1.0 and the
        // `!pastBand` term leaves `fast`, so `(fast ? 1.0 : proximity)` is 1.0 whichever way `fast`
        // lands: the gradient is the pre-law gradient to the bit, and `departurePressure` stops
        // being read at all.
        const auto proximity =
            legC && (setup.recoveryAuthority != RecoveryAuthority::Disabled) && referenceValid
                ? std::clamp((setup.slipEnter - guardSlip) / std::max(setup.slipEnter - setup.slipExit, 1e-9), 0.0, 1.0)
                : 1.0;
        const auto fast = state.pressure < state.departurePressure && !(legC && pastBand);

        state.pressure = std::min(request, state.pressure + reapplyGradient * (fast ? 1.0 : proximity) * deltaTime);

        if (losing)
        {
            state.phase = ModulatorPhase::Hold;
        }
        // **LEG C, second half — the Reapply -> Dump stuck protection.** The audit named this and
        // the taper as one leg because they are one function's slip-aware behaviour; they are
        // switched together here for exactly that reason, and section 16.0 records that they are
        // separable in principle.
        else if (legC && pastBand && excess < -setup.recoveryAcceleration &&
                 guardSlip > 2.0 * setup.slipEnter - setup.slipExit)
        {
            state.phase = ModulatorPhase::Dump;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        else if (state.pressure >= request)
        {
            state.phase = ModulatorPhase::Passive;
        }

        break;
    }
    }

    state.pressure = std::min(state.pressure, request);

    return state.pressure;
}

// Ground truth for the last physics tick. **Diagnostic only** — it labels samples offline and it
// never enters a candidate decision.
struct Truth
{
    std::array<bool, cornerCount> contact{true, true, true, true};
    std::array<double, cornerCount> load{};
    std::array<double, cornerCount> longitudinal{};
    std::array<double, cornerCount> effectiveRadius{};
    std::array<double, cornerCount> slipRatio{};
    double staticRearLoad = 1.0;
};

// The four populations of section 11, plus the slip-first partition `[.road-torque]`'s own audit
// prints so the two files' figures line up.
struct Counts
{
    std::size_t intended = 0;
    std::size_t lowLoad = 0;
    std::size_t highSlip = 0;
    std::size_t other = 0;

    std::size_t rolling = 0;
    std::size_t band = 0;
    std::size_t slipping = 0;
    std::size_t airborne = 0;

    [[nodiscard]] std::size_t total() const
    {
        return intended + lowLoad + highSlip + other;
    }

    void add(const Counts& more)
    {
        intended += more.intended;
        lowLoad += more.lowLoad;
        highSlip += more.highSlip;
        other += more.other;
        rolling += more.rolling;
        band += more.band;
        slipping += more.slipping;
        airborne += more.airborne;
    }
};

void classify(Counts& counts, const bool contact, const double load, const double slip, const double staticLoad)
{
    const auto magnitude = std::abs(slip);
    const auto loaded = contact && load >= loadedFraction * staticLoad;

    if (!loaded)
    {
        counts.lowLoad++;
    }
    else if (magnitude < lowSlip)
    {
        counts.intended++;
    }
    else if (magnitude > highSlip)
    {
        counts.highSlip++;
    }
    else
    {
        counts.other++;
    }

    if (!contact)
    {
        counts.airborne++;
    }
    else if (magnitude < lowSlip)
    {
        counts.rolling++;
    }
    else if (magnitude > highSlip)
    {
        counts.slipping++;
    }
    else
    {
        counts.band++;
    }
}

// One controller step of one channel, traced. Only the mechanism sections ask for these.
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

    double sensedSpeed = 0.0;
    double acceleration = 0.0;
    double excess = 0.0;
    double slip = 0.0;
    double guardSlip = 0.0;
    double sensorAge = 0.0;
    double brakeTorque = 0.0;
    double estimate = 0.0;

    bool pastBandUnqualified = false;
    bool emptyCaliper = false;
    bool stuckDecision = false;
    bool admitted = false;
    bool qualified = false;

    // --- added 2026-09-07 for section 18's per-leg event counts -------------------------------
    //
    // Two pieces of the channel's state as it stood BEFORE the call, recorded so the leg
    // predicates can be evaluated offline exactly as the control law writes them. `surgedBefore`
    // is `AntilockChannelState::surged` on entry — the Recover branch updates it before it tests
    // it, so the term leg D reads is `surgedBefore || surging`. `departurePressureBefore` is the
    // memory the Reapply stage's `fast` test reads. Recorded only; no decision reads either.
    bool surgedBefore = false;
    double departurePressureBefore = 0.0;

    double trueSlip = 0.0;
    double trueLoad = 0.0;
    double trueLongitudinal = 0.0;
    bool trueContact = true;
};

struct ShadowState
{
    AssistState assists{};
    std::array<ModulatorPhase, cornerCount> phase{};

    // Every controller step on a channel the candidate is allowed to look at, where `pastBand`
    // would have fired without it.
    std::size_t pastBandSteps = 0;
    // ...of which the SCOPE admitted, before the road-torque test was applied.
    std::size_t admittedSteps = 0;
    // ...of which the road-torque test then qualified. This is the changed-decision count.
    std::size_t qualifiedSteps = 0;
    // ...of those, the ones in a phase that actually READS `pastBand`. `Passive` and `Hold` do not,
    // so a qualification there changes nothing and inflates the count.
    std::size_t effectiveSteps = 0;

    Counts precision{};
    std::array<Counts, 2> byCaliper{};
    std::array<Counts, 5> byPhase{};
    std::array<Counts, cornerCount> byWheel{};
    // Which wheel the select-low channel was reading on every step where `pastBand` fired, changed
    // or not. Without it the changed-decision split by wheel cannot be read against a baseline.
    std::array<std::size_t, cornerCount> pastBandByWheel{};

    std::vector<double> qualifiedAge;
    std::size_t qualifiedStale = 0;
};

[[nodiscard]] bool admits(const Scope scope, const ModulatorPhase phase, const bool empty, const bool stuck)
{
    switch (scope)
    {
    case Scope::Everywhere:
        return true;
    case Scope::RecoverOnly:
        return phase == ModulatorPhase::Recover;
    case Scope::EmptyOnly:
        return empty;
    case Scope::RecoverAndEmpty:
        return phase == ModulatorPhase::Recover && empty;
    case Scope::StuckDecisionAndEmpty:
        return stuck && empty;
    }

    return false;
}

// The probe-local copy of `updateAssists`'s outer loop, following `[.road-torque]`'s `shadowUpdate`.
// **The control law is never copied**: `advanceAntilockChannel` is called, with a local copy of the
// setup as the only intervention.
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

            // --- the terms the production call is ABOUT to evaluate ----------------------------
            //
            // A replication of `advanceAntilockChannel`'s sensor bookkeeping and of nothing else,
            // needed because the decision has to be classified at the instant it is taken. It is
            // self-checking: the assertions after the call compare it against what production left.
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

            const auto projected = std::max(wheelSpeed, predictedLastSpeed + state.reference.rate * predictedAge);
            const auto guardSlip = state.reference.valid ? estimatedSlip(state.reference.speed, projected) : 0.0;
            const auto pastBandUnqualified = (setup.antilock.recoveryAuthority != RecoveryAuthority::Disabled) &&
                                             state.reference.valid && guardSlip > setup.antilock.slipEnter;

            // The production law's own terms, from the same replicated bookkeeping. `losing`,
            // `excess` and `slip` are exactly the expressions `advanceAntilockChannel` forms.
            const auto sensedSlip = state.reference.valid ? estimatedSlip(state.reference.speed, wheelSpeed) : 0.0;
            const auto excess = predictedAcceleration - state.reference.rate;
            const auto losing = predictedAcceleration < setup.antilock.lockDeceleration;

            // **The Recover -> Dump decision, written as the production branch writes it.** In
            // `Recover`, a channel whose wheel is not departing again takes the stuck branch when
            // `pastBand && excess < recoveryAcceleration`. Scope E is that decision and no other.
            const auto stuckDecision = snapshot.phase == ModulatorPhase::Recover &&
                                       !(sensedSlip > setup.antilock.slipEnter && losing) &&
                                       excess < setup.antilock.recoveryAcceleration;

            const auto empty = caliperEmpty(snapshot.pressure);

            // --- the observable ---------------------------------------------------------------
            const auto brakeTorque = std::max(0.0, snapshot.pressure) * calibration.torquePerPressure[controlWheel];
            const auto estimate = brakeTorque + calibration.inertia * predictedAcceleration / calibration.radius;

            // --- the qualification ------------------------------------------------------------
            auto local = setup.antilock;
            auto qualified = false;
            auto admitted = false;

            if (candidate.enabled && (!candidate.rearOnly || index == 2) && pastBandUnqualified)
            {
                shadow.pastBandSteps++;
                shadow.pastBandByWheel[controlWheel]++;

                admitted = admits(candidate.scope, snapshot.phase, empty, stuckDecision);

                if (admitted)
                {
                    shadow.admittedSteps++;
                    qualified = estimate < candidate.accept * calibration.peakTorque[controlWheel];
                }

                if (qualified)
                {
                    local.recoveryAuthority = RecoveryAuthority::Disabled;
                }
            }

            // Production unless section 16 asks for the copy. `Candidate::replica` defaults false,
            // so every arm in sections 1-15 evaluates the production function and nothing else.
            const auto pressure =
                candidate.replica
                    ? ablatedAntilockChannel(local, channel, channelState, readings[controlWheel], wheelSpeed,
                                             state.reference.speed, state.reference.rate, state.reference.valid,
                                             channelRequest, period, candidate.legA, candidate.legB, candidate.legC,
                                             candidate.legD)
                    : advanceAntilockChannel(local, channel, channelState, readings[controlWheel], wheelSpeed,
                                             state.reference.speed, state.reference.rate, state.reference.valid,
                                             channelRequest, AntilockChannelInputs{}, period);

            // Cheap in the common path on purpose: a `REQUIRE` per controller step per channel is a
            // quarter of a million Catch2 assertions per ensemble arm.
            if (channelState.acceleration != predictedAcceleration)
            {
                REQUIRE(channelState.acceleration == predictedAcceleration);
            }

            {
                const auto after =
                    std::max(wheelSpeed, channelState.lastSpeed + state.reference.rate * channelState.sinceUpdate);
                if (after != projected)
                {
                    REQUIRE(after == projected);
                }
            }

            if (qualified)
            {
                shadow.qualifiedSteps++;

                if (snapshot.phase == ModulatorPhase::Dump || snapshot.phase == ModulatorPhase::Recover ||
                    snapshot.phase == ModulatorPhase::Reapply)
                {
                    shadow.effectiveSteps++;
                }

                shadow.qualifiedAge.push_back(predictedAge);
                shadow.qualifiedStale += predictedAge > staleAge ? 1 : 0;

                classify(shadow.precision, truth.contact[controlWheel], truth.load[controlWheel],
                         truth.slipRatio[controlWheel], truth.staticRearLoad);
                classify(shadow.byCaliper[empty ? 1 : 0], truth.contact[controlWheel], truth.load[controlWheel],
                         truth.slipRatio[controlWheel], truth.staticRearLoad);
                classify(shadow.byPhase[phaseIndex(snapshot.phase)], truth.contact[controlWheel],
                         truth.load[controlWheel], truth.slipRatio[controlWheel], truth.staticRearLoad);
                classify(shadow.byWheel[controlWheel], truth.contact[controlWheel], truth.load[controlWheel],
                         truth.slipRatio[controlWheel], truth.staticRearLoad);
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
                record.excess = excess;
                record.slip = sensedSlip;
                record.guardSlip = guardSlip;
                record.sensorAge = predictedAge;
                record.brakeTorque = brakeTorque;
                record.estimate = estimate;
                record.pastBandUnqualified = pastBandUnqualified;
                record.emptyCaliper = empty;
                record.stuckDecision = stuckDecision;
                record.admitted = admitted;
                record.qualified = qualified;
                record.surgedBefore = snapshot.surged;
                record.departurePressureBefore = snapshot.departurePressure;
                record.trueSlip = truth.slipRatio[controlWheel];
                record.trueLoad = truth.load[controlWheel];
                record.trueLongitudinal = truth.longitudinal[controlWheel];
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

// --- one physics tick, for the run statistics and the mechanism trace -------------------------

struct WheelTick
{
    double load = 0.0;
    double longitudinal = 0.0;
    double lateral = 0.0;
    double capacity = 0.0;
    double slipRatio = 0.0;
    double grip = 0.0;
    double pressure = 0.0;
    double brakeTorqueCommanded = 0.0;
    ModulatorPhase phase = ModulatorPhase::Passive;
    bool inContact = false;

    // --- added 2026-09-07 for the steered low-mu re-scoring at the foot of this file ----------
    //
    // All four come off channels the tyre already publishes, and nothing here is a new tyre metric:
    // `lateralCapacity` is the same `tyreFriction` call `capacity` above is, on the other axis;
    // `gripUsed` and `peakSlip` are `TyreForces::gripUsed` and `TyreForces::longitudinalPeakSlip`
    // as the tyre reported them that tick; `slipAngle` is the slip the tyre was evaluated at.
    // They are recorded, never read by any decision.
    double lateralCapacity = 0.0;
    double gripUsed = 0.0;
    double peakSlip = 0.0;
    double slipAngle = 0.0;

    // --- added 2026-09-07 for section 17's split-mu trace ------------------------------------
    // The wheel's own angular speed as the state carries it, rad/s. Recorded, never read by any
    // decision; the controller sees the tone-ring sensor and not this.
    double wheelSpeed = 0.0;
};

struct Tick
{
    double time = 0.0;
    double speed = 0.0;
    double yaw = 0.0;
    double yawRate = 0.0;
    // Lateral displacement from where the pedal went down, metres. The steered fixture's primary
    // quantity, recorded per tick so the trajectories can be aligned rather than only compared at
    // the end.
    double lateralTravel = 0.0;
    // Lateral acceleration as the telemetry reports it, m/s^2 — section 17's split-mu trace asks
    // for it beside yaw rate. Recorded only.
    double lateralAcceleration = 0.0;
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
    double staticRearLoad = 1.0;
    bool stopped = false;
    std::size_t rearAirborneTicks = 0;

    std::size_t pastBandSteps = 0;
    std::size_t admittedSteps = 0;
    std::size_t qualifiedSteps = 0;
    std::size_t effectiveSteps = 0;
    std::size_t qualifiedStale = 0;

    Counts precision{};
    std::array<Counts, 2> byCaliper{};
    std::array<Counts, 5> byPhase{};
    std::array<Counts, cornerCount> byWheel{};
    std::array<std::size_t, cornerCount> pastBandByWheel{};
    std::vector<double> qualifiedAge;
};

[[nodiscard]] Run record(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                         const RoadTorqueCalibration& calibration, const Candidate& candidate, const double pedal,
                         const double entrySpeed, const double steering = 0.0, const bool traceSteps = false)
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

    truth.staticRearLoad =
        std::max(1.0, 0.5 * (lastStep.corners[2].forces.tireVertical + lastStep.corners[3].forces.tireVertical));

    // The settle runs the shadow with the candidate OFF, so anything it counted is not an
    // intervention. Cleared here so a run's counters describe the braking event alone.
    shadow.pastBandSteps = 0;
    shadow.admittedSteps = 0;
    shadow.qualifiedSteps = 0;
    shadow.effectiveSteps = 0;

    auto run = Run{};
    run.staticRearLoad = truth.staticRearLoad;
    run.entrySpeed = state.chassis.linearVelocity.z;

    run.ticks.reserve(360 * 8);

    const auto start = state.chassis.position;

    auto input = VehicleInput{};
    input.brake = pedal;
    input.steering = steering;

    const auto pressures = brakeCircuitPressures(setup, pedal);

    for (auto step = 0; step < 360 * 30; step++)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];

            truth.contact[index] = lastStep.telemetry.wheels[index].inContact;
            truth.load[index] = solution.forces.tireVertical;
            truth.longitudinal[index] = solution.contact.tyre.longitudinal;
            truth.effectiveRadius[index] = solution.contact.effectiveRadius;
            truth.slipRatio[index] = solution.contact.slip.slipRatio;
        }

        const auto brakes = shadowUpdate(assists, shadow, sense(), pedal, pressures, tick, calibration, candidate,
                                         truth, traceSteps ? &run.steps : nullptr, run.time);

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        run.time += tick;
        run.peakYawRate = std::max(run.peakYawRate, std::abs(lastStep.telemetry.yawRate));

        auto sample = Tick{};
        sample.time = run.time;
        sample.speed = state.chassis.linearVelocity.z;
        sample.yaw = lastStep.telemetry.yaw;
        sample.yawRate = lastStep.telemetry.yawRate;
        sample.lateralTravel = state.chassis.position.x - start.x;
        sample.lateralAcceleration = lastStep.telemetry.acceleration.x;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];
            auto& wheel = sample.wheels[index];

            wheel.load = solution.forces.tireVertical;
            wheel.longitudinal = solution.contact.tyre.longitudinal;
            wheel.lateral = solution.contact.tyre.lateral;
            wheel.capacity = tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, wheel.load,
                                          solution.patch.gripMultiplier) *
                             wheel.load;
            wheel.lateralCapacity =
                tyreFriction(setup.corners[index].tyre, TyreAxis::Lateral, wheel.load, solution.patch.gripMultiplier) *
                wheel.load;
            wheel.gripUsed = solution.contact.tyre.gripUsed;
            wheel.peakSlip = solution.contact.tyre.longitudinalPeakSlip;
            wheel.slipAngle = solution.contact.slip.slipAngle;
            wheel.slipRatio = solution.contact.slip.slipRatio;
            wheel.grip = solution.patch.gripMultiplier;
            wheel.pressure = shadow.assists.pressure[index];
            wheel.brakeTorqueCommanded = std::max(0.0, brakes.wheels[index]);
            wheel.phase = shadow.phase[index];
            wheel.inContact = lastStep.telemetry.wheels[index].inContact;
            wheel.wheelSpeed = state.corners[index].wheelSpeed;
        }

        run.rearAirborneTicks += (sample.wheels[2].inContact && sample.wheels[3].inContact) ? 0 : 1;

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

    run.pastBandSteps = shadow.pastBandSteps;
    run.admittedSteps = shadow.admittedSteps;
    run.qualifiedSteps = shadow.qualifiedSteps;
    run.effectiveSteps = shadow.effectiveSteps;
    run.qualifiedStale = shadow.qualifiedStale;
    run.precision = shadow.precision;
    run.byCaliper = shadow.byCaliper;
    run.byPhase = shadow.byPhase;
    run.byWheel = shadow.byWheel;
    run.pastBandByWheel = shadow.pastBandByWheel;
    run.qualifiedAge = std::move(shadow.qualifiedAge);

    return run;
}

// --- statistics ------------------------------------------------------------------------------

struct Distribution
{
    std::size_t n = 0;
    double minimum = 0.0;
    double p10 = 0.0;
    // Section 17 asks for the quartiles as well. Added 2026-09-07; every existing printer names its
    // fields explicitly, so nothing above this line changes by a character.
    double p25 = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double p75 = 0.0;
    double p90 = 0.0;
    double maximum = 0.0;
    double deviation = 0.0;
};

[[nodiscard]] double percentile(const std::vector<double>& sorted, const double fraction)
{
    if (sorted.empty())
    {
        return 0.0;
    }

    const auto position = fraction * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, sorted.size() - 1);

    return sorted[lower] + (position - static_cast<double>(lower)) * (sorted[upper] - sorted[lower]);
}

[[nodiscard]] Distribution distributionOf(std::vector<double> values)
{
    auto distribution = Distribution{};

    if (values.empty())
    {
        return distribution;
    }

    std::sort(values.begin(), values.end());

    distribution.n = values.size();
    distribution.minimum = values.front();
    distribution.maximum = values.back();
    distribution.p10 = percentile(values, 0.10);
    distribution.p25 = percentile(values, 0.25);
    distribution.median = percentile(values, 0.50);
    distribution.p75 = percentile(values, 0.75);
    distribution.p90 = percentile(values, 0.90);

    for (const auto value : values)
    {
        distribution.mean += value;
    }
    distribution.mean /= static_cast<double>(values.size());

    for (const auto value : values)
    {
        distribution.deviation += (value - distribution.mean) * (value - distribution.mean);
    }
    distribution.deviation = std::sqrt(distribution.deviation / static_cast<double>(values.size()));

    return distribution;
}

void distributionHeader(const char* what)
{
    std::printf("\n  %-38s %6s %10s %10s %10s %10s %10s %10s %10s\n", what, "n", "min", "P10", "MEDIAN", "mean", "P90",
                "max", "sd");
}

void distributionRow(const char* name, const Distribution& value)
{
    std::printf("  %-38s %6zu %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", name, value.n, value.minimum,
                value.p10, value.median, value.mean, value.p90, value.maximum, value.deviation);
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

[[nodiscard]] AssistSetup withoutSlipAwareRecovery(const AssistSetup& assists)
{
    auto copy = assists;
    copy.antilock.recoveryAuthority = RecoveryAuthority::Disabled;

    return copy;
}

[[nodiscard]] Candidate scoped(const Scope scope, const double accept = fixedAccept)
{
    auto candidate = Candidate{};
    candidate.enabled = true;
    candidate.accept = accept;
    candidate.rearOnly = true;
    candidate.scope = scope;

    return candidate;
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

// Seconds a wheel spent essentially stopped while the car was not. The split-mu mechanism's own
// measurement, and the reason a rear wheel that is given pressure back can cost yaw.
[[nodiscard]] double lockedTimeOf(const Run& run, const std::size_t wheel)
{
    auto ticks = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        ticks += (sample.wheels[wheel].inContact && std::abs(sample.wheels[wheel].slipRatio) > 0.90) ? 1 : 0;
    }

    return static_cast<double>(ticks) * tick;
}

[[nodiscard]] double meanOf(const Run& run, const std::size_t wheel, double WheelTick::* field,
                            const double until = 1e9)
{
    auto total = 0.0;
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        if (sample.time > until)
        {
            break;
        }

        total += sample.wheels[wheel].*field;
        count++;
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

[[nodiscard]] double meanMagnitudeOf(const Run& run, const std::size_t wheel, double WheelTick::* field,
                                     const double until = 1e9)
{
    auto total = 0.0;
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        if (sample.time > until)
        {
            break;
        }

        total += std::abs(sample.wheels[wheel].*field);
        count++;
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

// The share of a window in which the channel commanded nothing at all — the stranded state's own
// signature, measured rather than inferred from a utilisation number.
[[nodiscard]] double emptyShare(const Run& run, const std::size_t wheel, const double until)
{
    auto empty = std::size_t{0};
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        if (sample.time > until)
        {
            break;
        }

        empty += sample.wheels[wheel].pressure <= 0.0 ? 1 : 0;
        count++;
    }

    return count > 0 ? static_cast<double>(empty) / static_cast<double>(count) : 0.0;
}

[[nodiscard]] double yawAt(const Run& run, const double when)
{
    auto yaw = 0.0;

    for (const auto& sample : run.ticks)
    {
        if (sample.time > when)
        {
            break;
        }

        yaw = sample.yaw;
    }

    return yaw;
}

// --- the steered low-mu re-scoring's read-outs (2026-09-07) -----------------------------------
//
// Every one of these is arithmetic over the ticks `record` already stores. None of them is read by
// any controller decision anywhere in this file, and none of them introduces a tyre metric the
// tyre does not already publish.

[[nodiscard]] double axleMean(const Run& run, const std::size_t first, const std::size_t last,
                              double WheelTick::* field, const bool magnitude)
{
    auto total = 0.0;
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        for (auto index = first; index <= last; index++)
        {
            const auto value = sample.wheels[index].*field;
            total += magnitude ? std::abs(value) : value;
            count++;
        }
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

// Lateral force delivered against lateral capacity, summed over an axle over the whole event. The
// same construction `axleUse` uses for the longitudinal axis, on the other axis of the same tyre.
[[nodiscard]] double axleLateralUse(const Run& run, const std::size_t first, const std::size_t last)
{
    auto force = 0.0;
    auto capacity = 0.0;

    for (const auto& sample : run.ticks)
    {
        for (auto index = first; index <= last; index++)
        {
            force += std::abs(sample.wheels[index].lateral);
            capacity += sample.wheels[index].lateralCapacity;
        }
    }

    return capacity > 1.0 ? force / capacity : 0.0;
}

// The share of wheel-ticks an axle spends with its longitudinal slip past where its OWN curve
// peaked that tick, at that load and that grip. **Not a threshold this file picks**:
// `TyreForces::longitudinalPeakSlip` is the tyre saying where its force stops growing, and it moves
// with load and surface — which is the whole reason a fixed slip number would be the wrong question
// on mu 0.35.
[[nodiscard]] double axleBeyondPeak(const Run& run, const std::size_t first, const std::size_t last,
                                    const double multiple)
{
    auto past = std::size_t{0};
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        for (auto index = first; index <= last; index++)
        {
            const auto& wheel = sample.wheels[index];

            past += (wheel.peakSlip > 0.0 && std::abs(wheel.slipRatio) > multiple * wheel.peakSlip) ? 1 : 0;
            count++;
        }
    }

    return count > 0 ? static_cast<double>(past) / static_cast<double>(count) : 0.0;
}

// Sign changes in the yaw RATE, with a deadband so wobble either side of zero is not a reversal.
// A reporting statistic off the recorded path and nothing else.
[[nodiscard]] std::size_t yawReversalsOf(const Run& run, const double deadband)
{
    auto reversals = std::size_t{0};
    auto sign = 0;

    for (const auto& sample : run.ticks)
    {
        const auto now = sample.yawRate > deadband ? 1 : (sample.yawRate < -deadband ? -1 : 0);

        if (now != 0)
        {
            reversals += (sign != 0 && now != sign) ? 1 : 0;
            sign = now;
        }
    }

    return reversals;
}

// How long after the pedal went down the car first moved a tenth of a metre sideways. A reporting
// statistic off the recorded path; it is NOT a path-following controller metric and nothing is
// tuned against it.
[[nodiscard]] double lateralDelayOf(const Run& run, const double target)
{
    for (const auto& sample : run.ticks)
    {
        if (std::abs(sample.lateralTravel) >= target)
        {
            return sample.time;
        }
    }

    return 0.0;
}

[[nodiscard]] char phaseLetter(const ModulatorPhase phase)
{
    switch (phase)
    {
    case ModulatorPhase::Passive:
        return '.';
    case ModulatorPhase::Hold:
        return 'H';
    case ModulatorPhase::Dump:
        return 'D';
    case ModulatorPhase::Recover:
        return 'R';
    case ModulatorPhase::Reapply:
        return 'A';
    }

    return '?';
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
    double frontEngaged = 0.0;
    double rearEngaged = 0.0;
    double time = 0.0;
    bool stopped = false;

    // --- the steered low-mu re-scoring's read-outs (2026-09-07). Diagnostic, all of them. ------
    double meanSlipFront = 0.0;
    double meanSlipRear = 0.0;
    double meanSlipAll = 0.0;
    double gripUsedFront = 0.0;
    double gripUsedRear = 0.0;
    double lateralForceFront = 0.0;
    double lateralForceRear = 0.0;
    double lateralUseFront = 0.0;
    double lateralUseRear = 0.0;
    double beyondPeakFront = 0.0;
    double beyondPeakRear = 0.0;
    std::size_t yawReversals = 0;
    double lateralDelay = 0.0;

    std::size_t pastBandSteps = 0;
    std::size_t admittedSteps = 0;
    std::size_t qualifiedSteps = 0;
    std::size_t effectiveSteps = 0;
    std::size_t qualifiedStale = 0;

    Counts precision{};
    std::array<Counts, 2> byCaliper{};
    std::array<Counts, 5> byPhase{};
    std::array<Counts, cornerCount> byWheel{};
    std::array<std::size_t, cornerCount> pastBandByWheel{};
    std::vector<double> qualifiedAge;

    [[nodiscard]] bool stranded() const
    {
        return rearUtilisation < strandedRear;
    }
};

[[nodiscard]] Member memberOf(const std::size_t index, const double entry, const Run& run)
{
    const auto use = axleUse(run);

    auto member = Member{};
    member.index = index;
    member.entry = entry;
    member.distance = run.distance;
    member.frontUtilisation = use.front;
    member.rearUtilisation = use.rear;
    member.lateralTravel = run.lateralTravel;
    member.finalYaw = run.finalYaw;
    member.peakYawRate = run.peakYawRate;
    member.rearAirborne = run.rearAirborneTicks;
    member.frontCycles = cyclesOf(run, 0);
    member.rearCycles = cyclesOf(run, 2);
    member.frontEngaged = engagedOf(run, 0);
    member.rearEngaged = engagedOf(run, 2);
    member.time = run.time;
    member.stopped = run.stopped;

    member.meanSlipFront = axleMean(run, 0, 1, &WheelTick::slipRatio, true);
    member.meanSlipRear = axleMean(run, 2, 3, &WheelTick::slipRatio, true);
    member.meanSlipAll = axleMean(run, 0, 3, &WheelTick::slipRatio, true);
    member.gripUsedFront = axleMean(run, 0, 1, &WheelTick::gripUsed, false);
    member.gripUsedRear = axleMean(run, 2, 3, &WheelTick::gripUsed, false);
    member.lateralForceFront = axleMean(run, 0, 1, &WheelTick::lateral, true);
    member.lateralForceRear = axleMean(run, 2, 3, &WheelTick::lateral, true);
    member.lateralUseFront = axleLateralUse(run, 0, 1);
    member.lateralUseRear = axleLateralUse(run, 2, 3);
    member.beyondPeakFront = axleBeyondPeak(run, 0, 1, 1.0);
    member.beyondPeakRear = axleBeyondPeak(run, 2, 3, 1.0);
    member.yawReversals = yawReversalsOf(run, 0.02);
    member.lateralDelay = lateralDelayOf(run, 0.10);

    member.pastBandSteps = run.pastBandSteps;
    member.admittedSteps = run.admittedSteps;
    member.qualifiedSteps = run.qualifiedSteps;
    member.effectiveSteps = run.effectiveSteps;
    member.qualifiedStale = run.qualifiedStale;
    member.precision = run.precision;
    member.byCaliper = run.byCaliper;
    member.byPhase = run.byPhase;
    member.byWheel = run.byWheel;
    member.pastBandByWheel = run.pastBandByWheel;
    member.qualifiedAge = run.qualifiedAge;

    return member;
}

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
        members.push_back(
            memberOf(index, entry, record(setup, world, assists, calibration, candidate, pedal, entry, steering)));
    }

    return members;
}

struct ArmSummary
{
    std::string name;
    Distribution distance;
    Distribution rearUtilisation;
    Distribution frontUtilisation;
    std::size_t stranded = 0;
    std::size_t n = 0;
    double meanRearCycles = 0.0;
    double meanPastBand = 0.0;
    double meanAdmitted = 0.0;
    double meanQualified = 0.0;
    double meanEffective = 0.0;
    std::size_t stale = 0;
    Counts precision{};
    std::array<Counts, 2> byCaliper{};
    std::array<Counts, 5> byPhase{};
    std::array<Counts, cornerCount> byWheel{};
    std::vector<double> ages;
};

[[nodiscard]] ArmSummary summarise(const std::string& name, const std::vector<Member>& members)
{
    auto summary = ArmSummary{};
    summary.name = name;

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
        summary.meanRearCycles += static_cast<double>(member.rearCycles);
        summary.meanPastBand += static_cast<double>(member.pastBandSteps);
        summary.meanAdmitted += static_cast<double>(member.admittedSteps);
        summary.meanQualified += static_cast<double>(member.qualifiedSteps);
        summary.meanEffective += static_cast<double>(member.effectiveSteps);
        summary.stale += member.qualifiedStale;
        summary.precision.add(member.precision);

        for (auto index = std::size_t{0}; index < 2; index++)
        {
            summary.byCaliper[index].add(member.byCaliper[index]);
        }
        for (auto index = std::size_t{0}; index < 5; index++)
        {
            summary.byPhase[index].add(member.byPhase[index]);
        }
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            summary.byWheel[index].add(member.byWheel[index]);
        }

        summary.ages.insert(summary.ages.end(), member.qualifiedAge.begin(), member.qualifiedAge.end());
    }

    const auto count = static_cast<double>(std::max(members.size(), std::size_t{1}));
    summary.n = members.size();
    summary.meanRearCycles /= count;
    summary.meanPastBand /= count;
    summary.meanAdmitted /= count;
    summary.meanQualified /= count;
    summary.meanEffective /= count;
    summary.distance = distributionOf(distances);
    summary.rearUtilisation = distributionOf(rear);
    summary.frontUtilisation = distributionOf(front);

    return summary;
}

void printArmHeader()
{
    std::printf("\n  %-38s  stranded      P10   MEDIAN      P90       sd   rear util  front util   rearCyc"
                "   pastBand  admitted  changed  effective\n",
                "arm");
}

void printArm(const ArmSummary& summary)
{
    std::printf("  %-38s   %2zu/%2zu   %7.3f  %7.3f  %7.3f  %7.3f     %6.3f      %6.3f   %6.1f   %8.0f  %8.0f "
                "%8.0f   %8.0f\n",
                summary.name.c_str(), summary.stranded, summary.n, summary.distance.p10, summary.distance.median,
                summary.distance.p90, summary.distance.deviation, summary.rearUtilisation.median,
                summary.frontUtilisation.median, summary.meanRearCycles, summary.meanPastBand, summary.meanAdmitted,
                summary.meanQualified, summary.meanEffective);
}

void printPrecisionHeader()
{
    std::printf("\n  %-38s %8s   %8s %8s %8s %8s   %8s\n", "arm", "changed", "INTENDED", "low load", "high slip",
                "other", "precision");
}

void printPrecision(const char* name, const Counts& counts)
{
    const auto total = static_cast<double>(std::max(counts.total(), std::size_t{1}));

    std::printf("  %-38s %8zu   %8zu %8zu %8zu %8zu   %7.1f%%\n", name, counts.total(), counts.intended, counts.lowLoad,
                counts.highSlip, counts.other, 100.0 * static_cast<double>(counts.intended) / total);
}

void printSlipFirst(const char* name, const Counts& counts)
{
    std::printf("  %-38s   rolling %6zu   in band %6zu   slipping %6zu   airborne %6zu\n", name, counts.rolling,
                counts.band, counts.slipping, counts.airborne);
}

} // namespace

// =============================================================================================
// 0. THE SHADOW CONTROLLER IS THE PRODUCTION CONTROLLER
// =============================================================================================

TEST_CASE("the scoping shadow with every arm off is the production controller, to the bit", "[.road-torque-scope]")
{
    // **Nothing below this case means anything without it.** This file carries its own copy of
    // `updateAssists`'s outer loop so that the qualification can act between the channel's decision
    // and the caliper; this is what says that copy is the same loop.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto off = Candidate{};

    std::printf("\n=== the scoping shadow against production `updateAssists`, %zu members ===\n", ensembleCount);

    auto identical = std::size_t{0};

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);
        const auto shadow = record(setup.value(), world.value(), assists, calibration, off, 1.0, entry);

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), entry);

        auto assistState = AssistState{};
        auto lastStep = VehicleStep{};
        auto force = std::array<double, 2>{};
        auto capacity = std::array<double, 2>{};

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

            for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
            {
                const auto& solution = lastStep.corners[corner];
                const auto axle = corner < 2 ? std::size_t{0} : std::size_t{1};

                force[axle] += std::abs(solution.contact.tyre.longitudinal);
                capacity[axle] += tyreFriction(setup->corners[corner].tyre, TyreAxis::Longitudinal,
                                               solution.forces.tireVertical, solution.patch.gripMultiplier) *
                                  solution.forces.tireVertical;
            }

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

        identical += (distance == shadow.distance && rearUse == shadowUse.rear) ? 1 : 0;
    }

    std::printf("  identical on distance and rear utilisation: %zu of %zu\n", identical, ensembleCount);

    REQUIRE(identical == ensembleCount);
}

// =============================================================================================
// 1-2. THE AVAILABLE STATE, AND "CALIPER EMPTY" READ OFF THE EXISTING ACTUATOR BOUND
// =============================================================================================

TEST_CASE("what the channel already knows at the decision point, and how often its caliper is empty",
          "[.road-torque-scope]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    std::printf("\n=== SECTION 1: the state available at the decision point, and nothing else ===\n");
    std::printf("  Read straight off `advanceAntilockChannel`'s parameter list and\n");
    std::printf("  `AntilockChannelState`. No new signal is added anywhere in this file.\n\n");
    std::printf("    AntilockChannelState::phase             Passive|Hold|Dump|Recover|Reapply\n");
    std::printf("    AntilockChannelState::pressure          what the channel commands, Pa\n");
    std::printf("    AntilockChannelState::departurePressure the pressure the wheel last departed at, Pa\n");
    std::printf("    AntilockChannelState::acceleration      peripheral, from the last two teeth, m/s^2\n");
    std::printf("    AntilockChannelState::sinceUpdate       age of that reading, s (the pulse age)\n");
    std::printf("    AntilockChannelState::lastSpeed         the held reading, m/s\n");
    std::printf("    AntilockChannelState::lastPulses        the tooth count it was taken at\n");
    std::printf("    AntilockChannelState::surged            has the wheel surged since the dump ended\n");
    std::printf("    AntilockChannelState::cycles            entries to Dump\n");
    std::printf("    argument requestedPressure              what upstream wants, Pa\n");
    std::printf("    argument referenceSpeed / rate / valid  the estimator's own belief\n");
    std::printf("    argument wheel.valid / wheel.pulses     the tone ring\n");
    std::printf("  DERIVED IN THE LAW ITSELF: slip, guardSlip, excess, losing, surging, pastBand.\n");
    std::printf("  NOT AVAILABLE and not used: Fz, contact, true slip, road speed, tyre force,\n");
    std::printf("  suspension state, surface mu.\n");

    std::printf("\n=== SECTION 2: `caliper empty` is an EXISTING discrete bound, not a new threshold ===\n");
    std::printf("  `BrakeControlImpl`'s dump branch is\n\n");
    std::printf("      state.pressure = std::max(0.0, state.pressure - dumpGradient * deltaTime);\n\n");
    std::printf("  so the actuator has a hard LOWER BOUND at zero and reaches it exactly. Sitting on\n");
    std::printf("  that bound is the state `no further dump authority remains`, stated by the\n");
    std::printf("  implementation. The test below is `pressure <= 0.0` on the value the decision is\n");
    std::printf("  taken against, and nothing else.\n");

    // How reachable that bound is, and how much of the failing state lives on it.
    const auto stranded = std::size_t{4};
    const auto healthy = std::size_t{27};

    for (const auto member : {stranded, healthy})
    {
        const auto entry = memberEntry(member, ensembleCount);
        const auto run = record(setup.value(), world.value(), assists, calibration, Candidate{}, 1.0, entry, 0.0, true);

        auto rearSteps = std::size_t{0};
        auto rearEmpty = std::size_t{0};
        auto rearPastBand = std::size_t{0};
        auto rearPastBandEmpty = std::size_t{0};
        auto exactZero = std::size_t{0};
        auto nearZero = std::size_t{0};
        auto byPhaseEmpty = std::array<std::size_t, 5>{};
        auto byPhaseAll = std::array<std::size_t, 5>{};

        for (const auto& step : run.steps)
        {
            if (step.channel != 2)
            {
                continue;
            }

            rearSteps++;
            byPhaseAll[phaseIndex(step.before)]++;

            if (step.pressureBefore <= 0.0)
            {
                rearEmpty++;
                exactZero += step.pressureBefore == 0.0 ? 1 : 0;
                byPhaseEmpty[phaseIndex(step.before)]++;
            }
            else if (step.pressureBefore < 1.0 * bar)
            {
                nearZero++;
            }

            if (step.pastBandUnqualified)
            {
                rearPastBand++;
                rearPastBandEmpty += step.pressureBefore <= 0.0 ? 1 : 0;
            }
        }

        std::printf("\n  --- member %zu at %.3f km/h (%s), rear channel, %.3f m ---\n", member, 3.6 * entry,
                    member == stranded ? "production-STRANDED" : "production-recovered", run.distance);
        std::printf("    rear controller steps                        %8zu\n", rearSteps);
        std::printf("    ...with pressure <= 0                        %8zu  (%.1f%%)\n", rearEmpty,
                    100.0 * static_cast<double>(rearEmpty) / static_cast<double>(std::max(rearSteps, std::size_t{1})));
        std::printf("    ...of which EXACTLY 0.0 Pa                   %8zu\n", exactZero);
        std::printf("    ...with 0 < pressure < 1 bar                 %8zu\n", nearZero);
        std::printf("    ...where pastBand fired                      %8zu\n", rearPastBand);
        std::printf("    ...where pastBand fired AND caliper empty    %8zu  (%.1f%% of pastBand)\n", rearPastBandEmpty,
                    100.0 * static_cast<double>(rearPastBandEmpty) /
                        static_cast<double>(std::max(rearPastBand, std::size_t{1})));

        std::printf("    empty-caliper steps by phase:");
        for (auto index = std::size_t{0}; index < 5; index++)
        {
            const auto phase = index == 0   ? ModulatorPhase::Passive
                               : index == 1 ? ModulatorPhase::Hold
                               : index == 2 ? ModulatorPhase::Dump
                               : index == 3 ? ModulatorPhase::Recover
                                            : ModulatorPhase::Reapply;
            std::printf("  %s %zu/%zu", modulatorName(phase), byPhaseEmpty[index], byPhaseAll[index]);
        }
        std::printf("\n");
    }

    std::printf("\n  The bound is reached EXACTLY, so the condition needs no tolerance and no bar\n");
    std::printf("  figure. `pressure < 1 bar but not zero` is printed beside it only to show that a\n");
    std::printf("  near-empty band would not add a materially different population.\n");
}

// =============================================================================================
// 3-4. DOES EMPTY-CALIPER / PHASE SEPARATE THE INTENDED FROM THE UNINTENDED?
// =============================================================================================

TEST_CASE("the broad candidate's changed decisions, partitioned by caliper state and by phase", "[.road-torque-scope]")
{
    // **The scoping hypothesis stands or falls here, before any arm is run.** If the intended
    // loaded/rolling overrides are concentrated in the empty-caliper state and the high-slip
    // low-load ones are not, scoping can work. If both populations live in the same state, no
    // amount of gating separates them and the hypothesis is falsified on the data.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto broad = scoped(Scope::Everywhere);

    std::printf("\n=== the broad candidate at %.3f x peak (rear peak %.1f N.m), partitioned ===\n", fixedAccept,
                calibration.peakTorque[2]);
    std::printf("  Labels are DIAGNOSTIC TRUTH and are used offline only:\n");
    std::printf("    INTENDED   in contact, load >= %.2f x static, |true slip| < %.2f  (false pastBand)\n",
                loadedFraction, lowSlip);
    std::printf("    low load   airborne, or load < %.2f x static\n", loadedFraction);
    std::printf("    high slip  loaded and |true slip| > %.2f\n", highSlip);
    std::printf("    other      loaded and inside the band\n");

    // The representative stranded member first, so the figures line up with `[.road-torque]`'s
    // event audit.
    const auto member = std::size_t{4};
    const auto entry = memberEntry(member, ensembleCount);
    const auto run = record(setup.value(), world.value(), assists, calibration, broad, 1.0, entry, 0.0, true);

    std::printf("\n  --- member %zu at %.3f km/h: %zu changed decisions ---\n", member, 3.6 * entry,
                run.qualifiedSteps);

    printPrecisionHeader();
    printPrecision("all changed decisions", run.precision);
    printPrecision("  ...with the caliper EMPTY", run.byCaliper[1]);
    printPrecision("  ...with the caliper NON-EMPTY", run.byCaliper[0]);

    std::printf("\n  the same, in `[.road-torque]`'s own slip-first partition:\n");
    printSlipFirst("all changed decisions", run.precision);
    printSlipFirst("  ...caliper EMPTY", run.byCaliper[1]);
    printSlipFirst("  ...caliper NON-EMPTY", run.byCaliper[0]);

    std::printf("\n  --- SECTION 4: the same decisions by the phase they were taken in ---\n");
    printPrecisionHeader();
    for (auto index = std::size_t{0}; index < 5; index++)
    {
        const auto phase = index == 0   ? ModulatorPhase::Passive
                           : index == 1 ? ModulatorPhase::Hold
                           : index == 2 ? ModulatorPhase::Dump
                           : index == 3 ? ModulatorPhase::Recover
                                        : ModulatorPhase::Reapply;

        auto name = std::string{"phase "};
        name += modulatorName(phase);
        printPrecision(name.c_str(), run.byPhase[index]);
    }

    std::printf("\n  `Passive` and `Hold` do not read `pastBand` at all, so a qualification there is\n");
    std::printf("  inert and only inflates the changed-decision count.\n");

    // And the two-way split the hypothesis actually asks for.
    std::printf("\n  --- the joint question: phase x caliper, on this member ---\n");
    std::printf("  %-30s %10s %10s %10s %10s\n", "cell", "INTENDED", "low load", "high slip", "other");

    for (auto index = std::size_t{0}; index < 5; index++)
    {
        const auto phase = index == 0   ? ModulatorPhase::Passive
                           : index == 1 ? ModulatorPhase::Hold
                           : index == 2 ? ModulatorPhase::Dump
                           : index == 3 ? ModulatorPhase::Recover
                                        : ModulatorPhase::Reapply;

        auto empty = Counts{};
        auto full = Counts{};

        for (const auto& step : run.steps)
        {
            if (!step.qualified || step.before != phase)
            {
                continue;
            }

            classify(step.emptyCaliper ? empty : full, step.trueContact, step.trueLoad, step.trueSlip,
                     run.staticRearLoad);
        }

        std::printf("  %-24s EMPTY %10zu %10zu %10zu %10zu\n", modulatorName(phase), empty.intended, empty.lowLoad,
                    empty.highSlip, empty.other);
        std::printf("  %-24s full  %10zu %10zu %10zu %10zu\n", modulatorName(phase), full.intended, full.lowLoad,
                    full.highSlip, full.other);
    }

    // The whole dry ensemble, so the answer is not one member's.
    const auto ensemble =
        summarise("broad, whole ensemble", ensembleOf(setup.value(), world.value(), assists, calibration, broad, 1.0));

    std::printf("\n  --- the same partition over all %zu dry members ---\n", ensembleCount);
    printPrecisionHeader();
    printPrecision("all changed decisions", ensemble.precision);
    printPrecision("  ...with the caliper EMPTY", ensemble.byCaliper[1]);
    printPrecision("  ...with the caliper NON-EMPTY", ensemble.byCaliper[0]);

    printPrecisionHeader();
    for (auto index = std::size_t{0}; index < 5; index++)
    {
        const auto phase = index == 0   ? ModulatorPhase::Passive
                           : index == 1 ? ModulatorPhase::Hold
                           : index == 2 ? ModulatorPhase::Dump
                           : index == 3 ? ModulatorPhase::Recover
                                        : ModulatorPhase::Reapply;

        auto name = std::string{"phase "};
        name += modulatorName(phase);
        printPrecision(name.c_str(), ensemble.byPhase[index]);
    }
}

// =============================================================================================
// 5-7. THE ARCHITECTURE ARMS ON THE DRY ENSEMBLE
// =============================================================================================

namespace
{

[[nodiscard]] std::vector<ArmSummary> dryArms(const VehicleSetup& setup, const PhysicsWorld& world,
                                              const AssistSetup& assists, const RoadTorqueCalibration& calibration)
{
    auto arms = std::vector<ArmSummary>{};

    arms.push_back(summarise("production (law ON)", ensembleOf(setup, world, assists, calibration, Candidate{}, 1.0)));
    arms.push_back(summarise("slipAwareRecovery OFF", ensembleOf(setup, world, withoutSlipAwareRecovery(assists),
                                                                 calibration, Candidate{}, 1.0)));

    for (const auto scope : {Scope::Everywhere, Scope::RecoverOnly, Scope::EmptyOnly, Scope::RecoverAndEmpty,
                             Scope::StuckDecisionAndEmpty})
    {
        arms.push_back(summarise(scopeName(scope), ensembleOf(setup, world, assists, calibration, scoped(scope), 1.0)));
    }

    return arms;
}

} // namespace

TEST_CASE("the five architecture scopes on the dry ensemble, at one fixed threshold", "[.road-torque-scope]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    std::printf("\n=== DRY, %zu members, full pedal. Road-torque threshold FIXED at %.3f x peak ===\n", ensembleCount,
                fixedAccept);
    std::printf("  The only thing that differs between the scoped arms is WHERE the qualification is\n");
    std::printf("  allowed to act. Nothing here is optimised; stopping distance is not a target.\n");
    std::printf("  Primary requirement: preserve the elimination of the stranded cluster.\n");

    const auto arms = dryArms(setup.value(), world.value(), assists, calibration);

    printArmHeader();
    for (const auto& arm : arms)
    {
        printArm(arm);
    }

    std::printf("\n  --- intervention precision, whole dry ensemble ---\n");
    printPrecisionHeader();
    for (const auto& arm : arms)
    {
        if (arm.precision.total() > 0)
        {
            printPrecision(arm.name.c_str(), arm.precision);
        }
    }

    std::printf("\n  --- the same arms in the slip-first partition ---\n");
    for (const auto& arm : arms)
    {
        if (arm.precision.total() > 0)
        {
            printSlipFirst(arm.name.c_str(), arm.precision);
        }
    }

    std::printf("\n  --- SECTION 6 robustness: the two scoped arms at 0.05 / 0.15 / 0.30 ---\n");
    printArmHeader();

    for (const auto scope : {Scope::RecoverAndEmpty, Scope::StuckDecisionAndEmpty})
    {
        for (const auto accept : {0.05, 0.15, 0.30})
        {
            auto name = std::string{scopeName(scope)};
            name += " @ ";
            name += std::to_string(accept).substr(0, 4);

            printArm(summarise(
                name, ensembleOf(setup.value(), world.value(), assists, calibration, scoped(scope, accept), 1.0)));
        }
    }

    std::printf("\n  Three points, not a sweep. The scoping comparison above is at one threshold.\n");
}

// =============================================================================================
// 8. SPLIT MU — THE DECIDING TEST
// =============================================================================================

namespace
{

void printSurfaceHeader(const char* what)
{
    std::printf("\n  %-38s   %8s %8s %8s   %8s %8s   %8s   %8s\n", what, "stop P10", "MEDIAN", "P90", "front u",
                "rear u", "r cyc/s", "changed");
}

void printSurfaceArm(const std::string& name, const std::vector<Member>& members)
{
    auto distances = std::vector<double>{};
    auto front = std::vector<double>{};
    auto rear = std::vector<double>{};
    auto rearRate = std::vector<double>{};
    auto changed = 0.0;

    for (const auto& member : members)
    {
        distances.push_back(member.distance);
        front.push_back(member.frontUtilisation);
        rear.push_back(member.rearUtilisation);
        rearRate.push_back(member.rearEngaged > 0.0 ? static_cast<double>(member.rearCycles) / member.rearEngaged
                                                    : 0.0);
        changed += static_cast<double>(member.qualifiedSteps);
    }

    const auto stop = distributionOf(distances);

    std::printf("  %-38s   %8.3f %8.3f %8.3f   %8.4f %8.4f   %8.2f   %8.0f\n", name.c_str(), stop.p10, stop.median,
                stop.p90, distributionOf(front).median, distributionOf(rear).median, distributionOf(rearRate).median,
                changed / static_cast<double>(std::max(members.size(), std::size_t{1})));
}

void printYawArm(const std::string& name, const std::vector<Member>& members)
{
    auto yaw = std::vector<double>{};
    auto lateral = std::vector<double>{};
    auto rate = std::vector<double>{};
    auto inside = std::size_t{0};

    for (const auto& member : members)
    {
        yaw.push_back(std::abs(member.finalYaw) * degrees);
        lateral.push_back(std::abs(member.lateralTravel));
        rate.push_back(member.peakYawRate);
        inside += std::abs(member.finalYaw) < 45.0 / degrees ? 1 : 0;
    }

    const auto distribution = distributionOf(yaw);

    std::printf("  %-38s   %9.3f %9.3f %9.3f %9.3f   %9.3f %9.4f   %6zu/%zu\n", name.c_str(), distribution.p10,
                distribution.median, distribution.p90, distribution.maximum, distributionOf(lateral).median,
                distributionOf(rate).median, inside, members.size());
}

} // namespace

TEST_CASE("the five architecture scopes on the split-mu ensemble", "[.road-torque-scope]")
{
    // No yaw control is redesigned and `yawMomentDelay` is off on every arm.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    REQUIRE_FALSE(assists.antilock.yawMomentDelay);

    std::printf("\n=== SPLIT MU (1.00 at x<0, 0.35 at x>0), %zu members, full pedal ===\n", ensembleCount);
    std::printf("  The primary question: does empty-caliper / recovery scoping remove the MONOTONIC\n");
    std::printf("  yaw penalty that broad road-torque qualification carries?\n");

    struct Arm
    {
        std::string name;
        std::vector<Member> members;
    };

    auto arms = std::vector<Arm>{};

    arms.push_back(
        Arm{"production (law ON)", ensembleOf(setup.value(), split.value(), assists, calibration, Candidate{}, 1.0)});
    arms.push_back(
        Arm{"slipAwareRecovery OFF", ensembleOf(setup.value(), split.value(), withoutSlipAwareRecovery(assists),
                                                calibration, Candidate{}, 1.0)});

    for (const auto scope : {Scope::Everywhere, Scope::RecoverOnly, Scope::EmptyOnly, Scope::RecoverAndEmpty,
                             Scope::StuckDecisionAndEmpty})
    {
        arms.push_back(
            Arm{scopeName(scope), ensembleOf(setup.value(), split.value(), assists, calibration, scoped(scope), 1.0)});
    }

    printSurfaceHeader("arm");
    for (const auto& arm : arms)
    {
        printSurfaceArm(arm.name, arm.members);
    }

    std::printf("\n  --- the directional half ---\n");
    std::printf("  %-38s   %9s %9s %9s %9s   %9s %9s   %9s\n", "arm", "yaw P10", "yaw MED", "yaw P90", "yaw MAX",
                "lateral", "peak rate", "in 45 deg");

    for (const auto& arm : arms)
    {
        printYawArm(arm.name, arm.members);
    }

    std::printf("\n  --- changed decisions by which rear wheel the select-low channel was on ---\n");
    std::printf("  The rear channel is SELECT-LOW, so its control wheel says which side the decision\n");
    std::printf("  was taken from. **Which corner index sits on which surface is measured, not\n");
    std::printf("  asserted from the frame** -- see the mechanism trace, which reads it off the patch\n");
    std::printf("  grip in the first 0.3 s.\n");
    std::printf("  %-38s %10s %10s %10s %10s %10s %10s\n", "arm", "chg c2", "chg c3", "pastB c2", "pastB c3", "empty",
                "non-empty");

    for (const auto& arm : arms)
    {
        auto second = Counts{};
        auto third = Counts{};
        auto empty = Counts{};
        auto full = Counts{};
        auto pastBand = std::array<std::size_t, cornerCount>{};

        for (const auto& member : arm.members)
        {
            second.add(member.byWheel[2]);
            third.add(member.byWheel[3]);
            empty.add(member.byCaliper[1]);
            full.add(member.byCaliper[0]);

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                pastBand[index] += member.pastBandByWheel[index];
            }
        }

        if (second.total() + third.total() == 0)
        {
            continue;
        }

        std::printf("  %-38s %10zu %10zu %10zu %10zu %10zu %10zu\n", arm.name.c_str(), second.total(), third.total(),
                    pastBand[2], pastBand[3], empty.total(), full.total());
    }

    std::printf("\n  `pastB` is every step where `pastBand` fired on that control wheel, changed or not.\n");
    std::printf("  It is the baseline the changed-decision split has to be read against: a scope that\n");
    std::printf("  merely fires proportionally is not selecting a side.\n");

    std::printf("\n  --- intervention precision on split mu ---\n");
    printPrecisionHeader();
    for (const auto& arm : arms)
    {
        auto counts = Counts{};
        for (const auto& member : arm.members)
        {
            counts.add(member.precision);
        }

        if (counts.total() > 0)
        {
            printPrecision(arm.name.c_str(), counts);
        }
    }
}

// =============================================================================================
// 9. WHY BROAD QUALIFICATION COSTS YAW
// =============================================================================================

TEST_CASE("one split-mu member, traced: production against broad against scoped", "[.road-torque-scope]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto entry = memberEntry(ensembleCount / 2, ensembleCount);

    std::printf("\n=== one split-mu member at %.3f km/h, four arms ===\n", 3.6 * entry);
    std::printf("  ONLY the difference this qualification causes is explained. No general split-mu\n");
    std::printf("  investigation is performed and no yaw control is touched.\n");

    struct Arm
    {
        std::string name;
        Candidate candidate;
        bool lawOff;
    };

    const auto arms = std::array<Arm, 4>{Arm{"production (law ON)", Candidate{}, false},
                                         Arm{"slipAwareRecovery OFF", Candidate{}, true},
                                         Arm{"A broad", scoped(Scope::Everywhere), false},
                                         Arm{"D Recover AND empty", scoped(Scope::RecoverAndEmpty), false}};

    auto runs = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        const auto used = arm.lawOff ? withoutSlipAwareRecovery(assists) : assists;
        auto run = record(setup.value(), split.value(), used, calibration, arm.candidate, 1.0, entry);
        REQUIRE(run.stopped);
        runs.push_back(std::move(run));
    }

    // **Which corner is on which surface, measured and not asserted from the frame.** The plate is
    // grip 1.00 at x < 0 and 0.35 at x > 0; which corner index that is depends on the vehicle
    // frame's handedness, so it is read off the first 0.3 s of the production run, before the car
    // has yawed far enough to move a wheel across the boundary.
    std::printf("\n  --- which corner starts on which surface (mean patch grip, first 0.3 s) ---\n");
    auto lowMuRear = std::size_t{2};
    auto highMuRear = std::size_t{3};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("    corner %zu   grip %.4f   (whole stop %.4f)\n", index,
                    meanOf(runs[0], index, &WheelTick::grip, 0.3), meanOf(runs[0], index, &WheelTick::grip));
    }

    if (meanOf(runs[0], 3, &WheelTick::grip, 0.3) < meanOf(runs[0], 2, &WheelTick::grip, 0.3))
    {
        lowMuRear = 3;
        highMuRear = 2;
    }

    std::printf("    => LOW-mu rear corner is %zu, HIGH-mu rear corner is %zu\n", lowMuRear, highMuRear);

    // **The yaw is built early, so the whole-stop mean hides the mechanism.** Everything below is
    // reported over the first second as well as over the stop.
    std::printf("\n  --- the stop, and when the yaw is built ---\n");
    std::printf("  %-24s %8s   %9s %9s %9s %9s\n", "arm", "stop m", "yaw 0.5s", "yaw 1.0s", "yaw 2.0s", "yaw end");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("  %-24s %8.3f   %9.3f %9.3f %9.3f %9.3f\n", arms[index].name.c_str(), runs[index].distance,
                    degrees * yawAt(runs[index], 0.5), degrees * yawAt(runs[index], 1.0),
                    degrees * yawAt(runs[index], 2.0), degrees * runs[index].finalYaw);
    }

    std::printf("\n  --- the rear axle over the first second, which is where the yaw comes from ---\n");
    std::printf("  %-24s %10s %9s   %9s %9s   %9s %9s\n", "arm", "rear N.m", "empty %", "|slip| lo", "|slip| hi",
                "|Fy| lo N", "|Fy| hi N");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& run = runs[index];

        std::printf("  %-24s %10.2f %9.1f   %9.4f %9.4f   %9.1f %9.1f\n", arms[index].name.c_str(),
                    meanOf(run, lowMuRear, &WheelTick::brakeTorqueCommanded, 1.0), 100.0 * emptyShare(run, 2, 1.0),
                    meanMagnitudeOf(run, lowMuRear, &WheelTick::slipRatio, 1.0),
                    meanMagnitudeOf(run, highMuRear, &WheelTick::slipRatio, 1.0),
                    meanMagnitudeOf(run, lowMuRear, &WheelTick::lateral, 1.0),
                    meanMagnitudeOf(run, highMuRear, &WheelTick::lateral, 1.0));
    }

    std::printf("\n  `rear N.m` is the rear channel's commanded brake torque -- one channel drives both\n");
    std::printf("  rear wheels, so there is NO left/right rear pressure asymmetry to explain the yaw\n");
    std::printf("  with. `empty %%` is the share of the first second on the actuator's lower bound.\n");

    std::printf("\n  --- the same over the whole stop ---\n");
    std::printf("  %-24s %10s %9s   %9s %9s   %9s %9s   %8s %8s\n", "arm", "rear N.m", "empty %", "|slip| lo",
                "|slip| hi", "|Fy| lo N", "|Fy| hi N", "lock lo", "lock hi");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& run = runs[index];

        std::printf("  %-24s %10.2f %9.1f   %9.4f %9.4f   %9.1f %9.1f   %8.4f %8.4f\n", arms[index].name.c_str(),
                    meanOf(run, lowMuRear, &WheelTick::brakeTorqueCommanded), 100.0 * emptyShare(run, 2, 1e9),
                    meanMagnitudeOf(run, lowMuRear, &WheelTick::slipRatio),
                    meanMagnitudeOf(run, highMuRear, &WheelTick::slipRatio),
                    meanMagnitudeOf(run, lowMuRear, &WheelTick::lateral),
                    meanMagnitudeOf(run, highMuRear, &WheelTick::lateral), lockedTimeOf(run, lowMuRear),
                    lockedTimeOf(run, highMuRear));
    }

    std::printf("\n  --- the front axle, which the rear-only qualification never touches directly ---\n");
    std::printf("  %-24s %10s %10s   %10s %10s   %8s %8s\n", "arm", "front lo", "front hi", "|Fy| lo N", "|Fy| hi N",
                "lock lo", "lock hi");

    const auto lowMuFront = lowMuRear == 2 ? std::size_t{0} : std::size_t{1};
    const auto highMuFront = lowMuFront == 0 ? std::size_t{1} : std::size_t{0};

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& run = runs[index];

        std::printf("  %-24s %10.2f %10.2f   %10.1f %10.1f   %8.4f %8.4f\n", arms[index].name.c_str(),
                    meanOf(run, lowMuFront, &WheelTick::brakeTorqueCommanded),
                    meanOf(run, highMuFront, &WheelTick::brakeTorqueCommanded),
                    meanMagnitudeOf(run, lowMuFront, &WheelTick::lateral),
                    meanMagnitudeOf(run, highMuFront, &WheelTick::lateral), lockedTimeOf(run, lowMuFront),
                    lockedTimeOf(run, highMuFront));
    }

    std::printf("\n  A front left/right difference here is a CONSEQUENCE of the yaw and of the load it\n");
    std::printf("  moves, not its cause: the qualification is rear-only on every arm.\n");
}

// =============================================================================================
// 10. LOW MU
// =============================================================================================

TEST_CASE("the five architecture scopes on the uniform low-mu ensemble", "[.road-torque-scope]")
{
    // The low-mu red is not touched. What this answers is whether a scoped arm creates a NEW
    // regression on a uniformly slippery surface.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto slippery = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(slippery.has_value());

    const auto assists = withAntilock(setup.value());
    const auto plain = golfGtiMk7Assists(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    std::printf("\n=== LOW MU (0.35 everywhere), %zu members, full pedal ===\n", ensembleCount);

    struct Arm
    {
        std::string name;
        std::vector<Member> members;
    };

    auto arms = std::vector<Arm>{};

    arms.push_back(Arm{"no electronics (locked wheels)",
                       ensembleOf(setup.value(), slippery.value(), plain, calibration, Candidate{}, 1.0)});
    arms.push_back(Arm{"production (law ON)",
                       ensembleOf(setup.value(), slippery.value(), assists, calibration, Candidate{}, 1.0)});
    arms.push_back(
        Arm{"slipAwareRecovery OFF", ensembleOf(setup.value(), slippery.value(), withoutSlipAwareRecovery(assists),
                                                calibration, Candidate{}, 1.0)});

    for (const auto scope : {Scope::Everywhere, Scope::RecoverOnly, Scope::EmptyOnly, Scope::RecoverAndEmpty,
                             Scope::StuckDecisionAndEmpty})
    {
        arms.push_back(Arm{scopeName(scope),
                           ensembleOf(setup.value(), slippery.value(), assists, calibration, scoped(scope), 1.0)});
    }

    printSurfaceHeader("arm");
    for (const auto& arm : arms)
    {
        printSurfaceArm(arm.name, arm.members);
    }

    std::printf("\n  --- what the anti-lock system is worth here, PAIRED member by member ---\n");

    const auto& locked = arms[0].members;

    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        auto worth = std::vector<double>{};
        auto positive = std::size_t{0};

        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            const auto value =
                100.0 * (locked[member].distance - arms[index].members[member].distance) / locked[member].distance;
            worth.push_back(value);
            positive += value > 0.0 ? 1 : 0;
        }

        const auto distribution = distributionOf(worth);

        std::printf("  %-38s  worth %+7.3f%% median (%+7.3f to %+7.3f), %2zu/%zu positive\n", arms[index].name.c_str(),
                    distribution.median, distribution.minimum, distribution.maximum, positive, ensembleCount);
    }
}

// =============================================================================================
// 13. THE SENSOR TAIL AT THE SCOPED INTERVENTIONS
// =============================================================================================

TEST_CASE("the pulse age at the decisions each scope actually changes", "[.road-torque-scope]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    std::printf("\n=== the sensor tail: pulse age WHERE the qualification changes a decision ===\n");
    std::printf("  The whole-run figures the previous investigation reported were `fresh tooth on\n");
    std::printf("  27.84%% of steps, pulse age P90 10 ms, max 264 ms`. What matters for a production\n");
    std::printf("  design is the age at the interventions themselves.\n");

    distributionHeader("pulse age at changed decisions [ms]");

    for (const auto scope : {Scope::Everywhere, Scope::RecoverOnly, Scope::EmptyOnly, Scope::RecoverAndEmpty,
                             Scope::StuckDecisionAndEmpty})
    {
        const auto arm = summarise(scopeName(scope),
                                   ensembleOf(setup.value(), world.value(), assists, calibration, scoped(scope), 1.0));

        auto ages = std::vector<double>{};
        ages.reserve(arm.ages.size());
        for (const auto age : arm.ages)
        {
            ages.push_back(1000.0 * age);
        }

        distributionRow(scopeName(scope), distributionOf(ages));

        std::printf("  %-38s   fresh tooth %6.2f%%, stale (> %.0f ms) %6.2f%% of %zu\n", "",
                    100.0 * static_cast<double>(std::count(arm.ages.begin(), arm.ages.end(), 0.0)) /
                        static_cast<double>(std::max(arm.ages.size(), std::size_t{1})),
                    1000.0 * staleAge,
                    100.0 * static_cast<double>(arm.stale) /
                        static_cast<double>(std::max(arm.ages.size(), std::size_t{1})),
                    arm.ages.size());
    }

    std::printf("\n  A scope that removes the stale-sensor interventions removes the need for a filter\n");
    std::printf("  at the tail. One that does not leaves that requirement standing.\n");
}

// =============================================================================================
// 14. TONE-RING ROBUSTNESS
// =============================================================================================

TEST_CASE("what the tone ring's resolution does to the scoped arms", "[.road-torque-scope]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto base = withAntilock(setup.value());

    std::printf("\n=== tone-ring resolution, production against the broad and the scoped arms ===\n");
    std::printf("  `ToneRing::teeth` is marked a SOFT source in `WheelSensors.cppm`. The plant's\n");
    std::printf("  sensor and the ECU's estimate both change with it, so production moves too.\n");

    printArmHeader();

    for (const auto teeth :
         {std::uint32_t{24}, std::uint32_t{44}, std::uint32_t{48}, std::uint32_t{60}, std::uint32_t{96}})
    {
        auto assists = base;
        assists.toneRing.teeth = teeth;
        const auto calibration = calibrationOf(setup.value(), assists);

        auto name = std::string{"ring "};
        name += std::to_string(teeth);
        name += ", production";
        printArm(summarise(name, ensembleOf(setup.value(), world.value(), assists, calibration, Candidate{}, 1.0)));

        for (const auto scope : {Scope::Everywhere, Scope::RecoverAndEmpty, Scope::StuckDecisionAndEmpty})
        {
            auto arm = std::string{"ring "};
            arm += std::to_string(teeth);
            arm += ", ";
            arm += scopeName(scope);
            printArm(
                summarise(arm, ensembleOf(setup.value(), world.value(), assists, calibration, scoped(scope), 1.0)));
        }
    }

    std::printf("\n  The question is whether a scope narrows the spread of stranded fraction and of\n");
    std::printf("  intervention count across the five rings, not whether any one ring is better.\n");
}

// =============================================================================================
// 15. THE STEERED UNIFORM LOW-MU ENSEMBLE — RE-SCORING THE FOUR EXISTING ARMS
//
// `./EngineTests "[.steered-lowmu]"`. Added 2026-09-07, after the forensic audit of
// `AntilockSetup::slipAwareRecovery` found the case every straight-line comparison in this file and
// in `[.road-torque]` had missed: **uniform low grip WITH steering applied while braking.** On the
// straight the law measures worse on every axis — 20/29 stranded dry, ABS worth negative on mu
// 0.35, split-mu yaw eight times the law-off arm's — and on criterion 5's steered fixture it is the
// only thing keeping the car steerable at all: median 5.33 x the locked run's lateral travel
// against the law-off arm's 1.10 x.
//
// **The question here is narrow and is the only one asked.** Is that steered benefit unique to the
// current law, or does one of the two already-characterised road-torque arms preserve it? Nothing
// is tuned, no threshold is swept, no new scope is introduced and no production file is touched.
// The two road-torque arms are exactly the ones the dry and split-mu tables above already carry:
// `Scope::Everywhere` at 0.150 x peak (the broad arm) and `Scope::RecoverAndEmpty` at 0.150 x peak
// (the scoped arm D), both rear-channel only, both unaltered.
//
// **The ensemble is criterion 5's own**, not a new one: `memberEntry(index, 15)` reproduces
// `AntilockBrakingTests.cpp`'s `steeringEnsemble` entry speeds exactly — hundred x (1 + 0.001 k) for
// k in [-7, 7] — and `record`'s settle, roll-in, pedal step and stopping condition are that
// fixture's. The locked control is the same locked control.
// =============================================================================================

namespace
{

struct SteeredArm
{
    std::string name;
    AssistSetup assists;
    Candidate candidate;
};

constexpr auto steeredCount = std::size_t{15};
constexpr auto steeredLock = 0.35;
constexpr auto steeredGrip = 0.35;

[[nodiscard]] std::vector<double> columnOf(const std::vector<Member>& members, double Member::* field,
                                           const bool magnitude)
{
    auto values = std::vector<double>{};
    values.reserve(members.size());

    for (const auto& member : members)
    {
        values.push_back(magnitude ? std::abs(member.*field) : member.*field);
    }

    return values;
}

[[nodiscard]] std::vector<SteeredArm> steeredArms(const AssistSetup& assists)
{
    auto arms = std::vector<SteeredArm>{};

    arms.push_back(SteeredArm{"A production (law ON)", assists, Candidate{}});
    arms.push_back(SteeredArm{"B slipAwareRecovery OFF", withoutSlipAwareRecovery(assists), Candidate{}});
    arms.push_back(SteeredArm{"C broad road-torque 0.150", assists, scoped(Scope::Everywhere)});
    arms.push_back(SteeredArm{"D scoped Recover+empty 0.150", assists, scoped(Scope::RecoverAndEmpty)});

    return arms;
}

// The index whose value sits at the median of its own arm. For an odd ensemble that member exists
// and is a real run rather than an interpolation, which is what a trace needs.
[[nodiscard]] std::size_t medianIndexOf(const std::vector<double>& values)
{
    auto order = std::vector<std::size_t>{};
    order.reserve(values.size());

    for (auto index = std::size_t{0}; index < values.size(); index++)
    {
        order.push_back(index);
    }

    std::sort(order.begin(), order.end(),
              [&](const std::size_t left, const std::size_t right) { return values[left] < values[right]; });

    return order[order.size() / 2];
}

} // namespace

TEST_CASE("the four recovery architectures on the steered uniform low-mu ensemble", "[.steered-lowmu]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(steeredGrip, steeredGrip));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto plain = golfGtiMk7Assists(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto arms = steeredArms(assists);

    std::printf("\n=== STEERED UNIFORM LOW MU: grip %.2f both sides, full pedal, %.2f lock, %zu members ===\n",
                steeredGrip, steeredLock, steeredCount);
    std::printf("  Entry speeds are criterion 5's own: %.3f to %.3f m/s in %zu deterministic steps.\n",
                memberEntry(0, steeredCount), memberEntry(steeredCount - 1, steeredCount), steeredCount);

    // --- 0. THE LOCKED BASELINE -----------------------------------------------------------------
    //
    // One control, and it has to be ONE control: every ratio below is paired against it member by
    // member. The locked car has no electronics at all, so no arm can reach it — but "cannot" is a
    // claim about this file's own plumbing, and the shadow controller's counters do run on a locked
    // run. So it is verified rather than argued: the locked ensemble is re-run under each arm's
    // candidate and its assist setup's recovery flag, and every member must come back bit-identical.
    std::printf("\n--- 0. the locked control, re-run under each arm's context ---\n");

    const auto locked =
        ensembleOf(setup.value(), world.value(), plain, calibration, Candidate{}, 1.0, steeredLock, steeredCount);

    for (const auto& arm : arms)
    {
        auto lockedAssists = plain;
        lockedAssists.antilock.recoveryAuthority = arm.assists.antilock.recoveryAuthority;

        const auto again = ensembleOf(setup.value(), world.value(), lockedAssists, calibration, arm.candidate, 1.0,
                                      steeredLock, steeredCount);

        REQUIRE(again.size() == locked.size());

        auto identical = true;

        for (auto index = std::size_t{0}; index < locked.size(); index++)
        {
            identical = identical && again[index].distance == locked[index].distance &&
                        again[index].lateralTravel == locked[index].lateralTravel &&
                        again[index].finalYaw == locked[index].finalYaw &&
                        again[index].peakYawRate == locked[index].peakYawRate &&
                        again[index].time == locked[index].time;
        }

        std::printf("  %-32s locked control %s\n", arm.name.c_str(), identical ? "IDENTICAL" : "*** DIFFERENT ***");

        // Fixture contamination if this ever fails, and the whole comparison below is void.
        REQUIRE(identical);
    }

    auto ensembles = std::vector<std::vector<Member>>{};
    for (const auto& arm : arms)
    {
        ensembles.push_back(ensembleOf(setup.value(), world.value(), arm.assists, calibration, arm.candidate, 1.0,
                                       steeredLock, steeredCount));
    }

    for (const auto& members : ensembles)
    {
        for (const auto& member : members)
        {
            REQUIRE(member.stopped);
        }
    }

    // --- 1. LATERAL TRAVEL ------------------------------------------------------------------------
    std::printf("\n--- 1. LATERAL TRAVEL [m], magnitude. The primary steerability quantity. ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::lateralTravel, true)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(),
                        distributionOf(columnOf(ensembles[index], &Member::lateralTravel, true)));
    }

    std::printf("\n  --- signed, so a sign flip cannot hide inside a magnitude ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::lateralTravel, false)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(),
                        distributionOf(columnOf(ensembles[index], &Member::lateralTravel, false)));
    }

    std::printf("\n  --- PAIRED, member by member: |travel| and |travel| / |locked travel| ---\n");
    std::printf("\n  %-4s %9s %9s", "k", "entry", "locked");
    for (const auto& arm : arms)
    {
        std::printf(" | %8s %6s", arm.name.substr(0, 8).c_str(), "ratio");
    }
    std::printf("\n  %s\n", "--------------------------------------------------------------------------------------"
                            "--------------");

    auto passes = std::vector<std::size_t>(arms.size(), 0);
    auto ratioColumns = std::vector<std::vector<double>>(arms.size());

    for (auto member = std::size_t{0}; member < steeredCount; member++)
    {
        const auto reference = std::abs(locked[member].lateralTravel);

        std::printf("  %-4d %9.3f %9.3f", static_cast<int>(member) - 7, locked[member].entry, reference);

        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            const auto travel = std::abs(ensembles[index][member].lateralTravel);
            const auto ratio = reference > 1e-9 ? travel / reference : 0.0;

            ratioColumns[index].push_back(ratio);
            passes[index] += ratio > 3.0 ? 1 : 0;

            std::printf(" | %8.3f %6.2f", travel, ratio);
        }

        std::printf("\n");
    }

    std::printf("\n  --- the criterion, UNCHANGED: lateral travel > 3 x locked ---\n");
    std::printf("\n  %-32s %10s %12s %12s %10s\n", "arm", "members", "median ratio", "min ratio", "max ratio");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto ratios = distributionOf(ratioColumns[index]);

        std::printf("  %-32s %5zu/%-4zu %12.3f %12.3f %10.3f\n", arms[index].name.c_str(), passes[index], steeredCount,
                    ratios.median, ratios.minimum, ratios.maximum);
    }

    std::printf("\n  --- and as criterion 5 itself writes it: median(|assisted|) > 3 x median(|locked|) ---\n");

    const auto lockedMedian = distributionOf(columnOf(locked, &Member::lateralTravel, true)).median;

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto armMedian = distributionOf(columnOf(ensembles[index], &Member::lateralTravel, true)).median;

        std::printf("  %-32s %8.3f m against %8.3f m needed   %s   (%.2f x locked)\n", arms[index].name.c_str(),
                    armMedian, 3.0 * lockedMedian, armMedian > 3.0 * lockedMedian ? "PASS" : "FAIL",
                    armMedian / lockedMedian);
    }

    // --- 2. STOPPING DISTANCE ---------------------------------------------------------------------
    std::printf("\n--- 2. STOPPING DISTANCE [m]. Secondary here, and not optimised. ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::distance, false)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(), distributionOf(columnOf(ensembles[index], &Member::distance, false)));
    }

    // --- 3. PEAK YAW RATE -------------------------------------------------------------------------
    std::printf("\n--- 3. PEAK YAW RATE [rad/s]. NOT a steerability metric on its own. ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::peakYawRate, false)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(),
                        distributionOf(columnOf(ensembles[index], &Member::peakYawRate, false)));
    }

    // --- 4. MEAN TRUE SLIP ------------------------------------------------------------------------
    std::printf("\n--- 4. MEAN TRUE SLIP, from the tyre and not from the estimator. Diagnostic only. ---\n");

    for (const auto& axis :
         std::vector<std::pair<const char*, double Member::*>>{{"front axle", &Member::meanSlipFront},
                                                               {"rear axle", &Member::meanSlipRear},
                                                               {"all four", &Member::meanSlipAll}})
    {
        std::printf("\n  mean |slip ratio|, %s\n", axis.first);
        distributionHeader("arm");
        distributionRow("locked (control)", distributionOf(columnOf(locked, axis.second, false)));
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            distributionRow(arms[index].name.c_str(), distributionOf(columnOf(ensembles[index], axis.second, false)));
        }
    }

    // --- 5. TYRE UTILISATION ----------------------------------------------------------------------
    std::printf("\n--- 5. TYRE UTILISATION, per axle, over the whole braking event ---\n");
    std::printf("  longUse = sum|Fx| / sum(longitudinal capacity); latUse = sum|Fy| / sum(lateral capacity);\n");
    std::printf("  ellipse = mean of the tyre's own `TyreForces::gripUsed`; pastPeak = share of wheel-ticks\n");
    std::printf("  with |slip| past that tyre's own `longitudinalPeakSlip` at that load and grip.\n");
    std::printf("\n  %-32s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "arm", "F longUse", "R longUse", "F latUse",
                "R latUse", "F |Fy| N", "R |Fy| N", "F ellips", "R ellips", "F past%", "R past%");
    std::printf("  %s\n", "-----------------------------------------------------------------------------------------"
                          "-------------------------------------------");

    const auto utilisationRow = [](const char* name, const std::vector<Member>& members)
    {
        const auto value = [&](double Member::* field)
        {
            return distributionOf(columnOf(members, field, false)).median;
        };

        std::printf("  %-32s %9.4f %9.4f %9.4f %9.4f %9.1f %9.1f %9.4f %9.4f %9.1f %9.1f\n", name,
                    value(&Member::frontUtilisation), value(&Member::rearUtilisation), value(&Member::lateralUseFront),
                    value(&Member::lateralUseRear), value(&Member::lateralForceFront), value(&Member::lateralForceRear),
                    value(&Member::gripUsedFront), value(&Member::gripUsedRear),
                    100.0 * value(&Member::beyondPeakFront), 100.0 * value(&Member::beyondPeakRear));
    };

    utilisationRow("locked (control)", locked);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        utilisationRow(arms[index].name.c_str(), ensembles[index]);
    }

    std::printf("\n  --- the same quantities as distributions, not medians ---\n");

    for (const auto& column : std::vector<std::pair<const char*, double Member::*>>{
             {"front lateral utilisation", &Member::lateralUseFront},
             {"rear lateral utilisation", &Member::lateralUseRear},
             {"front share past own peak slip", &Member::beyondPeakFront},
             {"rear share past own peak slip", &Member::beyondPeakRear}})
    {
        std::printf("\n  %s\n", column.first);
        distributionHeader("arm");
        distributionRow("locked (control)", distributionOf(columnOf(locked, column.second, false)));
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            distributionRow(arms[index].name.c_str(), distributionOf(columnOf(ensembles[index], column.second, false)));
        }
    }

    // --- 6. PATH QUALITY --------------------------------------------------------------------------
    std::printf("\n--- 6. PATH QUALITY: is the car going somewhere, or turning on the spot? ---\n");
    std::printf("\n  %-32s %11s %11s %11s %10s %10s\n", "arm", "final yaw", "final lat", "peak yaw/s", "reversals",
                "0.1 m at");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------");

    const auto pathRow = [](const char* name, const std::vector<Member>& members)
    {
        const auto value = [&](double Member::* field, const bool magnitude)
        {
            return distributionOf(columnOf(members, field, magnitude)).median;
        };

        auto reversals = std::vector<double>{};
        for (const auto& member : members)
        {
            reversals.push_back(static_cast<double>(member.yawReversals));
        }

        std::printf("  %-32s %11.3f %11.3f %11.5f %10.1f %10.3f\n", name, degrees * value(&Member::finalYaw, false),
                    value(&Member::lateralTravel, false), value(&Member::peakYawRate, false),
                    distributionOf(reversals).median, value(&Member::lateralDelay, false));
    };

    pathRow("locked (control)", locked);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        pathRow(arms[index].name.c_str(), ensembles[index]);
    }

    std::printf("\n  final yaw is degrees at rest; final lat is signed metres; `0.1 m at` is the second the\n");
    std::printf("  car first reached a tenth of a metre of lateral displacement. Reporting statistics off\n");
    std::printf("  the recorded path; none of them is a controller target.\n");

    // --- 7. THE ROAD-TORQUE INTERVENTION AUDIT, C AND D ONLY ---------------------------------------
    std::printf("\n--- 7. ROAD-TORQUE INTERVENTION AUDIT on THIS fixture (arms C and D only) ---\n");
    printArmHeader();
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printArm(summarise(arms[index].name, ensembles[index]));
    }

    printPrecisionHeader();
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        printPrecision(arms[index].name.c_str(), summarise(arms[index].name, ensembles[index]).precision);
    }

    std::printf("\n  --- the same changed decisions in the slip-first partition ---\n");
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        printSlipFirst(arms[index].name.c_str(), summarise(arms[index].name, ensembles[index]).precision);
    }

    std::printf("\n  --- by caliper state at the decision (0 = pressure left, 1 = already empty) ---\n");
    printPrecisionHeader();
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        const auto summary = summarise(arms[index].name, ensembles[index]);

        for (auto caliper = std::size_t{0}; caliper < 2; caliper++)
        {
            auto name = arms[index].name + (caliper == 0 ? "  [has pressure]" : "  [empty]");
            printPrecision(name.c_str(), summary.byCaliper[caliper]);
        }
    }

    std::printf("\n  --- by the phase the channel was in ---\n");
    printPrecisionHeader();
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        const auto summary = summarise(arms[index].name, ensembles[index]);

        for (auto phase = std::size_t{0}; phase < 5; phase++)
        {
            if (summary.byPhase[phase].total() == 0)
            {
                continue;
            }

            auto name = arms[index].name + "  [" + modulatorName(static_cast<ModulatorPhase>(phase)) + "]";
            printPrecision(name.c_str(), summary.byPhase[phase]);
        }
    }

    std::printf("\n  --- by which wheel the select-low rear channel was reading ---\n");
    std::printf("  Both arms are REAR-CHANNEL ONLY, so there is no front/rear split to report: every\n");
    std::printf("  changed decision below is the rear channel's. Left/right is the select-low wheel.\n");
    printPrecisionHeader();
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        const auto summary = summarise(arms[index].name, ensembles[index]);

        for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
        {
            if (summary.byWheel[wheel].total() == 0)
            {
                continue;
            }

            auto name = arms[index].name + "  [wheel " + std::to_string(wheel) + "]";
            printPrecision(name.c_str(), summary.byWheel[wheel]);
        }
    }

    // --- 8. SENSOR QUALITY AT THE CHANGED DECISIONS ------------------------------------------------
    std::printf("\n--- 8. PULSE AGE at the decisions each road-torque arm actually changed [ms] ---\n");
    distributionHeader("arm");
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        const auto summary = summarise(arms[index].name, ensembles[index]);

        auto ages = std::vector<double>{};
        ages.reserve(summary.ages.size());
        for (const auto age : summary.ages)
        {
            ages.push_back(1000.0 * age);
        }

        distributionRow(arms[index].name.c_str(), distributionOf(ages));

        std::printf("  %-38s   fresh tooth %6.2f%%, stale (> %.0f ms) %6.2f%% of %zu\n", "",
                    100.0 * static_cast<double>(std::count(summary.ages.begin(), summary.ages.end(), 0.0)) /
                        static_cast<double>(std::max(summary.ages.size(), std::size_t{1})),
                    1000.0 * staleAge,
                    100.0 * static_cast<double>(summary.stale) /
                        static_cast<double>(std::max(summary.ages.size(), std::size_t{1})),
                    summary.ages.size());
    }
}

TEST_CASE("one steered low-mu member, traced through all four arms, with the state machine that goes with it",
          "[.steered-lowmu]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(steeredGrip, steeredGrip));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto plain = golfGtiMk7Assists(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto arms = steeredArms(assists);

    std::printf("\n=== ONE MEMBER, TRACED. Steered low mu, grip %.2f, full pedal, %.2f lock. ===\n", steeredGrip,
                steeredLock);

    // **The traced member has to be the SAME entry speed in every arm**, or the traces cannot be
    // aligned and "when did they diverge" has no meaning. So the ensembles are run first, each arm's
    // own median member is identified, and the member actually traced is production's — with every
    // arm's own median printed beside it so the reader can see how representative it is for that arm.
    auto ensembles = std::vector<std::vector<Member>>{};
    for (const auto& arm : arms)
    {
        ensembles.push_back(ensembleOf(setup.value(), world.value(), arm.assists, calibration, arm.candidate, 1.0,
                                       steeredLock, steeredCount));
    }

    std::printf("\n  %-32s %14s %14s %14s\n", "arm", "median |lat| m", "median member", "that member m");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto column = columnOf(ensembles[index], &Member::lateralTravel, true);
        const auto median = medianIndexOf(column);

        std::printf("  %-32s %14.3f %14d %14.3f\n", arms[index].name.c_str(), distributionOf(column).median,
                    static_cast<int>(median) - 7, column[median]);
    }

    const auto chosen = medianIndexOf(columnOf(ensembles[0], &Member::lateralTravel, true));
    const auto entry = memberEntry(chosen, steeredCount);

    std::printf("\n  TRACED MEMBER: k = %+d, entry %.4f m/s (production's own median member).\n",
                static_cast<int>(chosen) - 7, entry);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("    %-32s |lat| %8.3f m   (its own median %8.3f m)\n", arms[index].name.c_str(),
                    std::abs(ensembles[index][chosen].lateralTravel),
                    distributionOf(columnOf(ensembles[index], &Member::lateralTravel, true)).median);
    }

    auto runs = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        runs.push_back(record(setup.value(), world.value(), arm.assists, calibration, arm.candidate, 1.0, entry,
                              steeredLock, true));
    }

    const auto lockedRun =
        record(setup.value(), world.value(), plain, calibration, Candidate{}, 1.0, entry, steeredLock, false);

    // --- the aligned traces -----------------------------------------------------------------------
    std::printf("\n--- ALIGNED TIME HISTORIES. Steering input is a constant %.2f of lock from t = 0. ---\n",
                steeredLock);
    std::printf("  Fx and Fy are axle sums, newtons, signed as the tyre reports them. `slip` is the mean\n");
    std::printf("  |slip ratio| over the axle. `phase` is FL/FR/rear. `bar` is the commanded caliper\n");
    std::printf("  pressure at that wheel.\n");

    const auto traceOf = [](const char* name, const Run& run)
    {
        std::printf("\n  %s\n", name);
        std::printf("  %7s %7s %9s %9s | %7s %7s | %9s %9s | %9s %9s | %5s %8s %8s\n", "t s", "v m/s", "yaw/s", "lat m",
                    "F slip", "R slip", "F Fx N", "R Fx N", "F Fy N", "R Fy N", "phase", "F bar", "R bar");

        auto next = 0.0;

        for (const auto& sample : run.ticks)
        {
            if (sample.time + 1e-9 < next)
            {
                continue;
            }

            next = sample.time < 2.0 ? sample.time + 0.05 : sample.time + 0.25;

            const auto frontSlip = 0.5 * (std::abs(sample.wheels[0].slipRatio) + std::abs(sample.wheels[1].slipRatio));
            const auto rearSlip = 0.5 * (std::abs(sample.wheels[2].slipRatio) + std::abs(sample.wheels[3].slipRatio));

            std::printf("  %7.3f %7.3f %9.5f %9.4f | %7.4f %7.4f | %9.1f %9.1f | %9.1f %9.1f | %c%c%c   %8.1f %8.1f\n",
                        sample.time, sample.speed, sample.yawRate, sample.lateralTravel, frontSlip, rearSlip,
                        sample.wheels[0].longitudinal + sample.wheels[1].longitudinal,
                        sample.wheels[2].longitudinal + sample.wheels[3].longitudinal,
                        sample.wheels[0].lateral + sample.wheels[1].lateral,
                        sample.wheels[2].lateral + sample.wheels[3].lateral, phaseLetter(sample.wheels[0].phase),
                        phaseLetter(sample.wheels[1].phase), phaseLetter(sample.wheels[2].phase),
                        sample.wheels[0].pressure / bar, sample.wheels[2].pressure / bar);
        }
    };

    traceOf("LOCKED (control)", lockedRun);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        traceOf(arms[index].name.c_str(), runs[index]);
    }

    // --- where they part ---------------------------------------------------------------------------
    std::printf("\n--- WHEN THE TRAJECTORIES DIVERGE, each arm against production ---\n");
    std::printf("\n  %-32s %14s %14s %14s %14s\n", "arm", "|dlat| > 5 cm", "|dlat| > 25 cm", "|dlat| > 1 m",
                "|dslip| > 0.05");

    const auto divergence = [&](const Run& left, const Run& right, const double target, const bool slip)
    {
        const auto count = std::min(left.ticks.size(), right.ticks.size());

        for (auto index = std::size_t{0}; index < count; index++)
        {
            if (slip)
            {
                auto a = 0.0;
                auto b = 0.0;

                for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
                {
                    a += std::abs(left.ticks[index].wheels[wheel].slipRatio) / cornerCount;
                    b += std::abs(right.ticks[index].wheels[wheel].slipRatio) / cornerCount;
                }

                if (std::abs(a - b) > target)
                {
                    return left.ticks[index].time;
                }
            }
            else if (std::abs(left.ticks[index].lateralTravel - right.ticks[index].lateralTravel) > target)
            {
                return left.ticks[index].time;
            }
        }

        return -1.0;
    };

    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        std::printf("  %-32s %14.3f %14.3f %14.3f %14.3f\n", arms[index].name.c_str(),
                    divergence(runs[0], runs[index], 0.05, false), divergence(runs[0], runs[index], 0.25, false),
                    divergence(runs[0], runs[index], 1.0, false), divergence(runs[0], runs[index], 0.05, true));
    }

    std::printf("  %-32s %14.3f %14.3f %14.3f %14.3f\n", "locked, against production",
                divergence(runs[0], lockedRun, 0.05, false), divergence(runs[0], lockedRun, 0.25, false),
                divergence(runs[0], lockedRun, 1.0, false), divergence(runs[0], lockedRun, 0.05, true));

    std::printf("\n  -1.000 means the threshold was never reached before the shorter run ended.\n");

    // --- the windowed picture -----------------------------------------------------------------------
    std::printf("\n--- THE SAME MEMBER IN HALF-SECOND WINDOWS: where the lateral travel is actually made ---\n");
    std::printf("\n  %-32s", "arm");
    for (auto window = 0; window < 12; window++)
    {
        std::printf(" %6.1f", 0.5 * static_cast<double>(window + 1));
    }
    std::printf("   (cumulative |lateral travel|, m, at each half second)\n");

    const auto windowRow = [](const char* name, const Run& run)
    {
        std::printf("  %-32s", name);

        for (auto window = 0; window < 12; window++)
        {
            const auto when = 0.5 * static_cast<double>(window + 1);
            auto value = 0.0;

            for (const auto& sample : run.ticks)
            {
                if (sample.time > when)
                {
                    break;
                }

                value = std::abs(sample.lateralTravel);
            }

            std::printf(" %6.2f", value);
        }

        std::printf("\n");
    };

    windowRow("locked (control)", lockedRun);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        windowRow(arms[index].name.c_str(), runs[index]);
    }

    std::printf("\n  ...and the mean |slip ratio| per axle over the same windows\n");

    const auto slipWindowRow = [](const char* name, const Run& run, const std::size_t first, const std::size_t last)
    {
        std::printf("  %-32s", name);

        for (auto window = 0; window < 12; window++)
        {
            const auto from = 0.5 * static_cast<double>(window);
            const auto to = 0.5 * static_cast<double>(window + 1);
            auto total = 0.0;
            auto count = std::size_t{0};

            for (const auto& sample : run.ticks)
            {
                if (sample.time <= from || sample.time > to)
                {
                    continue;
                }

                for (auto index = first; index <= last; index++)
                {
                    total += std::abs(sample.wheels[index].slipRatio);
                    count++;
                }
            }

            std::printf(" %6.3f", count > 0 ? total / static_cast<double>(count) : 0.0);
        }

        std::printf("\n");
    };

    std::printf("\n  FRONT axle\n");
    slipWindowRow("locked (control)", lockedRun, 0, 1);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        slipWindowRow(arms[index].name.c_str(), runs[index], 0, 1);
    }

    std::printf("\n  REAR axle\n");
    slipWindowRow("locked (control)", lockedRun, 2, 3);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        slipWindowRow(arms[index].name.c_str(), runs[index], 2, 3);
    }

    std::printf("\n  ...and the axle lateral force, newtons, over the same windows\n");

    const auto lateralWindowRow = [](const char* name, const Run& run, const std::size_t first, const std::size_t last)
    {
        std::printf("  %-32s", name);

        for (auto window = 0; window < 12; window++)
        {
            const auto from = 0.5 * static_cast<double>(window);
            const auto to = 0.5 * static_cast<double>(window + 1);
            auto total = 0.0;
            auto count = std::size_t{0};

            for (const auto& sample : run.ticks)
            {
                if (sample.time <= from || sample.time > to)
                {
                    continue;
                }

                auto axle = 0.0;
                for (auto index = first; index <= last; index++)
                {
                    axle += sample.wheels[index].lateral;
                }

                total += std::abs(axle);
                count++;
            }

            std::printf(" %6.0f", count > 0 ? total / static_cast<double>(count) : 0.0);
        }

        std::printf("\n");
    };

    std::printf("\n  FRONT axle |Fy|\n");
    lateralWindowRow("locked (control)", lockedRun, 0, 1);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        lateralWindowRow(arms[index].name.c_str(), runs[index], 0, 1);
    }

    std::printf("\n  REAR axle |Fy|\n");
    lateralWindowRow("locked (control)", lockedRun, 2, 3);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        lateralWindowRow(arms[index].name.c_str(), runs[index], 2, 3);
    }

    // --- the state machine ---------------------------------------------------------------------------
    std::printf("\n--- STATE-MACHINE DIFFERENTIAL on the traced member, per channel ---\n");
    std::printf("  Counted off the controller's own steps (%.0f Hz), not off the physics tick.\n", assists.controlRate);
    std::printf("\n  %-32s %6s | %7s %7s %7s | %7s %7s %7s | %9s\n", "arm", "chan", "Dump in", "Rec in", "Hold in",
                "R->D", "R->A", "A->D", "empty s");
    std::printf("  %s\n", "----------------------------------------------------------------------------------------"
                          "------------");

    const auto period = assists.controlRate > 0.0 ? 1.0 / assists.controlRate : tick;

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            auto dumpIn = std::size_t{0};
            auto recoverIn = std::size_t{0};
            auto holdIn = std::size_t{0};
            auto recoverDump = std::size_t{0};
            auto recoverReapply = std::size_t{0};
            auto reapplyDump = std::size_t{0};
            auto emptySteps = std::size_t{0};

            for (const auto& step : runs[index].steps)
            {
                if (step.channel != channel)
                {
                    continue;
                }

                dumpIn += (step.before != ModulatorPhase::Dump && step.after == ModulatorPhase::Dump) ? 1 : 0;
                recoverIn += (step.before != ModulatorPhase::Recover && step.after == ModulatorPhase::Recover) ? 1 : 0;
                holdIn += (step.before != ModulatorPhase::Hold && step.after == ModulatorPhase::Hold) ? 1 : 0;
                recoverDump += (step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Dump) ? 1 : 0;
                recoverReapply +=
                    (step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Reapply) ? 1 : 0;
                reapplyDump += (step.before == ModulatorPhase::Reapply && step.after == ModulatorPhase::Dump) ? 1 : 0;
                emptySteps += step.pressureAfter <= 0.0 ? 1 : 0;
            }

            std::printf("  %-32s %6s | %7zu %7zu %7zu | %7zu %7zu %7zu | %9.3f\n",
                        channel == 0 ? arms[index].name.c_str() : "",
                        channel == 0 ? "FL" : (channel == 1 ? "FR" : "rear"), dumpIn, recoverIn, holdIn, recoverDump,
                        recoverReapply, reapplyDump, static_cast<double>(emptySteps) * period);
        }
    }

    std::printf("\n  --- phase residency on the traced member, share of BRAKING ticks per channel ---\n");
    std::printf("\n  %-32s %6s | %8s %8s %8s %8s %8s\n", "arm", "chan", "passive", "hold", "dump", "recover",
                "reapply");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        for (const auto wheel : {std::size_t{0}, std::size_t{1}, std::size_t{2}})
        {
            auto residency = std::array<std::size_t, 5>{};

            for (const auto& sample : runs[index].ticks)
            {
                residency[phaseIndex(sample.wheels[wheel].phase)]++;
            }

            const auto total = static_cast<double>(std::max(runs[index].ticks.size(), std::size_t{1}));

            std::printf(
                "  %-32s %6s | %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", wheel == 0 ? arms[index].name.c_str() : "",
                wheel == 0 ? "FL" : (wheel == 1 ? "FR" : "rear"), 100.0 * static_cast<double>(residency[0]) / total,
                100.0 * static_cast<double>(residency[1]) / total, 100.0 * static_cast<double>(residency[2]) / total,
                100.0 * static_cast<double>(residency[3]) / total, 100.0 * static_cast<double>(residency[4]) / total);
        }
    }

    std::printf("\n  --- and what the two road-torque arms changed on THIS member ---\n");
    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        auto qualified = std::size_t{0};
        auto first = -1.0;
        auto last = -1.0;

        for (const auto& step : runs[index].steps)
        {
            if (!step.qualified)
            {
                continue;
            }

            qualified++;
            first = first < 0.0 ? step.time : first;
            last = step.time;
        }

        std::printf("  %-32s changed %4zu decisions, first at %7.3f s, last at %7.3f s\n", arms[index].name.c_str(),
                    qualified, first, last);
    }
}

// =============================================================================================
// 16. THE THREE-LEG CAUSAL ABLATION OF `slipAwareRecovery`
//
// `./EngineTests "[.leg-ablation]"`. Added 2026-09-07, after section 15 established that the law
// AS A WHOLE carries the steered low-mu invariant — median 5.24 x the locked run's lateral travel
// against the law-off arm's 1.12 x — and that neither road-torque architecture is a shippable way
// to keep it. What section 15 could not say is WHICH PART of the law does the carrying, because
// `AntilockSetup::slipAwareRecovery` is one switch over three legs.
//
// **This file changes no production file and designs no controller.** It routes the channel
// through a verbatim probe-local copy of `advanceAntilockChannel` (`ablatedAntilockChannel`,
// above) whose only power is to switch one named leg off, proves that copy bit-identical to
// production with every leg on, and then measures four one-leg-off arms against the two controls
// on the fixture section 15 already validated. No threshold moves, no gradient moves, no
// estimator changes, no road-torque evidence appears anywhere.
// =============================================================================================

namespace
{

struct LegArm
{
    std::string name;
    AssistSetup assists;
    Candidate candidate;
};

// Every state-machine transition the deliverable asks for, per channel, counted off the
// controller's own steps. Arithmetic over `Run::steps`; nothing here is read by any decision.
struct Transitions
{
    std::array<std::size_t, brakeChannelCount> steps{};
    std::array<std::size_t, brakeChannelCount> dumpIn{};
    std::array<std::size_t, brakeChannelCount> recoverIn{};
    std::array<std::size_t, brakeChannelCount> holdIn{};
    std::array<std::size_t, brakeChannelCount> reapplyIn{};
    std::array<std::size_t, brakeChannelCount> recoverDump{};
    std::array<std::size_t, brakeChannelCount> recoverReapply{};
    std::array<std::size_t, brakeChannelCount> reapplyDump{};
    std::array<std::size_t, brakeChannelCount> emptySteps{};

    // Phase residency comes off the physics ticks rather than the controller steps, because that is
    // what the car actually experienced. Indexed [wheel][phase] for FL, FR and the rear pair's left.
    std::array<std::array<std::size_t, 5>, 3> residency{};
    std::size_t ticks = 0;

    void add(const Transitions& more)
    {
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            steps[channel] += more.steps[channel];
            dumpIn[channel] += more.dumpIn[channel];
            recoverIn[channel] += more.recoverIn[channel];
            holdIn[channel] += more.holdIn[channel];
            reapplyIn[channel] += more.reapplyIn[channel];
            recoverDump[channel] += more.recoverDump[channel];
            recoverReapply[channel] += more.recoverReapply[channel];
            reapplyDump[channel] += more.reapplyDump[channel];
            emptySteps[channel] += more.emptySteps[channel];
        }

        for (auto wheel = std::size_t{0}; wheel < 3; wheel++)
        {
            for (auto phase = std::size_t{0}; phase < 5; phase++)
            {
                residency[wheel][phase] += more.residency[wheel][phase];
            }
        }

        ticks += more.ticks;
    }
};

[[nodiscard]] Transitions transitionsOf(const Run& run)
{
    auto counts = Transitions{};

    for (const auto& step : run.steps)
    {
        const auto channel = step.channel;

        counts.steps[channel]++;
        counts.dumpIn[channel] += (step.before != ModulatorPhase::Dump && step.after == ModulatorPhase::Dump) ? 1 : 0;
        counts.recoverIn[channel] +=
            (step.before != ModulatorPhase::Recover && step.after == ModulatorPhase::Recover) ? 1 : 0;
        counts.holdIn[channel] += (step.before != ModulatorPhase::Hold && step.after == ModulatorPhase::Hold) ? 1 : 0;
        counts.reapplyIn[channel] +=
            (step.before != ModulatorPhase::Reapply && step.after == ModulatorPhase::Reapply) ? 1 : 0;
        counts.recoverDump[channel] +=
            (step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Dump) ? 1 : 0;
        counts.recoverReapply[channel] +=
            (step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Reapply) ? 1 : 0;
        counts.reapplyDump[channel] +=
            (step.before == ModulatorPhase::Reapply && step.after == ModulatorPhase::Dump) ? 1 : 0;
        counts.emptySteps[channel] += step.pressureAfter <= 0.0 ? 1 : 0;
    }

    for (const auto& sample : run.ticks)
    {
        for (auto wheel = std::size_t{0}; wheel < 3; wheel++)
        {
            counts.residency[wheel][phaseIndex(sample.wheels[wheel].phase)]++;
        }
    }

    counts.ticks = run.ticks.size();

    return counts;
}

struct LegEnsemble
{
    std::vector<Member> members;
    Transitions transitions;
};

// The ensemble, with the controller step trace on so the state machine can be counted. `record` is
// section 15's own; the trace flag adds a vector and changes no arithmetic.
[[nodiscard]] LegEnsemble legEnsemble(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                                      const RoadTorqueCalibration& calibration, const Candidate& candidate)
{
    auto ensemble = LegEnsemble{};
    ensemble.members.reserve(steeredCount);

    for (auto index = std::size_t{0}; index < steeredCount; index++)
    {
        const auto entry = memberEntry(index, steeredCount);
        const auto run = record(setup, world, assists, calibration, candidate, 1.0, entry, steeredLock, true);

        ensemble.members.push_back(memberOf(index, entry, run));
        ensemble.transitions.add(transitionsOf(run));
    }

    return ensemble;
}

// The earliest controller step at which an arm's trace stops matching production's. The two traces
// are index-aligned by construction: `shadowUpdate`'s clock advances on `deltaTime` alone, so the
// number of controller steps per physics tick and the channel order within a step are the same
// sequence whatever the controller decides.
struct Divergence
{
    bool found = false;
    std::size_t index = 0;
    double time = 0.0;
    std::size_t channel = 0;

    ModulatorPhase before = ModulatorPhase::Passive;
    ModulatorPhase productionAfter = ModulatorPhase::Passive;
    ModulatorPhase armAfter = ModulatorPhase::Passive;

    double productionPressure = 0.0;
    double armPressure = 0.0;

    double guardSlip = 0.0;
    double slip = 0.0;
    double excess = 0.0;
    double acceleration = 0.0;
    bool pastBand = false;

    bool phaseDiffers = false;
    bool pressureDiffers = false;
};

[[nodiscard]] Divergence firstDivergence(const Run& base, const Run& arm, const long wanted)
{
    const auto limit = std::min(base.steps.size(), arm.steps.size());

    for (auto index = std::size_t{0}; index < limit; index++)
    {
        const auto& left = base.steps[index];
        const auto& right = arm.steps[index];

        if (wanted >= 0 && left.channel != static_cast<std::size_t>(wanted))
        {
            continue;
        }

        const auto phaseDiffers = left.after != right.after;
        const auto pressureDiffers = left.pressureAfter != right.pressureAfter;

        if (!phaseDiffers && !pressureDiffers)
        {
            continue;
        }

        auto divergence = Divergence{};
        divergence.found = true;
        divergence.index = index;
        divergence.time = left.time;
        divergence.channel = left.channel;
        divergence.before = left.before;
        divergence.productionAfter = left.after;
        divergence.armAfter = right.after;
        divergence.productionPressure = left.pressureAfter;
        divergence.armPressure = right.pressureAfter;
        divergence.guardSlip = left.guardSlip;
        divergence.slip = left.slip;
        divergence.excess = left.excess;
        divergence.acceleration = left.acceleration;
        divergence.pastBand = left.pastBandUnqualified;
        divergence.phaseDiffers = phaseDiffers;
        divergence.pressureDiffers = pressureDiffers;

        return divergence;
    }

    return Divergence{};
}

[[nodiscard]] bool sameMember(const Member& left, const Member& right)
{
    return left.distance == right.distance && left.lateralTravel == right.lateralTravel &&
           left.finalYaw == right.finalYaw && left.peakYawRate == right.peakYawRate && left.time == right.time &&
           left.frontUtilisation == right.frontUtilisation && left.rearUtilisation == right.rearUtilisation &&
           left.meanSlipAll == right.meanSlipAll && left.lateralForceFront == right.lateralForceFront &&
           left.lateralUseFront == right.lateralUseFront && left.frontCycles == right.frontCycles &&
           left.rearCycles == right.rearCycles && left.stopped == right.stopped;
}

// Mean of a per-tick axle quantity inside a half-second window, for the divergence narrative.
[[nodiscard]] double windowAxleMean(const Run& run, const std::size_t first, const std::size_t last,
                                    double WheelTick::* field, const double from, const double to, const bool magnitude)
{
    auto total = 0.0;
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        if (sample.time <= from || sample.time > to)
        {
            continue;
        }

        for (auto index = first; index <= last; index++)
        {
            const auto value = sample.wheels[index].*field;
            total += magnitude ? std::abs(value) : value;
            count++;
        }
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

} // namespace

TEST_CASE("which leg of the slip-aware recovery law carries the steered low-mu invariant", "[.leg-ablation]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(steeredGrip, steeredGrip));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());
    const auto plain = golfGtiMk7Assists(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);

    const auto replicaOf = [](const bool legA, const bool legB, const bool legC)
    {
        auto candidate = Candidate{};
        candidate.replica = true;
        candidate.legA = legA;
        candidate.legB = legB;
        candidate.legC = legC;

        return candidate;
    };

    std::printf("\n=== THREE-LEG ABLATION OF slipAwareRecovery ===\n");
    std::printf("  Fixture: gripPlate(%.2f, %.2f), full pedal, %.2f lock, %zu deterministic members,\n", steeredGrip,
                steeredGrip, steeredLock, steeredCount);
    std::printf("  entry %.4f to %.4f m/s. Section 15's fixture, unaltered.\n", memberEntry(0, steeredCount),
                memberEntry(steeredCount - 1, steeredCount));

    // --- 16.0 THE LEG DECOMPOSITION, BEFORE ANY RESULT ------------------------------------------
    std::printf("\n--- 16.0 LEG DECOMPOSITION: the production expressions, and one the task does not name ---\n");
    std::printf("\n  `pastBand` is the law's single term, formed once per channel per step:\n");
    std::printf("      pastBand = (setup.recoveryAuthority != RecoveryAuthority::Disabled) && referenceValid && "
                "guardSlip > setup.slipEnter\n");
    std::printf("  and `guardSlip` is the staleness-projected slip, computed unconditionally.\n");
    std::printf("\n  FIVE production expressions read it. The task's three legs are three of them.\n");
    std::printf("\n  LEG A  Dump exit          if (excess > recoveryAcceleration || (!losing && !pastBand))\n");
    std::printf("         OFF                 if (excess > recoveryAcceleration || !losing)\n");
    std::printf("         can change          Dump -> Recover\n");
    std::printf("\n  LEG B  Recover stuck      else if (pastBand && excess < recoveryAcceleration)  -> Dump\n");
    std::printf("         OFF                 branch deleted; control falls to the surge test below it\n");
    std::printf("         can change          Recover -> Dump\n");
    std::printf("\n  LEG C  Reapply, BOTH halves of one function's slip-aware behaviour:\n");
    std::printf("     C1  proximity taper     proximity = clamp((slipEnter - guardSlip)/(slipEnter - slipExit),0,1)\n");
    std::printf("                             fast      = pressure < departurePressure && !pastBand\n");
    std::printf("                             pressure += reapplyGradient * (fast ? 1 : proximity) * dt\n");
    std::printf("         OFF                 proximity = 1, fast term loses !pastBand; the product is 1\n");
    std::printf("                             either way, which IS the pre-law gradient\n");
    std::printf("     C2  Reapply stuck       else if (pastBand && excess < -recoveryAcceleration &&\n");
    std::printf("                                      guardSlip > 2*slipEnter - slipExit)          -> Dump\n");
    std::printf("         OFF                 branch deleted\n");
    std::printf("         can change          the Reapply pressure trajectory, and Reapply -> Dump\n");
    std::printf("\n  **LEG D, AND THE THREE-LEG DECOMPOSITION DOES NOT COVER IT.** Inside Recover:\n");
    std::printf("         else if (!(surged && surging)) { if (!pastBand) { phase = Reapply; } }\n");
    std::printf("  This is a FOURTH pastBand consumer, in the Recover phase, and it is not leg B: leg B\n");
    std::printf("  is the branch that returns to Dump, this is the gate that refuses to leave Recover for\n");
    std::printf("  Reapply. It postdates the three legs `AntilockSetup::slipAwareRecovery`'s own comment\n");
    std::printf("  lists. It is held ON in arms A, B and C, and it is OFF only in control O.\n");
    std::printf("\n  CONSEQUENCE, STATED BEFORE THE RESULTS: 'B OFF' cannot reproduce the law-off arm's\n");
    std::printf("  initial rear Recover -> Reapply. With B off a stuck rear channel falls through to the\n");
    std::printf("  surge test, meets leg D's `!pastBand`, and HOLDS in Recover at constant pressure. The\n");
    std::printf("  hypothesis in section 12 of the brief is therefore testable only in the weaker form\n");
    std::printf("  'does removing the stuck dump alone recreate the law-off outcome', and the answer is\n");
    std::printf("  predicted by the code to be no. It is measured below rather than assumed.\n");
    std::printf("\n  SEPARABILITY: C1 and C2 are separable in principle — C1 is a gradient and C2 is a\n");
    std::printf("  transition — but they are the same function's response to the same term and the audit\n");
    std::printf("  named them one leg. They are switched together. Splitting them would make four arms,\n");
    std::printf("  which this task forbids; it is recorded as the obvious follow-up and not run here.\n");
    std::printf("\n  STATE THAT GOES UNREAD UNDER AN ABLATION:\n");
    std::printf("    C OFF  -> `slipExit` is read by nothing (both its readers are C1 and C2), and\n");
    std::printf("              `departurePressure` is still written on every Dump entry but read by\n");
    std::printf("              nothing that can change a number: `fast` no longer scales the gradient.\n");
    std::printf("    B OFF  -> `departurePressure` loses one of its three writers.\n");
    std::printf("    A OFF  -> nothing becomes unread; `guardSlip` and `pastBand` still feed B, C and D.\n");
    std::printf("    O      -> `pastBand` is constant false, so `guardSlip`, `slipExit` and\n");
    std::printf("              `departurePressure` are all inert together.\n");

    // --- 16.1 THE REPLICA IS PROVED BEFORE IT IS USED --------------------------------------------
    std::printf("\n--- 16.1 REPLICA FIDELITY: the probe copy with all three legs ON, against production ---\n");

    const auto production = legEnsemble(setup.value(), world.value(), assists, calibration, Candidate{});
    const auto mirrored = legEnsemble(setup.value(), world.value(), assists, calibration, replicaOf(true, true, true));

    REQUIRE(mirrored.members.size() == production.members.size());

    auto faithful = true;
    for (auto index = std::size_t{0}; index < production.members.size(); index++)
    {
        faithful = faithful && sameMember(production.members[index], mirrored.members[index]);
    }

    std::printf("  15 members, every recorded scalar: %s\n", faithful ? "BIT-IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(faithful);

    // --- 16.2 THE ARMS ---------------------------------------------------------------------------
    auto arms = std::vector<LegArm>{};
    arms.push_back(LegArm{"P production (all legs)", assists, Candidate{}});
    arms.push_back(LegArm{"O whole law OFF", withoutSlipAwareRecovery(assists), Candidate{}});
    arms.push_back(LegArm{"A dump-exit guard OFF", assists, replicaOf(false, true, true)});
    arms.push_back(LegArm{"B recover stuck OFF", assists, replicaOf(true, false, true)});
    arms.push_back(LegArm{"C reapply control OFF", assists, replicaOf(true, true, false)});

    // --- 16.3 THE LOCKED CONTROL, RE-RUN UNDER EVERY ARM -----------------------------------------
    std::printf("\n--- 16.3 the locked control, re-run under each arm's context ---\n");

    const auto locked =
        ensembleOf(setup.value(), world.value(), plain, calibration, Candidate{}, 1.0, steeredLock, steeredCount);

    for (const auto& arm : arms)
    {
        auto lockedAssists = plain;
        lockedAssists.antilock.recoveryAuthority = arm.assists.antilock.recoveryAuthority;

        const auto again = ensembleOf(setup.value(), world.value(), lockedAssists, calibration, arm.candidate, 1.0,
                                      steeredLock, steeredCount);

        REQUIRE(again.size() == locked.size());

        auto identical = true;
        for (auto index = std::size_t{0}; index < locked.size(); index++)
        {
            identical = identical && sameMember(again[index], locked[index]);
        }

        std::printf("  %-28s locked control %s\n", arm.name.c_str(), identical ? "IDENTICAL" : "*** DIFFERENT ***");

        // Fixture contamination if this ever fails, and every number below it is void.
        REQUIRE(identical);
    }

    auto ensembles = std::vector<LegEnsemble>{};
    ensembles.push_back(production);
    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        ensembles.push_back(
            legEnsemble(setup.value(), world.value(), arms[index].assists, calibration, arms[index].candidate));
    }

    for (const auto& ensemble : ensembles)
    {
        for (const auto& member : ensemble.members)
        {
            REQUIRE(member.stopped);
        }
    }

    const auto column = [&](const std::size_t arm, double Member::* field, const bool magnitude)
    {
        return columnOf(ensembles[arm].members, field, magnitude);
    };

    // --- 16.4 LATERAL TRAVEL, THE PRIMARY METRIC --------------------------------------------------
    std::printf("\n--- 16.4 LATERAL TRAVEL [m], magnitude. The primary steerability quantity. ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::lateralTravel, true)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(), distributionOf(column(index, &Member::lateralTravel, true)));
    }

    std::printf("\n  --- signed, so a sign flip cannot hide inside a magnitude ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::lateralTravel, false)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(), distributionOf(column(index, &Member::lateralTravel, false)));
    }

    std::printf("\n  --- PAIRED, member by member: |travel| / |locked travel| ---\n");
    std::printf("\n  %-4s %9s %8s", "k", "entry", "locked");
    for (const auto& arm : arms)
    {
        std::printf(" | %7s %6s", arm.name.substr(0, 7).c_str(), "ratio");
    }
    std::printf("\n  %s\n", "--------------------------------------------------------------------------------------"
                            "-------------------");

    auto passes = std::vector<std::size_t>(arms.size(), 0);
    auto ratioColumns = std::vector<std::vector<double>>(arms.size());

    for (auto member = std::size_t{0}; member < steeredCount; member++)
    {
        const auto reference = std::abs(locked[member].lateralTravel);

        std::printf("  %-4d %9.4f %8.3f", static_cast<int>(member) - 7, locked[member].entry, reference);

        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            const auto travel = std::abs(ensembles[index].members[member].lateralTravel);
            const auto ratio = reference > 1e-9 ? travel / reference : 0.0;

            ratioColumns[index].push_back(ratio);
            passes[index] += ratio > 3.0 ? 1 : 0;

            std::printf(" | %7.3f %6.2f", travel, ratio);
        }

        std::printf("\n");
    }

    std::printf("\n  --- the criterion, UNCHANGED: member lateral travel > 3 x that member's locked travel ---\n");
    std::printf("\n  %-28s %10s %12s %12s %10s %10s\n", "arm", "members", "median ratio", "min ratio", "max ratio",
                "sd");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto ratios = distributionOf(ratioColumns[index]);

        std::printf("  %-28s %5zu/%-4zu %12.3f %12.3f %10.3f %10.3f\n", arms[index].name.c_str(), passes[index],
                    steeredCount, ratios.median, ratios.minimum, ratios.maximum, ratios.deviation);
    }

    std::printf("\n  --- and as criterion 5 itself writes it: median(|assisted|) > 3 x median(|locked|) ---\n");

    const auto lockedMedian = distributionOf(columnOf(locked, &Member::lateralTravel, true)).median;

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto armMedian = distributionOf(column(index, &Member::lateralTravel, true)).median;

        std::printf("  %-28s %8.3f m against %8.3f m needed   %s   (%.3f x locked)\n", arms[index].name.c_str(),
                    armMedian, 3.0 * lockedMedian, armMedian > 3.0 * lockedMedian ? "PASS" : "FAIL",
                    armMedian / lockedMedian);
    }

    // --- 16.5 STOPPING DISTANCE -------------------------------------------------------------------
    std::printf("\n--- 16.5 STOPPING DISTANCE [m]. Secondary, and NOT optimised anywhere here. ---\n");
    distributionHeader("arm");
    distributionRow("locked (control)", distributionOf(columnOf(locked, &Member::distance, false)));
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        distributionRow(arms[index].name.c_str(), distributionOf(column(index, &Member::distance, false)));
    }

    // --- 16.6 TRUE SLIP ---------------------------------------------------------------------------
    std::printf("\n--- 16.6 MEAN TRUE SLIP, from the tyre and not from the estimator. Diagnostic. ---\n");
    std::printf("  The discriminator: does an ablation recreate the law-OFF high-slip state?\n");

    for (const auto& axis :
         std::vector<std::pair<const char*, double Member::*>>{{"front axle", &Member::meanSlipFront},
                                                               {"rear axle", &Member::meanSlipRear},
                                                               {"all four", &Member::meanSlipAll}})
    {
        std::printf("\n  mean |slip ratio|, %s\n", axis.first);
        distributionHeader("arm");
        distributionRow("locked (control)", distributionOf(columnOf(locked, axis.second, false)));
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            distributionRow(arms[index].name.c_str(), distributionOf(column(index, axis.second, false)));
        }
    }

    // --- 16.7 LATERAL FORCE AND UTILISATION -------------------------------------------------------
    std::printf("\n--- 16.7 LATERAL FORCE AND UTILISATION. `tyreFriction`-derived; NOT `gripUsed`. ---\n");
    std::printf("  latUse = sum|Fy| / sum(lateral capacity), the capacity being the same `tyreFriction`\n");
    std::printf("  call on the lateral axis at that tick's load and grip. `TyreForces::gripUsed` is out\n");
    std::printf("  of scope in this task and is not printed.\n");
    std::printf("\n  %-28s %10s %10s %10s %10s %10s %10s\n", "arm", "F |Fy| N", "R |Fy| N", "F latUse", "R latUse",
                "F longUse", "R longUse");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------");

    const auto forceRow = [](const char* name, const std::vector<Member>& members)
    {
        const auto value = [&](double Member::* field)
        {
            return distributionOf(columnOf(members, field, false)).median;
        };

        std::printf("  %-28s %10.1f %10.1f %10.4f %10.4f %10.4f %10.4f\n", name, value(&Member::lateralForceFront),
                    value(&Member::lateralForceRear), value(&Member::lateralUseFront), value(&Member::lateralUseRear),
                    value(&Member::frontUtilisation), value(&Member::rearUtilisation));
    };

    forceRow("locked (control)", locked);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        forceRow(arms[index].name.c_str(), ensembles[index].members);
    }

    std::printf("\n  --- the same quantities as distributions ---\n");

    for (const auto& field :
         std::vector<std::pair<const char*, double Member::*>>{{"front mean |Fy| [N]", &Member::lateralForceFront},
                                                               {"rear mean |Fy| [N]", &Member::lateralForceRear},
                                                               {"front lateral utilisation", &Member::lateralUseFront},
                                                               {"rear lateral utilisation", &Member::lateralUseRear}})
    {
        std::printf("\n  %s\n", field.first);
        distributionHeader("arm");
        distributionRow("locked (control)", distributionOf(columnOf(locked, field.second, false)));
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            distributionRow(arms[index].name.c_str(), distributionOf(column(index, field.second, false)));
        }
    }

    // --- 16.8 STATE-MACHINE DIFFERENTIAL, WHOLE ENSEMBLE ------------------------------------------
    std::printf("\n--- 16.8 STATE-MACHINE DIFFERENTIAL, summed over all %zu members, per channel ---\n", steeredCount);
    std::printf("  Counted off the controller's own steps (%.0f Hz). `empty s` is time at the actuator's\n",
                assists.controlRate);
    std::printf("  lower bound, seconds, summed over the ensemble.\n");
    std::printf("\n  %-28s %5s | %8s %8s %8s %8s | %7s %7s %7s | %9s\n", "arm", "chan", "Dump in", "Rec in", "Hold in",
                "Reapp in", "R->D", "R->A", "A->D", "empty s");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------"
                          "-------------------");

    const auto period = assists.controlRate > 0.0 ? 1.0 / assists.controlRate : tick;

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& counts = ensembles[index].transitions;

        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-28s %5s | %8zu %8zu %8zu %8zu | %7zu %7zu %7zu | %9.3f\n",
                        channel == 0 ? arms[index].name.c_str() : "",
                        channel == 0 ? "FL" : (channel == 1 ? "FR" : "rear"), counts.dumpIn[channel],
                        counts.recoverIn[channel], counts.holdIn[channel], counts.reapplyIn[channel],
                        counts.recoverDump[channel], counts.recoverReapply[channel], counts.reapplyDump[channel],
                        static_cast<double>(counts.emptySteps[channel]) * period);
        }
    }

    std::printf("\n  --- phase residency, share of braking ticks, summed over the ensemble ---\n");
    std::printf("\n  %-28s %5s | %8s %8s %8s %8s %8s\n", "arm", "wheel", "passive", "hold", "dump", "recover",
                "reapply");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& counts = ensembles[index].transitions;
        const auto total = static_cast<double>(std::max(counts.ticks, std::size_t{1}));

        for (auto wheel = std::size_t{0}; wheel < 3; wheel++)
        {
            std::printf("  %-28s %5s | %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n",
                        wheel == 0 ? arms[index].name.c_str() : "", wheel == 0 ? "FL" : (wheel == 1 ? "FR" : "rear"),
                        100.0 * static_cast<double>(counts.residency[wheel][0]) / total,
                        100.0 * static_cast<double>(counts.residency[wheel][1]) / total,
                        100.0 * static_cast<double>(counts.residency[wheel][2]) / total,
                        100.0 * static_cast<double>(counts.residency[wheel][3]) / total,
                        100.0 * static_cast<double>(counts.residency[wheel][4]) / total);
        }
    }

    std::printf("\n  --- cycling: dump entries per second of engagement, per channel, ensemble means ---\n");
    std::printf("\n  %-28s %12s %12s %12s\n", "arm", "FL Hz", "FR Hz", "rear Hz");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto& counts = ensembles[index].transitions;
        const auto engaged = [&](const std::size_t wheel)
        {
            const auto notPassive = counts.residency[wheel][1] + counts.residency[wheel][2] +
                                    counts.residency[wheel][3] + counts.residency[wheel][4];

            return static_cast<double>(notPassive) * tick;
        };

        std::printf("  %-28s %12.3f %12.3f %12.3f\n", arms[index].name.c_str(),
                    engaged(0) > 0.0 ? static_cast<double>(counts.dumpIn[0]) / engaged(0) : 0.0,
                    engaged(1) > 0.0 ? static_cast<double>(counts.dumpIn[1]) / engaged(1) : 0.0,
                    engaged(2) > 0.0 ? static_cast<double>(counts.dumpIn[2]) / engaged(2) : 0.0);
    }

    // --- 16.9 THE OUTLIER AND ANY NEW ONES --------------------------------------------------------
    std::printf("\n--- 16.9 THE k = -4 OUTLIER, and any new isolated bifurcation ---\n");
    std::printf("  Production's own outlier: member k = -4, lateral travel 8.209 m, paired ratio 2.787.\n");
    std::printf("\n  %-28s %10s %10s %10s %10s\n", "arm", "k=-4 [m]", "ratio", "members<3x", "which k");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto list = std::string{};

        for (auto member = std::size_t{0}; member < steeredCount; member++)
        {
            if (ratioColumns[index][member] <= 3.0)
            {
                list += list.empty() ? "" : ",";
                list += std::to_string(static_cast<int>(member) - 7);
            }
        }

        std::printf("  %-28s %10.3f %10.3f %10zu   %s\n", arms[index].name.c_str(),
                    std::abs(ensembles[index].members[3].lateralTravel), ratioColumns[index][3],
                    steeredCount - passes[index], list.empty() ? "none" : list.c_str());
    }

    // --- 16.10 THE REPRESENTATIVE MEMBER, TRACED --------------------------------------------------
    constexpr auto traced = std::size_t{2};

    std::printf("\n--- 16.10 THE REPRESENTATIVE MEMBER: k = %d, entry %.4f m/s ---\n", static_cast<int>(traced) - 7,
                memberEntry(traced, steeredCount));

    auto runs = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        runs.push_back(record(setup.value(), world.value(), arm.assists, calibration, arm.candidate, 1.0,
                              memberEntry(traced, steeredCount), steeredLock, true));
    }

    const auto lockedRun = record(setup.value(), world.value(), plain, calibration, Candidate{}, 1.0,
                                  memberEntry(traced, steeredCount), steeredLock, true);

    std::printf("\n  %-28s %9s %9s %9s %9s %9s %9s %9s\n", "arm", "stop [m]", "lat [m]", "t [s]", "F slip", "R slip",
                "F |Fy| N", "R |Fy| N");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------"
                          "----------");

    const auto memberRow = [&](const char* name, const Run& run)
    {
        const auto member = memberOf(traced, run.entrySpeed, run);

        std::printf("  %-28s %9.3f %9.3f %9.3f %9.4f %9.4f %9.1f %9.1f\n", name, member.distance, member.lateralTravel,
                    member.time, member.meanSlipFront, member.meanSlipRear, member.lateralForceFront,
                    member.lateralForceRear);
    };

    memberRow("locked (control)", lockedRun);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        memberRow(arms[index].name.c_str(), runs[index]);
    }

    std::printf("\n  --- half-second windows on the traced member. Speed, lateral travel, axle slip, |Fy| ---\n");

    const auto windows = std::size_t{16};

    const auto windowRow = [&](const char* name, const Run& run, const char* what, auto&& value)
    {
        std::printf("  %-24s %-9s", name, what);

        for (auto window = std::size_t{0}; window < windows; window++)
        {
            const auto from = 0.5 * static_cast<double>(window);
            const auto to = 0.5 * static_cast<double>(window + 1);

            std::printf(" %7.3f", value(run, from, to));
        }

        std::printf("\n");
    };

    std::printf("\n  seconds:                     ");
    for (auto window = std::size_t{0}; window < windows; window++)
    {
        std::printf(" %7.1f", 0.5 * static_cast<double>(window + 1));
    }
    std::printf("\n");

    const auto speedIn = [](const Run& run, const double from, const double to)
    {
        auto last = 0.0;
        for (const auto& sample : run.ticks)
        {
            if (sample.time > from && sample.time <= to)
            {
                last = sample.speed;
            }
        }

        return last;
    };

    const auto lateralIn = [](const Run& run, const double from, const double to)
    {
        auto last = 0.0;
        for (const auto& sample : run.ticks)
        {
            if (sample.time > from && sample.time <= to)
            {
                last = sample.lateralTravel;
            }
        }

        return last;
    };

    std::printf("\n  VEHICLE SPEED [m/s], value at the end of each window\n");
    windowRow("locked (control)", lockedRun, "speed", speedIn);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        windowRow(arms[index].name.c_str(), runs[index], "speed", speedIn);
    }

    std::printf("\n  LATERAL DISPLACEMENT [m], signed, value at the end of each window\n");
    windowRow("locked (control)", lockedRun, "lateral", lateralIn);
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        windowRow(arms[index].name.c_str(), runs[index], "lateral", lateralIn);
    }

    struct Axle
    {
        const char* name;
        std::size_t first;
        std::size_t last;
    };

    for (const auto& axle : std::vector<Axle>{{"FRONT", 0, 1}, {"REAR", 2, 3}})
    {
        const auto first = axle.first;
        const auto last = axle.last;

        std::printf("\n  %s axle TRUE |slip ratio|\n", axle.name);
        const auto slipIn = [&](const Run& run, const double from, const double to)
        {
            return windowAxleMean(run, first, last, &WheelTick::slipRatio, from, to, true);
        };

        windowRow("locked (control)", lockedRun, "slip", slipIn);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            windowRow(arms[index].name.c_str(), runs[index], "slip", slipIn);
        }

        std::printf("\n  %s axle |Fy| per wheel [N]\n", axle.name);
        const auto lateralForceIn = [&](const Run& run, const double from, const double to)
        {
            return windowAxleMean(run, first, last, &WheelTick::lateral, from, to, true);
        };

        windowRow("locked (control)", lockedRun, "Fy", lateralForceIn);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            windowRow(arms[index].name.c_str(), runs[index], "Fy", lateralForceIn);
        }

        std::printf("\n  %s axle commanded PRESSURE [bar]\n", axle.name);
        const auto pressureIn = [&](const Run& run, const double from, const double to)
        {
            return windowAxleMean(run, first, last, &WheelTick::pressure, from, to, false) / bar;
        };

        windowRow("locked (control)", lockedRun, "bar", pressureIn);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            windowRow(arms[index].name.c_str(), runs[index], "bar", pressureIn);
        }
    }

    std::printf("\n  --- the first 0.6 s of the traced member at 20 ms, all five arms, per channel ---\n");
    std::printf("  phase letters: . passive  H hold  D dump  R recover  A reapply\n");
    std::printf("  Columns are FL / FR / rear. `gS` is the guard slip the law reads, `sS` the raw sensed\n");
    std::printf("  slip the pre-law transitions read; both are the FL channel's.\n");
    std::printf("\n  %-24s %6s | %5s %5s %5s | %7s %7s %7s | %6s %6s\n", "arm", "t [s]", "FL", "FR", "rear", "FL bar",
                "FR bar", "R bar", "gS", "sS");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------"
                          "----");

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        for (auto slot = std::size_t{0}; slot <= 30; slot++)
        {
            const auto when = 0.02 * static_cast<double>(slot);

            const Tick* found = nullptr;
            for (const auto& sample : runs[index].ticks)
            {
                if (sample.time >= when)
                {
                    found = &sample;
                    break;
                }
            }

            if (found == nullptr)
            {
                break;
            }

            auto guardSlipAt = 0.0;
            auto sensedSlipAt = 0.0;
            for (const auto& step : runs[index].steps)
            {
                if (step.channel != 0 || step.time > found->time)
                {
                    continue;
                }

                guardSlipAt = step.guardSlip;
                sensedSlipAt = step.slip;
            }

            std::printf("  %-24s %6.3f | %5c %5c %5c | %7.1f %7.1f %7.1f | %6.3f %6.3f\n",
                        slot == 0 ? arms[index].name.c_str() : "", found->time, phaseLetter(found->wheels[0].phase),
                        phaseLetter(found->wheels[1].phase), phaseLetter(found->wheels[2].phase),
                        found->wheels[0].pressure / bar, found->wheels[1].pressure / bar,
                        found->wheels[2].pressure / bar, guardSlipAt, sensedSlipAt);
        }

        std::printf("\n");
    }

    // --- 16.11 FIRST DIVERGENCE -------------------------------------------------------------------
    std::printf("\n--- 16.11 FIRST CAUSAL DIVERGENCE FROM PRODUCTION, on the traced member ---\n");
    std::printf("  The two step traces are index-aligned: the controller clock advances on deltaTime\n");
    std::printf("  alone, so the step sequence is identical whatever the controller decides. The first\n");
    std::printf("  index at which anything differs is therefore the first causal difference, and every\n");
    std::printf("  later difference is downstream of it.\n");

    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        const auto any = firstDivergence(runs[0], runs[index], -1);

        std::printf("\n  %s\n", arms[index].name.c_str());

        if (!any.found)
        {
            std::printf("    NO DIVERGENCE at any controller step. This arm is bit-identical to production.\n");
            continue;
        }

        std::printf("    first difference   t = %.5f s, step %zu, channel %s\n", any.time, any.index,
                    any.channel == 0 ? "FL" : (any.channel == 1 ? "FR" : "rear"));
        std::printf("    state before       %s,  pastBand %s,  guardSlip %.4f,  sensed slip %.4f\n",
                    modulatorName(any.before), any.pastBand ? "TRUE" : "false", any.guardSlip, any.slip);
        std::printf("    wheel accel        %.2f m/s^2,  excess %.2f m/s^2,  losing %s\n", any.acceleration, any.excess,
                    any.acceleration < assists.antilock.lockDeceleration ? "TRUE" : "false");
        std::printf("    production takes   %s -> %s,  pressure %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.productionAfter), any.productionPressure / bar);
        std::printf("    this arm takes     %s -> %s,  pressure %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.armAfter), any.armPressure / bar);
        std::printf("    pressure created   %+.5f bar (%s)\n", (any.armPressure - any.productionPressure) / bar,
                    any.phaseDiffers ? "a transition difference" : "a gradient difference, same phase");

        std::printf("    per channel:");
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            const auto each = firstDivergence(runs[0], runs[index], static_cast<long>(channel));

            std::printf("  %s %s", channel == 0 ? "FL" : (channel == 1 ? "FR" : "rear"), each.found ? "" : "never");
            if (each.found)
            {
                std::printf("%.4f s", each.time);
            }
        }
        std::printf("\n");

        // What the divergence then did, in the fixture's own quantities.
        const auto after = std::min(any.time + 1.0, 30.0);
        std::printf("    in the second after it (%.3f-%.3f s), production -> this arm:\n", any.time, after);
        std::printf("      front slip  %.4f -> %.4f     rear slip  %.4f -> %.4f\n",
                    windowAxleMean(runs[0], 0, 1, &WheelTick::slipRatio, any.time, after, true),
                    windowAxleMean(runs[index], 0, 1, &WheelTick::slipRatio, any.time, after, true),
                    windowAxleMean(runs[0], 2, 3, &WheelTick::slipRatio, any.time, after, true),
                    windowAxleMean(runs[index], 2, 3, &WheelTick::slipRatio, any.time, after, true));
        std::printf("      front |Fy|  %.1f -> %.1f N   rear |Fy|  %.1f -> %.1f N\n",
                    windowAxleMean(runs[0], 0, 1, &WheelTick::lateral, any.time, after, true),
                    windowAxleMean(runs[index], 0, 1, &WheelTick::lateral, any.time, after, true),
                    windowAxleMean(runs[0], 2, 3, &WheelTick::lateral, any.time, after, true),
                    windowAxleMean(runs[index], 2, 3, &WheelTick::lateral, any.time, after, true));
        std::printf("      front bar   %.2f -> %.2f      rear bar   %.2f -> %.2f\n",
                    windowAxleMean(runs[0], 0, 1, &WheelTick::pressure, any.time, after, false) / bar,
                    windowAxleMean(runs[index], 0, 1, &WheelTick::pressure, any.time, after, false) / bar,
                    windowAxleMean(runs[0], 2, 3, &WheelTick::pressure, any.time, after, false) / bar,
                    windowAxleMean(runs[index], 2, 3, &WheelTick::pressure, any.time, after, false) / bar);
        std::printf("    eventual path: lateral %.3f m against production's %.3f m; stop %.3f m against %.3f m\n",
                    runs[index].lateralTravel, runs[0].lateralTravel, runs[index].distance, runs[0].distance);
    }

    // --- 16.12 THE REAR-FIRST HYPOTHESIS ----------------------------------------------------------
    std::printf("\n--- 16.12 REAR-FIRST OR FRONT-FIRST, per arm, on the traced member ---\n");
    std::printf("\n  %-28s %12s %12s %12s   %s\n", "arm", "FL first", "FR first", "rear first", "verdict");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------"
                          "----");

    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        auto when = std::array<double, brakeChannelCount>{};
        auto seen = std::array<bool, brakeChannelCount>{};

        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            const auto each = firstDivergence(runs[0], runs[index], static_cast<long>(channel));
            seen[channel] = each.found;
            when[channel] = each.found ? each.time : 1e9;
        }

        const auto frontFirst = std::min(when[0], when[1]);
        const auto rearFirst = when[2];

        const auto verdict = (!seen[0] && !seen[1] && !seen[2]) ? "no divergence at all"
                             : (rearFirst < frontFirst)         ? "REAR diverges first"
                             : (frontFirst < rearFirst)         ? "FRONT diverges first"
                                                                : "front and rear together";

        const auto text = [&](const std::size_t channel)
        {
            return seen[channel] ? std::to_string(when[channel]) : std::string{"never"};
        };

        std::printf("  %-28s %12s %12s %12s   %s\n", arms[index].name.c_str(), text(0).c_str(), text(1).c_str(),
                    text(2).c_str(), verdict);
    }

    std::printf("\n  --- and what the REAR channel did at production's first stuck-branch dump ---\n");
    std::printf("  Production's Recover -> Dump on the rear is leg B firing. Section 16.0 predicts that\n");
    std::printf("  'B OFF' HOLDS in Recover there rather than re-applying, because leg D is still on.\n");

    auto stuckTime = -1.0;
    for (const auto& step : runs[0].steps)
    {
        if (step.channel == 2 && step.before == ModulatorPhase::Recover && step.after == ModulatorPhase::Dump &&
            !(step.slip > assists.antilock.slipEnter && step.acceleration < assists.antilock.lockDeceleration))
        {
            stuckTime = step.time;
            break;
        }
    }

    std::printf("\n  production's first rear stuck-branch dump: %s", stuckTime < 0.0 ? "never\n" : "");
    if (stuckTime >= 0.0)
    {
        std::printf("t = %.5f s\n", stuckTime);

        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            const Step* at = nullptr;
            for (const auto& step : runs[index].steps)
            {
                if (step.channel == 2 && step.time >= stuckTime)
                {
                    at = &step;
                    break;
                }
            }

            if (at == nullptr)
            {
                continue;
            }

            std::printf("    %-28s rear at t=%.5f: %s -> %s, %.3f bar, guardSlip %.4f, excess %+.2f\n",
                        arms[index].name.c_str(), at->time, modulatorName(at->before), modulatorName(at->after),
                        at->pressureAfter / bar, at->guardSlip, at->excess);
        }
    }

    // --- 16.13 CLOSING SUMMARY --------------------------------------------------------------------
    std::printf("\n--- 16.13 SUMMARY TABLE: every headline quantity, one row per arm ---\n");
    std::printf("\n  %-28s %9s %7s %7s %9s %8s %8s %9s\n", "arm", "lat med", "ratio", ">3x", "stop med", "slip F",
                "slip R", "F latUse");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------"
                          "-------");

    {
        const auto value = [&](const std::vector<Member>& members, double Member::* field, const bool magnitude)
        {
            return distributionOf(columnOf(members, field, magnitude)).median;
        };

        std::printf("  %-28s %9.3f %7s %7s %9.3f %8.4f %8.4f %9.4f\n", "locked (control)",
                    value(locked, &Member::lateralTravel, true), "1.000", "-", value(locked, &Member::distance, false),
                    value(locked, &Member::meanSlipFront, false), value(locked, &Member::meanSlipRear, false),
                    value(locked, &Member::lateralUseFront, false));

        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            const auto& members = ensembles[index].members;
            auto count = std::to_string(passes[index]) + "/" + std::to_string(steeredCount);

            std::printf("  %-28s %9.3f %7.3f %7s %9.3f %8.4f %8.4f %9.4f\n", arms[index].name.c_str(),
                        value(members, &Member::lateralTravel, true), distributionOf(ratioColumns[index]).median,
                        count.c_str(), value(members, &Member::distance, false),
                        value(members, &Member::meanSlipFront, false), value(members, &Member::meanSlipRear, false),
                        value(members, &Member::lateralUseFront, false));
        }
    }

    std::printf("\n  Production is unchanged by this file. `Candidate::replica` is false everywhere above\n");
    std::printf("  section 16, and section 16.1 proves the copy reproduces production to the bit.\n");
}

// =============================================================================================
// 17. LEG A ACROSS THE SURFACES — DOES THE LEG THAT CARRIES THE BENEFIT ALSO CARRY THE COST?
//
// Section 16 established that ONE of the three legs, the Dump-exit guard, carries essentially all
// of the steered low-mu steerability the law buys. This section asks the other half of the
// question and nothing else: on the two surfaces where the law is known to COST something — the
// dry straight stop, where it strands the rear axle, and split mu, where it costs yaw — does the
// same leg carry that cost?
//
// THREE ARMS, and the task fixes them:
//
//   P  production                    all four pastBand consumers live
//   O  whole law OFF                 `slipAwareRecovery` false; `pastBand` constant false
//   A  leg A OFF only                the probe replica with legA=false, legB=legC=true
//
// **Leg D is ON in P and in A, and OFF only in O.** That is the same asymmetry section 16.0
// stated, and it is why an A-vs-O difference is not by itself an attribution to leg A alone.
//
// Nothing is tuned. No threshold moves. No new controller exists. Every fixture below is the one
// already in this file: `ensembleCount` members on `gripPlate`, full pedal, the same settle, the
// same stopping condition, the same stranding separator `rearUtilisation < strandedRear`.
// =============================================================================================

namespace
{

struct CrossArm
{
    std::string name;
    std::string tag;
    AssistSetup assists;
    Candidate candidate;
};

// The ensemble, its per-member last-airborne time, and the state machine underneath it. One pass:
// the trace flag is on, so the transitions come off the same runs the members do.
struct CrossEnsemble
{
    std::vector<Member> members;
    std::vector<double> lastAirborne;
    Transitions transitions;
};

// The last instant either rear wheel was off the ground. Context for the stranded state, and the
// n=29 result already falsified it as a separator, so it is printed and not leaned on.
[[nodiscard]] double lastAirborneOf(const Run& run)
{
    auto last = 0.0;

    for (const auto& sample : run.ticks)
    {
        if (!sample.wheels[2].inContact || !sample.wheels[3].inContact)
        {
            last = sample.time;
        }
    }

    return last;
}

[[nodiscard]] CrossEnsemble crossEnsemble(const VehicleSetup& setup, const PhysicsWorld& world,
                                          const AssistSetup& assists, const RoadTorqueCalibration& calibration,
                                          const Candidate& candidate, const double steering)
{
    auto ensemble = CrossEnsemble{};
    ensemble.members.reserve(ensembleCount);
    ensemble.lastAirborne.reserve(ensembleCount);

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);
        const auto run = record(setup, world, assists, calibration, candidate, 1.0, entry, steering, true);

        ensemble.members.push_back(memberOf(index, entry, run));
        ensemble.lastAirborne.push_back(lastAirborneOf(run));
        ensemble.transitions.add(transitionsOf(run));
    }

    return ensemble;
}

void wideHeader(const char* what)
{
    std::printf("\n  %-26s %4s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", what, "n", "min", "P10", "P25", "MEDIAN", "mean",
                "P75", "P90", "max", "sd");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------------------------------------");
}

void wideRow(const char* name, const Distribution& value)
{
    std::printf("  %-26s %4zu %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n", name, value.n, value.minimum,
                value.p10, value.p25, value.median, value.mean, value.p75, value.p90, value.maximum, value.deviation);
}

[[nodiscard]] std::size_t strandedCount(const std::vector<Member>& members)
{
    auto count = std::size_t{0};

    for (const auto& member : members)
    {
        count += member.stranded() ? 1 : 0;
    }

    return count;
}

[[nodiscard]] std::vector<double> airborneColumn(const std::vector<Member>& members)
{
    auto values = std::vector<double>{};

    for (const auto& member : members)
    {
        values.push_back(static_cast<double>(member.rearAirborne));
    }

    return values;
}

[[nodiscard]] const char* channelName(const std::size_t channel)
{
    return channel == 0 ? "FL" : (channel == 1 ? "FR" : "rear");
}

// Every transition the deliverable names, per channel, in seconds where a count is a duration.
void printTransitionBlock(const char* name, const Transitions& counts, const double period)
{
    for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
    {
        const auto steps = static_cast<double>(std::max(counts.steps[channel], std::size_t{1}));

        std::printf("  %-24s %5s %9zu %8zu %8zu %8zu %8zu %8zu %8zu %10.3f %8.1f%%\n", name, channelName(channel),
                    counts.steps[channel], counts.dumpIn[channel], counts.recoverIn[channel], counts.holdIn[channel],
                    counts.reapplyIn[channel], counts.recoverDump[channel], counts.recoverReapply[channel],
                    static_cast<double>(counts.emptySteps[channel]) * period,
                    100.0 * static_cast<double>(counts.emptySteps[channel]) / steps);
        name = "";
    }
}

void printTransitionHeader()
{
    std::printf("\n  %-24s %5s %9s %8s %8s %8s %8s %8s %8s %10s %9s\n", "arm", "chan", "steps", "Dump in", "Recov in",
                "Hold in", "Reapp in", "Rec->Dmp", "Rec->Rea", "empty [s]", "empty %");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "--------------------------------");
}

void printResidency(const char* name, const Transitions& counts)
{
    const auto ticks = static_cast<double>(std::max(counts.ticks, std::size_t{1}));

    for (auto wheel = std::size_t{0}; wheel < 3; wheel++)
    {
        std::printf("  %-24s %6s", name, wheel == 0 ? "FL" : (wheel == 1 ? "FR" : "RL"));

        for (auto phase = std::size_t{0}; phase < 5; phase++)
        {
            std::printf(" %9.2f%%", 100.0 * static_cast<double>(counts.residency[wheel][phase]) / ticks);
        }

        std::printf("\n");
        name = "";
    }
}

// The Reapply -> Dump count the deliverable asks for, kept separate because the header above is
// already at its width.
void printReapplyDump(const char* name, const Transitions& counts)
{
    std::printf("  %-24s", name);

    for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
    {
        std::printf("  %s %6zu", channelName(channel), counts.reapplyDump[channel]);
    }

    std::printf("\n");
}

// The latest controller step of a channel at or before each physics tick, so a per-tick table can
// print what the ECU had decided by the time that tick was integrated. Both sequences are sorted
// in time, so this is a merge and not a search.
[[nodiscard]] std::vector<const Step*> perTickSteps(const Run& run, const std::size_t channel)
{
    auto ordered = std::vector<const Step*>{};

    for (const auto& step : run.steps)
    {
        if (step.channel == channel)
        {
            ordered.push_back(&step);
        }
    }

    auto latest = std::vector<const Step*>(run.ticks.size(), nullptr);
    auto cursor = std::size_t{0};
    const Step* current = nullptr;

    for (auto index = std::size_t{0}; index < run.ticks.size(); index++)
    {
        const auto until = run.ticks[index].time + 1.0e-9;

        while (cursor < ordered.size() && ordered[cursor]->time <= until)
        {
            current = ordered[cursor];
            cursor++;
        }

        latest[index] = current;
    }

    return latest;
}

[[nodiscard]] double axleShare(const Tick& sample, const std::size_t first, const std::size_t last)
{
    auto force = 0.0;
    auto capacity = 0.0;

    for (auto index = first; index <= last; index++)
    {
        force += std::abs(sample.wheels[index].longitudinal);
        capacity += sample.wheels[index].capacity;
    }

    return capacity > 1.0 ? force / capacity : 0.0;
}

// The dry mechanism table: what the rear channel knew, what it commanded, and what the axle did.
void printDryTrace(const char* name, const Run& run, const double every)
{
    const auto steps = perTickSteps(run, 2);

    std::printf("\n  --- %s: rear channel, every %.0f ms ---\n", name, every * 1000.0);
    std::printf("  %8s %9s %8s %9s %9s %9s %9s %9s %8s %8s %9s\n", "t [s]", "phase", "bar", "raw slip", "guardSlip",
                "true slip", "rear Fz N", "front u", "rear u", "v [m/s]", "dist [m]");

    auto travelled = 0.0;
    auto next = 0.0;

    for (auto index = std::size_t{0}; index < run.ticks.size(); index++)
    {
        const auto& sample = run.ticks[index];
        travelled += sample.speed * tick;

        if (sample.time + 1.0e-9 < next)
        {
            continue;
        }
        next = sample.time + every;

        const auto* step = steps[index];

        std::printf("  %8.3f %9s %8.3f %9.4f %9.4f %9.4f %9.1f %9.4f %8.4f %8.3f %9.3f\n", sample.time,
                    modulatorName(sample.wheels[2].phase), sample.wheels[2].pressure / bar,
                    step != nullptr ? step->slip : 0.0, step != nullptr ? step->guardSlip : 0.0,
                    sample.wheels[2].slipRatio, sample.wheels[2].load + sample.wheels[3].load, axleShare(sample, 0, 1),
                    axleShare(sample, 2, 3), sample.speed, travelled);
    }
}

// The split-mu mechanism table: the two sides side by side, and what their difference did to yaw.
void printSplitTrace(const char* name, const Run& run, const double every)
{
    const auto front = perTickSteps(run, 0);
    const auto rightFront = perTickSteps(run, 1);
    const auto rear = perTickSteps(run, 2);

    std::printf("\n  --- %s: high-mu LEFT against low-mu RIGHT, every %.0f ms ---\n", name, every * 1000.0);
    std::printf("  %7s %7s %7s %7s %7s %8s %8s %8s %8s %9s %9s %9s %8s %8s %8s\n", "t [s]", "wL r/s", "wR r/s", "rawL",
                "rawR", "guardL", "guardR", "phaseL", "phaseR", "barL", "barR", "Fx L N", "Fx R N", "yawRate",
                "yaw deg");

    auto next = 0.0;

    for (auto index = std::size_t{0}; index < run.ticks.size(); index++)
    {
        const auto& sample = run.ticks[index];

        if (sample.time + 1.0e-9 < next)
        {
            continue;
        }
        next = sample.time + every;

        const auto* left = front[index];
        const auto* right = rightFront[index];

        std::printf(
            "  %7.3f %7.2f %7.2f %7.4f %7.4f %8.4f %8.4f %8s %8s %9.3f %9.3f %9.1f %8.1f %8.3f %8.3f\n", sample.time,
            sample.wheels[0].wheelSpeed, sample.wheels[1].wheelSpeed, left != nullptr ? left->slip : 0.0,
            right != nullptr ? right->slip : 0.0, left != nullptr ? left->guardSlip : 0.0,
            right != nullptr ? right->guardSlip : 0.0, modulatorName(sample.wheels[0].phase),
            modulatorName(sample.wheels[1].phase), sample.wheels[0].pressure / bar, sample.wheels[1].pressure / bar,
            sample.wheels[0].longitudinal + sample.wheels[2].longitudinal,
            sample.wheels[1].longitudinal + sample.wheels[3].longitudinal, sample.yawRate, sample.yaw * degrees);
    }

    // The rear channel is one channel for both wheels, so it gets its own line rather than a column.
    auto rearDumps = std::size_t{0};
    for (const auto* step : rear)
    {
        rearDumps += (step != nullptr && step->after == ModulatorPhase::Dump) ? 1 : 0;
    }
    std::printf("  rear channel (one for both wheels): %zu of %zu ticks commanded Dump\n", rearDumps, run.ticks.size());
}

// The left/right longitudinal force imbalance, and the yaw moment it makes about the car's centre,
// integrated over a window. Diagnostic arithmetic on recorded tyre forces; nothing reads it.
[[nodiscard]] double meanImbalance(const Run& run, const double from, const double to)
{
    auto total = 0.0;
    auto count = std::size_t{0};

    for (const auto& sample : run.ticks)
    {
        if (sample.time <= from || sample.time > to)
        {
            continue;
        }

        const auto left = sample.wheels[0].longitudinal + sample.wheels[2].longitudinal;
        const auto right = sample.wheels[1].longitudinal + sample.wheels[3].longitudinal;

        total += left - right;
        count++;
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

} // namespace

TEST_CASE("does the leg that carries the steered low-mu benefit also carry the dry and split-mu costs",
          "[.leg-a-cross]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto dry = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(dry.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto period = assists.controlRate > 0.0 ? 1.0 / assists.controlRate : tick;

    REQUIRE_FALSE(assists.antilock.yawMomentDelay);

    const auto replicaOfLegs = [](const bool legA, const bool legB, const bool legC)
    {
        auto candidate = Candidate{};
        candidate.replica = true;
        candidate.legA = legA;
        candidate.legB = legB;
        candidate.legC = legC;

        return candidate;
    };

    auto arms = std::vector<CrossArm>{};
    arms.push_back(CrossArm{"P production", "P", assists, Candidate{}});
    arms.push_back(CrossArm{"O whole law OFF", "O", withoutSlipAwareRecovery(assists), Candidate{}});
    arms.push_back(CrossArm{"A leg-A (dump exit) OFF", "A", assists, replicaOfLegs(false, true, true)});

    std::printf("\n=== SECTION 17: LEG A ACROSS THE SURFACES ===\n");
    std::printf("  Three arms and no others. P production, O whole law off, A leg-A off with B, C and D\n");
    std::printf("  still live. %zu deterministic members per surface, entry %.4f to %.4f m/s, full pedal,\n",
                ensembleCount, memberEntry(0, ensembleCount), memberEntry(ensembleCount - 1, ensembleCount));
    std::printf("  no steering. Stranding is the existing separator: rear utilisation < %.2f.\n", strandedRear);
    std::printf("  Controller period %.5f s (%.0f Hz).\n", period, assists.controlRate);
    std::printf("\n  CARRIED FORWARD, NOT RE-RUN — the steered low-mu result of section 16:\n");
    std::printf("      arm   lateral med   paired ratio   >3x     front slip   front |Fy|\n");
    std::printf("      P        15.991 m        5.242    14/15        0.175      498.5 N\n");
    std::printf("      O         3.310 m        1.124     0/15        0.358      391.9 N\n");
    std::printf("      A         6.654 m        2.189     3/15        0.408      320.9 N\n");

    // --- 17.1 THE REPLICA IS PROVED ON BOTH SURFACES BEFORE ANYTHING IS READ ---------------------
    std::printf("\n--- 17.1 REPLICA FIDELITY: the probe copy with all three legs ON, against production ---\n");
    std::printf("  Section 16.1 proved this on the steered low-mu fixture. These are the two fixtures\n");
    std::printf("  this task actually reads, so the identity is re-proved on each of them.\n");

    const auto dryProduction = crossEnsemble(setup.value(), dry.value(), assists, calibration, Candidate{}, 0.0);
    const auto dryMirror =
        crossEnsemble(setup.value(), dry.value(), assists, calibration, replicaOfLegs(true, true, true), 0.0);

    REQUIRE(dryMirror.members.size() == dryProduction.members.size());

    auto dryFaithful = true;
    for (auto index = std::size_t{0}; index < dryProduction.members.size(); index++)
    {
        dryFaithful = dryFaithful && sameMember(dryProduction.members[index], dryMirror.members[index]);
    }

    std::printf("  DRY,      %zu members, every recorded scalar: %s\n", ensembleCount,
                dryFaithful ? "BIT-IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(dryFaithful);

    const auto splitProduction = crossEnsemble(setup.value(), split.value(), assists, calibration, Candidate{}, 0.0);
    const auto splitMirror =
        crossEnsemble(setup.value(), split.value(), assists, calibration, replicaOfLegs(true, true, true), 0.0);

    REQUIRE(splitMirror.members.size() == splitProduction.members.size());

    auto splitFaithful = true;
    for (auto index = std::size_t{0}; index < splitProduction.members.size(); index++)
    {
        splitFaithful = splitFaithful && sameMember(splitProduction.members[index], splitMirror.members[index]);
    }

    std::printf("  SPLIT MU, %zu members, every recorded scalar: %s\n", ensembleCount,
                splitFaithful ? "BIT-IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(splitFaithful);

    // =============================================================================================
    // EXPERIMENT 1 — DRY STRAIGHT
    // =============================================================================================
    std::printf("\n\n=== EXPERIMENT 1: DRY STRAIGHT, grip 1.00 both sides ===\n");

    auto dryEnsembles = std::vector<CrossEnsemble>{};
    dryEnsembles.push_back(dryProduction);
    dryEnsembles.push_back(
        crossEnsemble(setup.value(), dry.value(), arms[1].assists, calibration, arms[1].candidate, 0.0));
    dryEnsembles.push_back(
        crossEnsemble(setup.value(), dry.value(), arms[2].assists, calibration, arms[2].candidate, 0.0));

    for (const auto& ensemble : dryEnsembles)
    {
        for (const auto& member : ensemble.members)
        {
            REQUIRE(member.stopped);
        }
    }

    std::printf("\n--- 17.2 DRY STOPPING DISTANCE [m] ---\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(),
                distributionOf(columnOf(dryEnsembles[index].members, &Member::distance, false)));
    }

    std::printf("\n--- 17.3 DRY REAR STRANDING, the existing separator (rear utilisation < %.2f) ---\n", strandedRear);
    std::printf("\n  %-26s %12s %12s\n", "arm", "stranded", "fraction");
    std::printf("  %s\n", "----------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto count = strandedCount(dryEnsembles[index].members);
        std::printf("  %-26s %8zu/%-3zu %11.3f\n", arms[index].name.c_str(), count, ensembleCount,
                    static_cast<double>(count) / static_cast<double>(ensembleCount));
    }

    std::printf("\n--- 17.4 DRY UTILISATION AND AIRBORNE CONTEXT ---\n");
    std::printf("  Airborne duration is CONTEXT. The n=%zu result already falsified a simple\n", ensembleCount);
    std::printf("  airborne-tick separator, and nothing below treats it as a predictor.\n");

    for (const auto& axis : std::vector<std::pair<const char*, double Member::*>>{
             {"REAR utilisation", &Member::rearUtilisation}, {"FRONT utilisation", &Member::frontUtilisation}})
    {
        std::printf("\n  %s\n", axis.first);
        wideHeader("arm");
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            wideRow(arms[index].name.c_str(),
                    distributionOf(columnOf(dryEnsembles[index].members, axis.second, false)));
        }
    }

    std::printf("\n  REAR AIRBORNE TICKS (both rear wheels down = not airborne)\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(), distributionOf(airborneColumn(dryEnsembles[index].members)));
    }

    std::printf("\n  LAST AIRBORNE TIME [s]\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(), distributionOf(dryEnsembles[index].lastAirborne));
    }

    // --- 17.5 THE MEMBER MAP ----------------------------------------------------------------------
    std::printf("\n--- 17.5 DRY MEMBER-BY-MEMBER MAP, all %zu deterministic members ---\n", ensembleCount);
    std::printf("  The question this table answers: does A remove the SAME stranded population that O\n");
    std::printf("  removes, or does it move the boundary of a chaotic cluster?\n");
    std::printf("\n  %-3s %9s | %8s %8s %8s | %5s %5s %5s | %8s %8s %8s\n", "k", "entry m/s", "P stop", "O stop",
                "A stop", "P str", "O str", "A str", "P rear u", "O rear u", "A rear u");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "------------");

    auto dryStrandedP = std::vector<std::size_t>{};
    auto dryStrandedA = std::vector<std::size_t>{};

    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        const auto& p = dryEnsembles[0].members[member];
        const auto& o = dryEnsembles[1].members[member];
        const auto& a = dryEnsembles[2].members[member];

        if (p.stranded())
        {
            dryStrandedP.push_back(member);
        }
        if (a.stranded())
        {
            dryStrandedA.push_back(member);
        }

        std::printf("  %-3zu %9.4f | %8.3f %8.3f %8.3f | %5s %5s %5s | %8.4f %8.4f %8.4f\n", member, p.entry,
                    p.distance, o.distance, a.distance, p.stranded() ? "YES" : ".", o.stranded() ? "YES" : ".",
                    a.stranded() ? "YES" : ".", p.rearUtilisation, o.rearUtilisation, a.rearUtilisation);
    }

    {
        auto shared = std::size_t{0};
        for (const auto member : dryStrandedA)
        {
            shared += dryEnsembles[0].members[member].stranded() ? 1 : 0;
        }

        auto newlyStranded = std::size_t{0};
        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            newlyStranded +=
                (dryEnsembles[2].members[member].stranded() && !dryEnsembles[0].members[member].stranded()) ? 1 : 0;
        }

        std::printf("\n  production-stranded set: %zu members. leg-A-off stranded set: %zu members.\n",
                    dryStrandedP.size(), dryStrandedA.size());
        std::printf("  members stranded under BOTH P and A: %zu. Stranded under A but NOT under P: %zu.\n", shared,
                    newlyStranded);
        std::printf("  whole-law-off stranded set: %zu members.\n", strandedCount(dryEnsembles[1].members));
    }

    // --- 17.6 THE STATE MACHINE -------------------------------------------------------------------
    std::printf("\n--- 17.6 DRY STATE-MACHINE DIFFERENTIAL, summed over all %zu members ---\n", ensembleCount);
    printTransitionHeader();
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printTransitionBlock(arms[index].name.c_str(), dryEnsembles[index].transitions, period);
    }

    std::printf("\n  Reapply -> Dump, per channel\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printReapplyDump(arms[index].name.c_str(), dryEnsembles[index].transitions);
    }

    std::printf("\n  PHASE RESIDENCY, share of physics ticks (FL, FR and the rear pair's left wheel)\n");
    std::printf("\n  %-24s %6s %10s %10s %10s %10s %10s\n", "arm", "wheel", "passive", "hold", "dump", "recover",
                "reapply");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-----");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printResidency(arms[index].name.c_str(), dryEnsembles[index].transitions);
    }

    std::printf("\n  The interaction section 16 recorded: with A off, leg B can oscillate Dump <-> Recover.\n");
    std::printf("  Rear Recover -> Dump on the steered fixture went 357 (P) to 1854 (A). The dry column\n");
    std::printf("  above is the same count on this surface: P %zu, O %zu, A %zu.\n",
                dryEnsembles[0].transitions.recoverDump[2], dryEnsembles[1].transitions.recoverDump[2],
                dryEnsembles[2].transitions.recoverDump[2]);

    // --- 17.7 THE REPRESENTATIVE DRY MEMBER -------------------------------------------------------
    std::printf("\n--- 17.7 DRY REPRESENTATIVE MEMBER: production-stranded, traced under P, O and A ---\n");

    REQUIRE_FALSE(dryStrandedP.empty());

    auto representative = dryStrandedP.front();
    auto established = false;
    for (const auto member : dryStrandedP)
    {
        if (member == std::size_t{4})
        {
            representative = member;
            established = true;
        }
    }

    if (!established)
    {
        for (const auto member : dryStrandedP)
        {
            if (dryEnsembles[0].members[member].rearUtilisation <
                dryEnsembles[0].members[representative].rearUtilisation)
            {
                representative = member;
            }
        }
    }

    std::printf("  member %zu, entry %.4f m/s%s. Production rear utilisation %.4f.\n", representative,
                memberEntry(representative, ensembleCount),
                established ? " — the established representative of `[.road-torque-scope]` section 2" : "",
                dryEnsembles[0].members[representative].rearUtilisation);

    auto dryRuns = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        dryRuns.push_back(record(setup.value(), dry.value(), arm.assists, calibration, arm.candidate, 1.0,
                                 memberEntry(representative, ensembleCount), 0.0, true));
    }

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printDryTrace(arms[index].name.c_str(), dryRuns[index], 0.10);
    }

    std::printf("\n  --- first causal divergence from production, on this member ---\n");
    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        const auto any = firstDivergence(dryRuns[0], dryRuns[index], -1);

        std::printf("\n  %s\n", arms[index].name.c_str());

        if (!any.found)
        {
            std::printf("    NO DIVERGENCE at any controller step.\n");
            continue;
        }

        std::printf("    first difference   t = %.5f s, step %zu, channel %s\n", any.time, any.index,
                    channelName(any.channel));
        std::printf("    state before       %s, pastBand %s, guardSlip %.4f, sensed slip %.4f\n",
                    modulatorName(any.before), any.pastBand ? "TRUE" : "false", any.guardSlip, any.slip);
        std::printf("    wheel accel        %.2f m/s^2, excess %.2f m/s^2, losing %s\n", any.acceleration, any.excess,
                    any.acceleration < assists.antilock.lockDeceleration ? "TRUE" : "false");
        std::printf("    production takes   %s -> %s, %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.productionAfter), any.productionPressure / bar);
        std::printf("    this arm takes     %s -> %s, %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.armAfter), any.armPressure / bar);
        std::printf("    is it the low-mu mechanism? Dump exit on !losing while pastBand: %s\n",
                    (any.before == ModulatorPhase::Dump && any.pastBand &&
                     any.acceleration >= assists.antilock.lockDeceleration &&
                     any.productionAfter == ModulatorPhase::Dump && any.armAfter == ModulatorPhase::Recover)
                        ? "YES — the same transition and the same terms as the steered fixture"
                        : "NO — a different transition on this surface");

        std::printf("    per channel:");
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            const auto each = firstDivergence(dryRuns[0], dryRuns[index], static_cast<long>(channel));
            std::printf("  %s ", channelName(channel));
            if (each.found)
            {
                std::printf("%.4f s", each.time);
            }
            else
            {
                std::printf("never");
            }
        }
        std::printf("\n");

        const auto after = std::min(any.time + 1.0, 30.0);
        std::printf("    in the second after it, production -> this arm:\n");
        std::printf("      rear true slip %.4f -> %.4f    rear bar %.3f -> %.3f    rear Fz %.1f -> %.1f N\n",
                    windowAxleMean(dryRuns[0], 2, 3, &WheelTick::slipRatio, any.time, after, true),
                    windowAxleMean(dryRuns[index], 2, 3, &WheelTick::slipRatio, any.time, after, true),
                    windowAxleMean(dryRuns[0], 2, 3, &WheelTick::pressure, any.time, after, false) / bar,
                    windowAxleMean(dryRuns[index], 2, 3, &WheelTick::pressure, any.time, after, false) / bar,
                    2.0 * windowAxleMean(dryRuns[0], 2, 3, &WheelTick::load, any.time, after, false),
                    2.0 * windowAxleMean(dryRuns[index], 2, 3, &WheelTick::load, any.time, after, false));
        std::printf("    whole run: rear utilisation %.4f -> %.4f, stop %.3f -> %.3f m\n", axleUse(dryRuns[0]).rear,
                    axleUse(dryRuns[index]).rear, dryRuns[0].distance, dryRuns[index].distance);
    }

    // =============================================================================================
    // EXPERIMENT 2 — SPLIT MU
    // =============================================================================================
    std::printf("\n\n=== EXPERIMENT 2: SPLIT MU, grip 1.00 at x<0 (LEFT), 0.35 at x>0 (RIGHT) ===\n");
    std::printf("  The fixture is unaltered and `yawMomentDelay` is off on every arm.\n");

    auto splitEnsembles = std::vector<CrossEnsemble>{};
    splitEnsembles.push_back(splitProduction);
    splitEnsembles.push_back(
        crossEnsemble(setup.value(), split.value(), arms[1].assists, calibration, arms[1].candidate, 0.0));
    splitEnsembles.push_back(
        crossEnsemble(setup.value(), split.value(), arms[2].assists, calibration, arms[2].candidate, 0.0));

    for (const auto& ensemble : splitEnsembles)
    {
        for (const auto& member : ensemble.members)
        {
            REQUIRE(member.stopped);
        }
    }

    std::printf("\n--- 17.8 SPLIT-MU STOPPING DISTANCE [m] ---\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(),
                distributionOf(columnOf(splitEnsembles[index].members, &Member::distance, false)));
    }

    std::printf("\n--- 17.9 SPLIT-MU FINAL YAW ---\n");
    std::printf("\n  magnitude [deg]\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto values = columnOf(splitEnsembles[index].members, &Member::finalYaw, true);
        for (auto& value : values)
        {
            value *= degrees;
        }
        wideRow(arms[index].name.c_str(), distributionOf(values));
    }

    std::printf("\n  signed [deg], so a sign flip cannot hide inside a magnitude\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto values = columnOf(splitEnsembles[index].members, &Member::finalYaw, false);
        for (auto& value : values)
        {
            value *= degrees;
        }
        wideRow(arms[index].name.c_str(), distributionOf(values));
    }

    std::printf("\n  sign, range and reversals\n");
    std::printf("\n  %-26s %10s %10s %12s %12s %12s\n", "arm", "positive", "negative", "range [deg]", "sign flips",
                "peak rate");
    std::printf("  %s\n", "--------------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto positive = std::size_t{0};
        auto negative = std::size_t{0};
        auto reversals = std::size_t{0};
        auto signedYaw = std::vector<double>{};
        auto rates = std::vector<double>{};

        for (const auto& member : splitEnsembles[index].members)
        {
            positive += member.finalYaw > 0.0 ? 1 : 0;
            negative += member.finalYaw < 0.0 ? 1 : 0;
            reversals += member.yawReversals;
            signedYaw.push_back(member.finalYaw * degrees);
            rates.push_back(member.peakYawRate);
        }

        const auto spread = distributionOf(signedYaw);

        std::printf("  %-26s %10zu %10zu %12.3f %12zu %12.4f\n", arms[index].name.c_str(), positive, negative,
                    spread.maximum - spread.minimum, reversals, distributionOf(rates).median);
    }

    std::printf("\n  the existing split-mu criterion, unchanged: |final yaw| < 45 deg\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto inside = std::size_t{0};
        for (const auto& member : splitEnsembles[index].members)
        {
            inside += std::abs(member.finalYaw) * degrees < 45.0 ? 1 : 0;
        }
        std::printf("  %-26s %zu/%zu inside\n", arms[index].name.c_str(), inside, ensembleCount);
    }

    // --- 17.10 SPLIT-MU MEMBER MAP ----------------------------------------------------------------
    std::printf("\n--- 17.10 SPLIT-MU MEMBER MAP, all %zu members. The distribution is broad; the median\n",
                ensembleCount);
    std::printf("    alone is not the result. ---\n");
    std::printf("\n  %-3s %9s | %10s %10s %10s | %8s %8s %8s\n", "k", "entry m/s", "P yaw deg", "O yaw deg",
                "A yaw deg", "P stop", "O stop", "A stop");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "----");

    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        const auto& p = splitEnsembles[0].members[member];
        const auto& o = splitEnsembles[1].members[member];
        const auto& a = splitEnsembles[2].members[member];

        std::printf("  %-3zu %9.4f | %10.3f %10.3f %10.3f | %8.3f %8.3f %8.3f\n", member, p.entry, p.finalYaw * degrees,
                    o.finalYaw * degrees, a.finalYaw * degrees, p.distance, o.distance, a.distance);
    }

    {
        auto closerToO = std::size_t{0};
        auto closerToP = std::size_t{0};

        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            const auto toP = std::abs(std::abs(splitEnsembles[2].members[member].finalYaw) -
                                      std::abs(splitEnsembles[0].members[member].finalYaw));
            const auto toO = std::abs(std::abs(splitEnsembles[2].members[member].finalYaw) -
                                      std::abs(splitEnsembles[1].members[member].finalYaw));

            closerToO += toO < toP ? 1 : 0;
            closerToP += toP < toO ? 1 : 0;
        }

        std::printf("\n  members whose leg-A-off |yaw| sits nearer whole-law-off: %zu of %zu; nearer production:"
                    " %zu.\n",
                    closerToO, ensembleCount, closerToP);
    }

    // --- 17.11 SPLIT-MU STATE MACHINE -------------------------------------------------------------
    std::printf("\n--- 17.11 SPLIT-MU STATE-MACHINE DIFFERENTIAL, summed over all %zu members ---\n", ensembleCount);
    printTransitionHeader();
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printTransitionBlock(arms[index].name.c_str(), splitEnsembles[index].transitions, period);
    }

    std::printf("\n  Reapply -> Dump, per channel\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printReapplyDump(arms[index].name.c_str(), splitEnsembles[index].transitions);
    }

    std::printf("\n  PHASE RESIDENCY, share of physics ticks (FL is the HIGH-mu side, FR the LOW-mu side)\n");
    std::printf("\n  %-24s %6s %10s %10s %10s %10s %10s\n", "arm", "wheel", "passive", "hold", "dump", "recover",
                "reapply");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-----");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printResidency(arms[index].name.c_str(), splitEnsembles[index].transitions);
    }

    // --- 17.12 SPLIT-MU REPRESENTATIVE ------------------------------------------------------------
    std::printf("\n--- 17.12 SPLIT-MU REPRESENTATIVE MEMBER: the one at production's median |yaw| ---\n");

    auto splitYaw = std::vector<double>{};
    for (const auto& member : splitEnsembles[0].members)
    {
        splitYaw.push_back(std::abs(member.finalYaw));
    }

    const auto splitRepresentative = medianIndexOf(splitYaw);

    std::printf("  member %zu, entry %.4f m/s, production final yaw %.3f deg.\n", splitRepresentative,
                memberEntry(splitRepresentative, ensembleCount),
                splitEnsembles[0].members[splitRepresentative].finalYaw * degrees);

    auto splitRuns = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        splitRuns.push_back(record(setup.value(), split.value(), arm.assists, calibration, arm.candidate, 1.0,
                                   memberEntry(splitRepresentative, ensembleCount), 0.0, true));
    }

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printSplitTrace(arms[index].name.c_str(), splitRuns[index], 0.10);
    }

    std::printf("\n  --- first causal divergence, and what it did to the left/right imbalance ---\n");
    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        const auto any = firstDivergence(splitRuns[0], splitRuns[index], -1);

        std::printf("\n  %s\n", arms[index].name.c_str());

        if (!any.found)
        {
            std::printf("    NO DIVERGENCE at any controller step.\n");
            continue;
        }

        std::printf("    first difference   t = %.5f s, channel %s\n", any.time, channelName(any.channel));
        std::printf("    state before       %s, pastBand %s, guardSlip %.4f, sensed slip %.4f\n",
                    modulatorName(any.before), any.pastBand ? "TRUE" : "false", any.guardSlip, any.slip);
        std::printf("    wheel accel        %.2f m/s^2, excess %.2f m/s^2, losing %s\n", any.acceleration, any.excess,
                    any.acceleration < assists.antilock.lockDeceleration ? "TRUE" : "false");
        std::printf("    production takes   %s -> %s, %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.productionAfter), any.productionPressure / bar);
        std::printf("    this arm takes     %s -> %s, %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.armAfter), any.armPressure / bar);
        std::printf("    is it the Dump exit on !losing while pastBand? %s\n",
                    (any.before == ModulatorPhase::Dump && any.pastBand &&
                     any.acceleration >= assists.antilock.lockDeceleration &&
                     any.productionAfter == ModulatorPhase::Dump && any.armAfter == ModulatorPhase::Recover)
                        ? "YES"
                        : "NO");

        const auto after = std::min(any.time + 1.0, 30.0);
        std::printf("    mean LEFT-RIGHT longitudinal force in the second after it: production %+.1f N,"
                    " this arm %+.1f N\n",
                    meanImbalance(splitRuns[0], any.time, after), meanImbalance(splitRuns[index], any.time, after));
        std::printf("    mean over the WHOLE stop:                                production %+.1f N,"
                    " this arm %+.1f N\n",
                    meanImbalance(splitRuns[0], 0.0, 1.0e9), meanImbalance(splitRuns[index], 0.0, 1.0e9));
        std::printf("    yaw at the divergence  %+.3f deg; at +1 s  production %+.3f deg, this arm %+.3f deg\n",
                    yawAt(splitRuns[0], any.time) * degrees, yawAt(splitRuns[0], after) * degrees,
                    yawAt(splitRuns[index], after) * degrees);
        std::printf("    final yaw              production %+.3f deg, this arm %+.3f deg;"
                    " stop %.3f against %.3f m\n",
                    splitRuns[0].finalYaw * degrees, splitRuns[index].finalYaw * degrees, splitRuns[0].distance,
                    splitRuns[index].distance);
    }

    // --- 17.13 THE CROSS-SURFACE MATRIX -----------------------------------------------------------
    std::printf("\n\n--- 17.13 CROSS-SURFACE LEG-A MATRIX ---\n");
    std::printf("  STEERED LOW MU is EXISTING (section 16, n=15, 0.35 lock). DRY and SPLIT MU are NEW\n");
    std::printf("  THIS TASK (n=%zu each, no steering).\n", ensembleCount);

    std::printf("\n  %-6s | %27s | %26s | %26s\n", "arm", "STEERED LOW MU (EXISTING)", "DRY (NEW)", "SPLIT MU (NEW)");
    std::printf("  %-6s | %6s %5s %6s %6s | %8s %8s %7s | %7s %8s %8s\n", "", "ratio", ">3x", "slipF", "|Fy|",
                "stranded", "stop med", "rear u", "yaw med", "yaw wid", "stop med");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "------------------------");

    {
        const auto existing = std::array<std::array<double, 4>, 3>{std::array<double, 4>{5.242, 14.0, 0.175, 498.5},
                                                                   std::array<double, 4>{1.124, 0.0, 0.358, 391.9},
                                                                   std::array<double, 4>{2.189, 3.0, 0.408, 320.9}};

        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            const auto stop = distributionOf(columnOf(dryEnsembles[index].members, &Member::distance, false));
            const auto rear = distributionOf(columnOf(dryEnsembles[index].members, &Member::rearUtilisation, false));

            auto yaw = columnOf(splitEnsembles[index].members, &Member::finalYaw, true);
            for (auto& value : yaw)
            {
                value *= degrees;
            }
            const auto yawSpread = distributionOf(yaw);
            const auto splitStop = distributionOf(columnOf(splitEnsembles[index].members, &Member::distance, false));

            std::printf("  %-6s | %6.3f %2.0f/15 %6.3f %6.1f | %5zu/%-2zu %8.3f %7.4f | %7.3f %8.3f %8.3f\n",
                        arms[index].tag.c_str(), existing[index][0], existing[index][1], existing[index][2],
                        existing[index][3], strandedCount(dryEnsembles[index].members), ensembleCount, stop.median,
                        rear.median, yawSpread.median, yawSpread.p90 - yawSpread.p10, splitStop.median);
        }
    }

    std::printf("\n  yaw wid = P90 - P10 of |final yaw|, the distribution width the task asks for.\n");
    std::printf("\n  Production is unchanged by this file. `Candidate::replica` is false in arms P and O,\n");
    std::printf("  and section 17.1 proves the copy with every leg on reproduces production to the bit on\n");
    std::printf("  both fixtures. No threshold, gain or tyre parameter is touched anywhere in it.\n");
}

// =============================================================================================
// 18. WHICH OF B, C AND D CARRIES THE REMAINING SPLIT-MU YAW COST, WITH LEG A HELD ON
//
// `./EngineTests "[.leg-bcd]"`. Added 2026-09-07, after section 17 established that leg A carries
// about 85% of the split-mu STOPPING-DISTANCE interval between production and the whole law off
// but only about 29% of the median yaw interval and 37% of the mean. Distance and yaw are
// therefore separable, and the yaw cost is mostly somewhere else.
//
// **Leg A is ON in every arm of this section, production included.** Section 17 measured what
// happens when it is not: with A off, leg B's own operating population explodes — the ensemble's
// Recover -> Dump count went 137 to 1878 dry and 248 to 2856 split — while leg D still holds
// Recover -> Reapply near production. An A-off arm is therefore a compound altered state machine
// and NOT a clean "production minus A", so it is not the baseline for anything here and does not
// appear as an arm.
//
// FIVE ARMS AND NO OTHERS:
//
//   P  production            A ON   B ON   C ON   D ON
//   O  whole law OFF         `slipAwareRecovery` false; `pastBand` is constant false
//   B  leg B off only        A ON   B OFF  C ON   D ON
//   C  leg C off only        A ON   B ON   C OFF  D ON
//   D  leg D off only        A ON   B ON   C ON   D OFF
//
// These are ONE-AT-A-TIME ablations in the production operating context. They measure marginal
// behaviour and they do NOT prove additivity: section 18.12 reports, rather than assumes away,
// every place where switching one leg changes the population another leg sees.
//
// **Nothing is designed, tuned or calibrated.** No threshold, gradient, estimator constant, pulse
// model, brake bias or tyre parameter is touched. No road-torque evidence appears. No new
// observable is introduced: the actuator lower bound is this file's existing definition — a
// controller step whose resulting pressure is <= 0 — which `Transitions::emptySteps` has counted
// since section 16.
//
// The fixture is the existing split-mu ensemble unchanged: `gripPlate(1.00, 0.35)`, so LEFT is the
// HIGH-mu side and FL is the high-mu front channel, `ensembleCount` deterministic members on the
// same +/-0.7% entry-speed band, full pedal, no steering, the same settle, the same stopping
// condition, `yawMomentDelay` off, `OSR_ASSISTS` irrelevant because the setup is built in code.
// =============================================================================================

namespace
{

// --- the actuator lower bound, and it is NOT redefined here -----------------------------------
//
// A controller step whose resulting pressure is <= 0. That is exactly what `transitionsOf` counts
// into `Transitions::emptySteps`, and the episode statistics below are runs of consecutive such
// steps on one channel. Nothing here introduces an epsilon.
[[nodiscard]] std::vector<const Step*> channelSteps(const Run& run, const std::size_t channel)
{
    auto ordered = std::vector<const Step*>{};

    for (const auto& step : run.steps)
    {
        if (step.channel == channel)
        {
            ordered.push_back(&step);
        }
    }

    return ordered;
}

struct Episodes
{
    std::size_t steps = 0;
    std::size_t count = 0;
    std::size_t longest = 0;
    std::vector<double> durations;

    void add(const Episodes& more)
    {
        steps += more.steps;
        count += more.count;
        longest = std::max(longest, more.longest);
        durations.insert(durations.end(), more.durations.begin(), more.durations.end());
    }
};

[[nodiscard]] Episodes lowerBoundEpisodes(const Run& run, const std::size_t channel, const double period)
{
    auto episodes = Episodes{};
    auto current = std::size_t{0};

    const auto close = [&]
    {
        if (current > 0)
        {
            episodes.count++;
            episodes.longest = std::max(episodes.longest, current);
            episodes.durations.push_back(static_cast<double>(current) * period);
            current = 0;
        }
    };

    for (const auto* step : channelSteps(run, channel))
    {
        if (step->pressureAfter <= 0.0)
        {
            episodes.steps++;
            current++;
        }
        else
        {
            close();
        }
    }

    close();

    return episodes;
}

// --- the leg predicates, written exactly as the control law writes them ------------------------
//
// Every term comes off the recorded step and the setup's own calibrated constants. `losing` and
// `surging` are the law's expressions; `surged` is the law's own updated value, because the
// Recover branch writes `state.surged = state.surged || surging` before it tests it.
// **TWO readings of `pastBand`, and the difference is what makes the arms comparable.**
//
//  - `pastBandLaw` is what THIS arm's control law evaluated. It is constant false in the whole-law
//    -off arm, because `slipAwareRecovery` is false there, so a census built on it reads 0 for
//    every leg in O — which is true and useless, since it cannot distinguish "the state never
//    arose" from "the law was not looking".
//  - `pastBandState` is the geometry alone: the estimated slip the guards read is past the
//    calibrated band. `guardSlip` is recorded on every step of every arm and is 0 when the
//    reference is invalid, so this term is what `slipAwareRecovery && referenceValid && guardSlip >
//    slipEnter` would have been on an arm that has the law switched off.
//
// The event counts below are built on the STATE reading, so a leg's population is comparable
// across all five arms; what each arm actually DID with that population is the transition columns
// beside them. In production the two readings coincide by construction, so P's counts are the
// production law's own.
struct LegTerms
{
    bool losing = false;
    bool surging = false;
    bool surged = false;
    bool pastBandLaw = false;
    bool pastBandState = false;
    bool redeparting = false; // the first Recover branch
    bool bState = false;      // the leg-B stuck branch's own condition, with the branch reached
    bool dState = false;      // leg D alone standing between Recover and Reapply
    bool fast = false;
    double proximity = 1.0;
    bool c1State = false;
    bool c1Closed = false;
    bool c2State = false;
};

[[nodiscard]] LegTerms termsOf(const AntilockSetup& setup, const Step& step)
{
    auto terms = LegTerms{};

    terms.losing = step.acceleration < setup.lockDeceleration;
    terms.surging = step.excess > setup.recoverySurge;
    terms.surged = step.surgedBefore || terms.surging;
    terms.pastBandLaw = step.pastBandUnqualified;
    terms.pastBandState = step.guardSlip > setup.slipEnter;

    if (step.before == ModulatorPhase::Recover)
    {
        terms.redeparting = step.slip > setup.slipEnter && terms.losing;
        terms.bState = !terms.redeparting && terms.pastBandState && step.excess < setup.recoveryAcceleration;
        terms.dState = !terms.redeparting && !terms.bState && !(terms.surged && terms.surging) && terms.pastBandState;
    }

    if (step.before == ModulatorPhase::Reapply)
    {
        terms.fast = step.pressureBefore < step.departurePressureBefore && !terms.pastBandState;
        terms.proximity =
            std::clamp((setup.slipEnter - step.guardSlip) / std::max(setup.slipEnter - setup.slipExit, 1e-9), 0.0, 1.0);
        terms.c1State = !terms.fast && terms.proximity < 1.0;
        terms.c1Closed = !terms.fast && terms.proximity <= 0.0;
        terms.c2State = !terms.losing && terms.pastBandState && step.excess < -setup.recoveryAcceleration &&
                        step.guardSlip > 2.0 * setup.slipEnter - setup.slipExit;
    }

    return terms;
}

struct LegEvents
{
    // Leg B: the qualifying state, its episodes, and what the arm actually did with it.
    std::array<std::size_t, brakeChannelCount> bSteps{};
    std::array<std::size_t, brakeChannelCount> bEpisodes{};
    std::array<std::size_t, brakeChannelCount> bTookDump{};
    std::array<std::size_t, brakeChannelCount> bStayedRecover{};

    // Leg D: the blocked-exit state, its episodes and its longest run.
    std::array<std::size_t, brakeChannelCount> dSteps{};
    std::array<std::size_t, brakeChannelCount> dEpisodes{};
    std::array<std::size_t, brakeChannelCount> dLongest{};
    std::array<std::size_t, brakeChannelCount> dTookReapply{};

    // Leg C, both halves together, as the previous experiment grouped them.
    std::array<std::size_t, brakeChannelCount> reapplySteps{};
    std::array<std::size_t, brakeChannelCount> c1Tapered{};
    std::array<std::size_t, brakeChannelCount> c1Closed{};
    std::array<std::size_t, brakeChannelCount> c2Steps{};
    std::array<std::size_t, brakeChannelCount> c2TookDump{};

    void add(const LegEvents& more)
    {
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            bSteps[channel] += more.bSteps[channel];
            bEpisodes[channel] += more.bEpisodes[channel];
            bTookDump[channel] += more.bTookDump[channel];
            bStayedRecover[channel] += more.bStayedRecover[channel];
            dSteps[channel] += more.dSteps[channel];
            dEpisodes[channel] += more.dEpisodes[channel];
            dLongest[channel] = std::max(dLongest[channel], more.dLongest[channel]);
            dTookReapply[channel] += more.dTookReapply[channel];
            reapplySteps[channel] += more.reapplySteps[channel];
            c1Tapered[channel] += more.c1Tapered[channel];
            c1Closed[channel] += more.c1Closed[channel];
            c2Steps[channel] += more.c2Steps[channel];
            c2TookDump[channel] += more.c2TookDump[channel];
        }
    }
};

[[nodiscard]] LegEvents legEventsOf(const AntilockSetup& setup, const Run& run)
{
    auto events = LegEvents{};

    for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
    {
        auto inB = false;
        auto inD = std::size_t{0};

        for (const auto* step : channelSteps(run, channel))
        {
            const auto terms = termsOf(setup, *step);

            if (terms.bState)
            {
                events.bSteps[channel]++;
                events.bEpisodes[channel] += inB ? 0 : 1;
                inB = true;
                events.bTookDump[channel] += step->after == ModulatorPhase::Dump ? 1 : 0;
                events.bStayedRecover[channel] += step->after == ModulatorPhase::Recover ? 1 : 0;
            }
            else
            {
                inB = false;
            }

            if (terms.dState)
            {
                events.dSteps[channel]++;
                events.dEpisodes[channel] += inD == 0 ? 1 : 0;
                inD++;
                events.dLongest[channel] = std::max(events.dLongest[channel], inD);
                events.dTookReapply[channel] += step->after == ModulatorPhase::Reapply ? 1 : 0;
            }
            else
            {
                inD = 0;
            }

            if (step->before == ModulatorPhase::Reapply)
            {
                events.reapplySteps[channel]++;
                events.c1Tapered[channel] += terms.c1State ? 1 : 0;
                events.c1Closed[channel] += terms.c1Closed ? 1 : 0;

                if (terms.c2State)
                {
                    events.c2Steps[channel]++;
                    events.c2TookDump[channel] += step->after == ModulatorPhase::Dump ? 1 : 0;
                }
            }
        }
    }

    return events;
}

// --- what happens after a would-be blocked exit, or after a suppressed stuck dump ---------------
//
// One census, taken forward from a named event on one channel, inside the SAME arm. There is no
// cross-arm alignment here on purpose: once two arms have diverged their step indices no longer
// describe the same physical instant, and the question "does the exit survive" is about the arm
// that took it.
struct Followon
{
    std::size_t events = 0;
    std::array<std::array<std::size_t, 5>, 4> phaseAt{}; // [offset][phase]
    std::array<double, 4> pressureDelta{};               // mean bar change from the event
    std::array<double, 4> guardSlipAt{};
    std::size_t backToDump = 0; // reached Dump or Hold within the window
    std::vector<double> delay;  // seconds to the first Reapply, where one comes
};

constexpr auto followOffsets = std::array<double, 4>{0.010, 0.050, 0.100, 0.250};

[[nodiscard]] Followon followOn(const AntilockSetup& setup, const Run& run, const std::size_t channel,
                                const double period, const bool wantD)
{
    auto census = Followon{};
    const auto steps = channelSteps(run, channel);

    auto stride = std::array<std::size_t, 4>{};
    for (auto slot = std::size_t{0}; slot < followOffsets.size(); slot++)
    {
        stride[slot] = static_cast<std::size_t>(followOffsets[slot] / period + 0.5);
    }

    for (auto index = std::size_t{0}; index < steps.size(); index++)
    {
        const auto terms = termsOf(setup, *steps[index]);

        // The D population is a Recover step that leg D would have held and that this arm let go
        // to Reapply. The B population is a Recover step qualifying for the stuck dump that this
        // arm did NOT dump.
        const auto wanted = wantD ? (terms.dState && steps[index]->after == ModulatorPhase::Reapply)
                                  : (terms.bState && steps[index]->after != ModulatorPhase::Dump);

        if (!wanted)
        {
            continue;
        }

        census.events++;

        const auto base = steps[index]->pressureAfter;
        auto fell = false;
        auto arrived = -1.0;

        for (auto ahead = std::size_t{1}; ahead <= stride.back() && index + ahead < steps.size(); ahead++)
        {
            const auto& later = *steps[index + ahead];

            if (arrived < 0.0 && later.after == ModulatorPhase::Reapply)
            {
                arrived = static_cast<double>(ahead) * period;
            }

            fell = fell || later.after == ModulatorPhase::Dump || later.after == ModulatorPhase::Hold;
        }

        census.backToDump += fell ? 1 : 0;

        if (arrived >= 0.0)
        {
            census.delay.push_back(arrived);
        }

        for (auto slot = std::size_t{0}; slot < stride.size(); slot++)
        {
            const auto at = index + stride[slot];

            if (at >= steps.size())
            {
                continue;
            }

            census.phaseAt[slot][phaseIndex(steps[at]->after)]++;
            census.pressureDelta[slot] += (steps[at]->pressureAfter - base) / bar;
            census.guardSlipAt[slot] += steps[at]->guardSlip;
        }
    }

    for (auto slot = std::size_t{0}; slot < census.pressureDelta.size(); slot++)
    {
        const auto n = static_cast<double>(std::max(census.events, std::size_t{1}));
        census.pressureDelta[slot] /= n;
        census.guardSlipAt[slot] /= n;
    }

    return census;
}

// --- correlation, both kinds, on n = 29 --------------------------------------------------------
[[nodiscard]] double pearsonOf(const std::vector<double>& left, const std::vector<double>& right)
{
    if (left.size() != right.size() || left.size() < 2)
    {
        return 0.0;
    }

    const auto n = static_cast<double>(left.size());
    auto meanLeft = 0.0;
    auto meanRight = 0.0;

    for (auto index = std::size_t{0}; index < left.size(); index++)
    {
        meanLeft += left[index];
        meanRight += right[index];
    }
    meanLeft /= n;
    meanRight /= n;

    auto covariance = 0.0;
    auto varianceLeft = 0.0;
    auto varianceRight = 0.0;

    for (auto index = std::size_t{0}; index < left.size(); index++)
    {
        const auto a = left[index] - meanLeft;
        const auto b = right[index] - meanRight;
        covariance += a * b;
        varianceLeft += a * a;
        varianceRight += b * b;
    }

    const auto denominator = std::sqrt(varianceLeft * varianceRight);

    return denominator > 0.0 ? covariance / denominator : 0.0;
}

// Fractional ranks, ties averaged, so Spearman is Pearson on the ranks.
[[nodiscard]] std::vector<double> ranksOf(const std::vector<double>& values)
{
    auto order = std::vector<std::size_t>(values.size());
    for (auto index = std::size_t{0}; index < order.size(); index++)
    {
        order[index] = index;
    }

    std::sort(order.begin(), order.end(),
              [&](const std::size_t a, const std::size_t b) { return values[a] < values[b]; });

    auto ranks = std::vector<double>(values.size(), 0.0);

    for (auto index = std::size_t{0}; index < order.size();)
    {
        auto last = index;
        while (last + 1 < order.size() && values[order[last + 1]] == values[order[index]])
        {
            last++;
        }

        const auto shared = 0.5 * (static_cast<double>(index) + static_cast<double>(last)) + 1.0;
        for (auto tie = index; tie <= last; tie++)
        {
            ranks[order[tie]] = shared;
        }

        index = last + 1;
    }

    return ranks;
}

[[nodiscard]] double spearmanOf(const std::vector<double>& left, const std::vector<double>& right)
{
    return pearsonOf(ranksOf(left), ranksOf(right));
}

// --- distribution shape, from the deterministic member map and not from a mixture model ---------
//
// The members are sorted, the gaps between neighbours are ranked, and the rule is stated rather
// than fitted to the arms under test: a gap is a SEPARATION when it is at least `gapMultiple`
// times the sample's OWN median neighbour gap and leaves at least `gapTail` members on each side.
// One separation is BIMODAL, two or more MULTIMODAL / IRREGULAR, none UNIMODAL.
//
// **The two constants have exactly one calibration point, and it is not one of this section's own
// arms.** Section 17 established by eye that the whole-law-off control on this fixture is bimodal:
// 19 members at 0.28-4.17 deg and 10 at 7.51-22.02, with a clear gap between. The rule must
// reproduce that split and only that split in O, and these are the values that do — the first
// pass, written before the numbers existed, used a fraction of the range and missed it, because a
// long single-sided tail makes the range a bad yardstick for a gap. The rule is then applied
// unchanged to all five arms, and the three largest gaps of every arm are printed beside the
// verdict so the classification can be checked rather than believed.
constexpr auto gapMultiple = 8.0;
constexpr auto gapTail = std::size_t{5};

struct Gap
{
    double size = 0.0;
    double multiple = 0.0;
    double edgeLow = 0.0;
    double edgeHigh = 0.0;
    std::size_t below = 0;
    std::size_t above = 0;
    bool separates = false;
};

struct Shape
{
    std::string verdict;
    std::size_t separations = 0;
    double medianGap = 0.0;
    std::vector<Gap> ranked;
};

[[nodiscard]] Shape shapeOf(std::vector<double> values)
{
    auto shape = Shape{};

    if (values.size() < 2 * gapTail)
    {
        shape.verdict = "too few members";
        return shape;
    }

    std::sort(values.begin(), values.end());

    auto sizes = std::vector<double>{};
    for (auto index = std::size_t{1}; index < values.size(); index++)
    {
        sizes.push_back(values[index] - values[index - 1]);
    }

    shape.medianGap = distributionOf(sizes).median;

    // **Every SEGMENT must be able to be a mode, not only every side of one gap.** Two qualifying
    // gaps a member apart would otherwise be counted as two separations while the thing between
    // them is a single member, which is a thin region and not a third mode. Accepting a candidate
    // only when it is at least `gapTail` members past the last accepted one enforces that, and it
    // is the same constant the two sides already use.
    auto lastAccepted = std::size_t{0};

    for (auto index = std::size_t{1}; index < values.size(); index++)
    {
        auto gap = Gap{};
        gap.size = values[index] - values[index - 1];
        gap.multiple = shape.medianGap > 0.0 ? gap.size / shape.medianGap : 0.0;
        gap.edgeLow = values[index - 1];
        gap.edgeHigh = values[index];
        gap.below = index;
        gap.above = values.size() - index;
        gap.separates = gap.multiple >= gapMultiple && gap.below >= gapTail && gap.above >= gapTail &&
                        (lastAccepted == 0 || index - lastAccepted >= gapTail);

        if (gap.separates)
        {
            lastAccepted = index;
            shape.separations++;
        }

        shape.ranked.push_back(gap);
    }

    std::sort(shape.ranked.begin(), shape.ranked.end(),
              [](const Gap& left, const Gap& right) { return left.size > right.size; });

    shape.verdict = shape.separations == 0 ? "UNIMODAL" : shape.separations == 1 ? "BIMODAL" : "MULTIMODAL / IRREGULAR";

    return shape;
}

// --- the five-arm split-mu ensemble ------------------------------------------------------------
struct BcdArm
{
    std::string name;
    std::string tag;
    AssistSetup assists;
    Candidate candidate;
};

struct BcdEnsemble
{
    std::vector<Member> members;
    Transitions transitions;
    LegEvents events;

    // Per member, so the correlation and the member map can be paired.
    std::vector<double> flLowerBound;
    std::vector<double> yawMagnitude;

    Episodes flEpisodes;
    Followon flFollow;
    Followon rearFollow;
};

[[nodiscard]] BcdEnsemble bcdEnsemble(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                                      const RoadTorqueCalibration& calibration, const Candidate& candidate,
                                      const double period, const bool wantD)
{
    auto ensemble = BcdEnsemble{};
    ensemble.members.reserve(ensembleCount);

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);
        const auto run = record(setup, world, assists, calibration, candidate, 1.0, entry, 0.0, true);

        ensemble.members.push_back(memberOf(index, entry, run));
        ensemble.transitions.add(transitionsOf(run));
        ensemble.events.add(legEventsOf(assists.antilock, run));

        const auto episodes = lowerBoundEpisodes(run, 0, period);
        ensemble.flEpisodes.add(episodes);
        ensemble.flLowerBound.push_back(static_cast<double>(episodes.steps) * period);
        ensemble.yawMagnitude.push_back(std::abs(run.finalYaw) * degrees);

        const auto front = followOn(assists.antilock, run, 0, period, wantD);
        const auto rear = followOn(assists.antilock, run, 2, period, wantD);

        ensemble.flFollow.events += front.events;
        ensemble.flFollow.backToDump += front.backToDump;
        ensemble.rearFollow.events += rear.events;
        ensemble.rearFollow.backToDump += rear.backToDump;

        for (auto slot = std::size_t{0}; slot < followOffsets.size(); slot++)
        {
            for (auto phase = std::size_t{0}; phase < 5; phase++)
            {
                ensemble.flFollow.phaseAt[slot][phase] += front.phaseAt[slot][phase];
                ensemble.rearFollow.phaseAt[slot][phase] += rear.phaseAt[slot][phase];
            }

            ensemble.flFollow.pressureDelta[slot] += front.pressureDelta[slot] * static_cast<double>(front.events);
            ensemble.flFollow.guardSlipAt[slot] += front.guardSlipAt[slot] * static_cast<double>(front.events);
            ensemble.rearFollow.pressureDelta[slot] += rear.pressureDelta[slot] * static_cast<double>(rear.events);
            ensemble.rearFollow.guardSlipAt[slot] += rear.guardSlipAt[slot] * static_cast<double>(rear.events);
        }

        ensemble.flFollow.delay.insert(ensemble.flFollow.delay.end(), front.delay.begin(), front.delay.end());
        ensemble.rearFollow.delay.insert(ensemble.rearFollow.delay.end(), rear.delay.begin(), rear.delay.end());
    }

    for (auto slot = std::size_t{0}; slot < followOffsets.size(); slot++)
    {
        ensemble.flFollow.pressureDelta[slot] /=
            static_cast<double>(std::max(ensemble.flFollow.events, std::size_t{1}));
        ensemble.flFollow.guardSlipAt[slot] /= static_cast<double>(std::max(ensemble.flFollow.events, std::size_t{1}));
        ensemble.rearFollow.pressureDelta[slot] /=
            static_cast<double>(std::max(ensemble.rearFollow.events, std::size_t{1}));
        ensemble.rearFollow.guardSlipAt[slot] /=
            static_cast<double>(std::max(ensemble.rearFollow.events, std::size_t{1}));
    }

    return ensemble;
}

// The tick nearest a given time, for the causal chain at fixed offsets after a divergence.
[[nodiscard]] const Tick* tickAt(const Run& run, const double when)
{
    const Tick* best = nullptr;

    for (const auto& sample : run.ticks)
    {
        if (sample.time <= when + 1.0e-9)
        {
            best = &sample;
        }
        else
        {
            break;
        }
    }

    return best;
}

[[nodiscard]] double sideForce(const Tick& sample, const bool left)
{
    return left ? sample.wheels[0].longitudinal + sample.wheels[2].longitudinal
                : sample.wheels[1].longitudinal + sample.wheels[3].longitudinal;
}

// The causal chain the deliverable asks for, at +10, +50, +100, +250 and +500 ms after the first
// controller step that differs from production: pressure, then left/right force, then yaw rate,
// then accumulated yaw.
void printCausalChain(const Run& base, const Run& arm, const Divergence& divergence)
{
    static constexpr auto offsets = std::array<double, 5>{0.010, 0.050, 0.100, 0.250, 0.500};

    std::printf("    %9s | %17s | %17s | %19s | %17s | %17s\n", "offset", "FL bar  P / arm", "FR bar  P / arm",
                "left-right Fx [N]", "yaw rate [rad/s]", "yaw [deg]");
    std::printf("    %s\n", "---------------------------------------------------------------------------------"
                            "---------------------------------------------");

    for (const auto offset : offsets)
    {
        const auto when = divergence.time + offset;
        const auto* p = tickAt(base, when);
        const auto* a = tickAt(arm, when);

        if (p == nullptr || a == nullptr)
        {
            std::printf("    %+8.0f ms | %s\n", offset * 1000.0, "past the end of one of the two runs");
            continue;
        }

        std::printf("    %+8.0f ms | %7.3f %7.3f  | %7.3f %7.3f  | %8.1f %8.1f  | %8.4f %8.4f | %8.3f %8.3f\n",
                    offset * 1000.0, p->wheels[0].pressure / bar, a->wheels[0].pressure / bar,
                    p->wheels[1].pressure / bar, a->wheels[1].pressure / bar,
                    sideForce(*p, true) - sideForce(*p, false), sideForce(*a, true) - sideForce(*a, false), p->yawRate,
                    a->yawRate, p->yaw * degrees, a->yaw * degrees);
    }
}

// The representative trace the deliverable asks for, with the two terms `printSplitTrace` does not
// carry: the boolean law terms per front channel, and lateral acceleration.
void printWideSplitTrace(const AntilockSetup& setup, const char* name, const Run& run, const double every)
{
    const auto left = perTickSteps(run, 0);
    const auto right = perTickSteps(run, 1);
    const auto rear = perTickSteps(run, 2);

    std::printf("\n  --- %s: FL is HIGH mu, FR is LOW mu. Every %.0f ms. ---\n", name, every * 1000.0);
    std::printf("  %6s %6s %6s %6s %6s %6s %6s %4s %4s %8s %8s %7s %7s %8s %8s %8s %7s %7s\n", "t [s]", "wL r/s",
                "wR r/s", "rawL", "guardL", "rawR", "guardR", "phL", "phR", "excessL", "excessR", "barL", "barR",
                "FxL [N]", "FxR [N]", "yawRate", "yaw deg", "ay");

    auto next = 0.0;

    for (auto index = std::size_t{0}; index < run.ticks.size(); index++)
    {
        const auto& sample = run.ticks[index];

        if (sample.time + 1.0e-9 < next)
        {
            continue;
        }
        next = sample.time + every;

        const auto* stepLeft = left[index];
        const auto* stepRight = right[index];

        auto flags = std::array<std::array<char, 4>, 2>{};
        const auto pair = std::array<const Step*, 2>{stepLeft, stepRight};

        for (auto side = std::size_t{0}; side < 2; side++)
        {
            flags[side] = {'.', '.', '.', '\0'};

            if (pair[side] == nullptr)
            {
                continue;
            }

            const auto terms = termsOf(setup, *pair[side]);
            flags[side][0] = terms.losing ? 'L' : '.';
            flags[side][1] = terms.surging ? 'S' : '.';
            flags[side][2] = terms.pastBandLaw ? 'B' : '.';
        }

        std::printf("  %6.3f %6.2f %6.2f %6.3f %6.3f %6.3f %6.3f %c%3s %c%3s %8.1f %8.1f %7.3f %7.3f %8.1f %8.1f"
                    " %8.4f %7.3f %7.2f\n",
                    sample.time, sample.wheels[0].wheelSpeed, sample.wheels[1].wheelSpeed,
                    stepLeft != nullptr ? stepLeft->slip : 0.0, stepLeft != nullptr ? stepLeft->guardSlip : 0.0,
                    stepRight != nullptr ? stepRight->slip : 0.0, stepRight != nullptr ? stepRight->guardSlip : 0.0,
                    phaseLetter(sample.wheels[0].phase), flags[0].data(), phaseLetter(sample.wheels[1].phase),
                    flags[1].data(), stepLeft != nullptr ? stepLeft->excess : 0.0,
                    stepRight != nullptr ? stepRight->excess : 0.0, sample.wheels[0].pressure / bar,
                    sample.wheels[1].pressure / bar, sideForce(sample, true), sideForce(sample, false), sample.yawRate,
                    sample.yaw * degrees, sample.lateralAcceleration);
    }

    auto rearDumps = std::size_t{0};
    for (const auto* step : rear)
    {
        rearDumps += (step != nullptr && step->after == ModulatorPhase::Dump) ? 1 : 0;
    }
    std::printf("  rear channel (one for both wheels): %zu of %zu ticks commanded Dump. Phase letters:"
                " . Passive, H Hold, D Dump, R Recover, A Reapply; flags L losing, S surging, B pastBand.\n",
                rearDumps, run.ticks.size());
}

void printFollowon(const char* what, const Followon& census, const double period)
{
    std::printf("  %-30s events %zu", what, census.events);

    if (census.events == 0)
    {
        std::printf("  — the state never arose in this arm\n");
        return;
    }

    std::printf(", of which %zu (%.1f%%) saw Dump or Hold within %.0f ms\n", census.backToDump,
                100.0 * static_cast<double>(census.backToDump) / static_cast<double>(census.events),
                followOffsets.back() * 1000.0);
    std::printf("    %8s %9s %9s %9s %9s %9s %11s %11s\n", "offset", "passive", "hold", "dump", "recover", "reapply",
                "d bar", "guardSlip");

    for (auto slot = std::size_t{0}; slot < followOffsets.size(); slot++)
    {
        std::printf("    %+6.0f ms", followOffsets[slot] * 1000.0);
        for (auto phase = std::size_t{0}; phase < 5; phase++)
        {
            std::printf(" %9zu", census.phaseAt[slot][phase]);
        }
        std::printf(" %11.3f %11.4f\n", census.pressureDelta[slot], census.guardSlipAt[slot]);
    }

    if (!census.delay.empty())
    {
        const auto spread = distributionOf(census.delay);
        std::printf("    first Reapply within the window: %zu of %zu, median %.4f s, P90 %.4f s"
                    " (%.1f / %.1f controller periods)\n",
                    census.delay.size(), census.events, spread.median, spread.p90, spread.median / period,
                    spread.p90 / period);
    }
    else
    {
        std::printf("    first Reapply within the window: none\n");
    }
}

void printEpisodes(const char* name, const Episodes& episodes, const double period, const std::size_t steps)
{
    const auto spread = distributionOf(episodes.durations);

    std::printf("  %-22s %11.3f %11.3f%% %9zu %11.4f %11.4f %11.4f\n", name,
                static_cast<double>(episodes.steps) * period,
                100.0 * static_cast<double>(episodes.steps) / static_cast<double>(std::max(steps, std::size_t{1})),
                episodes.count, static_cast<double>(episodes.longest) * period, spread.median, spread.p90);
}

} // namespace

TEST_CASE("which of legs B, C and D carries the remaining split-mu yaw cost, with leg A held on", "[.leg-bcd]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto period = assists.controlRate > 0.0 ? 1.0 / assists.controlRate : tick;

    REQUIRE_FALSE(assists.antilock.yawMomentDelay);
    REQUIRE(assists.antilock.recoveryAuthority != RecoveryAuthority::Disabled);

    const auto replicaOfLegs = [](const bool legA, const bool legB, const bool legC, const bool legD)
    {
        auto candidate = Candidate{};
        candidate.replica = true;
        candidate.legA = legA;
        candidate.legB = legB;
        candidate.legC = legC;
        candidate.legD = legD;

        return candidate;
    };

    auto arms = std::vector<BcdArm>{};
    arms.push_back(BcdArm{"P production", "P", assists, Candidate{}});
    arms.push_back(BcdArm{"O whole law OFF", "O", withoutSlipAwareRecovery(assists), Candidate{}});
    arms.push_back(BcdArm{"B leg-B OFF (stuck dump)", "B", assists, replicaOfLegs(true, false, true, true)});
    arms.push_back(BcdArm{"C leg-C OFF (reapply)", "C", assists, replicaOfLegs(true, true, false, true)});
    arms.push_back(BcdArm{"D leg-D OFF (rec->rea gate)", "D", assists, replicaOfLegs(true, true, true, false)});

    std::printf("\n=== SECTION 18: LEGS B, C AND D ON SPLIT MU, WITH LEG A HELD IN PRODUCTION STATE ===\n");
    std::printf("  Five arms and no others. Leg A is ON in P, B, C and D; it is off only inside O, where\n");
    std::printf("  the whole law is off. %zu deterministic members, entry %.4f to %.4f m/s, full pedal, no\n",
                ensembleCount, memberEntry(0, ensembleCount), memberEntry(ensembleCount - 1, ensembleCount));
    std::printf("  steering, grip 1.00 at x<0 (LEFT, and FL is therefore the HIGH-mu front channel) and\n");
    std::printf("  0.35 at x>0 (RIGHT, FR the low-mu one). Controller period %.5f s (%.0f Hz).\n", period,
                assists.controlRate);
    std::printf("\n  CARRIED FORWARD, NOT RE-RUN — section 17's split-mu result, for orientation only:\n");
    std::printf("      arm   stop med    |yaw| med   |yaw| mean   |yaw| P90   |yaw| max\n");
    std::printf("      P       56.602       17.263       17.728       25.436      32.680\n");
    std::printf("      O       55.519        2.092        6.110       18.929      22.024\n");
    std::printf("      A OFF   55.687       12.816       13.411       16.445      18.626\n");

    // --- 18.1 THE REPLICA, NOW WITH FOUR LEGS, IS PROVED BEFORE ANYTHING IS READ ------------------
    std::printf("\n--- 18.1 REPLICA IDENTITY: the probe copy with A, B, C AND D all ON, against production ---\n");
    std::printf("  Sections 16.1 and 17.1 proved the three-leg copy. Leg D is a new switch in the same\n");
    std::printf("  copy, so the identity is re-proved here with all four legs on. Every recorded scalar of\n");
    std::printf("  every member must match, and nothing below this line may be read if it does not.\n");

    const auto production = bcdEnsemble(setup.value(), split.value(), assists, calibration, Candidate{}, period, true);
    const auto mirror = bcdEnsemble(setup.value(), split.value(), assists, calibration,
                                    replicaOfLegs(true, true, true, true), period, true);

    REQUIRE(mirror.members.size() == production.members.size());

    auto faithful = true;
    for (auto index = std::size_t{0}; index < production.members.size(); index++)
    {
        faithful = faithful && sameMember(production.members[index], mirror.members[index]);
    }

    std::printf("  SPLIT MU, %zu members, every recorded scalar: %s\n", ensembleCount,
                faithful ? "BIT-IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(faithful);

    // The state machine has to match too, not only the run scalars: a leg switched off that leaves
    // every scalar alone but changes a transition count would be a drifted copy that this section's
    // whole method reads.
    auto sameMachine = true;
    for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
    {
        sameMachine = sameMachine && production.transitions.dumpIn[channel] == mirror.transitions.dumpIn[channel] &&
                      production.transitions.recoverIn[channel] == mirror.transitions.recoverIn[channel] &&
                      production.transitions.holdIn[channel] == mirror.transitions.holdIn[channel] &&
                      production.transitions.reapplyIn[channel] == mirror.transitions.reapplyIn[channel] &&
                      production.transitions.recoverDump[channel] == mirror.transitions.recoverDump[channel] &&
                      production.transitions.recoverReapply[channel] == mirror.transitions.recoverReapply[channel] &&
                      production.transitions.reapplyDump[channel] == mirror.transitions.reapplyDump[channel] &&
                      production.transitions.emptySteps[channel] == mirror.transitions.emptySteps[channel];
    }

    std::printf("  ...and every state-machine transition count, per channel: %s\n",
                sameMachine ? "IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(sameMachine);

    // --- the five ensembles -----------------------------------------------------------------------
    auto ensembles = std::vector<BcdEnsemble>{};
    ensembles.push_back(production);
    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        ensembles.push_back(bcdEnsemble(setup.value(), split.value(), arms[index].assists, calibration,
                                        arms[index].candidate, period, arms[index].tag == "D"));
    }

    for (const auto& ensemble : ensembles)
    {
        for (const auto& member : ensemble.members)
        {
            REQUIRE(member.stopped);
        }
    }

    // --- 18.2 THE DISTRIBUTIONS --------------------------------------------------------------------
    std::printf("\n--- 18.2 SPLIT-MU |FINAL YAW| [deg] ---\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(), distributionOf(ensembles[index].yawMagnitude));
    }

    std::printf("\n  SIGNED final yaw [deg], so a sign flip cannot hide inside a magnitude\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto values = columnOf(ensembles[index].members, &Member::finalYaw, false);
        for (auto& value : values)
        {
            value *= degrees;
        }
        wideRow(arms[index].name.c_str(), distributionOf(values));
    }

    std::printf("\n  SIGN, REVERSALS AND RATE\n");
    std::printf("\n  %-28s %9s %9s %11s %13s %13s %13s\n", "arm", "positive", "negative", "range deg", "head sign rev",
                "in-run flips", "med peak rate");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-----------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto positive = std::size_t{0};
        auto negative = std::size_t{0};
        auto flips = std::size_t{0};
        auto signedYaw = std::vector<double>{};
        auto rates = std::vector<double>{};
        auto headingReversals = std::size_t{0};
        auto previous = 0;

        for (const auto& member : ensembles[index].members)
        {
            positive += member.finalYaw > 0.0 ? 1 : 0;
            negative += member.finalYaw < 0.0 ? 1 : 0;
            flips += member.yawReversals;
            signedYaw.push_back(member.finalYaw * degrees);
            rates.push_back(member.peakYawRate);

            const auto sign = member.finalYaw > 0.0 ? 1 : (member.finalYaw < 0.0 ? -1 : 0);
            if (sign != 0)
            {
                headingReversals += (previous != 0 && sign != previous) ? 1 : 0;
                previous = sign;
            }
        }

        const auto spread = distributionOf(signedYaw);

        std::printf("  %-28s %9zu %9zu %11.3f %13zu %13zu %13.4f\n", arms[index].name.c_str(), positive, negative,
                    spread.maximum - spread.minimum, headingReversals, flips, distributionOf(rates).median);
    }
    std::printf("\n  \"head sign rev\" counts sign changes of the FINAL heading as the entry speed is swept\n");
    std::printf("  through the %zu members in order — the deterministic bifurcation signature, not a\n", ensembleCount);
    std::printf("  within-run quantity. \"in-run flips\" is the yaw-rate sign flip count, deadband 0.02 rad/s.\n");

    std::printf("\n--- 18.3 DISTRIBUTION SHAPE, from the sorted member map ---\n");
    std::printf("  Rule, stated not fitted to these arms: a gap between neighbouring members is a\n");
    std::printf("  SEPARATION when it is >= %.0fx the sample's OWN median neighbour gap and leaves >= %zu\n",
                gapMultiple, gapTail);
    std::printf("  members each side, and be at least %zu members past the previous separation so that every\n",
                gapTail);
    std::printf("  SEGMENT can be a mode. One separation is BIMODAL, two or more MULTIMODAL / IRREGULAR, none\n");
    std::printf("  UNIMODAL. The two constants have ONE calibration point and it is not an arm of this\n");
    std::printf("  section: section 17 found the whole-law-off control bimodal at 19 members 0.28-4.17 deg\n");
    std::printf("  against 10 at 7.51-22.02, and the rule must reproduce that split in O and only that\n");
    std::printf("  one. The three largest gaps of every arm are printed so the verdict can be checked.\n");
    std::printf("\n  %-28s %-22s %10s %10s\n", "arm", "verdict", "med gap", "separations");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto shape = shapeOf(ensembles[index].yawMagnitude);
        std::printf("  %-28s %-22s %10.4f %10zu\n", arms[index].name.c_str(), shape.verdict.c_str(), shape.medianGap,
                    shape.separations);
    }

    std::printf("\n  the three largest gaps of each arm, and whether the rule calls each a separation\n");
    std::printf("\n  %-28s %10s %9s %16s %8s %8s %12s\n", "arm", "gap deg", "x median", "between", "below", "above",
                "separation?");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "----");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto shape = shapeOf(ensembles[index].yawMagnitude);
        const auto* name = arms[index].name.c_str();

        for (auto rank = std::size_t{0}; rank < std::min(std::size_t{3}, shape.ranked.size()); rank++)
        {
            const auto& gap = shape.ranked[rank];
            auto where = std::array<char, 64>{};
            std::snprintf(where.data(), where.size(), "%.2f / %.2f", gap.edgeLow, gap.edgeHigh);
            std::printf("  %-28s %10.3f %9.1f %16s %8zu %8zu %12s\n", name, gap.size, gap.multiple, where.data(),
                        gap.below, gap.above, gap.separates ? "YES" : ".");
            name = "";
        }
    }

    std::printf("\n  the sorted |yaw| ladders, so the shape is visible and not only classified\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto sorted = ensembles[index].yawMagnitude;
        std::sort(sorted.begin(), sorted.end());

        std::printf("\n  %s\n   ", arms[index].name.c_str());
        for (auto member = std::size_t{0}; member < sorted.size(); member++)
        {
            std::printf(" %7.3f", sorted[member]);
            if ((member + 1) % 10 == 0)
            {
                std::printf("\n   ");
            }
        }
        std::printf("\n");
    }

    std::printf("\n--- 18.4 SPLIT-MU STOPPING DISTANCE [m] — context for the yaw result, not a target ---\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(), distributionOf(columnOf(ensembles[index].members, &Member::distance, false)));
    }

    std::printf("\n  the existing split-mu criterion, unchanged: |final yaw| < 45 deg\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto inside = std::size_t{0};
        for (const auto value : ensembles[index].yawMagnitude)
        {
            inside += value < 45.0 ? 1 : 0;
        }
        std::printf("  %-28s %zu/%zu inside\n", arms[index].name.c_str(), inside, ensembleCount);
    }

    // --- 18.5 THE MEMBER MAP ------------------------------------------------------------------------
    std::printf("\n--- 18.5 MEMBER-BY-MEMBER MAP, all %zu deterministic members ---\n", ensembleCount);
    std::printf("\n  %-3s %9s | %8s %8s %8s %8s %8s | %8s %8s %8s %8s %8s\n", "k", "entry m/s", "P yaw", "O yaw",
                "B yaw", "C yaw", "D yaw", "P stop", "O stop", "B stop", "C stop", "D stop");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------------------------");

    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        std::printf("  %-3zu %9.4f |", member, ensembles[0].members[member].entry);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            std::printf(" %8.3f", ensembles[index].members[member].finalYaw * degrees);
        }
        std::printf(" |");
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            std::printf(" %8.3f", ensembles[index].members[member].distance);
        }
        std::printf("\n");
    }

    std::printf("\n  Signed yaw above. The counts below are on |yaw|, and the whole-law-off control is\n");
    std::printf("  bimodal, so \"nearer O\" is a statement about one member and not about a cluster.\n");
    std::printf("\n  %-28s %14s %14s %14s %14s\n", "ablation", "nearer P", "nearer O", "|yaw| < P", "|yaw| > P");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");

    for (auto index = std::size_t{2}; index < arms.size(); index++)
    {
        auto nearerP = std::size_t{0};
        auto nearerO = std::size_t{0};
        auto lower = std::size_t{0};
        auto higher = std::size_t{0};

        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            const auto value = ensembles[index].yawMagnitude[member];
            const auto toP = std::abs(value - ensembles[0].yawMagnitude[member]);
            const auto toO = std::abs(value - ensembles[1].yawMagnitude[member]);

            nearerP += toP < toO ? 1 : 0;
            nearerO += toO < toP ? 1 : 0;
            lower += value < ensembles[0].yawMagnitude[member] ? 1 : 0;
            higher += value > ensembles[0].yawMagnitude[member] ? 1 : 0;
        }

        std::printf("  %-28s %14zu %14zu %14zu %14zu\n", arms[index].name.c_str(), nearerP, nearerO, lower, higher);
    }

    // --- 18.6 THE HIGH-MU CHANNEL'S EMPTY CALIPER ---------------------------------------------------
    std::printf("\n--- 18.6 HIGH-MU FL AT THE ACTUATOR LOWER BOUND ---\n");
    std::printf("  Definition unchanged and not re-derived: a controller step whose resulting pressure is\n");
    std::printf("  <= 0. Summed over all %zu members. Section 17's figures for the same quantity were\n",
                ensembleCount);
    std::printf("  P 15.459 s, A OFF 13.006 s, O 11.368 s.\n");
    std::printf("\n  %-22s %11s %12s %9s %11s %11s %11s\n", "arm", "total [s]", "fraction", "episodes", "longest [s]",
                "median [s]", "P90 [s]");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "--");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printEpisodes(arms[index].name.c_str(), ensembles[index].flEpisodes, period,
                      ensembles[index].transitions.steps[0]);
    }

    std::printf("\n  the same quantity for the LOW-mu FR channel and for the rear, as context\n");
    std::printf("\n  %-22s %13s %13s %13s\n", "arm", "FL [s]", "FR [s]", "rear [s]");
    std::printf("  %s\n", "--------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("  %-22s", arms[index].name.c_str());
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf(" %13.3f", static_cast<double>(ensembles[index].transitions.emptySteps[channel]) * period);
        }
        std::printf("\n");
    }

    std::printf("\n--- 18.7 DOES FL LOWER-BOUND EXPOSURE TRACK YAW? ---\n");
    std::printf("  A DIAGNOSTIC CORRELATE and nothing else. It is not a controller threshold, it is not a\n");
    std::printf("  calibration target, and no arm is scored on it.\n");
    std::printf("\n  %-28s %12s %12s %14s %14s\n", "arm", "Pearson r", "Spearman", "FL mean [s]", "|yaw| mean");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("  %-28s %12.4f %12.4f %14.4f %14.3f\n", arms[index].name.c_str(),
                    pearsonOf(ensembles[index].flLowerBound, ensembles[index].yawMagnitude),
                    spearmanOf(ensembles[index].flLowerBound, ensembles[index].yawMagnitude),
                    distributionOf(ensembles[index].flLowerBound).mean,
                    distributionOf(ensembles[index].yawMagnitude).mean);
    }

    std::printf("\n  the paired member table the correlation is computed on\n");
    std::printf("\n  %-3s | %17s | %17s | %17s | %17s | %17s\n", "k", "P  FL s / yaw", "O  FL s / yaw", "B  FL s / yaw",
                "C  FL s / yaw", "D  FL s / yaw");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "--------------------------");
    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        std::printf("  %-3zu |", member);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            std::printf(" %8.4f %8.3f |", ensembles[index].flLowerBound[member], ensembles[index].yawMagnitude[member]);
        }
        std::printf("\n");
    }

    // --- 18.8 THE STATE MACHINE ---------------------------------------------------------------------
    std::printf("\n--- 18.8 STATE-MACHINE DIFFERENTIAL, summed over all %zu members ---\n", ensembleCount);
    printTransitionHeader();
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printTransitionBlock(arms[index].name.c_str(), ensembles[index].transitions, period);
    }

    std::printf("\n  Reapply -> Dump, per channel\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printReapplyDump(arms[index].name.c_str(), ensembles[index].transitions);
    }

    std::printf("\n  PHASE RESIDENCY, share of physics ticks (FL is HIGH mu, FR is LOW mu, RL the rear pair)\n");
    std::printf("\n  %-24s %6s %10s %10s %10s %10s %10s\n", "arm", "wheel", "passive", "hold", "dump", "recover",
                "reapply");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-----");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printResidency(arms[index].name.c_str(), ensembles[index].transitions);
    }

    // --- 18.9 THE PER-LEG EVENT COUNTS --------------------------------------------------------------
    std::printf("\n--- 18.9 LEG B: the exact event, `phase == Recover && pastBand && excess < recoveryAccel`"
                " ---\n");
    std::printf("  A STATE census on each arm's own steps, with `pastBand` read geometrically so O is\n");
    std::printf("  comparable (see `LegTerms`). P's column IS the production law's own event count, because\n");
    std::printf("  in P the two readings coincide. `qual steps` is time in the state; an episode is a\n");
    std::printf("  maximal run of consecutive qualifying steps on one channel. The last two columns say\n");
    std::printf("  what the arm DID with the state: in P the branch fires on every step of it, in B it\n");
    std::printf("  cannot fire at all, and the population is then free to grow.\n");
    std::printf("\n  %-28s %5s %12s %10s %11s %12s %12s\n", "arm", "chan", "qual steps", "time [s]", "episodes",
                "-> Dump", "stayed Rec");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "---");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto* name = arms[index].name.c_str();
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-28s %5s %12zu %10.3f %11zu %12zu %12zu\n", name, channelName(channel),
                        ensembles[index].events.bSteps[channel],
                        static_cast<double>(ensembles[index].events.bSteps[channel]) * period,
                        ensembles[index].events.bEpisodes[channel], ensembles[index].events.bTookDump[channel],
                        ensembles[index].events.bStayedRecover[channel]);
            name = "";
        }
    }

    std::printf("\n  UNDER B OFF: what became of the suppressed events. Same arm, forward from each one.\n");
    printFollowon("B arm, high-mu FL", ensembles[2].flFollow, period);
    printFollowon("B arm, rear channel", ensembles[2].rearFollow, period);

    std::printf("\n--- 18.10 LEG C: the two halves, counted together as the previous experiment grouped"
                " them ---\n");
    std::printf("  Both halves are STATE counts, on the geometric `pastBand` (see `LegTerms`), so the arms\n");
    std::printf("  compare. C1 is the proximity taper's state: a Reapply step NOT in the fast stage whose\n");
    std::printf("  taper would be below 1.0, and `closed` is the subset where it would be exactly zero (a de\n");
    std::printf("  facto Hold). C2 is the Reapply -> Dump stuck protection's own condition. In P and in the\n");
    std::printf("  arms that keep C, `C2 -> Dump` equals `C2 qual` because the branch fires whenever it\n");
    std::printf("  qualifies; in C it is 0 by construction and the state count is what grew.\n");
    std::printf("\n  %-28s %5s %12s %11s %11s %11s %11s\n", "arm", "chan", "Reapply st", "C1 tapered", "C1 closed",
                "C2 qual", "C2 -> Dump");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "--");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto* name = arms[index].name.c_str();
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-28s %5s %12zu %11zu %11zu %11zu %11zu\n", name, channelName(channel),
                        ensembles[index].events.reapplySteps[channel], ensembles[index].events.c1Tapered[channel],
                        ensembles[index].events.c1Closed[channel], ensembles[index].events.c2Steps[channel],
                        ensembles[index].events.c2TookDump[channel]);
            name = "";
        }
    }

    std::printf("\n--- 18.11 LEG D: the blocked exit, `Recover && !(surged && surging) && pastBand` with the\n");
    std::printf("    two branches above it not taken — the state where D ALONE stands between Recover and\n");
    std::printf("    Reapply. A state census on the geometric `pastBand`, as 18.9 and 18.10 are. ---\n");
    std::printf("\n  %-28s %5s %12s %10s %11s %12s %12s\n", "arm", "chan", "blocked st", "time [s]", "episodes",
                "longest [s]", "-> Reapply");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "---");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto* name = arms[index].name.c_str();
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-28s %5s %12zu %10.3f %11zu %12.4f %12zu\n", name, channelName(channel),
                        ensembles[index].events.dSteps[channel],
                        static_cast<double>(ensembles[index].events.dSteps[channel]) * period,
                        ensembles[index].events.dEpisodes[channel],
                        static_cast<double>(ensembles[index].events.dLongest[channel]) * period,
                        ensembles[index].events.dTookReapply[channel]);
            name = "";
        }
    }

    std::printf("\n  UNDER D OFF: what happened immediately after each would-be blocked exit.\n");
    printFollowon("D arm, high-mu FL", ensembles[4].flFollow, period);
    printFollowon("D arm, rear channel", ensembles[4].rearFollow, period);

    // --- 18.12 THE REPRESENTATIVE MEMBER --------------------------------------------------------------
    std::printf("\n--- 18.12 REPRESENTATIVE MEMBER, traced through all five arms ---\n");

    const auto representative = std::size_t{23};
    const auto medianMember = medianIndexOf(ensembles[0].yawMagnitude);

    std::printf("  The established split-mu representative is member %zu, entry %.4f m/s. Production final\n",
                representative, memberEntry(representative, ensembleCount));
    std::printf("  yaw here is %+.3f deg; the member at production's median |yaw| this run is %zu.\n",
                ensembles[0].members[representative].finalYaw * degrees, medianMember);

    auto runs = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        runs.push_back(record(setup.value(), split.value(), arm.assists, calibration, arm.candidate, 1.0,
                              memberEntry(representative, ensembleCount), 0.0, true));
    }

    std::printf("\n  %-28s %10s %10s %12s %12s %12s\n", "arm", "yaw deg", "stop m", "FL empty s", "FL cycles",
                "rear cycles");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto episodes = lowerBoundEpisodes(runs[index], 0, period);
        std::printf("  %-28s %10.3f %10.3f %12.4f %12zu %12zu\n", arms[index].name.c_str(),
                    runs[index].finalYaw * degrees, runs[index].distance, static_cast<double>(episodes.steps) * period,
                    cyclesOf(runs[index], 0), cyclesOf(runs[index], 2));
    }

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printWideSplitTrace(assists.antilock, arms[index].name.c_str(), runs[index], 0.20);
    }

    // --- 18.13 THE FIRST-DIVERGENCE AUDIT -------------------------------------------------------------
    std::printf("\n--- 18.13 FIRST-DIVERGENCE AUDIT on member %zu: controller transition -> pressure ->\n",
                representative);
    std::printf("    left/right Fx -> yaw rate -> accumulated yaw ---\n");

    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        const auto any = firstDivergence(runs[0], runs[index], -1);

        std::printf("\n  %s\n", arms[index].name.c_str());

        if (!any.found)
        {
            std::printf("    NO DIVERGENCE at any controller step.\n");
            continue;
        }

        const auto& step = runs[0].steps[any.index];
        const auto terms = termsOf(assists.antilock, step);

        std::printf("    first difference   t = %.5f s, step %zu, channel %s%s\n", any.time, any.index,
                    channelName(any.channel), any.channel == 0 ? " (the HIGH-mu front)" : "");
        std::printf("    phase before       %s\n", modulatorName(any.before));
        std::printf("    terms              pastBand %s, guardSlip %.4f, sensed slip %.4f, excess %+.2f m/s^2,\n",
                    any.pastBand ? "TRUE" : "false", any.guardSlip, any.slip, any.excess);
        std::printf("                       losing %s, surging %s, surged %s, accel %+.2f m/s^2\n",
                    terms.losing ? "TRUE" : "false", terms.surging ? "TRUE" : "false", terms.surged ? "TRUE" : "false",
                    any.acceleration);
        std::printf("    which leg          B state %s, D state %s, C1 taper state %s, C2 state %s\n",
                    terms.bState ? "TRUE" : "false", terms.dState ? "TRUE" : "false", terms.c1State ? "TRUE" : "false",
                    terms.c2State ? "TRUE" : "false");
        std::printf("    production takes   %s -> %s, %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.productionAfter), any.productionPressure / bar);
        std::printf("    this arm takes     %s -> %s, %.4f bar\n", modulatorName(any.before),
                    modulatorName(any.armAfter), any.armPressure / bar);

        std::printf("    per channel:");
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            const auto each = firstDivergence(runs[0], runs[index], static_cast<long>(channel));
            std::printf("  %s ", channelName(channel));
            if (each.found)
            {
                std::printf("%.4f s", each.time);
            }
            else
            {
                std::printf("never");
            }
        }
        std::printf("\n\n");

        printCausalChain(runs[0], runs[index], any);

        const auto after = std::min(any.time + 1.0, 30.0);
        std::printf("    mean left-right Fx in the second after it: production %+.1f N, this arm %+.1f N\n",
                    meanImbalance(runs[0], any.time, after), meanImbalance(runs[index], any.time, after));
        std::printf("    mean over the whole stop:                  production %+.1f N, this arm %+.1f N\n",
                    meanImbalance(runs[0], 0.0, 1.0e9), meanImbalance(runs[index], 0.0, 1.0e9));
        std::printf("    final yaw production %+.3f deg, this arm %+.3f deg; stop %.3f against %.3f m\n",
                    runs[0].finalYaw * degrees, runs[index].finalYaw * degrees, runs[0].distance, runs[index].distance);
    }

    // --- 18.14 THE INTERACTION LEDGER -----------------------------------------------------------------
    std::printf("\n--- 18.14 INTERACTION: does switching one leg change another leg's operating population?"
                " ---\n");
    std::printf("  These are ONE-AT-A-TIME ablations. They do not prove additivity and no percentage below\n");
    std::printf("  is summed anywhere. The table is the geometric STATE census of every leg in every arm,\n");
    std::printf("  summed over all three channels, so a leg whose population moves when a DIFFERENT leg is\n");
    std::printf("  switched is visible — including in O, where every leg is inert but the states still\n");
    std::printf("  arise and can be counted.\n");
    std::printf("\n  %-28s %13s %13s %13s %13s\n", "arm", "B state", "D state", "C1 state", "C2 state");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto b = std::size_t{0};
        auto d = std::size_t{0};
        auto c1 = std::size_t{0};
        auto c2 = std::size_t{0};

        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            b += ensembles[index].events.bSteps[channel];
            d += ensembles[index].events.dSteps[channel];
            c1 += ensembles[index].events.c1Tapered[channel];
            c2 += ensembles[index].events.c2Steps[channel];
        }

        std::printf("  %-28s %13zu %13zu %13zu %13zu\n", arms[index].name.c_str(), b, d, c1, c2);
    }

    std::printf("\n  and the transition counts the interaction shows up in most directly\n");
    std::printf("\n  %-28s %14s %14s %14s %14s\n", "arm", "FL Rec->Dump", "FL Rec->Rea", "rear Rec->Dump",
                "rear Rec->Rea");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("  %-28s %14zu %14zu %14zu %14zu\n", arms[index].name.c_str(),
                    ensembles[index].transitions.recoverDump[0], ensembles[index].transitions.recoverReapply[0],
                    ensembles[index].transitions.recoverDump[2], ensembles[index].transitions.recoverReapply[2]);
    }

    std::printf("\n  Production is unchanged by this file. `Candidate::replica` is false in arms P and O;\n");
    std::printf("  section 18.1 proves the four-leg copy reproduces production to the bit and with an\n");
    std::printf("  identical state machine. No threshold, gradient, estimator constant or tyre parameter is\n");
    std::printf("  touched anywhere in it, and no controller is designed.\n");
}

// =============================================================================================
// SECTION 19 — THE A x D INTERACTION, ON THE SPLIT-MU n = 29 ENSEMBLE.
// `./EngineTests "[.leg-ad]"` — hidden, registers no ctest entry, changes no production file.
//
// **The question is `A x D`, not "is D bad".** Section 18 already answered the second: with leg A
// held in its production state, leg D OFF changes the state machine enormously, shortens the stop
// and cuts pressure starvation, and moves the ensemble median |yaw| by about one degree. None of
// B, C or D individually explains the yaw that separates production from the whole-law-off
// control. This section asks the one pairwise question those results justify: whether removing D
// *inside the A-OFF operating regime* is a different intervention from removing D in production's,
// because A OFF is known to move the Dump/Recover-side population by an order of magnitude
// (split-mu Recover -> Dump, rear channel: production 248, A OFF 2856).
//
// FOUR ARMS AND NO OTHERS:
//
//   P    production                     A ON  B ON  C ON  D ON
//   O    whole law OFF                  the existing control, `slipAwareRecovery` false
//   A    leg A OFF only                 A OFF B ON  C ON  D ON
//   AD   legs A and D OFF               A OFF B ON  C ON  D OFF
//
// **The D-OFF arm is LOADED, not re-run.** Section 18's D arm is on the same fixture, the same
// deterministic 29-member entry sweep and the same replica, and its per-member result is recorded
// in `traces/leg-bcd-splitmu-20260907.txt`. It is transcribed below at the three decimals that
// trace prints, and 19.1 proves the transcription belongs to this fixture by checking the recorded
// P and O columns of the same table against this run's own live P and O to that precision. If that
// check fails the loaded D is from a different world and nothing that uses it may be read.
//
// **The deciding evidence is member-paired, not aggregate.** The system is deterministic and
// bifurcating: leg A OFF changes which events legs B, C and D ever see, so a pair effect larger
// than the sum of two single median effects is not by itself proof of interaction. The ordinary
// aggregate residual is computed for description; the primary instrument is the per-member
// residual `I_i = yAD_i - yA_i - yD_i + yP_i` and the per-member distance to that member's own
// whole-law-off result.
//
// Nothing here designs a controller, changes a threshold, adds an observable or touches
// `RaceEngine/Assists`. No new fixture and no new ABS implementation: the ensemble, the surface,
// the settle, the pedal, the stopping condition, the percentile definition and the
// distribution-shape rule are section 18's, used unchanged.
// =============================================================================================
namespace
{

// --- SECTION 18's SPLIT-MU MEMBER MAP, TRANSCRIBED ---------------------------------------------
//
// Signed final yaw in degrees and stopping distance in metres, member 0 to 28 in entry-speed
// order, from `traces/leg-bcd-splitmu-20260907.txt` lines 143-171. The D column is the one this
// section needs; the P and O columns are here only so the transcription can be checked against a
// live run of the same two arms, which is what licenses reading D at all.
constexpr auto recordedProductionYaw = std::array<double, ensembleCount>{
    -18.534, -12.619, -32.680, -14.585, -12.138, -18.218, -17.499, -16.959, -9.490,  -17.827,
    -14.098, -15.181, -20.671, -20.690, -12.171, -20.483, -14.793, -17.756, -25.919, -26.711,
    -16.112, -21.131, -13.416, -17.263, -15.289, -10.578, -20.556, -15.416, -25.315};

constexpr auto recordedControlYaw = std::array<double, ensembleCount>{
    -18.850, -1.306, -0.734,  -3.026,  -2.092, -20.341, -1.116, -1.695,  -0.811, -9.097,
    -0.280,  -1.081, -19.248, -7.511,  -2.481, -1.130,  -1.079, -22.024, -1.632, -4.169,
    -0.649,  -2.479, -13.008, -11.307, -0.657, -1.418,  -1.977, -12.527, -13.471};

constexpr auto recordedDumpGateYaw = std::array<double, ensembleCount>{
    -12.273, -13.736, -15.452, -27.461, -24.374, -9.305,  -16.689, -29.020, -18.014, -21.631,
    -16.615, -8.497,  -14.686, -18.090, -24.429, -15.773, -15.873, -16.573, -24.601, -16.197,
    -14.961, -12.313, -18.737, -15.429, -34.788, -12.083, -15.605, -15.847, -16.837};

constexpr auto recordedProductionStop =
    std::array<double, ensembleCount>{56.406, 56.468, 56.825, 55.509, 54.892, 56.602, 56.634, 55.587, 55.228, 57.127,
                                      55.833, 57.728, 57.770, 56.530, 56.671, 57.730, 56.246, 56.458, 57.019, 57.508,
                                      56.214, 57.717, 55.475, 56.764, 57.220, 56.386, 57.893, 56.566, 56.836};

constexpr auto recordedControlStop =
    std::array<double, ensembleCount>{55.375, 55.512, 54.878, 54.600, 54.924, 55.556, 55.168, 54.820, 55.196, 54.871,
                                      55.398, 55.261, 54.817, 55.807, 55.519, 55.729, 55.375, 56.947, 55.931, 55.235,
                                      55.939, 55.838, 55.672, 55.679, 55.965, 56.469, 55.869, 56.840, 56.242};

constexpr auto recordedDumpGateStop =
    std::array<double, ensembleCount>{54.369, 54.641, 54.881, 55.921, 55.737, 55.681, 55.380, 56.359, 55.587, 55.737,
                                      54.906, 55.644, 55.381, 55.639, 56.479, 56.224, 56.467, 55.216, 56.421, 56.200,
                                      56.118, 55.575, 55.903, 55.302, 57.463, 55.491, 56.269, 56.092, 55.771};

// The trace prints three decimals, so a live value and its transcription agree when they are
// within half of the last printed place. This is the tolerance of a TRANSCRIPTION CHECK and it is
// used for nothing else: every arithmetic result below is computed on live values except where the
// D arm is named, and D carries this quantisation with it.
constexpr auto transcriptionTolerance = 0.0005 + 1.0e-9;

[[nodiscard]] std::vector<double> magnitudeOf(const std::array<double, ensembleCount>& values)
{
    auto out = std::vector<double>{};
    out.reserve(ensembleCount);

    for (const auto value : values)
    {
        out.push_back(std::abs(value));
    }

    return out;
}

[[nodiscard]] std::vector<double> plainOf(const std::array<double, ensembleCount>& values)
{
    return std::vector<double>(values.begin(), values.end());
}

// A fixed-bin histogram, so "did a population move" can be read rather than asserted. The edges
// are stated here and are not fitted to any arm: they are round degrees spanning the range every
// arm of section 18 occupied.
constexpr auto yawBinEdges = std::array<double, 5>{5.0, 10.0, 15.0, 20.0, 30.0};

[[nodiscard]] std::array<std::size_t, 6> histogramOf(const std::vector<double>& values)
{
    auto bins = std::array<std::size_t, 6>{};

    for (const auto value : values)
    {
        auto slot = std::size_t{0};

        while (slot < yawBinEdges.size() && value >= yawBinEdges[slot])
        {
            slot++;
        }

        bins[slot]++;
    }

    return bins;
}

struct AdArm
{
    std::string name;
    std::string tag;
    AssistSetup assists;
    Candidate candidate;
};

// The causal chain, with both column labels named rather than assumed to be production. Section
// 18's `printCausalChain` hard-codes "P" for its base, which is right there and wrong here: this
// section's key comparison has arm A as the base.
void printPairCausalChain(const char* baseName, const char* armName, const Run& base, const Run& arm,
                          const Divergence& divergence)
{
    static constexpr auto offsets = std::array<double, 5>{0.010, 0.050, 0.100, 0.250, 0.500};

    std::printf("    base = %s, arm = %s\n", baseName, armName);
    std::printf("    %9s | %17s | %17s | %19s | %17s | %17s\n", "offset", "FL bar base/arm", "FR bar base/arm",
                "left-right Fx [N]", "yaw rate [rad/s]", "yaw [deg]");
    std::printf("    %s\n", "---------------------------------------------------------------------------------"
                            "---------------------------------------------");

    for (const auto offset : offsets)
    {
        const auto when = divergence.time + offset;
        const auto* first = tickAt(base, when);
        const auto* second = tickAt(arm, when);

        if (first == nullptr || second == nullptr)
        {
            std::printf("    %+8.0f ms | %s\n", offset * 1000.0, "past the end of one of the two runs");
            continue;
        }

        std::printf("    %+8.0f ms | %7.3f %7.3f  | %7.3f %7.3f  | %8.1f %8.1f  | %8.4f %8.4f | %8.3f %8.3f\n",
                    offset * 1000.0, first->wheels[0].pressure / bar, second->wheels[0].pressure / bar,
                    first->wheels[1].pressure / bar, second->wheels[1].pressure / bar,
                    sideForce(*first, true) - sideForce(*first, false),
                    sideForce(*second, true) - sideForce(*second, false), first->yawRate, second->yawRate,
                    first->yaw * degrees, second->yaw * degrees);
    }
}

// The first-divergence audit between any two arms, with the leg terms read off the BASE arm's step
// because that is the state both arms were looking at when they parted.
void printPairDivergence(const AntilockSetup& setup, const char* baseName, const char* armName, const Run& base,
                         const Run& arm)
{
    const auto any = firstDivergence(base, arm, -1);

    if (!any.found)
    {
        std::printf("    NO DIVERGENCE at any controller step.\n");
        return;
    }

    const auto& step = base.steps[any.index];
    const auto terms = termsOf(setup, step);

    std::printf("    first difference   t = %.5f s, step %zu, channel %s%s\n", any.time, any.index,
                channelName(any.channel), any.channel == 0 ? " (the HIGH-mu front)" : "");
    std::printf("    phase before       %s\n", modulatorName(any.before));
    std::printf("    terms              pastBand %s, guardSlip %.4f, raw sensed slip %.4f, excess %+.2f m/s^2,\n",
                any.pastBand ? "TRUE" : "false", any.guardSlip, any.slip, any.excess);
    std::printf("                       losing %s, surging %s, surged %s, accel %+.2f m/s^2\n",
                terms.losing ? "TRUE" : "false", terms.surging ? "TRUE" : "false", terms.surged ? "TRUE" : "false",
                any.acceleration);
    std::printf("    which leg          B state %s, D state %s, C1 taper state %s, C2 state %s\n",
                terms.bState ? "TRUE" : "false", terms.dState ? "TRUE" : "false", terms.c1State ? "TRUE" : "false",
                terms.c2State ? "TRUE" : "false");
    std::printf("    %-6s takes        %s -> %s, %.4f bar\n", baseName, modulatorName(any.before),
                modulatorName(any.productionAfter), any.productionPressure / bar);
    std::printf("    %-6s takes        %s -> %s, %.4f bar\n", armName, modulatorName(any.before),
                modulatorName(any.armAfter), any.armPressure / bar);

    std::printf("    per channel:");
    for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
    {
        const auto each = firstDivergence(base, arm, static_cast<long>(channel));
        std::printf("  %s ", channelName(channel));

        if (each.found)
        {
            std::printf("%.4f s", each.time);
        }
        else
        {
            std::printf("never");
        }
    }
    std::printf("\n\n");

    printPairCausalChain(baseName, armName, base, arm, any);

    const auto after = std::min(any.time + 1.0, 30.0);
    std::printf("    mean left-right Fx in the second after it: %s %+.1f N, %s %+.1f N\n", baseName,
                meanImbalance(base, any.time, after), armName, meanImbalance(arm, any.time, after));
    std::printf("    mean over the whole stop:                  %s %+.1f N, %s %+.1f N\n", baseName,
                meanImbalance(base, 0.0, 1.0e9), armName, meanImbalance(arm, 0.0, 1.0e9));
    std::printf("    final yaw %s %+.3f deg, %s %+.3f deg; stop %.3f against %.3f m\n", baseName,
                base.finalYaw * degrees, armName, arm.finalYaw * degrees, base.distance, arm.distance);
}

} // namespace

TEST_CASE("does removing leg D inside the leg-A-off regime expose an A x D interaction on split mu", "[.leg-ad]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto calibration = calibrationOf(setup.value(), assists);
    const auto period = assists.controlRate > 0.0 ? 1.0 / assists.controlRate : tick;

    REQUIRE_FALSE(assists.antilock.yawMomentDelay);
    REQUIRE(assists.antilock.recoveryAuthority != RecoveryAuthority::Disabled);

    const auto replicaOfLegs = [](const bool legA, const bool legB, const bool legC, const bool legD)
    {
        auto candidate = Candidate{};
        candidate.replica = true;
        candidate.legA = legA;
        candidate.legB = legB;
        candidate.legC = legC;
        candidate.legD = legD;

        return candidate;
    };

    auto arms = std::vector<AdArm>{};
    arms.push_back(AdArm{"P production", "P", assists, Candidate{}});
    arms.push_back(AdArm{"O whole law OFF", "O", withoutSlipAwareRecovery(assists), Candidate{}});
    arms.push_back(AdArm{"A leg-A OFF", "A", assists, replicaOfLegs(false, true, true, true)});
    arms.push_back(AdArm{"AD legs A and D OFF", "AD", assists, replicaOfLegs(false, true, true, false)});

    std::printf("\n=== SECTION 19: THE A x D INTERACTION ON THE SPLIT-MU n = %zu ENSEMBLE ===\n", ensembleCount);
    std::printf("  Four simulated arms and no others: P, O, A (leg A off) and AD (legs A and D off). The\n");
    std::printf("  D-OFF arm is LOADED from traces/leg-bcd-splitmu-20260907.txt and is not re-run.\n");
    std::printf("  %zu deterministic members, entry %.4f to %.4f m/s, full pedal, no steering, grip 1.00 at\n",
                ensembleCount, memberEntry(0, ensembleCount), memberEntry(ensembleCount - 1, ensembleCount));
    std::printf("  x<0 (LEFT, so FL is the HIGH-mu front channel) and 0.35 at x>0 (RIGHT, FR the low-mu\n");
    std::printf("  one). Controller period %.5f s (%.0f Hz). Nothing is calibrated and no arm is scored.\n", period,
                assists.controlRate);

    // --- 19.1 THE REPLICA, AND THE TRANSCRIPTION ------------------------------------------------
    std::printf("\n--- 19.1 REPLICA IDENTITY: the probe copy with A, B, C AND D all ON, against production ---\n");
    std::printf("  Re-proved here rather than carried forward from 18.1, because this section reads the\n");
    std::printf("  copy with two legs off and nothing below may be read if the copy has drifted.\n");

    const auto production = bcdEnsemble(setup.value(), split.value(), assists, calibration, Candidate{}, period, true);
    const auto mirror = bcdEnsemble(setup.value(), split.value(), assists, calibration,
                                    replicaOfLegs(true, true, true, true), period, true);

    REQUIRE(mirror.members.size() == production.members.size());

    auto faithful = true;
    for (auto index = std::size_t{0}; index < production.members.size(); index++)
    {
        faithful = faithful && sameMember(production.members[index], mirror.members[index]);
    }

    std::printf("  SPLIT MU, %zu members, every recorded scalar: %s\n", ensembleCount,
                faithful ? "BIT-IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(faithful);

    auto sameMachine = true;
    for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
    {
        sameMachine = sameMachine && production.transitions.dumpIn[channel] == mirror.transitions.dumpIn[channel] &&
                      production.transitions.recoverIn[channel] == mirror.transitions.recoverIn[channel] &&
                      production.transitions.holdIn[channel] == mirror.transitions.holdIn[channel] &&
                      production.transitions.reapplyIn[channel] == mirror.transitions.reapplyIn[channel] &&
                      production.transitions.recoverDump[channel] == mirror.transitions.recoverDump[channel] &&
                      production.transitions.recoverReapply[channel] == mirror.transitions.recoverReapply[channel] &&
                      production.transitions.reapplyDump[channel] == mirror.transitions.reapplyDump[channel] &&
                      production.transitions.emptySteps[channel] == mirror.transitions.emptySteps[channel];
    }

    std::printf("  ...and every state-machine transition count, per channel: %s\n",
                sameMachine ? "IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(sameMachine);

    auto ensembles = std::vector<BcdEnsemble>{};
    ensembles.push_back(production);
    for (auto index = std::size_t{1}; index < arms.size(); index++)
    {
        ensembles.push_back(bcdEnsemble(setup.value(), split.value(), arms[index].assists, calibration,
                                        arms[index].candidate, period, true));
    }

    for (const auto& ensemble : ensembles)
    {
        for (const auto& member : ensemble.members)
        {
            REQUIRE(member.stopped);
        }
    }

    std::printf("\n  THE LOADED D ARM, checked against this run rather than trusted. The same table that\n");
    std::printf("  carries the D column carries a P and an O column; both are re-run live here, and every\n");
    std::printf("  member of both must agree with the transcription to the trace's own printed precision\n");
    std::printf("  (three decimals, so +/- %.4f). If it does not, the loaded D is from a different world.\n",
                transcriptionTolerance);

    auto worstYaw = 0.0;
    auto worstStop = 0.0;
    auto transcribed = true;

    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        const auto yawGap = std::array<double, 2>{
            std::abs(ensembles[0].members[member].finalYaw * degrees - recordedProductionYaw[member]),
            std::abs(ensembles[1].members[member].finalYaw * degrees - recordedControlYaw[member])};
        const auto stopGap =
            std::array<double, 2>{std::abs(ensembles[0].members[member].distance - recordedProductionStop[member]),
                                  std::abs(ensembles[1].members[member].distance - recordedControlStop[member])};

        for (auto side = std::size_t{0}; side < 2; side++)
        {
            worstYaw = std::max(worstYaw, yawGap[side]);
            worstStop = std::max(worstStop, stopGap[side]);
            transcribed =
                transcribed && yawGap[side] <= transcriptionTolerance && stopGap[side] <= transcriptionTolerance;
        }
    }

    std::printf("  worst live-against-recorded difference over P and O, all %zu members: yaw %.6f deg,\n",
                ensembleCount, worstYaw);
    std::printf("  stop %.6f m — %s\n", worstStop, transcribed ? "WITHIN THE PRINTED PRECISION" : "*** OUTSIDE ***");
    REQUIRE(transcribed);

    // The four live arms' magnitudes, and the loaded D's, in one place.
    const auto yawP = ensembles[0].yawMagnitude;
    const auto yawO = ensembles[1].yawMagnitude;
    const auto yawA = ensembles[2].yawMagnitude;
    const auto yawAD = ensembles[3].yawMagnitude;
    const auto yawD = magnitudeOf(recordedDumpGateYaw);

    const auto stopP = columnOf(ensembles[0].members, &Member::distance, false);
    const auto stopO = columnOf(ensembles[1].members, &Member::distance, false);
    const auto stopA = columnOf(ensembles[2].members, &Member::distance, false);
    const auto stopAD = columnOf(ensembles[3].members, &Member::distance, false);
    const auto stopD = plainOf(recordedDumpGateStop);

    // --- 19.2 THE YAW DISTRIBUTIONS ---------------------------------------------------------------
    std::printf("\n--- 19.2 SPLIT-MU |FINAL YAW| [deg] ---\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(), distributionOf(ensembles[index].yawMagnitude));
    }
    wideRow("D leg-D OFF (loaded)", distributionOf(yawD));

    std::printf("\n  SIGNED final yaw [deg], so a sign flip cannot hide inside a magnitude\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto values = columnOf(ensembles[index].members, &Member::finalYaw, false);
        for (auto& value : values)
        {
            value *= degrees;
        }
        wideRow(arms[index].name.c_str(), distributionOf(values));
    }
    wideRow("D leg-D OFF (loaded)", distributionOf(plainOf(recordedDumpGateYaw)));

    std::printf("\n  SIGN, REVERSALS AND RATE (live arms only — the loaded D column carries final yaw and\n");
    std::printf("  stopping distance and no within-run quantity)\n");
    std::printf("\n  %-24s %9s %9s %11s %13s %13s %13s\n", "arm", "positive", "negative", "range deg", "head sign rev",
                "in-run flips", "med peak rate");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        auto positive = std::size_t{0};
        auto negative = std::size_t{0};
        auto flips = std::size_t{0};
        auto signedYaw = std::vector<double>{};
        auto rates = std::vector<double>{};
        auto headingReversals = std::size_t{0};
        auto previous = 0;

        for (const auto& member : ensembles[index].members)
        {
            positive += member.finalYaw > 0.0 ? 1 : 0;
            negative += member.finalYaw < 0.0 ? 1 : 0;
            flips += member.yawReversals;
            signedYaw.push_back(member.finalYaw * degrees);
            rates.push_back(member.peakYawRate);

            const auto sign = member.finalYaw > 0.0 ? 1 : (member.finalYaw < 0.0 ? -1 : 0);
            if (sign != 0)
            {
                headingReversals += (previous != 0 && sign != previous) ? 1 : 0;
                previous = sign;
            }
        }

        const auto spread = distributionOf(signedYaw);

        std::printf("  %-24s %9zu %9zu %11.3f %13zu %13zu %13.4f\n", arms[index].name.c_str(), positive, negative,
                    spread.maximum - spread.minimum, headingReversals, flips, distributionOf(rates).median);
    }
    std::printf("\n  \"head sign rev\" counts sign changes of the FINAL heading as the entry speed is swept\n");
    std::printf("  through the %zu members in order — the deterministic bifurcation signature, not a\n", ensembleCount);
    std::printf("  within-run quantity. \"in-run flips\" is the yaw-rate sign flip count, deadband 0.02 rad/s.\n");

    std::printf("\n--- 19.3 DISTRIBUTION SHAPE, by section 18's rule, applied unchanged ---\n");
    std::printf("  A gap between neighbouring sorted members is a SEPARATION when it is >= %.0fx the\n", gapMultiple);
    std::printf("  sample's OWN median neighbour gap, leaves >= %zu members each side and is >= %zu members\n", gapTail,
                gapTail);
    std::printf("  past the previous separation. One separation BIMODAL, two or more MULTIMODAL /\n");
    std::printf("  IRREGULAR, none UNIMODAL. The rule's one calibration point is section 17's whole-law-off\n");
    std::printf("  control, and it is not an arm of this section. O is re-run live here, so the rule can be\n");
    std::printf("  seen to reproduce that 19/10 split before AD is classified by it.\n");
    std::printf("\n  %-24s %-22s %10s %10s\n", "arm", "verdict", "med gap", "separations");
    std::printf("  %s\n", "---------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto shape = shapeOf(ensembles[index].yawMagnitude);
        std::printf("  %-24s %-22s %10.4f %10zu\n", arms[index].name.c_str(), shape.verdict.c_str(), shape.medianGap,
                    shape.separations);
    }
    {
        const auto shape = shapeOf(yawD);
        std::printf("  %-24s %-22s %10.4f %10zu\n", "D leg-D OFF (loaded)", shape.verdict.c_str(), shape.medianGap,
                    shape.separations);
    }

    std::printf("\n  the three largest gaps of each arm, and whether the rule calls each a separation\n");
    std::printf("\n  %-24s %10s %9s %16s %8s %8s %12s\n", "arm", "gap deg", "x median", "between", "below", "above",
                "separation?");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "");
    for (auto index = std::size_t{0}; index <= arms.size(); index++)
    {
        const auto values = index < arms.size() ? ensembles[index].yawMagnitude : yawD;
        const auto shape = shapeOf(values);
        const auto* name = index < arms.size() ? arms[index].name.c_str() : "D leg-D OFF (loaded)";

        for (auto rank = std::size_t{0}; rank < std::min(std::size_t{3}, shape.ranked.size()); rank++)
        {
            const auto& gap = shape.ranked[rank];
            auto where = std::array<char, 64>{};
            std::snprintf(where.data(), where.size(), "%.2f / %.2f", gap.edgeLow, gap.edgeHigh);
            std::printf("  %-24s %10.3f %9.1f %16s %8zu %8zu %12s\n", name, gap.size, gap.multiple, where.data(),
                        gap.below, gap.above, gap.separates ? "YES" : ".");
            name = "";
        }
    }

    std::printf("\n  the sorted |yaw| ladders, so the shape is visible and not only classified\n");
    for (auto index = std::size_t{0}; index <= arms.size(); index++)
    {
        auto sorted = index < arms.size() ? ensembles[index].yawMagnitude : yawD;
        std::sort(sorted.begin(), sorted.end());

        std::printf("\n  %s\n   ", index < arms.size() ? arms[index].name.c_str() : "D leg-D OFF (loaded)");
        for (auto member = std::size_t{0}; member < sorted.size(); member++)
        {
            std::printf(" %7.3f", sorted[member]);
            if ((member + 1) % 10 == 0)
            {
                std::printf("\n   ");
            }
        }
        std::printf("\n");
    }

    std::printf("\n  and the same ladders as a FIXED-BIN histogram, so a population that moves can be read\n");
    std::printf("  off rather than inferred. The edges are stated here and fitted to nothing: they are\n");
    std::printf("  round degrees spanning the range every arm of section 18 occupied.\n");
    std::printf("\n  %-24s %9s %9s %9s %9s %9s %9s\n", "arm", "<5", "5-10", "10-15", "15-20", "20-30", ">=30");
    std::printf("  %s\n", "---------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index <= arms.size(); index++)
    {
        const auto bins = histogramOf(index < arms.size() ? ensembles[index].yawMagnitude : yawD);
        std::printf("  %-24s", index < arms.size() ? arms[index].name.c_str() : "D leg-D OFF (loaded)");
        for (const auto count : bins)
        {
            std::printf(" %9zu", count);
        }
        std::printf("\n");
    }

    // --- 19.4 STOPPING DISTANCE -------------------------------------------------------------------
    std::printf("\n--- 19.4 SPLIT-MU STOPPING DISTANCE [m] — secondary context, not a target ---\n");
    wideHeader("arm");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        wideRow(arms[index].name.c_str(), distributionOf(columnOf(ensembles[index].members, &Member::distance, false)));
    }
    wideRow("D leg-D OFF (loaded)", distributionOf(stopD));

    std::printf("\n  the existing split-mu criterion, unchanged: |final yaw| < 45 deg\n");
    for (auto index = std::size_t{0}; index <= arms.size(); index++)
    {
        const auto& values = index < arms.size() ? ensembles[index].yawMagnitude : yawD;
        auto inside = std::size_t{0};
        for (const auto value : values)
        {
            inside += value < 45.0 ? 1 : 0;
        }
        std::printf("  %-24s %zu/%zu inside\n", index < arms.size() ? arms[index].name.c_str() : "D leg-D OFF (loaded)",
                    inside, ensembleCount);
    }

    // --- 19.5 THE MEMBER MAP AND THE PAIRED RESIDUAL ----------------------------------------------
    std::printf("\n--- 19.5 MEMBER-BY-MEMBER MAP, all %zu deterministic members ---\n", ensembleCount);
    std::printf("  Signed yaw and stop for every arm; then the paired interaction residual on MAGNITUDES,\n");
    std::printf("      I_i = |yaw AD| - |yaw A| - |yaw D| + |yaw P|\n");
    std::printf("  I_i < 0 means removing D was MORE yaw-beneficial for that member with A already off than\n");
    std::printf("  it was in production's context; I_i > 0 means less. No epsilon is applied to zero.\n");
    std::printf("\n  %-3s %9s | %8s %8s %8s %8s %8s | %8s %8s %8s %8s %8s | %9s\n", "k", "entry m/s", "P yaw", "O yaw",
                "A yaw", "AD yaw", "D yaw", "P stop", "O stop", "A stop", "AD stop", "D stop", "I_i deg");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------------------------------------------");

    auto residual = std::vector<double>{};
    residual.reserve(ensembleCount);

    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        const auto value = yawAD[member] - yawA[member] - yawD[member] + yawP[member];
        residual.push_back(value);

        std::printf("  %-3zu %9.4f |", member, ensembles[0].members[member].entry);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            std::printf(" %8.3f", ensembles[index].members[member].finalYaw * degrees);
        }
        std::printf(" %8.3f |", recordedDumpGateYaw[member]);
        std::printf(" %8.3f %8.3f %8.3f %8.3f %8.3f |", stopP[member], stopO[member], stopA[member], stopAD[member],
                    stopD[member]);
        std::printf(" %9.3f\n", value);
    }

    std::printf("\n  AD AGAINST A, AND AD AGAINST P, on |yaw|\n");
    {
        auto lowerThanA = std::size_t{0};
        auto higherThanA = std::size_t{0};
        auto equalToA = std::size_t{0};
        auto nearerO = std::size_t{0};
        auto nearerA = std::size_t{0};
        auto lowerThanP = std::size_t{0};
        auto higherThanP = std::size_t{0};

        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            lowerThanA += yawAD[member] < yawA[member] ? 1 : 0;
            higherThanA += yawAD[member] > yawA[member] ? 1 : 0;
            equalToA += yawAD[member] == yawA[member] ? 1 : 0;
            lowerThanP += yawAD[member] < yawP[member] ? 1 : 0;
            higherThanP += yawAD[member] > yawP[member] ? 1 : 0;

            const auto toO = std::abs(yawAD[member] - yawO[member]);
            const auto toA = std::abs(yawAD[member] - yawA[member]);
            nearerO += toO < toA ? 1 : 0;
            nearerA += toA < toO ? 1 : 0;
        }

        std::printf("  |yaw AD| <  |yaw A|      %zu/%zu\n", lowerThanA, ensembleCount);
        std::printf("  |yaw AD| >  |yaw A|      %zu/%zu\n", higherThanA, ensembleCount);
        std::printf("  |yaw AD| == |yaw A|      %zu/%zu\n", equalToA, ensembleCount);
        std::printf("  AD nearer its own O than its own A   %zu/%zu\n", nearerO, ensembleCount);
        std::printf("  AD nearer its own A than its own O   %zu/%zu\n", nearerA, ensembleCount);
        std::printf("  |yaw AD| <  |yaw P|      %zu/%zu\n", lowerThanP, ensembleCount);
        std::printf("  |yaw AD| >  |yaw P|      %zu/%zu\n", higherThanP, ensembleCount);
    }

    std::printf("\n--- 19.6 THE PAIRED INTERACTION RESIDUAL I_i ---\n");
    std::printf("  THE PRIMARY INTERACTION INSTRUMENT. The aggregate residual below it is descriptive\n");
    std::printf("  only: A OFF changes the event population B, C and D ever see, so median additivity is\n");
    std::printf("  not a test of anything on a bifurcating deterministic system.\n");
    wideHeader("quantity");
    wideRow("I_i [deg]", distributionOf(residual));
    {
        auto negative = std::size_t{0};
        auto positive = std::size_t{0};
        auto zero = std::size_t{0};

        for (const auto value : residual)
        {
            negative += value < 0.0 ? 1 : 0;
            positive += value > 0.0 ? 1 : 0;
            zero += value == 0.0 ? 1 : 0;
        }

        std::printf("\n  I_i <  0 (D more beneficial with A off)   %zu/%zu\n", negative, ensembleCount);
        std::printf("  I_i >  0 (D less beneficial with A off)   %zu/%zu\n", positive, ensembleCount);
        std::printf("  I_i == 0 exactly                          %zu/%zu\n", zero, ensembleCount);
    }

    std::printf("\n  THE ORDINARY AGGREGATE INTERACTION, for description and for nothing else.\n");
    std::printf("      dA = Y(A) - Y(P),  dD = Y(D) - Y(P),  dAD = Y(AD) - Y(P),  I = dAD - (dA + dD)\n");
    std::printf("\n  %-22s %10s %10s %10s %10s %10s %10s %10s\n", "quantity", "Y(P)", "Y(O)", "Y(A)", "Y(D)", "Y(AD)",
                "dA+dD", "I");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "----");
    {
        const auto rowOf =
            [](const char* what, const double p, const double o, const double a, const double d, const double ad)
        {
            const auto deltaA = a - p;
            const auto deltaD = d - p;
            const auto deltaAD = ad - p;

            std::printf("  %-22s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", what, p, o, a, d, ad,
                        deltaA + deltaD, deltaAD - (deltaA + deltaD));
        };

        const auto dP = distributionOf(yawP);
        const auto dO = distributionOf(yawO);
        const auto dA = distributionOf(yawA);
        const auto dD = distributionOf(yawD);
        const auto dAD = distributionOf(yawAD);

        rowOf("median |yaw| [deg]", dP.median, dO.median, dA.median, dD.median, dAD.median);
        rowOf("mean   |yaw| [deg]", dP.mean, dO.mean, dA.mean, dD.mean, dAD.mean);
        rowOf("P90    |yaw| [deg]", dP.p90, dO.p90, dA.p90, dD.p90, dAD.p90);

        rowOf("median stop [m]", distributionOf(stopP).median, distributionOf(stopO).median,
              distributionOf(stopA).median, distributionOf(stopD).median, distributionOf(stopAD).median);
    }

    // --- 19.7 CONTROL-RELATIVE CLOSURE ------------------------------------------------------------
    std::printf("\n--- 19.7 CONTROL-RELATIVE CLOSURE: does AD move each member toward ITS OWN whole-law-off\n");
    std::printf("    result? dA_i = | |yaw A| - |yaw O| |, dAD_i = | |yaw AD| - |yaw O| |. ---\n");
    {
        auto dAValues = std::vector<double>{};
        auto dADValues = std::vector<double>{};
        auto closer = std::size_t{0};
        auto further = std::size_t{0};
        auto same = std::size_t{0};

        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            const auto toA = std::abs(yawA[member] - yawO[member]);
            const auto toAD = std::abs(yawAD[member] - yawO[member]);

            dAValues.push_back(toA);
            dADValues.push_back(toAD);

            closer += toAD < toA ? 1 : 0;
            further += toAD > toA ? 1 : 0;
            same += toAD == toA ? 1 : 0;
        }

        wideHeader("distance to own O");
        wideRow("dA_i  [deg]", distributionOf(dAValues));
        wideRow("dAD_i [deg]", distributionOf(dADValues));

        std::printf("\n  dAD_i <  dA_i  (AD nearer the control)   %zu/%zu\n", closer, ensembleCount);
        std::printf("  dAD_i >  dA_i  (AD further from it)      %zu/%zu\n", further, ensembleCount);
        std::printf("  dAD_i == dA_i                            %zu/%zu\n", same, ensembleCount);

        std::printf("\n  %-3s %11s %11s %11s %11s %11s\n", "k", "|yaw O|", "|yaw A|", "|yaw AD|", "dA_i", "dAD_i");
        std::printf("  %s\n", "----------------------------------------------------------------------");
        for (auto member = std::size_t{0}; member < ensembleCount; member++)
        {
            std::printf("  %-3zu %11.3f %11.3f %11.3f %11.3f %11.3f\n", member, yawO[member], yawA[member],
                        yawAD[member], dAValues[member], dADValues[member]);
        }
    }

    // --- 19.8 THE STATE MACHINE -------------------------------------------------------------------
    std::printf("\n--- 19.8 STATE-MACHINE DIFFERENTIAL, summed over all %zu members ---\n", ensembleCount);
    std::printf("  P and O are here as the two ends of the interval. The hypothesis under test is that A\n");
    std::printf("  OFF creates a large Recover -> Dump population which leg D then holds out of Reapply,\n");
    std::printf("  and that removing D in THAT environment is a different intervention from removing it in\n");
    std::printf("  production's. Section 18's D-OFF row, for orientation and not re-run: FL Rec->Dmp 54,\n");
    std::printf("  Rec->Rea 669, empty 13.398 s; rear Rec->Dmp 56, Rec->Rea 656, empty 11.705 s.\n");
    printTransitionHeader();
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printTransitionBlock(arms[index].name.c_str(), ensembles[index].transitions, period);
    }

    std::printf("\n  Reapply -> Dump, per channel\n");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printReapplyDump(arms[index].name.c_str(), ensembles[index].transitions);
    }

    std::printf("\n  PHASE RESIDENCY, share of physics ticks (FL is HIGH mu, FR is LOW mu, RL the rear pair)\n");
    std::printf("\n  %-24s %6s %10s %10s %10s %10s %10s\n", "arm", "wheel", "passive", "hold", "dump", "recover",
                "reapply");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-----");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printResidency(arms[index].name.c_str(), ensembles[index].transitions);
    }

    // --- 19.9 LEG B ---------------------------------------------------------------------------------
    std::printf("\n--- 19.9 LEG B POPULATION: `phase == Recover && pastBand && excess < recoveryAccel`, with\n");
    std::printf("    the re-departure branch not taken. A STATE census on each arm's own steps, with\n");
    std::printf("    `pastBand` read geometrically so O is comparable (see `LegTerms`). ---\n");
    std::printf("  Established context: split-mu Recover -> Dump on the rear channel is 248 in production\n");
    std::printf("  and 2856 with A OFF. The question here is whether D OFF collapses that A-induced\n");
    std::printf("  population, and by how much.\n");
    std::printf("\n  %-24s %5s %12s %10s %11s %12s %12s %12s\n", "arm", "chan", "qual steps", "time [s]", "episodes",
                "-> Dump", "stayed Rec", "Rec->Dump");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto* name = arms[index].name.c_str();
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-24s %5s %12zu %10.3f %11zu %12zu %12zu %12zu\n", name, channelName(channel),
                        ensembles[index].events.bSteps[channel],
                        static_cast<double>(ensembles[index].events.bSteps[channel]) * period,
                        ensembles[index].events.bEpisodes[channel], ensembles[index].events.bTookDump[channel],
                        ensembles[index].events.bStayedRecover[channel],
                        ensembles[index].transitions.recoverDump[channel]);
            name = "";
        }
    }

    // --- 19.10 LEG D --------------------------------------------------------------------------------
    std::printf("\n--- 19.10 LEG D POPULATION: `phase == Recover`, re-departure not taken, leg-B branch not\n");
    std::printf("    taken, `!(surged && surging)` and `pastBand` — the state where D ALONE stands between\n");
    std::printf("    Recover and Reapply. ---\n");
    std::printf("\n  %-24s %5s %12s %10s %11s %12s %12s\n", "arm", "chan", "blocked st", "time [s]", "episodes",
                "longest [s]", "-> Reapply");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "---");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto* name = arms[index].name.c_str();
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-24s %5s %12zu %10.3f %11zu %12.4f %12zu\n", name, channelName(channel),
                        ensembles[index].events.dSteps[channel],
                        static_cast<double>(ensembles[index].events.dSteps[channel]) * period,
                        ensembles[index].events.dEpisodes[channel],
                        static_cast<double>(ensembles[index].events.dLongest[channel]) * period,
                        ensembles[index].events.dTookReapply[channel]);
            name = "";
        }
    }

    std::printf("\n  WHAT BECAME OF THE RELEASED EXITS. Forward from each Recover step that leg D would have\n");
    std::printf("  held and that the arm let go to Reapply, inside that same arm. In A the gate is ON, so\n");
    std::printf("  the population is zero by construction and the census is empty; in AD it is the whole\n");
    std::printf("  point of the arm.\n");
    printFollowon("A arm, high-mu FL", ensembles[2].flFollow, period);
    printFollowon("A arm, rear channel", ensembles[2].rearFollow, period);
    printFollowon("AD arm, high-mu FL", ensembles[3].flFollow, period);
    printFollowon("AD arm, rear channel", ensembles[3].rearFollow, period);

    // --- 19.11 LEG C --------------------------------------------------------------------------------
    std::printf("\n--- 19.11 LEG C POPULATION: does removing D merely move the A-induced population\n");
    std::printf("    downstream into C? Leg C is NOT ablated anywhere in this section. ---\n");
    std::printf("\n  %-24s %5s %12s %11s %11s %11s %11s\n", "arm", "chan", "Reapply st", "C1 tapered", "C1 closed",
                "C2 qual", "C2 -> Dump");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "--");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto* name = arms[index].name.c_str();
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf("  %-24s %5s %12zu %11zu %11zu %11zu %11zu\n", name, channelName(channel),
                        ensembles[index].events.reapplySteps[channel], ensembles[index].events.c1Tapered[channel],
                        ensembles[index].events.c1Closed[channel], ensembles[index].events.c2Steps[channel],
                        ensembles[index].events.c2TookDump[channel]);
            name = "";
        }
    }

    // --- 19.12 PRESSURE STARVATION ------------------------------------------------------------------
    std::printf("\n--- 19.12 HIGH-MU FL AT THE ACTUATOR LOWER BOUND ---\n");
    std::printf("  Definition unchanged and not re-derived: a controller step whose resulting pressure is\n");
    std::printf("  <= 0, no epsilon. Summed over all %zu members. Section 18's D-OFF figure for FL was\n",
                ensembleCount);
    std::printf("  13.398 s, against P 15.459 s and O 11.368 s.\n");
    std::printf("\n  %-22s %11s %12s %9s %11s %11s %11s\n", "arm", "total [s]", "fraction", "episodes", "longest [s]",
                "median [s]", "P90 [s]");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "--");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printEpisodes(arms[index].name.c_str(), ensembles[index].flEpisodes, period,
                      ensembles[index].transitions.steps[0]);
    }

    std::printf("\n  the same quantity for the LOW-mu FR channel and for the rear, as context\n");
    std::printf("\n  %-22s %13s %13s %13s\n", "arm", "FL [s]", "FR [s]", "rear [s]");
    std::printf("  %s\n", "--------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("  %-22s", arms[index].name.c_str());
        for (auto channel = std::size_t{0}; channel < brakeChannelCount; channel++)
        {
            std::printf(" %13.3f", static_cast<double>(ensembles[index].transitions.emptySteps[channel]) * period);
        }
        std::printf("\n");
    }

    std::printf("\n  how much of the A -> O FL gap each arm closes\n");
    {
        const auto flOf = [&](const std::size_t index)
        {
            return static_cast<double>(ensembles[index].transitions.emptySteps[0]) * period;
        };

        const auto valueA = flOf(2);
        const auto valueO = flOf(1);
        const auto valueAD = flOf(3);
        const auto span = valueA - valueO;

        std::printf("  A %.3f s, O %.3f s, AD %.3f s. AD closes %.3f s of the %.3f s A -> O gap (%.1f%%).\n", valueA,
                    valueO, valueAD, valueA - valueAD, span, span != 0.0 ? 100.0 * (valueA - valueAD) / span : 0.0);
    }

    std::printf("\n--- 19.13 DOES FL LOWER-BOUND EXPOSURE TRACK YAW? ---\n");
    std::printf("  A DIAGNOSTIC CORRELATE and nothing else. It is not a controller threshold, it is not a\n");
    std::printf("  calibration target, no arm is scored on it, and nothing is fitted to it. The previous\n");
    std::printf("  experiment already falsified the simple relationship: per-member |r| <= 0.32 with the\n");
    std::printf("  sign changing across arms. AD is the arm the deliverable asks for; the rest are context.\n");
    std::printf("\n  %-24s %12s %12s %14s %14s\n", "arm", "Pearson r", "Spearman", "FL mean [s]", "|yaw| mean");
    std::printf("  %s\n", "---------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        std::printf("  %-24s %12.4f %12.4f %14.4f %14.3f\n", arms[index].name.c_str(),
                    pearsonOf(ensembles[index].flLowerBound, ensembles[index].yawMagnitude),
                    spearmanOf(ensembles[index].flLowerBound, ensembles[index].yawMagnitude),
                    distributionOf(ensembles[index].flLowerBound).mean,
                    distributionOf(ensembles[index].yawMagnitude).mean);
    }

    std::printf("\n  the paired member table the correlation is computed on\n");
    std::printf("\n  %-3s | %17s | %17s | %17s | %17s\n", "k", "P  FL s / yaw", "O  FL s / yaw", "A  FL s / yaw",
                "AD FL s / yaw");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "----");
    for (auto member = std::size_t{0}; member < ensembleCount; member++)
    {
        std::printf("  %-3zu |", member);
        for (auto index = std::size_t{0}; index < arms.size(); index++)
        {
            std::printf(" %8.4f %8.3f |", ensembles[index].flLowerBound[member], ensembles[index].yawMagnitude[member]);
        }
        std::printf("\n");
    }

    // --- 19.14 THE REPRESENTATIVE MEMBER ------------------------------------------------------------
    std::printf("\n--- 19.14 REPRESENTATIVE MEMBER, traced through P, A, AD and O ---\n");

    const auto representative = std::size_t{23};

    auto extreme = std::size_t{0};
    for (auto member = std::size_t{1}; member < ensembleCount; member++)
    {
        if (std::abs(residual[member]) > std::abs(residual[extreme]))
        {
            extreme = member;
        }
    }

    const auto residualSpread = distributionOf(
        [&]
        {
            auto values = std::vector<double>{};
            for (const auto value : residual)
            {
                values.push_back(std::abs(value));
            }
            return values;
        }());

    std::printf("  The established split-mu representative is member %zu, entry %.4f m/s. Its paired\n", representative,
                memberEntry(representative, ensembleCount));
    std::printf("  residual is I = %+.3f deg against an ensemble median |I| of %.3f, so it is %s the\n",
                residual[representative], residualSpread.median,
                std::abs(residual[representative]) >= residualSpread.median ? "at or above" : "below");
    std::printf("  typical member for this interaction. The member with the largest |I| is %zu at %+.3f deg,\n",
                extreme, residual[extreme]);
    std::printf("  and it is traced as well so the representative cannot flatter or hide the mechanism.\n");

    auto runs = std::vector<Run>{};
    for (const auto& arm : arms)
    {
        runs.push_back(record(setup.value(), split.value(), arm.assists, calibration, arm.candidate, 1.0,
                              memberEntry(representative, ensembleCount), 0.0, true));
    }

    std::printf("\n  %-24s %10s %10s %12s %12s %12s\n", "arm", "yaw deg", "stop m", "FL empty s", "FL cycles",
                "rear cycles");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");
    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto episodes = lowerBoundEpisodes(runs[index], 0, period);
        std::printf("  %-24s %10.3f %10.3f %12.4f %12zu %12zu\n", arms[index].name.c_str(),
                    runs[index].finalYaw * degrees, runs[index].distance, static_cast<double>(episodes.steps) * period,
                    cyclesOf(runs[index], 0), cyclesOf(runs[index], 2));
    }

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        printWideSplitTrace(assists.antilock, arms[index].name.c_str(), runs[index], 0.20);
    }

    // --- 19.15 THE FIRST A-vs-AD DIVERGENCE ---------------------------------------------------------
    std::printf("\n--- 19.15 FIRST A-vs-AD DIVERGENCE on member %zu: the D decision -> pressure -> wheel\n",
                representative);
    std::printf("    longitudinal force -> left/right Fx imbalance -> yaw rate -> accumulated yaw ---\n\n");
    printPairDivergence(assists.antilock, "A", "AD", runs[2], runs[3]);

    std::printf("\n  the same audit against production and against the control, for orientation\n");
    std::printf("\n  P -> AD\n");
    printPairDivergence(assists.antilock, "P", "AD", runs[0], runs[3]);
    std::printf("\n  P -> A\n");
    printPairDivergence(assists.antilock, "P", "A", runs[0], runs[2]);

    if (extreme != representative)
    {
        std::printf("\n--- 19.16 THE SAME AUDIT ON MEMBER %zu, the largest paired residual ---\n", extreme);

        const auto extremeA = record(setup.value(), split.value(), arms[2].assists, calibration, arms[2].candidate, 1.0,
                                     memberEntry(extreme, ensembleCount), 0.0, true);
        const auto extremeAD = record(setup.value(), split.value(), arms[3].assists, calibration, arms[3].candidate,
                                      1.0, memberEntry(extreme, ensembleCount), 0.0, true);

        std::printf("  entry %.4f m/s, I = %+.3f deg; A final yaw %+.3f deg, AD %+.3f deg.\n\n",
                    memberEntry(extreme, ensembleCount), residual[extreme], extremeA.finalYaw * degrees,
                    extremeAD.finalYaw * degrees);
        printPairDivergence(assists.antilock, "A", "AD", extremeA, extremeAD);
    }

    std::printf("\n  Production is unchanged by this file. `Candidate::replica` is false in arms P and O;\n");
    std::printf("  19.1 proves the four-leg copy reproduces production to the bit and with an identical\n");
    std::printf("  state machine, and proves the loaded D column belongs to this fixture. No threshold,\n");
    std::printf("  gradient, estimator constant or tyre parameter is touched anywhere in it, no controller\n");
    std::printf("  is designed and no observable is added.\n");
}

// =============================================================================================
// SECTION 20 — THE YAW-MOMENT LEDGER, AND WHAT THE SUCCESSFUL WHOLE-LAW-OFF MEMBERS ACTUALLY DO.
// `./EngineTests "[.yaw-ledger]"` — hidden, registers no ctest entry, changes no production file.
//
// **The last forensic section.** Sections 16 to 19 decomposed the slip-aware recovery law into
// four Boolean legs and measured every single and one pair. The decomposition is spent: leg A off
// closes 29% of the production -> whole-law-off median yaw interval, no other single leg closes
// any of it, and the A+D pair adds 3.6 points more while reproducing the control's front-channel
// phase residency almost exactly. Phase population does not explain the control's yaw advantage,
// so the remaining difference is a closed-loop force history and must be measured as one.
//
// TWO ARMS AND NO OTHERS: P (production) and O (`slipAwareRecovery` false). No leg switches, no
// replica, no candidate: `Candidate::replica` is false in both, so both run the production
// control law and O differs from P by one field of the shipped setup.
//
// THREE POPULATIONS: all 29 production members, and the whole-law-off arm's own two clusters as
// section 17 observed them and section 19 re-verified them — 19 members at |yaw| <= 4.17 deg and
// 10 at >= 7.51 deg, with an empty gap between. **The clusters are the existing observed
// membership, taken from this run's own O arm by that stated gap.** Nothing is fitted.
//
// **THE LEDGER IS PROVED BEFORE ANYTHING IS INTERPRETED.** The chassis is integrated on angular
// *momentum* (`RigidBodyState::angularMomentum`, world frame) and `integrate` advances it as
// `L' = L + tau*dt` with `ForceAccumulator::angularDamping` never written by the vehicle, so the
// statement `dL/dt = tau` is exact rather than approximate and there is no gyroscopic term to
// account for. Section 20.1 reconstructs `tau_y` from the step's own forces and geometry and
// compares it against the measured `dL_y` tick by tick. If the ledger does not close, nothing
// below it is read.
//
// THE COMPLETE INVENTORY OF WHAT PUTS A YAW MOMENT ON THIS CHASSIS, read out of `stepVehicle`'s
// force pass rather than assumed:
//
//   1. gravity, at the centre of mass                      — no moment at all
//   2. `tireVertical * patch.normal` at the contact point   — vertical, so no yaw on a level plate
//   3. the unsprung mass's inertia reaction at the wheel centre — **has a yaw component on this
//      car**, because the Golf states `geometricLoadPath` and the reaction then points along the
//      wheel centre's own direction of travel rather than along world up
//   4. `tyre.longitudinal * contact.forward + tyre.lateral * contact.lateral` at the patch centre
//      — the term this section exists to decompose
//   5. `tyre.aligningMoment * patch.normal`, a couple
//   6. aero drag and lift at each surface's own point
//   7. the driveline reaction `-(I_wheel * alpha_wheel)` about the spin axis — **also on for this
//      car**, `drivelineReaction = true` since 2026-08-27
//   8. `resolveContacts`, bodywork against the world — counted, and zero on a bare grip plate
//
// Items 3 and 7 are the two the naive "FL Fx minus FR Fx" reading leaves out, and both are live
// on the Golf. Item 7 is recomputed from the wheel speeds rather than from the torques, because
// `stepVehicle` advances the spin as `(roadTorque + drive)/I*dt` and then clamps it with the brake:
// the net is exactly `I*(w' - w)/dt`, which the recorder has on both sides of the call.
//
// Nothing here designs a controller, changes a threshold, adds a production observable or touches
// `RaceEngine/Assists`. The fixture, the ensemble, the surface, the settle, the pedal, the
// stopping condition and the percentile definition are section 18's, used unchanged.
// =============================================================================================
namespace
{

// --- the windows, stated and not tuned ---------------------------------------------------------
constexpr auto windowCount = std::size_t{6};
constexpr auto windowEdges = std::array<double, 5>{0.25, 0.50, 1.00, 2.00, 3.00};

[[nodiscard]] std::size_t windowOf(const double when)
{
    auto slot = std::size_t{0};

    while (slot < windowEdges.size() && when > windowEdges[slot])
    {
        slot++;
    }

    return slot;
}

[[nodiscard]] const char* windowName(const std::size_t slot)
{
    static constexpr auto names = std::array<const char*, windowCount>{"0.00-0.25 s", "0.25-0.50 s", "0.50-1.00 s",
                                                                       "1.00-2.00 s", "2.00-3.00 s", "3.00 s-stop"};

    return names[slot];
}

// The four instants the deliverable asks the cluster question at.
constexpr auto probeCount = std::size_t{4};
constexpr auto probeTimes = std::array<double, probeCount>{0.25, 0.50, 1.00, 2.00};

// The cumulative curves, on a stated 50 ms grid out to 4 s.
constexpr auto curveStep = 0.05;
constexpr auto curvePoints = std::size_t{81};

// The whole-law-off cluster boundary, as section 17 observed it and section 19 re-printed it: an
// empty band between 4.17 and 7.51 deg. Stated here, applied to the O arm only, and asserted to
// reproduce the 19/10 split — if the gap has moved, the split this section analyses is not the
// one the evidence was built on and the run says so.
constexpr auto clusterFloor = 4.17;
constexpr auto clusterCeiling = 7.51;
constexpr auto lowClusterCount = std::size_t{19};
constexpr auto highClusterCount = std::size_t{10};

struct MomentMember
{
    std::size_t index = 0;
    double entry = 0.0;
    double finalYaw = 0.0; // degrees, signed
    double distance = 0.0;
    double stopTime = 0.0;
    bool stopped = false;

    // --- windowed yaw impulses, N.m.s, signed ---
    std::array<double, windowCount> jLong{};  // from tyre.longitudinal * contact.forward
    std::array<double, windowCount> jLat{};   // from tyre.lateral * contact.lateral
    std::array<double, windowCount> jAlign{}; // the aligning-moment couples
    std::array<double, windowCount> jOther{}; // vertical + unsprung + aero + driveline
    std::array<double, windowCount> jTotal{}; // every term, which is dL_y over the window

    // --- windowed force impulses, N.s, signed ---
    std::array<double, windowCount> flLong{};
    std::array<double, windowCount> frLong{};
    std::array<double, windowCount> frontLat{};
    std::array<double, windowCount> rearLong{};
    std::array<double, windowCount> rearLat{};

    // --- what the body did over the window ---
    std::array<double, windowCount> yawRateChange{};
    std::array<double, windowCount> yawChange{};
    std::array<double, windowCount> seconds{};

    // --- cumulative curves on the 50 ms grid ---
    std::array<double, curvePoints> cumLong{};
    std::array<double, curvePoints> cumLat{};
    std::array<double, curvePoints> cumTotal{};
    std::array<double, curvePoints> curveYaw{};
    std::array<double, curvePoints> curveYawRate{};
    std::array<double, curvePoints> curveImbalance{};

    // --- snapshots at the four probe instants ---
    std::array<double, probeCount> atCumLong{};
    std::array<double, probeCount> atCumLat{};
    std::array<double, probeCount> atCumTotal{};
    std::array<double, probeCount> atYaw{};
    std::array<double, probeCount> atYawRate{};
    std::array<double, probeCount> atFlFx{};
    std::array<double, probeCount> atFrFx{};
    std::array<double, probeCount> atFlSlip{};
    std::array<double, probeCount> atFrSlip{};
    std::array<double, probeCount> atFlBar{};
    std::array<double, probeCount> atFrBar{};
    std::array<double, probeCount> atFlImpulse{};
    std::array<double, probeCount> atFrImpulse{};
    std::array<std::size_t, probeCount> atFlPhase{};
    std::array<std::size_t, probeCount> atFrPhase{};
    std::array<double, probeCount> atSensorAge{};

    // --- the ledger ---
    double measuredDelta = 0.0;
    double reconstructedDelta = 0.0;
    double worstTickResidual = 0.0;
    double worstTickTorque = 0.0;
    double sumAbsTorqueImpulse = 0.0;
    std::size_t bodyContacts = 0;
    double worstNormalTilt = 0.0;

    // --- per-term totals over the whole stop, N.m.s ---
    double termLong = 0.0;
    double termLat = 0.0;
    double termVertical = 0.0;
    double termAlign = 0.0;
    double termUnsprung = 0.0;
    double termAero = 0.0;
    double termDriveline = 0.0;

    // --- timing ---
    double centroidAbs = 0.0;
    double centroidSigned = 0.0;
    double absImpulse = 0.0;
    bool signedCentroidUsable = false;
};

// The recorder. **A copy of `record`'s loop and nothing else**, with the momentum ledger taken
// around the `stepVehicle` call. Every decision, every argument and the settle above it are
// character for character the same, and 20.1 proves the two produce the same run.
[[nodiscard]] MomentMember recordLedger(const VehicleSetup& setup, const PhysicsWorld& world,
                                        const AssistSetup& assists, const RoadTorqueCalibration& calibration,
                                        const double pedal, const double entrySpeed, const std::size_t index)
{
    auto state = VehicleState{};
    settle(setup, state, world, entrySpeed);

    auto shadow = ShadowState{};
    auto lastStep = VehicleStep{};
    auto truth = Truth{};
    const auto passive = Candidate{};
    const auto candidate = Candidate{};

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
        const auto command = shadowUpdate(assists, shadow, sense(), 0.0, noBrakePressure, tick, calibration, passive,
                                          truth, nullptr, 0.0);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    truth.staticRearLoad =
        std::max(1.0, 0.5 * (lastStep.corners[2].forces.tireVertical + lastStep.corners[3].forces.tireVertical));

    shadow.pastBandSteps = 0;
    shadow.admittedSteps = 0;
    shadow.qualifiedSteps = 0;
    shadow.effectiveSteps = 0;

    auto member = MomentMember{};
    member.index = index;
    member.entry = entrySpeed;

    const auto start = state.chassis.position;

    auto input = VehicleInput{};
    input.brake = pedal;
    input.steering = 0.0;

    const auto pressures = brakeCircuitPressures(setup, pedal);

    auto steps = std::vector<Step>{};
    steps.reserve(4096);

    auto elapsed = 0.0;
    auto cumulativeLong = 0.0;
    auto cumulativeLat = 0.0;
    auto cumulativeTotal = 0.0;
    auto cumulativeFl = 0.0;
    auto cumulativeFr = 0.0;
    auto centroidNumerator = 0.0;
    auto signedNumerator = 0.0;
    auto signedDenominator = 0.0;
    auto nextCurve = std::size_t{0};
    auto nextProbe = std::size_t{0};
    auto previousYawRate = 0.0;
    auto previousYaw = 0.0;
    auto windowRateStart = std::array<double, windowCount>{};
    auto windowRateEnd = std::array<double, windowCount>{};
    auto windowYawStart = std::array<double, windowCount>{};
    auto windowYawEnd = std::array<double, windowCount>{};
    auto windowSeen = std::array<bool, windowCount>{};
    const auto startMomentum = state.chassis.angularMomentum.y;

    for (auto step = 0; step < 360 * 30; step++)
    {
        for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
        {
            const auto& solution = lastStep.corners[corner];

            truth.contact[corner] = lastStep.telemetry.wheels[corner].inContact;
            truth.load[corner] = solution.forces.tireVertical;
            truth.longitudinal[corner] = solution.contact.tyre.longitudinal;
            truth.effectiveRadius[corner] = solution.contact.effectiveRadius;
            truth.slipRatio[corner] = solution.contact.slip.slipRatio;
        }

        const auto brakes = shadowUpdate(assists, shadow, sense(), pedal, pressures, tick, calibration, candidate,
                                         truth, &steps, elapsed);

        // --- the two sides of the ledger, taken around the call ---------------------------------
        const auto before = state.chassis;
        auto beforeSpin = std::array<double, cornerCount>{};
        for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
        {
            beforeSpin[corner] = state.corners[corner].wheelSpeed;
        }

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        const auto measured = state.chassis.angularMomentum.y - before.angularMomentum.y;

        auto termLong = 0.0;
        auto termLat = 0.0;
        auto termVertical = 0.0;
        auto termAlign = 0.0;
        auto termUnsprung = 0.0;
        auto termDriveline = 0.0;

        for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
        {
            const auto& solution = lastStep.corners[corner];
            const auto& suspension = solution.suspension;
            const auto& setupCorner = setup.corners[corner];

            const auto wheelWorld = raceengine::bodyToWorld(before, suspension.wheelCentre);
            const auto contactPoint = solution.patch.inContact ? solution.patch.centre : wheelWorld;
            const auto patchArm = solution.patch.centre - before.position;
            const auto contactArm = contactPoint - before.position;
            const auto wheelArm = wheelWorld - before.position;

            // The yaw component of `r x F` and nothing else, so the split is exactly the split the
            // force pass applies: one term per force vector, at the point that force acts.
            const auto yawOf = [](const glm::dvec3& arm, const glm::dvec3& force)
            {
                return arm.z * force.x - arm.x * force.z;
            };

            termLong += yawOf(patchArm, solution.contact.tyre.longitudinal * solution.contact.forward);
            termLat += yawOf(patchArm, solution.contact.tyre.lateral * solution.contact.lateral);
            termVertical += yawOf(contactArm, solution.forces.tireVertical * solution.patch.normal);
            termAlign += (solution.contact.tyre.aligningMoment * solution.patch.normal).y;

            const auto acceleration = solution.generalisedForce / solution.generalisedInertia;

            if (setup.geometricLoadPath)
            {
                termUnsprung += yawOf(wheelArm, before.orientation * (-setupCorner.unsprungMass * acceleration *
                                                                      suspension.wheelCentrePerAngle));
            }
            else
            {
                termUnsprung +=
                    yawOf(wheelArm,
                          glm::dvec3(0.0, -setupCorner.unsprungMass * suspension.travelPerAngle * acceleration, 0.0));
            }

            if (setup.drivelineReaction)
            {
                const auto spinAxis = before.orientation * (suspension.uprightOrientation * glm::dvec3(1.0, 0.0, 0.0));
                const auto net =
                    setupCorner.wheelInertia * (state.corners[corner].wheelSpeed - beforeSpin[corner]) / tick;

                termDriveline += (-net * spinAxis).y;
            }

            member.worstNormalTilt = std::max(
                member.worstNormalTilt, std::max(std::abs(solution.patch.normal.x), std::abs(solution.patch.normal.z)));
        }

        auto termAero = 0.0;
        const auto airspeed = glm::length(before.linearVelocity);
        if (airspeed > 1e-6)
        {
            const auto flow = before.linearVelocity / airspeed;
            const auto pressure = 0.5 * setup.airDensity * airspeed * airspeed;

            for (const auto& surface : setup.aero)
            {
                const auto force =
                    -pressure * surface.dragArea * flow + glm::dvec3(0.0, pressure * surface.liftArea, 0.0);
                const auto arm = raceengine::bodyToWorld(before, surface.centre) - before.position;

                termAero += arm.z * force.x - arm.x * force.z;
            }
        }

        const auto reconstructed =
            (termLong + termLat + termVertical + termAlign + termUnsprung + termAero + termDriveline) * tick;

        member.measuredDelta += measured;
        member.reconstructedDelta += reconstructed;
        member.worstTickResidual = std::max(member.worstTickResidual, std::abs(measured - reconstructed));
        member.worstTickTorque = std::max(member.worstTickTorque, std::abs(measured) / tick);
        member.sumAbsTorqueImpulse += std::abs(measured);
        member.bodyContacts += lastStep.contacts.points.size();

        member.termLong += termLong * tick;
        member.termLat += termLat * tick;
        member.termVertical += termVertical * tick;
        member.termAlign += termAlign * tick;
        member.termUnsprung += termUnsprung * tick;
        member.termAero += termAero * tick;
        member.termDriveline += termDriveline * tick;

        elapsed += tick;

        const auto slot = windowOf(elapsed);
        const auto yaw = lastStep.telemetry.yaw * degrees;
        const auto yawRate = lastStep.telemetry.yawRate;

        if (!windowSeen[slot])
        {
            windowSeen[slot] = true;
            windowRateStart[slot] = previousYawRate;
            windowYawStart[slot] = previousYaw;
        }
        windowRateEnd[slot] = yawRate;
        windowYawEnd[slot] = yaw;
        member.seconds[slot] += tick;

        member.jLong[slot] += termLong * tick;
        member.jLat[slot] += termLat * tick;
        member.jAlign[slot] += termAlign * tick;
        member.jOther[slot] += (termVertical + termUnsprung + termAero + termDriveline) * tick;
        member.jTotal[slot] += measured;

        const auto flFx = lastStep.corners[0].contact.tyre.longitudinal;
        const auto frFx = lastStep.corners[1].contact.tyre.longitudinal;

        member.flLong[slot] += flFx * tick;
        member.frLong[slot] += frFx * tick;
        member.frontLat[slot] +=
            (lastStep.corners[0].contact.tyre.lateral + lastStep.corners[1].contact.tyre.lateral) * tick;
        member.rearLong[slot] +=
            (lastStep.corners[2].contact.tyre.longitudinal + lastStep.corners[3].contact.tyre.longitudinal) * tick;
        member.rearLat[slot] +=
            (lastStep.corners[2].contact.tyre.lateral + lastStep.corners[3].contact.tyre.lateral) * tick;

        cumulativeLong += termLong * tick;
        cumulativeLat += termLat * tick;
        cumulativeTotal += measured;
        cumulativeFl += flFx * tick;
        cumulativeFr += frFx * tick;

        centroidNumerator += elapsed * std::abs(measured);
        member.absImpulse += std::abs(measured);
        signedNumerator += elapsed * measured;
        signedDenominator += measured;

        while (nextCurve < curvePoints && elapsed + 1.0e-9 >= static_cast<double>(nextCurve) * curveStep)
        {
            member.cumLong[nextCurve] = cumulativeLong;
            member.cumLat[nextCurve] = cumulativeLat;
            member.cumTotal[nextCurve] = cumulativeTotal;
            member.curveYaw[nextCurve] = yaw;
            member.curveYawRate[nextCurve] = yawRate;
            member.curveImbalance[nextCurve] = flFx - frFx;
            nextCurve++;
        }

        while (nextProbe < probeCount && elapsed + 1.0e-9 >= probeTimes[nextProbe])
        {
            member.atCumLong[nextProbe] = cumulativeLong;
            member.atCumLat[nextProbe] = cumulativeLat;
            member.atCumTotal[nextProbe] = cumulativeTotal;
            member.atYaw[nextProbe] = yaw;
            member.atYawRate[nextProbe] = yawRate;
            member.atFlFx[nextProbe] = flFx;
            member.atFrFx[nextProbe] = frFx;
            member.atFlSlip[nextProbe] = lastStep.corners[0].contact.slip.slipRatio;
            member.atFrSlip[nextProbe] = lastStep.corners[1].contact.slip.slipRatio;
            member.atFlBar[nextProbe] = shadow.assists.pressure[0] / bar;
            member.atFrBar[nextProbe] = shadow.assists.pressure[1] / bar;
            member.atFlImpulse[nextProbe] = cumulativeFl;
            member.atFrImpulse[nextProbe] = cumulativeFr;
            member.atFlPhase[nextProbe] = phaseIndex(shadow.phase[0]);
            member.atFrPhase[nextProbe] = phaseIndex(shadow.phase[1]);
            nextProbe++;
        }

        previousYawRate = yawRate;
        previousYaw = yaw;

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            member.stopped = true;
            break;
        }
    }

    member.stopTime = elapsed;
    member.distance = state.chassis.position.z - start.z;
    member.finalYaw = lastStep.telemetry.yaw * degrees;
    member.measuredDelta = state.chassis.angularMomentum.y - startMomentum;

    for (auto slot = std::size_t{0}; slot < windowCount; slot++)
    {
        member.yawRateChange[slot] = windowSeen[slot] ? windowRateEnd[slot] - windowRateStart[slot] : 0.0;
        member.yawChange[slot] = windowSeen[slot] ? windowYawEnd[slot] - windowYawStart[slot] : 0.0;
    }

    while (nextCurve < curvePoints)
    {
        member.cumLong[nextCurve] = cumulativeLong;
        member.cumLat[nextCurve] = cumulativeLat;
        member.cumTotal[nextCurve] = cumulativeTotal;
        member.curveYaw[nextCurve] = previousYaw;
        member.curveYawRate[nextCurve] = previousYawRate;
        member.curveImbalance[nextCurve] = 0.0;
        nextCurve++;
    }

    member.centroidAbs = member.absImpulse > 0.0 ? centroidNumerator / member.absImpulse : 0.0;
    member.centroidSigned = signedDenominator != 0.0 ? signedNumerator / signedDenominator : 0.0;
    // A signed centroid is only readable when the signed total is large against the absolute one.
    // Below that the denominator is a difference of two big cancelling numbers and the quotient is
    // arbitrary, so it is reported and marked rather than used.
    member.signedCentroidUsable = member.absImpulse > 0.0 && std::abs(signedDenominator) > 0.25 * member.absImpulse;

    // The sensor age around the divergence window, from the channel steps this run already
    // recorded. Front channels only, and the mean over the 100 ms bracketing each probe instant.
    for (auto probe = std::size_t{0}; probe < probeCount; probe++)
    {
        auto total = 0.0;
        auto count = std::size_t{0};

        for (const auto& one : steps)
        {
            if (one.channel > 1 || one.time < probeTimes[probe] - 0.05 || one.time > probeTimes[probe] + 0.05)
            {
                continue;
            }

            total += one.sensorAge;
            count++;
        }

        member.atSensorAge[probe] = count > 0 ? total / static_cast<double>(count) : 0.0;
    }

    return member;
}

// --- the three populations, and the small amount of statistics they need ------------------------
struct Populations
{
    std::vector<const MomentMember*> production;
    std::vector<const MomentMember*> low;
    std::vector<const MomentMember*> high;
};

using Selector = double (*)(const MomentMember&, std::size_t);

[[nodiscard]] std::vector<double> gather(const std::vector<const MomentMember*>& group, const Selector field,
                                         const std::size_t slot)
{
    auto values = std::vector<double>{};
    values.reserve(group.size());

    for (const auto* member : group)
    {
        values.push_back(field(*member, slot));
    }

    return values;
}

// The rank-based area under the curve for "does this quantity separate the high cluster from the
// low one". Ties score a half, so a constant quantity reads 0.5 rather than 1.0. **A description
// of overlap and not a classifier**: no threshold is chosen anywhere, and nothing downstream
// consumes it.
[[nodiscard]] double aucOf(const std::vector<double>& low, const std::vector<double>& high)
{
    if (low.empty() || high.empty())
    {
        return 0.5;
    }

    auto total = 0.0;

    for (const auto right : high)
    {
        for (const auto left : low)
        {
            total += right > left ? 1.0 : (right == left ? 0.5 : 0.0);
        }
    }

    return total / (static_cast<double>(low.size()) * static_cast<double>(high.size()));
}

// How many members of one group fall inside the other group's closed range — the plainest
// statement of overlap there is, and it does not depend on a rank convention.
[[nodiscard]] std::size_t insideRange(const std::vector<double>& outer, const std::vector<double>& inner)
{
    if (outer.empty())
    {
        return 0;
    }

    const auto low = *std::min_element(outer.begin(), outer.end());
    const auto high = *std::max_element(outer.begin(), outer.end());

    auto count = std::size_t{0};
    for (const auto value : inner)
    {
        count += (value >= low && value <= high) ? 1 : 0;
    }

    return count;
}

void printGroupRow(const char* name, const std::vector<double>& values)
{
    const auto spread = distributionOf(values);

    std::printf("  %-22s %4zu %11.3f %11.3f %11.3f %11.3f %11.3f\n", name, spread.n, spread.minimum, spread.p10,
                spread.median, spread.p90, spread.maximum);
}

void printGroupHeader(const char* what)
{
    std::printf("\n  %-22s %4s %11s %11s %11s %11s %11s\n", what, "n", "min", "P10", "MEDIAN", "P90", "max");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------");
}

// One quantity, all six windows, all three populations.
void printWindowed(const char* what, const char* units, const Populations& groups, const Selector field)
{
    std::printf("\n  %s [%s]\n", what, units);
    std::printf("\n  %-13s %-9s %4s %11s %11s %11s %11s %11s\n", "window", "group", "n", "min", "P10", "MEDIAN", "P90",
                "max");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "------");

    for (auto slot = std::size_t{0}; slot < windowCount; slot++)
    {
        const auto rows = std::array<std::pair<const char*, const std::vector<const MomentMember*>*>, 3>{
            std::pair{"P", &groups.production}, std::pair{"O-LOW", &groups.low}, std::pair{"O-HIGH", &groups.high}};

        const auto* label = windowName(slot);

        for (const auto& row : rows)
        {
            const auto spread = distributionOf(gather(*row.second, field, slot));
            std::printf("  %-13s %-9s %4zu %11.3f %11.3f %11.3f %11.3f %11.3f\n", label, row.first, spread.n,
                        spread.minimum, spread.p10, spread.median, spread.p90, spread.maximum);
            label = "";
        }
    }
}

// One quantity at the four probe instants, with the separation statistics beside it.
void printProbed(const char* what, const char* units, const Populations& groups, const Selector field)
{
    std::printf("\n  %s [%s]\n", what, units);
    std::printf("\n  %-8s %11s %11s %11s %11s %9s %11s %11s\n", "at", "P med", "O-LOW med", "O-HIGH med", "L-H gap",
                "AUC", "H in L range", "L in H range");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------");

    for (auto probe = std::size_t{0}; probe < probeCount; probe++)
    {
        const auto lowValues = gather(groups.low, field, probe);
        const auto highValues = gather(groups.high, field, probe);
        const auto lowSpread = distributionOf(lowValues);
        const auto highSpread = distributionOf(highValues);

        std::printf("  %6.2f s %11.3f %11.3f %11.3f %11.3f %9.3f %11zu %11zu\n", probeTimes[probe],
                    distributionOf(gather(groups.production, field, probe)).median, lowSpread.median, highSpread.median,
                    lowSpread.median - highSpread.median, aucOf(lowValues, highValues),
                    insideRange(lowValues, highValues), insideRange(highValues, lowValues));
    }
}

// The group median of a cumulative curve, printed on a 0.1 s grid with the count of members still
// running at each time beside it, so a curve is never read past the point where members have left.
void printCurveField(const char* what, const char* units, const Populations& groups,
                     const std::array<double, curvePoints> MomentMember::* curve)
{
    std::printf("\n  %s [%s], group medians on the stated 50 ms grid, printed every 0.2 s\n", what, units);
    std::printf("\n  %-8s %12s %12s %12s %12s %10s %10s %10s\n", "t [s]", "P", "O-LOW", "O-HIGH", "LOW-HIGH", "P live",
                "LOW live", "HIGH live");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "----------");

    for (auto point = std::size_t{0}; point < curvePoints; point += 4)
    {
        const auto when = static_cast<double>(point) * curveStep;

        const auto medianOf = [&](const std::vector<const MomentMember*>& group)
        {
            auto values = std::vector<double>{};
            for (const auto* member : group)
            {
                values.push_back((member->*curve)[point]);
            }
            return distributionOf(values).median;
        };

        const auto liveOf = [&](const std::vector<const MomentMember*>& group)
        {
            auto count = std::size_t{0};
            for (const auto* member : group)
            {
                count += member->stopTime >= when ? 1 : 0;
            }
            return count;
        };

        std::printf("  %8.2f %12.3f %12.3f %12.3f %12.3f %10zu %10zu %10zu\n", when, medianOf(groups.production),
                    medianOf(groups.low), medianOf(groups.high), medianOf(groups.low) - medianOf(groups.high),
                    liveOf(groups.production), liveOf(groups.low), liveOf(groups.high));
    }
}

} // namespace

TEST_CASE("what force history separates the successful whole-law-off split-mu members", "[.yaw-ledger]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    const auto assists = withAntilock(setup.value());
    const auto control = withoutSlipAwareRecovery(assists);
    const auto calibration = calibrationOf(setup.value(), assists);

    REQUIRE_FALSE(assists.antilock.yawMomentDelay);
    REQUIRE(assists.antilock.recoveryAuthority != RecoveryAuthority::Disabled);
    REQUIRE(control.antilock.recoveryAuthority == RecoveryAuthority::Disabled);

    std::printf("\n=== SECTION 20: THE YAW-MOMENT LEDGER ON THE SPLIT-MU n = %zu ENSEMBLE ===\n", ensembleCount);
    std::printf("  Two arms and no others: P (production) and O (the whole slip-aware recovery law off).\n");
    std::printf("  %zu deterministic members, entry %.4f to %.4f m/s, full pedal, no steering, grip 1.00 at\n",
                ensembleCount, memberEntry(0, ensembleCount), memberEntry(ensembleCount - 1, ensembleCount));
    std::printf("  x<0 and 0.35 at x>0. **Section 20A MEASURES which corner stands on which surface**\n");
    std::printf("  and the answer is that corner 0 (FL) is the LOW-mu wheel at x = +0.77 and corner 1 (FR)\n");
    std::printf("  is the HIGH-mu one at x = -0.77, because `outboardSign` puts the car's LEFT at +x. The\n");
    std::printf("  channel names are corner indices and carry no statement about grip; every mu label below\n");
    std::printf("  comes from that measurement. **Sections 17 to 19 printed the opposite label**, which\n");
    std::printf("  changes none of their numbers and inverts every sentence that named a surface.\n");

    // --- 20.0 THE FRAME AND THE SIGNS, READ OFF THE CAR RATHER THAN ASSUMED ----------------------
    std::printf("\n--- 20.0 THE FRAME AND THE SIGN CONVENTION, verified against the implementation ---\n");
    std::printf("  The chassis carries angular MOMENTUM in the world frame and `integrate` advances it as\n");
    std::printf("  L' = L + tau*dt. `ForceAccumulator::angularDamping` is never written by `stepVehicle`, so\n");
    std::printf("  that update is exact and there is no gyroscopic term: dL/dt = tau, to the bit. The yaw\n");
    std::printf("  channel is therefore the WORLD y component of L, not the body-frame yaw rate telemetry\n");
    std::printf("  reports, and the two are compared below rather than assumed equal.\n");
    std::printf("  `TelemetryFrame::yaw` is atan2(forward.x, forward.z), a heading and not an integral, so\n");
    std::printf("  a POSITIVE yaw is a rotation about world +y and turns the car toward +x, which is the\n");
    std::printf("  car's LEFT and is the LOW-mu side. Every member of every arm ends NEGATIVE, the car\n");
    std::printf("  pulling toward the HIGH-mu side. That is asserted rather than stated below.\n");

    // --- the two ensembles -----------------------------------------------------------------------
    auto productionMembers = std::vector<MomentMember>{};
    auto controlMembers = std::vector<MomentMember>{};
    productionMembers.reserve(ensembleCount);
    controlMembers.reserve(ensembleCount);

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);

        productionMembers.push_back(
            recordLedger(setup.value(), split.value(), assists, calibration, 1.0, entry, index));
        controlMembers.push_back(recordLedger(setup.value(), split.value(), control, calibration, 1.0, entry, index));
    }

    // --- 20.1 THE LEDGER --------------------------------------------------------------------------
    std::printf("\n--- 20.1 THE YAW-MOMENT LEDGER: reconstructed torque against measured momentum ---\n");
    std::printf("  Per controller-free physics tick, over every tick of all %zu runs. The left side is\n",
                2 * ensembleCount);
    std::printf("  L_y(after) - L_y(before) taken around `stepVehicle`. The right side is the yaw component\n");
    std::printf("  of every force and couple the force pass applies, times dt, reconstructed from that\n");
    std::printf("  tick's own step and the chassis state it was computed at. The eight contributors are\n");
    std::printf("  listed in this section's header and none is left out.\n");

    const auto reportLedger = [](const char* name, const std::vector<MomentMember>& members)
    {
        auto worstTick = 0.0;
        auto worstRun = 0.0;
        auto worstRelative = 0.0;
        auto contacts = std::size_t{0};
        auto worstTilt = 0.0;
        auto worstTorque = 0.0;

        for (const auto& member : members)
        {
            worstTick = std::max(worstTick, member.worstTickResidual);
            worstRun = std::max(worstRun, std::abs(member.measuredDelta - member.reconstructedDelta));
            worstTorque = std::max(worstTorque, member.worstTickTorque);
            contacts += member.bodyContacts;
            worstTilt = std::max(worstTilt, member.worstNormalTilt);

            if (member.sumAbsTorqueImpulse > 0.0)
            {
                worstRelative = std::max(worstRelative, std::abs(member.measuredDelta - member.reconstructedDelta) /
                                                            member.sumAbsTorqueImpulse);
            }
        }

        std::printf("  %-14s worst tick residual %.3e N.m.s, worst whole-run residual %.3e N.m.s\n", name, worstTick,
                    worstRun);
        std::printf("  %-14s worst run residual as a fraction of that run's total |impulse|: %.3e\n", "",
                    worstRelative);
        std::printf("  %-14s largest tick yaw torque seen %.1f N.m; bodywork contact points %zu;\n", "", worstTorque,
                    contacts);
        std::printf("  %-14s largest |patch normal| horizontal component %.3e (a level plate reads 0)\n", "",
                    worstTilt);

        return std::array<double, 3>{worstTick, worstRun, worstRelative};
    };

    const auto productionLedger = reportLedger("P production", productionMembers);
    const auto controlLedger = reportLedger("O law OFF", controlMembers);

    // The ledger must close to floating-point noise, not to a tolerance somebody chose. The scale
    // is the largest single-tick yaw torque the car ever sees times dt, which is order 10 N.m.s;
    // a residual eight orders below that is the arithmetic and nothing else.
    const auto ledgerLimit = 1.0e-6;
    const auto closed = productionLedger[0] < ledgerLimit && controlLedger[0] < ledgerLimit &&
                        productionLedger[1] < ledgerLimit && controlLedger[1] < ledgerLimit;

    std::printf("\n  LEDGER %s (limit %.1e N.m.s on both the per-tick and the whole-run residual)\n",
                closed ? "CLOSES" : "*** DOES NOT CLOSE — NOTHING BELOW MAY BE INTERPRETED ***", ledgerLimit);
    REQUIRE(closed);

    std::printf("\n  WHAT EACH CONTRIBUTOR IS WORTH over a whole stop, N.m.s, ensemble median of the signed\n");
    std::printf("  total. This is the inventory, and it is what says the naive FL-minus-FR reading is or is\n");
    std::printf("  not the whole story on this car.\n");
    std::printf("\n  %-16s %13s %13s %13s %13s %13s %13s %13s\n", "arm", "longitudinal", "lateral", "vertical",
                "aligning", "unsprung", "aero", "driveline");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "---------------------------");

    const auto termRow = [](const char* name, const std::vector<MomentMember>& members)
    {
        const auto columnOfTerm = [&](double MomentMember::* field)
        {
            auto values = std::vector<double>{};
            for (const auto& member : members)
            {
                values.push_back(member.*field);
            }
            return distributionOf(values).median;
        };

        std::printf("  %-16s %13.3f %13.3f %13.3f %13.3f %13.3f %13.3f %13.3f\n", name,
                    columnOfTerm(&MomentMember::termLong), columnOfTerm(&MomentMember::termLat),
                    columnOfTerm(&MomentMember::termVertical), columnOfTerm(&MomentMember::termAlign),
                    columnOfTerm(&MomentMember::termUnsprung), columnOfTerm(&MomentMember::termAero),
                    columnOfTerm(&MomentMember::termDriveline));
    };

    termRow("P production", productionMembers);
    termRow("O law OFF", controlMembers);

    std::printf("\n  This car states `geometricLoadPath` and `drivelineReaction` both TRUE, so the unsprung\n");
    std::printf("  and driveline columns are live rather than structurally zero, and the two of them are\n");
    std::printf("  exactly what a `FL Fx - FR Fx` reading would have thrown away.\n");

    // --- the recorder is a copy of `record`, and it is proved -------------------------------------
    std::printf("\n  THE RECORDER IS A COPY OF `record` AND IS PROVED AGAINST IT. The ledger is taken around\n");
    std::printf("  the same `stepVehicle` call in the same loop with the same arguments; the run must be\n");
    std::printf("  the same run to the bit or the histories below belong to a different car.\n");

    auto faithful = true;
    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);
        const auto reference =
            record(setup.value(), split.value(), assists, calibration, Candidate{}, 1.0, entry, 0.0, false);

        faithful = faithful && reference.finalYaw * degrees == productionMembers[index].finalYaw &&
                   reference.distance == productionMembers[index].distance &&
                   reference.stopped == productionMembers[index].stopped;
    }

    std::printf("  P arm, %zu members, final yaw / distance / stopped against `record`: %s\n", ensembleCount,
                faithful ? "BIT-IDENTICAL" : "*** DIFFERENT ***");
    REQUIRE(faithful);

    // --- 20.2 THE CLUSTERS ------------------------------------------------------------------------
    std::printf("\n--- 20.2 CLUSTER STRUCTURE ---\n");
    std::printf("  The clusters are the EXISTING observed membership: section 17 found an empty band in the\n");
    std::printf("  whole-law-off arm between %.2f and %.2f deg with %zu members below and %zu above, and\n",
                clusterFloor, clusterCeiling, lowClusterCount, highClusterCount);
    std::printf("  section 19 reproduced it. The band is applied here unchanged. No model is fitted, and if\n");
    std::printf("  the split does not come out %zu/%zu this run the analysis is not the one the evidence\n",
                lowClusterCount, highClusterCount);
    std::printf("  was built on.\n");

    auto groups = Populations{};
    auto negativeYaw = true;

    for (const auto& member : productionMembers)
    {
        groups.production.push_back(&member);
        negativeYaw = negativeYaw && member.finalYaw < 0.0;
    }

    auto inBand = std::size_t{0};
    for (const auto& member : controlMembers)
    {
        const auto magnitude = std::abs(member.finalYaw);
        negativeYaw = negativeYaw && member.finalYaw < 0.0;

        if (magnitude <= clusterFloor)
        {
            groups.low.push_back(&member);
        }
        else if (magnitude >= clusterCeiling)
        {
            groups.high.push_back(&member);
        }
        else
        {
            inBand++;
        }
    }

    std::printf("\n  O-LOW %zu members, O-HIGH %zu members, inside the empty band %zu — %s\n", groups.low.size(),
                groups.high.size(), inBand,
                (groups.low.size() == lowClusterCount && groups.high.size() == highClusterCount && inBand == 0)
                    ? "THE EXISTING SPLIT"
                    : "*** NOT THE EXISTING SPLIT ***");
    REQUIRE(groups.low.size() == lowClusterCount);
    REQUIRE(groups.high.size() == highClusterCount);
    REQUIRE(inBand == 0);

    std::printf("  every member of both arms ends with NEGATIVE yaw (toward the HIGH-mu side): %s\n",
                negativeYaw ? "TRUE" : "*** FALSE ***");
    REQUIRE(negativeYaw);

    std::printf("\n  CLUSTER MEMBERSHIP AGAINST ENTRY SPEED — is the cluster a speed band or a bifurcation?\n");
    std::printf("\n  %-3s %11s %11s %11s %8s %11s %11s\n", "k", "entry m/s", "P yaw", "O yaw", "cluster", "P stop",
                "O stop");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");

    auto runs = std::size_t{0};
    auto previousLow = -1;

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto isLow = std::abs(controlMembers[index].finalYaw) <= clusterFloor;
        const auto flag = isLow ? 1 : 0;

        runs += (previousLow >= 0 && flag != previousLow) ? 1 : 0;
        previousLow = flag;

        std::printf("  %-3zu %11.4f %11.3f %11.3f %8s %11.3f %11.3f\n", index, productionMembers[index].entry,
                    productionMembers[index].finalYaw, controlMembers[index].finalYaw, isLow ? "LOW" : "HIGH",
                    productionMembers[index].distance, controlMembers[index].distance);
    }

    std::printf("\n  contiguous runs of one cluster label across the sweep: %zu (2 would be two clean bands,\n",
                runs + 1);
    std::printf("  %zu is the most interleaved the sweep can be). The sweep spans %.4f m/s, which is 1.4%%\n",
                ensembleCount, memberEntry(ensembleCount - 1, ensembleCount) - memberEntry(0, ensembleCount));
    std::printf("  of the entry speed, so a cluster that interleaves is a bifurcation and not a speed mode.\n");

    // --- 20.3 THE TIME-RESOLVED YAW IMPULSE -------------------------------------------------------
    std::printf("\n--- 20.3 TIME-RESOLVED YAW IMPULSE, N.m.s, signed ---\n");
    std::printf("  Windows are STATED and not tuned: 0-0.25, 0.25-0.50, 0.50-1.00, 1.00-2.00, 2.00-3.00 and\n");
    std::printf("  3.00 s to the stop. A NEGATIVE impulse turns the car toward the HIGH-mu side, which is\n");
    std::printf("  the direction every member of both arms finishes in.\n");

    printWindowed("LONGITUDINAL-FORCE yaw impulse", "N.m.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.jLong[slot]; });
    printWindowed("LATERAL-FORCE yaw impulse", "N.m.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.jLat[slot]; });
    printWindowed("TOTAL yaw impulse (the measured dL_y)", "N.m.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.jTotal[slot]; });
    printWindowed("ALIGNING-MOMENT yaw impulse", "N.m.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.jAlign[slot]; });
    printWindowed("EVERYTHING ELSE (vertical + unsprung + aero + driveline)", "N.m.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.jOther[slot]; });

    // --- 20.4 THE FORCE HISTORY -------------------------------------------------------------------
    std::printf("\n--- 20.4 FORCE HISTORY, N.s, signed. Braking force is NEGATIVE (it opposes +z travel) ---\n");

    printWindowed("LOW-mu FL (corner 0) longitudinal impulse", "N.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.flLong[slot]; });
    printWindowed("HIGH-mu FR (corner 1) longitudinal impulse", "N.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.frLong[slot]; });
    printWindowed("FRONT longitudinal IMBALANCE, LOW-mu FL minus HIGH-mu FR", "N.s", groups,
                  [](const MomentMember& member, const std::size_t slot)
                  { return member.flLong[slot] - member.frLong[slot]; });
    printWindowed("FRONT lateral impulse", "N.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.frontLat[slot]; });
    printWindowed("REAR longitudinal impulse", "N.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.rearLong[slot]; });
    printWindowed("REAR lateral impulse", "N.s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.rearLat[slot]; });
    printWindowed("YAW-RATE CHANGE over the window", "rad/s", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.yawRateChange[slot]; });
    printWindowed("YAW-ANGLE CHANGE over the window", "deg", groups,
                  [](const MomentMember& member, const std::size_t slot) { return member.yawChange[slot]; });

    // --- 20.5 THE CUMULATIVE HISTORY --------------------------------------------------------------
    std::printf("\n--- 20.5 CUMULATIVE HISTORY: when does the low cluster separate? ---\n");

    printCurveField("CUMULATIVE LONGITUDINAL yaw impulse", "N.m.s", groups, &MomentMember::cumLong);
    printCurveField("CUMULATIVE LATERAL yaw impulse", "N.m.s", groups, &MomentMember::cumLat);
    printCurveField("CUMULATIVE TOTAL yaw impulse", "N.m.s", groups, &MomentMember::cumTotal);
    printCurveField("YAW ANGLE", "deg", groups, &MomentMember::curveYaw);
    printCurveField("YAW RATE", "rad/s", groups, &MomentMember::curveYawRate);
    printCurveField("INSTANTANEOUS LOW-mu FL minus HIGH-mu FR longitudinal force", "N", groups,
                    &MomentMember::curveImbalance);

    // --- 20.6 SEPARATION AT THE FOUR PROBE INSTANTS -----------------------------------------------
    std::printf("\n--- 20.6 SEPARATION OF O-LOW FROM O-HIGH AT THE FOUR PROBE INSTANTS ---\n");
    std::printf("  AUC is the rank statistic P(O-HIGH value > O-LOW value), ties counted a half. 0.5 is no\n");
    std::printf("  separation, 0 or 1 is perfect separation. The two range counts beside it are the plainest\n");
    std::printf("  statement of overlap there is. **This is a description of overlap and NOT a classifier**:\n");
    std::printf("  no threshold is chosen here and nothing downstream consumes any of it.\n");

    printProbed("cumulative LONGITUDINAL yaw impulse", "N.m.s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atCumLong[probe]; });
    printProbed("cumulative LATERAL yaw impulse", "N.m.s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atCumLat[probe]; });
    printProbed("cumulative TOTAL yaw impulse", "N.m.s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atCumTotal[probe]; });
    printProbed("YAW ANGLE", "deg", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atYaw[probe]; });
    printProbed("YAW RATE", "rad/s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atYawRate[probe]; });
    printProbed("LOW-mu FL cumulative longitudinal impulse", "N.s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFlImpulse[probe]; });
    printProbed("HIGH-mu FR cumulative longitudinal impulse", "N.s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFrImpulse[probe]; });
    printProbed("LOW-mu FL minus HIGH-mu FR cumulative longitudinal impulse", "N.s", groups,
                [](const MomentMember& member, const std::size_t probe)
                { return member.atFlImpulse[probe] - member.atFrImpulse[probe]; });
    printProbed("instantaneous LOW-mu FL longitudinal force", "N", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFlFx[probe]; });
    printProbed("instantaneous HIGH-mu FR longitudinal force", "N", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFrFx[probe]; });
    printProbed("LOW-mu FL slip ratio", "-", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFlSlip[probe]; });
    printProbed("HIGH-mu FR slip ratio", "-", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFrSlip[probe]; });
    printProbed("LOW-mu FL brake pressure", "bar", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFlBar[probe]; });
    printProbed("HIGH-mu FR brake pressure", "bar", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atFrBar[probe]; });

    std::printf("\n  CONTROLLER PHASE at each probe instant, member counts (0 Passive, 1 Hold, 2 Dump,\n");
    std::printf("  3 Recover, 4 Reapply). Reported AFTER the physical picture above, not before it.\n");
    std::printf("\n  the LOW-mu wheel is FL (corner 0) and the HIGH-mu wheel is FR (corner 1), measured\n");
    std::printf("\n  %-8s %-9s %9s %9s %9s %9s %9s | %9s %9s %9s %9s %9s\n", "at", "group", "loFL pas", "loFL hold",
                "loFL dump", "loFL rec", "loFL rea", "hiFR pas", "hiFR hold", "hiFR dump", "hiFR rec", "hiFR rea");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-----------------------------------");

    for (auto probe = std::size_t{0}; probe < probeCount; probe++)
    {
        const auto rows = std::array<std::pair<const char*, const std::vector<const MomentMember*>*>, 3>{
            std::pair{"P", &groups.production}, std::pair{"O-LOW", &groups.low}, std::pair{"O-HIGH", &groups.high}};

        auto label = std::array<char, 16>{};
        std::snprintf(label.data(), label.size(), "%.2f s", probeTimes[probe]);
        const char* shown = label.data();

        for (const auto& row : rows)
        {
            auto front = std::array<std::size_t, 5>{};
            auto right = std::array<std::size_t, 5>{};

            for (const auto* member : *row.second)
            {
                front[member->atFlPhase[probe]]++;
                right[member->atFrPhase[probe]]++;
            }

            std::printf("  %-8s %-9s %9zu %9zu %9zu %9zu %9zu | %9zu %9zu %9zu %9zu %9zu\n", shown, row.first, front[0],
                        front[1], front[2], front[3], front[4], right[0], right[1], right[2], right[3], right[4]);
            shown = "";
        }
    }

    // --- 20.7 IMPULSE AGAINST TIMING --------------------------------------------------------------
    std::printf("\n--- 20.7 IMPULSE AGAINST TIMING ---\n");
    std::printf("  Two members can carry the same total yaw impulse and finish at different headings if the\n");
    std::printf("  moment arrives at different times. The absolute centroid is\n");
    std::printf("      t_c = integral(t * |dL_y|) / integral(|dL_y|)\n");
    std::printf("  and it is DIAGNOSTIC ONLY — nothing reads it and no controller signal is derived from\n");
    std::printf("  it. The signed centroid is printed beside it with the count of members for which its\n");
    std::printf("  denominator is large enough against the absolute total to mean anything.\n");

    printGroupHeader("absolute centroid [s]");
    printGroupRow(
        "P",
        gather(groups.production, [](const MomentMember& member, const std::size_t) { return member.centroidAbs; }, 0));
    printGroupRow(
        "O-LOW",
        gather(groups.low, [](const MomentMember& member, const std::size_t) { return member.centroidAbs; }, 0));
    printGroupRow(
        "O-HIGH",
        gather(groups.high, [](const MomentMember& member, const std::size_t) { return member.centroidAbs; }, 0));

    printGroupHeader("total |dL_y| [N.m.s]");
    printGroupRow(
        "P",
        gather(groups.production, [](const MomentMember& member, const std::size_t) { return member.absImpulse; }, 0));
    printGroupRow(
        "O-LOW",
        gather(groups.low, [](const MomentMember& member, const std::size_t) { return member.absImpulse; }, 0));
    printGroupRow(
        "O-HIGH",
        gather(groups.high, [](const MomentMember& member, const std::size_t) { return member.absImpulse; }, 0));

    printGroupHeader("net dL_y over the stop [N.m.s]");
    printGroupRow("P", gather(
                           groups.production,
                           [](const MomentMember& member, const std::size_t) { return member.measuredDelta; }, 0));
    printGroupRow(
        "O-LOW",
        gather(groups.low, [](const MomentMember& member, const std::size_t) { return member.measuredDelta; }, 0));
    printGroupRow(
        "O-HIGH",
        gather(groups.high, [](const MomentMember& member, const std::size_t) { return member.measuredDelta; }, 0));

    {
        auto usable = std::size_t{0};
        for (const auto& member : controlMembers)
        {
            usable += member.signedCentroidUsable ? 1 : 0;
        }
        auto usableProduction = std::size_t{0};
        for (const auto& member : productionMembers)
        {
            usableProduction += member.signedCentroidUsable ? 1 : 0;
        }

        std::printf("\n  signed centroid readable (|net| > 25%% of total |impulse|): P %zu/%zu, O %zu/%zu.\n",
                    usableProduction, ensembleCount, usable, ensembleCount);
        std::printf("  Where it is not readable the denominator is a difference of two large cancelling\n");
        std::printf("  numbers and the quotient is arbitrary, so the ABSOLUTE centroid above is what the\n");
        std::printf("  timing statement rests on.\n");
    }

    printGroupHeader("stop duration [s]");
    printGroupRow(
        "P",
        gather(groups.production, [](const MomentMember& member, const std::size_t) { return member.stopTime; }, 0));
    printGroupRow("O-LOW",
                  gather(groups.low, [](const MomentMember& member, const std::size_t) { return member.stopTime; }, 0));
    printGroupRow(
        "O-HIGH",
        gather(groups.high, [](const MomentMember& member, const std::size_t) { return member.stopTime; }, 0));

    printGroupHeader("stopping distance [m]");
    printGroupRow(
        "P",
        gather(groups.production, [](const MomentMember& member, const std::size_t) { return member.distance; }, 0));
    printGroupRow("O-LOW",
                  gather(groups.low, [](const MomentMember& member, const std::size_t) { return member.distance; }, 0));
    printGroupRow(
        "O-HIGH",
        gather(groups.high, [](const MomentMember& member, const std::size_t) { return member.distance; }, 0));

    // --- 20.8 SENSOR TIMING -------------------------------------------------------------------------
    std::printf("\n--- 20.8 SENSOR TIMING AROUND THE PROBE INSTANTS ---\n");
    std::printf("  The mean front-channel `sensorAge` in the 100 ms bracketing each probe instant, from the\n");
    std::printf("  steps these runs already recorded. **No sweep is run and no filter is redesigned.** The\n");
    std::printf("  only question asked is whether the cluster split sits on a one-tick or one-pulse timing\n");
    std::printf("  difference. The controller period is %.5f s and the physics tick is %.5f s.\n",
                assists.controlRate > 0.0 ? 1.0 / assists.controlRate : tick, tick);

    printProbed("front-channel mean sensor age", "s", groups,
                [](const MomentMember& member, const std::size_t probe) { return member.atSensorAge[probe]; });

    // --- 20.9 THE MEMBER MAP ------------------------------------------------------------------------
    std::printf("\n--- 20.9 MEMBER MAP: the whole-law-off arm, by cluster ---\n");
    std::printf("\n  %-3s %8s %8s %10s %10s %10s %10s %10s %10s %8s\n", "k", "entry", "yaw", "J_long", "J_lat",
                "J_total", "FL imp", "FR imp", "t_c abs", "cluster");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "----------------");

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto& member = controlMembers[index];
        auto totalLong = 0.0;
        auto totalLat = 0.0;
        auto totalFl = 0.0;
        auto totalFr = 0.0;

        for (auto slot = std::size_t{0}; slot < windowCount; slot++)
        {
            totalLong += member.jLong[slot];
            totalLat += member.jLat[slot];
            totalFl += member.flLong[slot];
            totalFr += member.frLong[slot];
        }

        std::printf("  %-3zu %8.4f %8.3f %10.1f %10.1f %10.1f %10.1f %10.1f %10.3f %8s\n", index, member.entry,
                    member.finalYaw, totalLong, totalLat, member.measuredDelta, totalFl, totalFr, member.centroidAbs,
                    std::abs(member.finalYaw) <= clusterFloor ? "LOW" : "HIGH");
    }

    std::printf("\n  and the same for production, which has no cluster structure\n");
    std::printf("\n  %-3s %8s %8s %10s %10s %10s %10s %10s %10s\n", "k", "entry", "yaw", "J_long", "J_lat", "J_total",
                "FL imp", "FR imp", "t_c abs");
    std::printf("  %s\n", "-------------------------------------------------------------------------------------"
                          "-------");

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto& member = productionMembers[index];
        auto totalLong = 0.0;
        auto totalLat = 0.0;
        auto totalFl = 0.0;
        auto totalFr = 0.0;

        for (auto slot = std::size_t{0}; slot < windowCount; slot++)
        {
            totalLong += member.jLong[slot];
            totalLat += member.jLat[slot];
            totalFl += member.flLong[slot];
            totalFr += member.frLong[slot];
        }

        std::printf("  %-3zu %8.4f %8.3f %10.1f %10.1f %10.1f %10.1f %10.1f %10.3f\n", index, member.entry,
                    member.finalYaw, totalLong, totalLat, member.measuredDelta, totalFl, totalFr, member.centroidAbs);
    }

    std::printf("\n  Production is unchanged by this file. Both arms run the production control law with\n");
    std::printf("  `Candidate::replica` false; O differs from P by one field of the shipped setup and by\n");
    std::printf("  nothing else. 20.1 proves the momentum ledger closes to floating-point noise and that the\n");
    std::printf("  recorder reproduces `record` to the bit. No threshold, gradient, estimator constant or\n");
    std::printf("  tyre parameter is touched anywhere in it, no controller is designed, and no classifier or\n");
    std::printf("  threshold is selected from any separation statistic above.\n");
}

// =============================================================================================
// SECTION 20A — WHICH CORNER IS ON WHICH SURFACE, MEASURED.
// `./EngineTests "[.mu-side]"` — two seconds, settles one car on the split plate and prints what
// each corner is standing on. It exists because section 20's causal statements are about a *side*
// and the frame this project uses is the one thing in it that has been got wrong before:
// `Suspension.cppm` records that "+x is the car's left" was stated in five places and was wrong in
// all five, and that the only oracle that settled it was a picture. A yaw-moment decomposition
// cannot be read off a side label that has not been measured, so this measures it.
// =============================================================================================
TEST_CASE("which front corner stands on which surface of the split-mu plate", "[.mu-side]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto split = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(split.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, split.value(), memberEntry(0, ensembleCount));

    const auto stepped = stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, split.value(), tick);
    REQUIRE(stepped.has_value());

    std::printf("\n=== SECTION 20A: WHICH CORNER STANDS ON WHICH SURFACE ===\n");
    std::printf("  `gripPlate(1.00, 0.35)` gives grip 1.00 to every triangle whose centroid has x < 0 and\n");
    std::printf("  0.35 to the rest. `Suspension.cppm` states that +x is the car's LEFT and that the frame\n");
    std::printf("  is left-handed. Neither is assumed below: the grip each corner's own contact patch\n");
    std::printf("  reports is what decides the label.\n");
    std::printf("\n  %-14s %12s %12s %14s %12s\n", "corner", "world x", "chassis x", "grip", "surface");
    std::printf("  %s\n", "----------------------------------------------------------------------------");

    static constexpr auto names =
        std::array<const char*, cornerCount>{"0 FrontLeft", "1 FrontRight", "2 RearLeft", "3 RearRight"};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& solution = stepped.value().corners[index];

        std::printf("  %-14s %12.4f %12.4f %14.3f %12s\n", names[index], solution.patch.centre.x,
                    solution.suspension.contactPatch.x, solution.patch.gripMultiplier,
                    solution.patch.gripMultiplier > 0.5 ? "HIGH mu" : "LOW mu");
    }

    const auto& frontLeft = stepped.value().corners[0];
    const auto& frontRight = stepped.value().corners[1];

    std::printf("\n  VERDICT: corner 0 (FrontLeft) is the %s wheel and corner 1 (FrontRight) is the %s one.\n",
                frontLeft.patch.gripMultiplier > 0.5 ? "HIGH-mu" : "LOW-mu",
                frontRight.patch.gripMultiplier > 0.5 ? "HIGH-mu" : "LOW-mu");
    std::printf("  The channel names `channelName` prints — FL, FR — are corner indices and carry no\n");
    std::printf("  statement about grip. Any reading that attaches a mu to one of them must come from\n");
    std::printf("  this table.\n");

    REQUIRE(frontLeft.patch.gripMultiplier != frontRight.patch.gripMultiplier);
}
