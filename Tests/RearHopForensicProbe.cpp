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
using raceengine::WheelSpeedReadings;

// The rear-axle hop forensic instrument. `./EngineTests "[.rear-hop]"`.
//
// **Characterisation only. Nothing in this file is a proposal and nothing in it is a gate.** There
// is no `REQUIRE` on any behavioural figure anywhere below: every `REQUIRE` is a precondition of the
// kind `docs/` calls for [fixtures-must-assert-preconditions], plus the one bit-identity assertion
// that licenses the shadow controller described next. No production file is touched by this probe;
// no acceptance criterion is read, moved or added.
//
// **What is being characterised.** `docs/braking-chain-brief.md` §15 nominates the rear-axle hop as
// the one open mechanism: the production anti-lock arm splits into two clean clusters over a
// deterministic 29-member entry-speed ensemble — 9 members recover the rear axle (rear utilisation
// 0.566-0.715, stop 43.5-44.8 m) and 20 strand it (0.327-0.472, 45.1-47.7 m) — and the airborne-tick
// discriminator that used to separate them has been falsified. This file answers *why* the rear
// channel can sustain the hop, and what the controller cannot see that would break it.
//
// **The shadow controller, and why it exists.** Every oracle below has to intervene *inside* one
// controller period, between the rear channel's decision and the rear caliper. `updateAssists` runs
// that loop internally, so the probe carries its own copy of the loop — `shadowUpdate` — that calls
// the same exported production entry points (`sampleWheelSensors`, `advanceReferenceSpeed`,
// `advanceYawMomentDelay`, `antilockControlWheel`, `advanceAntilockChannel`) in the same order with
// the same arguments. **The control law itself is NOT copied**: `advanceAntilockChannel` is called,
// never reimplemented, so there is no second copy of the state machine to drift. What the shadow
// adds is a probe-local actuator override applied *after* the production call returns.
//
// The first test case below asserts the shadow reproduces production to the bit across the whole
// ensemble with every oracle off. Without that assertion nothing else here means anything
// [check-the-control-variable].

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto gravity = 9.80665;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto bar = 1.0e5;
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;

// The ensemble, stated exactly as `[.brake-ledger]` states it so a member index here is the same
// member there: +/-0.7% of the fixture's own 100 km/h, uniformly spaced, odd count.
constexpr auto ensembleBand = 0.007;
constexpr auto ensembleCount = std::size_t{29};

// The separator `[.brake-gap]` found in the data rather than chose. Re-checked on every arm below
// before it is quoted.
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
// The probe-only diagnostic oracles.
//
// **Every one of these is an actuator override applied after `advanceAntilockChannel` returns.** The
// production state machine runs unmodified and sees its own real state; what the oracle changes is
// what reaches the caliper. That is the weakest intervention that can answer the causal question,
// and it keeps the production law out of this file entirely.
//
// **None of these is a proposal and none of their numbers is a calibration.** Where a threshold
// appears it is a ground-truth quantity no wheel-speed sensor can observe, chosen to be obviously
// diagnostic (zero load, the physical contact flag) rather than tuned.
// ---------------------------------------------------------------------------------------------

enum class RearMode : std::uint8_t
{
    // The production channel, untouched.
    Production,
    // A: no rear pressure RISE while the rear axle is unloaded. Dumps still work.
    ContactGate,
    // B: no rear pressure rise for a fixed interval after physical contact is restored.
    LandingDelay,
    // B2: no rear pressure rise for a fixed interval after every dump exit. **Knows nothing about
    // contact** — this is the timing-only control for section 12.
    BlindDelay,
    // C: the state machine runs untouched and the rear brake TORQUE is zeroed for a fixed interval
    // from the instant contact is restored.
    LandingTorqueCut,
    // 13a: the state machine runs untouched and the rear actuator realises its demand with no valve
    // lag at all — dump to zero in one step, re-apply to the request in one step.
    IdealActuator,
    // 13b: the production valve, driven by a per-wheel peak-slip oracle instead of the state
    // machine. The complement of 13a: ideal decision, real actuator.
    OracleThroughValve
};

struct RearOracle
{
    RearMode mode = RearMode::Production;

    // ContactGate: the axle is "unloaded" below this fraction of one rear wheel's static load. 0.0
    // means literal zero load, which is the definition that needs no number at all.
    double loadFraction = 0.0;
    // ContactGate: gate on the physical contact flag rather than on load.
    bool useContactFlag = false;

    // LandingDelay / BlindDelay / LandingTorqueCut, seconds.
    double delay = 0.0;
};

// What the oracle is allowed to see, and it is ground truth: the simulator's own rear tyre loads and
// contact flags, one physics tick old. **No production controller has any of this** — that is the
// point of section 18, and every use of it below is labelled a PHYSICS ORACLE.
struct RearTruth
{
    std::array<bool, 2> contact{true, true};
    std::array<double, 2> load{};
    double staticWheelLoad = 1.0;
    // Seconds since the rear axle last regained contact, +infinity-ish before the first loss.
    double sinceLanding = 1e9;
    bool unloaded = false;
};

// One controller step's worth of the rear channel's own decision, reconstructed from the channel
// state the production call leaves behind. Nothing here is computed by a copy of the law: `slip`,
// `guardSlip`, `excess`, `losing`, `surging` and `pastBand` are the same expressions
// `advanceAntilockChannel` evaluates, applied to the state IT left, which is what makes them exact.
struct ControlStep
{
    double time = 0.0;
    std::size_t channel = 0;
    std::size_t controlWheel = 0;

    ModulatorPhase before = ModulatorPhase::Passive;
    ModulatorPhase after = ModulatorPhase::Passive;
    double pressureBefore = 0.0;
    double pressureAfter = 0.0;
    double request = 0.0;
    double departurePressure = 0.0;
    bool surgedBefore = false;

    double sensedSpeed = 0.0;
    double acceleration = 0.0;
    double excess = 0.0;
    double slip = 0.0;
    double guardSlip = 0.0;
    double referenceSpeed = 0.0;
    double referenceRate = 0.0;
    bool referenceValid = false;
    bool losing = false;
    bool surging = false;
    bool pastBand = false;

    // What the oracle did, and what the world was actually doing while it did it.
    bool gated = false;
    double truthLoadMin = 0.0;
    bool truthContact = true;
};

struct ShadowState
{
    AssistState assists{};
    // BlindDelay's own timer: seconds since the rear channel last left the dump.
    double sinceDumpExit = 1e9;
    std::size_t gatedSteps = 0;
    double gatedSeconds = 0.0;

    // What the ECU ended the tick believing, per wheel, so the sampler does not have to run the
    // sensors a second time to find out.
    std::array<double, cornerCount> sensedSpeed{};
    std::array<double, cornerCount> estimatedSlip{};
    std::array<ModulatorPhase, cornerCount> phase{};
    std::array<std::uint32_t, cornerCount> cycles{};
};

// The probe-local copy of `updateAssists`'s outer loop. Traction control and the cornering brake are
// deliberately not stepped: this fixture leaves both off (`golfGtiMk7Assists` defaults
// `TractionMode::Off` and `cornering.enabled` false), neither touches the sensors or the reference,
// and the bit-identity case below is what proves the omission costs nothing.
[[nodiscard]] BrakeCommand shadowUpdate(const AssistSetup& setup, ShadowState& shadow, const AssistSensors& sensors,
                                        const double brakeDemand, const std::array<double, cornerCount>& wheelPressure,
                                        const double deltaTime, const RearOracle& oracle, const RearTruth& truth,
                                        const std::array<double, cornerCount>& oracleTorque,
                                        std::vector<ControlStep>* trace, const double runTime)
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
        shadow.sinceDumpExit += period;

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

            auto pressure = advanceAntilockChannel(
                setup.antilock, channel, channelState, readings[controlWheel], wheelSpeed, state.reference.speed,
                state.reference.rate, state.reference.valid, channelRequest, AntilockChannelInputs{}, period);

            const auto rear = index == 2;
            auto gated = false;

            if (rear && snapshot.phase == ModulatorPhase::Dump && channelState.phase != ModulatorPhase::Dump)
            {
                shadow.sinceDumpExit = 0.0;
            }

            // --- the probe-only override -------------------------------------------------------
            if (rear && oracle.mode != RearMode::Production)
            {
                switch (oracle.mode)
                {
                case RearMode::ContactGate:
                {
                    const auto blocked = oracle.useContactFlag ? (!truth.contact[0] || !truth.contact[1])
                                                               : std::min(truth.load[0], truth.load[1]) <=
                                                                     oracle.loadFraction * truth.staticWheelLoad;
                    if (blocked && pressure > snapshot.pressure)
                    {
                        pressure = snapshot.pressure;
                        channelState.pressure = pressure;
                        gated = true;
                    }
                    break;
                }
                case RearMode::LandingDelay:
                {
                    const auto blocked = truth.sinceLanding < oracle.delay;
                    if (blocked && pressure > snapshot.pressure)
                    {
                        pressure = snapshot.pressure;
                        channelState.pressure = pressure;
                        gated = true;
                    }
                    break;
                }
                case RearMode::BlindDelay:
                {
                    const auto blocked = shadow.sinceDumpExit < oracle.delay;
                    if (blocked && pressure > snapshot.pressure)
                    {
                        pressure = snapshot.pressure;
                        channelState.pressure = pressure;
                        gated = true;
                    }
                    break;
                }
                case RearMode::IdealActuator:
                {
                    if (channelState.phase == ModulatorPhase::Dump)
                    {
                        pressure = 0.0;
                    }
                    else if (channelState.phase == ModulatorPhase::Reapply)
                    {
                        pressure = channelRequest;
                    }

                    pressure = std::min(pressure, channelRequest);
                    channelState.pressure = pressure;
                    gated = pressure != snapshot.pressure;
                    break;
                }
                case RearMode::LandingTorqueCut:
                case RearMode::OracleThroughValve:
                case RearMode::Production:
                    break;
                }
            }

            if (gated)
            {
                shadow.gatedSteps++;
                shadow.gatedSeconds += period;
            }

            if (trace != nullptr)
            {
                auto record = ControlStep{};
                record.time = runTime + elapsed;
                record.channel = index;
                record.controlWheel = controlWheel;
                record.before = snapshot.phase;
                record.after = channelState.phase;
                record.pressureBefore = snapshot.pressure;
                record.pressureAfter = pressure;
                record.request = channelRequest;
                record.departurePressure = snapshot.departurePressure;
                record.surgedBefore = snapshot.surged;
                record.sensedSpeed = wheelSpeed;
                record.acceleration = channelState.acceleration;
                record.excess = channelState.acceleration - state.reference.rate;
                record.referenceSpeed = state.reference.speed;
                record.referenceRate = state.reference.rate;
                record.referenceValid = state.reference.valid;
                record.slip = state.reference.valid ? estimatedSlip(state.reference.speed, wheelSpeed) : 0.0;

                const auto projected =
                    std::max(wheelSpeed, channelState.lastSpeed + state.reference.rate * channelState.sinceUpdate);
                record.guardSlip = state.reference.valid ? estimatedSlip(state.reference.speed, projected) : 0.0;
                record.losing = channelState.acceleration < setup.antilock.lockDeceleration;
                record.surging = record.excess > setup.antilock.recoverySurge;
                record.pastBand = (setup.antilock.recoveryAuthority != RecoveryAuthority::Disabled) &&
                                  state.reference.valid && record.guardSlip > setup.antilock.slipEnter;
                record.gated = gated;
                record.truthLoadMin = std::min(truth.load[0], truth.load[1]);
                record.truthContact = truth.contact[0] && truth.contact[1];

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
                shadow.cycles[wheel] = state.antilock.channels[index].cycles;
            }
        }
    }

    if (oracle.mode == RearMode::LandingTorqueCut && truth.sinceLanding < oracle.delay)
    {
        command.wheels[2] = 0.0;
        command.wheels[3] = 0.0;
    }

    if (oracle.mode == RearMode::OracleThroughValve)
    {
        command.wheels[2] = oracleTorque[2];
        command.wheels[3] = oracleTorque[3];
    }

    return command;
}

} // namespace

namespace
{

// --- what one wheel was doing on one physics tick ---

struct WheelSample
{
    double load = 0.0;
    double forceLongitudinal = 0.0;
    double friction = 0.0;
    double slipRatio = 0.0;
    double peakSlip = 0.0;
    double effectiveRadius = 0.0;
    double wheelSpeed = 0.0;
    double wheelAcceleration = 0.0;
    double pressure = 0.0;
    double brakeTorque = 0.0;
    double estimatedSlip = 0.0;
    double sensedSpeed = 0.0;
    double suspensionTravel = 0.0;
    double travelPerAngle = 0.0;
    // The longitudinal component of d(contact patch)/d(wishbone angle) — the geometric load path's
    // own term, and the ONE channel that says how much vertical work a longitudinal tyre force does
    // on this corner's degree of freedom.
    double patchPerAngleZ = 0.0;
    double damperVelocity = 0.0;
    double damperForce = 0.0;
    double springForce = 0.0;
    double bumpStopForce = 0.0;
    double droopStopForce = 0.0;
    ModulatorPhase phase = ModulatorPhase::Passive;
    bool inContact = false;

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
    double pitchRate = 0.0;
    double referenceSpeed = 0.0;
    double referenceRate = 0.0;
    double referenceCoasting = 0.0;
    bool referenceValid = false;
    // Which wheel is the estimator's own candidate this tick: under braking it takes the FASTEST
    // sensed wheel, so this says whether the reference is currently being set by a rear wheel.
    std::size_t fastestWheel = 0;
    bool rearContact = true;
    double rearAxleLoad = 0.0;
    std::array<WheelSample, cornerCount> wheels{};
};

struct Run
{
    std::vector<Sample> samples;
    std::vector<ControlStep> steps;
    double distance = 0.0;
    double time = 0.0;
    double entrySpeed = 0.0;
    bool stopped = false;
    bool grounded = true;
    std::size_t rearAirborneTicks = 0;
    std::size_t gatedSteps = 0;
    double gatedSeconds = 0.0;
    // How many rear landings the run contained, and what the oracle blocked at each.
    std::size_t landings = 0;
};

// The oracle's trim gain, copied from `[.brake-utilisation]` so the `OracleThroughValve` arm is the
// same per-wheel peak-slip controller the ledger's oracle arms use.
constexpr auto oracleTrim = 0.5;

// One stop. `oracle` selects the probe-only rear intervention; `Production` is the shipped car.
[[nodiscard]] Run record(const VehicleSetup& setup, const PhysicsWorld& world, const AssistSetup& assists,
                         const double pedal, const RearOracle& oracle, const double entrySpeed,
                         const bool traceSteps = false)
{
    auto state = VehicleState{};
    settle(setup, state, world, entrySpeed);

    auto shadow = ShadowState{};
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

    auto truth = RearTruth{};
    auto passive = RearOracle{};
    const auto noOracleTorque = std::array<double, cornerCount>{};

    for (auto step = 0; step < 180; step++)
    {
        const auto command = shadowUpdate(assists, shadow, sense(), 0.0, noBrakePressure, tick, passive, truth,
                                          noOracleTorque, nullptr, 0.0);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
        REQUIRE(lastStep.telemetry.wheels[index].gripMultiplier > 0.0);
    }
    REQUIRE(std::abs(state.chassis.linearVelocity.z - entrySpeed) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - designHeight) < 0.1);
    REQUIRE(std::abs(state.chassis.position.x) < 0.05);

    // The static rear wheel load the oracle's normalised threshold is a fraction of, taken from the
    // settled roll rather than stated: it is the car's own, so the same fraction means the same
    // thing on a different vehicle [borrowed-numbers-need-a-platform-match].
    truth.staticWheelLoad =
        std::max(1.0, 0.5 * (lastStep.corners[2].forces.tireVertical + lastStep.corners[3].forces.tireVertical));

    auto run = Run{};
    run.entrySpeed = state.chassis.linearVelocity.z;
    run.samples.reserve(360 * 8);

    const auto start = state.chassis.position;
    auto previousSpeed = run.entrySpeed;
    auto previousWheelSpeed = std::array<double, cornerCount>{};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        previousWheelSpeed[index] = state.corners[index].wheelSpeed;
    }

    auto input = VehicleInput{};
    input.brake = pedal;

    const auto pressures = brakeCircuitPressures(setup, pedal);

    for (auto step = 0; step < 360 * 30; step++)
    {
        // Ground truth for the oracle, one physics tick old — which is what a probe standing outside
        // the tick has and is the honest lag to give it.
        for (auto index = std::size_t{0}; index < 2; index++)
        {
            truth.contact[index] = lastStep.telemetry.wheels[index + 2].inContact;
            truth.load[index] = lastStep.corners[index + 2].forces.tireVertical;
        }

        const auto nowUnloaded = !truth.contact[0] || !truth.contact[1];
        if (truth.unloaded && !nowUnloaded)
        {
            truth.sinceLanding = 0.0;
            run.landings++;
        }
        else
        {
            truth.sinceLanding += tick;
        }
        truth.unloaded = nowUnloaded;

        // The per-wheel peak-slip command the `OracleThroughValve` arm drives the rear with, built
        // exactly as `[.brake-utilisation]`'s bounded oracle builds it.
        auto oracleTorque = std::array<double, cornerCount>{};
        if (oracle.mode == RearMode::OracleThroughValve)
        {
            for (auto index = std::size_t{2}; index < cornerCount; index++)
            {
                const auto& solution = lastStep.corners[index];
                const auto& corner = setup.corners[index];
                const auto peakSlip = std::max(solution.contact.tyre.longitudinalPeakSlip, 1e-3);
                const auto slip = std::abs(solution.contact.slip.slipRatio);
                const auto radius = std::max(solution.contact.effectiveRadius, 1e-3);
                const auto feedforward = std::abs(solution.contact.tyre.longitudinal) * radius;
                const auto error = std::clamp((peakSlip - slip) / peakSlip, -1.0, 1.0);
                const auto trim = oracleTrim * solution.forces.tireVertical * radius * error;

                auto command = std::max(0.0, feedforward + trim);
                command = std::max(0.0, command - corner.rollingResistance * solution.forces.tireVertical * radius);
                oracleTorque[index] = std::min(command, corner.brakeTorque);
            }
        }

        const auto brakes = shadowUpdate(assists, shadow, sense(), pedal, pressures, tick, oracle, truth, oracleTorque,
                                         traceSteps ? &run.steps : nullptr, run.time);

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        run.time += tick;

        auto sample = Sample{};
        sample.time = run.time;
        sample.speed = state.chassis.linearVelocity.z;
        sample.deceleration = (previousSpeed - sample.speed) / tick;
        sample.pitch = lastStep.telemetry.pitch;
        sample.pitchRate = lastStep.telemetry.pitchRate;
        sample.referenceSpeed = shadow.assists.reference.speed;
        sample.referenceRate = shadow.assists.reference.rate;
        sample.referenceValid = shadow.assists.reference.valid;
        sample.referenceCoasting = shadow.assists.reference.coasting;
        previousSpeed = sample.speed;

        for (auto index = std::size_t{1}; index < cornerCount; index++)
        {
            if (std::abs(shadow.sensedSpeed[index]) > std::abs(shadow.sensedSpeed[sample.fastestWheel]))
            {
                sample.fastestWheel = index;
            }
        }

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];
            auto& wheel = sample.wheels[index];

            wheel.load = solution.forces.tireVertical;
            wheel.forceLongitudinal = solution.contact.tyre.longitudinal;
            wheel.friction = tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, wheel.load,
                                          solution.patch.gripMultiplier);
            wheel.slipRatio = solution.contact.slip.slipRatio;
            wheel.peakSlip = solution.contact.tyre.longitudinalPeakSlip;
            wheel.effectiveRadius = solution.contact.effectiveRadius;
            wheel.wheelSpeed = state.corners[index].wheelSpeed;
            wheel.wheelAcceleration = (wheel.wheelSpeed - previousWheelSpeed[index]) / tick;
            previousWheelSpeed[index] = wheel.wheelSpeed;
            wheel.pressure = shadow.assists.pressure[index];
            wheel.brakeTorque = brakes.wheels[index];
            wheel.estimatedSlip = shadow.estimatedSlip[index];
            wheel.sensedSpeed = shadow.sensedSpeed[index];
            wheel.phase = shadow.phase[index];
            wheel.suspensionTravel = solution.suspension.wheelTravel;
            wheel.travelPerAngle = solution.suspension.travelPerAngle;
            wheel.patchPerAngleZ = solution.suspension.patchPerAngle.z;
            wheel.damperVelocity = solution.damperVelocity;
            wheel.damperForce = solution.forces.damper;
            wheel.springForce = solution.forces.spring;
            wheel.bumpStopForce = solution.forces.bumpStop;
            wheel.droopStopForce = solution.forces.droopStop;
            wheel.inContact = lastStep.telemetry.wheels[index].inContact;

            run.grounded = run.grounded && wheel.inContact;
        }

        sample.rearContact = sample.wheels[2].inContact && sample.wheels[3].inContact;
        sample.rearAxleLoad = sample.wheels[2].load + sample.wheels[3].load;
        run.rearAirborneTicks += sample.rearContact ? 0 : 1;

        run.samples.push_back(sample);

        if (sample.speed <= 0.0)
        {
            run.stopped = true;
            break;
        }
    }

    run.distance = state.chassis.position.z - start.z;
    run.gatedSteps = shadow.gatedSteps;
    run.gatedSeconds = shadow.gatedSeconds;

    return run;
}

// --- the statistics the ledger reports, restated here so this file stands alone ---

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

struct AxleUse
{
    double frontUtilisation = 0.0;
    double rearUtilisation = 0.0;
    double minimumRearLoad = 1e9;
};

[[nodiscard]] AxleUse axleUse(const Run& run)
{
    auto use = AxleUse{};
    auto force = std::array<double, 2>{};
    auto capacity = std::array<double, 2>{};

    for (const auto& sample : run.samples)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& wheel = sample.wheels[index];
            const auto axle = index < 2 ? std::size_t{0} : std::size_t{1};

            force[axle] += std::abs(wheel.forceLongitudinal);
            capacity[axle] += wheel.capacity();
        }

        use.minimumRearLoad = std::min(use.minimumRearLoad, sample.rearAxleLoad);
    }

    use.frontUtilisation = capacity[0] > 1.0 ? force[0] / capacity[0] : 0.0;
    use.rearUtilisation = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0;

    return use;
}

struct Member
{
    std::size_t index = 0;
    double entry = 0.0;
    double distance = 0.0;
    double frontUtilisation = 0.0;
    double rearUtilisation = 0.0;
    double minimumRearLoad = 0.0;
    std::size_t rearAirborne = 0;
    std::size_t landings = 0;
    std::size_t frontCycles = 0;
    std::size_t rearCycles = 0;
    double gatedSeconds = 0.0;
    bool grounded = false;
    bool stopped = false;

    [[nodiscard]] bool stranded() const
    {
        return rearUtilisation < strandedRear;
    }
};

[[nodiscard]] std::size_t cyclesOf(const Run& run, const std::size_t wheel)
{
    // Entries to the dump, counted off the sampled phase the way `AntilockChannelState::cycles`
    // counts them, so the number means the same thing.
    auto cycles = std::size_t{0};
    auto previous = ModulatorPhase::Passive;

    for (const auto& sample : run.samples)
    {
        const auto phase = sample.wheels[wheel].phase;
        cycles += (phase == ModulatorPhase::Dump && previous != ModulatorPhase::Dump) ? 1 : 0;
        previous = phase;
    }

    return cycles;
}

[[nodiscard]] std::vector<Member> ensembleOf(const VehicleSetup& setup, const PhysicsWorld& world,
                                             const AssistSetup& assists, const double pedal, const RearOracle& oracle,
                                             const std::size_t count = ensembleCount)
{
    auto members = std::vector<Member>{};
    members.reserve(count);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto entry = memberEntry(index, count);
        const auto run = record(setup, world, assists, pedal, oracle, entry);
        const auto use = axleUse(run);

        members.push_back(Member{.index = index,
                                 .entry = entry,
                                 .distance = run.distance,
                                 .frontUtilisation = use.frontUtilisation,
                                 .rearUtilisation = use.rearUtilisation,
                                 .minimumRearLoad = use.minimumRearLoad,
                                 .rearAirborne = run.rearAirborneTicks,
                                 .landings = run.landings,
                                 .frontCycles = cyclesOf(run, 0),
                                 .rearCycles = cyclesOf(run, 2),
                                 .gatedSeconds = run.gatedSeconds,
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
    std::size_t grounded = 0;
    double highestStranded = 0.0;
    double lowestRecovered = 1e9;
    double meanRearAirborne = 0.0;
    double meanRearCycles = 0.0;
    double meanFrontCycles = 0.0;
};

[[nodiscard]] ArmSummary summarise(const std::vector<Member>& members)
{
    auto summary = ArmSummary{};
    auto distances = std::vector<double>{};
    auto rear = std::vector<double>{};
    auto front = std::vector<double>{};

    for (const auto& member : members)
    {
        distances.push_back(member.distance);
        rear.push_back(member.rearUtilisation);
        front.push_back(member.frontUtilisation);

        summary.stranded += member.stranded() ? 1 : 0;
        summary.grounded += member.grounded ? 1 : 0;
        summary.meanRearAirborne += static_cast<double>(member.rearAirborne);
        summary.meanRearCycles += static_cast<double>(member.rearCycles);
        summary.meanFrontCycles += static_cast<double>(member.frontCycles);

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
    summary.meanRearAirborne /= count;
    summary.meanRearCycles /= count;
    summary.meanFrontCycles /= count;
    summary.distance = distributionOf(distances);
    summary.rearUtilisation = distributionOf(rear);
    summary.frontUtilisation = distributionOf(front);

    return summary;
}

void printArm(const char* name, const ArmSummary& summary)
{
    std::printf("  %-40s  %2zu/%2zu   %7.3f  %7.3f  %7.3f   %6.3f  %6.3f  %6.3f   %6.3f   %2zu/%2zu  %6.1f  %5.1f\n",
                name, summary.stranded, summary.distance.n, summary.distance.p10, summary.distance.median,
                summary.distance.p90, summary.rearUtilisation.p10, summary.rearUtilisation.median,
                summary.rearUtilisation.p90, summary.frontUtilisation.median, summary.grounded, summary.distance.n,
                summary.meanRearAirborne, summary.meanRearCycles);
}

void printArmHeader()
{
    std::printf("\n  %-40s  stranded    P10   MEDIAN      P90    rear util P10 / MED / P90   front"
                "   4wheels  rearOff  cyc\n",
                "arm");
}

} // namespace

namespace
{

// The same stop driven through the **production** `updateAssists`, for the bit-identity check
// alone. It records only what the check compares.
struct ProductionRun
{
    double distance = 0.0;
    double rearUtilisation = 0.0;
    double frontUtilisation = 0.0;
    std::size_t rearAirborneTicks = 0;
    std::uint32_t rearCycles = 0;
    bool stopped = false;
};

[[nodiscard]] ProductionRun recordProduction(const VehicleSetup& setup, const PhysicsWorld& world,
                                             const AssistSetup& assists, const double pedal, const double entrySpeed)
{
    auto state = VehicleState{};
    settle(setup, state, world, entrySpeed);

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

    auto run = ProductionRun{};
    const auto start = state.chassis.position;

    auto input = VehicleInput{};
    input.brake = pedal;

    auto force = std::array<double, 2>{};
    auto capacity = std::array<double, 2>{};

    for (auto step = 0; step < 360 * 30; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = pedal, .throttle = 0.0},
                                           brakeCircuitPressures(setup, pedal), tick);
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        auto rearOff = false;

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& solution = lastStep.corners[index];
            const auto axle = index < 2 ? std::size_t{0} : std::size_t{1};
            const auto load = solution.forces.tireVertical;

            force[axle] += std::abs(solution.contact.tyre.longitudinal);
            capacity[axle] += load * tyreFriction(setup.corners[index].tyre, TyreAxis::Longitudinal, load,
                                                  solution.patch.gripMultiplier);

            rearOff = rearOff || (axle == 1 && !lastStep.telemetry.wheels[index].inContact);
        }

        run.rearAirborneTicks += rearOff ? 1 : 0;
        run.rearCycles = command.channels.antilockCycles[2];

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            run.stopped = true;
            break;
        }
    }

    run.distance = state.chassis.position.z - start.z;
    run.frontUtilisation = capacity[0] > 1.0 ? force[0] / capacity[0] : 0.0;
    run.rearUtilisation = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0;

    return run;
}

} // namespace

TEST_CASE("the shadow controller is the production controller, to the bit, on every member", "[.rear-hop]")
{
    // **Section 0, and nothing below is worth reading without it.** The shadow reproduces
    // `updateAssists`'s outer loop and calls the production `advanceAntilockChannel` unmodified;
    // what it adds is an override applied after that call returns, disabled here. If the two arms
    // disagree by so much as a bit, every oracle result in this file is measuring the shadow's own
    // divergence rather than the intervention [check-the-control-variable].
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto oracle = RearOracle{};

    std::printf("\n=== the shadow against production, all %zu members ===\n", ensembleCount);
    std::printf("\n   member   entry [km/h]   shadow [m]   production [m]   identical\n");

    auto identical = std::size_t{0};

    for (auto index = std::size_t{0}; index < ensembleCount; index++)
    {
        const auto entry = memberEntry(index, ensembleCount);
        const auto mine = record(setup.value(), world.value(), assists, 1.0, oracle, entry);
        const auto theirs = recordProduction(setup.value(), world.value(), assists, 1.0, entry);
        const auto use = axleUse(mine);

        const auto same = mine.distance == theirs.distance && use.rearUtilisation == theirs.rearUtilisation &&
                          use.frontUtilisation == theirs.frontUtilisation &&
                          mine.rearAirborneTicks == theirs.rearAirborneTicks && mine.stopped == theirs.stopped;

        identical += same ? 1 : 0;

        if (index % 7 == 0 || !same)
        {
            std::printf("   %6zu   %12.4f   %10.6f   %14.6f   %s\n", index, 3.6 * entry, mine.distance, theirs.distance,
                        same ? "yes" : "NO");
        }

        REQUIRE(mine.distance == theirs.distance);
        REQUIRE(use.rearUtilisation == theirs.rearUtilisation);
        REQUIRE(use.frontUtilisation == theirs.frontUtilisation);
        REQUIRE(mine.rearAirborneTicks == theirs.rearAirborneTicks);
    }

    std::printf("\n  identical members: %zu of %zu\n", identical, ensembleCount);
    REQUIRE(identical == ensembleCount);
}

TEST_CASE("the three representative members: recovered, stranded, and the nearest boundary pair", "[.rear-hop]")
{
    // **Section 1.** The ensemble is re-run here rather than quoted, because a member index has to
    // be a member of *this* build's ensemble for the traces below to be traces of anything. The
    // clusters are read off rear utilisation against the 0.50 separator `[.brake-gap]` found in an
    // empty region of the data, and that emptiness is re-checked before the fractions are printed.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto members = ensembleOf(setup.value(), world.value(), assists, 1.0, RearOracle{});

    std::printf("\n=== the production anti-lock ensemble, member by member ===\n");
    std::printf("\n   idx   entry [km/h]   stop [m]   rear util  front util   rearOff  landings  rearCyc  frontCyc"
                "   minFz [N]   cluster\n");

    for (const auto& member : members)
    {
        std::printf("   %3zu   %12.4f   %8.3f   %9.4f  %10.4f   %7zu  %8zu  %7zu  %8zu   %9.1f   %s\n", member.index,
                    3.6 * member.entry, member.distance, member.rearUtilisation, member.frontUtilisation,
                    member.rearAirborne, member.landings, member.rearCycles, member.frontCycles, member.minimumRearLoad,
                    member.stranded() ? "STRANDED" : "recovered");

        REQUIRE(member.stopped);
    }

    const auto summary = summarise(members);

    std::printf("\n  stranded %zu of %zu (%.3f).  separator 0.50: highest stranded %.4f, lowest recovered %.4f — %s\n",
                summary.stranded, members.size(),
                static_cast<double>(summary.stranded) / static_cast<double>(members.size()), summary.highestStranded,
                summary.lowestRecovered, summary.lowestRecovered > summary.highestStranded ? "still empty" : "CLOSED");
    std::printf("  stop median %.3f m, P10 %.3f, P90 %.3f, sd %.3f.  four wheels down: %zu of %zu\n",
                summary.distance.median, summary.distance.p10, summary.distance.p90, summary.distance.deviation,
                summary.grounded, members.size());

    // The three representatives. "Clearly" is the extreme of each cluster rather than a member near
    // the separator, and the boundary pair is the closest ADJACENT pair whose cluster changes —
    // adjacent because the plant difference is then one ensemble step and nothing else.
    auto bestRecovered = std::size_t{0};
    auto worstStranded = std::size_t{0};
    auto haveRecovered = false;
    auto haveStranded = false;

    for (const auto& member : members)
    {
        if (!member.stranded() && (!haveRecovered || member.rearUtilisation > members[bestRecovered].rearUtilisation))
        {
            bestRecovered = member.index;
            haveRecovered = true;
        }

        if (member.stranded() && (!haveStranded || member.rearUtilisation < members[worstStranded].rearUtilisation))
        {
            worstStranded = member.index;
            haveStranded = true;
        }
    }

    REQUIRE(haveRecovered);
    REQUIRE(haveStranded);

    std::printf("\n  --- the representatives ---\n");
    std::printf("  A  clearly RECOVERED  member %zu, entry %.4f km/h, %.3f m, rear util %.4f\n", bestRecovered,
                3.6 * members[bestRecovered].entry, members[bestRecovered].distance,
                members[bestRecovered].rearUtilisation);
    std::printf("  B  clearly STRANDED   member %zu, entry %.4f km/h, %.3f m, rear util %.4f\n", worstStranded,
                3.6 * members[worstStranded].entry, members[worstStranded].distance,
                members[worstStranded].rearUtilisation);

    std::printf("\n  C  every ADJACENT pair whose cluster changes (the plant step between them is one member):\n");

    for (auto index = std::size_t{1}; index < members.size(); index++)
    {
        const auto& lower = members[index - 1];
        const auto& upper = members[index];

        if (lower.stranded() == upper.stranded())
        {
            continue;
        }

        std::printf("     %3zu (%.4f km/h, %s, %.3f m, util %.4f)  ->  %3zu (%.4f km/h, %s, %.3f m, util %.4f)"
                    "   d(entry) %.4f km/h, d(stop) %+.3f m\n",
                    lower.index, 3.6 * lower.entry, lower.stranded() ? "STR" : "rec", lower.distance,
                    lower.rearUtilisation, upper.index, 3.6 * upper.entry, upper.stranded() ? "STR" : "rec",
                    upper.distance, upper.rearUtilisation, 3.6 * (upper.entry - lower.entry),
                    upper.distance - lower.distance);
    }
}

TEST_CASE("every rear-channel transition the production state machine actually takes", "[.rear-hop]")
{
    // **Section 2, measured rather than read.** The transition conditions themselves are read off
    // `Assists/Impl/BrakeControlImpl.cpp` and restated in the deliverable; what this case adds is
    // which of them the rear channel actually takes, how often, and what the decision terms were
    // when it took them. Every term below is the same expression the production law evaluates,
    // applied to the channel state that call left behind — so nothing here is a second copy of the
    // law and nothing can drift from it.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto members = ensembleOf(setup.value(), world.value(), assists, 1.0, RearOracle{});
    auto stranded = std::size_t{0};
    auto recovered = std::size_t{0};
    auto haveStranded = false;
    auto haveRecovered = false;

    for (const auto& member : members)
    {
        if (member.stranded() && (!haveStranded || member.rearUtilisation < members[stranded].rearUtilisation))
        {
            stranded = member.index;
            haveStranded = true;
        }
        if (!member.stranded() && (!haveRecovered || member.rearUtilisation > members[recovered].rearUtilisation))
        {
            recovered = member.index;
            haveRecovered = true;
        }
    }

    REQUIRE(haveStranded);
    REQUIRE(haveRecovered);

    const auto names = std::array<const char*, 5>{"passive", "hold", "dump", "recover", "reapply"};

    const auto census = [&](const char* label, const std::size_t index)
    {
        const auto run =
            record(setup.value(), world.value(), assists, 1.0, RearOracle{}, memberEntry(index, ensembleCount), true);

        auto counts = std::array<std::array<std::size_t, 5>, 5>{};
        auto dwell = std::array<std::size_t, 5>{};
        auto steps = std::size_t{0};

        // What the world was doing at each transition, so a transition taken while the axle is in
        // the air is separable from the same transition taken on the road.
        auto airborneEntries = std::array<std::size_t, 5>{};
        auto loadAtEntry = std::array<double, 5>{};

        for (const auto& step : run.steps)
        {
            if (step.channel != 2)
            {
                continue;
            }

            steps++;
            const auto from = static_cast<std::size_t>(step.before);
            const auto to = static_cast<std::size_t>(step.after);
            counts[from][to]++;
            dwell[to]++;

            if (from != to)
            {
                airborneEntries[to] += step.truthContact ? 0 : 1;
                loadAtEntry[to] += step.truthLoadMin;
            }
        }

        std::printf("\n  --- %s member %zu: rear-channel transition census over %zu controller steps ---\n", label,
                    index, steps);
        std::printf("\n     from \\ to    passive     hold     dump  recover  reapply\n");

        for (auto from = std::size_t{0}; from < 5; from++)
        {
            std::printf("     %-10s", names[from]);
            for (auto to = std::size_t{0}; to < 5; to++)
            {
                if (from == to)
                {
                    std::printf("        .");
                    continue;
                }
                std::printf("   %6zu", counts[from][to]);
            }
            std::printf("\n");
        }

        std::printf("\n     phase dwell [controller steps / %% of stop]:");
        for (auto phase = std::size_t{0}; phase < 5; phase++)
        {
            std::printf("  %s %zu (%.1f%%)", names[phase], dwell[phase],
                        100.0 * static_cast<double>(dwell[phase]) /
                            static_cast<double>(std::max(steps, std::size_t{1})));
        }
        std::printf("\n");

        std::printf("\n     entries taken with a rear wheel OFF THE ROAD, and the mean rear axle load at entry:\n");
        for (auto phase = std::size_t{0}; phase < 5; phase++)
        {
            auto entries = std::size_t{0};
            for (auto from = std::size_t{0}; from < 5; from++)
            {
                entries += from == phase ? 0 : counts[from][phase];
            }

            std::printf("       -> %-8s  entries %5zu   airborne %5zu (%5.1f%%)   mean min rear Fz %8.1f N\n",
                        names[phase], entries, airborneEntries[phase],
                        100.0 * static_cast<double>(airborneEntries[phase]) /
                            static_cast<double>(std::max(entries, std::size_t{1})),
                        loadAtEntry[phase] / static_cast<double>(std::max(entries, std::size_t{1})));
        }
    };

    std::printf("\n=== the rear channel's reachable transitions, as taken ===\n");
    census("RECOVERED", recovered);
    census("STRANDED ", stranded);
}

TEST_CASE("the rear actuator: how long a decision takes to reach the caliper", "[.rear-hop]")
{
    // **Section 3.** Two halves. The first is arithmetic on the modulator's own stated gradients,
    // which is what the hardware can do; the second is what the loop actually spends, measured off
    // every dump entry and every re-apply entry the stranded member takes.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto pressures = brakeCircuitPressures(setup.value(), 1.0);
    const auto& modulator = assists.antilock.modulator;

    std::printf("\n=== the actuator, from its own stated rates ===\n");
    std::printf("\n  controller period          %.4f ms   (%.0f Hz), physics tick %.4f ms (%.0f Hz)\n",
                1000.0 / assists.controlRate, assists.controlRate, 1000.0 * tick, 1.0 / tick);
    std::printf("  dump gradient              %.3e Pa/s  (%.0f bar/s), front and rear the SAME valve\n",
                modulator.dumpGradient, modulator.dumpGradient / bar);
    std::printf("  re-apply gradient, front   %.3e Pa/s  (%.0f bar/s)\n", modulator.reapplyGradient,
                modulator.reapplyGradient / bar);
    std::printf("  re-apply gradient, REAR    %.3e Pa/s  (%.0f bar/s)   %s\n", modulator.rearReapplyGradient,
                modulator.rearReapplyGradient / bar,
                modulator.rearReapplyGradient == modulator.reapplyGradient ? "-- ships at the front's rate"
                                                                           : "-- metered separately");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto perPascal = pressures[index] > 0.0 ? setup->corners[index].brakeTorque / pressures[index] : 0.0;
        std::printf("  corner %zu: full pedal %6.2f bar, brake %8.1f N.m, %.3e N.m/Pa;  full dump %6.1f ms,"
                    "  full re-apply %6.1f ms\n",
                    index, pressures[index] / bar, setup->corners[index].brakeTorque, perPascal,
                    1000.0 * pressures[index] / modulator.dumpGradient,
                    1000.0 * pressures[index] /
                        (index < 2 ? modulator.reapplyGradient : modulator.rearReapplyGradient));
    }

    std::printf("\n  Notes on the seams this file could not find, because they do not exist in the model:\n");
    std::printf("    - no transport delay, no valve slew ramp, no minimum command, no residual torque:\n");
    std::printf("      the pressure integrates at a constant gradient from the step it is commanded on.\n");
    std::printf("    - the controller's `pressure` IS the actuator state; there is no plant/observer split,\n");
    std::printf("      so the loop observes its own actuator PERFECTLY and never has to estimate it.\n");
    std::printf("    - brake torque = pressure x (N.m/Pa) with no lag, so caliper torque is instantaneous\n");
    std::printf("      in the pressure. Every millisecond below is the PRESSURE's, not the pad's.\n");

    const auto members = ensembleOf(setup.value(), world.value(), assists, 1.0, RearOracle{});
    auto stranded = std::size_t{0};
    auto haveStranded = false;
    for (const auto& member : members)
    {
        if (member.stranded() && (!haveStranded || member.rearUtilisation < members[stranded].rearUtilisation))
        {
            stranded = member.index;
            haveStranded = true;
        }
    }
    REQUIRE(haveStranded);

    const auto run =
        record(setup.value(), world.value(), assists, 1.0, RearOracle{}, memberEntry(stranded, ensembleCount), true);

    // What the loop actually spends. A dump is measured to the moment the pressure is a tenth of
    // what it entered at; a re-apply to the moment it has recovered half of what the dump took off.
    auto dumpTimes = std::vector<double>{};
    auto dumpFractions = std::vector<double>{};
    auto reapplyTimes = std::vector<double>{};
    auto reapplyRegained = std::vector<double>{};

    const auto period = 1.0 / assists.controlRate;

    for (auto index = std::size_t{0}; index < run.steps.size(); index++)
    {
        const auto& step = run.steps[index];
        if (step.channel != 2)
        {
            continue;
        }

        if (step.before != ModulatorPhase::Dump && step.after == ModulatorPhase::Dump)
        {
            const auto entry = step.pressureBefore;
            auto low = entry;

            for (auto forward = index; forward < run.steps.size(); forward++)
            {
                const auto& later = run.steps[forward];
                if (later.channel != 2)
                {
                    continue;
                }
                if (later.after != ModulatorPhase::Dump)
                {
                    low = later.pressureAfter;
                    dumpTimes.push_back(later.time - step.time);
                    break;
                }
            }

            dumpFractions.push_back(entry > 1.0 ? low / entry : 0.0);
        }

        if (step.before != ModulatorPhase::Reapply && step.after == ModulatorPhase::Reapply)
        {
            const auto entry = step.pressureBefore;

            for (auto forward = index; forward < run.steps.size(); forward++)
            {
                const auto& later = run.steps[forward];
                if (later.channel != 2)
                {
                    continue;
                }
                if (later.after != ModulatorPhase::Reapply)
                {
                    reapplyTimes.push_back(later.time - step.time);
                    reapplyRegained.push_back((later.pressureAfter - entry) / bar);
                    break;
                }
            }
        }
    }

    const auto dumpDuration = distributionOf(dumpTimes);
    const auto dumpDepth = distributionOf(dumpFractions);
    const auto reapplyDuration = distributionOf(reapplyTimes);
    const auto regained = distributionOf(reapplyRegained);

    std::printf("\n=== and what the LOOP spends, stranded member %zu ===\n", stranded);
    std::printf("\n  rear dump episodes     %3zu   duration ms  min %6.2f  median %6.2f  max %7.2f   "
                "(= %.1f / %.1f controller steps at the median / max)\n",
                dumpDuration.n, 1000.0 * dumpDuration.minimum, 1000.0 * dumpDuration.median,
                1000.0 * dumpDuration.maximum, dumpDuration.median / period, dumpDuration.maximum / period);
    std::printf("  ...pressure left at exit, as a fraction of entry: min %.3f median %.3f max %.3f\n",
                dumpDepth.minimum, dumpDepth.median, dumpDepth.maximum);
    std::printf("  rear re-apply episodes %3zu   duration ms  min %6.2f  median %6.2f  max %7.2f\n", reapplyDuration.n,
                1000.0 * reapplyDuration.minimum, 1000.0 * reapplyDuration.median, 1000.0 * reapplyDuration.maximum);
    std::printf("  ...bar regained per re-apply episode: min %+7.2f median %+7.2f max %+7.2f\n", regained.minimum,
                regained.median, regained.maximum);
}

namespace
{

// The three members every trace case below uses, found the same way every time so the indices agree
// across cases.
struct Representatives
{
    std::size_t recovered = 0;
    std::size_t stranded = 0;
    std::size_t boundaryLower = 0;
    std::size_t boundaryUpper = 0;
    std::vector<Member> members;
};

[[nodiscard]] Representatives representativesOf(const VehicleSetup& setup, const PhysicsWorld& world,
                                                const AssistSetup& assists)
{
    auto found = Representatives{};
    found.members = ensembleOf(setup, world, assists, 1.0, RearOracle{});

    auto haveRecovered = false;
    auto haveStranded = false;

    for (const auto& member : found.members)
    {
        if (!member.stranded() &&
            (!haveRecovered || member.rearUtilisation > found.members[found.recovered].rearUtilisation))
        {
            found.recovered = member.index;
            haveRecovered = true;
        }
        if (member.stranded() &&
            (!haveStranded || member.rearUtilisation < found.members[found.stranded].rearUtilisation))
        {
            found.stranded = member.index;
            haveStranded = true;
        }
    }

    REQUIRE(haveRecovered);
    REQUIRE(haveStranded);

    auto haveBoundary = false;
    for (auto index = std::size_t{1}; index < found.members.size(); index++)
    {
        if (found.members[index - 1].stranded() != found.members[index].stranded())
        {
            found.boundaryLower = index - 1;
            found.boundaryUpper = index;
            haveBoundary = true;
            break;
        }
    }

    REQUIRE(haveBoundary);

    return found;
}

// The last controller step of each physics tick, per channel, so a per-tick table can print what the
// ECU had decided by the time the tick was integrated.
[[nodiscard]] std::vector<const ControlStep*> perTick(const Run& run, const std::size_t channel)
{
    auto latest = std::vector<const ControlStep*>(run.samples.size(), nullptr);

    for (const auto& step : run.steps)
    {
        if (step.channel != channel)
        {
            continue;
        }

        const auto index = static_cast<std::size_t>(step.time / tick);
        if (index < latest.size())
        {
            latest[index] = &step;
        }
    }

    return latest;
}

} // namespace

TEST_CASE("one sustained hop cycle, event by event", "[.rear-hop]")
{
    // **Sections 4 and 5.** The cycle is delimited by a PHYSICAL event — the rear axle's load
    // crossing back up through a tenth of its static value — and everything else is reported in the
    // order the trace shows it rather than in the order the hypothesis expects.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);
    const auto run = record(setup.value(), world.value(), assists, 1.0, RearOracle{},
                            memberEntry(found.stranded, ensembleCount), true);
    const auto steps = perTick(run, 2);

    // The static rear wheel load, from the first tick of the roll before the pedal moved.
    const auto staticRear = 0.5 * run.samples.front().rearAxleLoad;

    std::printf("\n=== the hop, stranded member %zu (entry %.4f km/h, %.3f m) ===\n", found.stranded,
                3.6 * memberEntry(found.stranded, ensembleCount), run.distance);
    std::printf("  static rear axle load at the braking point: %.1f N (%.1f N per wheel)\n",
                run.samples.front().rearAxleLoad, staticRear);

    // Find the sustained part: the last contact loss is deep in the stop, so take the cycle whose
    // start is nearest the middle of the airborne episodes.
    auto losses = std::vector<std::size_t>{};
    for (auto index = std::size_t{1}; index < run.samples.size(); index++)
    {
        if (run.samples[index - 1].rearContact && !run.samples[index].rearContact)
        {
            losses.push_back(index);
        }
    }

    std::printf("  rear contact losses: %zu, first at %.4f s, last at %.4f s\n", losses.size(),
                losses.empty() ? 0.0 : run.samples[losses.front()].time,
                losses.empty() ? 0.0 : run.samples[losses.back()].time);

    REQUIRE(!losses.empty());

    // A cycle in the steady part: skip the entry transient entirely.
    auto chosen = losses.front();
    for (const auto loss : losses)
    {
        if (run.samples[loss].time > 1.0)
        {
            chosen = loss;
            break;
        }
    }

    const auto from = chosen > 40 ? chosen - 40 : std::size_t{0};
    const auto to = std::min(run.samples.size(), chosen + 150);

    std::printf("\n  --- the cycle around the contact loss at %.4f s, every physics tick ---\n",
                run.samples[chosen].time);
    std::printf("\n      t [s]  ev |  Fz_RL   Fz_RR   Fx_RL   util |  omega   v_car   v_ref    slip  est.slip"
                "   dw/dt |   T_req   T_act  |  phase    bar |  travel   damper |  pitch  pRate\n");

    auto previousPhase = ModulatorPhase::Passive;
    auto previousContact = true;

    for (auto index = from; index < to; index++)
    {
        const auto& sample = run.samples[index];
        const auto& rl = sample.wheels[2];
        const auto& rr = sample.wheels[3];
        const auto* step = steps[index];

        auto event = "  ";
        if (previousContact && !sample.rearContact)
        {
            event = "LO";
        }
        else if (!previousContact && sample.rearContact)
        {
            event = "LA";
        }
        else if (rl.phase != previousPhase)
        {
            switch (rl.phase)
            {
            case ModulatorPhase::Dump:
                event = "D>";
                break;
            case ModulatorPhase::Recover:
                event = "R>";
                break;
            case ModulatorPhase::Reapply:
                event = "A>";
                break;
            case ModulatorPhase::Hold:
                event = "H>";
                break;
            case ModulatorPhase::Passive:
                event = "P>";
                break;
            }
        }

        previousPhase = rl.phase;
        previousContact = sample.rearContact;

        std::printf("   %8.4f  %s | %6.0f  %6.0f  %6.0f  %5.3f | %6.2f  %6.2f  %6.2f  %6.3f  %8.3f  %7.0f |"
                    " %7.1f %7.1f  | %-7s %5.1f | %+6.4f  %+7.3f | %+6.3f %+6.3f\n",
                    sample.time, event, rl.load, rr.load, rl.forceLongitudinal, rl.utilisation(), rl.wheelSpeed,
                    sample.speed, sample.referenceSpeed, rl.slipRatio, rl.estimatedSlip, rl.wheelAcceleration,
                    step != nullptr ? step->request * setup->corners[2].brakeTorque /
                                          std::max(brakeCircuitPressures(setup.value(), 1.0)[2], 1.0)
                                    : 0.0,
                    rl.brakeTorque, modulatorName(rl.phase), rl.pressure / bar, rl.suspensionTravel, rl.damperVelocity,
                    sample.pitch, sample.pitchRate);
    }

    std::printf("\n  events: LO contact lost, LA contact restored, D> dump entered, R> recover, A> re-apply,"
                " H> hold, P> passive.\n");
    std::printf("  omega is the wheel's own rad/s; dw/dt is its angular acceleration, rad/s^2.\n");
    std::printf("  T_req is what the channel is asking the caliper for, T_act what it got, both N.m.\n");

    // The sequence, stated as the trace shows it rather than as the brief guessed it.
    std::printf("\n  --- the same cycle as an event list ---\n");

    previousContact = true;
    previousPhase = ModulatorPhase::Passive;

    for (auto index = from; index < to; index++)
    {
        const auto& sample = run.samples[index];
        const auto& rl = sample.wheels[2];

        if (previousContact != sample.rearContact)
        {
            std::printf("   %8.4f  %-22s  Fz %6.0f N   slip %6.3f   phase %-7s   bar %5.1f   omega %6.2f\n",
                        sample.time, sample.rearContact ? "CONTACT RESTORED" : "CONTACT LOST", sample.rearAxleLoad,
                        rl.slipRatio, modulatorName(rl.phase), rl.pressure / bar, rl.wheelSpeed);
        }

        if (previousPhase != rl.phase)
        {
            std::printf("   %8.4f  -> %-19s  Fz %6.0f N   slip %6.3f   est %6.3f   bar %5.1f   dw/dt %7.0f\n",
                        sample.time, modulatorName(rl.phase), sample.rearAxleLoad, rl.slipRatio, rl.estimatedSlip,
                        rl.pressure / bar, rl.wheelAcceleration);
        }

        previousContact = sample.rearContact;
        previousPhase = rl.phase;
    }
}

TEST_CASE("recovered against stranded, aligned on the first rear unload", "[.rear-hop]")
{
    // **Section 6.** Both members are aligned on the FIRST tick either of them loses rear contact,
    // and compared forward until they part. The question is what happens *before* the stranded
    // member is obviously stranded, so nothing after the divergence is offered as a cause.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);

    const auto a = record(setup.value(), world.value(), assists, 1.0, RearOracle{},
                          memberEntry(found.recovered, ensembleCount), true);
    const auto b = record(setup.value(), world.value(), assists, 1.0, RearOracle{},
                          memberEntry(found.stranded, ensembleCount), true);

    std::printf("\n=== recovered member %zu against stranded member %zu ===\n", found.recovered, found.stranded);
    std::printf("  recovered  entry %.4f km/h  %.3f m  rear util %.4f  rear off %zu ticks  landings %zu\n",
                3.6 * memberEntry(found.recovered, ensembleCount), a.distance, axleUse(a).rearUtilisation,
                a.rearAirborneTicks, a.landings);
    std::printf("  stranded   entry %.4f km/h  %.3f m  rear util %.4f  rear off %zu ticks  landings %zu\n",
                3.6 * memberEntry(found.stranded, ensembleCount), b.distance, axleUse(b).rearUtilisation,
                b.rearAirborneTicks, b.landings);

    const auto firstLoss = [](const Run& run)
    {
        for (auto index = std::size_t{0}; index < run.samples.size(); index++)
        {
            if (!run.samples[index].rearContact)
            {
                return index;
            }
        }
        return run.samples.size();
    };

    const auto lossA = firstLoss(a);
    const auto lossB = firstLoss(b);

    REQUIRE(lossA < a.samples.size());
    REQUIRE(lossB < b.samples.size());

    std::printf("\n  first rear contact loss: recovered tick %zu (%.4f s), stranded tick %zu (%.4f s)\n", lossA,
                a.samples[lossA].time, lossB, b.samples[lossB].time);

    // Where the two trajectories first part, channel by channel. Every threshold below is a
    // tolerance on a printed quantity and NOT a criterion.
    const auto rows = std::min(a.samples.size() - lossA, b.samples.size() - lossB);

    auto firstPhase = rows;
    auto firstPressure = rows;
    auto firstLoad = rows;
    auto firstSlip = rows;
    auto firstContact = rows;

    for (auto offset = std::size_t{0}; offset < rows; offset++)
    {
        const auto& sa = a.samples[lossA + offset];
        const auto& sb = b.samples[lossB + offset];

        if (firstPhase == rows && sa.wheels[2].phase != sb.wheels[2].phase)
        {
            firstPhase = offset;
        }
        if (firstPressure == rows && std::abs(sa.wheels[2].pressure - sb.wheels[2].pressure) > 0.5 * bar)
        {
            firstPressure = offset;
        }
        if (firstLoad == rows && std::abs(sa.rearAxleLoad - sb.rearAxleLoad) > 200.0)
        {
            firstLoad = offset;
        }
        if (firstSlip == rows && std::abs(sa.wheels[2].slipRatio - sb.wheels[2].slipRatio) > 0.05)
        {
            firstSlip = offset;
        }
        if (firstContact == rows && sa.rearContact != sb.rearContact)
        {
            firstContact = offset;
        }
    }

    std::printf("\n  first divergence, in ticks after the aligned first unload (%.4f ms each):\n", 1000.0 * tick);
    std::printf("    modulator phase      %s\n",
                firstPhase < rows ? std::to_string(firstPhase).c_str() : "never within the aligned window");
    std::printf("    rear pressure > 0.5 bar apart   %s\n",
                firstPressure < rows ? std::to_string(firstPressure).c_str() : "never");
    std::printf("    rear axle load > 200 N apart    %s\n",
                firstLoad < rows ? std::to_string(firstLoad).c_str() : "never");
    std::printf("    rear slip > 0.05 apart          %s\n",
                firstSlip < rows ? std::to_string(firstSlip).c_str() : "never");
    std::printf("    contact state differs           %s\n",
                firstContact < rows ? std::to_string(firstContact).c_str() : "never");

    std::printf("\n  --- aligned, every 4th tick, through the first 0.9 s after the unload ---\n");
    std::printf("\n     +ticks |  recovered: Fz    slip  phase     bar  omega  cont |  stranded: Fz    slip"
                "  phase     bar  omega  cont\n");

    for (auto offset = std::size_t{0}; offset < rows && offset < 324; offset += 4)
    {
        const auto& sa = a.samples[lossA + offset];
        const auto& sb = b.samples[lossB + offset];

        std::printf("     %6zu | %14.0f  %6.3f  %-7s %5.1f %6.2f  %3s | %13.0f  %6.3f  %-7s %5.1f %6.2f  %3s\n", offset,
                    sa.rearAxleLoad, sa.wheels[2].slipRatio, modulatorName(sa.wheels[2].phase),
                    sa.wheels[2].pressure / bar, sa.wheels[2].wheelSpeed, sa.rearContact ? "on" : "OFF",
                    sb.rearAxleLoad, sb.wheels[2].slipRatio, modulatorName(sb.wheels[2].phase),
                    sb.wheels[2].pressure / bar, sb.wheels[2].wheelSpeed, sb.rearContact ? "on" : "OFF");
    }
}

TEST_CASE("the boundary pair, where the plant difference is one ensemble step", "[.rear-hop]")
{
    // **Section 7, and it is the strongest instrument in the file**: two adjacent members of the
    // same deterministic sweep whose clusters differ. Everything about the car is identical and the
    // entry speed differs by 0.05% of itself, so whatever separates them is inside the loop.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);
    const auto& lower = found.members[found.boundaryLower];
    const auto& upper = found.members[found.boundaryUpper];

    const auto a = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, lower.entry, true);
    const auto b = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, upper.entry, true);

    std::printf("\n=== the boundary pair: members %zu and %zu ===\n", lower.index, upper.index);
    std::printf("  %zu  entry %.5f km/h  %.3f m  rear util %.4f  %s\n", lower.index, 3.6 * lower.entry, a.distance,
                axleUse(a).rearUtilisation, lower.stranded() ? "STRANDED" : "recovered");
    std::printf("  %zu  entry %.5f km/h  %.3f m  rear util %.4f  %s\n", upper.index, 3.6 * upper.entry, b.distance,
                axleUse(b).rearUtilisation, upper.stranded() ? "STRANDED" : "recovered");
    std::printf("  entry difference %.5f km/h (%.4f%% of the entry speed)\n", 3.6 * (upper.entry - lower.entry),
                100.0 * (upper.entry - lower.entry) / lower.entry);

    const auto rows = std::min(a.samples.size(), b.samples.size());
    const auto stepsA = perTick(a, 2);
    const auto stepsB = perTick(b, 2);

    auto firstPhase = rows;
    auto firstPressure = rows;
    auto firstSlip = rows;
    auto firstContact = rows;
    auto firstLoad = rows;
    auto firstDecision = rows;

    for (auto index = std::size_t{0}; index < rows; index++)
    {
        const auto& sa = a.samples[index];
        const auto& sb = b.samples[index];

        if (firstLoad == rows && std::abs(sa.rearAxleLoad - sb.rearAxleLoad) > 50.0)
        {
            firstLoad = index;
        }
        if (firstSlip == rows && std::abs(sa.wheels[2].slipRatio - sb.wheels[2].slipRatio) > 0.02)
        {
            firstSlip = index;
        }
        if (firstPhase == rows && sa.wheels[2].phase != sb.wheels[2].phase)
        {
            firstPhase = index;
        }
        if (firstPressure == rows && std::abs(sa.wheels[2].pressure - sb.wheels[2].pressure) > 0.5 * bar)
        {
            firstPressure = index;
        }
        if (firstContact == rows && sa.rearContact != sb.rearContact)
        {
            firstContact = index;
        }
        if (firstDecision == rows && stepsA[index] != nullptr && stepsB[index] != nullptr &&
            (stepsA[index]->losing != stepsB[index]->losing || stepsA[index]->pastBand != stepsB[index]->pastBand ||
             stepsA[index]->surging != stepsB[index]->surging))
        {
            firstDecision = index;
        }
    }

    const auto when = [&](const std::size_t index)
    {
        return index < rows ? a.samples[index].time : -1.0;
    };

    std::printf("\n  --- in chronological order ---\n");
    std::printf("    first PHYSICAL divergence   rear axle load > 50 N apart      tick %5zu   t = %+8.4f s\n",
                firstLoad, when(firstLoad));
    std::printf("    first WHEEL-SLIP divergence slip > 0.02 apart                tick %5zu   t = %+8.4f s\n",
                firstSlip, when(firstSlip));
    std::printf("    first CONTROLLER-TERM split losing/pastBand/surging disagree tick %5zu   t = %+8.4f s\n",
                firstDecision, when(firstDecision));
    std::printf("    first CONTROLLER-STATE split modulator phase differs         tick %5zu   t = %+8.4f s\n",
                firstPhase, when(firstPhase));
    std::printf("    first ACTUATOR divergence   rear pressure > 0.5 bar apart    tick %5zu   t = %+8.4f s\n",
                firstPressure, when(firstPressure));
    std::printf("    first CONTACT divergence    one axle up, the other down      tick %5zu   t = %+8.4f s\n",
                firstContact, when(firstContact));

    const auto first = std::min({firstLoad, firstSlip, firstDecision, firstPhase, firstPressure, firstContact});
    const auto window = first < rows ? first : std::size_t{0};

    std::printf("\n  --- around the earliest of them, every tick ---\n");
    std::printf("\n      t [s] |  A: Fz_axle    slip  phase     bar  cont  losing pastB |  B: Fz_axle    slip"
                "  phase     bar  cont  losing pastB\n");

    const auto printWindow = [&](const std::size_t at, const std::size_t span)
    {
        const auto begin = at > 20 ? at - 20 : std::size_t{0};

        for (auto index = begin; index < std::min(rows, at + span); index++)
        {
            const auto& sa = a.samples[index];
            const auto& sb = b.samples[index];
            const auto* ca = stepsA[index];
            const auto* cb = stepsB[index];

            std::printf("   %8.4f | %10.0f  %6.3f  %-7s %5.1f  %4s  %6s %5s | %10.0f  %6.3f  %-7s %5.1f  %4s"
                        "  %6s %5s\n",
                        sa.time, sa.rearAxleLoad, sa.wheels[2].slipRatio, modulatorName(sa.wheels[2].phase),
                        sa.wheels[2].pressure / bar, sa.rearContact ? "on" : "OFF",
                        ca != nullptr && ca->losing ? "yes" : "no", ca != nullptr && ca->pastBand ? "yes" : "no",
                        sb.rearAxleLoad, sb.wheels[2].slipRatio, modulatorName(sb.wheels[2].phase),
                        sb.wheels[2].pressure / bar, sb.rearContact ? "on" : "OFF",
                        cb != nullptr && cb->losing ? "yes" : "no", cb != nullptr && cb->pastBand ? "yes" : "no");
        }
    };

    printWindow(window, 40);

    std::printf("\n  --- and around the first PHYSICAL divergence, every tick ---\n");
    std::printf("\n      t [s] |  A: Fz_axle    slip  phase     bar  cont  losing pastB |  B: Fz_axle    slip"
                "  phase     bar  cont  losing pastB\n");
    printWindow(firstLoad < rows ? firstLoad : std::size_t{0}, 60);
}

TEST_CASE("what the rear channel knows when it decides to re-apply", "[.rear-hop]")
{
    // **Section 8, and the hypothesis is tested rather than assumed.** Every entry to `Reapply` on
    // the rear channel, across the whole ensemble, with the ground truth at that instant beside the
    // ECU's own belief. The two populations — recovered members and stranded members — are then
    // compared, because the hypothesis is a claim about a DIFFERENCE and not about a level.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);

    struct Entries
    {
        std::vector<double> load;
        std::vector<double> loadFraction;
        std::vector<double> loadRate;
        std::vector<double> slip;
        std::vector<double> estimated;
        std::vector<double> acceleration;
        std::vector<double> torque;
        std::vector<double> force;
        std::vector<double> utilisation;
        std::size_t airborne = 0;
        std::size_t total = 0;
    };

    auto recoveredEntries = Entries{};
    auto strandedEntries = Entries{};

    for (const auto& member : found.members)
    {
        const auto run = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, member.entry, true);
        auto& into = member.stranded() ? strandedEntries : recoveredEntries;

        const auto staticRear = 0.5 * run.samples.front().rearAxleLoad;

        auto previousPhase = ModulatorPhase::Passive;

        for (auto index = std::size_t{1}; index < run.samples.size(); index++)
        {
            const auto& sample = run.samples[index];
            const auto& previous = run.samples[index - 1];
            const auto& rl = sample.wheels[2];

            const auto entered = rl.phase == ModulatorPhase::Reapply && previousPhase != ModulatorPhase::Reapply;
            previousPhase = rl.phase;

            if (!entered)
            {
                continue;
            }

            into.total++;
            into.airborne += sample.rearContact ? 0 : 1;
            into.load.push_back(sample.rearAxleLoad);
            into.loadFraction.push_back(sample.rearAxleLoad / std::max(2.0 * staticRear, 1.0));
            into.loadRate.push_back((sample.rearAxleLoad - previous.rearAxleLoad) / tick);
            into.slip.push_back(std::abs(rl.slipRatio));
            into.estimated.push_back(std::abs(rl.estimatedSlip));
            into.acceleration.push_back(rl.wheelAcceleration);
            into.torque.push_back(rl.brakeTorque);
            into.force.push_back(std::abs(rl.forceLongitudinal));
            into.utilisation.push_back(rl.utilisation());
        }
    }

    const auto report = [](const char* label, const Entries& entries)
    {
        std::printf("\n  --- %s: %zu rear re-apply entries, %zu of them with a rear wheel OFF THE ROAD (%.1f%%) ---\n",
                    label, entries.total, entries.airborne,
                    100.0 * static_cast<double>(entries.airborne) /
                        static_cast<double>(std::max(entries.total, std::size_t{1})));

        const auto row = [](const char* name, const std::vector<double>& values, const char* unit)
        {
            const auto distribution = distributionOf(values);
            std::printf("    %-28s  min %10.3f   P10 %10.3f   MEDIAN %10.3f   P90 %10.3f   max %10.3f  %s\n", name,
                        distribution.minimum, distribution.p10, distribution.median, distribution.p90,
                        distribution.maximum, unit);
        };

        row("rear axle load", entries.load, "N");
        row("...as a fraction of static", entries.loadFraction, "-");
        row("dFz/dt (axle)", entries.loadRate, "N/s");
        row("true slip ratio", entries.slip, "-");
        row("ECU's estimated slip", entries.estimated, "-");
        row("wheel angular acceleration", entries.acceleration, "rad/s^2");
        row("brake torque at the wheel", entries.torque, "N.m");
        row("tyre longitudinal force", entries.force, "N");
        row("tyre utilisation", entries.utilisation, "-");
    };

    std::printf("\n=== the state of the world at every rear RE-APPLY entry, across the whole ensemble ===\n");
    report("RECOVERED members", recoveredEntries);
    report("STRANDED  members", strandedEntries);
}

TEST_CASE("what the production controller can and cannot see about rear contact", "[.rear-hop]")
{
    // **Section 9, and it is an architecture inventory with one measurement in it.**
    //
    // DIRECTLY AVAILABLE to `advanceAntilockChannel` today, from its own parameters and state:
    //   - the tone ring's speed reading and its age (`WheelSpeedReading::speed`, `.age`, `.valid`)
    //   - peripheral acceleration between tooth crossings (`AntilockChannelState::acceleration`)
    //   - the reference speed, its validity and its rate (`ReferenceSpeedState`)
    //   - the estimated slip, and the staleness-projected guard slip
    //   - its own commanded pressure and the request above it
    //   - elsewhere in the layer: yaw rate, lateral acceleration, the brake light switch
    //
    // DERIVABLE WITHOUT NEW PHYSICS — every term is already a parameter of the layer:
    //   - commanded brake torque: `pressure * brakeTorquePerPressure[wheel]`, already computed in
    //     `updateAssists`
    //   - the wheel's own angular acceleration: the peripheral figure over the nominal radius
    //   - **and therefore the tyre's longitudinal force**, from the wheel's equation of motion:
    //     `Fx ~= (T_brake + I * dw/dt) / r`. `I` is ECU calibration of the same kind
    //     `brakeTorquePerPressure` already is. This is the one derivable quantity that separates the
    //     two cases the controller currently confuses, and the measurement below is of it.
    //
    // NOT AVAILABLE, and a real road-car ECU does not have it either:
    //   - tyre normal load, the contact flag, suspension travel or damper velocity
    //   - the surface friction coefficient, and therefore the tyre's own peak slip
    //   - true road speed
    //
    // Nothing is implemented here. What is measured is whether the derivable quantity actually
    // separates "unloaded wheel spinning freely" from "loaded wheel recovering from slip", using
    // ONLY signals in the first two lists.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);
    const auto inertia = raceengine::wheelInertias(setup.value());
    const auto radius = assists.reference.nominalRadius;
    const auto perPascal = assists.brakeTorquePerPressure[2];

    std::printf("\n=== can a controller-observable quantity tell an unloaded rear wheel from a slipping one? ===\n");
    std::printf("  rear wheel inertia %.4f kg.m^2, ECU nominal radius %.4f m, %.3e N.m/Pa\n", inertia[2], radius,
                perPascal);
    std::printf("  the estimator is  Fx_hat = (T_brake + I * dw/dt) / r, with dw/dt the tone ring's own\n");
    std::printf("  peripheral acceleration over the nominal radius. Nothing in it is privileged.\n");

    struct Bucket
    {
        std::vector<double> estimate;
        std::vector<double> truth;
        std::vector<double> load;
        std::size_t count = 0;
    };

    auto airborne = Bucket{};
    auto loaded = Bucket{};

    for (const auto& member : found.members)
    {
        const auto run = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, member.entry, true);
        const auto steps = perTick(run, 2);

        for (auto index = std::size_t{0}; index < run.samples.size(); index++)
        {
            const auto* step = steps[index];
            if (step == nullptr || !step->referenceValid)
            {
                continue;
            }

            const auto& sample = run.samples[index];
            const auto& rl = sample.wheels[2];

            // Only while the channel is actually intervening: a passive channel is not the case in
            // question, and including it would flatter the separation with thousands of easy rows.
            if (rl.phase == ModulatorPhase::Passive)
            {
                continue;
            }

            const auto torque = step->pressureAfter * perPascal;
            const auto angular = step->acceleration / radius;
            const auto estimate = (torque + inertia[2] * angular) / radius;

            auto& bucket = sample.rearContact ? loaded : airborne;
            bucket.count++;
            bucket.estimate.push_back(estimate);
            bucket.truth.push_back(std::abs(rl.forceLongitudinal));
            bucket.load.push_back(sample.rearAxleLoad);
        }
    }

    const auto report = [](const char* label, const Bucket& bucket)
    {
        const auto estimate = distributionOf(bucket.estimate);
        const auto truth = distributionOf(bucket.truth);
        const auto load = distributionOf(bucket.load);

        std::printf("\n  --- %s: %zu intervening ticks ---\n", label, bucket.count);
        std::printf("    Fx_hat [N] from ECU signals   min %9.1f  P10 %9.1f  MEDIAN %9.1f  P90 %9.1f  max %9.1f\n",
                    estimate.minimum, estimate.p10, estimate.median, estimate.p90, estimate.maximum);
        std::printf("    |Fx| [N] TRUE                 min %9.1f  P10 %9.1f  MEDIAN %9.1f  P90 %9.1f  max %9.1f\n",
                    truth.minimum, truth.p10, truth.median, truth.p90, truth.maximum);
        std::printf("    rear axle Fz [N] TRUE         min %9.1f  P10 %9.1f  MEDIAN %9.1f  P90 %9.1f  max %9.1f\n",
                    load.minimum, load.p10, load.median, load.p90, load.maximum);
    };

    report("rear axle OFF the road", airborne);
    report("rear axle ON  the road", loaded);

    // How well it would separate, swept over a threshold on the estimate. Reported as a table and
    // NOT as a chosen number: picking one is controller design and this file does not do it.
    std::printf("\n  --- separation, swept (a table, not a threshold) ---\n");
    std::printf("\n    Fx_hat below [N]   airborne ticks caught   loaded ticks wrongly caught\n");

    for (const auto level : std::array<double, 7>{0.0, 100.0, 200.0, 400.0, 800.0, 1200.0, 2000.0})
    {
        auto caught = std::size_t{0};
        auto wrong = std::size_t{0};

        for (const auto value : airborne.estimate)
        {
            caught += value < level ? 1 : 0;
        }
        for (const auto value : loaded.estimate)
        {
            wrong += value < level ? 1 : 0;
        }

        std::printf("    %16.0f   %8zu / %8zu (%5.1f%%)   %8zu / %8zu (%5.1f%%)\n", level, caught, airborne.count,
                    100.0 * static_cast<double>(caught) / static_cast<double>(std::max(airborne.count, std::size_t{1})),
                    wrong, loaded.count,
                    100.0 * static_cast<double>(wrong) / static_cast<double>(std::max(loaded.count, std::size_t{1})));
    }
}

TEST_CASE("the diagnostic oracles, on the full ensemble", "[.rear-hop]")
{
    // **Sections 10, 11 and 12.** Every arm is the same 29 members, the same plant and the same
    // production state machine; only the probe-local rear actuator override differs. The primary
    // criterion is the stranded-rear fraction and not the stopping distance.
    //
    // **A PHYSICS ORACLE is not a proposal.** Arms A and B read the simulator's own rear tyre loads
    // and contact flags — signals no ECU has (section 9) — deliberately, because the first job is
    // causal proof. Arm B2 is the timing-only control: it knows nothing about contact at all.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    struct Arm
    {
        const char* name;
        RearOracle oracle;
    };

    const auto arms = std::vector<Arm>{
        {"production (control)", RearOracle{}},
        {"A1 gate: Fz <= 0", RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.0}},
        {"A2 gate: contact flag false", RearOracle{.mode = RearMode::ContactGate, .useContactFlag = true}},
        {"A3 gate: Fz <= 0.10 x static", RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.10}},
        {"A4 gate: Fz <= 0.25 x static", RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.25}},
        {"A5 gate: Fz <= 0.50 x static", RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.50}},
        {"B1 landing delay 20 ms", RearOracle{.mode = RearMode::LandingDelay, .delay = 0.020}},
        {"B2 landing delay 50 ms", RearOracle{.mode = RearMode::LandingDelay, .delay = 0.050}},
        {"B3 landing delay 100 ms", RearOracle{.mode = RearMode::LandingDelay, .delay = 0.100}},
        {"B4 landing delay 200 ms", RearOracle{.mode = RearMode::LandingDelay, .delay = 0.200}},
        {"T1 blind delay 20 ms  (timing)", RearOracle{.mode = RearMode::BlindDelay, .delay = 0.020}},
        {"T2 blind delay 50 ms  (timing)", RearOracle{.mode = RearMode::BlindDelay, .delay = 0.050}},
        {"T3 blind delay 100 ms (timing)", RearOracle{.mode = RearMode::BlindDelay, .delay = 0.100}},
        {"T4 blind delay 200 ms (timing)", RearOracle{.mode = RearMode::BlindDelay, .delay = 0.200}},
        {"C1 torque cut 20 ms at landing", RearOracle{.mode = RearMode::LandingTorqueCut, .delay = 0.020}},
        {"C2 torque cut 50 ms at landing", RearOracle{.mode = RearMode::LandingTorqueCut, .delay = 0.050}},
        {"C3 torque cut 100 ms at landing", RearOracle{.mode = RearMode::LandingTorqueCut, .delay = 0.100}}};

    std::printf("\n=== the diagnostic oracles, %zu members each ===\n", ensembleCount);
    printArmHeader();

    auto baseline = ArmSummary{};

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto members = ensembleOf(setup.value(), world.value(), assists, 1.0, arms[index].oracle);
        const auto summary = summarise(members);

        if (index == 0)
        {
            baseline = summary;
        }

        printArm(arms[index].name, summary);

        auto blocked = 0.0;
        for (const auto& member : members)
        {
            blocked += member.gatedSeconds;
        }

        if (index > 0)
        {
            std::printf("      %-36s  stranded %+d, stop median %+.3f m, rear util %+.4f, mean gated %.4f s\n", "",
                        static_cast<int>(summary.stranded) - static_cast<int>(baseline.stranded),
                        summary.distance.median - baseline.distance.median,
                        summary.rearUtilisation.median - baseline.rearUtilisation.median,
                        blocked / static_cast<double>(members.size()));
        }
    }

    std::printf("\n  Read the stranded column first. The classification of each arm is in the deliverable.\n");
    std::printf("  'mean gated' is how long per stop the override actually held the rear pressure down.\n");
}

TEST_CASE("the actuator against the state decision", "[.rear-hop]")
{
    // **Section 13.** Two complementary counterfactuals, both probe-local.
    //   13a IDEAL ACTUATOR: the production state machine, its demand realised with no valve lag.
    //   13b IDEAL DECISION: the production valve, driven by a per-wheel peak-slip oracle instead of
    //       the state machine.
    // Between them they say which half of the loop owns the limit cycle.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    struct Arm
    {
        const char* name;
        RearOracle oracle;
    };

    const auto arms = std::vector<Arm>{
        {"production (control)", RearOracle{}},
        {"13a rear actuator IDEAL", RearOracle{.mode = RearMode::IdealActuator}},
        {"13b rear DECISION ideal, real valve", RearOracle{.mode = RearMode::OracleThroughValve}},
        {"A1 gate: Fz <= 0 (for reference)", RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.0}}};

    std::printf("\n=== actuator lag against state decision ===\n");
    printArmHeader();

    for (const auto& arm : arms)
    {
        const auto members = ensembleOf(setup.value(), world.value(), assists, 1.0, arm.oracle);
        printArm(arm.name, summarise(members));
    }
}

TEST_CASE("the front channel while the rear is held by an oracle", "[.rear-hop]")
{
    // **Section 14.** The rear channel takes the minimal oracle that breaks the hop; the front
    // channel is production and untouched. What is reported is what the FRONT did.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto arms = std::array<std::pair<const char*, RearOracle>, 2>{
        std::pair<const char*, RearOracle>{"production", RearOracle{}},
        std::pair<const char*, RearOracle>{"rear gated on Fz <= 0",
                                           RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.0}}};

    std::printf("\n=== the front channel, with and without the rear oracle ===\n");
    std::printf("\n  %-24s  front util   rear util   front dumps   rear dumps   front share of |Fx|   stop [m]\n",
                "arm");

    for (const auto& arm : arms)
    {
        auto frontUse = std::vector<double>{};
        auto rearUse = std::vector<double>{};
        auto frontCycles = std::vector<double>{};
        auto rearCycles = std::vector<double>{};
        auto share = std::vector<double>{};
        auto stops = std::vector<double>{};

        for (auto index = std::size_t{0}; index < ensembleCount; index++)
        {
            const auto run =
                record(setup.value(), world.value(), assists, 1.0, arm.second, memberEntry(index, ensembleCount));
            const auto use = axleUse(run);

            auto frontImpulse = 0.0;
            auto totalImpulse = 0.0;
            for (const auto& sample : run.samples)
            {
                for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
                {
                    const auto value = std::abs(sample.wheels[wheel].forceLongitudinal);
                    totalImpulse += value;
                    frontImpulse += wheel < 2 ? value : 0.0;
                }
            }

            frontUse.push_back(use.frontUtilisation);
            rearUse.push_back(use.rearUtilisation);
            frontCycles.push_back(static_cast<double>(cyclesOf(run, 0)));
            rearCycles.push_back(static_cast<double>(cyclesOf(run, 2)));
            share.push_back(totalImpulse > 0.0 ? frontImpulse / totalImpulse : 0.0);
            stops.push_back(run.distance);
        }

        std::printf("  %-24s  %10.4f  %10.4f  %12.1f %12.1f  %19.4f  %9.3f\n", arm.first,
                    distributionOf(frontUse).median, distributionOf(rearUse).median, distributionOf(frontCycles).median,
                    distributionOf(rearCycles).median, distributionOf(share).median, distributionOf(stops).median);
    }

    std::printf("\n  Every figure is the ensemble median over %zu members. The front controller is production\n",
                ensembleCount);
    std::printf("  in both rows and was not redesigned, read or altered.\n");
}

TEST_CASE("the half-pedal four-wheels red, against the same oracle", "[.rear-hop]")
{
    // **Section 15.** The criterion's own fixture is `pedal 0.50` with the ELECTRONICS OFF, so there
    // is no rear channel in it to break — which is itself the answer to half the question. What is
    // measured here is three arms: the criterion's own fixture, the same pedal with the anti-lock
    // system on, and the same pedal with the anti-lock system on and the rear oracle holding.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto off = golfGtiMk7Assists(setup.value());
    auto on = off;
    on.antilock.enabled = true;

    struct Arm
    {
        const char* name;
        const AssistSetup* assists;
        RearOracle oracle;
    };

    const auto arms = std::vector<Arm>{
        {"criterion's own fixture: electronics OFF", &off, RearOracle{}},
        {"anti-lock ON, production rear", &on, RearOracle{}},
        {"anti-lock ON, rear gated on Fz <= 0", &on, RearOracle{.mode = RearMode::ContactGate, .loadFraction = 0.0}}};

    std::printf("\n=== the half-pedal four-wheels criterion, pedal 0.50, %zu members ===\n", ensembleCount);
    std::printf("\n  %-42s  four down   min rear Fz [N]   rear off ticks   stop [m]\n", "arm");

    for (const auto& arm : arms)
    {
        auto grounded = std::size_t{0};
        auto loads = std::vector<double>{};
        auto offTicks = std::vector<double>{};
        auto stops = std::vector<double>{};

        for (auto index = std::size_t{0}; index < ensembleCount; index++)
        {
            const auto run =
                record(setup.value(), world.value(), *arm.assists, 0.50, arm.oracle, memberEntry(index, ensembleCount));
            const auto use = axleUse(run);

            grounded += run.grounded ? 1 : 0;
            loads.push_back(use.minimumRearLoad);
            offTicks.push_back(static_cast<double>(run.rearAirborneTicks));
            stops.push_back(run.distance);
        }

        std::printf("  %-42s  %2zu / %2zu   %15.3f   %14.1f   %8.3f\n", arm.name, grounded, ensembleCount,
                    distributionOf(loads).median, distributionOf(offTicks).median, distributionOf(stops).median);
    }
}

TEST_CASE("does the controller add energy to the hop, or fail to take it out", "[.rear-hop]")
{
    // **Section 16.** Over one representative hop cycle, on the corner's own degree of freedom:
    //
    //   damper work   = integral of F_damper * d(wheelTravel)/dt
    //   brake-force work on the SUSPENSION = integral of Fx * (patchPerAngle.z / travelPerAngle)
    //                                        * d(wheelTravel)/dt
    //
    // The second term is the geometric load path's own: a longitudinal tyre force does vertical work
    // on this corner exactly in proportion to the longitudinal component of the patch's velocity per
    // unit of wishbone rotation, which is the same dot product `docs/suspension-load-path-brief.md`
    // is about. If it is positive over a cycle the brake force is FEEDING the vertical mode.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);

    struct Cycle
    {
        double from = 0.0;
        double to = 0.0;
        double damperWork = 0.0;
        double brakeForceWork = 0.0;
        double stopWork = 0.0;
        double impulse = 0.0;
        double brakeTorqueWork = 0.0;
        double travelSwing = 0.0;
        // The whole-car path, which the per-corner geometric term above cannot see: the pitch moment
        // the four longitudinal tyre forces make about the centre of mass, working on the pitch
        // rate. Reported per metre of centre-of-mass height, so no height has to be invented.
        double pitchWorkPerMetre = 0.0;
        double rearPitchWorkPerMetre = 0.0;
    };

    const auto measure = [&](const char* label, const std::size_t member)
    {
        const auto run =
            record(setup.value(), world.value(), assists, 1.0, RearOracle{}, memberEntry(member, ensembleCount));

        // Cycle boundaries: the rear axle load crossing UP through a tenth of its static value, which
        // is a physical event and needs no controller state.
        const auto staticAxle = run.samples.front().rearAxleLoad;
        const auto level = 0.10 * staticAxle;

        auto crossings = std::vector<std::size_t>{};
        for (auto index = std::size_t{1}; index < run.samples.size(); index++)
        {
            if (run.samples[index - 1].rearAxleLoad <= level && run.samples[index].rearAxleLoad > level)
            {
                crossings.push_back(index);
            }
        }

        std::printf("\n  --- %s member %zu: %zu upward crossings of %.1f N (a tenth of the static %.1f N) ---\n", label,
                    member, crossings.size(), level, staticAxle);

        if (crossings.size() < 2)
        {
            std::printf("    fewer than two cycles: no sustained hop to account for.\n");
            return;
        }

        auto cycles = std::vector<Cycle>{};

        for (auto index = std::size_t{1}; index < crossings.size(); index++)
        {
            auto cycle = Cycle{};
            cycle.from = run.samples[crossings[index - 1]].time;
            cycle.to = run.samples[crossings[index]].time;

            auto minimumTravel = 1e9;
            auto maximumTravel = -1e9;

            // The cycle's own mean longitudinal force, so the steady dive is out of the pitch term
            // and only its FLUCTUATION is accounted.
            auto meanForce = 0.0;
            auto meanRearForce = 0.0;
            auto counted = 0.0;
            for (auto at = crossings[index - 1] + 1; at <= crossings[index]; at++)
            {
                for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
                {
                    meanForce += run.samples[at].wheels[wheel].forceLongitudinal;
                    meanRearForce += wheel >= 2 ? run.samples[at].wheels[wheel].forceLongitudinal : 0.0;
                }
                counted += 1.0;
            }
            meanForce /= std::max(counted, 1.0);
            meanRearForce /= std::max(counted, 1.0);

            for (auto at = crossings[index - 1] + 1; at <= crossings[index]; at++)
            {
                const auto& sample = run.samples[at];
                const auto& previous = run.samples[at - 1];

                auto totalForce = 0.0;
                auto rearForce = 0.0;
                for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
                {
                    totalForce += sample.wheels[wheel].forceLongitudinal;
                    rearForce += wheel >= 2 ? sample.wheels[wheel].forceLongitudinal : 0.0;
                }

                cycle.pitchWorkPerMetre += (totalForce - meanForce) * sample.pitchRate * tick;
                cycle.rearPitchWorkPerMetre += (rearForce - meanRearForce) * sample.pitchRate * tick;

                for (auto wheel = std::size_t{2}; wheel < cornerCount; wheel++)
                {
                    const auto& now = sample.wheels[wheel];
                    const auto travelRate = (now.suspensionTravel - previous.wheels[wheel].suspensionTravel) / tick;
                    const auto ratio =
                        std::abs(now.travelPerAngle) > 1e-9 ? now.patchPerAngleZ / now.travelPerAngle : 0.0;

                    cycle.damperWork += now.damperForce * travelRate * tick;
                    cycle.brakeForceWork += now.forceLongitudinal * ratio * travelRate * tick;
                    cycle.stopWork += (now.bumpStopForce + now.droopStopForce) * travelRate * tick;
                    cycle.impulse += std::abs(now.forceLongitudinal) * tick;
                    cycle.brakeTorqueWork += now.brakeTorque * std::abs(now.wheelSpeed) * tick;

                    if (wheel == 2)
                    {
                        minimumTravel = std::min(minimumTravel, now.suspensionTravel);
                        maximumTravel = std::max(maximumTravel, now.suspensionTravel);
                    }
                }
            }

            cycle.travelSwing = maximumTravel - minimumTravel;
            cycles.push_back(cycle);
        }

        std::printf("\n      from      to   period [ms]   damper W [J]   Fx-on-suspension W [J]   stops W [J]"
                    "   |Fx| impulse [N.s]   brake W [J]   travel swing [mm]   pitch W/h [J/m]   rear pitch W/h\n");

        auto sumDamper = 0.0;
        auto sumBrakeForce = 0.0;
        auto sumPitch = 0.0;
        auto sumRearPitch = 0.0;
        auto counted = std::size_t{0};

        for (const auto& cycle : cycles)
        {
            if (cycle.from < 0.6)
            {
                continue;
            }

            counted++;
            sumDamper += cycle.damperWork;
            sumBrakeForce += cycle.brakeForceWork;
            sumPitch += cycle.pitchWorkPerMetre;
            sumRearPitch += cycle.rearPitchWorkPerMetre;

            std::printf("   %7.4f %7.4f   %11.1f   %12.2f   %22.2f   %11.2f   %18.1f   %11.1f   %17.2f   %15.2f"
                        "   %14.2f\n",
                        cycle.from, cycle.to, 1000.0 * (cycle.to - cycle.from), cycle.damperWork, cycle.brakeForceWork,
                        cycle.stopWork, cycle.impulse, cycle.brakeTorqueWork, 1000.0 * cycle.travelSwing,
                        cycle.pitchWorkPerMetre, cycle.rearPitchWorkPerMetre);
        }

        if (counted > 0)
        {
            std::printf("\n    over %zu sustained cycles: damper %.2f J, Fx-on-suspension %+.2f J,"
                        " pitch %+.2f J/m of CG height (rear share %+.2f J/m).\n",
                        counted, sumDamper, sumBrakeForce, sumPitch, sumRearPitch);
            std::printf("    %s;  %s\n",
                        sumBrakeForce > 0.0 ? "the per-corner geometric path FEEDS the mode"
                                            : "the per-corner geometric path does net negative work",
                        sumPitch > 0.0 ? "the whole-car pitch-moment path FEEDS the mode"
                                       : "the whole-car pitch-moment path does net negative work");
        }

        std::printf("    (sign convention: positive damper work is the damper being driven; positive"
                    " Fx-on-suspension work\n     is the longitudinal tyre force FEEDING the corner's"
                    " vertical degree of freedom.)\n");
    };

    std::printf("\n=== energy over the cycle, at %.5f m/s^2 of gravity ===\n", gravity);
    measure("RECOVERED", found.recovered);
    measure("STRANDED ", found.stranded);

    // **The sign control, and it is what turns the two numbers above from a claim into a
    // measurement** [check-the-control-variable]. The same two integrals over the pitch ring-down of
    // an ELECTRONICS-OFF constant-pedal stop, where the mode is known to decay: whatever sign a
    // decaying mode gives is the sign that means "damping", and the arms above are read against it.
    const auto off = golfGtiMk7Assists(setup.value());
    const auto control = record(setup.value(), world.value(), off, 0.38, RearOracle{}, memberEntry(14, ensembleCount));

    auto meanForce = 0.0;
    auto counted = 0.0;
    for (const auto& sample : control.samples)
    {
        if (sample.time < 0.05 || sample.time > 1.50)
        {
            continue;
        }
        for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
        {
            meanForce += sample.wheels[wheel].forceLongitudinal;
        }
        counted += 1.0;
    }
    meanForce /= std::max(counted, 1.0);

    auto pitchWork = 0.0;
    auto damperWork = 0.0;
    auto earlySwing = 0.0;
    auto lateSwing = 0.0;
    auto earlyLow = 1e9;
    auto earlyHigh = -1e9;
    auto lateLow = 1e9;
    auto lateHigh = -1e9;

    for (auto index = std::size_t{1}; index < control.samples.size(); index++)
    {
        const auto& sample = control.samples[index];
        if (sample.time < 0.05 || sample.time > 1.50)
        {
            continue;
        }

        auto totalForce = 0.0;
        for (auto wheel = std::size_t{0}; wheel < cornerCount; wheel++)
        {
            totalForce += sample.wheels[wheel].forceLongitudinal;
        }

        pitchWork += (totalForce - meanForce) * sample.pitchRate * tick;

        for (auto wheel = std::size_t{2}; wheel < cornerCount; wheel++)
        {
            const auto rate =
                (sample.wheels[wheel].suspensionTravel - control.samples[index - 1].wheels[wheel].suspensionTravel) /
                tick;
            damperWork += sample.wheels[wheel].damperForce * rate * tick;
        }

        if (sample.time < 0.60)
        {
            earlyLow = std::min(earlyLow, sample.wheels[2].suspensionTravel);
            earlyHigh = std::max(earlyHigh, sample.wheels[2].suspensionTravel);
        }
        else
        {
            lateLow = std::min(lateLow, sample.wheels[2].suspensionTravel);
            lateHigh = std::max(lateHigh, sample.wheels[2].suspensionTravel);
        }
    }

    earlySwing = earlyHigh - earlyLow;
    lateSwing = lateHigh - lateLow;

    std::printf("\n  --- the SIGN CONTROL: electronics off, pedal 0.38, the pitch ring-down 0.05-1.50 s ---\n");
    std::printf("    rear travel swing 0.05-0.60 s: %.2f mm;  0.60-1.50 s: %.2f mm   (%s)\n", 1000.0 * earlySwing,
                1000.0 * lateSwing, lateSwing < earlySwing ? "the mode DECAYS" : "the mode does not decay");
    std::printf("    damper work over the window: %+.2f J   pitch work: %+.2f J per metre of CG height\n", damperWork,
                pitchWork);
    std::printf("    stop %.3f m, rear off %zu ticks, four wheels down: %s\n", control.distance,
                control.rearAirborneTicks, control.grounded ? "yes" : "NO");
    std::printf("    Read the arms above against this row's SIGNS, not against zero.\n");
}

TEST_CASE("what the rear channel believes after the axle lands", "[.rear-hop]")
{
    // **The measurement the oracle results sent this file back for.** Every load-gated and
    // timing-gated intervention above did nothing, and the re-apply population turned out to be the
    // opposite of the hypothesis, so the question becomes: what is the rear channel doing during the
    // hundreds of milliseconds when its pressure is zero and its wheel is on the road?
    //
    // Three things are measured here, all of them off signals already sampled:
    //   - the reference speed estimate against the truth, split by whether the rear axle is on the
    //     road, because a wheel that has left the ground is no longer a road speed measurement;
    //   - how often the `pastBand` guard — the gate on every leg of the slip-aware recovery law — is
    //     true while the tyre's REAL slip is negligible;
    //   - the longest run of ticks in each stop with essentially no rear brake pressure while the
    //     rear axle is on the road and carrying real load, which is where the distance goes.
    //
    // **This is not an estimator investigation.** No estimator constant is read for a value to
    // change and nothing about `advanceReferenceSpeed` is proposed. What is being established is
    // which signal the sustaining loop runs on.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    const auto found = representativesOf(setup.value(), world.value(), assists);

    struct Bucket
    {
        std::vector<double> referenceError;
        std::vector<double> guardSlip;
        std::vector<double> trueSlip;
        std::size_t ticks = 0;
        std::size_t pastBand = 0;
        std::size_t pastBandRolling = 0;
        std::size_t zeroPressureLoaded = 0;
    };

    auto recoveredOn = Bucket{};
    auto strandedOn = Bucket{};
    auto strandedOff = Bucket{};

    // The stuck branch: `Recover -> Dump`, which only the slip-aware law can take.
    struct Stuck
    {
        std::vector<double> trueSlip;
        std::vector<double> guardSlip;
        std::vector<double> referenceError;
        std::vector<double> load;
        std::vector<double> pressure;
    };

    auto stuck = Stuck{};

    // The longest run of near-zero rear pressure while the axle is loaded, per member.
    auto darkRecovered = std::vector<double>{};
    auto darkStranded = std::vector<double>{};
    auto darkImpulseLost = std::vector<double>{};

    for (const auto& member : found.members)
    {
        const auto run = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, member.entry, true);
        const auto steps = perTick(run, 2);
        const auto staticAxle = run.samples.front().rearAxleLoad;

        auto longest = 0.0;
        auto current = 0.0;
        auto lostCapacity = 0.0;

        for (auto index = std::size_t{0}; index < run.samples.size(); index++)
        {
            const auto& sample = run.samples[index];
            const auto* step = steps[index];

            if (step == nullptr || !step->referenceValid || sample.speed < 1.0)
            {
                continue;
            }

            auto& bucket = !sample.rearContact ? strandedOff : (member.stranded() ? strandedOn : recoveredOn);

            if (sample.rearContact || member.stranded())
            {
                bucket.ticks++;
                bucket.referenceError.push_back((sample.referenceSpeed - sample.speed) / std::max(sample.speed, 1e-6));
                bucket.guardSlip.push_back(step->guardSlip);
                bucket.trueSlip.push_back(std::abs(sample.wheels[2].slipRatio));
                bucket.pastBand += step->pastBand ? 1 : 0;
                bucket.pastBandRolling += (step->pastBand && std::abs(sample.wheels[2].slipRatio) < 0.05) ? 1 : 0;
            }

            const auto dark =
                sample.wheels[2].pressure < 1.0 * bar && sample.rearContact && sample.rearAxleLoad > 0.5 * staticAxle;

            if (dark)
            {
                current += tick;
                longest = std::max(longest, current);
                lostCapacity += sample.wheels[2].capacity() * tick + sample.wheels[3].capacity() * tick;
            }
            else
            {
                current = 0.0;
            }

            if (step->before == ModulatorPhase::Recover && step->after == ModulatorPhase::Dump)
            {
                stuck.trueSlip.push_back(std::abs(sample.wheels[2].slipRatio));
                stuck.guardSlip.push_back(step->guardSlip);
                stuck.referenceError.push_back((sample.referenceSpeed - sample.speed) / std::max(sample.speed, 1e-6));
                stuck.load.push_back(sample.rearAxleLoad);
                stuck.pressure.push_back(step->pressureBefore / bar);
            }
        }

        (member.stranded() ? darkStranded : darkRecovered).push_back(longest);
        darkImpulseLost.push_back(lostCapacity);
    }

    const auto report = [](const char* label, const Bucket& bucket)
    {
        const auto error = distributionOf(bucket.referenceError);
        const auto guardSlip = distributionOf(bucket.guardSlip);
        const auto trueSlip = distributionOf(bucket.trueSlip);

        std::printf("\n  --- %s: %zu ticks ---\n", label, bucket.ticks);
        std::printf("    reference error (ref - true)/true   min %+8.4f  P10 %+8.4f  MEDIAN %+8.4f  P90 %+8.4f"
                    "  max %+8.4f\n",
                    error.minimum, error.p10, error.median, error.p90, error.maximum);
        std::printf("    the guard's slip (what the law reads) min %8.4f  P10 %8.4f  MEDIAN %8.4f  P90 %8.4f"
                    "  max %8.4f\n",
                    guardSlip.minimum, guardSlip.p10, guardSlip.median, guardSlip.p90, guardSlip.maximum);
        std::printf("    the tyre's REAL slip                 min %8.4f  P10 %8.4f  MEDIAN %8.4f  P90 %8.4f"
                    "  max %8.4f\n",
                    trueSlip.minimum, trueSlip.p10, trueSlip.median, trueSlip.p90, trueSlip.maximum);
        std::printf("    pastBand true: %zu ticks (%.1f%%);  of those, %zu (%.1f%% of all ticks) had a REAL slip"
                    " below 0.05\n",
                    bucket.pastBand,
                    100.0 * static_cast<double>(bucket.pastBand) /
                        static_cast<double>(std::max(bucket.ticks, std::size_t{1})),
                    bucket.pastBandRolling,
                    100.0 * static_cast<double>(bucket.pastBandRolling) /
                        static_cast<double>(std::max(bucket.ticks, std::size_t{1})));
    };

    std::printf("\n=== the reference speed, the guard it feeds, and the dark time it buys ===\n");
    report("RECOVERED members, rear axle ON the road", recoveredOn);
    report("STRANDED  members, rear axle ON the road", strandedOn);
    report("ANY member, rear axle OFF the road", strandedOff);

    std::printf("\n  --- every Recover -> Dump (the slip-aware law's STUCK branch), whole ensemble: %zu ---\n",
                stuck.trueSlip.size());

    const auto row = [](const char* name, const std::vector<double>& values, const char* unit)
    {
        const auto distribution = distributionOf(values);
        std::printf("    %-30s  min %9.4f   P10 %9.4f   MEDIAN %9.4f   P90 %9.4f   max %9.4f  %s\n", name,
                    distribution.minimum, distribution.p10, distribution.median, distribution.p90, distribution.maximum,
                    unit);
    };

    row("the tyre's REAL slip", stuck.trueSlip, "-");
    row("the guard's slip", stuck.guardSlip, "-");
    row("reference error", stuck.referenceError, "-");
    row("rear axle load", stuck.load, "N");
    row("rear pressure at the dump", stuck.pressure, "bar");

    const auto dark1 = distributionOf(darkRecovered);
    const auto dark2 = distributionOf(darkStranded);
    const auto lost = distributionOf(darkImpulseLost);

    std::printf("\n  --- the longest continuous run with rear pressure < 1 bar while the rear axle is DOWN and\n");
    std::printf("      carrying more than half its static load ---\n");
    std::printf("    RECOVERED members (%zu):  min %6.1f ms   MEDIAN %6.1f ms   max %6.1f ms\n", dark1.n,
                1000.0 * dark1.minimum, 1000.0 * dark1.median, 1000.0 * dark1.maximum);
    std::printf("    STRANDED  members (%zu):  min %6.1f ms   MEDIAN %6.1f ms   max %6.1f ms\n", dark2.n,
                1000.0 * dark2.minimum, 1000.0 * dark2.median, 1000.0 * dark2.maximum);
    std::printf("    rear grip capacity going past unused while the axle was down, whole ensemble:"
                " median %.1f N.s, max %.1f N.s\n",
                lost.median, lost.maximum);

    // And whether that dark time is where the distance actually goes. Per member, the TOTAL time the
    // rear caliper spent under a bar with the axle down and loaded, against the stop.
    std::printf("\n  --- per member: total dark rear time against the stop ---\n");
    std::printf("\n     idx   dark total [ms]   rear util   stop [m]   cluster\n");

    auto darkTotals = std::vector<double>{};
    auto stops = std::vector<double>{};

    for (const auto& member : found.members)
    {
        const auto run = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, member.entry);
        const auto staticAxle = run.samples.front().rearAxleLoad;

        auto total = 0.0;
        for (const auto& sample : run.samples)
        {
            total +=
                sample.wheels[2].pressure < 1.0 * bar && sample.rearContact && sample.rearAxleLoad > 0.5 * staticAxle
                    ? tick
                    : 0.0;
        }

        darkTotals.push_back(1000.0 * total);
        stops.push_back(member.distance);

        std::printf("     %3zu   %15.1f   %9.4f   %8.3f   %s\n", member.index, 1000.0 * total, member.rearUtilisation,
                    member.distance, member.stranded() ? "STRANDED" : "recovered");
    }

    auto meanDark = 0.0;
    auto meanStop = 0.0;
    for (auto index = std::size_t{0}; index < darkTotals.size(); index++)
    {
        meanDark += darkTotals[index];
        meanStop += stops[index];
    }
    meanDark /= static_cast<double>(darkTotals.size());
    meanStop /= static_cast<double>(stops.size());

    auto covariance = 0.0;
    auto varianceDark = 0.0;
    auto varianceStop = 0.0;
    for (auto index = std::size_t{0}; index < darkTotals.size(); index++)
    {
        covariance += (darkTotals[index] - meanDark) * (stops[index] - meanStop);
        varianceDark += (darkTotals[index] - meanDark) * (darkTotals[index] - meanDark);
        varianceStop += (stops[index] - meanStop) * (stops[index] - meanStop);
    }

    // A descriptive statistic and nothing else.
    std::printf("\n     mean dark %.1f ms, mean stop %.3f m, correlation r = %.4f over %zu members\n", meanDark,
                meanStop, covariance / std::max(std::sqrt(varianceDark * varianceStop), 1e-9), darkTotals.size());
}

TEST_CASE("where the reference speed error comes from, and what it costs", "[.rear-hop]")
{
    // **The last link in the chain, measured rather than argued.** `advanceReferenceSpeed` takes the
    // FASTEST sensed wheel under braking and will not let the estimate fall below it, nor fall faster
    // than `fallLimit` (1.3 g). A rear wheel in the air with its brake dumped keeps the speed it left
    // the road with while the car goes on slowing, so it becomes the fastest wheel by construction.
    // This case measures whether that is what actually happens, how big the error is when the axle
    // lands, and how long it takes to wash out.
    //
    // **Nothing here proposes an estimator change.** `docs/braking-chain-brief.md` records that two
    // estimator corrections were built, made the estimate honest and made the car worse, because the
    // controller's thresholds are calibrated against these errors. This is a measurement of a
    // TRANSIENT corruption the hop creates, which is a different object from the systematic bias that
    // was already ruled out by sign.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto assists = golfGtiMk7Assists(setup.value());
    assists.antilock.enabled = true;
    // **Pinned to the pre-2026-09-07 controller** (`docs/abs-architecture-design.md`). This probe is
    // a record of a closed investigation into that controller, so it keeps measuring it rather than
    // silently becoming a measurement of its replacement.
    assists.antilock.recoveryAuthority = RecoveryAuthority::Unconditional;

    std::printf("\n=== the reference speed's inputs, and the error the hop puts into it ===\n");
    std::printf("  fall limit %.3f m/s^2, rise limit %.3f m/s^2, rate smoothing %.3f s\n", assists.reference.fallLimit,
                assists.reference.riseLimit, assists.reference.rateSmoothing);
    std::printf("  under braking the estimate may not fall below the fastest sensed wheel, nor faster than\n");
    std::printf("  the fall limit. A car stopping at ~0.95 g therefore sheds an error at ~3 m/s^2 at best.\n");

    const auto found = representativesOf(setup.value(), world.value(), assists);

    auto rearSetsItOn = std::size_t{0};
    auto rearSetsItOff = std::size_t{0};
    auto ticksOn = std::size_t{0};
    auto ticksOff = std::size_t{0};

    struct Landing
    {
        double airborne = 0.0;
        double errorAtLanding = 0.0;
        double peakError = 0.0;
        double washout = 0.0;
        double darkAfter = 0.0;
        bool stranded = false;
    };

    auto landings = std::vector<Landing>{};

    for (const auto& member : found.members)
    {
        const auto run = record(setup.value(), world.value(), assists, 1.0, RearOracle{}, member.entry);

        auto liftedAt = std::size_t{0};
        auto lifted = false;

        for (auto index = std::size_t{1}; index < run.samples.size(); index++)
        {
            const auto& sample = run.samples[index];

            if (sample.speed < 1.0 || !sample.referenceValid)
            {
                continue;
            }

            if (sample.rearContact)
            {
                ticksOn++;
                rearSetsItOn += sample.fastestWheel >= 2 ? 1 : 0;
            }
            else
            {
                ticksOff++;
                rearSetsItOff += sample.fastestWheel >= 2 ? 1 : 0;
            }

            if (run.samples[index - 1].rearContact && !sample.rearContact)
            {
                liftedAt = index;
                lifted = true;
            }

            if (lifted && !run.samples[index - 1].rearContact && sample.rearContact)
            {
                auto landing = Landing{};
                landing.stranded = member.stranded();
                landing.airborne = sample.time - run.samples[liftedAt].time;
                landing.errorAtLanding = sample.referenceSpeed - sample.speed;

                for (auto forward = index; forward < run.samples.size(); forward++)
                {
                    const auto& later = run.samples[forward];
                    if (later.speed < 1.0)
                    {
                        break;
                    }

                    landing.peakError = std::max(landing.peakError, later.referenceSpeed - later.speed);
                    landing.darkAfter += later.wheels[2].pressure < 1.0 * bar && later.rearContact ? tick : 0.0;

                    if (later.referenceSpeed - later.speed < 0.05 * later.speed)
                    {
                        landing.washout = later.time - sample.time;
                        break;
                    }

                    if (!later.rearContact)
                    {
                        landing.washout = later.time - sample.time;
                        landing.darkAfter = 0.0;
                        break;
                    }
                }

                landings.push_back(landing);
                lifted = false;
            }
        }
    }

    std::printf("\n  the estimator's candidate is a REAR wheel on:\n");
    std::printf("    %6.2f%% of ticks with the rear axle ON the road   (%zu of %zu)\n",
                100.0 * static_cast<double>(rearSetsItOn) / static_cast<double>(std::max(ticksOn, std::size_t{1})),
                rearSetsItOn, ticksOn);
    std::printf("    %6.2f%% of ticks with the rear axle OFF the road  (%zu of %zu)\n",
                100.0 * static_cast<double>(rearSetsItOff) / static_cast<double>(std::max(ticksOff, std::size_t{1})),
                rearSetsItOff, ticksOff);

    std::printf("\n  --- every rear landing in the ensemble: %zu ---\n", landings.size());
    std::printf("\n     airborne [ms]   ref error at landing [m/s]   peak error after [m/s]   washout to <5%% [ms]"
                "   dark rear [ms]   member\n");

    for (const auto& landing : landings)
    {
        std::printf("     %13.1f   %25.3f   %21.3f   %19.1f   %14.1f   %s\n", 1000.0 * landing.airborne,
                    landing.errorAtLanding, landing.peakError, 1000.0 * landing.washout, 1000.0 * landing.darkAfter,
                    landing.stranded ? "STRANDED" : "recovered");
    }

    auto airborne = std::vector<double>{};
    auto peak = std::vector<double>{};
    auto dark = std::vector<double>{};

    for (const auto& landing : landings)
    {
        airborne.push_back(1000.0 * landing.airborne);
        peak.push_back(landing.peakError);
        dark.push_back(1000.0 * landing.darkAfter);
    }

    const auto airborneDistribution = distributionOf(airborne);
    const auto peakDistribution = distributionOf(peak);
    const auto darkDistribution = distributionOf(dark);

    std::printf("\n     airborne [ms]        median %8.1f   max %8.1f\n", airborneDistribution.median,
                airborneDistribution.maximum);
    std::printf("     peak ref error [m/s] median %8.3f   max %8.3f\n", peakDistribution.median,
                peakDistribution.maximum);
    std::printf("     dark rear time [ms]  median %8.1f   max %8.1f\n", darkDistribution.median,
                darkDistribution.maximum);
}

TEST_CASE("which single term the loop needs: the guards, or the rear re-apply rate", "[.rear-hop]")
{
    // **The last two counterfactuals, and neither is a probe hack**: both are existing fields of
    // `AntilockSetup`, set on a LOCAL copy for one ensemble each. Nothing is written to any
    // production default and no file is edited.
    //
    //   G  `slipAwareRecovery = false` — the switch the law was built with, off. It removes
    //      `pastBand` from every leg at once, on all three channels, and is the A/B
    //      `AntilockSetup::slipAwareRecovery` exists for.
    //   V  `modulator.rearReapplyGradient` multiplied — the ONE actuator rate that is already a rear
    //      field, so the front valve and both dump rates stay exactly as shipped. It separates "the
    //      rear re-apply is too slow" from "the rear dump is too fast" and from anything the front
    //      channel does.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0));
    REQUIRE(world.has_value());

    auto shipped = golfGtiMk7Assists(setup.value());
    shipped.antilock.enabled = true;

    auto noGuards = shipped;
    noGuards.antilock.recoveryAuthority = RecoveryAuthority::Disabled;

    struct Arm
    {
        const char* name;
        AssistSetup assists;
    };

    auto arms =
        std::vector<Arm>{{"production (control)", shipped}, {"G  slipAwareRecovery OFF (all channels)", noGuards}};

    for (const auto factor : std::array<double, 4>{2.0, 4.0, 10.0, 100.0})
    {
        auto faster = shipped;
        faster.antilock.modulator.rearReapplyGradient = factor * shipped.antilock.modulator.reapplyGradient;
        arms.push_back(Arm{"V  rear re-apply gradient x", faster});
    }

    // And the two together, because the oracle arms already showed the loop is an interaction.
    auto both = noGuards;
    both.antilock.modulator.rearReapplyGradient = 10.0 * shipped.antilock.modulator.reapplyGradient;
    arms.push_back(Arm{"G+V  guards off AND rear x10", both});

    std::printf("\n=== the two production switches, each on its own ensemble ===\n");
    printArmHeader();

    const auto factors = std::array<double, 4>{2.0, 4.0, 10.0, 100.0};
    auto label = std::string{};

    for (auto index = std::size_t{0}; index < arms.size(); index++)
    {
        const auto members = ensembleOf(setup.value(), world.value(), arms[index].assists, 1.0, RearOracle{});

        label = arms[index].name;
        if (index >= 2 && index < 2 + factors.size())
        {
            label += std::to_string(static_cast<int>(factors[index - 2]));
            label += "  (";
            label +=
                std::to_string(static_cast<int>(factors[index - 2] * shipped.antilock.modulator.reapplyGradient / bar));
            label += " bar/s)";
        }

        printArm(label.c_str(), summarise(members));
    }

    std::printf("\n  The rear re-apply gradient is the only actuator rate that is already a REAR field.\n");
    std::printf("  Every arm above leaves the front channel, both dump rates and the whole control law\n");
    std::printf("  exactly as shipped except where its own name says otherwise.\n");
}
