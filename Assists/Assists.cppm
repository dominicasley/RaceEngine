module;

#include <array>
#include <cstdint>
#include <type_traits>

export module raceengine.assists;

export import :WheelSensors;
export import :BrakeControl;
export import :TractionControl;

namespace raceengine
{

// The electronics between the driver and the actuators: anti-lock braking, traction control and the
// cornering brake, sharing one set of wheel speed sensors and one hydraulic unit.
//
// **Why this is a module and not a partition of `raceengine.physics` is stated at the top of
// `Api/WheelSensors.cppm`** and is the most important thing about the whole layer: nothing in here
// can reach vehicle state, because reaching it would mean importing the module that already imports
// this one, and a module cycle is a build failure rather than a review comment.
//
// The unit is a pure function of (setup, state, sensors, demand, dt) in the same sense the vehicle
// tick is: no clock is read, no global is touched, nothing is random, and its state is bytes that
// can be saved and restored with the car's.

// What the car's own sensors report this tick. Every field is something a real ECU has on the bus.
export struct AssistSensors
{
    // Wheel rotation in rad/s. **The only crossing point in the whole design**: it goes to the tone
    // rings, and what comes out the other side of them is all any controller here ever sees.
    std::array<double, wheelCount> wheelSpeeds{};

    // The yaw rate sensor and the accelerometer, in the car's own frame. Yaw rate is read by nothing
    // yet and is here because a stability controller needs it, needs it from a sensor rather than
    // from the chassis, and adding it later would move every call site.
    double yawRate = 0.0;
    double lateralAcceleration = 0.0;

    // The steering angle sensor, radians at the wheel. The cornering brake's kinematics would be
    // slightly better for reading it; nothing does yet.
    double steeringWheelAngle = 0.0;
};

// What the driver is asking for, before any of it reaches an actuator.
export struct AssistDemand
{
    // Pedal travel, 0 to 1. Read for the brake-light switch and for nothing else — a pedal is not a
    // pressure, and everything the hydraulics do to it belongs upstream of this layer.
    double brake = 0.0;
    double throttle = 0.0;
};

export struct AssistSetup
{
    ToneRing toneRing;
    ReferenceSpeedSetup reference;
    AntilockSetup antilock;
    TractionSetup traction;
    CorneringBrakeSetup cornering;

    // What each corner's brake makes per pascal of line pressure, N.m/Pa, and what the system's
    // pressure is at a fully applied pedal, Pa.
    //
    // **Both replace one array of peak torques** (2026-08-23, `docs/brake-model-brief.md` stage 2),
    // and the reason is that the peak was the only scale in the whole layer: `BrakeModulator`'s rates
    // were fractions of it per second, so re-deriving this car's brakes from its calipers — which
    // moved the peak from 5600 N.m to 10688 — silently doubled every pressure gradient the modulator
    // works at. A controller calibrated in fractions of an actuator's range is a controller that has
    // to be re-tuned every time the actuator changes, which is what it is not allowed to need.
    //
    // Still ECU calibration rather than vehicle state: a unit that cannot say what its own actuator
    // is worth cannot ask it for a torque. What it must not have is anything about the *car* — load,
    // grip, mass — and it does not.
    std::array<double, wheelCount> brakeTorquePerPressure{};

    // What each caliper is at with the pedal fully applied, Pa. **Per wheel and not one number**,
    // because a car with a proportioning valve on its rear circuit does not put the same pressure on
    // all four — and the two controllers below that command a *fraction* of full braking need to know
    // what full braking is at the wheel they are commanding it at.
    std::array<double, wheelCount> maximumWheelPressure{};

    // The controller clock, Hz. One number rather than three, because the three systems share an
    // ECU on a real car and the per-system fields exist for a future one that does not.
    double controlRate = 1000.0;

    // Brake travel at which the brake light switch trips, which is the signal the reference speed
    // estimator changes which wheels it believes on. A twentieth of the pedal, the same figure and
    // the same reasoning as `AutoClutch::creepBrakeCut`.
    double brakeSwitch = 0.05;
};

export struct AssistState
{
    WheelSensorStates sensors{};
    ReferenceSpeedState reference{};
    AntilockState antilock{};
    TractionState traction{};
    CorneringBrakeState cornering{};

    // What the hydraulic unit is holding at each wheel, **pascals**. Carried across ticks because the
    // controller runs on its own clock: a tick shorter than one controller period runs no controller
    // step at all and the actuator holds where it was, which is what an actuator does.
    std::array<double, wheelCount> pressure{};

    // Controller time not yet spent, seconds.
    double clockRemainder = 0.0;
};

static_assert(std::is_trivially_copyable_v<AssistState>, "the harness saves and restores this by copying its bytes");
static_assert(std::is_standard_layout_v<AssistState>, "and rollback will later");

// What the layer did, for telemetry and for tests. Every intervention is reported by **source**
// rather than as one total, because "the brakes came on" is not a finding and "traction control
// applied 210 N.m to the front left while the engine channel was still winding in" is.
export struct AssistChannels
{
    // The ECU's own view of the world, which is not the world.
    double referenceSpeed = 0.0;
    bool referenceValid = false;
    double referenceCoasting = 0.0;
    // How fast the ECU believes the *car* is changing speed, m/s^2. Every anti-lock threshold is
    // measured against this, so it is the channel to read first when the system misbehaves.
    double referenceAcceleration = 0.0;

    std::array<double, wheelCount> sensedWheelSpeed{};
    std::array<double, wheelCount> sensorAge{};
    std::array<double, wheelCount> estimatedSlip{};
    // Peripheral acceleration as the controller measured it — between tooth crossings, not between
    // control steps. The channel that says whether the ECU could see what the wheel was doing.
    std::array<double, wheelCount> sensedWheelAcceleration{};
    std::array<ModulatorPhase, wheelCount> antilockPhase{};

    // What the actuator ended the tick at, **pascals**, and what each source asked for in N.m.
    std::array<double, wheelCount> pressure{};
    std::array<double, wheelCount> driverBrakeTorque{};
    std::array<double, wheelCount> antilockBrakeTorque{};
    std::array<double, wheelCount> tractionBrakeTorque{};
    std::array<double, wheelCount> corneringBrakeTorque{};

    // 0 with the driver's foot untouched, 1 with the throttle shut.
    double engineTorqueReduction = 0.0;

    std::array<bool, wheelCount> antilockActive{};
    std::array<std::uint32_t, wheelCount> antilockCycles{};
    bool tractionBrakeActive = false;
    bool tractionEngineActive = false;
    bool corneringActive = false;
};

export struct AssistOutput
{
    BrakeCommand brakes;
    // What the driver's throttle becomes on its way to the engine, 0 to 1.
    double throttleScale = 1.0;
    AssistChannels channels;
};

// One tick of the whole layer. `deltaTime` is the *physics* tick: the sensors are sampled across it
// and the controller then runs however many of its own periods fit inside, carrying the remainder.
//
// `wheelPressure` is what the driver's pedal has already put at each caliper, pascals — the master
// cylinder's own characteristic on the front circuit and whatever the proportioning valve has done to
// it on the rear.
//
// **A parameter and not a field on `AssistDemand`**, deliberately. A pedal is what the driver asks
// for and a caliper pressure is what the plumbing made of it, so they are not the same kind of thing;
// and a defaulted field would have let all twenty-three call sites keep compiling while quietly
// commanding zero pressure, which is a car with no brakes that still builds. `brakeCircuitPressures`
// in `raceengine.physics` is the one place that map is written down — this layer cannot see the car's
// hydraulics any more than it can see its mass, and a real unit is in the same position: it has a
// pressure sensor and a knowledge of its own plumbing, not a model of the pedal box.
export [[nodiscard]] AssistOutput updateAssists(const AssistSetup& setup, AssistState& state,
                                                const AssistSensors& sensors, const AssistDemand& demand,
                                                const std::array<double, wheelCount>& wheelPressure,
                                                const double deltaTime);

// Whether anything is switched on. The vehicle model brakes exactly as it always did when nothing
// is, which is what keeps an unassisted car bit-identical rather than merely close.
export [[nodiscard]] bool assistsEngaged(const AssistSetup& setup);

} // namespace raceengine
