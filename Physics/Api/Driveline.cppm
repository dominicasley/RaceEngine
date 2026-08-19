module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:Driveline;

import :Clutch;
import :Coupling;
import :Telemetry;
import :Vehicle;

namespace raceengine
{

// The driveline as a chain of rotating inertias coupled by torque-transfer elements, which is the
// whole of why it is written this way rather than as a ratio calculation.
//
// Every simplification below is only safe because of that structure. Written naively — engine torque
// split by a fixed ratio straight to the wheels — each of the deferred features is a rewrite:
// clutch slip needs an engine speed that is not derived from wheel speed, an LSD needs a
// differential that is asked a question rather than performing a division, and driveline compliance
// needs two inertias with something between them. Written as a chain they are all drop-ins, and the
// cost of writing it this way now is one extra state variable and one indirection.

export enum class DrivenAxle : std::uint32_t { Front, Rear, All };

// Front then rear, which is how many differentials a car without a centre one has. A centre
// differential is the same interface a third time and is somebody else's milestone.
export inline constexpr std::size_t axleCount = 2;

// The idle controller, and it is an air bypass rather than a governor on the fuelling because that
// is the device. A real engine idles on a path *around* the closed throttle plate and a closed-loop
// ECU runs a PI on that valve, so the brief's two options are the same object here. Stating it as the
// bypass is also what makes its combination with the driver's pedal obvious: two parallel air paths
// mean the engine sees whichever flows more and never their sum.
export struct IdleGovernor
{
    // Sized on the engine's own torque at idle rather than guessed. About 145 N.m at full throttle
    // at 89 rad/s against 0.15 kg.m^2 gives a natural frequency of sqrt(145 * ki / J) and a damping
    // ratio of 145 * kp / (2 * J * wn), so these are 2 Hz and very nearly critical — fast enough that
    // a clutch bite recovers in under half a second, slow enough that it is not fighting the driver.
    double proportional = 0.022;
    double integral = 0.15;
    double maximumBypass = 1.0;
};

export struct EngineModel
{
    // Torque against engine speed in rad/s, at full throttle. A curve rather than a peak and a
    // shape, because a real engine is measured rather than described.
    Curve torque;

    // Placeholder: a small turbocharged four.
    double inertia = 0.15;
    double idleSpeed = 89.0;     // ~850 rpm
    double limiterSpeed = 712.0; // ~6800 rpm
    // Below this it is not turning slowly, it has stopped: no engine keeps itself alight at a third
    // of its idle speed. ~380 rpm, under cranking speed.
    double stallSpeed = 40.0;

    IdleGovernor governor;

    // Engine braking: what it absorbs at the limiter with the throttle shut, falling linearly to
    // nothing at rest. Deliberately separate from the torque curve, which is a full-throttle
    // measurement and has no business carrying the closed-throttle behaviour as a negative number.
    double coastTorque = 75.0;
};

// Torque at the flywheel. Positive drives, negative brakes.
export [[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle)
{
    const auto demand = std::clamp(throttle, 0.0, 1.0);

    // The limiter cuts fuel rather than shaping the curve, which is why it is a cliff here and not a
    // taper: that abruptness is what a driver feels. And cutting fuel does not merely stop the
    // engine driving — it makes it *brake*, because a cylinder still pumping with nothing burning in
    // it is a compressor. Scaling the braking by a shut throttle alone would have an engine on its
    // limiter coasting freely, which is not what one does.
    const auto onLimiter = speed >= engine.limiterSpeed;

    const auto driving = onLimiter ? 0.0 : demand * engine.torque.at(speed);
    const auto pumping = onLimiter ? 1.0 : 1.0 - demand;
    const auto braking = pumping * engine.coastTorque * std::clamp(speed / engine.limiterSpeed, 0.0, 1.0);

    return driving - braking;
}

// How much bypass the governor is asking for, and the integral it is asking through. Exported and
// taking its integral by reference because what holds an idle is worth pinning on its own, without a
// driveline or a car around it.
export [[nodiscard]] double idleBypass(const EngineModel& engine, const double speed, double& integral,
                                       const double deltaTime)
{
    const auto error = engine.idleSpeed - speed;

    // A valve can only add air, so the integral is clamped to the range the output can ever be asked
    // for and never runs negative. An integrator left running while the output is saturated takes as
    // long to come back as it took to wind up, which reads as an engine that hangs after every
    // clutch release rather than as a controller fault.
    integral = std::clamp(integral + engine.governor.integral * error * deltaTime, 0.0, engine.governor.maximumBypass);

    return std::clamp(engine.governor.proportional * error + integral, 0.0, engine.governor.maximumBypass);
}

export struct Gearbox
{
    // Ratios by gear, index 0 being first. Reverse and neutral are handled by `gear` below rather
    // than by living in here.
    std::vector<double> ratios;
    double finalDrive = 4.37;
    double reverseRatio = 3.6;

    // Total reduction from the flywheel to the differential input. Zero in neutral, which is what
    // disconnects the chain — not a special case anywhere else, just a ratio of nothing.
    [[nodiscard]] double reduction(const std::int32_t gear) const
    {
        if (gear == 0 || ratios.empty())
        {
            return 0.0;
        }

        if (gear < 0)
        {
            return -reverseRatio * finalDrive;
        }

        const auto index = std::min(static_cast<std::size_t>(gear - 1), ratios.size() - 1);

        return ratios[index] * finalDrive;
    }
};

export struct DifferentialTorques
{
    double left = 0.0;
    double right = 0.0;
};

// What the pack did last tick. The lock/slip machine in `:Coupling` is what will decide this
// instead of the clamp in `split`, and these two are precisely the numbers it judges — how much the
// pack was asked to carry against how much it could — so they are recorded where they are known
// rather than recovered afterwards from torques that have already been split.
export struct DifferentialState
{
    double transfer = 0.0;
    double capacity = 0.0;
};

// The differential, asked a question rather than performing a division.
//
// One struct and one code path covers open, spool and clutch-pack, because the difference between
// them is entirely in these numbers: an open diff locks with nothing, a spool locks with everything,
// and an LSD locks with a preload plus a ramp that differs on and off power. Adding the LSD is
// therefore a change to data and not to code, which is the whole point of the brief's insistence
// that this be an interface.
export struct Differential
{
    // Torque it will transfer across itself regardless of what is going through it. What holds a
    // car straight under power with one wheel on ice.
    double preload = 0.0;
    // Fraction of the input torque that becomes locking torque, on power and off it. Real ramp
    // angles give different numbers for the two, and that asymmetry is most of an LSD's character.
    double powerRamp = 0.0;
    double coastRamp = 0.0;

    // How hard it resists a speed difference before it reaches its locking limit. Large: this is a
    // friction element, and the limit is what shapes it, not the stiffness.
    double lockingStiffness = 400.0;

    // The state is the axle's, not the differential's: a `Differential` is setup and is handed
    // around by const reference, and all-wheel drive asks the *same* one twice. Two axles sharing
    // one state object would have the front's lock stamping on the rear's, which is why the state
    // arrives here rather than living in the struct.
    [[nodiscard]] DifferentialTorques split(DifferentialState& state, const double leftSpeed, const double rightSpeed,
                                            const double input, const double deltaTime) const
    {
        // The tick the lock/slip machine will want when it replaces the clamp below. Taken now
        // because adding it later moves every call site, and consumed by nothing yet.
        static_cast<void>(deltaTime);

        const auto ramp = input >= 0.0 ? powerRamp : coastRamp;
        const auto capacity = preload + ramp * std::abs(input);

        // Opposes the difference, up to what the pack can hold. Beyond that the diff is simply open
        // and the faster wheel keeps the extra.
        const auto transfer = std::clamp(lockingStiffness * (leftSpeed - rightSpeed), -capacity, capacity);

        state.transfer = transfer;
        state.capacity = capacity;

        return DifferentialTorques{.left = 0.5 * input - transfer, .right = 0.5 * input + transfer};
    }
};

export [[nodiscard]] Differential openDifferential()
{
    return Differential{};
}

export [[nodiscard]] Differential spool()
{
    // Locked solid: an enormous capacity, so the transfer term is never the binding constraint and
    // the two wheels are held to the same speed.
    return Differential{.preload = 1e6, .powerRamp = 0.0, .coastRamp = 0.0, .lockingStiffness = 1e5};
}

export [[nodiscard]] Differential clutchPackLsd(const double preload, const double powerRamp, const double coastRamp)
{
    return Differential{.preload = preload, .powerRamp = powerRamp, .coastRamp = coastRamp};
}

export struct DrivelineSetup
{
    EngineModel engine;
    Gearbox gearbox;
    // The slot between the engine's inertia and the gearbox input, and this file names neither of the
    // two things that can be in it after this line.
    DriveCoupling coupling;
    AutoClutch autoClutch;
    Differential differential = openDifferential();
    DrivenAxle driven = DrivenAxle::Front;
};

// Whether the engine is alight. An enum rather than a bool so that a cranking state has somewhere to
// go, and so that `DrivelineState` stays trivially copyable either way.
export enum class EngineState : std::uint32_t { Stalled, Running };

// The driveline's own state, and it is here rather than in `VehicleState` for the reason every
// other split in this module is made: what integrates a quantity owns it. Engine speed sat in the
// vehicle's state where nothing in the vehicle model read it or wrote it, which made `:Vehicle` the
// keeper of a number belonging to a partition it does not even import.
//
// Trivially copyable and standard layout, and it stays that way — save and restore is a memcpy and
// rollback will later lean on that. Scalars, enums and fixed arrays only: no `Curve`, no vector, no
// string. The shift phase and timer, driveline wind-up, the gearbox output shaft and the converter's
// turbine still belong here and are still not here.
export struct DrivelineState
{
    // Independent state from the start even though a locked clutch makes it derivable, because
    // making it independent later is a restructure and making it independent now is free.
    double engineSpeed = 0.0;

    // A default-constructed driveline is a car with the key out: nothing has started this engine, so
    // it is not turning. `startEngine` is what changes that, and it is deliberately not an input.
    EngineState engine = EngineState::Stalled;

    // The coupling's own lock/slip machine, whichever kind is fitted.
    CouplingState clutch{};
    // The pedal actually being held, which is the driver's when the driver is on it and the
    // automation's when nobody is. Kept rather than recomputed so the handover between the two is
    // continuous — see `advanceClutchPedal`.
    double clutchPedal = 0.0;

    double idleIntegral = 0.0;

    // What the clutch has turned into heat since the run began, in joules. A running total rather
    // than a per-tick figure: the thermal model that will read it integrates, and a channel that had
    // to be integrated by whoever plots it is a channel that gets integrated differently twice.
    double slipEnergy = 0.0;

    // One per axle, and separate rather than shared, for the reason given at `split`.
    std::array<DifferentialState, axleCount> differentials{};
};

static_assert(std::is_trivially_copyable_v<DrivelineState>, "the harness saves and restores this by copying its bytes");
static_assert(std::is_standard_layout_v<DrivelineState>, "and rollback will later");

export struct DrivelineTorques
{
    // Per corner, in the same order as everything else.
    std::array<double, cornerCount> wheel{};
    double clutch = 0.0;
    double engine = 0.0;
    // How far out of step the two sides of the clutch are, in rad/s. It does not reach zero when the
    // coupling locks: only the engine's half of the constraint is integrated here, the driveline's
    // half being the vehicle tick's, so a locked clutch converges over a handful of ticks rather
    // than within one.
    double clutchSlip = 0.0;
    bool clutchLocked = false;
    double slipEnergy = 0.0;
};

// How a stalled engine comes back, and it is a function the game calls rather than a field on
// `VehicleInput`. There is no starter here and an ignition model is a milestone of its own; more to
// the point a restart is a *command*, and `VehicleInput` is the packet a rollback netcode transmits
// and replays every tick, where a level-triggered starter bit would fire again on every replayed
// tick of the restart. When a cranking model does arrive this is what ends the crank, and nothing
// above it moves.
export void startEngine(const DrivelineSetup& setup, DrivelineState& state)
{
    state.engine = EngineState::Running;
    state.engineSpeed = std::max(state.engineSpeed, setup.engine.idleSpeed);
    state.idleIntegral = 0.0;
}

namespace
{

// What replaced the three `std::max(0.0, ...)` floors. They were the only thing keeping engine speed
// off the negative axis, and removing them without this leaves nothing catching it — an engine
// dragged below the speed at which it can keep itself alight does not turn slowly backwards, it
// stops. So the floor is still there and it is now a consequence of the model rather than a clamp
// bolted under it.
void settleEngineSpeed(const EngineModel& engine, DrivelineState& state)
{
    if (state.engine == EngineState::Stalled || state.engineSpeed < engine.stallSpeed)
    {
        state.engine = EngineState::Stalled;
        state.engineSpeed = 0.0;
        state.idleIntegral = 0.0;
    }
}

} // namespace

// One tick of the chain. `state` in, torques out, and the caller integrates the wheels.
//
// Fallible, and the one thing that can fail is the coupling slot: a torque converter is a kind
// `DriveCoupling` can be set to and is not a model this engine has yet, so it says so rather than
// answering. Nothing else in here knows which coupling is fitted.
//
// The driver's packet arrives whole rather than as three loose scalars, because throttle, clutch and
// gear are all its fields and passing them separately is three chances to pass them in the wrong
// order. `roadTorques` is `roadTorques(previousStep)` and is what the road did to the wheels last
// tick — passed rather than restated, exactly as `wheelInertias` is, and lagged by one tick because
// the tire's answer for this one does not exist until the vehicle tick has run.
export [[nodiscard]] std::expected<DrivelineTorques, std::string>
stepDriveline(const DrivelineSetup& setup, DrivelineState& state, const std::array<double, cornerCount>& wheelSpeeds,
              const std::array<double, cornerCount>& wheelInertias, const std::array<double, cornerCount>& roadTorques,
              const VehicleInput& input, const double deltaTime)
{
    auto result = DrivelineTorques{};

    const auto engineInertia = std::max(setup.engine.inertia, 1e-9);
    const auto running = state.engine == EngineState::Running;

    // The bypass and the driver's pedal are parallel air paths, so the engine gets the larger of the
    // two and never their sum: a governor that added to a wide-open throttle would be making torque
    // that no valve was opened for.
    const auto bypass = running ? idleBypass(setup.engine, state.engineSpeed, state.idleIntegral, deltaTime) : 0.0;
    const auto demand = std::max(std::clamp(input.throttle, 0.0, 1.0), bypass);

    const auto flywheel = running ? engineTorque(setup.engine, state.engineSpeed, demand) : 0.0;
    const auto reduction = setup.gearbox.reduction(input.gear);

    result.engine = flywheel;

    const auto driven = setup.driven;
    const auto isDriven = [driven](const std::size_t index)
    {
        if (driven == DrivenAxle::All)
        {
            return true;
        }

        return driven == DrivenAxle::Front ? index < 2 : index >= 2;
    };

    // The driveline's speed and inertia, both referred to the clutch. Referring rather than
    // simulating each shaft is the deferred-compliance simplification, and it is exact while the
    // shafts are rigid.
    auto axleSpeed = 0.0;
    auto axleInertia = 0.0;
    auto axleRoadTorque = 0.0;
    auto drivenCount = 0;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        if (!isDriven(index))
        {
            continue;
        }

        axleSpeed += wheelSpeeds[index];
        axleInertia += wheelInertias[index];
        axleRoadTorque += roadTorques[index];
        drivenCount++;
    }

    const auto connected = std::abs(reduction) >= 1e-9 && drivenCount > 0;

    axleSpeed = drivenCount > 0 ? axleSpeed / static_cast<double>(drivenCount) : 0.0;
    const auto clutchSideSpeed = connected ? axleSpeed * reduction : 0.0;

    // The pedal, whoever is on it, and the automation is a layer over exactly this one number.
    const auto automatic = autoClutchPedal(setup.autoClutch, setup.engine.idleSpeed, clutchSideSpeed, state.engineSpeed,
                                           input.throttle, connected);
    state.clutchPedal = advanceClutchPedal(setup.autoClutch, state.clutchPedal, input.clutch, automatic, deltaTime);

    result.slipEnergy = state.slipEnergy;

    // Neutral, or a car with nothing driven: the engine is on its own, spinning against its own
    // friction, and the coupling is holding no two things together to have a mode about.
    if (!connected)
    {
        state.clutch = CouplingState{};
        state.engineSpeed += (flywheel / engineInertia) * deltaTime;
        settleEngineSpeed(setup.engine, state);

        return result;
    }

    // The wheels' own inertia and nothing else, which is the right one over a tick: the car's mass
    // reaches the wheel through the tire, and the tire builds its force over a relaxation *length*
    // rather than instantly. Folding the car in here instead — 122 kg.m^2 against the wheels' 2.4 —
    // was tried and is unstable, because the vehicle tick then integrates the wheel against 1.2 while
    // the coupling sized its torque against a hundred times that: measured, it turned a launch's one
    // lock transition into forty of them banging between plus and minus the whole capacity.
    const auto referredInertia = axleInertia / (reduction * reduction);

    const auto sides = CouplingSides{.drivingSpeed = state.engineSpeed,
                                     .drivenSpeed = clutchSideSpeed,
                                     .drivingInertia = engineInertia,
                                     .drivenInertia = std::max(referredInertia, 1e-9),
                                     .drivingTorque = flywheel,
                                     .drivenTorque = axleRoadTorque / reduction,
                                     .capacity = 0.0};

    const auto solved = stepDriveCoupling(setup.coupling, state.clutch, sides, state.clutchPedal, deltaTime);
    if (!solved)
    {
        return std::unexpected(solved.error());
    }

    const auto clutchTorque = solved->torque;

    // The engine takes what the clutch did not. A stalled one takes nothing: `settleEngineSpeed`
    // holds it at rest, which is what its own compression does, and the torque still crosses to the
    // driveline — that is why a stalled car in gear drags itself to a stop rather than coasting.
    state.engineSpeed += ((flywheel - clutchTorque) / engineInertia) * deltaTime;
    settleEngineSpeed(setup.engine, state);

    state.slipEnergy += std::abs(clutchTorque * solved->slipSpeed) * deltaTime;

    // Through the gearing to the differential, and out to the wheels it decides between.
    const auto axleTorque = clutchTorque * reduction;

    if (driven == DrivenAxle::All)
    {
        // Split evenly front to rear before each differential, which is a centre spool. A centre
        // differential is the same interface again and is somebody else's milestone. The two
        // differentials are the same setup asked twice and each keeps its own state.
        const auto front = setup.differential.split(state.differentials[0], wheelSpeeds[0], wheelSpeeds[1],
                                                    0.5 * axleTorque, deltaTime);
        const auto rear = setup.differential.split(state.differentials[1], wheelSpeeds[2], wheelSpeeds[3],
                                                   0.5 * axleTorque, deltaTime);

        result.wheel = {front.left, front.right, rear.left, rear.right};
    }
    else if (driven == DrivenAxle::Front)
    {
        const auto split =
            setup.differential.split(state.differentials[0], wheelSpeeds[0], wheelSpeeds[1], axleTorque, deltaTime);
        result.wheel = {split.left, split.right, 0.0, 0.0};
    }
    else
    {
        const auto split =
            setup.differential.split(state.differentials[1], wheelSpeeds[2], wheelSpeeds[3], axleTorque, deltaTime);
        result.wheel = {0.0, 0.0, split.left, split.right};
    }

    result.clutch = clutchTorque;
    result.clutchSlip = solved->slipSpeed;
    result.clutchLocked = solved->locked;
    result.slipEnergy = state.slipEnergy;

    return result;
}

// The driveline's own telemetry channels, filled by whoever stepped it. `:Vehicle` fills the rest of
// the frame and cannot fill these — it does not import this partition and must not — so a caller
// stepping both is what joins them, and this is that caller's tool rather than five assignments
// restated at every site.
export void fillDrivelineTelemetry(TelemetryFrame& frame, const DrivelineState& state, const DrivelineTorques& torques)
{
    frame.engineSpeed = state.engineSpeed;
    frame.engineTorque = torques.engine;
    frame.clutchTorque = torques.clutch;
    frame.clutchSlip = torques.clutchSlip;
    frame.clutchSlipEnergy = torques.slipEnergy;
}

// The placeholder car's driveline: a small turbocharged four driving the front wheels through a
// six-speed and an open differential. Every number a placeholder, and the torque curve shaped like a
// modern turbo's — flat and early — rather than like a naturally aspirated one.
export [[nodiscard]] DrivelineSetup placeholderDriveline()
{
    auto setup = DrivelineSetup{};

    // rad/s against N.m. 1000 rpm is 105 rad/s.
    setup.engine.torque = Curve{.points = {glm::dvec2(0.0, 60.0), glm::dvec2(105.0, 160.0), glm::dvec2(200.0, 310.0),
                                           glm::dvec2(300.0, 350.0), glm::dvec2(470.0, 350.0), glm::dvec2(600.0, 300.0),
                                           glm::dvec2(712.0, 250.0)}};
    setup.engine.inertia = 0.15;
    setup.engine.limiterSpeed = 712.0;
    setup.engine.coastTorque = 75.0;

    setup.gearbox.ratios = {3.19, 2.08, 1.47, 1.20, 0.99, 0.80};
    setup.gearbox.finalDrive = 4.37;

    // A single dry plate, and the defaults on `FrictionClutch` are this car's. Stated rather than
    // left implied, because it is the line that says which kind is in the slot.
    setup.coupling.kind = DriveCouplingKind::FrictionClutch;

    setup.driven = DrivenAxle::Front;
    setup.differential = openDifferential();

    return setup;
}

// `DrivelineTorques::wheel` is handed straight to `stepVehicle`, which is the whole of how the
// driveline reaches the road. It used to be applied here instead, integrating `wheelSpeed` before
// the vehicle tick integrated the same field again from the road — so the brake clamp inside that
// tick sized itself against one of the two torques and knew nothing of the other. Throttle and
// brake together were therefore inconsistent, and launch, creep and converter stall are all exactly
// that case.
//
// Calling this from the game's loop rather than from inside `stepVehicle` is unchanged and is still
// the point: which wheels a car drives is a property of the car and not of its suspension, and
// keeping them apart is what let the whole vehicle be built and validated before an engine existed.

} // namespace raceengine
