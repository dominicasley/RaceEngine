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

    // How far the speed has to fall before the fuel comes back. A bare threshold re-arms on the
    // very tick the cut has slowed the engine past it, so the limiter chatters at the *timestep's*
    // frequency rather than at the engine's — halve the tick and it chatters twice as fast, which is
    // the signature of an artefact rather than a device. 16 rad/s is about 150 rpm. Zero reproduces
    // the bare threshold exactly, which is how the chatter is measured against itself.
    double limiterRestoreBand = 16.0;

    IdleGovernor governor;

    // Engine braking: what it absorbs at the limiter with the throttle shut, falling linearly to
    // nothing at rest. Deliberately separate from the torque curve, which is a full-throttle
    // measurement and has no business carrying the closed-throttle behaviour as a negative number.
    double coastTorque = 75.0;
};

// Torque at the flywheel. Positive drives, negative brakes.
//
// The limiter cuts fuel rather than shaping the curve, which is why it is a cliff here and not a
// taper: that abruptness is what a driver feels. And cutting fuel does not merely stop the engine
// driving — it makes it *brake*, because a cylinder still pumping with nothing burning in it is a
// compressor. Scaling the braking by a shut throttle alone would have an engine on its limiter
// coasting freely, which is not what one does.
//
// Whether the fuel is cut arrives as an argument because the *decision* has memory and this function
// has none: see `advanceRevLimiter`.
export [[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle,
                                         const bool fuelCut)
{
    const auto demand = std::clamp(throttle, 0.0, 1.0);

    const auto driving = fuelCut ? 0.0 : demand * engine.torque.at(speed);
    const auto pumping = fuelCut ? 1.0 : 1.0 - demand;
    const auto braking = pumping * engine.coastTorque * std::clamp(speed / engine.limiterSpeed, 0.0, 1.0);

    return driving - braking;
}

// The same engine with the limiter answered from this instant alone, which is the right question for
// anything sweeping the curve and the wrong one for anything running it.
export [[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle)
{
    return engineTorque(engine, speed, throttle, speed >= engine.limiterSpeed);
}

// Where the fuel cut goes next, given where it is. Two thresholds, and deliberately *not*
// `stepCoupling`'s two-thresholds-and-a-dwell: that machine answers "hold or slide" for a pair of
// inertias and hands back a torque, and a limiter has no second side, no capacity and no torque to
// clamp. Reaching it through `CouplingSides` would mean inventing all three and then reading the
// answer out of a bool riding on a torque solver — an interface that reads as reuse and behaves as a
// comment, which is `peakSlipScale`'s failure written again. What is shared is the *idea*, and the
// dwell is not even part of it here: the band is crossed at a rate the engine's own inertia sets, so
// there is nothing left for a dwell to stop.
export [[nodiscard]] bool advanceRevLimiter(const EngineModel& engine, const bool fuelCut, const double speed)
{
    return fuelCut ? speed > engine.limiterSpeed - std::max(engine.limiterRestoreBand, 0.0)
                   : speed >= engine.limiterSpeed;
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

// Where a gearbox is between two gears. The middle three are one torque interrupt — nothing reaches
// the wheels while a ratio is being changed — and they are three rather than one because the rules
// differ across them: the ratio is selected on entering `Neutral`, that window is the only place the
// target may still move, and `Engaging` is the box committed.
export enum class ShiftPhase : std::uint32_t { Engaged, Disengaging, Neutral, Engaging };

export struct ShiftTiming
{
    // Not one number stated twice. An upshift is a clutch-to-clutch handover on a box that has
    // already pre-selected the gear on its other input shaft, and 8 ms is what this car's data
    // states. A downshift has first to raise the engine by a third of its speed, and 100 ms is very
    // nearly what 0.15 kg.m^2 takes to gain 200 rad/s on 300 N.m — so the downshift figure is a
    // rev-match time, and that is why the duration belongs to the *direction* rather than to the box.
    double upshiftTime = 0.008;
    double downshiftTime = 0.100;

    // Where the three sub-phases fall inside it. Not equal work: the middle is where the new ratio is
    // selected and where a blip has to happen, so it gets the half.
    double disengageFraction = 0.25;
    double engageFraction = 0.25;
};

export struct Gearbox
{
    // Ratios by gear, index 0 being first. Reverse and neutral are handled by `gear` below rather
    // than by living in here.
    std::vector<double> ratios;
    double finalDrive = 4.37;
    double reverseRatio = 3.6;

    ShiftTiming shift;

    // The highest forward gear this box actually has.
    [[nodiscard]] std::int32_t topGear() const
    {
        return static_cast<std::int32_t>(ratios.size());
    }

    // Every gear number this model produces goes through here first. `reduction` below answers an
    // impossible gear with the *top* ratio — an eighth-gear request in a six-speed comes back as
    // sixth, which is a plausible number and therefore the expensive kind of wrong. Clamping where
    // the number is made means that clamp is never the thing that answers.
    [[nodiscard]] std::int32_t clampGear(const std::int32_t gear) const
    {
        return std::clamp(gear, -1, topGear());
    }

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

// Rev matching, and it is *one* controller in both directions rather than a blip and a cut bolted
// together. The target is the speed the gear being engaged will demand: asking for it while the
// engine is below it is a blip, and asking for it while the engine is above it is a shut throttle,
// which is exactly the torque cut an upshift wants. Off, the driver's own pedal stands through the
// shift and the coupling takes up whatever mismatch is left — which is the comparison the slip
// energy is read from.
export struct ShiftAssist
{
    bool revMatch = true;

    // Throttle per rad/s of error: full throttle 50 rad/s — about 480 rpm — short of the target.
    double gain = 0.02;
};

// The throttle a rev match is asking for. Pure and exported, because what an assist asks for is
// worth pinning without a car around it, and because it is a P controller and nothing more: the
// engine's own inertia is the plant and the blip goes through `engineTorque` exactly as the driver's
// pedal does. Nothing here reaches past the model to set a speed.
export [[nodiscard]] double revMatchThrottle(const EngineModel& engine, const ShiftAssist& assist,
                                             const double targetSpeed, const double engineSpeed)
{
    // A downshift whose target is past the limiter is a money shift: the assist blips to the limiter
    // and no further, and what the *game* does about the rest of that request is a gameplay decision
    // taken somewhere else.
    const auto target = std::clamp(targetSpeed, engine.idleSpeed, engine.limiterSpeed);

    return std::clamp(assist.gain * (target - engineSpeed), 0.0, 1.0);
}

export struct DrivelineSetup
{
    EngineModel engine;
    Gearbox gearbox;
    // The slot between the engine's inertia and the gearbox input, and this file names neither of the
    // two things that can be in it after this line.
    DriveCoupling coupling;
    AutoClutch autoClutch;
    ShiftAssist shiftAssist;
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
// string. Driveline wind-up, the gearbox output shaft and the converter's turbine still belong here
// and are still not here.
export struct DrivelineState
{
    // Independent state from the start even though a locked clutch makes it derivable, because
    // making it independent later is a restructure and making it independent now is free.
    double engineSpeed = 0.0;

    // The gear actually in mesh, which is not the gear the driver has asked for: `VehicleInput::gear`
    // is a *demand* and this is what the shift machine has got round to. A demand is a level, so it
    // survives a replayed tick unchanged; the paddles that produce it are events and are handled a
    // layer up, in `operateTransmission`.
    std::int32_t gear = 0;
    std::int32_t targetGear = 0;
    // The gear the shift in progress left. Kept because the duration is a property of the direction
    // and a retarget inside the neutral window can change which direction that is.
    std::int32_t shiftFrom = 0;

    ShiftPhase shiftPhase = ShiftPhase::Engaged;
    double shiftTimer = 0.0;

    // The limiter's one bit of memory, which is the whole of what stops it chattering.
    bool fuelCut = false;

    // A default-constructed driveline is a car with the key out: nothing has started this engine, so
    // it is not turning. `startEngine` is what changes that, and it is deliberately not an input.
    EngineState engine = EngineState::Stalled;

    // The coupling slot's own state, whichever kind is fitted.
    DriveCouplingState coupling{};
    // The pedal actually being held, which is the driver's when the driver is on it and the
    // automation's when nobody is. Kept rather than recomputed so the handover between the two is
    // continuous — see `advanceClutchPedal`.
    double clutchPedal = 0.0;

    double idleIntegral = 0.0;

    // What the coupling has turned into heat since the run began, in joules — a plate's friction or a
    // converter's fluid. A running total rather than a per-tick figure: the thermal model that will
    // read it integrates, and a channel that had to be integrated by whoever plots it is a channel
    // that gets integrated differently twice.
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
    // What the coupling delivered to the gearbox input.
    double clutch = 0.0;
    // And what it took off the engine, which is the same number for a friction clutch and smaller
    // for a converter by exactly the torque ratio — the difference is the stator's reaction into the
    // housing. Without both, an engine's own torque balance cannot be reconstructed from telemetry.
    double clutchReaction = 0.0;
    double engine = 0.0;
    // How far out of step the two sides of the clutch are, in rad/s. It does not reach zero when the
    // coupling locks: only the engine's half of the constraint is integrated here, the driveline's
    // half being the vehicle tick's, so a locked clutch converges over a handful of ticks rather
    // than within one.
    double clutchSlip = 0.0;
    bool clutchLocked = false;
    double slipEnergy = 0.0;

    // The gear in mesh and where the box is between two of them, so nothing has to reach into the
    // state to plot the one channel a shift shows up in.
    std::int32_t gear = 0;
    ShiftPhase shiftPhase = ShiftPhase::Engaged;

    // The driven axle's inertia referred through the gearing, and it is reported rather than left
    // internal because it is precisely the number a shift written as a shrinking ratio destroys: it
    // goes as 1/reduction^2, so a ratio taken toward zero sends it to infinity and the solve with it.
    // Reported every tick, so "it stays finite through every phase" is a measurement rather than an
    // assurance. Zero in neutral, where there is no gearing to refer anything through.
    double referredInertia = 0.0;

    bool fuelCut = false;
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

// Only two forward gears have a shift between them worth timing. Neutral at either end is not a
// ratio change under load — there is nothing to pull out of, and the auto-clutch is already the
// device that matches the two speeds — so it engages directly. That is also what keeps a caller who
// simply states the gear it wants from paying for a state machine it never asked for.
[[nodiscard]] bool timedShift(const std::int32_t from, const std::int32_t to)
{
    return from >= 1 && to >= 1 && from != to;
}

[[nodiscard]] double shiftDuration(const Gearbox& gearbox, const std::int32_t from, const std::int32_t to)
{
    return std::max(to > from ? gearbox.shift.upshiftTime : gearbox.shift.downshiftTime, 0.0);
}

// The shift machine. `demanded` is a *level* — the gear the driver is still asking for — which is
// what makes a request neither repeat on a replayed tick nor ever be lost: a paddle pulled while the
// box is busy leaves the demand standing, and the machine picks it up the moment it is free. That is
// the whole of the queue-or-drop question, answered by having no queue to get wrong.
void advanceShift(const Gearbox& gearbox, DrivelineState& state, const std::int32_t demanded, const double deltaTime)
{
    const auto demand = gearbox.clampGear(demanded);

    if (state.shiftPhase == ShiftPhase::Engaged)
    {
        if (demand == state.gear)
        {
            return;
        }

        if (!timedShift(state.gear, demand))
        {
            state.gear = demand;
            state.targetGear = demand;
            state.shiftFrom = demand;

            return;
        }

        state.shiftFrom = state.gear;
        state.targetGear = demand;
        state.shiftPhase = ShiftPhase::Disengaging;
        state.shiftTimer = 0.0;

        return;
    }

    state.shiftTimer += deltaTime;

    const auto total = shiftDuration(gearbox, state.shiftFrom, state.targetGear);
    const auto opened = total * std::clamp(gearbox.shift.disengageFraction, 0.0, 1.0);
    // Never before `opened`, whatever the two fractions add up to, or a box with generous ramps would
    // reach the committed phase without ever passing through the window the ratio changes in.
    const auto closing = std::max(total * (1.0 - std::clamp(gearbox.shift.engageFraction, 0.0, 1.0)), opened);

    if (state.shiftTimer >= total)
    {
        state.gear = state.targetGear;
        state.shiftFrom = state.targetGear;
        state.shiftPhase = ShiftPhase::Engaged;
        state.shiftTimer = 0.0;

        return;
    }

    if (state.shiftTimer >= closing)
    {
        state.shiftPhase = ShiftPhase::Engaging;
        state.gear = state.targetGear;

        return;
    }

    if (state.shiftTimer >= opened)
    {
        // The neutral window, and the only place the target may still move: before it the box has
        // selected nothing, after it the gear is going in. The *ratio* changes here too, where
        // nothing is being transmitted — which is the one way a shift may change it at all, because
        // the referred inertia goes as 1/reduction^2 and a ratio walked toward zero takes the solve
        // with it long before it arrives.
        state.shiftPhase = ShiftPhase::Neutral;

        if (demand != state.targetGear && timedShift(state.shiftFrom, demand))
        {
            state.targetGear = demand;
        }

        state.gear = state.targetGear;

        return;
    }

    state.shiftPhase = ShiftPhase::Disengaging;
}

} // namespace

// One tick of the chain. `state` in, torques out, and the caller integrates the wheels.
//
// Fallible, and the one thing that can fail is the coupling slot: it answers for the kinds it has a
// model for and refuses for any other rather than falling through to a neighbour's answer. Nothing
// else in here knows which coupling is fitted, which is why the driver's pedal and the gear both go
// into the slot whole and each kind drops what it does not read.
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

    axleSpeed = drivenCount > 0 ? axleSpeed / static_cast<double>(drivenCount) : 0.0;

    // The driver's gear is a demand; the machine below decides when the box has got there.
    advanceShift(setup.gearbox, state, input.gear, deltaTime);

    const auto reduction = setup.gearbox.reduction(state.gear);
    const auto geared = std::abs(reduction) >= 1e-9;

    result.gear = state.gear;
    result.shiftPhase = state.shiftPhase;
    result.referredInertia = geared ? axleInertia / (reduction * reduction) : 0.0;

    // A shift is a torque interrupt, and it is expressed as the gearbox being *open* rather than as
    // a ratio on its way to zero. Open, the path is the one this function already had for neutral —
    // one mechanism, no knowledge of which coupling is fitted, and an exact zero at the wheels — and
    // the ratio it will come back in changes while nothing is going through it.
    //
    // The softness of the re-engagement is left to the coupling, which already has the machinery for
    // it: at the instant the box closes, the plate is slipping across whatever the rev match left and
    // re-locks over a handful of ticks exactly as it does after a launch. That is a modelled
    // softness rather than an authored ramp, and it is why there is no partially-transmitting
    // gearbox here to get the torque balance wrong in.
    const auto connected = geared && drivenCount > 0 && state.shiftPhase == ShiftPhase::Engaged;

    const auto clutchSideSpeed = connected ? axleSpeed * reduction : 0.0;

    // The bypass and the driver's pedal are parallel air paths, so the engine gets the larger of the
    // two and never their sum: a governor that added to a wide-open throttle would be making torque
    // that no valve was opened for.
    const auto bypass = running ? idleBypass(setup.engine, state.engineSpeed, state.idleIntegral, deltaTime) : 0.0;

    // The rev match *takes the pedal away* rather than adding to it, and that is what separates it
    // from the bypass beside it: a bypass is a second air path, an ECU shift cut is the ECU holding
    // the plate shut against the driver's foot. The idle path still wins underneath, because the one
    // thing neither may do is stall the engine.
    const auto matching = state.shiftPhase != ShiftPhase::Engaged && setup.shiftAssist.revMatch;
    const auto pedal = matching
                           ? revMatchThrottle(setup.engine, setup.shiftAssist,
                                              axleSpeed * setup.gearbox.reduction(state.targetGear), state.engineSpeed)
                           : std::clamp(input.throttle, 0.0, 1.0);
    const auto demand = std::max(pedal, bypass);

    state.fuelCut = running && advanceRevLimiter(setup.engine, state.fuelCut, state.engineSpeed);

    const auto flywheel = running ? engineTorque(setup.engine, state.engineSpeed, demand, state.fuelCut) : 0.0;

    result.engine = flywheel;
    result.fuelCut = state.fuelCut;

    // The pedal, whoever is on it, and the automation is a layer over exactly this one number.
    const auto automatic = autoClutchPedal(setup.autoClutch, setup.engine.idleSpeed, clutchSideSpeed, state.engineSpeed,
                                           input.throttle, connected);
    state.clutchPedal = advanceClutchPedal(setup.autoClutch, state.clutchPedal, input.clutch, automatic, deltaTime);

    result.slipEnergy = state.slipEnergy;

    // Neutral, a gear change, or a car with nothing driven: the engine is on its own, spinning
    // against its own friction, and the coupling is holding no two things together to have a mode
    // about.
    if (!connected)
    {
        idleDriveCoupling(setup.coupling, state.coupling, deltaTime);
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
    const auto referredInertia = result.referredInertia;

    const auto sides = CouplingSides{.drivingSpeed = state.engineSpeed,
                                     .drivenSpeed = clutchSideSpeed,
                                     .drivingInertia = engineInertia,
                                     .drivenInertia = std::max(referredInertia, 1e-9),
                                     .drivingTorque = flywheel,
                                     .drivenTorque = axleRoadTorque / reduction,
                                     .capacity = 0.0};

    // The gear in mesh rather than the one asked for: a lockup clutch that read the demand would let
    // go a shift early and take hold one late.
    const auto command = DriveCouplingCommand{.clutchPedal = state.clutchPedal, .gear = state.gear};

    const auto solved = stepDriveCoupling(setup.coupling, state.coupling, sides, command, deltaTime);
    if (!solved)
    {
        return std::unexpected(solved.error());
    }

    // The engine takes what the coupling did not. A stalled one takes nothing: `settleEngineSpeed`
    // holds it at rest, which is what its own compression does, and the torque still crosses to the
    // driveline — that is why a stalled car in gear drags itself to a stop rather than coasting.
    state.engineSpeed += ((flywheel - solved->drivingTorque) / engineInertia) * deltaTime;
    settleEngineSpeed(setup.engine, state);

    state.slipEnergy += solved->slipPower * deltaTime;

    // Through the gearing to the differential, and out to the wheels it decides between. This is the
    // *delivered* torque and not the reaction: they part company the moment the slot holds anything
    // with a member grounded to its own housing.
    const auto axleTorque = solved->drivenTorque * reduction;

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

    result.clutch = solved->drivenTorque;
    result.clutchReaction = solved->drivingTorque;
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
    // The gear in mesh, over the demand the vehicle tick copied off the input packet. They are the
    // same number except during a shift, which is the one time anybody reads the channel.
    frame.gear = torques.gear;
    frame.shiftPhase = static_cast<std::uint32_t>(torques.shiftPhase);
}

// How a car is *driven*, which is a different question from what is in it. Every car in this game is
// driven in semi-manual at this stage — the driver picks gears with paddles whether the box behind
// them is a friction clutch and a manual gearbox or a converter and a planetary one — and that is
// exactly why the two are split. The transmission model is whatever it physically is; the operation
// mode translates intent into the commands that model already takes. Full manual (an H-pattern and
// the driver's own clutch) and full automatic (the mode picking gears itself) are then modes added
// here, and neither touches a transmission model.
//
// `SuspensionKind`'s shape again, and one case for the same reason `DriveCouplingKind` had one: a
// slot that quietly answered as its neighbour is a feature that reads as implemented and behaves as
// a comment.
export enum class TransmissionMode : std::uint32_t { SemiManual };

// The lever, as distinct from the paddles. Level-triggered on purpose: "the driver has selected
// reverse" is a state of the world and re-sending it on a replayed tick asks for nothing new.
export enum class GearRange : std::uint32_t { Reverse, Neutral, Drive };

export struct TransmissionOperation
{
    TransmissionMode mode = TransmissionMode::SemiManual;
};

// What the driver is doing, before any of it means anything to a gearbox.
export struct DriverIntent
{
    double steering = 0.0;
    double throttle = 0.0;
    double brake = 0.0;
    // The driver's own pedal, and it stays live in semi-manual on a friction-clutch car. The
    // hand-over is `advanceClutchPedal`'s and is not restated here: past the pedal's free play the
    // foot wins outright, so a driver slipping it from rest cannot be overridden and a driver with a
    // foot off it gets the automation. On a converter car the field is ignored, once, in the slot.
    double clutch = 0.0;

    // Shift requests are **counts, not levels**, and that is the one thing in this struct that had to
    // be decided rather than copied. A "shift up" bit is a level: held for a fifth of a second it
    // asks seventy times at 360 Hz, and an edge taken against *the previous packet* is worse still,
    // because a rollback restores the state and then replays inputs against whatever packet the
    // consumer happened to be holding. A monotonic count has neither failure — a held paddle changes
    // nothing, a replayed tick re-reads the same number against a `TransmissionState` that was
    // rolled back with everything else, and a packet lost on the wire still delivers its request,
    // because the count arrives late rather than not at all.
    std::uint32_t upshifts = 0;
    std::uint32_t downshifts = 0;

    GearRange range = GearRange::Drive;
};

// The operation mode's own state, and it is deliberately not part of `DrivelineState`. That struct is
// the transmission *model's*; this is the driver's side of the seam, and keeping them apart is what
// makes "adding full manual is a mode, not a change to the gearbox" true rather than intended.
export struct TransmissionState
{
    std::int32_t gearDemand = 0;
    std::uint32_t upshiftsSeen = 0;
    std::uint32_t downshiftsSeen = 0;
};

static_assert(std::is_trivially_copyable_v<TransmissionState>, "the mode's state is saved by copying its bytes too");
static_assert(std::is_standard_layout_v<TransmissionState>, "and rollback will later");

// One tick of the driver's side. Out comes the packet the transmission model already takes, so the
// caller hands the same `VehicleInput` to `stepVehicle` and `stepDriveline` and nothing downstream
// learns that paddles exist.
//
// `driveline` and `deltaTime` are read by nothing in semi-manual and are taken anyway, for
// `Differential::split`'s reason: an automatic selects its own gears from engine speed against a
// schedule with hysteresis measured in time, and adding either argument later moves every call site.
export [[nodiscard]] std::expected<VehicleInput, std::string>
operateTransmission(const DrivelineSetup& setup, const TransmissionOperation& operation, TransmissionState& state,
                    const DrivelineState& driveline, const DriverIntent& intent, const double deltaTime)
{
    switch (operation.mode)
    {
    case TransmissionMode::SemiManual:
    {
        static_cast<void>(driveline);
        static_cast<void>(deltaTime);

        const auto pulled = [](const std::uint32_t requested, std::uint32_t& seen)
        {
            // A count that went *backwards* is a state restored under a fresh input stream, not four
            // billion requests. Resynchronise and ask for nothing.
            const auto pending = requested >= seen ? std::min(requested - seen, std::uint32_t{8}) : std::uint32_t{0};
            seen = requested;

            return static_cast<std::int32_t>(pending);
        };

        const auto up = pulled(intent.upshifts, state.upshiftsSeen);
        const auto down = pulled(intent.downshifts, state.downshiftsSeen);

        switch (intent.range)
        {
        case GearRange::Reverse:
            state.gearDemand = -1;
            break;
        case GearRange::Neutral:
            state.gearDemand = 0;
            break;
        case GearRange::Drive:
            // Clamped into the gears the box has, here, where the number is made. Paddles cannot walk
            // out of first into neutral either: neutral and reverse are the lever's, which is both
            // what the car does and what stops a downshift under braking finding neutral.
            state.gearDemand =
                std::clamp(std::max(state.gearDemand, 1) + up - down, 1, std::max(setup.gearbox.topGear(), 1));
            break;
        }

        auto input = VehicleInput{};
        input.steering = intent.steering;
        input.throttle = intent.throttle;
        input.brake = intent.brake;
        input.clutch = intent.clutch;
        input.gear = state.gearDemand;

        return input;
    }
    }

    return std::unexpected("the transmission is being driven in a mode this build has no operation for");
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

// The same car with a torque converter in the slot instead of a plate. Its ratios are an automatic's
// rather than the manual's — a wider first, a taller top and a shorter final drive — because a
// converter already multiplies at the bottom and a gearbox behind one is geared for that. Nothing
// here touches the stall speed, which falls out of the converter's own curves and out of the engine's
// torque at the speed they cross.
export [[nodiscard]] DrivelineSetup placeholderAutomatic()
{
    auto setup = placeholderDriveline();

    setup.coupling.kind = DriveCouplingKind::TorqueConverter;

    setup.gearbox.ratios = {4.15, 2.37, 1.56, 1.16, 0.86, 0.69};
    setup.gearbox.finalDrive = 3.20;

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
