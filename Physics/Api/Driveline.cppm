module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:Driveline;

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

export struct EngineModel
{
    // Torque against engine speed in rad/s, at full throttle. A curve rather than a peak and a
    // shape, because a real engine is measured rather than described.
    Curve torque;

    // Placeholder: a small turbocharged four.
    double inertia = 0.15;
    double idleSpeed = 89.0;     // ~850 rpm
    double limiterSpeed = 712.0; // ~6800 rpm

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

// The clutch, currently locked, and an element in the chain either way.
//
// A locked clutch is a rigid coupling and a slipping one is a friction element with a capacity
// curve; both sit here, and both are `capacity`. The engine keeps its own speed through either,
// which is why turning slip on later is a change to one number rather than to the shape of the
// model.
export struct Clutch
{
    // What it can hold before it slips. The placeholder is far past anything this engine makes, so
    // it never does.
    double capacity = 100000.0;
    // How hard it pulls the two sides together. Solved implicitly below, so this can be as stiff as
    // a locked clutch needs without the timestep caring.
    double stiffness = 8000.0;
};

export struct DrivelineSetup
{
    EngineModel engine;
    Gearbox gearbox;
    Clutch clutch;
    Differential differential = openDifferential();
    DrivenAxle driven = DrivenAxle::Front;
};

// The driveline's own state, and it is here rather than in `VehicleState` for the reason every
// other split in this module is made: what integrates a quantity owns it. Engine speed sat in the
// vehicle's state where nothing in the vehicle model read it or wrote it, which made `:Vehicle` the
// keeper of a number belonging to a partition it does not even import.
//
// Trivially copyable and standard layout, and it stays that way — save and restore is a memcpy and
// rollback will later lean on that. Scalars, enums and fixed arrays only: no `Curve`, no vector, no
// string. The clutch's lock and dwell, the shift phase and timer, driveline wind-up, the gearbox
// output shaft, the converter's turbine and the slip-energy ledger all belong here and none of them
// is here yet.
export struct DrivelineState
{
    // Independent state from the start even though a locked clutch makes it derivable, because
    // making it independent later is a restructure and making it independent now is free.
    double engineSpeed = 0.0;

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
    // How far out of step the two sides of the clutch are. Zero while it is locked, and the channel
    // that will say how much it is slipping when it is allowed to.
    double clutchSlip = 0.0;
};

// One tick of the chain. Pure: state in, torques out, and the caller integrates.
//
// The clutch is solved implicitly rather than stepped. A locked clutch is a very stiff coupling
// between two small inertias, and stepped explicitly it is the single most unstable thing in a
// driveline — the engine and the gearbox trade energy back and forth at the timestep's frequency
// until one of them reaches infinity. Solved, it is unconditionally stable at any stiffness, which
// is what lets "locked" mean locked rather than "stiff enough to look locked".
export [[nodiscard]] DrivelineTorques stepDriveline(const DrivelineSetup& setup, DrivelineState& state,
                                                    const std::array<double, cornerCount>& wheelSpeeds,
                                                    const std::array<double, cornerCount>& wheelInertias,
                                                    const double throttle, const std::int32_t gear,
                                                    const double deltaTime)
{
    auto result = DrivelineTorques{};

    const auto flywheel = engineTorque(setup.engine, state.engineSpeed, throttle);
    const auto reduction = setup.gearbox.reduction(gear);

    // Neutral: the engine is on its own, spinning against its own friction. No torque reaches the
    // wheels and none comes back.
    //
    // The floor at zero here and at the two sites below is the only thing stopping the engine
    // turning backwards, and therefore the only thing stopping it stalling. It comes out together
    // with the stall model and the idle controller and not before: removing it on its own leaves
    // negative engine speeds with nothing to catch them.
    if (std::abs(reduction) < 1e-9)
    {
        state.engineSpeed =
            std::max(0.0, state.engineSpeed + (flywheel / std::max(setup.engine.inertia, 1e-9)) * deltaTime);
        result.engine = flywheel;

        return result;
    }

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
    auto drivenCount = 0;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        if (!isDriven(index))
        {
            continue;
        }

        axleSpeed += wheelSpeeds[index];
        axleInertia += wheelInertias[index];
        drivenCount++;
    }

    if (drivenCount == 0)
    {
        state.engineSpeed =
            std::max(0.0, state.engineSpeed + (flywheel / std::max(setup.engine.inertia, 1e-9)) * deltaTime);
        return result;
    }

    axleSpeed /= static_cast<double>(drivenCount);

    const auto clutchSideSpeed = axleSpeed * reduction;
    const auto referredInertia = axleInertia / (reduction * reduction);

    // Implicit solve for the slip across the clutch. Both sides accelerate under the torque it
    // carries, so the slip decays at a rate set by the stiffness and the two inertias.
    const auto engineInertia = std::max(setup.engine.inertia, 1e-9);
    const auto mobility = 1.0 / engineInertia + 1.0 / std::max(referredInertia, 1e-9);

    auto slip = state.engineSpeed - clutchSideSpeed;
    slip = (slip + deltaTime * (flywheel / engineInertia)) / (1.0 + deltaTime * setup.clutch.stiffness * mobility);

    const auto clutchTorque = std::clamp(setup.clutch.stiffness * slip, -setup.clutch.capacity, setup.clutch.capacity);

    // Engine takes what the clutch did not.
    state.engineSpeed = std::max(0.0, state.engineSpeed + ((flywheel - clutchTorque) / engineInertia) * deltaTime);

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
    result.engine = flywheel;
    result.clutchSlip = slip;

    return result;
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
