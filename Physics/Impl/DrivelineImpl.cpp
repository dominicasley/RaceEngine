// Driveline bodies. Declarations are in Api/Driveline.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <string>
#include <type_traits>
#include <vector>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] double throttleAirFlow(const EngineModel& engine, const double speed, const double throttle)
{
    const auto& body = engine.throttle;

    const auto span = 1.0 - std::cos(std::max(body.fullOpenAngle, 1e-6));
    const auto leak = std::clamp(body.leakArea, 0.0, 1.0);
    const auto plate = span > 1e-12 ? (1.0 - std::cos(std::clamp(throttle, 0.0, 1.0) * body.fullOpenAngle)) / span
                                    : std::clamp(throttle, 0.0, 1.0);

    const auto area = leak + (1.0 - leak) * std::clamp(plate, 0.0, 1.0);

    // The manifold pressure a plate of this area holds against an engine drawing air in proportion
    // to its own speed. Both flows equated and solved: the throttle passes with the square root of
    // what is left of the pressure drop, the engine swallows in proportion to what is there.
    const auto restriction = std::abs(speed) / std::max(body.chokedSpeed, 1e-9);
    const auto pressure = [restriction](const double open)
    {
        const auto denominator = std::sqrt(open * open + restriction * restriction);

        return denominator > 1e-12 ? open / denominator : 0.0;
    };

    const auto wideOpen = pressure(1.0);

    return wideOpen > 1e-12 ? std::clamp(pressure(area) / wideOpen, 0.0, 1.0) : 0.0;
}

[[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle,
                                  const bool fuelCut)
{
    const auto losses =
        std::max(engine.coastTorque, 0.0) * std::clamp(speed / std::max(engine.limiterSpeed, 1e-9), 0.0, 1.0);
    const auto full = engine.torque.at(speed);

    // Cutting fuel does not merely stop the engine driving — it makes it *brake*, because a cylinder
    // still pumping with nothing burning in it is a compressor. That is why the limiter is a cliff
    // here and not a taper: the abruptness is what a driver feels.
    if (fuelCut)
    {
        return -losses;
    }

    return throttleAirFlow(engine, speed, throttle) * (full + losses) - losses;
}

[[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle)
{
    const auto overrun = engine.overrun.enabled && throttle <= engine.overrun.throttleThreshold &&
                         speed >= engine.overrun.cutFraction * engine.idleSpeed;

    return engineTorque(engine, speed, throttle, speed >= engine.limiterSpeed || overrun);
}

[[nodiscard]] bool advanceRevLimiter(const EngineModel& engine, const bool fuelCut, const double speed)
{
    return fuelCut ? speed > engine.limiterSpeed - std::max(engine.limiterRestoreBand, 0.0)
                   : speed >= engine.limiterSpeed;
}

[[nodiscard]] bool advanceOverrunCut(const EngineModel& engine, const bool cut, const double speed,
                                     const double throttle)
{
    if (!engine.overrun.enabled || throttle > engine.overrun.throttleThreshold)
    {
        return false;
    }

    return cut ? speed > engine.overrun.restoreFraction * engine.idleSpeed
               : speed >= engine.overrun.cutFraction * engine.idleSpeed;
}

[[nodiscard]] double idleBypass(const EngineModel& engine, const double speed, double& integral, const double deltaTime)
{
    const auto error = engine.idleSpeed - speed;

    // A valve can only add air, so the integral is clamped to the range the output can ever be asked
    // for and never runs negative. An integrator left running while the output is saturated takes as
    // long to come back as it took to wind up, which reads as an engine that hangs after every
    // clutch release rather than as a controller fault.
    integral = std::clamp(integral + engine.governor.integral * error * deltaTime, 0.0, engine.governor.maximumBypass);

    return std::clamp(engine.governor.proportional * error + integral, 0.0, engine.governor.maximumBypass);
}

[[nodiscard]] Differential openDifferential()
{
    return Differential{};
}

[[nodiscard]] Differential spool()
{
    // Locked solid, and both numbers are needed to say so. An enormous capacity is what stops the
    // pack ever reaching its limit; a lock slip speed past anything a wheel can do is what makes it
    // hold across a difference rather than sliding at that capacity, which for 1e6 N.m would put a
    // megawatt of nonsense into the wheels on the first tick of a corner.
    return Differential{.preload = 1e6, .pack = {.lockSlipSpeed = 1e9, .slipDwell = 0.0}};
}

[[nodiscard]] Differential clutchPackLsd(const double preload, const double powerRamp, const double coastRamp)
{
    return Differential{.preload = preload, .powerRamp = powerRamp, .coastRamp = coastRamp};
}

[[nodiscard]] double rampLockFraction(const RampGeometry& geometry, const double rampAngle)
{
    const auto radius = std::max(geometry.rampRadius, 1e-9);
    const auto tangent = std::tan(std::clamp(rampAngle, 1e-6, 1.5707963267948966));

    return 0.5 * (geometry.plateRadius / radius) * static_cast<double>(geometry.faces) *
           std::max(geometry.frictionCoefficient, 0.0) / tangent;
}

[[nodiscard]] double torqueBiasRatio(const double lockFraction)
{
    const auto fraction = std::clamp(lockFraction, 0.0, 0.5 - 1e-9);

    return (0.5 + fraction) / (0.5 - fraction);
}

[[nodiscard]] Differential rampLsd(const double preload, const double powerAngle, const double coastAngle,
                                   const RampGeometry& geometry)
{
    return clutchPackLsd(preload, rampLockFraction(geometry, powerAngle), rampLockFraction(geometry, coastAngle));
}

[[nodiscard]] double revMatchThrottle(const EngineModel& engine, const ShiftAssist& assist, const double targetSpeed,
                                      const double engineSpeed)
{
    // A downshift whose target is past the limiter is a money shift: the assist blips to the limiter
    // and no further, and what the *game* does about the rest of that request is a gameplay decision
    // taken somewhere else.
    const auto target = std::clamp(targetSpeed, engine.idleSpeed, engine.limiterSpeed);

    return std::clamp(assist.gain * (target - engineSpeed), 0.0, 1.0);
}

void startEngine(const DrivelineSetup& setup, DrivelineState& state)
{
    state.engine = EngineState::Running;
    state.engineSpeed = std::max(state.engineSpeed, setup.engine.idleSpeed);
    state.idleIntegral = 0.0;
}

void placeDriveline(const DrivelineSetup& setup, DrivelineState& state, const double axleSpeed)
{
    // **The gear's own final drive**, which on a two-final transaxle is not one number. Placing a car
    // in sixth with the first axle's ratio would wind the shaft against a speed it never turns at.
    state.shaftSpeed = axleSpeed * setup.gearbox.finalFor(state.gear);
    state.windUp = 0.0;
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

// What reaches the differential, given what the gearbox output is carrying. Declared in
// Api/Driveline.cppm, where the reasoning lives.
[[nodiscard]] double throughDrivelineLosses(const DrivelineLosses& losses, const double torque, const double speed)
{
    // A driveline that ate more than 95% of what went through it is a seized one, and the reciprocal
    // below would turn a plausible number into an enormous one rather than failing visibly.
    const auto efficiency = std::clamp(losses.efficiency, 0.05, 1.0);

    // Torque and speed agreeing in sign is power flowing toward the wheels. At a standstill there is
    // no power flowing either way and no loss to take, so the driving branch — which multiplies — is
    // the safe one to fall into.
    return torque * speed >= 0.0 ? torque * efficiency : torque / efficiency;
}

[[nodiscard]] std::expected<DrivelineTorques, std::string>
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

    const auto gearRatio = setup.gearbox.ratio(state.gear);

    // **The final drive this gear runs through**, which is a property of the gear on a transaxle with
    // two of them. The compliant shaft sits between the gearbox output and the final drive, so which
    // axle is downstream of it changes with the gear — and so does the speed the shaft is wound
    // against. Modelling that pair as one lumped shaft is an approximation and it is recorded at
    // `DrivelineCompliance::stiffness`.
    const auto finalDrive = setup.gearbox.finalFor(state.gear);

    // The differential input's speed, at the gearbox output — the reference the compliant shaft and
    // every number describing it is stated in, because that is where the shaft physically is.
    const auto diffSpeed = axleSpeed * finalDrive;
    const auto compliant = setup.compliance.enabled && drivenCount > 0 && std::abs(finalDrive) >= 1e-9;

    // A rigid driveline has no state of its own: the shaft is whatever the wheels are doing and the
    // twist is nothing. Slaving the two rather than branching around them is what makes
    // `enabled = false` reproduce the arithmetic this function had before the compliance existed,
    // tick for tick — a second code path kept in step by hand would not.
    if (!compliant)
    {
        state.shaftSpeed = diffSpeed;
        state.windUp = 0.0;
    }

    const auto clutchSideSpeed = connected ? state.shaftSpeed * gearRatio : 0.0;

    // What the shaft is carrying as this tick opens. The clutch is solved against it rather than
    // against the road, which is what removes the one-tick lag the rigid path has to live with: the
    // driven side's external torque is now a spring this function owns both ends of.
    const auto shaftReaction = compliant ? std::max(setup.compliance.stiffness, 0.0) * state.windUp +
                                               std::max(setup.compliance.damping, 0.0) * (state.shaftSpeed - diffSpeed)
                                         : 0.0;

    // How hard the spring resists the shaft *changing* speed over this tick, as opposed to what it is
    // already pulling with. The coupling needs it because it is solving for a driven side this
    // function will then integrate against exactly this coefficient — the reaction above is the
    // explicit half of the same spring and the two are not interchangeable.
    const auto shaftResisting =
        compliant ? std::max(setup.compliance.stiffness, 0.0) * deltaTime + std::max(setup.compliance.damping, 0.0)
                  : 0.0;

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
    // Against the *demanded* pedal rather than the driver's, so the idle path and a rev match both
    // put the injectors back the way a foot would.
    state.overrunCut = running && advanceOverrunCut(setup.engine, state.overrunCut, state.engineSpeed, demand);

    const auto flywheel =
        running ? engineTorque(setup.engine, state.engineSpeed, demand, state.fuelCut || state.overrunCut) : 0.0;

    result.engine = flywheel;
    result.fuelCut = state.fuelCut;
    result.overrunCut = state.overrunCut;

    // The pedal, whoever is on it, and the automation is a layer over exactly this one number.
    const auto automatic = autoClutchPedal(setup.autoClutch, setup.engine.idleSpeed, clutchSideSpeed, state.engineSpeed,
                                           input.throttle, connected);

    // Creep is a **floor on the clamp**, and that is the whole of how it joins the rest of the
    // automation: whichever rule wants the clutch closed further gets it. Written that way there is no
    // handover to get wrong and no mode to be in. Away from a standstill it never binds — by the time
    // the car is doing four kilometres an hour `autoClutchPedal`'s own catching-up term is already
    // asking for more clamp than any creep pressure — so a rule that would be wrong at speed is not
    // switched off at speed, it is simply outvoted.
    //
    // It also cannot fight the two things that must win. The driver's own foot beats it past the
    // pedal's free play, inside `advanceClutchPedal`, exactly as it beats the rest of the automation;
    // and the anti-stall is a floor applied *over* the result, so a creep command on an engine being
    // dragged toward its floor opens the clutch rather than closing it. Creep does not defeat
    // anti-stall because it is not in a position to.
    state.creepCommand =
        advanceCreep(setup.autoClutch, state.creepCommand, input.brake, input.throttle, connected, running, deltaTime);
    result.creepCommand = state.creepCommand;

    // **Launch control owns the pedal outright while it is armed**, rather than voting with the others
    // — which is what makes it a mode and the rest of the automation a set of rules. Creep and the
    // catching-up term are both about a car that is trying to move; a launch is a car being
    // deliberately held, and `min`-ing a regulator against them would let whichever wanted the clutch
    // closer win at exactly the moment the regulator is trying to hold it open enough to keep the
    // engine alive.
    //
    // The two things that still beat it are the two that must: the driver's own foot past the pedal's
    // free play, inside `advanceClutchPedal`, and the anti-stall floor applied over the result.
    state.launchArmed = launchControlArmed(setup.autoClutch.launch, state.launchArmed, input.brake, input.throttle,
                                           axleSpeed, connected, running);

    const auto commanded =
        state.launchArmed ? launchControlPedal(setup.coupling, setup.autoClutch.launch, flywheel, state.engineSpeed)
                          : std::min(automatic, creepPedal(setup.coupling, state.creepCommand));

    // **A dead engine gets a disengaged clutch, at any speed, and this is not the anti-stall.** The
    // anti-stall prevents a stall and lives only under the creep band; this is what to do once one has
    // already happened, and the two were tangled together until that band went in.
    //
    // It goes in as the *floor* rather than as the automation's request for `advanceClutchPedal`'s own
    // reason: the floor is past the rate limit, and the rate limit models a foot. A dual clutch
    // dropping pressure on a dead engine is not a foot, and the difference is a whole quarter second
    // of a pedal walking open at four per second — long enough for a plate that happened to be closed
    // to put its entire 480 N·m through the driveline first.
    //
    // What it prevents is sharp. `settleEngineSpeed` pins a stalled engine at exactly zero, so a
    // closed clutch hands the compliant shaft an immovable wall to wind against: measured at 10 m/s in
    // first, the driven wheels ring **31 → 11 → 18 rad/s** with 2732 N·m peaks and 25 torque sign
    // reversals in 720 ticks — a car shaking itself apart rather than coasting to a halt. Bump
    // starting, where the wheels *motor* a dead engine instead of finding it welded, is what would
    // make closing it honest, and nothing in this model has it.
    const auto floor =
        running ? antiStallPedal(setup.autoClutch, setup.engine.idleSpeed, clutchSideSpeed, state.engineSpeed) : 1.0;

    state.clutchPedal =
        advanceClutchPedal(setup.autoClutch, state.clutchPedal, input.clutch, commanded, floor, deltaTime);

    result.slipEnergy = state.slipEnergy;

    // Neutral, a gear change, or a car with nothing driven: the engine is on its own, spinning
    // against its own friction, and the coupling is holding no two things together to have a mode
    // about. The shaft under it is still turning with the wheels either way, which is why this is no
    // longer where the function ends.
    auto coupling = DriveCouplingSolution{};

    if (connected)
    {
        // Rigid, this is the wheels' own inertia and nothing else, which is the right one over a
        // tick: the car's mass reaches the wheel through the tire, and the tire builds its force
        // over a relaxation *length* rather than instantly. Folding the car in here instead — 122
        // kg.m^2 against the wheels' 2.4 — was tried and is unstable, because the vehicle tick then
        // integrates the wheel against 1.2 while the coupling sized its torque against a hundred
        // times that: measured, it turned a launch's one lock transition into forty of them banging
        // between plus and minus the whole capacity.
        //
        // Compliant, the driven side is no longer the wheels at all: it is the shaft, whose inertia
        // is its own and whose external torque is the spring. That is the honest pair, and it is
        // also what makes the road's one-tick lag stop mattering here.
        const auto referredInertia = result.referredInertia;
        const auto drivenInertia = compliant
                                       ? setup.gearbox.inputInertia + setup.compliance.inertia / (gearRatio * gearRatio)
                                       : referredInertia;
        const auto drivenTorque = compliant ? -shaftReaction / gearRatio : axleRoadTorque / reduction;

        const auto sides = CouplingSides{.drivingSpeed = state.engineSpeed,
                                         .drivenSpeed = clutchSideSpeed,
                                         .drivingInertia = engineInertia,
                                         .drivenInertia = std::max(drivenInertia, 1e-9),
                                         .drivingTorque = flywheel,
                                         .drivenTorque = drivenTorque,
                                         .capacity = 0.0,
                                         .drivenResisting = compliant ? shaftResisting / (gearRatio * gearRatio) : 0.0};

        // The gear in mesh rather than the one asked for: a lockup clutch that read the demand would
        // let go a shift early and take hold one late.
        const auto command = DriveCouplingCommand{.clutchPedal = state.clutchPedal, .gear = state.gear};

        const auto solved = stepDriveCoupling(setup.coupling, state.coupling, sides, command, deltaTime);
        if (!solved)
        {
            return std::unexpected(solved.error());
        }

        coupling = solved.value();

        // The engine takes what the coupling did not. A stalled one takes nothing: `settleEngineSpeed`
        // holds it at rest, which is what its own compression does, and the torque still crosses to
        // the driveline — that is why a stalled car in gear drags itself to a stop rather than
        // coasting.
        state.engineSpeed += ((flywheel - coupling.drivingTorque) / engineInertia) * deltaTime;
        settleEngineSpeed(setup.engine, state);

        state.slipEnergy += coupling.slipPower * deltaTime;
    }
    else
    {
        idleDriveCoupling(setup.coupling, state.coupling, deltaTime);
        state.engineSpeed += (flywheel / engineInertia) * deltaTime;
        settleEngineSpeed(setup.engine, state);

        if (!compliant && drivenCount == 0)
        {
            return result;
        }
    }

    // What the box put into the shaft, at the gearbox output, and an **exact zero** through every
    // phase of a gear change and in neutral: an open gearbox is not a small torque. What the wheels
    // see through that window is a separate question once the shaft is compliant — a wound shaft
    // gives back what it stored, and that release is the shunt this element exists to produce.
    const auto gearboxTorque = connected ? coupling.drivenTorque * gearRatio : 0.0;

    // The shaft, *solved* rather than stepped. Its spring and its damper are both contributed as
    // coefficients on the left of `(J/dt + C)w' = (J/dt)w + T` — which is `integrate`'s rule for a
    // damper, extended to a spring by writing the twist implicitly as well — so it is stable at any
    // stiffness rather than at stiffnesses under `2J/dt`. A driveline is the stiffest element in
    // this model after the tyre's vertical rate, and stepping it explicitly is what substepping the
    // driveline would have been for.
    const auto shaftTorque = [&]
    {
        if (!compliant)
        {
            return gearboxTorque;
        }

        // The shaft and the gearbox input are one body while the box is closed, so this solve and the
        // coupling's must be told the same thing about it or they are moving two different cars. The
        // coupling sees it at the *input*, this sees it at the output, and the two differ by the
        // ratio squared — which is not a detail here: through first gear the input's own inertia is
        // twelve times the shaft's once referred, so leaving it out of this half makes the shaft the
        // light body all over again. Open, the box has disconnected them and the shaft is alone.
        const auto coupled = connected ? setup.gearbox.inputInertia * gearRatio * gearRatio : 0.0;
        const auto inertia = std::max(setup.compliance.inertia + coupled, 1e-9);
        const auto stiffness = std::max(setup.compliance.stiffness, 0.0);
        const auto damping = std::max(setup.compliance.damping, 0.0);
        const auto twist = std::max(setup.compliance.maximumTwist, 0.0);

        const auto mass = inertia / std::max(deltaTime, 1e-9);
        const auto resisting = stiffness * deltaTime + damping;

        state.shaftSpeed =
            (mass * state.shaftSpeed + gearboxTorque - stiffness * state.windUp + resisting * diffSpeed) /
            (mass + resisting);
        state.windUp = std::clamp(state.windUp + (state.shaftSpeed - diffSpeed) * deltaTime, -twist, twist);

        return stiffness * state.windUp + damping * (state.shaftSpeed - diffSpeed);
    }();

    result.gearbox = gearboxTorque;
    result.shaftTorque = shaftTorque;
    result.windUp = state.windUp;

    // Through the final drive to the differential, and out to the wheels it decides between. This is
    // the *delivered* torque and not the reaction: they part company the moment the slot holds
    // anything with a member grounded to its own housing.
    // **The losses go here and nowhere else** — between the gearbox output and the differential, which
    // is where the final drive, its bearings and the halfshaft joints physically are. See
    // `DrivelineLosses` for why this is not a multiplier on the engine's curve, and for the sign rule
    // that makes a lossy driveline coast down *harder* rather than more gently.
    const auto axleTorque = throughDrivelineLosses(setup.losses, shaftTorque, state.shaftSpeed) * finalDrive;

    // One statement of what a differential is handed, rather than the same seven fields written out
    // three times with the indices changed.
    const auto axleSides = [&](const std::size_t left, const double torque)
    {
        return DifferentialSides{.leftSpeed = wheelSpeeds[left],
                                 .rightSpeed = wheelSpeeds[left + 1],
                                 .leftInertia = wheelInertias[left],
                                 .rightInertia = wheelInertias[left + 1],
                                 .leftTorque = roadTorques[left],
                                 .rightTorque = roadTorques[left + 1],
                                 .input = torque};
    };

    if (driven == DrivenAxle::All)
    {
        // Split evenly front to rear before each differential, which is a centre spool. A centre
        // differential is the same interface again and is somebody else's milestone. The two
        // differentials are the same setup asked twice and each keeps its own state.
        const auto front = setup.differential.split(state.differentials[0], axleSides(0, 0.5 * axleTorque), deltaTime);
        const auto rear = setup.differential.split(state.differentials[1], axleSides(2, 0.5 * axleTorque), deltaTime);

        result.wheel = {front.left, front.right, rear.left, rear.right};
    }
    else if (driven == DrivenAxle::Front)
    {
        const auto split = setup.differential.split(state.differentials[0], axleSides(0, axleTorque), deltaTime);
        result.wheel = {split.left, split.right, 0.0, 0.0};
    }
    else
    {
        const auto split = setup.differential.split(state.differentials[1], axleSides(2, axleTorque), deltaTime);
        result.wheel = {0.0, 0.0, split.left, split.right};
    }

    result.clutch = coupling.drivenTorque;
    result.clutchReaction = coupling.drivingTorque;
    result.clutchSlip = coupling.slipSpeed;
    result.clutchLocked = coupling.locked;
    result.slipEnergy = state.slipEnergy;

    return result;
}

void fillDrivelineTelemetry(TelemetryFrame& frame, const DrivelineState& state, const DrivelineTorques& torques)
{
    frame.engineSpeed = state.engineSpeed;
    // Where the pedal ended up, which on a car nobody is declutching is the auto-clutch's doing.
    // Off the state rather than off the input packet: the input carries what the driver asked for
    // and the state carries what the clutch is at, and away from a standing start or a shift they
    // are the same number — which is precisely when the channel is uninteresting.
    frame.clutch = state.clutchPedal;
    frame.engineTorque = torques.engine;
    frame.clutchTorque = torques.clutch;
    frame.clutchSlip = torques.clutchSlip;
    frame.clutchSlipEnergy = torques.slipEnergy;
    // The gear in mesh, over the demand the vehicle tick copied off the input packet. They are the
    // same number except during a shift, which is the one time anybody reads the channel.
    frame.gear = torques.gear;
    frame.shiftPhase = static_cast<std::uint32_t>(torques.shiftPhase);
}

[[nodiscard]] std::expected<VehicleInput, std::string>
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

[[nodiscard]] DrivelineSetup placeholderDriveline()
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

[[nodiscard]] DrivelineSetup placeholderAutomatic()
{
    auto setup = placeholderDriveline();

    setup.coupling.kind = DriveCouplingKind::TorqueConverter;

    setup.gearbox.ratios = {4.15, 2.37, 1.56, 1.16, 0.86, 0.69};
    setup.gearbox.finalDrive = 3.20;

    // The turbine and the fluid it drags round with it, which is a far heavier thing than a manual's
    // driven plate. It is also the inertia the stall test is really measuring against: a turbine
    // held at rest is only held if there is something there to hold.
    setup.gearbox.inputInertia = 0.09;

    // The same halfshafts, restated for this car's final drive: a rate at the gearbox output is
    // `k_wheel / finalDrive^2`, so dropping 4.37 to 3.20 makes the identical shaft a stiffer number
    // here. Inheriting the manual's would have been a physically softer car by half, arrived at by
    // saying nothing.
    setup.compliance.stiffness = 2482.0;
    setup.compliance.damping = 13.6;

    return setup;
}

} // namespace raceengine
