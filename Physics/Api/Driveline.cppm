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

// The plate and the air path past it, which is the whole of why a part-load engine is not a
// full-throttle one scaled down.
//
// **Representative rather than measured, and loudly so.** Assetto Corsa's `engine.ini` states a
// wide-open power curve, a coast reference and a turbo; it says nothing whatever about what the
// engine makes at a third of the pedal, and neither does any published figure for this car. AC's own
// model multiplies the wide-open curve by the pedal, which is what this file did until now. What is
// here instead is the standard orifice-and-pump balance — a throttle passing air against a pressure
// ratio, an engine drawing it in proportion to its own speed — solved for the manifold pressure:
// Guzzella and Onder, *Introduction to Modelling and Control of Internal Combustion Engine Systems*
// 2nd ed. ch. 2; Heywood, *Internal Combustion Engine Fundamentals* ch. 7.
export struct ThrottleBody
{
    // How far the butterfly opens. A plate's flow area goes as one minus the cosine of its angle
    // from shut, which is what makes the first third of the pedal worth so much less than the last.
    double fullOpenAngle = 1.396; // 80 degrees

    // What is still open with the plate shut: the machined bypass an idle valve breathes through
    // plus the plate's own leakage, as a fraction of the fully open area. **Sized rather than
    // guessed** — it is the area at which the engine's own idle airflow exactly covers its friction,
    // so a warm engine holds its idle without the governor doing anything. It is also the whole of
    // what an overrun cut has to turn off: air at a shut throttle is what a coasting engine is still
    // burning, and cutting the fuel is what stops it.
    double leakArea = 0.006;

    // Where the plate stops being the restriction. At this speed a fully open throttle holds
    // 1/sqrt(2) of ambient in the manifold, so below it the pedal saturates early and above it the
    // same pedal opening passes steadily less of what the engine wants.
    double chokedSpeed = 900.0;
};

// The overrun cut, and it is two thresholds and no dwell for `advanceRevLimiter`'s reason: the band
// is crossed at a rate the car's own deceleration sets, and a speed cannot dither across it inside a
// tick the way a torque can.
//
// **What it does is the restore, not the cut**, and that is worth stating because the opposite reads
// as obvious. A shut throttle passes only the leak area whatever the injectors are doing, so cutting
// the fuel above the band changes the torque by exactly what that leak was making — real, and small.
// Coming *back* through the band is where it shows: the injectors return, the engine sustains itself
// again, and the braking lets go rather than tapering linearly to nothing at rest.
export struct OverrunCut
{
    bool enabled = true;

    // As multiples of idle speed, so they follow the engine rather than being restated against it.
    double cutFraction = 1.8;
    double restoreFraction = 1.4;

    // Past this the driver is asking for torque and the injectors are on whatever the speed. A rev
    // match counts as asking, which is why the *demanded* pedal is what this is read against.
    double throttleThreshold = 0.02;
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

    ThrottleBody throttle;
    OverrunCut overrun;
};

// The fraction of the air a fully open throttle would pass at this speed, which is what the cylinder
// actually gets. Exactly 1 at full pedal and exactly the leak area's worth at a shut one, at every
// speed — so the measured full-throttle curve is reproduced to the bit and only the middle of the
// pedal is reshaped.
//
// Pure and exported, because a part-load surface is the one thing here that has to be swept and read
// rather than believed.
export [[nodiscard]] double throttleAirFlow(const EngineModel& engine, const double speed, const double throttle)
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
// Part load is **not** the full-throttle curve scaled by the pedal, and that was the whole of what
// this function used to say. What the crankshaft delivers is the work the trapped air did less the
// friction and pumping the engine owes whatever the load — so a third of the air is a good deal less
// than a third of the torque, and high enough up the range it is none at all. Written this way the
// coast reference stops being a term bolted beside the curve and becomes the loss the part-load
// blend is taken from, which is what it physically is.
export [[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle,
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

// The same engine with both fuel decisions answered from this instant alone, which is the right
// question for anything sweeping the curve and the wrong one for anything running it: see
// `advanceRevLimiter` and `advanceOverrunCut` for why each of them has memory.
export [[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle)
{
    const auto overrun = engine.overrun.enabled && throttle <= engine.overrun.throttleThreshold &&
                         speed >= engine.overrun.cutFraction * engine.idleSpeed;

    return engineTorque(engine, speed, throttle, speed >= engine.limiterSpeed || overrun);
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

// Where the overrun cut goes next. The same two-thresholds-and-memory shape the limiter has, armed
// only while nobody is asking for torque — and `throttle` is the *demanded* pedal rather than the
// driver's, so a rev match's blip puts the injectors back exactly as a foot would.
export [[nodiscard]] bool advanceOverrunCut(const EngineModel& engine, const bool cut, const double speed,
                                            const double throttle)
{
    if (!engine.overrun.enabled || throttle > engine.overrun.throttleThreshold)
    {
        return false;
    }

    return cut ? speed > engine.overrun.restoreFraction * engine.idleSpeed
               : speed >= engine.overrun.cutFraction * engine.idleSpeed;
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

    // What turns at the gearbox *input*: the clutch's driven plate or the converter's turbine, plus
    // the input shaft and the gearset riding on it. A manual's is small; an automatic's turbine
    // carries entrained fluid and is several times it, which is why `placeholderAutomatic` states
    // its own.
    //
    // Rigidly coupled this was never needed — the wheels reach the gearbox input through the
    // gearing and outweigh it by two orders of magnitude, so leaving it out cost nothing. A
    // compliant shaft is what makes it load-bearing: the input is no longer tied to the wheels
    // within a tick, so its own inertia is the only thing the coupling has to push against, and
    // without it the coupling solves against the *shaft* alone and finds a body 141 times too light
    // to be pushed on. That reads as a clutch that transmits nothing and a converter that never
    // multiplies.
    double inputInertia = 0.02;

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

    // The box's own ratio, with nothing after it. Named separately from `reduction` because the
    // compliant shaft lives *between* the two: what twists is the gearbox output against the
    // differential input, so the two halves of the chain are needed apart as well as together.
    [[nodiscard]] double ratio(const std::int32_t gear) const
    {
        if (gear == 0 || ratios.empty())
        {
            return 0.0;
        }

        if (gear < 0)
        {
            return -reverseRatio;
        }

        return ratios[std::min(static_cast<std::size_t>(gear - 1), ratios.size() - 1)];
    }

    // Total reduction from the flywheel to the differential input. Zero in neutral, which is what
    // disconnects the chain — not a special case anywhere else, just a ratio of nothing.
    [[nodiscard]] double reduction(const std::int32_t gear) const
    {
        return ratio(gear) * finalDrive;
    }
};

export struct DifferentialTorques
{
    double left = 0.0;
    double right = 0.0;
};

// What the pack did last tick, and the lock/slip machine that decided it. `transfer` and `capacity`
// are precisely the two numbers that machine judges — how much the pack was asked to carry against
// how much it could — so they are recorded where they are known rather than recovered afterwards
// from torques that have already been split.
export struct DifferentialState
{
    double transfer = 0.0;
    double capacity = 0.0;
    CouplingState pack{};
};

// The two wheels as the differential needs to see them, and it is a struct for `CouplingSides`'
// reason: seven loose scalars at a call site are seven chances to pass them in the wrong order.
//
// The inertias and the road torques are here because **a friction differential cannot be solved
// without them**. `stepCoupling` holds a pair together by asking what torque that costs, and that
// question is `(Ir*Tl - Il*Tr)/(Il+Ir)` plus the slip through the reduced inertia — so a pack told
// the wheels have no external torque can only make up the difference with a speed difference, and
// "locks" by slipping for ever. That is the failure the clutch already had and had measured out of
// it; reproducing it one element further down the chain would be the same bug written twice.
export struct DifferentialSides
{
    double leftSpeed = 0.0;
    double rightSpeed = 0.0;
    double leftInertia = 1.0;
    double rightInertia = 1.0;
    // What the road did to each wheel, lagged one tick exactly as the clutch's is and for the same
    // reason: the tyre's answer for this tick does not exist until the vehicle tick has run.
    double leftTorque = 0.0;
    double rightTorque = 0.0;
    // What the gearing delivered to the differential's case.
    double input = 0.0;
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
    // The geometry that produces these lives in `rampLockFraction` and deliberately not in here —
    // see the note there for why.
    double powerRamp = 0.0;
    double coastRamp = 0.0;

    // The pack's own lock/slip machine, and its two departures from the plate's defaults are both
    // required rather than tuning.
    //
    // `lockSlipSpeed` is half a rad/s because that is what a differential is *for*: a corner of any
    // real radius turns the two wheels past each other faster than that, and a pack that considered
    // locking across it would be answering the question a spool answers. And `slipDwell` is zero
    // for `advanceRevLimiter`'s reason — the dwell exists because a *torque* can dither across both
    // thresholds inside one tick, and what this pack's constraint is dominated by is a speed
    // difference, which cannot. Left at the plate's 20 ms it is 20 ms of sliding friction at full
    // capacity every time a wheel is momentarily quicker than its partner, which is a car that
    // darts on the first tick of every launch.
    CouplingSetup pack{.lockSlipSpeed = 0.5, .slipDwell = 0.0};

    // The state is the axle's, not the differential's: a `Differential` is setup and is handed
    // around by const reference, and all-wheel drive asks the *same* one twice. Two axles sharing
    // one state object would have the front's lock stamping on the rear's, which is why the state
    // arrives here rather than living in the struct.
    [[nodiscard]] DifferentialTorques split(DifferentialState& state, const DifferentialSides& sides,
                                            const double deltaTime) const
    {
        const auto ramp = sides.input >= 0.0 ? powerRamp : coastRamp;
        const auto capacity = std::max(preload, 0.0) + std::max(ramp, 0.0) * std::abs(sides.input);

        // The pack is `stepCoupling`'s third consumer and holds the two wheels together exactly as
        // the plate holds the engine to the gearbox: the gears have already split the input evenly,
        // so what is left for the pack to carry is whatever asymmetry the road is imposing on top.
        const auto pair = CouplingSides{.drivingSpeed = sides.leftSpeed,
                                        .drivenSpeed = sides.rightSpeed,
                                        .drivingInertia = sides.leftInertia,
                                        .drivenInertia = sides.rightInertia,
                                        .drivingTorque = 0.5 * sides.input + sides.leftTorque,
                                        .drivenTorque = 0.5 * sides.input + sides.rightTorque,
                                        .capacity = capacity};

        const auto held = stepCoupling(pack, state.pack, pair, deltaTime);

        state.transfer = held.torque;
        state.capacity = capacity;

        return DifferentialTorques{.left = 0.5 * sides.input - held.torque, .right = 0.5 * sides.input + held.torque};
    }
};

export [[nodiscard]] Differential openDifferential()
{
    return Differential{};
}

export [[nodiscard]] Differential spool()
{
    // Locked solid, and both numbers are needed to say so. An enormous capacity is what stops the
    // pack ever reaching its limit; a lock slip speed past anything a wheel can do is what makes it
    // hold across a difference rather than sliding at that capacity, which for 1e6 N.m would put a
    // megawatt of nonsense into the wheels on the first tick of a corner.
    return Differential{.preload = 1e6, .pack = {.lockSlipSpeed = 1e9, .slipDwell = 0.0}};
}

export [[nodiscard]] Differential clutchPackLsd(const double preload, const double powerRamp, const double coastRamp)
{
    return Differential{.preload = preload, .powerRamp = powerRamp, .coastRamp = coastRamp};
}

// The pack a ramped differential actually has, for the cars whose data states the hardware rather
// than the behaviour.
//
// **The geometry lives here and not in `Differential`, and that is the decision this milestone had
// to take rather than inherit.** The brief specifies ramp angles, friction faces and a coefficient;
// the struct above stores two dimensionless torque fractions, and the two are not the same thing
// said twice — the fraction is *what the differential does* and the geometry is one particular way
// of arriving at it. Every source this project has states the fraction directly: AC's `POWER` and
// `COAST` are exactly it, a race engineer asks for "a 25% diff", and a torque bias ratio is a
// one-line function of it. Putting the geometry in the type would oblige the Golf — whose file
// states 0.25 and no hardware at all — to carry an invented ramp angle, a made-up plate count and a
// friction coefficient nobody measured, chosen so that their product came back to the number
// already known. That is the fictitious damper again: a plausible-looking geometry describing a
// pack that does not exist, and a second statement of a number with nothing making the two agree.
// So the conversion is a factory, `split` and its call sites never learn a ramp angle exists, and
// the two ways of stating a pack meet at one function that can be checked against published bias
// ratios without a car anywhere near it.
// Ramp angles are the one thing in this model quoted in degrees everywhere they are published, so
// the conversion is stated once here rather than as a magic number at every data site. The angle
// itself stays radians, as everything here is.
export inline constexpr auto degreesToRadians = 0.017453292519943295;

export struct RampGeometry
{
    // Where the cross pin rides its ramp, and where the plates rub. Both from the axis, in metres.
    double rampRadius = 0.0225;
    double plateRadius = 0.042;

    // Friction interfaces in **one** side's stack. The two stacks are symmetric and each carries the
    // same torque, but they carry it to opposite wheels — the sum is the input, and the *bias* is
    // one stack's worth, the other being what reacts it.
    std::uint32_t faces = 4;

    // Wet plates in gear oil, which is why it is a tenth rather than the third a dry clutch runs.
    double frictionCoefficient = 0.085;
};

// The fraction of the torque through the case that a ramp of this angle turns into locking torque.
//
// The angle is measured from the plane perpendicular to the differential's axis, which is the
// convention the two numbers on a Salisbury's spec sheet are quoted in: a face at 90 degrees is flat
// against the rotation, produces no axial thrust at all and locks with nothing, and every smaller
// angle locks harder. Tangential force at the pin is the input torque over the ramp radius; the
// wedge turns it into thrust through the cotangent; the pin is driven by ramps in *both* pressure
// rings so each takes half of it; and the thrust on one ring clamps that ring's stack.
export [[nodiscard]] double rampLockFraction(const RampGeometry& geometry, const double rampAngle)
{
    const auto radius = std::max(geometry.rampRadius, 1e-9);
    const auto tangent = std::tan(std::clamp(rampAngle, 1e-6, 1.5707963267948966));

    return 0.5 * (geometry.plateRadius / radius) * static_cast<double>(geometry.faces) *
           std::max(geometry.frictionCoefficient, 0.0) / tangent;
}

// The torque bias ratio a lock fraction produces: what the loaded wheel carries over what the
// unloaded one does. The number a differential is advertised by, and the one this model can be
// checked against published hardware with.
export [[nodiscard]] double torqueBiasRatio(const double lockFraction)
{
    const auto fraction = std::clamp(lockFraction, 0.0, 0.5 - 1e-9);

    return (0.5 + fraction) / (0.5 - fraction);
}

// A pack stated as hardware. Angles in radians, as everything here is.
export [[nodiscard]] Differential rampLsd(const double preload, const double powerAngle, const double coastAngle,
                                          const RampGeometry& geometry)
{
    return clutchPackLsd(preload, rampLockFraction(geometry, powerAngle), rampLockFraction(geometry, coastAngle));
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

// The shaft between the gearbox output and the differential input, and what it does when the
// throttle is dropped on it.
//
// One lumped torsional element standing for the whole shaft train below the box, stated at the
// **gearbox output** because that is where the element is. On a transverse transaxle the number is
// dominated by the two halfshafts referred through the final drive rather than by the pinion's own
// torsion, and referring a rate across gearing divides it by the ratio squared — the output turns
// `finalDrive` times faster than the wheel, so it twists that much further for the same halfshaft
// angle. Two halfshafts of a small hatchback are about 25 kN.m/rad at the wheel, which twists eleven
// degrees under this car's 4879 N.m of peak axle torque; through 4.37 that is **1331 N.m/rad here**,
// and the same shaft appears as a different number on a car with a different final drive, which is
// why `placeholderAutomatic` restates it rather than inheriting it. The damping is a ratio of 0.08
// on the driveline mode, which is why shunt rings a few times rather than either buzzing or
// vanishing.
//
// **It was 500, and that is worth recording because of how it failed.** Too soft is not merely a
// softer car: first gear then asks for more twist than `maximumTwist` allows — 6.1 rad on the
// automatic against a 3.0 rad guard — so the guard stops being a guard and becomes a silent 1500 N.m
// torque limiter in ordinary driving. What that looks like is a converter that never reaches stall
// (3312 rpm against a road automatic's 1800-2500), a car that will not drive away, and a torque
// interrupt that never appears, none of which reads as a stiffness.
//
// **The numbers are representative and the shape is not.** No published figure for this car states a
// halfshaft rate, and AC does not model driveline compliance at all — so these are sized from the
// frequency the element has to produce (a FWD hatchback shuffles at nine to eleven hertz) rather
// than measured. What is *not* representative is that it is one degree of freedom solved implicitly:
// that is the model, and the rate is the data.
export struct DrivelineCompliance
{
    bool enabled = true;

    // N.m per radian and N.m.s per radian, both at the gearbox output.
    double stiffness = 1331.0;
    double damping = 7.3;

    // What turns between the box and the final drive: the output shaft, the pinion and the
    // differential case. drivetrain.ini [GEARBOX] INERTIA for this car, whose own reference the file
    // does not state.
    double inertia = 0.017;

    // A shaft wound past this is not a driveline, it is a solve that has come apart. The clamp is
    // what keeps that diagnosable rather than a NaN spreading through the wheels.
    double maximumTwist = 3.0;
};

export struct DrivelineSetup
{
    EngineModel engine;
    Gearbox gearbox;
    // The slot between the engine's inertia and the gearbox input, and this file names neither of the
    // two things that can be in it after this line.
    DriveCoupling coupling;
    AutoClutch autoClutch;
    ShiftAssist shiftAssist;
    DrivelineCompliance compliance;
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
    // And the overrun's, which is a separate decision made against a separate band: one is the
    // engine being held down, the other the car driving it.
    bool overrunCut = false;

    // A default-constructed driveline is a car with the key out: nothing has started this engine, so
    // it is not turning. `startEngine` is what changes that, and it is deliberately not an input.
    EngineState engine = EngineState::Stalled;

    // The coupling slot's own state, whichever kind is fitted.
    DriveCouplingState coupling{};
    // The pedal actually being held, which is the driver's when the driver is on it and the
    // automation's when nobody is. Kept rather than recomputed so the handover between the two is
    // continuous — see `advanceClutchPedal`.
    double clutchPedal = 0.0;

    // The compliant shaft's two numbers, both at the gearbox output: what it is turning at and how
    // far it is wound against the differential. Slaved to the wheels and held at zero when the
    // compliance is off, which is what makes a rigid driveline the same arithmetic rather than a
    // second path.
    double shaftSpeed = 0.0;
    double windUp = 0.0;

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
    // What the gearbox delivered to the shaft, at its own output, and what the shaft delivered to the
    // differential. They are the same number only where the driveline is rigid or settled; the
    // difference between them *is* the wind-up, which is why both are reported rather than one.
    double gearbox = 0.0;
    double shaftTorque = 0.0;
    // How far the shaft is wound, in radians at the gearbox output. The channel shunt is read in.
    double windUp = 0.0;
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
    bool overrunCut = false;
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

// Placing a car at a road speed means placing all of it, and the compliant shaft is the part that is
// easy to forget. A `DrivelineState` written with the wheels turning and the shaft still at rest is
// not a slow car, it is an impossible one: the spring reads the difference as a twist nobody put
// there and hands the wheels a torque on the first tick that nothing in the car produced. Measured on
// the placeholder with the wheels at 40 rad/s, that fiction is **-2746 N.m** — a fault that presents
// as a driveline refusing to drive and is entirely an initial condition.
//
// A rigid driveline has no such state, which is why nothing needed this before and why leaving it out
// went unnoticed: `stepDriveline` slaves the shaft to the wheels every tick when the compliance is
// off, so the same fixture was consistent by construction.
//
// Separate from `startEngine` deliberately. Which of the two a caller wants is a real question — a
// car can be placed rolling with its engine dead, and is, every time one is pushed.
export void placeDriveline(const DrivelineSetup& setup, DrivelineState& state, const double axleSpeed)
{
    state.shaftSpeed = axleSpeed * setup.gearbox.finalDrive;
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

    const auto gearRatio = setup.gearbox.ratio(state.gear);
    const auto finalDrive = setup.gearbox.finalDrive;

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
    state.clutchPedal =
        advanceClutchPedal(setup.autoClutch, state.clutchPedal, input.clutch, automatic,
                           antiStallPedal(setup.autoClutch, setup.engine.idleSpeed, state.engineSpeed), deltaTime);

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
    const auto axleTorque = shaftTorque * finalDrive;

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
