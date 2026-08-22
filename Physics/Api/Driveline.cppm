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
export [[nodiscard]] double throttleAirFlow(const EngineModel& engine, const double speed, const double throttle);

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
                                         const bool fuelCut);

// The same engine with both fuel decisions answered from this instant alone, which is the right
// question for anything sweeping the curve and the wrong one for anything running it: see
// `advanceRevLimiter` and `advanceOverrunCut` for why each of them has memory.
export [[nodiscard]] double engineTorque(const EngineModel& engine, const double speed, const double throttle);

// Where the fuel cut goes next, given where it is. Two thresholds, and deliberately *not*
// `stepCoupling`'s two-thresholds-and-a-dwell: that machine answers "hold or slide" for a pair of
// inertias and hands back a torque, and a limiter has no second side, no capacity and no torque to
// clamp. Reaching it through `CouplingSides` would mean inventing all three and then reading the
// answer out of a bool riding on a torque solver — an interface that reads as reuse and behaves as a
// comment, which is `peakSlipScale`'s failure written again. What is shared is the *idea*, and the
// dwell is not even part of it here: the band is crossed at a rate the engine's own inertia sets, so
// there is nothing left for a dwell to stop.
export [[nodiscard]] bool advanceRevLimiter(const EngineModel& engine, const bool fuelCut, const double speed);

// Where the overrun cut goes next. The same two-thresholds-and-memory shape the limiter has, armed
// only while nobody is asking for torque — and `throttle` is the *demanded* pedal rather than the
// driver's, so a rev match's blip puts the injectors back exactly as a foot would.
export [[nodiscard]] bool advanceOverrunCut(const EngineModel& engine, const bool cut, const double speed,
                                            const double throttle);

// How much bypass the governor is asking for, and the integral it is asking through. Exported and
// taking its integral by reference because what holds an idle is worth pinning on its own, without a
// driveline or a car around it.
export [[nodiscard]] double idleBypass(const EngineModel& engine, const double speed, double& integral,
                                       const double deltaTime);

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

export [[nodiscard]] Differential openDifferential();

export [[nodiscard]] Differential spool();

export [[nodiscard]] Differential clutchPackLsd(const double preload, const double powerRamp, const double coastRamp);

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
export [[nodiscard]] double rampLockFraction(const RampGeometry& geometry, const double rampAngle);

// The torque bias ratio a lock fraction produces: what the loaded wheel carries over what the
// unloaded one does. The number a differential is advertised by, and the one this model can be
// checked against published hardware with.
export [[nodiscard]] double torqueBiasRatio(const double lockFraction);

// A pack stated as hardware. Angles in radians, as everything here is.
export [[nodiscard]] Differential rampLsd(const double preload, const double powerAngle, const double coastAngle,
                                          const RampGeometry& geometry);

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
                                             const double targetSpeed, const double engineSpeed);

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

    // The creep torque the transmission is currently commanding, N.m at the clutch. State rather than
    // a function of this instant for the same reason the pedal above is: what makes creep a controller
    // action rather than a switch is that the pressure is *ramped*, and a ramp has to remember where
    // it had got to.
    double creepCommand = 0.0;

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
    // What the creep rule is commanding, N.m at the clutch. Reported rather than left in the state
    // because it is the one channel that says whether a crawl is the controller's doing or the car
    // rolling down a hill, and those two look identical in speed.
    double creepCommand = 0.0;

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
export void startEngine(const DrivelineSetup& setup, DrivelineState& state);

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
export void placeDriveline(const DrivelineSetup& setup, DrivelineState& state, const double axleSpeed);

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
              const VehicleInput& input, const double deltaTime);

// The driveline's own telemetry channels, filled by whoever stepped it. `:Vehicle` fills the rest of
// the frame and cannot fill these — it does not import this partition and must not — so a caller
// stepping both is what joins them, and this is that caller's tool rather than five assignments
// restated at every site.
export void fillDrivelineTelemetry(TelemetryFrame& frame, const DrivelineState& state, const DrivelineTorques& torques);

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
                    const DrivelineState& driveline, const DriverIntent& intent, const double deltaTime);

// The placeholder car's driveline: a small turbocharged four driving the front wheels through a
// six-speed and an open differential. Every number a placeholder, and the torque curve shaped like a
// modern turbo's — flat and early — rather than like a naturally aspirated one.
export [[nodiscard]] DrivelineSetup placeholderDriveline();

// The same car with a torque converter in the slot instead of a plate. Its ratios are an automatic's
// rather than the manual's — a wider first, a taller top and a shorter final drive — because a
// converter already multiplies at the bottom and a gearbox behind one is geared for that. Nothing
// here touches the stall speed, which falls out of the converter's own curves and out of the engine's
// torque at the speed they cross.
export [[nodiscard]] DrivelineSetup placeholderAutomatic();

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
