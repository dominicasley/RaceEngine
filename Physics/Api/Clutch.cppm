module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>

#include <glm/glm.hpp>

export module raceengine.physics:Clutch;

import :Coupling;
import :Vehicle;

namespace raceengine
{

// The element between the engine's inertia and the gearbox input. It is an *element* rather than a
// clutch because a torque converter sits in the same slot, and nothing downstream may know which is
// fitted: the driveline asks this what the engine felt and what the gearbox got, which is the same
// sentence either way.
//
// Two kinds behind one function and one output struct, which is `SuspensionKind`'s shape and
// deliberately not an abstract base's. What a driveline wants from this slot does not vary by type;
// the pedal a friction clutch reads and the gear a converter's lockup reads are both in the
// arguments, and each kind discards the other's. Value semantics throughout, so `DrivelineSetup`
// stays assignable.
export enum class DriveCouplingKind : std::uint32_t { FrictionClutch, TorqueConverter };

// Pedal travel against how much of the clamp load reaches the plate, and it is data rather than code
// because a real clutch is nowhere near linear in it. The top of the travel is free play in the
// release mechanism, almost the whole of the clamp goes in over a narrow band in the middle, and the
// bottom is a plate already fully released. That band is what a driver launches on, and its width is
// the difference between a pedal and a switch.
export [[nodiscard]] Curve roadClutchEngagement();

export struct FrictionClutch
{
    // `clutchCapacity`'s inverse is defined against this struct below, so anything that changes how
    // pedal becomes clamp changes both directions at once and there is no second statement of the
    // curve to fall out of step with the first.

    // Placeholder, and a single 240 mm organic plate's numbers: 8 kN of diaphragm clamp, 0.30 dry
    // friction, a 0.100 m mean effective radius and the two faces one plate has come to 480 N.m,
    // which is 1.37 times what this engine makes. Sized under that it slips in top gear on its own.
    double clampForce = 8000.0;
    double frictionCoefficient = 0.30;
    double effectiveRadius = 0.100;
    std::uint32_t faces = 2;

    Curve engagement = roadClutchEngagement();

    CouplingSetup coupling;
};

// Torque capacity at a given pedal. Clamp force through the friction surfaces, scaled by where the
// pedal has the clamp — nothing here integrates or remembers anything.
export [[nodiscard]] double clutchCapacity(const FrictionClutch& clutch, const double pedal);

// The pedal that commands a given capacity out of this clutch: `clutchCapacity` read backwards.
//
// This exists because a transmission controller commands a *torque* and the actuator this model has
// is a pedal. A real TCU meters clutch pressure, which is capacity, and the pedal is the only way to
// ask this clutch for one — so the alternative to inverting the curve is a second control law written
// in pedal units, which is the same number stated twice with nothing keeping the two together.
//
// It returns the **least engaged** pedal that delivers the capacity, which matters at both ends
// because both ends of the curve are flat. Commanding nothing must give a fully released pedal and
// not the first pedal at which the clamp happens to reach zero: the difference is invisible in torque
// and very visible in the pedal channel, which is what a fixture asserts its preconditions on.
export [[nodiscard]] double clutchPedalForCapacity(const FrictionClutch& clutch, const double capacity);

// The other kind, and it is not a variant of the friction clutch. A converter is a fluid coupling
// with a third member, and all it shares with a plate is where it sits in the chain: the impeller
// throws oil at the turbine, the stator turns what comes back and reacts the difference into the
// housing, and that reaction is the extra torque. Nothing in here has a capacity, a pedal or a
// sliding surface.
//
// **Both curves below are representative rather than measured**, which is a distinction nothing else
// in this model has to make and is therefore worth making loudly. Every other number here comes out
// of a real car's own data, and neither car on this machine is a torque-converter automatic — the
// Golf is a dual clutch and the M3 is a manual — so there is nothing to extract these from. The
// shape and the sizing are the published single-stage three-element characteristic for a mid-size
// passenger car: Wong, *Theory of Ground Vehicles* 4th ed. ch. 3; Kotwicki, "Dynamic Models for
// Torque Converter Equipped Vehicles", SAE 820393; Heisler, *Advanced Vehicle Technology* 2nd ed.
// ch. 3.

// Speed ratio against torque ratio. Just over two at stall, falling to one at the coupling point
// near 0.88 and staying there — past that the stator freewheels and the converter is a plain fluid
// coupling. Efficiency is the product of the two, and this shape reaches 0.90 by a speed ratio of
// 0.8 — the published figure. Past that it keeps creeping toward one rather than turning over as a
// measured curve does: slip is the only loss modelled here, and the churning and pumping losses that
// bend a real curve back down would be a third table. There is next to no torque left up there to be
// inefficient with, which is why the omission costs nothing and why a lockup clutch exists anyway.
export [[nodiscard]] Curve converterTorqueRatio();

// Speed ratio against the capacity coefficient C = T_impeller / omega_impeller^2, in N.m per
// (rad/s)^2.
//
// The literature quotes the *capacity factor* K = omega_impeller / sqrt(T_impeller) instead, and
// this is that table squared and inverted: 140, 143, 150, 156, 166, 182, 212, 290 and infinity
// rpm/sqrt(lb.ft) at the speed ratios below, which is an ordinary passenger-car converter. Held as C
// rather than as K because K is *infinite* where the two wheels come into step. Interpolating a
// piecewise-linear K toward a large finite stand-in instead collapses the torque to nothing well
// before the speed ratio gets there — which is precisely the range a converter without a lockup
// spends its life cruising in. C is zero there, which is both finite and the truth: no relative
// motion, no flow, no torque.
export [[nodiscard]] Curve converterCapacity();

// The plate that bridges impeller and turbine once there is nothing left to multiply. It is in
// *parallel* with the fluid rather than downstream of it, which is what a real lockup clutch is
// bolted across, and it is `stepCoupling`'s third consumer for the reason that partition names no
// device: a lockup plate is a friction clutch with a different capacity in front of it.
export struct ConverterLockup
{
    bool enabled = true;

    // What the plate holds fully applied. 1.3 times what this engine makes, sized as the friction
    // clutch's is and for the same reason.
    double capacity = 450.0;

    // Turbine speed, rad/s. Two of them, and the gap is not a refinement: a plate answering one
    // threshold chatters every time the speed sits on it. Between them the apply level is simply
    // held, which makes that level its own latch and is why nothing else has to carry one.
    double releaseSpeed = 140.0;
    double engageSpeed = 180.0;
    // A lockup in first is a car with no launch device at all.
    std::int32_t lowestGear = 3;

    // Fractions of full apply per second. A real plate is ramped by the transmission controller over
    // about half a second, and it is that ramp rather than any threshold that stops the engagement
    // being a bang.
    double applyRate = 2.5;
    double releaseRate = 5.0;

    // It will lock across far more slip than the gearbox plate will, and that is not a relaxation of
    // the same rule. A friction clutch is closed by a driver who has already matched the two speeds,
    // so a rad/s is all it should ever have to slam shut on; a lockup plate is closed against
    // whatever slip the fluid has left it, which at a converter's coupling point is tens of rpm and
    // never less. Held at a rad/s the plate simply never engages, which reads as a lockup that was
    // never wired up.
    CouplingSetup coupling{.lockSlipSpeed = 12.0};
};

export struct TorqueConverter
{
    Curve torqueRatio = converterTorqueRatio();
    Curve capacity = converterCapacity();

    // Reverse flow, where the turbine is the pump. Representative, like the curves: published
    // reverse-region capacity is well under the forward figure, and this is why an automatic gives
    // so little engine braking.
    double overrunCapacityScale = 0.5;

    ConverterLockup lockup;
};

// What the fluid is doing. `impeller` is what the converter takes off the engine and `turbine` what
// it delivers to the gearbox input; they are not the same number, and the difference is what the
// stator reacts into the housing.
export struct ConverterFlow
{
    double impeller = 0.0;
    double turbine = 0.0;
};

// The fluid path alone, with no lockup in it. Pure, so what a converter does can be swept and read
// off without a car around it.
export [[nodiscard]] ConverterFlow converterFlow(const TorqueConverter& converter, const double impellerSpeed,
                                                 const double turbineSpeed);

// Where the lockup's apply pressure goes next, given what the car is doing.
export [[nodiscard]] double advanceLockup(const ConverterLockup& lockup, const double apply, const double turbineSpeed,
                                          const std::int32_t gear, const double deltaTime);

export struct DriveCoupling
{
    DriveCouplingKind kind = DriveCouplingKind::FrictionClutch;

    FrictionClutch clutch;
    TorqueConverter converter;
};

// The slot's own state. `CouplingState` is the lock/slip machine — the plate's for a clutch, the
// lockup's for a converter — and the apply level beside it is the one thing a converter integrates
// that a plate does not.
export struct DriveCouplingState
{
    CouplingState coupling{};
    double lockupApply = 0.0;
};

static_assert(std::is_trivially_copyable_v<DriveCouplingState>, "the driveline's state is saved by copying its bytes");
static_assert(std::is_standard_layout_v<DriveCouplingState>, "and rollback will later");

// What the slot is told each tick. Both fields are read by one kind and deliberately discarded by
// the other, which is exactly why they arrive here rather than in either device's own setup: what
// the driveline knows is that there is an element in the chain, and it may not know which.
export struct DriveCouplingCommand
{
    // 0 released and 1 fully depressed.
    double clutchPedal = 0.0;
    std::int32_t gear = 0;
};

export struct DriveCouplingSolution
{
    // What the slot took off the engine, and what it delivered to the gearbox input. One number does
    // for a friction clutch, whose two faces carry the same torque; a converter has a third member
    // grounded to the housing and the difference between these two is precisely what it reacts. A
    // slot reporting a single torque is a slot no converter can be honest in.
    double drivingTorque = 0.0;
    double drivenTorque = 0.0;

    double slipSpeed = 0.0;

    // Watts, and the caller integrates it. Each kind states its own loss because they are not the
    // same arithmetic: a plate dissipates torque times slip, a converter dissipates the power in
    // less the power out, and the two agree only where the torque ratio is one.
    double slipPower = 0.0;

    bool locked = false;
};

// One tick of whatever is fitted.
export [[nodiscard]] std::expected<DriveCouplingSolution, std::string>
stepDriveCoupling(const DriveCoupling& coupling, DriveCouplingState& state, const CouplingSides& sides,
                  const DriveCouplingCommand& command, const double deltaTime);

// What the slot does on a tick where nothing is connected through it — neutral, a gear change, or a
// car with no driven axle. It is here rather than in the driveline because the two kinds answer
// differently and the driveline may not know which is fitted.
//
// A plate simply forgets which side of its hysteresis it was on: there is nothing to hold. A
// converter's lockup *bleeds* rather than dropping, because the driveline used to clear this whole
// struct and that put a step in the one channel a shift is judged by — a plate that vanished in a
// tick and took half a second to come back, on every gear change, for no reason a transmission
// controller would recognise.
export void idleDriveCoupling(const DriveCoupling& coupling, DriveCouplingState& state, const double deltaTime);

// Launch control, as the car has it: brake and throttle together, the engine held at a target while
// the clutch slips, and the car released when the brake comes off.
//
// **It is a clutch-pressure regulator and not a rev limiter, and that distinction is the whole
// implementation.** Holding a car on its brakes at full throttle, the engine's speed is set by how
// much torque the clutch is taking off it: open the plate and the engine runs away, close it and the
// engine is dragged toward a stall. So the controller commands a *capacity* — which is what a real
// TCU meters as pressure — and the speed falls out. `clutchPedalForCapacity` is the inverse that
// makes that expressible against a pedal, and it exists for exactly this reason.
//
// **Off by default**, like every other assist in this project, because every measurement here is of
// the car underneath its electronics.
//
// A converter car does not get one and must not: a fluid coupling at stall already does this, and
// commanding a launch on top would be the same behaviour modelled twice. `launchControlPedal`
// dispatches on the kind for the same reason `creepPedal` does.
export struct LaunchControl
{
    bool enabled = false;

    // Where the engine is held while the brake is down, rad/s. ~3000 rpm, which is what this car
    // does — high enough to be on boost when the clutch takes up, low enough not to simply light the
    // tyres.
    double targetSpeed = 314.16;

    // How much of each pedal arms it. Both are needed at once, which is what makes it a deliberate
    // action rather than something that fires whenever somebody rests a foot on the brake.
    double armThrottle = 0.85;
    double armBrake = 0.40;

    // And where the launch is let go. Below this the brake is no longer holding the car and the
    // regulator hands the clutch back to the ordinary automation, which closes it on the throttle
    // that is already flat.
    double releaseBrake = 0.15;

    // Above this **wheel** speed there is nothing to launch, rad/s. What stops the regulator
    // re-arming mid-run if a driver brushes the brake with the throttle still down.
    //
    // In rad/s of wheel and not m/s of road because `stepDriveline` does not know the tyre's radius
    // and has no business learning it — the driveline deals in shaft speeds throughout. 6.3 rad/s is
    // about 2 m/s on this car's 0.3186 m tyre, which is walking pace.
    double maximumWheelSpeed = 6.3;

    // Clutch capacity per rad/s of engine speed error, N.m per rad/s.
    //
    // **Derived rather than tuned.** With the car held, the engine's equation is `I·dω/dt = T - C`;
    // commanding `C = T + g·(ω - ω_target)` makes that `I·dω/dt = -g·(ω - ω_target)`, a first-order
    // lag with time constant `I/g`. This engine's inertia is 0.15 kg·m², so 1.5 gives a tenth of a
    // second — fast enough to catch the engine before it runs past the target, slow enough not to
    // chatter the plate against it.
    double gain = 1.5;
};

// A layer over the *pedal*, not around the clutch. It produces a pedal position and nothing else, so
// the friction model underneath is identical whoever is pressing — which is the whole point, because
// the rig has a real clutch pedal and a human slipping it from rest is the best test this model has.
export struct AutoClutch
{
    bool enabled = true;

    // Where the clutch must be fully out and fully in, measured on the driveline side and stated as
    // fractions of idle speed so that they follow the engine rather than being restated against it.
    double releaseFraction = 0.40;
    double grabFraction = 1.20;

    // **Was the speed it held the engine at on a launch, and that rule is gone.** It held the clutch
    // open until the engine had climbed to this — 250 rad/s, near 2400 rpm — and only then fed the
    // clamp in against the revs. That is a launch assist and no dual clutch does it: from the seat it
    // is flooring the throttle, hearing the engine flare, and going nowhere. A DSG *closes* the clutch
    // when you ask for torque and lets the anti-stall protect the engine if it was too much.
    //
    // Kept as a field only because `[.vehicle-data]` enumerates it. Nothing reads it.
    double launchSpeed = 250.0;

    // What the clutch answers the driver with instead: the accelerator travel at which the clamp is
    // commanded fully home. A tenth of the pedal, so any real request for torque closes it and the
    // take-up is the pedal's own rate limit — a quarter of a second from open to shut — rather than a
    // regulator waiting on revs.
    double engageThrottle = 0.10;

    // A foot has a speed limit and so does this. Without it the pedal is a step input into a
    // friction element, which is the one thing a coupling cannot be asked to answer sensibly.
    double pedalRate = 4.0;

    // The free play a real pedal already has at the top of its travel, and the line this uses to
    // decide that the driver is on the pedal.
    double freePlay = 0.05;

    // The anti-stall band, as fractions of idle speed. Above `antiStallBegin` the engine is healthy
    // and this contributes nothing; by `antiStallOpen` the clutch is fully out. It begins *below*
    // idle so the governor's own excursions never brush it, and it is fully open well above the
    // engine's stall floor — 0.55 of an 850 rpm idle is 468 rpm against a stall floor of 382 —
    // because the whole term exists to keep the fight away from that floor, not to referee at it.
    double antiStallBegin = 0.90;
    double antiStallOpen = 0.55;

    // --- creep ---
    //
    // A dual clutch in gear with the brakes released and no throttle crawls forward, and it does so
    // because the controller *commands* it: there is nothing in a pair of dry plates that leaks
    // torque the way a fluid coupling does at stall. That is the whole reason this is a rule here and
    // a consequence of the hardware in `TorqueConverter`.
    bool creep = true;

    // **The torque, not the speed.** A real TCU commands a modest clutch pressure and lets the
    // resulting speed fall out of whatever is loading the car, which is why a DSG creeps slower
    // uphill, faster downhill and stalls against a kerb rather than climbing it. A speed regulator
    // gets all three wrong, and gets them wrong in the direction that feels like an assist.
    //
    // N.m at the clutch. What it buys is grade authority and nothing else: through this car's first
    // gear and final drive it is `creepTorque * 13.94 / 0.3186` newtons at the road, so 30 N.m is
    // 1313 N against a 1348 kg car — a little under a ten percent gradient before it gives up. The
    // speed it settles at on the flat is not set by this at all; see `creepPedal`.
    double creepTorque = 30.0;

    // How fast the commanded pressure may move, N.m per second. Two of them because a TCU raises
    // creep pressure gently and drops it promptly, and because the pedal's own rate limit is no
    // substitute: the engagement curve is nearly vertical where creep lives, so a pedal crawling
    // through it at `pedalRate` still delivers the whole creep torque inside fifteen milliseconds.
    //
    // **Measured, and it does exactly one of the two things it was written for.** Swept from 400 down
    // to 15 N.m/s it controls the clutch-torque rise precisely and by construction — 9.0 N.m per tick
    // at 400, 0.042 at 15, 2.8 at the 100 here — so the take-up is a ramp rather than a step. What it
    // does *not* touch is the engine: the worst idle excursion of a creep is **0.845 of idle at every
    // rate across that whole range**, because that excursion is not the take-up at all. It is the
    // lock-up a second and a half later, when `autoClutchPedal`'s catching-up term closes the clutch
    // on the last of the slip. Nothing in this brief moves that, and the anti-stall catches it — see
    // the note on it in `docs/vehicle-physics.md`.
    double creepApplyRate = 100.0;
    double creepReleaseRate = 400.0;

    // Brake travel at which creep is cancelled, so the car sits still at a light rather than slipping
    // its clutch against its own brakes for as long as the light is red.
    //
    // **A threshold and not a taper, and the difference is a burnt clutch.** A taper was tried first,
    // creep falling linearly to nothing over the first fifth of the pedal, and it leaves a band where
    // the brakes are already strong enough to hold the car and creep is still commanding torque into
    // them: measured at a tenth of the pedal, that is the car stationary with the clutch slipping at
    // **1.3 kW for as long as the driver waits** — 40 kJ over thirty seconds, which no dry plate
    // survives being asked for at a set of lights. A threshold has no such band. The transition is
    // ramped by `creepReleaseRate` in *time*, which is where a real controller ramps it, rather than
    // by pedal position.
    //
    // **Where it goes is derived rather than chosen, and the derivation moved when the brakes did.**
    // It has to sit below the brake application at which the brakes can hold the car against full
    // creep, or the band comes back.
    //
    // Full creep puts `creepTorque * 13.94` — 418 N.m — through this car's front axle. Against
    // `brakes.ini`'s 4200 N.m they crossed at a tenth of the pedal and a twentieth was half of that,
    // which is also about where a real brake-light switch trips. **Deriving the brakes from the
    // hardware moved the crossing** (2026-08-23, `docs/brake-model-brief.md`): the car makes 8791 N.m
    // at a fully applied pedal and the pedal is close to linear down here, so the brakes hold against
    // creep by **0.048** of the pedal and a cut at 0.05 was *above* the crossing — which is the band
    // this threshold exists to eliminate, and `CreepTests` caught it on the build that landed.
    //
    // 0.024 is half of the new crossing, on the same rule. It is below a brake-light switch's trip
    // rather than at it, and that is the right way round: the constraint that binds is the one about
    // the clutch, and a switch that trips later simply means creep is already gone by the time the
    // light comes on.
    //
    // A driver resting a foot exactly on it gets creep ramping up and down at `creepApplyRate` rather
    // than a step, which is the rate limit doing what it is for. If that ever proves to be felt, the
    // fix is the two-thresholds-and-hysteresis shape `advanceRevLimiter` and `ConverterLockup`
    // already use, and it is a real brake switch's shape too.
    double creepBrakeCut = 0.024;

    // The launch programme, which is the one rule here that owns the pedal outright while it is armed
    // rather than voting on it with the others.
    LaunchControl launch;

    // And the accelerator travel the controller reads as "not pressed". Creep is the no-demand state:
    // the moment the driver asks for torque the pedal map owns the clutch, and this is where that is
    // said. A dead band rather than a threshold because a real accelerator has one, and because a
    // sensor resting a percent off zero must not hold creep off for ever.
    double creepThrottleLift = 0.03;
};

// Whether launch control should be holding the car this tick, given what the driver is doing. Pure,
// and separate from the pedal it produces so that the *decision* can be tested without a clutch.
export [[nodiscard]] bool launchControlArmed(const LaunchControl& launch, const bool armed, const double brake,
                                             const double throttle, const double wheelSpeed, const bool inGear,
                                             const bool running);

// The pedal the launch regulator is asking for, given what the engine is doing and making. Returns a
// fully released pedal for a converter, which has no use for one.
export [[nodiscard]] double launchControlPedal(const DriveCoupling& coupling, const LaunchControl& launch,
                                               const double engineTorque, const double engineSpeed);

// Where the commanded creep torque goes next, in N.m at the clutch. `stepCoupling`'s shape: a target
// this instant answers for, and a rate the state may approach it at.
//
// Every one of the four things that cancels it cancels it outright, because every one of them is a
// statement about the car rather than a quantity to be traded off: out of gear there is nothing to
// creep against, a dead engine is not creeping anywhere, a driver on the accelerator has taken the
// clutch back, and a driver on the brake has asked the car to stay where it is. The smoothing is the
// rate, in time, and there is nowhere else it belongs — see `creepBrakeCut`.
export [[nodiscard]] double advanceCreep(const AutoClutch& assist, const double command, const double brake,
                                         const double throttle, const bool inGear, const bool running,
                                         const double deltaTime);

// The pedal the creep rule is asking for, given what it has commanded.
//
// **A converter car has no creep rule and must not be given one.** A fluid coupling at stall already
// passes torque — that is what a fluid coupling *is*, and it is why an automatic creeps without
// anything deciding that it should. Commanding a creep on top of one would be the same behaviour
// modelled twice, and the second copy would be the one with a number in it. So the dispatch is here,
// in the file that owns the kinds, and `stepDriveline` asks for a pedal without learning which is
// fitted — which is the same rule `stepDriveCoupling` is written under.
export [[nodiscard]] double creepPedal(const DriveCoupling& coupling, const double command);

// The pedal an automatic foot would be holding, given what the car is doing. `idleSpeed` arrives as a
// number rather than as the engine model because `:Driveline` imports this partition and not the
// other way about.
export [[nodiscard]] double autoClutchPedal(const AutoClutch& assist, const double idleSpeed,
                                            const double clutchSideSpeed, const double engineSpeed,
                                            const double throttle, const bool inGear);

// How far the clutch must be out *right now* because the engine is being dragged toward its stall
// floor. Zero for a healthy engine, one by `antiStallOpen`, and computed from the engine's own speed
// rather than the driveline side's — which is the hole the engagement logic above has: `caught`
// watches the road, and a full brake application arrests the wheels faster than any pedal follows,
// so the road-side answer arrives after the engine is already below its floor.
// **It only exists underneath the creep band.** Above it there is nothing for it to protect: a car
// travelling faster than idle through its gear is turning the engine rather than being turned by it,
// and an engine dragged down out there is being dragged down by a driver who has chosen the wrong
// gear — which is a thing a car is allowed to do, and a clutch that quietly opened itself to prevent
// it would be an assist nobody asked for. `grabFraction * idleSpeed` is the same line creep and the
// catching-up term already use, so there is one threshold in this file and not three.
//
// `clutchSideSpeed` is signed and is compared as such: a car rolling backwards is under the band from
// the wrong side, and that is exactly a stall coming.
export [[nodiscard]] double antiStallPedal(const AutoClutch& assist, const double idleSpeed,
                                           const double clutchSideSpeed, const double engineSpeed);

// Where the pedal goes next, given where it is, what the driver is doing with it and what the
// automation would like.
//
// The driver's foot wins outright past the free play, and neither blend works. Taking the more
// released of the two lets the automation let out a clutch the driver is deliberately biting, so a
// human cannot launch the car at all; taking the more clamped lets it clamp one the driver is
// deliberately slipping, so a human cannot protect the engine. The automation is an assist and the
// driver's own foot is what switches it off, exactly as a brake pedal cancels cruise control — and
// the line between the two is the free play a real pedal already has rather than an invented
// threshold.
export [[nodiscard]] double advanceClutchPedal(const AutoClutch& assist, const double pedal, const double driverPedal,
                                               const double automatic, const double antiStall, const double deltaTime);

} // namespace raceengine
