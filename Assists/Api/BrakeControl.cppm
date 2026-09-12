module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

export module raceengine.assists:BrakeControl;

import :WheelSensors;

namespace raceengine
{

// The brake actuator, and the anti-lock controller that owns it.
//
// **The actuator is modelled, not just the control law**, and that is most of what makes this feel
// like ABS rather than like a slip limiter. A hydraulic unit has three states per channel — build,
// hold and dump — each with a finite rate, and the cycling every driver recognises is those rates
// working against how fast the wheel can lock and how fast it re-accelerates once it is let go.
// Given instantaneous per-wheel pressure the same control law produces impossibly short stops and
// none of the feel.

// What the assist layer is clamping at each wheel this tick, N.m and never negative — a brake
// opposes rotation whichever way the wheel is turning, so a sign here would mean nothing.
//
// `commanded` false is not "no brakes": it is "nobody intervened", and the vehicle model then brakes
// on the driver's demand against each corner's own peak exactly as it did before this module
// existed. That is what keeps every fixture that names no assist bit-identical.
export struct BrakeCommand
{
    std::array<double, wheelCount> wheels{};
    bool commanded = false;
};

static_assert(std::is_trivially_copyable_v<BrakeCommand>, "the actuator command is copied by value into the tick");

// Channels, not wheels, and the difference is a design decision every real car makes.
//
// The two fronts are controlled individually because front braking is most of the car's stopping and
// because the fronts are what steers. The rear axle is **select-low**: both rear wheels follow
// whichever of them has less grip. That costs rear braking — the wheel on the good surface is held
// back to the bad one's pressure — and buys yaw stability, because the rear axle cannot develop a
// braking imbalance across it. On a split-mu surface it is the difference between stopping straight
// and spinning.
export enum class BrakeChannel : std::uint32_t
{
    FrontLeft,
    FrontRight,
    Rear
};

export inline constexpr std::size_t brakeChannelCount = 3;

// The classic phase sequence, and the order is load-bearing rather than descriptive.
//
// `Hold` is entered on the wheel's own **deceleration** and the dump is then confirmed by slip.
// `Dump` ends when the wheel starts coming back, `Recover` holds while it does, and `Reapply` puts
// pressure back until the wheel starts going again.
//
// **After the first cycle, slip stops being consulted at all** and the loop runs on the wheel's
// acceleration signature alone. That is not an optimisation — it is what makes the system survive
// its own reference speed estimate falling apart, which is exactly what happens when all four wheels
// slip together on a uniformly slippery surface. Built the other way round, with slip a *necessary*
// condition for intervening, this controller measured a 1% improvement over locked wheels on a
// mu 0.35 surface: the estimator collapsed, the estimated slip read near zero while the tyres were
// at 70% real slip, and the system politely handed the pressure back.
export enum class ModulatorPhase : std::uint32_t
{
    Passive,
    Hold,
    Dump,
    Recover,
    Reapply
};

// The hydraulic unit's rates, **pascals per second**.
//
// **They were fractions of full system pressure per second until 2026-08-23, and that was the fault
// rather than the units.** A rate stated as a fraction of the actuator's own range is not a property
// of the actuator at all: deriving this car's brakes from its calipers moved the peak from 5600 N.m
// to 10688 (`docs/brake-model-brief.md`), and every gradient in this file silently doubled with it.
// A modulator's valves do not know what the calipers on the other end of the pipe are.
//
// **Order of magnitude from the ABS literature, cross-checked by hand, and not tuned to a
// frequency.** Pressure-decrease gradients for a passenger-car modulator are on the order of
// 1000 bar/s and re-apply gradients a few hundred. The asymmetry is the hardware's: a dump opens the
// caliper to a low-pressure accumulator and a re-apply has to push fluid back in against the pump.
// This is the *same sourcing the fractional version had* — it is written down in the units it was
// sourced in, which is the whole change, and the previous 13/s and 3/s were those figures divided by
// a master cylinder pressure of "about 100 bar" that this car turns out not to have.
//
// The hand check the dump rate has to pass is that it can unload a wheel faster than the wheel can
// lock. On a low-grip surface — mu about 0.15, wet paint or polished ice — a front wheel carrying
// 5000 N sees 239 N.m of road against a brake that at this car's 2.94e-4 N.m/Pa makes that at 8 bar,
// so above 8 bar the excess spins the wheel down: from a full-pedal 125 bar the wheel is losing
// 3435 N.m into 1.45 kg.m^2, which is 2369 rad/s^2, and a wheel turning at 87 rad/s (100 km/h on a
// 0.3186 m tyre) is stopped in 37 ms. At 1000 bar/s the modulator is below that 8 bar in 117 ms and
// past the half that matters in 60 ms — **which is not inside it**, and is why this system's first
// dump on a slippery surface always ends with the wheel already stopped. That is a real
// characteristic of a real unit at full pedal and is what the recovery phase exists to handle.
//
// **The cycling frequency is not stated anywhere in this file.** It is whatever these rates and the
// wheel's own re-acceleration produce, and `AntilockBrakingTests` measures it.
export struct BrakeModulator
{
    // 1000 bar/s. Corroborated after the fact by a bench measurement of a passenger-car HCU:
    // decompression gradients of −90.5 to −94.8 MPa/s at master pressures of 8-15 MPa (Chang'an
    // University ABS test bench, J. Traffic & Transportation Eng. 2011-05).
    double dumpGradient = 1.0e8;
    // 300 bar/s. Real units pulse the inlet valve to get this; the average is what matters to the
    // wheel and the average is what is modelled.
    double reapplyGradient = 3.0e7;
    // The rear channel's own re-apply average, Pa/s — **the rear axle IS metered separately in
    // production, and that is sourced**: Robert Bosch GmbH, US patent 5,284,385 (filed 1990),
    // reduces the rear build rate against the front's by lengthening the build train's holding
    // phases (the duty factor of exactly the pulsing the front figure above averages), halving the
    // first pressure pulse, or holding outright — because dynamic axle-load transfer relieves the
    // rear axle, a rear build-up decoupled from the deceleration "leads inevitably to vehicle
    // instability", and the reduced rate "avoid[s] high control frequency at the rear axle".
    //
    // **No measured rear figure exists anywhere reachable** — the patent's own example is "e.g. to
    // halve it", which is an example and not a measurement, so the default is the front's value and
    // the mechanism ships bit-inert. A number here below the front's is a statement of Bosch's
    // mechanism at example grade; the pin in `AntilockBrakingTests` names that grade and flips the
    // day one is stated. (Burckhardt, *Radschlupf-Regelsysteme*, printed book only, is the one
    // place a real table may exist — on the human-fetch list.)
    double rearReapplyGradient = 3.0e7;
};

// How much authority the brake-recovery supervisor has over the five transition sites that used to
// read `pastBand` unconditionally. Three arms, and two of them are positive controls rather than
// options: an architecture that replaces two measured controllers has to be able to reproduce both.
export enum class RecoveryAuthority : std::uint32_t
{
    // `limitSlip == pastBand`. The pre-2026-09-07 production controller, to the bit.
    Unconditional,
    // `limitSlip == pastBand AND the road is taking torque AND the axle's lateral reserve is worth
    // protecting AND the caliper is not starved`. The shipped car.
    Supervised,
    // `limitSlip == false`. The pre-2026-08-25 acceleration-only recovery loop, to the bit.
    Disabled,
};

// What the road-action observable says about the wheel this channel is watching.
//
// **UNKNOWN is not a third opinion, it is the absence of one**, and it must PERMIT recovery. A stale
// or locked wheel holds an acceleration that reads deeply negative, so a design that let UNKNOWN
// refuse would be reading a frozen sensor as evidence that the tyre is loaded — which is exactly the
// artefact that made some of the diagnostic arms look better than they were. Refusal requires
// positive evidence; permission is the default.
export enum class RoadEvidence : std::uint32_t
{
    Unknown,
    Free,
    Loaded
};

// How many tooth crossings the road-action filter can hold. A fixed capacity because the channel
// state is copied by value into the car's saved bytes.
export inline constexpr std::size_t roadTorqueCapacity = 8;

// The vehicle-level lateral-authority gate: **an authority arbiter and not a stability controller**.
//
// It commands no pressure, adds no brake torque, performs no differential braking and has no
// actuator of any kind. Its whole output is one bounded number answering *how much of the yaw the car
// has now did the driver not ask for*, and the only thing that number does is decide whether
// preserving lateral tyre capacity is worth spending longitudinal capability on.
//
// **Why a vehicle-level signal is not optional here.** The same front-axle decision has to go two
// different ways on two surfaces a per-wheel controller cannot tell apart. Steering on uniform low
// grip, the front tyres must be kept near their longitudinal peak or the car goes straight on: law
// off, lateral travel collapses from 5.24 times the locked baseline to 1.12. Braking straight on a
// split surface, the *same* front lateral force is what sustains the rotation — the front axle's
// moment arm is +1.13 m ahead of the centre of mass against the rear's -1.51 m behind it, so a
// same-sign lateral force at the front holds the yaw and at the rear reverses it. Measured front
// lateral impulse over 1.00-2.00 s: 4385 N.s on the arm that yaws 17.3 degrees, 1325 N.s on the arm
// that yaws 2.1. The two populations do not overlap.
//
// The separator is driver intent and nothing else, which is why this reads steering.
export struct StabilitySetup
{
    // The bicycle reference's geometry. **COMMON PUBLIC DATA** for any road car.
    double wheelbase = 2.62;

    // Steering wheel angle to road wheel angle. **COMMON PUBLIC DATA**, and derivable once from an
    // authored rack. The reference only has to answer "is the driver asking for yaw at all", so an
    // error here is second order.
    double steeringRatio = 14.1;

    // Understeer gradient, rad per m/s^2. **ARCHETYPE DEFAULT**, and zero is the kinematic reference
    // rather than a missing value: it shapes how much yaw the driver asked for and not whether they
    // asked for any.
    double understeerGradient = 0.0;

    // The lateral acceleration the vehicle-relative yaw scale is drawn against, m/s^2, and the limit
    // above which this gate switches itself off entirely.
    //
    // **SOURCED, and it is already in this file**: Limpert, *Brake Design and Safety* 3rd ed. §9.3.1
    // switches the yaw moment build-up delay off above 0.4 g for the same physical reason — above it
    // the car is genuinely cornering, and a mechanism that exists for a straight-line split surface
    // has no business acting. Here it does two jobs. It sets `r_scale = lateralReference / v`, which
    // is what makes every threshold in this gate vehicle-relative and speed-relative rather than an
    // absolute rad/s somebody measured on one car. And it is the fail-safe: above it the gate returns
    // zero, so a car pulling real lateral acceleration keeps its front lateral protection whatever
    // the steering channel says. 0.4 * 9.80665 = 3.92266.
    double lateralReference = 3.92266;

    // What the car can pull laterally, m/s^2, used only to saturate the reference yaw rate — a car
    // cannot yaw faster than its grip allows however far the wheel is turned. **ARCHETYPE DEFAULT**:
    // a road car on dry tarmac. Deliberately a different number from `lateralReference`, because
    // clamping the reference at the scale would make "the driver is asking for yaw" unreachable at
    // full lock.
    //
    // **The accelerometer raises it and never lowers it**: a car demonstrably pulling more lateral
    // acceleration than the archetype allows is a car whose driver can be asking for more yaw than
    // the archetype admits, and the safe direction for this gate is always toward *asking*. Lowering
    // it on a slippery surface would be the unsafe direction — it would read a car that cannot
    // corner as a car whose driver did not ask to. 1.0 g.
    double lateralCapability = 9.80665;

    // The dimensionless share of `r_scale` that serves as both the intent deadband and the yaw-error
    // limit. One number rather than two, because two would be a tuning surface where the evidence
    // supports one significance scale. **ARCHETYPE DEFAULT.**
    //
    // **Monotone, and it only reaches the split-mu surface.** Measured across the four ensembles it
    // moves nothing on dry, straight low-mu or steered low-mu — none of those has yaw the driver did
    // not ask for — and on split mu the median heading error runs 8.00 degrees at 0.25, 12.08 at
    // 0.50, 16.29 at 1.00 and 22.33 at 2.00, against the previous controller's 17.26. A smaller share
    // makes the gate easier to trip, which biases longitudinal utilisation forward sooner. 0.25 is a
    // quarter of the yaw rate a 0.4 g corner implies at the car's current speed.
    double yawShare = 0.25;

    // The reference speed below which this gate is neutral, m/s. **DERIVED, not chosen**: one tooth
    // pitch over the reference estimator's own smoothing window — below it a wheel produces fewer
    // than one crossing per averaging interval and neither the speed nor the yaw reference means
    // anything. Filled by the composition root from the ring and the estimator; the default is the
    // Golf's 48-pole ring at 0.050 s smoothing, and any value at all keeps a divide bounded.
    double speedFloor = 0.834;
};

// What one channel needs that is not per-wheel sensor data: its own actuator calibration, what the
// driveline is doing to the wheel it watches, and the vehicle-level gate's answer.
//
// **A struct rather than five more parameters** because `advanceAntilockChannel` already takes nine,
// and because these five arrive together from one place — the composition root, which is the only
// thing that can see the whole car.
export struct AntilockChannelInputs
{
    // What this channel's brake makes per pascal, N.m/Pa, and what it makes at full system pressure.
    // The second is what `AntilockSetup::roadShare` is a fraction of.
    double torquePerPressure = 0.0;
    double peakBrakeTorque = 0.0;

    // The radius the ECU converts a rim acceleration into an angular one with, metres.
    double rollingRadius = 0.3186;

    // The capture timer this channel's sensor is read on, seconds. Not a control parameter: it is
    // what the quantisation floor of the road-action estimate is computed from, per sample.
    double timerResolution = 1e-6;

    // What the driveline is putting on the control wheel, N.m. **Identically zero on an undriven
    // wheel and DIRECTLY KNOWN on a driven one** — a car whose ECU commands the driveline knows this
    // number and does not estimate it.
    double driveTorque = 0.0;

    // Whether that number is available at all. **False makes the road evidence UNKNOWN, which
    // PERMITS** — the alternative is an observable that silently reads a driven wheel's drive torque
    // as zero, and with the caliper empty the drive term is not a correction to the estimate, it is
    // the whole of it.
    bool driveTorqueKnown = true;

    // How much of the car's current yaw the driver did not ask for, 0 to 1, from
    // `advanceStabilitySupervisor`. Zero is neutral and is what every fallback lands on.
    double yawDisturbance = 0.0;
};

export struct AntilockSetup
{
    bool enabled = false;

    BrakeModulator modulator;

    // The controller's own clock, Hz. Deliberately not the physics rate: a real ECU runs its loop
    // on a timer and consumes whatever the capture registers happen to hold, so the estimate it
    // reads has an age that has nothing to do with how often the world was integrated.
    double controlRate = 1000.0;

    // **The peak is found by the wheel running away from the car, not by a slip number**, and that
    // is the one decision in this file that everything else follows from.
    //
    // Below the tyre's peak a braked wheel is *stable*: more slip makes more road torque, so the
    // wheel settles wherever the brake and the road balance. Past the peak it is unstable — more
    // slip makes less road torque — and the wheel departs. So "this wheel is decelerating much
    // harder than the car is" **is** a peak detector, and it is one that needs to know nothing about
    // the tyre, the load or the surface.
    //
    // That matters here more than it would on a real car, because this tyre's peak moves a long way
    // with the surface: measured on the Golf's own compound, the longitudinal curve peaks at
    // kappa 0.09 on dry tarmac and at **kappa 0.03 on a mu 0.35 surface**, because scaling the
    // friction without scaling the stiffness moves the peak in proportion. A controller regulating
    // to a fixed 8-30% slip band — which is the published ABS control range, and which the first
    // version of this file used — is then working three times past the peak on anything slippery,
    // and it measured 1% better than locked wheels where there was 27% to be had.
    //
    // Slip stays as a *second* trigger, not as a gate. A wheel that has been dragging steadily since
    // before the deceleration transient is one this misses and slip catches.
    double slipEnter = 0.20;
    double slipExit = 0.08;

    // How much harder than the car a wheel must be decelerating before it is judged to be departing,
    // m/s^2 at the rim, negative. **Relative, and that is what makes it surface-independent**: the
    // absolute version of this threshold has to clear whatever the car itself is doing, which on
    // tarmac is 1 g and on ice is 0.3 g, so an absolute number is either deaf on one surface or
    // twitchy on the other.
    //
    // **A hand calculation against the sensor's own noise floor, and then some.** The tone ring's
    // microsecond timer resolves speed to about 0.03% at 100 km/h, and the controller differences
    // speed across one pulse interval — 1.5 ms there — so timer quantisation alone can manufacture
    // 0.0003 * 27.78 / 0.0015 = 5.6 m/s^2 of apparent acceleration. The car's own deceleration is
    // known only through `ReferenceSpeedState::rate`, which is an average and therefore lags the
    // pedal: at the start of a stop it still reads the coast the car was on, so a threshold close to
    // the noise floor fires on every wheel on the car the instant the pedal moves. Measured at 1 g:
    // 242 cycles on a dry stop that needed none, and 190 m against 41.
    // 1.5 g leaves room for both errors. 1.5 * 9.80665 = 14.710.
    double lockDeceleration = -14.710;

    // And how much harder than the car a wheel must be *accelerating* to count as recovering, m/s^2.
    // Half a g above the car, for the same margin reason.  0.5 * 9.80665 = 4.903.
    double recoveryAcceleration = 4.903;

    // The larger threshold that says the wheel is being **driven** back up by the road rather than
    // merely no longer dragged down by the brake, m/s^2 above the car.
    //
    // **A hand calculation about the worst surface, not about this one.** Release a locked wheel onto
    // mu = 0.2 — about the least a road car ever meets — carrying 4700 N: the road puts
    // 0.2 * 4700 * 0.3186 = 299 N.m into a 1.45 kg.m^2 wheel, which is 207 rad/s^2, or 66 m/s^2 at
    // the rim. 2 g is far below that on any surface and far above what the tone ring's coarseness
    // can manufacture, so it separates a wheel that has genuinely come back from one that has merely
    // stopped going away. 2.0 * 9.80665 = 19.613.
    double recoverySurge = 19.613;

    // **How much authority the recovery supervisor has**, and the field the 2026-09-07 architecture
    // replaced `slipAwareRecovery` with (`docs/abs-architecture-design.md`).
    //
    // The old boolean armed one term, `pastBand`, which was read unconditionally in five transition
    // sites and did three jobs at once: wheel protection, brake recovery and — without ever saying so
    // — vehicle stability, because what it actually bought was lateral tyre capacity under steering.
    // The forensic audit measured all three: it costs a dry stop (20 of 29 members strand the rear
    // axle, rear utilisation 0.416 against 0.745), it costs a straight low-mu stop (worse than locked
    // wheels on 23 of 29 members), it costs split-mu yaw (17.3 degrees against 2.1) — and it is the
    // only thing keeping the car steerable on a slippery surface, where switching it off collapses
    // lateral travel from 5.24 times the locked baseline to 1.12.
    //
    // So the term survives and its gating does not. `Supervised` is the shipped car: the five sites
    // keep their structure and the term that arms them is qualified on whether the road is taking
    // torque, on whether the axle's lateral reserve is worth spending longitudinal capability to
    // protect, and on whether the caliper has been empty longer than it takes to fill.
    //
    // The other two arms exist because a replacement that cannot reproduce both of the things it
    // replaces cannot be attributed. `Unconditional` is the pre-2026-09-07 production controller **to
    // the bit**, and `Disabled` is the pre-2026-08-25 acceleration-only loop **to the bit**. Both are
    // measured arms with published numbers, and both are the positive controls the architecture's
    // validation is blocked on.
    RecoveryAuthority recoveryAuthority = RecoveryAuthority::Supervised;

    // --- the road-action observable ------------------------------------------------------------
    //
    // `T_road_hat = T_brake_commanded + I_wheel * dOmega/dt - T_drive`, in N.m of spin-up torque.
    // With the caliper empty it degenerates to the inertial term, which is a direct measurement of
    // what the road is doing to a wheel nothing else is touching — and that is the question the dry
    // failure turns on, because a wheel the ECU believes is slipping and which is making no force is
    // a wheel there is nothing to protect.

    // The ECU's own figure for a wheel's rotational inertia, kg.m^2. **ARCHETYPE-ESTIMABLE and
    // measured to be so**: swept from half to twice the plant's value the dry ensemble does not move
    // (0 of 29 stranded at every point). It is the ECU's calibration and not the car's number, which
    // is why it lives here beside `brakeTorquePerPressure` rather than being imported from the
    // vehicle — this module cannot see a vehicle and must not learn to.
    double wheelInertia = 1.45;

    // The gate the estimate must clear to count as "the road is taking torque", **as a fraction of
    // this channel's own peak brake torque**. Dimensionless on purpose: the front axle's peak is
    // 3665.5 N.m and the rear's 729.8, a ratio of five, and a threshold in N.m does not even carry
    // across one car's two axles let alone across a fleet. Measured plateau on the dry ensemble:
    // 0.010 to 1.000 is 0 or 1 of 29 stranded, a factor of one hundred.
    //
    // **The value is bounded from both sides by physics and the two bounds are close together**,
    // which is the most important thing to know about this number. From below it must clear what a
    // nearly unloaded wheel offers: a rear tyre carrying 7% of its static load makes about 30 N.m,
    // and the sensor's own quantisation floor is of order 25, which is why the gate is the larger of
    // this share and a multiple of that floor. From above it must stay UNDER what a tyre can offer on
    // the worst surface the car still has to be protected on: at mu 0.35 a front tyre under about
    // 5000 N offers 0.35 * 5000 * 0.3186 = 558 N.m against a front peak brake torque of 3665, which
    // is **0.152 of peak**. A share at or above that reads the whole front axle as making no useful
    // force on every slippery surface, and the steered low-mu ensemble falls off a cliff there:
    // measured on the 15-member fixture the paired lateral-travel ratio is 4.67 at 0.100, 1.19 at
    // 0.120 and 0.26 at 0.150.
    //
    // 0.100 is two thirds of that upper bound and an order of magnitude above the lower one. The dry
    // ensemble is flat across the whole range (0 of 29 stranded from 0.010 to 0.150) and the straight
    // low-mu ensemble is the other side of the same trade, so the usable region for all four
    // ensembles at once is narrow: **0.100 is the only point on the measured grid that satisfies
    // every one of them**, and that narrowness is recorded rather than smoothed over.
    double roadShare = 0.100;

    // How many tooth crossings the estimate is filtered over. **Event domain and not time domain**,
    // and that is not a preference: the sample rate is proportional to wheel speed, so a time
    // constant averages a different number of teeth at every speed and is a different filter at
    // 100 km/h and at 20. A fixed count of crossings averages the same amount of sensor evidence
    // wherever it is read. Capped by `roadTorqueCapacity`.
    std::uint32_t roadSamples = 5;

    // How many of the wheel's own measured tooth intervals a sample may age before it is INVALID,
    // dimensionless. Stated against `WheelSpeedReading::period` rather than in seconds for the same
    // reason the filter is event-domain.
    double staleShare = 2.0;

    // How many multiples of the sensor's own quantisation floor the estimate must clear, on top of
    // `roadShare`. The floor is computed per sample from the capture timer and the measured period —
    // `(I/r) * (timerResolution/period) * (speed/period)` — and is of order 25 N.m at the rear at
    // 100 km/h, which is the same order as the separation the observable is asked to make. Without
    // this term a small `roadShare` on a coarse ring is a decision taken below the noise.
    double noiseShare = 1.0;

    // The vehicle-level lateral-authority gate. See `StabilitySetup`.
    StabilitySetup stability;

    // --- yaw moment build-up delay, and it ships OFF ---------------------------------------------
    //
    // **The one feature both split-mu criteria name as absent**, now with a citable description
    // rather than a folk memory of one. Limpert, *Brake Design and Safety* 3rd ed., §9.3.1
    // (`docs/fetched/limpert.txt`): on a split-coefficient surface, "brake line pressure is
    // increased at the 'high' wheel in stages or steps as soon as the first pressure reduction
    // caused by wheel lockup tendencies takes place at the 'low' wheel. When the pressure at the
    // 'high' wheel has reached its locking level, it is no longer influenced by the signals of the
    // 'low' wheel; that is, it is individually controlled."
    //
    // It is a mechanism **this car's class needs**: Limpert scopes it to "smaller vehicles with
    // lower values of mass moment of inertia", where the yaw builds faster than a driver can
    // countersteer, against heavy long-wheelbase cars that develop it slowly enough to manage
    // without.
    //
    // And it is explicitly a **compromise** — "between good steering response and minimized stopping
    // distance", with the book noting that "differences exist between ABS manufacturers" and that
    // directional stability may be lost where minimum stopping distance is the design objective. So
    // whether it goes on this car is a decision about what the car should be, not a defect to fix,
    // and it is not one to take without the driver.
    bool yawMomentDelay = false;

    // How much of the modulator's own re-apply gradient the high wheel may build at while the delay
    // is engaged. **PLACED**: Limpert describes the staging and publishes no rate for it, and
    // neither does the Bosch material behind the modulator's other gradients. Expressed as a share
    // of an existing calibrated quantity rather than as a new pascals-per-second, so that it cannot
    // drift away from the hydraulics it is a fraction of. Swept in `[.yaw-delay]`.
    double yawDelayApplyShare = 0.25;

    // The hold between steps, seconds — "in stages or steps". **PLACED, and cosmetic to the
    // average**: the share above sets the mean rate and this sets only how granular the staircase
    // is. Twenty controller periods at the default control rate.
    double yawDelayStagePeriod = 0.02;

    // Lateral acceleration above which the delay switches off, m/s². **Sourced, and one of the few
    // hard numbers in the passage**: "prior to the electronic stability system, to optimize braking
    // while turning, a lateral acceleration sensor switches off the yaw moment delay feature for
    // lateral acceleration exceeding 0.4 g" — because in a corner the large braking force wanted at
    // the outer wheel produces a moment that opposes the lateral force's, leaving a mildly
    // understeering car the driver can hold. 0.4 × 9.80665 = 3.923.
    double yawDelayLateralLimit = 3.92266;
};

// What the yaw moment delay is holding. Zero on every tick of every car until the feature is
// switched on, which is what makes it bit-inert rather than merely off.
export struct YawMomentDelayState
{
    bool engaged = false;

    // Which front channel is the "high" one — the wheel on the grippier side, identified by its
    // being the one that has **not** yet asked for a pressure reduction.
    std::size_t highChannel = 0;

    // The staged ceiling that channel's request is capped at, pascals. Meaningless while
    // `engaged` is false.
    double ceiling = 0.0;
    double sinceStage = 0.0;

    // Whether a brake application is under way, and what each front channel's cycle count stood at
    // when it began.
    //
    // **`AntilockChannelState::cycles` never resets, and reading it directly cost a seat lap.** The
    // trigger is "the first pressure reduction *of this brake application*", and the first version
    // of this asked whether the other front had ever reduced pressure at all. On a single-stop
    // fixture those are the same question. On a 372 s session they are not: both fronts crossed
    // zero cycles on the same tick 25 s in, and the delay was disarmed for the remaining 347 —
    // through 25 of the lap's 44 brake applications that each had a genuine window, one of them
    // 361 ms wide. Every fixture in this repository is one stop with a fresh state, so nothing
    // could see it (`traces/rack-exit-20260830-yawdelay-seat.csv`).
    bool applied = false;
    std::array<std::uint32_t, 2> baseCycles{};

    // Whether this brake application's delay is over. Once the high wheel has taken its own control
    // — or the staircase has climbed to everything the driver is asking for — the book is explicit
    // that it "is no longer influenced by the signals of the 'low' wheel", and that has to mean for
    // the rest of the application rather than until the next time the low wheel cycles. Without it
    // the staircase re-arms every time it finishes and the high wheel is held down for the whole
    // stop; measured, that turned a 7.7% distance penalty into the same steering benefit at 7.0%,
    // and it is the difference between the book's "small" increase and an open-ended one.
    bool completed = false;
};

// One channel of the modulator and of the controller that drives it.
export struct AntilockChannelState
{
    ModulatorPhase phase = ModulatorPhase::Passive;

    // What the channel is holding, pascals. Meaningless while `phase` is `Passive`, where the wheel
    // simply gets what the master cylinder is at.
    double pressure = 0.0;

    // The controller's own view of the wheel it is watching, updated **only when a new tooth has
    // gone past**. Differencing on the control clock instead would report zero for every step
    // between pulses and a spike on the one that gets a new number, which at walking pace is a
    // 30 ms-old spike. This is what a real ECU has and its staleness is the point.
    double lastSpeed = 0.0;
    double sinceUpdate = 0.0;
    std::uint64_t lastPulses = 0;
    // Peripheral acceleration, m/s^2, as measured between the last two sensor updates.
    double acceleration = 0.0;

    // Whether the wheel has surged past `recoverySurge` since the dump ended. Pressure goes back on
    // when that surge *subsides*, not when it starts: a wheel still accelerating hard is a wheel
    // still climbing back towards the road's speed, and re-applying into it puts it straight back
    // down again.
    bool surged = false;

    // What the channel was holding when the wheel last departed, pascals — the memory a production
    // unit's two-stage re-apply is built on, written on every entry to the dump. The re-apply runs
    // at the full gradient below it and hands over to the slip-proximity taper at it: below the
    // level that provably held this wheel moments ago there is nothing to probe for, and crawling
    // back through that region at the taper's rate was measured leaving the rear axle under-braked
    // for tenths of a second per cycle. Meaningless while the channel is passive.
    double departurePressure = 0.0;

    // How many times this channel has released pressure. **Counted on entering the dump and nowhere
    // else**, which is a correction: counting the entry to `Hold` instead counts a threshold
    // chattering as well as a cycle, and the two differ by a factor of four. Measured on a low-mu
    // stop, the Hold-based count said 19.3 Hz while the pressure — and therefore the pedal under the
    // driver's foot — moved 4.8 times a second.
    //
    // The channel that reports whether the system is working and, differenced against the time it
    // spent engaged, what frequency it is working at.
    std::uint32_t cycles = 0;

    // --- the road-action observable's own state (2026-09-07) ------------------------------------
    //
    // A ring of the last `roadSamples` estimates, **one per tooth crossing and never one per control
    // step**. A control step with no new crossing carries no new information about the road, and a
    // filter fed the held value at the control rate is a filter whose length in *evidence* changes
    // with speed.
    std::array<double, roadTorqueCapacity> roadSamples{};
    std::uint32_t roadCount = 0;
    std::uint32_t roadHead = 0;

    // The filtered estimate, N.m of spin-up torque, and what the supervisor made of it. Both are
    // telemetry as well as state: a gate nobody can see in a trace is a gate nobody can report on.
    double roadTorque = 0.0;
    RoadEvidence evidence = RoadEvidence::Unknown;

    // How long this channel's commanded pressure has been at the actuator's floor, seconds.
    //
    // **This is the whole answer to "the controller left the dump and the caliper is still empty".**
    // A phase is not a delivered torque: this unit empties a rear caliper in 54 ms and fills it in
    // 181, so a state machine that transitions in one control period can be three actuator time
    // constants away from the pressure its phase implies. Nothing else in this file carries actuator
    // history, and without it the dry limit cycle has no bound that does not depend on the sensor.
    double timeAtFloor = 0.0;

    // Whether the recovery supervisor is limiting slip on this channel this period — the term that
    // replaced `pastBand` — and whether the ECU's own belief that the wheel is past the band was
    // asserted at all. The two differ exactly where the supervisor changed a decision, which is what
    // makes "changed decisions per stop" a directly measured number rather than a shadow run.
    bool limited = false;
    bool banded = false;
};

export struct AntilockState
{
    std::array<AntilockChannelState, brakeChannelCount> channels{};

    YawMomentDelayState yawDelay;
};

// One step of the vehicle-level lateral-authority gate, answering **how much of the car's current
// yaw the driver did not ask for**, 0 to 1.
//
// Pure and stateless: a reference yaw rate from the steering angle and the estimated speed, compared
// against what the yaw sensor reports, both measured against a scale that is a property of the car
// and its speed rather than an absolute number somebody read off one vehicle.
//
// **It commands nothing.** The return value gates whether the recovery supervisor's lateral-capacity
// protection has authority at an axle, and it reaches no actuator by any other path.
//
// Every failure lands on **zero**, which retains the protection: a gate that withdrew lateral
// authority because a sensor went quiet would be a gate that turns a sensor fault into a
// straight-on. Non-finite readings, an invalid reference, a speed below the floor, and real lateral
// acceleration all return zero.
export [[nodiscard]] double advanceStabilitySupervisor(const StabilitySetup& setup, const double referenceSpeed,
                                                       const bool referenceValid, const double steeringWheelAngle,
                                                       const double yawRate, const double lateralAcceleration);

// Which wheel a channel is controlling on, given what the sensors report.
//
// Front channels answer with their own wheel. The rear channel answers with whichever rear wheel is
// turning slowest — the one with least grip — which is select-low stated as a function rather than
// buried in the controller.
export [[nodiscard]] std::size_t antilockControlWheel(const BrakeChannel channel, const WheelSpeedReadings& wheels);

// Which wheels a channel drives.
export [[nodiscard]] bool antilockDrivesWheel(const BrakeChannel channel, const std::size_t wheel);

// One controller step for one channel.
//
// `requestedPressure` is what everything upstream — the driver's foot, traction control, the
// cornering brake — wants at this wheel, in pascals. What comes back is what the actuator is
// actually at. With the channel passive the two are the same number, to the bit, which is what makes
// an unassisted car unchanged.
//
// `referenceAcceleration` is `ReferenceSpeedState::rate`: how fast the ECU believes the *car* is
// changing speed. Every threshold above is measured against it, so this controller carries the
// estimator's errors as well as its own — which is correct, and is where the character comes from.
//
// `channel` is which of the modulator's three channels this is, and it exists because the rear one
// is metered separately (`BrakeModulator::rearReapplyGradient` and its source above).
// One step of the yaw moment build-up delay, answering the ceiling the **high** front channel's
// request must be capped at. `state.engaged` says whether there is a cap at all and
// `state.highChannel` says which of the two front channels it applies to.
//
// It lives above `advanceAntilockChannel` rather than inside it because it is the one part of this
// controller that is **not** per-channel: the whole mechanism is one front wheel's pressure being
// held down by the other front wheel's signals, which a per-channel step cannot see.
//
// `left` and `right` are the two front channels as they stood at the start of this controller
// period, and `frontRequests` is what everything upstream wants at each of them. That one period of
// lag is deliberate and is what a real unit has: it reacts to the sample it has captured, not to one
// it has not taken yet.
// `braking` is whether the driver is on the pedal. **It is a scope condition and not a guard**: the
// book describes this feature entirely for braking on a split surface, and yaw during acceleration
// is traction control's business. Without it the delay arms on a traction-control brake
// intervention — measured, it fired for one tick of a scripted standing launch, which is a wheel
// being braked for spinning and nothing to do with a split surface.
export [[nodiscard]] double advanceYawMomentDelay(const AntilockSetup& setup, YawMomentDelayState& state,
                                                  const AntilockChannelState& left, const AntilockChannelState& right,
                                                  const std::array<double, 2>& frontRequests,
                                                  const double lateralAcceleration, const bool braking,
                                                  const double deltaTime);

// `inputs` is everything the channel needs that is not per-wheel sensor data: its own actuator
// calibration, the driveline torque at the wheel it watches, and the vehicle-level gate's answer.
// **A channel cannot compute any of them** — two are the composition root's and the third is the
// whole car's — which is the same reason the yaw moment delay lives outside this function.
export [[nodiscard]] double advanceAntilockChannel(const AntilockSetup& setup, const BrakeChannel channel,
                                                   AntilockChannelState& state, const WheelSpeedReading& wheel,
                                                   const double wheelRoadSpeed, const double referenceSpeed,
                                                   const double referenceAcceleration, const bool referenceValid,
                                                   const double requestedPressure, const AntilockChannelInputs& inputs,
                                                   const double deltaTime);

} // namespace raceengine
