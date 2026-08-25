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
export enum class BrakeChannel : std::uint32_t { FrontLeft, FrontRight, Rear };

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
export enum class ModulatorPhase : std::uint32_t { Passive, Hold, Dump, Recover, Reapply };

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
    // 1000 bar/s.
    double dumpGradient = 1.0e8;
    // 300 bar/s. Real units pulse the inlet valve to get this; the average is what matters to the
    // wheel and the average is what is modelled.
    double reapplyGradient = 3.0e7;
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

    // **The recovery law reads how far the estimated slip is from the controller's own band**, and
    // this is the switch that turns that off for an A/B. On, the law this closes over is the one
    // `docs/braking-chain-brief.md`'s instrument convicted: every recovery decision was written as
    // though "not departing" meant "recovered", and a lightly loaded wheel past the tyre's peak is
    // neither — road torque nearly balances brake torque there, so the wheel neither departs nor
    // surges, and the old law re-applied into that equilibrium and held the rear axle at three times
    // its peak slip for an entire stop, delivering 0.74 of its capacity against the fronts' 0.85.
    //
    // Three legs, all anchored on the `slipEnter`/`slipExit` band already calibrated above — no new
    // threshold is introduced anywhere:
    //
    //  - a dump does not end merely because the wheel stopped departing while the estimated slip is
    //    still past the band: equilibrium past the peak is what "stopped departing at 0.42 slip" is;
    //  - a channel in recovery whose wheel is neither departing nor genuinely re-accelerating, with
    //    slip still past the band, is **stuck** and dumps again rather than re-applying into it;
    //  - the re-apply gradient tapers as the estimated slip approaches the band from below, so
    //    pressure comes back gently near the peak and at the full rate well under it.
    //
    // **Every leg is gated on the reference being valid and reading past the band**, which is what
    // keeps the estimator-collapse case honest: on a uniformly slippery surface the estimated slip
    // reads far *below* the truth (the reference is biased low when every wheel slips at once), the
    // guards then never fire, and the channel behaves exactly as the previous law did. Slip reading
    // low disables this law; only the first version of this file, which required slip to *confirm*
    // an intervention, could be disarmed by it.
    bool slipAwareRecovery = true;
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
};

export struct AntilockState
{
    std::array<AntilockChannelState, brakeChannelCount> channels{};
};

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
export [[nodiscard]] double advanceAntilockChannel(const AntilockSetup& setup, AntilockChannelState& state,
                                                   const WheelSpeedReading& wheel, const double wheelRoadSpeed,
                                                   const double referenceSpeed, const double referenceAcceleration,
                                                   const bool referenceValid, const double requestedPressure,
                                                   const double deltaTime);

} // namespace raceengine
