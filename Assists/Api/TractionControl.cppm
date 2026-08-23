module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

export module raceengine.assists:TractionControl;

import :WheelSensors;

namespace raceengine
{

// Two systems that share one actuator and are otherwise unrelated, kept in one file because the
// per-wheel brake intervention path is the thing they share and splitting them would mean two copies
// of it. They are named separately throughout — a channel called `traction` and a channel called
// `cornering` — so that a later question about which of them did something has an answer.

export enum class TractionMode : std::uint32_t {
    // The driver's foot goes to the engine unmodified and no brake is applied for slip. What every
    // fixture in this repository runs under unless it says otherwise.
    Off,
    // Slip allowed past the tyre's peak. Slower off a low-grip surface and quicker to rotate the car,
    // which is what the mode is for.
    Sport,
    Full
};

export struct TractionSetup
{
    TractionMode mode = TractionMode::Off;

    // The controller's own clock, Hz. Shares the anti-lock unit's in practice; stated separately
    // because nothing makes that true and a stability controller added later will want its own.
    double controlRate = 1000.0;

    std::array<bool, wheelCount> driven{true, true, false, false};

    // **Where each mode regulates, from the published control range rather than from this tyre.**
    // The 8-30% band an anti-lock system works in is the same band a traction system works in, for
    // the same reason: it is where a longitudinal curve has its peak. Full mode holds 10%, just
    // under where a road tyre peaks, and sport holds 22%, comfortably past it. Neither number was
    // read off this model's own curve, and if a compound with a peak somewhere else is fitted these
    // stay where they are — which is exactly what happens on a real car.
    double fullTargetSlip = 0.10;
    double sportTargetSlip = 0.22;

    // --- the brake channel: an electronic differential lock, not a second engine channel ---
    //
    // **Reformed 2026-08-23, and what it was before was the wrong device.** It braked each driven
    // wheel on that wheel's slip against the *reference speed* — which is the engine channel's
    // signal wired to a valve. A real brake-based traction intervention (VW call it EDS, Bosch call
    // it BTCS) is a **differential lock**: it brakes the wheel turning faster than its partner so
    // that an open differential passes the torque to the other side. When both driven wheels spin
    // together there is no cross-axle torque to redistribute and the brakes cannot help at all —
    // that case belongs to the engine, and a system that brakes both wheels there is just heating
    // two discs while the engine drives through them.
    //
    // **The structural benefit is that it no longer divides by the reference speed.** The defect
    // found from the seat on 2026-08-23 — reference collapses after a lock-up, traction control
    // reads the recovering wheels as wheelspin and holds them down — could not have happened to a
    // channel keyed on left-against-right, because both wheels were equally slow and the difference
    // was zero. Bounding the reference error fixed that instance; this makes the channel immune to
    // the class.
    //
    // The Golf GTI Performance also has a **mechanical** limited-slip differential, so most of this
    // work is already done in hardware on this car and the channel is the smaller of the two.
    //
    // Torque per m/s of cross-axle departure, N.m, proportional and nothing else: an integrator on a
    // channel arbitrating with a much slower one winds up through every transient the slow one is
    // about to fix.
    //
    // **Sized so the ceiling is reached at one metre per second of departure.** At the contact patch
    // that is one driven wheel turning 3.1 rad/s faster than the other once the turn's own
    // kinematics are taken out — unambiguous wheelspin rather than a differential doing its job in a
    // corner. A third of this car's 3665 N.m front brake is 1221, and 1221 N.m per m/s puts the
    // ceiling exactly there.
    double brakeGain = 1221.0;

    // Below this the difference is the sensors talking rather than the axle, m/s. One tone ring
    // pulse at 100 km/h is 1.5 ms of staleness, which on a 48-pole ring is about 0.04 m/s of
    // quantisation per wheel; 0.10 leaves room for two of them and is still a twentieth of what
    // reaches the ceiling.
    double brakeDeadband = 0.10;

    // **The brake channel is speed limited and a real one has to be**, because it is a friction
    // device with the engine on the other side of it: everything it does becomes heat in one disc.
    // Volkswagen quote EDS as working to about 80 km/h and hand over to the engine above that, and
    // a disc temperature model — which this engine does not have — is what a production unit uses
    // to withdraw sooner. Faded rather than switched, over the last 10 km/h, because a step in brake
    // torque at a threshold is felt as a bump through the car and no hydraulic unit produces one.
    double brakeSpeedLimit = 22.222;
    double brakeSpeedFade = 2.778;

    // **How long the reference may be running on its own bound before the brake channel withdraws**,
    // seconds. `ReferenceSpeedState::coasting` reports exactly this and its own comment says it is
    // "what a plausibility check would watch"; this is that check.
    //
    // The case it exists for was measured rather than imagined: a lap of Bathurst with the car
    // landing from a crest at 126 km/h with **all four wheels stopped in the air**. Every reading is
    // then a wheel spinning back up at its own rate, the difference across the axle is noise, and a
    // differential lock acting on it is braking a wheel for no reason. 3.3% of this channel's
    // interventions over that lap were in that state.
    //
    // 20 ms, because a wheel crossing a bump has its reading carried for a control period or two and
    // a wheel in the air has it carried for as long as it is off the ground. Long enough not to fire
    // on the first, short enough to catch the second well inside a landing.
    double brakePlausibilityHold = 0.020;

    // Track widths, metres. **ECU calibration rather than vehicle state**, and the reason this
    // channel needs them: two driven wheels in a corner turn at genuinely different speeds and an
    // electronic diff lock that did not know it would brake the outside wheel all the way round
    // every bend. The undriven pair's own speed difference over the rear track is the yaw rate the
    // road is imposing, which predicts what the front pair's difference ought to be — and what is
    // left over is wheelspin. `advanceCorneringBrake` below has taken its kinematics this way since
    // it was written; this channel now does the same.
    double frontTrack = 1.539;
    double rearTrack = 1.516;

    // Ceiling, as a fraction of that corner's full brake torque.
    //
    // **Sized as a transient device, which is what it physically is.** A front wheel getting the
    // whole of first gear — 350 N.m at the crank through 13.94 is 4879 N.m at the axle, half of it
    // per wheel with an open differential — accelerates at 1683 rad/s^2 into a 1.45 kg.m^2 wheel.
    // A third of this car's 1575 N.m front brake is 525 N.m, which takes 362 rad/s^2 off that: it
    // cannot hold the wheel against the engine and is not meant to. It arrests the *departure* while
    // the engine channel, which is three powers of ten slower, takes the torque away. A ceiling large
    // enough to hold on its own would put the whole of that 25 kW into one disc for as long as the
    // surface lasted.
    double brakeCeiling = 0.333;

    // --- the engine channel: slow, shared, and sustainable ---
    //
    // **The only lever this model has is the throttle**, and that is a real limitation rather than a
    // simplification. A production traction system retards ignition or cuts injectors, both of which
    // act inside 20 ms and can take torque below what a shut throttle gives. Reaching either would
    // mean a term in the engine model, and the driveline is out of scope for this work — so the
    // engine channel here is throttle closing, and 100 ms is a drive-by-wire throttle body's
    // full-travel time. That makes it the slow channel by construction, which is the division of
    // labour the design wants, but it is slower than a real one and the report says so.
    double engineCloseTime = 0.100;
    // Torque comes back gently or the wheel simply goes again. Twice the closing time.
    double engineOpenTime = 0.200;
    // Throttle removed per unit of slip error beyond the target.
    double engineGain = 4.0;
};

export struct TractionState
{
    // 0 lets the driver's foot through untouched, 1 shuts the throttle.
    double engineReduction = 0.0;
    // What the brake channel is asking of each wheel, as a fraction of that corner's full torque.
    std::array<double, wheelCount> brakeFraction{};
    // Whether either channel did anything this step, kept apart so the telemetry can show which.
    bool brakeActive = false;
    bool engineActive = false;
};

static_assert(std::is_trivially_copyable_v<TractionState>, "assist state rides in the car's saved bytes");

// XDS, and it is neither traction control nor the differential.
//
// The Mk7 GTI brakes its **inside front** wheel while cornering to emulate a limited-slip
// differential: the open diff sends torque to whichever wheel is easiest to turn, which on corner
// exit in a front-drive car is the unloaded inside one, and braking it puts that torque back on the
// outside wheel where there is load to use it. It is triggered by *cornering*, not by wheelspin,
// which is the whole of what separates it from the traction controller above.
//
// Modelled off the sensors and nothing else. The undriven rear pair gives the turn's kinematics for
// free — their speed difference over the rear track is the yaw rate the road is imposing — and the
// inside front's excess over what those kinematics predict for it is the quantity to act on.
export struct CorneringBrakeSetup
{
    bool enabled = false;

    double controlRate = 1000.0;

    // Track widths, metres, which the ECU has as calibration and the sensors do not report.
    double frontTrack = 1.539;
    double rearTrack = 1.516;

    // Cornering below this does not get an intervention, m/s^2. **A hand calculation about what a
    // corner is**: 0.3 g on a 30 m radius is 34 km/h and on a 100 m radius is 62 km/h, which is a
    // firmly driven road corner rather than a lane change. 0.3 * 9.80665 = 2.942.
    double onsetLateralAcceleration = 2.942;

    // N.m per m/s of the inside front wheel's excess over its kinematic speed.
    double gain = 900.0;
    // And the ceiling, as a fraction of that corner's full brake torque. Volkswagen describe XDS as
    // a *slight* brake application; a seventh of full braking on one front wheel is about 225 N.m
    // here, which is 706 N at the tyre against an axle passing several thousand.
    double ceiling = 0.143;
};

export struct CorneringBrakeState
{
    std::array<double, wheelCount> brakeFraction{};
    bool active = false;
};

static_assert(std::is_trivially_copyable_v<CorneringBrakeState>, "and so does this");

// The slip target for a mode. `Off` has none and answers with zero, which no caller should be
// reading — it is here so the function is total.
export [[nodiscard]] double tractionTargetSlip(const TractionSetup& setup);

// One controller step of the traction system.
//
// `throttle` is what the driver's foot is asking for, 0 to 1. What comes back is in `state`:
// per-wheel brake fractions and an engine reduction, kept separate all the way to the telemetry
// because which of the two did the work is the question the design is answering.
//
// `brakePeakTorque` is what each corner's brake makes at full pressure, N.m. It is ECU calibration
// and not vehicle state — a controller has to know what its own actuator is worth or it cannot ask
// it for a torque — and it arrives as an argument so that the one array the whole assist unit holds
// is not restated in three setups that could then disagree.
export void advanceTractionControl(const TractionSetup& setup, TractionState& state, const WheelSpeedReadings& wheels,
                                   const ReferenceSpeedSetup& reference,
                                   const std::array<double, wheelCount>& brakePeakTorque, const double referenceSpeed,
                                   const bool referenceValid, const double referenceCoasting, const double throttle,
                                   const double deltaTime);

// What the throttle becomes once the engine channel has had it.
export [[nodiscard]] double tractionThrottleScale(const TractionState& state);

// One controller step of the cornering brake.
//
// `lateralAcceleration` is the accelerometer's, m/s^2, positive towards the car's own +x. It is a
// gate and not a gain — everything quantitative comes off the wheel speeds.
export void advanceCorneringBrake(const CorneringBrakeSetup& setup, CorneringBrakeState& state,
                                  const WheelSpeedReadings& wheels, const ReferenceSpeedSetup& reference,
                                  const std::array<double, wheelCount>& brakePeakTorque,
                                  const double lateralAcceleration, const double deltaTime);

} // namespace raceengine
