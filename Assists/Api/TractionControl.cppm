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

    // --- the brake channel: fast, per wheel, and expensive ---
    //
    // Torque per unit of slip error beyond the target, N.m. Proportional and nothing else: an
    // integrator on a channel that is arbitrating with a much slower one winds up during every
    // transient the slow one is about to fix.
    double brakeGain = 2600.0;

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
                                   const bool referenceValid, const double throttle, const double deltaTime);

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
