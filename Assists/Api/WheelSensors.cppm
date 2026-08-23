module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

export module raceengine.assists:WheelSensors;

namespace raceengine
{

// **The whole module is defined by what it cannot reach, and this is the file that states it.**
//
// A real ABS or traction controller knows wheel rotational speeds, yaw rate and lateral
// acceleration. It does not know road speed, slip ratio, tyre force, vertical load or the friction
// under any wheel — and a controller given those outperforms every system ever fitted to a car while
// feeling wrong in a way that is very hard to diagnose from the seat, because the failures that give
// ABS its character are all *estimation* failures.
//
// So `raceengine.assists` is a module of its own rather than a partition of `raceengine.physics`,
// and `raceengine.physics:Vehicle` imports *it*. That direction is the enforcement: an `import
// raceengine.physics;` added anywhere in here is a module dependency cycle, which is a hard build
// failure and not a review comment. `wheelCount` below is restated for the same reason — it cannot
// be imported from `:Telemetry` without opening the door — and `:Vehicle` static_asserts the two
// agree, so the two statements cannot drift.
//
// The second half of the enforcement is the parameter lists: every entry point here takes sensor
// readings and driver demand, so even inside a unit that had somehow named a `VehicleState` there
// would be no instance of one to read.

// Four. See above for why this is not `cornerCount`, and `Vehicle.cppm` for the assertion that ties
// them together.
export inline constexpr std::size_t wheelCount = 4;

// Wheel order matches the vehicle's — front left, front right, rear left, rear right — because the
// two arrays are copied across the boundary elementwise and a different order here would be a
// silent transposition rather than a type error.
export inline constexpr std::size_t frontLeft = 0;
export inline constexpr std::size_t frontRight = 1;
export inline constexpr std::size_t rearLeft = 2;
export inline constexpr std::size_t rearRight = 3;

// The toothed ring on the hub and the Hall element beside it.
//
// **A wheel speed sensor does not report angular velocity.** It reports a pulse per tooth, and the
// speed is *derived from the timing of those pulses* — so both the resolution and the latency of
// what the controller reads are functions of how fast the wheel is turning. That is not a detail:
// it is the mechanism behind every low-speed ABS behaviour there is, and a model that samples the
// model's own omega once per tick has thrown it away before the controller ever runs.
export struct ToneRing
{
    // Poles on the ring. A 48-pole encoder turning at one revolution per second produces 48 pulses
    // per second, which is the whole of the arithmetic.
    //
    // **A data parameter with a soft source, and it is marked as one.** 48 poles is the common VAG
    // magnetic-encoder-ring count and the exact Mk7 figure is not confirmed by anything to hand.
    // Nothing downstream is calibrated against 48: the resolution and dropout tests below assert
    // against `teeth` and the wheel radius rather than against a measured interval, so changing this
    // to the 44 at the other end of the plausible range moves the numbers and breaks no assertion.
    std::uint32_t teeth = 48;

    // The capture timer the period is measured on. A microsecond is an ordinary automotive input
    // capture resolution, and at 100 km/h it is 0.07% of a 1.5 ms pulse interval — so it is visible
    // in the reading and nowhere near the dominant error.
    double timerResolution = 1e-6;

    // **Seams, deliberately inert.** A damaged ring drops or doubles pulses and a noisy Hall element
    // jitters the edge timing; both are real failures and both change the character of the system.
    // Modelling either is its own piece of work with its own acceptance evidence, so they are stated
    // here and read by nothing rather than being invented now.
    double edgeJitter = 0.0;
    std::uint32_t damagedTeeth = 0;
};

// One sensor's own state, carried across ticks. Trivially copyable because it rides in whatever
// saves and restores the car.
export struct WheelSensorState
{
    // How far the wheel has turned since the last tooth crossing, radians, and how long ago that
    // crossing was.
    double angleSincePulse = 0.0;
    double timeSincePulse = 0.0;

    // The interval between the last two crossings, seconds. This is the measurement; everything the
    // controller reads is arithmetic on it.
    double measuredPeriod = 0.0;

    // Which way the wheel was turning at the last crossing. Direction-capable active sensors are
    // what a car with hill-hold has, and this car has one.
    double direction = 0.0;

    std::uint64_t pulses = 0;
};

// What the controller gets to look at.
export struct WheelSpeedReading
{
    // Angular velocity in rad/s, signed by the last measured direction.
    //
    // **Bounded above by the pulse that has not arrived yet**, which is the one inference every ECU
    // makes and is why no low-speed threshold has to be written anywhere: if the wheel were still
    // turning at the last measured speed, the next tooth would already have gone past. So the
    // reading is the smaller of the measured speed and one tooth pitch over the time since the last
    // crossing, and a wheel that stops has its reading decay hyperbolically towards zero rather than
    // being held at whatever it was doing when it stopped.
    double speed = 0.0;

    // How old the measurement is, seconds. 1.5 ms at 100 km/h on a 48-pole ring, 30 ms at 5 km/h.
    double age = 0.0;

    // False until the first tooth has ever gone past, which is a real state a controller has to
    // handle: a car that has been standing still has no wheel speed measurement at all.
    bool valid = false;

    std::uint64_t pulses = 0;
};

export using WheelSensorStates = std::array<WheelSensorState, wheelCount>;
export using WheelSpeedReadings = std::array<WheelSpeedReading, wheelCount>;

static_assert(std::is_trivially_copyable_v<WheelSensorState>, "sensor state rides in the car's saved bytes");
static_assert(std::is_standard_layout_v<WheelSensorState>, "and rollback will later");

// One tick of the four sensors.
//
// `wheelSpeeds` is the physical rotation of each wheel in rad/s and is the **only** place the model
// crosses into this module. The angle turned over the tick is accumulated and every tooth crossing
// inside it is counted with its own sub-tick instant, so at 100 km/h — where a tooth passes every
// 1.5 ms against a 2.78 ms tick — the pulses are not aliased away by the tick rate.
export [[nodiscard]] WheelSpeedReadings sampleWheelSensors(const ToneRing& ring, WheelSensorStates& states,
                                                           const std::array<double, wheelCount>& wheelSpeeds,
                                                           const double deltaTime);

// How the estimator decides which wheels to believe.
//
// **This is a real component with real failure modes and they are the character, not a defect.**
// The estimate is the fastest undriven wheel under braking and the slowest under power, bounded by
// how hard a car can plausibly change speed — and when every wheel slips together there is nothing
// left to take a reference from, so it coasts on the bound and drifts away from the truth. That
// case is exactly the one where a system that could see the truth would look superhuman.
export struct ReferenceSpeedSetup
{
    // The radius the ECU multiplies wheel speed by, metres.
    //
    // **Deliberately the nominal unloaded radius, and deliberately not corrected.** The effective
    // rolling radius shrinks as the tyre deflects, so under heavy braking the real radius is smaller
    // than this and every wheel reads slightly *slow* — a systematic error in the direction that
    // makes the system think there is more slip than there is. Real systems live with it.
    double nominalRadius = 0.3186;

    // Which wheels the engine drives. An undriven wheel is a free road-speed measurement and is why
    // traction control on a front-drive car is a much easier problem than on a four-wheel-drive one.
    std::array<bool, wheelCount> driven{true, true, false, false};

    // How fast the reference is allowed to fall and rise, m/s^2, when no wheel supports the change.
    //
    // **Sourced to what a car can physically do rather than to what this one does.** A road car on
    // dry asphalt decelerates at about 1.0-1.2 g and a tyre cannot exceed roughly 1.3 g on any
    // surface a road car meets, so a reference falling faster than 1.3 g is being fooled by a
    // locked wheel rather than following the car. 12.7 m/s^2 is 1.3 g at standard gravity
    // (1.3 * 9.80665 = 12.749).
    //
    // **This is the hard bound and not the working rate**, and the difference decides whether the
    // estimator survives a slippery road. Falling at 1.3 g the moment every wheel goes is right on
    // tarmac and useless on ice, where the car is doing 0.3 g: measured on a mu 0.35 surface the
    // estimate went from 27 m/s to 8 m/s in a second and a half while the car was still doing 22,
    // the estimated slip read near zero against 70% of real slip, and the anti-lock system politely
    // handed the pressure back. What the estimate actually coasts on is
    // `ReferenceSpeedState::rate` — the deceleration it was last *observed* to have — and this only
    // ever caps it.
    double fallLimit = 12.749;
    // And rising. A front-drive car is traction limited at the driven axle whatever its power: with
    // 61.4% of the weight on the front at rest and load moving *off* it as the car accelerates, the
    // front carries around 55% at the point it matters, so the best it can do is about 0.55 of the
    // tyre's own coefficient. At a road tyre's 1.1 that is 0.61 g, and 5.884 m/s^2 is 0.6 g — a
    // reference rising faster than that is being dragged up by wheels that are spinning.
    double riseLimit = 5.884;

    // How long the observed rate of change is averaged over before it is coasted on, seconds.
    //
    // **Bounded at both ends by things that are not preferences.** It has to settle well inside one
    // anti-lock cycle, which is of the order of 100 ms, or the controller's own thresholds — every
    // one of which is measured against this — lag the pedal by longer than a cycle. And it has to
    // average many tooth crossings rather than a few, or it carries the ring's coarseness straight
    // into the threshold: 1.5 ms between pulses at 100 km/h, 30 ms at 5. A twentieth of a second is
    // a third of a cycle and thirty pulses at speed, which satisfies both.
    double rateSmoothing = 0.050;
};

export struct ReferenceSpeedState
{
    // The ECU's estimate of road speed, m/s. Positive is forwards.
    double speed = 0.0;

    // How fast the estimate was last observed to be changing while a wheel was still supporting it,
    // m/s^2, signed. **This is what the estimate coasts on when every wheel goes**, rather than a
    // fixed limit: a car that was decelerating at 0.3 g a moment ago is still decelerating at about
    // 0.3 g now, whatever surface it is on, and assuming otherwise is what makes an estimator
    // useless in the exact conditions it exists for. It lags a genuine change of surface by
    // `rateSmoothing`, which is a real error and is the honest one to have.
    double rate = 0.0;

    // A first-order lag of `speed`, m/s, which is what `rate` is differenced against. Carried in the
    // state rather than recomputed because a differentiator needs a memory and this is it.
    double lagged = 0.0;

    // True once any wheel has ever produced a measurement.
    bool valid = false;

    // How far the estimate is currently being carried by its own rate limit rather than by a wheel,
    // seconds. A system running on the bound is a system that has lost its reference, and this is
    // the channel that says so — it is what a plausibility check would watch and what an ESC
    // retrofit will want.
    double coasting = 0.0;
};

// One controller step of the reference speed estimator.
//
// `braking` is the brake light switch: which end of the wheel population to believe flips with it,
// because under braking every wheel reads low and under power the driven ones read high.
export void advanceReferenceSpeed(const ReferenceSpeedSetup& setup, ReferenceSpeedState& state,
                                  const WheelSpeedReadings& wheels, const bool braking, const double deltaTime);

// A wheel's road speed as the ECU computes it, m/s: the reading through the nominal radius and
// nothing else. Exported because both controllers and every test does it, and three copies of one
// multiplication is three places for the radius to be wrong.
export [[nodiscard]] double sensedRoadSpeed(const ReferenceSpeedSetup& setup, const WheelSpeedReading& wheel);

// The slip the controller *believes* it is at, which is not the slip the tyre is at. Positive means
// the wheel is turning slower than the reference (braking slip); negative means faster (drive slip).
// Both are reported by the same expression so a sign convention cannot differ between the two
// controllers that read it.
export [[nodiscard]] double estimatedSlip(const double referenceSpeed, const double wheelSpeed);

} // namespace raceengine
