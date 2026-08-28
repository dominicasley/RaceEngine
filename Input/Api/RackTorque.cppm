module;

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.input:RackTorque;

namespace raceengine
{

namespace
{

// Kept here as well as in the implementation unit, and that is legal rather than sloppy: an
// unnamed namespace is internal to each translation unit, so these are two distinct copies
// and not one entity defined twice. This side is needed because an inline or constexpr
// function below calls them, and those cannot move — a caller has to see their bodies.
inline void appendRackInteger(std::string& text, const long long value)
{
    auto buffer = std::array<char, 32>{};
    const auto written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    text.append(buffer.data(), written.ptr);
}

inline void appendRackNumber(std::string& text, const double value, const int precision)
{
    auto buffer = std::array<char, 64>{};
    const auto written =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, precision);

    text.append(buffer.data(), written.ptr);
}

constexpr auto rackGravity = 9.80665;

constexpr auto rackMetresPerSecondToKilometresPerHour = 3.6;

constexpr auto rackRadiansPerSecondToRevolutionsPerMinute = 9.549296585513721;

constexpr auto rackRadiansToDegrees = 57.29577951308232;

} // namespace

// Stage one of three: what the road is doing to the steering, in newton metres, with no notion
// anywhere in it of what is going to display that.
//
// It knows nothing about Fanatec, nothing about evdev, nothing about eight bits and nothing about
// eight newton metres. That is the whole point of it being its own partition. A road car's rack
// sees far more at the limit than any belt-driven base can produce, so something has to compress
// it — and the failure this separation exists to prevent is that the compression gets done by
// raising the tyre's self-aligning moment instead, at which point the *tyre model* is silently
// carrying a hardware compensation and the car feels right on one base and wrong everywhere else.
//
// Nothing here imports the physics module either, which is deliberate: what it takes is the
// geometry the kinematic solve already produced and the forces the tyre model already produced,
// as plain numbers, so the whole derivation can be pinned by tests that stand up no vehicle at
// all. The caller does the extraction, because the caller is the one holding a tick's result.
//
// Chassis frame throughout, SI: **+x is the car's left** (`outboardSign` in the physics module
// carries why, and the day it cost), +y up, +z forward, metres, newtons, radians.

// One steered corner, as the solve left it. Every point is where it *is* this tick, not where it
// was authored — which is what makes camber gain, bump steer and a wheel half on a kerb arrive
// through this path without a single term written for any of them.
export struct SteeredCorner
{
    // The kingpin axis, by its two ends. A double wishbone's upper end is its upper ball joint and
    // a strut's is its top bearing; the solve reports both in the same field, so nothing here has
    // to know which linkage it is holding.
    glm::dvec3 lowerBallJoint{0.0};
    glm::dvec3 upperBallJoint{0.0};

    // Where the tie rod picks up on the upright, solved, and where it picks up on the rack, with
    // the rack's travel already added. The line between them is the tie rod, and a tie rod is a
    // two-force member — which is what makes the whole steering Jacobian below closed form.
    glm::dvec3 steeringArm{0.0};
    glm::dvec3 rackOuter{0.0};

    glm::dvec3 contactPatch{0.0};
    glm::dvec3 patchNormal{0.0, 1.0, 0.0};

    // The tyre's whole resultant at the patch — vertical, longitudinal and lateral together — in
    // the chassis frame. All three matter and only one of them is obvious: the vertical load acting
    // through the kingpin's offset from the patch is what a parked car's steering weight *is*, and
    // it is also how a kerb reaches the driver's hands.
    glm::dvec3 tyreForce{0.0};

    // The self-aligning moment, N·m, as a couple about the patch normal. Pneumatic trail lives in
    // here; mechanical trail does not and must not, because mechanical trail is the geometry above
    // and would then be counted twice.
    double aligningMoment = 0.0;
};

// The car's electric power steering, as a boost curve.
//
// **This is a property of the vehicle and not of whatever is displaying it**, and saying so is the
// whole reason it lives here rather than where it used to. Until 2026-08-21 the assist was two
// literals in `SimulatedCar::publishRackTorque` — a flat multiplier of 0.22 at rest rising to 0.55
// at speed, applied between stage one returning and the publish — and its stated justification was
// that the unassisted parked rack "saturates the sheet's 4 N·m ceiling within ±1.6 mm". A sheet's
// ceiling is a number about a wheel base. A car parameter sized against one is a hardware
// compensation living in the car, which is exactly what the three-stage split exists to prevent;
// the split protected the three stages that were named and the compensation went to the seam
// between them.
//
// Both halves of that justification turn out to be false, measured (`[.steering-geometry]`):
//
//   - The parked rack is nothing like that stiff. Swept quasi-statically with the brake held, this
//     car reads **1.67 N·m at the rim at 38° of lock and 7.00 N·m at 340°** — 0.11 N·m per mm of
//     rack, against the 2.5 the comment claimed. It is wrong by a factor of about twenty-five, and
//     ±1.6 mm of travel is worth 0.4 N·m rather than 4.
//   - The rail-to-rail shake it was introduced to tame was independently diagnosed two passes
//     later as the *damper* closing a loop through the hardware — 3.1 N·m of damper against
//     0.33 N·m of carcass spring at a 0.58 mm wiggle — and fixed by band-limiting it.
//
// What replaces it is the shape an electric rack actually has: **assist force is a boost on the
// driver's own effort, and the boost falls both with road speed and with how hard the rack is
// loaded.** Both of those inputs are the car's.
//
// **The load term is the one that matters, and it took getting backwards once to see why.** A first
// attempt had the boost *rising* with load — help where the effort is, which sounds right and is
// what a naive reading of "progressive" suggests. Measured on the 35 m/s sweep it took the fall in
// rim torque as the fronts go past the grip peak from **50% to 34%**. That fall *is* the limit cue;
// it is the entire mechanism criterion 13 tests, so a curve that compresses it attacks the
// milestone's acceptance criterion at precisely the moment it counts. A real electric rack sheds
// assist as steering effort rises for exactly this reason, and with the sign the right way round the
// same sweep gives **53.6%** — a slight *enhancement* of what the bare rack says rather than a
// dilution of it. **That ratio is the acceptance test for this map, not the endpoint torques.**
//
// Two more properties worth knowing before touching any of it:
//
//   - **Past the motor's capacity the incremental gain is exactly one.** A flat multiplier scales a
//     kerb strike down along with everything else; a saturating motor hands the driver every newton
//     past its own ceiling, undiminished.
//   - **This model's parked rack cannot support a parking target, and the numbers below say so.**
//     Fitted to 6 N·m at the cornering limit, the map lands parking at 1.78 N·m against the 2.0–2.5
//     a Mk7 asks of a driver's arms — and it cannot be raised without inverting the load term,
//     because a heavier parking effort needs *less* boost at parking than at the limit, which is the
//     anti-EPS shape that costs the cue. The cause is upstream: the unassisted parked rack measures
//     7.0 N·m at full lock where a real car's is sixty to a hundred, because the tyre model has no
//     turn-slip torque — a parked tyre here twists rather than scrubbing. The assist is asked to
//     remove an effort that was never there. **Fix the tyre, not this.**
//
// **Every number below is a placeholder in the sense every vehicle figure in this repository is**:
// no measured Mk7 boost curve was available, so the level is anchored on one torque-at-the-rim
// target — 6 N·m at the cornering limit — and the two shaping terms are set to keep the limit cue at
// or above what the bare rack gives. What they are emphatically *not* fitted to is what fits under
// eight newton metres.
export struct PowerAssist
{
    // The most boost this curve ever gives, and it gives it at `sqrt(knee * taper)` of rack force —
    // the peak of the hump the two forces below make between them. Zero is an unassisted rack, which
    // is what a default-constructed steering box is: an assist is something a car has, so a car has
    // to say so.
    double peakBoost = 0.0;

    // **Below this the motor holds back, and that is what leaves the on-centre road feel.** A real
    // boost curve has a low-gain region at small driver effort — it is what makes a car track
    // straight and what lets the driver feel the road through the first few newton metres — and
    // without one the assist is at its strongest exactly where the signal is smallest. Measured on
    // the scripted launch, adding this lifted the median torque at the rim by three quarters.
    double boostKneeForce = 600.0;

    // **Above this the motor gives the load back, and that is what leaves the limit cue.** See the
    // account above for the measurement that settles the direction; every real electric rack sheds
    // assist as steering effort rises, and this is where it starts.
    //
    // **These two are independent now, and until 2026-08-22 they were not.** The shape used to be
    // `F/((F+k)(F+t))`, which is *symmetric in k and t* — swapping the two fields leaves the curve
    // bit-identical — so "hold back on centre" and "give the load back at the limit" were one
    // parameter pulling in opposite directions. On this car the limit sits about four times the
    // on-centre force, which is far too close together for one hump to do both: every repositioning
    // that bought a bigger limit cue paid for it in on-centre feel, measured, at a rate of about one
    // for one. That is not a tuning problem, it is the shape, and it is why the taper had been left
    // where it did no work rather than moved somewhere it did harm. See `assistBoost`.
    double boostTaperForce = 1500.0;

    // How much of the boost the motor gives up by motorway speed, and the speed at which half of
    // that has been given up. Every electric rack schedules this way — light in a car park, firm on
    // a motorway — and it is the second of the two inputs a real boost curve is drawn against.
    double speedFalloff = 0.20;
    double falloffSpeed = 20.0;

    // What the motor can put into the rack, newtons. An MQB-class rack-parallel unit is good for
    // something like this, and the figure matters at exactly one place: past it the driver carries
    // every further newton themselves, which is the kerb strike that must not be scaled down.
    double maximumForce = 8000.0;
};

// The boost this rack gives at a given load and road speed. Exported because it is the curve, and a
// reader checking whether the wheel should be this heavy wants to evaluate it rather than infer it.
//
// **The load term is a band pass in the rack force with second-order corners.** A rise term
// `F²/(F² + knee²)` suppresses the boost below the knee, a shed term `taper²/(taper² + F²)` gives it
// back above the taper, and the product is normalised so that `peakBoost` is literally the peak: it
// occurs at `F = sqrt(knee · taper)` with a value of `(taper/(taper + knee))²`, which is closed
// form, so the field means what it says rather than meaning it after a scaling nobody can see.
//
// **What it replaced, and what the difference actually is.** The term was
// `F·N/((F + knee)(F + taper))` — a band pass with *first-order* corners. Both are symmetric under
// swapping knee and taper (`(500, 2500)` and `(2500, 500)` are the same curve in either), and a test
// pins that, because it was got wrong once here: the fix is not asymmetry, it is **steepness**.
//
// The old term rolls off as `F` below the knee and as `1/F` above the taper; this one as `F²` and
// `1/F²`. At twice the peak's force the old shape is still at 89% of its peak and this one is at
// 64%; at four times, 64% against 22%. That is the whole of it, and it is what makes the two forces
// *placeable*: with first-order corners, shedding enough assist across a limit region only 1.5×
// wide in force means dragging the peak far below it, which drags the on-centre suppression down
// with it. Measured against Dominic's 201-second Bathurst session, every repositioning of the old
// shape traded one end for the other at about one for one — buying a limit cue took the on-centre
// ratio from 0.80 down to between 0.57 and 0.71, and that ratio is what the knee exists to protect.
// With second-order corners the same session gives **0.99 on centre and half again the limit cue**,
// from a peak that barely moves. Both halves measured; neither is a preference.
export [[nodiscard]] inline double assistBoost(const PowerAssist& assist, const double rackForce,
                                               const double roadSpeed)
{
    if (assist.peakBoost <= 0.0)
    {
        return 0.0;
    }

    const auto speed = std::max(roadSpeed, 0.0);
    const auto schedule = 1.0 - assist.speedFalloff * speed / (speed + std::max(assist.falloffSpeed, 1e-9));

    const auto knee = std::max(assist.boostKneeForce, 1e-9);
    const auto taper = std::max(assist.boostTaperForce, 1e-9);
    const auto load = std::abs(rackForce);

    const auto rise = load * load / (load * load + knee * knee);
    const auto shed = taper * taper / (taper * taper + load * load);

    // The product's own peak, in closed form, so dividing by it makes `peakBoost` the peak.
    const auto atPeak = (taper / (taper + knee)) * (taper / (taper + knee));

    return assist.peakBoost * std::max(schedule, 0.0) * rise * shed / atPeak;
}

// Where the taper goes, as a fraction of the rack force at which the car's steering limit sits, and
// how far below it the knee goes.
//
// **Two fractions and one measured number per car, which is the whole point of the exercise.** The
// limit's position in rack force is a property of the vehicle — its front axle load at the limit,
// its tyre's aligning-moment peak and its steering geometry — and `steeringLimitLoad` with
// `tyreAligningPeak` computes it from the car's own data without anybody driving. So a second car
// gets its assist placed by construction rather than by a seat session, which is the failure the
// three hand-written constants this replaces would have repeated for every vehicle added.
//
// The taper sits just *below* the limit so the motor is already shedding by the time the tyre
// starts to: placed at the limit it is still at full boost through the first half of the cue, which
// is precisely the fault being fixed. The knee sits at the taper over `assistTaperOverKnee`, which
// makes the pass band a little over an octave wide — wide enough that the peak is a plateau rather
// than a spike, narrow enough that the on-centre suppression is not fighting the shed.
export inline constexpr double assistTaperOfLimit = 0.90;
export inline constexpr double assistTaperOverKnee = 2.5;

// The road speed the level is anchored at, m/s. The target below is a torque a driver's arms feel in
// a corner, so it is stated at a cornering speed rather than parked — the speed schedule is a real
// part of the curve and an anchor that ignored it would put the level wherever the schedule happened
// to be.
export inline constexpr double assistAnchorSpeed = 20.0;

// The steering box, stated in the units its parts are actually specified in.
export struct SteeringRack
{
    // Metres of rack per unit of steering demand, so full lock is this much travel. The vehicle
    // setup owns the number; it is copied here rather than derived, because this partition does not
    // import the one that states it.
    double travelPerInput = 0.055;

    // What the rim turns lock to lock. Together with the travel above this is the pinion radius,
    // and the pinion radius is the whole of the steering ratio on a rack and pinion: there is no
    // second reduction between the pinion and the rim, so torque at the pinion *is* torque at the
    // rim. That identity is why this stage can report newton metres at all.
    double lockToLockDegrees = 756.0;

    // Coulomb friction at the rack, newtons. Seal drag, the pinion's own mesh and the ball joints.
    // **Placeholder**, in the sense every number in this repository's vehicle data is: a documented
    // order of magnitude for a rack-and-pinion steering box rather than a measured figure for this
    // one. It is here rather than left out because a rack with no friction at all makes a wheel that
    // wanders on centre, which reads as a physics fault.
    double friction = 120.0;

    // Viscous damping at the rack, newtons per metre per second. Same provenance.
    double damping = 900.0;

    // Below this the Coulomb term is regularised rather than switched. A friction force written as
    // `-F * sign(v)` flips its whole magnitude between two consecutive ticks at a standstill, which
    // on a force feedback base is an audible buzz at the output rate and on a physics trace is a
    // square wave nobody can read past. `tanh` is the same force everywhere the rack is actually
    // moving and a steep ramp through zero where it is not.
    double frictionReferenceSpeed = 0.01;

    // The motor between the driver and all of the above, if this car has one. Default is none, so a
    // bare steering box is a bare steering box and every case that never mentions it is unassisted.
    PowerAssist assist{};

    // The steering system's own rotational inertia at the rim, kg·m²: the wheel and column, the
    // rack referred through the pinion, and the two front corners' rotation about their kingpins
    // referred through the steering ratio.
    //
    // **It is the car's, and it used to be the wheel base's.** The parking damper was sized as
    // 2·√(K·J) with J = 3.1e-3 — a figure fitted from *this base's* limit cycle, which is a device
    // inertia wearing a car number's clothes and the second half of the compensation `PowerAssist`
    // above is the first half of. Reckoned honestly for this car it is about thirty times larger:
    // a 2.5 kg rim at a 0.185 m radius of gyration is 0.086 on its own, and the two front
    // assemblies contribute 2 × 0.67/13.8² ≈ 0.007 through the ratio.
    //
    // **Read `docs/post-m2-remediation-brief.md` before using this to size anything.** The number
    // is right and the system it describes is not in the simulation: the rack angle is commanded
    // straight from the demand, so there is no steering degree of freedom anywhere and the only
    // inertia physically in the loop a damper closes is the base's own rim. The two answers differ
    // by 5.5x and both are correct about different oscillators. What makes them one oscillator is a
    // rack with a mass and a compliance under the driver's demand, which is a change to the vehicle
    // model rather than a number.
    double steeringInertia = 0.093;
};

// The pinion's radius, which is the rack's travel divided by the rim's rotation. Exported because
// it is the number that converts everything here between newtons and newton metres, and a reader
// checking the absolute magnitudes wants to see it.
export [[nodiscard]] inline double pinionRadius(const SteeringRack& rack)
{
    const auto lockToLock = rack.lockToLockDegrees * 0.017453292519943295;
    if (lockToLock <= 0.0)
    {
        return 0.0;
    }

    // Full travel is both ends: the demand runs -1 to 1.
    return 2.0 * rack.travelPerInput / lockToLock;
}

// The whole assist for a car whose steering limit sits at `limitRackForce` newtons, leaving
// `targetRimTorque` newton metres in the driver's hands when it gets there.
//
// **Three stated numbers and one measured one, and none of them is a wheel base's.** The two
// fractions above place the shape; this target places the level; and the measured one is the car's
// own limit, which `steeringLimitLoad` and `tyreAligningPeak` compute from its data with nobody
// driving. That is the property the 2026-08-22 brief asked for in as many words — "whatever
// mechanism you use to place the knee and taper should be able to be re-derived from a trace rather
// than re-guessed per car" — and computing it from the car instead of from a trace is the same
// requirement met one step earlier.
//
// **The level is solved rather than fitted.** `peakBoost` is the most boost the curve ever gives,
// the target is what the driver should feel at the limit, and the shape at the limit is closed form,
// so the boost needed there is `unassisted/target - 1` and the peak is that divided by the shape and
// the speed schedule. One line, no search — and it re-solves when the shape moves, which is what
// stops the two drifting apart. They did drift once: the level was fitted to a boost curve whose
// taper was later found to be doing no work, so it was carrying the level for a shape nobody was
// running.
export [[nodiscard]] inline PowerAssist assistPlacedAtLimit(const SteeringRack& rack, const double limitRackForce,
                                                            const double targetRimTorque)
{
    auto assist = PowerAssist{};

    const auto limit = std::max(limitRackForce, 0.0);
    const auto taper = limit * assistTaperOfLimit;

    assist.boostTaperForce = taper;
    assist.boostKneeForce = taper / assistTaperOverKnee;

    const auto unassisted = limit * pinionRadius(rack);
    if (!(targetRimTorque > 0.0) || !(unassisted > targetRimTorque))
    {
        // A rack already lighter than the target needs no motor, and asking for one would have it
        // *adding* effort. An unassisted rack is the honest answer and is what a default-constructed
        // `PowerAssist` already is.
        return assist;
    }

    // The shape at the limit, with `peakBoost` factored out — which is exactly `assistBoost` with a
    // unit peak, so it is asked of `assistBoost` rather than written out a second time.
    auto unit = assist;
    unit.peakBoost = 1.0;

    const auto shapeAtLimit = assistBoost(unit, limit, assistAnchorSpeed);
    if (!(shapeAtLimit > 0.0))
    {
        return assist;
    }

    assist.peakBoost = (unassisted / targetRimTorque - 1.0) / shapeAtLimit;

    return assist;
}

export inline constexpr std::size_t steeredCornerLimit = 2;

export struct RackTorque
{
    // Per corner, N·m about that corner's own kingpin axis. Diagnostic, and the first place to look
    // when the sign of the whole thing is wrong.
    std::array<double, steeredCornerLimit> kingpinTorque{};

    // Newtons at the rack, from the tyres alone, and then the two resistances that oppose the
    // rack's own motion. Split rather than summed because the tyre term is what the car is doing
    // and the other two are what the steering box is doing, and confusing the two is how a rack's
    // friction ends up being tuned to fix a tyre.
    double tyreForce = 0.0;
    double frictionForce = 0.0;
    double dampingForce = 0.0;
    double rackForce = 0.0;

    // **The deliverable, and it is the *unassisted* one.** Newton metres at the pinion, which on a
    // rack and pinion is newton metres at the rim. Positive in the same sense as a positive
    // steering demand, so a self-aligning tyre reports the opposite sign to the lock it is under.
    //
    // This is the number that is comparable across cars, sessions and hardware, and the one
    // criterion 10 is checked against — no motor, no device, nothing but the road and the geometry.
    double steeringTorque = 0.0;

    // What this car's own electric rack contributed, newtons, and what is therefore left for the
    // driver's hands. Zero on a car with no assist, in which case the two torques below are equal.
    double assistForce = 0.0;
    double driverRackForce = 0.0;

    // **What the driver actually feels**, N·m at the rim: the deliverable above with this car's own
    // power steering between it and their hands. This is what goes to a wheel base; the one above
    // is what goes in the physics trace beside it.
    //
    // Keeping them apart is the point. Conflated into one field they were invisible for a whole
    // milestone, and the trace column documented as the hardware-independent artefact was carrying
    // a number that had been through a hardware-sized multiplier.
    double assistedTorque = 0.0;

    // False when anything upstream handed this a value that is not a number. The caller must treat
    // that as a reason to let go of the wheel rather than as a number to pass on.
    bool finite = true;
};

// The whole derivation, and it is two cross products and a ratio per corner.
//
// The tyre's resultant acts at the contact patch. Its moment about the kingpin axis is what the
// steering has to hold, and taking it as a moment about the *solved* axis is what makes mechanical
// trail, scrub radius, caster and kingpin inclination all arrive without any of them being written
// down: they are the geometry, and the geometry is an input.
//
//     T = k . [ (patch - lowerBallJoint) x F + Mz * n ]
//
// The tie rod then converts that to a force at the rack. It is a two-force member, so its force is
// along its own line; write u for that line, and the rack's own axis is chassis +x. A virtual
// displacement of the rack must keep the tie rod's length, which gives
//
//     dx * (u . x)  =  dtheta * ( k . ((arm - lowerBallJoint) x u) )
//
// so the ratio of those two brackets is dtheta/dx exactly, and virtual work turns the kingpin
// moment into a rack force with it. No small-angle approximation, no authored steering ratio and no
// lever arm measured off a drawing: change a hardpoint and this changes with it, which is the same
// rule the suspension solve keeps.
//
// `roadSpeed` is metres per second and is the car's, not the rack's: it is the second input the
// power assist's boost curve is drawn against, and it reaches nothing else here. Left at its default
// the whole derivation is exactly what it was before the assist existed.
export [[nodiscard]] RackTorque steeringRackTorque(const SteeringRack& rack,
                                                   const std::span<const SteeredCorner> corners,
                                                   const double rackVelocity, const double roadSpeed = 0.0);

// What the car was doing on the tick that produced the torque beside it.
//
// **This is a projection of the physics module's `TelemetryFrame`, not a second computation of
// anything**, and that sentence is the whole design. Nothing in this partition imports the physics
// module — see the note at the top of the file — so the caller extracts, exactly as it already does
// for `SteeredCorner` and `SlippingWheel`. What that buys is that the heading-invariance gate over
// `TelemetryFrame` covers these columns too: there is one place each number is worked out, and this
// is a copy of it rather than a rival to it.
//
// Why it is here at all: the first seat session measured the assist ratio bottoming out at 0.29 in
// the 10-20 N·m band, and a rack-only trace cannot say where the front grip limit falls in rack
// torque. If the limit is inside that band the taper is aimed too high. `aligningMoment` against
// `slipAngle` is the cue that settles it, and it has to be on the same time base as the rack to be
// read against it.
//
// SI throughout, like everything else in this partition. The CSV converts at its boundary.
export inline constexpr std::size_t tracedCornerCount = 4;

// FL, FR, RL, RR, and **that order is load-bearing**: it is `Corner`'s own order in the physics
// module, the array is filled by index and read by index, and the CSV's header loop walks the same
// list. A column that carried its neighbour's corner is the fault this ordering exists to make
// impossible, and `RackTraceCornerTests` asserts it per corner and per channel rather than trusting
// the header.
export inline constexpr std::array<const char*, tracedCornerCount> tracedCornerAbbreviations{"FL", "FR", "RL", "RR"};

export struct WheelTrace
{
    double slipAngle = 0.0;
    double slipRatio = 0.0;

    // The tyre's three forces in its own patch frame, and its aligning moment. `aligningMoment` is
    // the one the exercise is for: it falls away past the peak while lateral force is still near it,
    // and that fall *is* the limit cue the driver feels through the rack.
    double verticalLoad = 0.0;
    double lateralForce = 0.0;
    double longitudinalForce = 0.0;
    double aligningMoment = 0.0;

    double suspensionTravel = 0.0;
    double damperVelocity = 0.0;

    // How many of the patch grid's samples are touching. One integer that turns the enveloping
    // question from a scoping judgement into a measurement — see `WheelTelemetry::contactingSamples`
    // for what it is being asked.
    std::uint32_t contactingSamples = 0;

    // Deepest minus shallowest contacting depth across the patch grid, metres. It rides beside the
    // count because the count alone cannot separate a loaded wheel on a kerb edge from a lightly
    // loaded one on flat road, and the enveloping model is chosen against the distribution of the
    // pair over a real lap — see `WheelTelemetry::patchDepthSpread`.
    double patchDepthSpread = 0.0;

    // --- what the electronics were doing at this wheel ---
    //
    // **Added 2026-08-23 because a trace could not answer the question it was being asked.** Working
    // out whether a lap had been driven with the anti-lock system on meant going and reading the
    // setup sheet, which by then said something else. A trace has to be self-describing: the state of
    // the systems on the tick belongs beside the tyre they acted on.
    bool antilockActive = false;
    // Cumulative dumps on this wheel's channel. Differenced across a window it is the engagement
    // rate; as a level it says whether the unit did anything at all on this lap.
    std::uint32_t antilockCycles = 0;
    // What the caliper is actually holding, pascals — the modulator's own state, and the channel that
    // separates "the driver asked for full pressure" from "the wheel got it".
    double brakePressure = 0.0;

    // The tread **core**'s temperature, degrees Celsius, joined 2026-08-28 with the thermal tyre.
    //
    // One column and not the three the telemetry CSV carries, because this file is the *seat's*
    // artefact rather than the physics ledger: the core is the layer grip reads, so it is the one a
    // report like "it went away after three laps" has to be read against. It reads the seed
    // temperature on a car with `tyreThermal` off, which is the value the rest of the model is
    // assuming rather than a measurement — and that is worth having in the trace too, because the
    // question a trace has to answer first is what the car was set up as.
    double treadCoreTemperature = 0.0;

    // And the brake disc's, degrees Celsius, joined 2026-08-28 with the fade model. It belongs on the
    // seat's artefact more than most channels do: "the pedal went long at the end of the lap" is a
    // report this column either confirms or refutes in one glance.
    double discTemperature = 0.0;

    // And the wheel's, joined with stage 3. It earns a column because it is the *answer* to stage 3
    // rather than an input to it: a trace showing a disc at 500 °C, a rim at 90 and a tread at 50
    // says in three numbers how much of a brake's heat a wheel lets past, which is the whole
    // question. docs/brake-thermal-brief.md.
    double wheelTemperature = 0.0;
};

export struct VehicleTrace
{
    // World frame, metres, and the centre of mass rather than the model origin. Present so a corner
    // can be found on the circuit without a second file to line up against.
    double centreOfMassX = 0.0;
    double centreOfMassZ = 0.0;

    double heading = 0.0;
    double speed = 0.0;

    // The car's own frame, m/s^2. Long is +z and lat is +x, the same roles `TelemetryFrame` states
    // them in and the same ones that were carrying world-frame components until 2026-08-21.
    double lateralAcceleration = 0.0;
    double longitudinalAcceleration = 0.0;
    double yawRate = 0.0;

    double rideHeightFront = 0.0;
    double rideHeightRear = 0.0;

    double throttle = 0.0;
    double brake = 0.0;
    double clutch = 0.0;
    std::int32_t gear = 0;
    double engineSpeed = 0.0;

    // --- which electronics were fitted, and what the car-wide ones were doing ---
    //
    // **`antilockEnabled` and `tractionMode` are constant for a run and are written every row on
    // purpose.** They are what makes a lap self-describing: two traces taken ten minutes apart
    // differed only in a setup-sheet line, and telling them apart meant reading a file that had since
    // been edited. A column that never changes is cheap; a trace nobody can date is not.
    //
    // **"Constant for a run" was a promise the writer did not keep until 2026-08-28**, and it cost a
    // wrong reading of a seat trace the same day. The car publishes its first frames before the setup
    // sheet has landed on it, so these read the *factory* configuration — everything off — for about
    // four ticks and the session's own from the fifth. A reader who samples row 0, which is the
    // obvious thing to do with a documented constant, is told the car had no electronics. The writer
    // now stamps the settled value across every row; see `rackTorqueToCsv`.
    //
    // `tractionMode` is 0 off, 1 full, 2 sport — an integer rather than the enum because this module
    // is `raceengine.input` and naming `TractionMode` here would couple the trace format to the
    // assist layer for one column.
    bool antilockEnabled = false;
    std::uint32_t tractionMode = 0;

    bool tractionBrakeActive = false;
    bool tractionEngineActive = false;

    // **Fitted and active are two different questions and the trace needs both.** `corneringActive`
    // is only true while XDS is applying a brake, so a lap driven with it switched on but never
    // triggered reads exactly like a lap driven with it off — which is the confusion `ABS Fitted`
    // and `TC Mode` were added to end for the other two, and XDS was left without its half of it.
    // Found the hard way on 2026-08-23: a trace with 0 in the active column could not be told from
    // one where the assist was never enabled, and the answer needed the setup sheet, which had been
    // edited since.
    bool corneringEnabled = false;
    bool corneringActive = false;
    // 0 with the driver's foot untouched, 1 with the throttle shut.
    double engineTorqueReduction = 0.0;

    std::array<WheelTrace, tracedCornerCount> wheels{};
};

// The stage-one trace, and it is the artefact rather than instrumentation.
//
// It is in newton metres at the rim whatever is plugged in, so a run on this base, a run on a
// direct drive base and a run with nothing attached at all produce the same numbers for the same
// lap. The stage-two columns ride beside it because a clip is only legible against what was asked
// for, but the stage-one column is the one that is comparable across hardware and sessions, and it
// is written whether or not any device took it.
export struct RackTorqueFrame
{
    double time = 0.0;
    std::uint64_t sequence = 0;

    // --- stage one, always, in N·m at the rim ---
    //
    // **`steeringTorque` is the unassisted rack**, which is what this column has always been
    // documented as and, until 2026-08-21, was not: the assist was applied a layer out and every
    // channel here carried it. That is what made a car parameter sized against a wheel base
    // invisible for a milestone, so the two are separate fields now and the CSV names them apart.
    double steeringTorque = 0.0;
    // The same tick with this car's own power steering between it and the driver's hands. This is
    // what stage two was given; the one above is what the road did.
    double assistedTorque = 0.0;
    double rackForce = 0.0;
    double tyreRackForce = 0.0;
    double rackTravel = 0.0;
    double rackVelocity = 0.0;

    // --- stage two, for this device ---
    double requestedTorque = 0.0;
    double commandedTorque = 0.0;
    double deliveredTorque = 0.0;
    bool clipped = false;

    // End to end, device report to the write returning, in milliseconds. Zero when nothing has been
    // written yet.
    double latencyMilliseconds = 0.0;

    // --- the car, on the tick that produced the columns above ---
    VehicleTrace vehicle{};
};

// The same ring the physics telemetry uses, and for the same reason: a validation run is minutes
// long and the interesting part is at the end, and a recorder that grew would allocate on the
// output thread.
export class RackTorqueRecorder
{
public:
    explicit RackTorqueRecorder(const std::size_t capacity) :
        frames(capacity)
    {
    }

    void record(const RackTorqueFrame& frame)
    {
        if (frames.empty())
        {
            return;
        }

        frames[next] = frame;
        next = (next + 1) % frames.size();
        filled = filled < frames.size() ? filled + 1 : frames.size();
    }

    // Oldest first, which is the opposite of storage order once the ring has wrapped.
    [[nodiscard]] std::vector<RackTorqueFrame> inOrder() const
    {
        auto ordered = std::vector<RackTorqueFrame>{};
        ordered.reserve(filled);

        const auto start = filled < frames.size() ? std::size_t{0} : next;
        for (auto offset = std::size_t{0}; offset < filled; offset++)
        {
            ordered.push_back(frames[(start + offset) % frames.size()]);
        }

        return ordered;
    }

    [[nodiscard]] std::size_t size() const
    {
        return filled;
    }

    [[nodiscard]] std::size_t capacity() const
    {
        return frames.size();
    }

private:
    std::vector<RackTorqueFrame> frames;
    std::size_t next = 0;
    std::size_t filled = 0;
};

// Pure, like `telemetryToCsv` and for the same two reasons: what is in the text is worth testing and
// opening a file is not, and `std::to_chars` rather than a stream because a CSV written on a machine
// with a comma decimal separator is not a CSV.
export [[nodiscard]] inline std::string rackTorqueToCsv(const std::vector<RackTorqueFrame>& frames)
{
    auto text = std::string{};
    // Sixty-three columns now rather than thirteen, so the row estimate goes up with them. A session
    // is tens of megabytes of text and getting this wrong costs a handful of reallocations of a
    // string that large, which is the one part of writing the file worth a thought.
    text.reserve(frames.size() * 640 + 2048);

    // **Which electronics this run was driven with, taken from the last frame and written on every
    // row.** These three columns document the *session*, and the car publishes about four frames
    // before its setup sheet lands on it — so the head of the file otherwise reports the factory
    // configuration, which is everything off. It read as a lap driven with no ABS on 2026-08-28 and
    // was only caught because the driver knew otherwise. The last frame is the settled answer: a
    // sheet reloaded mid-session states what the car finished under, which is the same rule a
    // reloaded sheet already follows everywhere else.
    const auto fitted = frames.empty() ? VehicleTrace{} : frames.back().vehicle;

    text += "Time [s],Sequence,"
            "Steering Torque [Nm],Assisted Torque [Nm],Rack Force [N],Tyre Rack Force [N],"
            "Rack Travel [mm],Rack Vel [mm/s],"
            "Requested Torque [Nm],Commanded Torque [Nm],Delivered Torque [Nm],Clipped [],Latency [ms],"
            "CoG X [m],CoG Z [m],Heading [deg],Speed [kph],"
            "G Force Lat [g],G Force Long [g],Yaw Rate [deg/s],"
            "Ride Height F [mm],Ride Height R [mm],"
            "Throttle Pos [%],Brake Pos [%],Clutch Pos [%],Gear [],Engine RPM [rpm],"
            "ABS Fitted [],TC Mode [],TC Brake [],TC Engine [],XDS Fitted [],XDS Active [],Engine Reduction [%]";

    for (const auto* tag : tracedCornerAbbreviations)
    {
        const auto corner = std::string(" ") + tag;

        text += ",Slip Angle" + corner + " [deg]";
        text += ",Slip Ratio" + corner + " []";
        text += ",Tyre Fz" + corner + " [N]";
        text += ",Tyre Fy" + corner + " [N]";
        text += ",Tyre Fx" + corner + " [N]";
        text += ",Tyre Mz" + corner + " [Nm]";
        text += ",Susp Pos" + corner + " [mm]";
        text += ",Damper Vel" + corner + " [mm/s]";
        text += ",Contact Samples" + corner + " []";
        text += ",Patch Depth Spread" + corner + " [mm]";
        text += ",ABS Active" + corner + " []";
        text += ",ABS Cycles" + corner + " []";
        text += ",Brake Pressure" + corner + " [bar]";
        text += ",Tyre Temp Core" + corner + " [C]";
        text += ",Disc Temp" + corner + " [C]";
        text += ",Wheel Temp" + corner + " [C]";
    }

    text += "\n";

    for (const auto& frame : frames)
    {
        appendRackNumber(text, frame.time, 6);
        text += ",";

        auto buffer = std::array<char, 32>{};
        const auto written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), frame.sequence);
        text.append(buffer.data(), written.ptr);

        for (const auto value : {frame.steeringTorque, frame.assistedTorque, frame.rackForce, frame.tyreRackForce})
        {
            text += ",";
            appendRackNumber(text, value, 4);
        }

        for (const auto value : {frame.rackTravel * 1000.0, frame.rackVelocity * 1000.0})
        {
            text += ",";
            appendRackNumber(text, value, 3);
        }

        for (const auto value : {frame.requestedTorque, frame.commandedTorque, frame.deliveredTorque})
        {
            text += ",";
            appendRackNumber(text, value, 4);
        }

        text += frame.clipped ? ",1," : ",0,";
        appendRackNumber(text, frame.latencyMilliseconds, 3);

        const auto& car = frame.vehicle;

        for (const auto value : {car.centreOfMassX, car.centreOfMassZ})
        {
            text += ",";
            appendRackNumber(text, value, 4);
        }

        text += ",";
        appendRackNumber(text, car.heading * rackRadiansToDegrees, 4);
        text += ",";
        appendRackNumber(text, car.speed * rackMetresPerSecondToKilometresPerHour, 4);

        for (const auto value : {car.lateralAcceleration, car.longitudinalAcceleration})
        {
            text += ",";
            appendRackNumber(text, value / rackGravity, 5);
        }

        text += ",";
        appendRackNumber(text, car.yawRate * rackRadiansToDegrees, 4);

        for (const auto value : {car.rideHeightFront, car.rideHeightRear})
        {
            text += ",";
            appendRackNumber(text, value * 1000.0, 3);
        }

        for (const auto value : {car.throttle, car.brake, car.clutch})
        {
            text += ",";
            appendRackNumber(text, value * 100.0, 3);
        }

        text += ",";
        appendRackInteger(text, car.gear);
        text += ",";
        appendRackNumber(text, car.engineSpeed * rackRadiansPerSecondToRevolutionsPerMinute, 1);

        // **The settled configuration and not this frame's**, which is what these three columns have
        // always claimed to be. See `RackTorqueVehicle` for the four ticks that made the claim false
        // and the seat trace it misread. The per-tick answers live in `tractionBrakeActive`,
        // `tractionEngineActive`, `corneringActive` and the per-corner `ABS Active`, and those are
        // untouched — fitted and active are two different questions and this only settles the first.
        text += fitted.antilockEnabled ? ",1," : ",0,";
        appendRackInteger(text, static_cast<long long>(fitted.tractionMode));
        text += car.tractionBrakeActive ? ",1" : ",0";
        text += car.tractionEngineActive ? ",1" : ",0";
        text += fitted.corneringEnabled ? ",1" : ",0";
        text += car.corneringActive ? ",1," : ",0,";
        appendRackNumber(text, car.engineTorqueReduction * 100.0, 3);

        // By index, in the order the header was written. Nothing here names a corner: the loop and
        // the header loop walk the same array, so a mapping fault would have to be a fault in the
        // array itself rather than in one of thirty-six hand-written lines.
        for (const auto& wheel : car.wheels)
        {
            text += ",";
            appendRackNumber(text, wheel.slipAngle * rackRadiansToDegrees, 4);
            text += ",";
            appendRackNumber(text, wheel.slipRatio, 5);

            for (const auto value :
                 {wheel.verticalLoad, wheel.lateralForce, wheel.longitudinalForce, wheel.aligningMoment})
            {
                text += ",";
                appendRackNumber(text, value, 3);
            }

            for (const auto value : {wheel.suspensionTravel, wheel.damperVelocity})
            {
                text += ",";
                appendRackNumber(text, value * 1000.0, 3);
            }

            text += ",";
            appendRackInteger(text, static_cast<long long>(wheel.contactingSamples));
            text += ",";
            appendRackNumber(text, wheel.patchDepthSpread * 1000.0, 3);

            text += wheel.antilockActive ? ",1," : ",0,";
            appendRackInteger(text, static_cast<long long>(wheel.antilockCycles));
            text += ",";
            // Bar, like every other pressure a person reads. The model carries pascals.
            appendRackNumber(text, wheel.brakePressure / 1.0e5, 3);
            text += ",";
            appendRackNumber(text, wheel.treadCoreTemperature, 2);
            text += ",";
            appendRackNumber(text, wheel.discTemperature, 2);
            text += ",";
            appendRackNumber(text, wheel.wheelTemperature, 2);
        }

        text += "\n";
    }

    return text;
}

} // namespace raceengine
