module;

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:PublishedCars;

import :Brakes;
import :Contact;
import :ContactPatch;
import :Coupling;
import :Driveline;
import :RigidBody;
import :Suspension;
import :Telemetry;
import :Tyre;
import :Vehicle;

namespace raceengine
{

// A real car, from its own published physics data rather than from plausible invention.
//
// The source is a publicly distributed Assetto Corsa model of a Volkswagen Golf GTI Mk7.5, unpacked
// and parsed by `scripts/import-ac-car.py`. What is here is what that data states, or arithmetic on
// what it states; where AC does not model something this engine does, the number is marked and the
// reason is given. Nothing in this file is a number somebody thought looked about right, and the
// distinction from `placeholderSedan()` is exactly that — the placeholder is a *fixture*, calibrated
// so the model can be validated against a car whose every figure is chosen, and it stays.
//
// **The conversion of the pickup points is verified rather than assumed.** AC states them relative
// to the wheel centre, x measured inboard from the wheel's centre plane, y up and z forward:
//
//     x = track / 2 - x_ac
//     y = y_ac + tyreRadius        (so the design contact patch sits on y = 0)
//     z = z_ac + axleStation
//
// Converting with that and reading the track back off the resulting wheel centre reproduces the
// TRACK the same file states, on both axles, to four decimal places. Any other convention misses by
// tens of centimetres, which is why that check is a test rather than a comment.
//
// The model frame is the engine's: **+x is the car's left** (`outboardSign`), +y up, +z forward,
// origin midway between the axles at
// road level. AC's own frame has its origin at the centre of gravity, so everything AC states in it
// — aero surfaces, the collider — is carried across by the centre of gravity's position below.

// suspensions.ini [BASIC] WHEELBASE, and the axles placed symmetrically about the origin.
inline constexpr auto golfWheelbase = 2.638;
inline constexpr auto golfFrontAxle = 0.5 * golfWheelbase;
inline constexpr auto golfRearAxle = -0.5 * golfWheelbase;

// suspensions.ini [FRONT]/[REAR] TRACK.
inline constexpr auto golfFrontTrack = 1.539;
inline constexpr auto golfRearTrack = 1.516;

// **Not tyres.ini RADIUS, and this is the first number in this file taken away from the mod.**
//
// The file states 0.3298 m, which is 3.5% larger than any factory fitment for this car: a 225/40 R18
// is 0.3186 m unloaded (18 × 25.4 / 2 + 225 × 0.40, in millimetres) and the Performance's 225/35 R19
// is 0.3200. The import is faithful — nothing in `scripts/import-ac-car.py` is wrong — the source
// disagrees with the car, so the published fitment wins and the disagreement is written down here
// rather than absorbed.
//
// **It is a uniform scale factor, which is exactly what shape-based validation cannot see.** Road
// speed, the effective final drive, every slip ratio and the rolling-resistance lever all move by
// the same 3.5%, so every monotonicity, every ordering and every ratio the suite asserts is
// preserved by construction. The two things that do move are absolute: what the speedometer says,
// and what the coastdown fit reports — and a two-parameter fit will happily absorb a 3.5% lever
// error into `Crr` and `CdA` and still look agreeable, which is why the pre-correction figures
// coming out near their targets was never evidence the radius was right.
//
// One consequence worth stating because it is not obvious: this is also every corner's pickup-point
// datum. The conversion at the top of this file puts the design contact patch on y = 0 by adding the
// radius to AC's wheel-centre-relative y, so correcting the radius lowers every hardpoint by 11.2 mm
// relative to the road and leaves the *suspension* geometry — the relative positions the linkage is
// made of — untouched. That is the right behaviour: the linkage is what AC states, and where the
// road is relative to it is what the tyre states.
inline constexpr auto golfTyreRadius = 0.3186;

// **The ninth number taken away from the mod** (2026-08-24), and it is the mass of the whole car.
//
// `car.ini [BASIC] TOTALMASS` states **1348 kg**. The manufacturer's tare mass — the vehicle with
// standard equipment and unoccupied — is **1377 kg**, and a car being driven has a driver in it. EC
// 1230/2012's "weight in running order" is the convention that adds one and it adds exactly 75 kg, so
// what is actually being accelerated is **1452 kg** and the model was **7.2% light**.
//
// **Whether AC's figure includes a driver could not be resolved and does not need to be.** The claim
// that `TOTALMASS` carries one is community lore; the primary source for it says "speculated" in as
// many words. Either reading gives the same shortfall in the total that accelerates: excluding a
// driver the car is 29 kg under tare *and* has nobody in it, and including one the car itself is
// 104 kg under tare. The number below is derived from the manufacturer's figure and the regulation's
// occupant, so it does not depend on settling the mod's convention.
//
// It matters as much as the torque curve does and that is measured rather than asserted: elapsed time
// in an in-gear pull moves **1.00% per 1% of mass** against 1.18% per 1% of `T·η`, which is why a
// mass error hides so well behind an engine that looks about right.
// `docs/engine-curve-validation-brief.md`.
inline constexpr auto golfTareMass = 1377.0;
inline constexpr auto golfDriverMass = 75.0;
inline constexpr auto golfTotalMass = golfTareMass + golfDriverMass;

// **Not suspensions.ini HUB_MASS, and this is the second number taken away from the mod.**
//
// The file states 80 kg per front corner and 85 per rear — 330 kg, a quarter of the whole car, on a
// hatchback. It is not a plausible figure for any road car and it was invisible in every total
// because the sprung side is *solved* from it (below), so `TOTALMASS` came out right whatever this
// said. What it moves is the load path: wheel hop goes as sqrt(k_tyre / m_unsprung), and unsprung
// mass transfers load directly rather than through the springs.
//
// **There is no published per-corner figure for this car**, so what replaces it is a component
// build-up rather than a citation, and the lines are written out so the number can be argued with
// rather than only accepted. Published breakdowns for a passenger-car corner put wheel at 9-13 kg,
// tyre at 10-14, the brake assembly at 8-12 and the outboard suspension share at 8-11, for 35-45 kg
// all in; this car sits at the top of that because it is the 245 PS car on 18s.
//
//     front, MacPherson strut, driven          rear, multi-link, undriven
//     ---------------------------------        --------------------------------
//     225/40 R18 tyre            10.0          225/40 R18 tyre            10.0
//     18 x 7.5J cast alloy       11.0          18 x 7.5J cast alloy       11.0
//     340 x 30 vented disc        9.5          310 x 22 vented disc        6.5
//     caliper and carrier         4.5          caliper, carrier, EPB       5.0
//     hub and bearing             2.5          hub and bearing             2.5
//     steering knuckle            3.5          wheel carrier               3.5
//     outboard share of arms      3.0          outboard share of links     3.5
//     outboard half of driveshaft 3.0          --
//     strut's unsprung share      2.5          damper and spring share     1.5
//     ---------------------------------        --------------------------------
//     total                      49.5          total                      43.5
//
// Rounded to the precision the build-up deserves, which is not three figures. The front carries the
// larger brake and a driveshaft the rear does not have, which is why the two differ — and it is the
// opposite ordering to the mod's, whose rear was the heavier of the two.
inline constexpr auto golfFrontHubMass = 49.0;
inline constexpr auto golfRearHubMass = 43.0;

// Front: MacPherson strut, from suspensions.ini [FRONT]. The strut's top bearing is STRUT_CAR and
// its lower end is the lower ball joint, which is WBTYRE_BOTTOM — the same point STRUT_TYRE names,
// and the two agreeing is the file's own statement that the strut picks up on the ball joint.
export [[nodiscard]] CornerHardpoints golfMk7FrontCorner(const CornerSide side);

// Rear: double wishbone, from suspensions.ini [REAR].
//
// A production Mk7.5 has a multi-link rear rather than a wishbone; the model approximates it as one,
// which is why the lower arm's pivot axis is 0.74 m long — that is a trailing arm wearing a
// wishbone's clothes. The curves it produces are a road car's, but it is a model of a model.
export [[nodiscard]] CornerHardpoints golfMk7RearCorner(const CornerSide side);

// The Golf's brakes, as the parts on the car rather than as `brakes.ini`'s scalar.
//
// **This is the sixth number taken away from the mod** (2026-08-23), after the wheel radius, the
// unsprung mass, `CG_LOCATION`, the tyre peaks and `MAX_TORQUE` itself — and it is the one that
// deletes a number rather than replacing it, because nothing below was chosen. `MAX_TORQUE` and
// `FRONT_SHARE` are both gone: the torque falls out of the piston, the pad and the disc, and the
// share falls out of the two of them against one line pressure. Every part is sourced in
// `PublishedCarsImpl.cpp` and the two that had to be estimated are marked there.
//
// The car is the **Performance** variant, which is what the mod is, so these are the Performance
// Pack's brakes: 340 x 30 vented front, 310 x 22 vented rear, single-piston sliding calipers on both
// axles.
export [[nodiscard]] BrakeHardware golfMk7FrontBrake();
export [[nodiscard]] BrakeHardware golfMk7RearBrake();

// And the one hydraulic system that feeds them. Per car, which is the whole reason the front/rear
// split is a property of the calipers and not a setting — and, with the valve below, the reason it
// is not a *constant* either.
export [[nodiscard]] BrakeHydraulics golfMk7Hydraulics();

// The rear circuit's proportioning valve, which is the second half of the bias: the calipers make
// 0.686 out of one pressure, and the pressure the rear calipers see is not the one the fronts do.
// Derived from this car's own two lock pressures — see `PublishedCarsImpl.cpp`.
export [[nodiscard]] ProportioningValve golfMk7RearProportioningValve();

// The Golf GTI Mk7.5, assembled. Fallible for the same reasons `placeholderSedan` is: every corner's
// geometry is swept and its setup validated before the car is handed back, so a bad number is an
// error at load time rather than a car that flies away on the tick that first reaches it.
export [[nodiscard]] std::expected<VehicleSetup, std::string> golfGtiMk7();

// The same car's driveline, from engine.ini, drivetrain.ini and the turbocharged torque curve the
// extractor computed off power.lut.
//
// **power.lut is torque in newton metres and it is the naturally aspirated curve**, which AC layers
// boost onto per turbo as min(wastegate, maxBoost * (rpm/reference)^gamma). Read as final it reports
// 159 N.m and 110 bhp for a car that makes 350 and 243 — a wrong number that looks like a right one,
// which is the only kind worth guarding against. The points below are the boosted curve.
export [[nodiscard]] DrivelineSetup golfGtiMk7Driveline();

// The same car's electronics, calibrated against the car that was just built rather than against a
// second copy of its numbers: brake peaks come off `setup.corners[i].brakeTorque`, the tone rings
// and the reference radius off the wheel, the tracks off the geometry.
//
// **Everything is switched off.** A Mk7 GTI leaves the factory with all of it switched on, and this
// does not, deliberately: the suite's job is to validate the car underneath the electronics, and a
// default that quietly assisted every fixture would be a suite that could no longer see the car at
// all. A game or a test that wants the systems says so, once, at the call.
export [[nodiscard]] AssistSetup golfGtiMk7Assists(const VehicleSetup& setup);

} // namespace raceengine
