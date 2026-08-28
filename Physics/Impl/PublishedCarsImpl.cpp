// PublishedCars bodies. Declarations are in Api/PublishedCars.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] CornerHardpoints golfMk7FrontCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * (0.5 * golfFrontTrack - x), y + golfTyreRadius, z + golfFrontAxle);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;
    corner.kind = SuspensionKind::MacPhersonStrut;

    corner.lower = Wishbone{.frontPivot = at(0.40769, -0.10955, 0.01423),
                            .rearPivot = at(0.40944, -0.12017, -0.27373),
                            .ballJoint = at(0.07174, -0.13710, 0.01213)};

    // **The ninth number taken away from the mod** (2026-08-26): the strut top's longitudinal
    // position. As imported (`-0.04296`) the kingpin leans back only 4.59 deg where a Mk7 is
    // published near 7.5 deg of caster — the discrepancy step 1 of the suspension audit recorded
    // and step 15 sized (docs/suspension-geometry-audit.md). The lateral and vertical positions
    // are the file's own, so KPI and scrub are untouched; the longitudinal position is **solved**
    // from the held ball joint and the published caster rather than typed, which is why the z
    // lands in chassis space below instead of carrying a literal that would go stale if the ball
    // joint ever moved. Mechanical trail rises 26.7 -> 36.0 mm with it, which is the whole of the
    // steering-feel change and is awaiting its seat session.
    corner.strutTop = at(0.25366, 0.54933, 0.0);
    corner.strutTop.z =
        corner.lower.ballJoint.z - (corner.strutTop.y - corner.lower.ballJoint.y) * std::tan(glm::radians(7.5));
    corner.steeringRackOuter = at(0.31000, -0.06546, -0.22200);
    corner.steeringArm = at(0.05200, -0.06757, -0.12586);
    corner.wheelCentre = at(0.0, 0.0, 0.0);
    corner.wheelRadius = golfTyreRadius;

    corner.droopAngle = -0.16;
    corner.bumpAngle = 0.16;

    return corner;
}

[[nodiscard]] CornerHardpoints golfMk7RearCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * (0.5 * golfRearTrack - x), y + golfTyreRadius, z + golfRearAxle);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;

    corner.lower = Wishbone{.frontPivot = at(0.17702, -0.11686, 0.50420),
                            .rearPivot = at(0.57702, -0.11686, -0.11540),
                            .ballJoint = at(0.04602, -0.12706, 0.01870)};
    corner.upper = Wishbone{.frontPivot = at(0.34842, 0.09370, 0.22140),
                            .rearPivot = at(0.39800, 0.08540, 0.03080),
                            .ballJoint = at(0.06622, 0.10070, -0.00650)};
    corner.steeringRackOuter = at(0.58002, -0.09806, -0.10540);
    corner.steeringArm = at(0.07202, -0.10406, -0.09540);
    corner.wheelCentre = at(0.0, 0.0, 0.0);
    corner.wheelRadius = golfTyreRadius;

    // **AC states no damper pickup points for a double wishbone**, because it states spring and
    // damper rates at the wheel and never needs them. Left defaulted, both mounts land on the origin
    // and the motion ratio that comes out is a plausible-looking number describing a linkage that
    // does not exist — `validateCornerSetup` refuses exactly that, and `PublishedGeometryTests` pins
    // the refusal.
    //
    // So this is the one piece of geometry here that is not from the file, and it is placed rather
    // than fitted: the Mk7's rear damper picks up on the wheel carrier and stands very nearly
    // upright, so the lower eye goes on the ball joint and the upper mount directly above it. That
    // arrangement has a motion ratio of about one, which is what makes AC's at-the-wheel rates pass
    // through the conversion below essentially unchanged — the least the choice can be made to
    // matter. Only the *length* is free, and nothing reads it but the spring's rest length.
    corner.damperWishbone = corner.lower.ballJoint;
    corner.damperChassis = corner.lower.ballJoint + glm::dvec3(0.0, 0.40, 0.0);

    corner.droopAngle = -0.14;
    corner.bumpAngle = 0.14;

    return corner;
}

namespace
{

inline constexpr auto standardGravity = 9.80665;
inline constexpr auto rpmToRadiansPerSecond = 0.10471975511965977;

// The rack travel that steers the front wheels to the lock the car states, **signed**. AC gives the
// steering wheel's own lock and the ratio between the two, so the road wheel's angle is data; the
// travel that produces it is a property of this linkage and is solved for rather than guessed.
// Monotonic in the magnitude over the range a rack has, so a bisection is enough and cannot pick a
// second root.
//
// The sign is the half that used to be missing, and it was missing invisibly. The bisection below
// searches positive travel against `std::abs(toe)`, so it can only ever answer *how far*; which way
// was then taken from `copysign(..., steerRatio)` — the sign of a steering *ratio*, which is a
// different quantity that happens to carry one. On this car that produced a rack that steered left
// when the driver asked for right, on both the wheel and the keyboard, and nothing downstream could
// tell: every stage from the kinematic solve to the force feedback was faithfully consistent with a
// rack going the wrong way.
//
// Taken from the linkage instead, and **read off the wheel's own nose rather than off its toe**.
// Toe is documented as positive when the front of the wheel points toward the car on both sides,
// which makes it a mirrored quantity and therefore the worst possible thing to read a left-or-right
// answer from: two separate attempts reasoned from it and both came out inverted. The second was
// worse than the first, because it was "calibrated" against a chassis yaw rate whose own sign was
// read in a frame nobody had pinned down — the answer looked measured and was not.
//
// What it reads now is where the nose of the wheel moves, against `outboardSign`, which is the one
// place the frame's handedness is stated and the only thing here that has been checked against a
// picture. No mirrored quantity is involved and no car-specific assumption either: a rack behind the
// axle steers the opposite way to one ahead of it, and this measures whichever this car has.
//
// The magnitude is searched **along the direction that steers right** rather than along +x. Rack
// travel is applied toward +x on both corners, so it is outboard on one and inboard on the other,
// and a linkage that reaches full lock one way need not reach it the other within a rack's travel —
// searching the wrong way is an honest-looking "cannot reach the lock the car states".
[[nodiscard]] std::expected<double, std::string> rackTravelForSteer(const CornerHardpoints& hardpoints,
                                                                    const double roadWheelAngle)
{
    // Where this wheel is pointing, as a direction rather than as an angle with a convention on it.
    const auto noseAt = [&hardpoints](const double travel) -> std::expected<glm::dvec3, std::string>
    {
        const auto solved = solveCorner(hardpoints, 0.0, travel);
        if (!solved)
        {
            return std::unexpected(solved.error());
        }

        return solved->uprightOrientation * glm::dvec3(0.0, 0.0, 1.0);
    };

    // Small enough to be unambiguously in the linear part of the linkage, large enough to be clear
    // of the solve's own noise.
    constexpr auto probeTravel = 0.01;

    const auto centred = noseAt(0.0);
    const auto probed = noseAt(probeTravel);
    if (!centred || !probed)
    {
        return std::unexpected("the steering linkage will not solve at a small rack travel: " +
                               (centred ? probed.error() : centred.error()));
    }

    // The car's right, from the one function allowed to know which way that is.
    const auto towardRight = outboardSign(CornerSide::Right);
    const auto direction = (probed->x - centred->x) * towardRight > 0.0 ? 1.0 : -1.0;

    auto low = 0.0;

    const auto steerAt = [&hardpoints, direction](const double travel) -> std::expected<double, std::string>
    {
        const auto solved = solveCorner(hardpoints, 0.0, direction * travel);
        if (!solved)
        {
            return std::unexpected(solved.error());
        }

        return std::abs(solved->toe);
    };

    const auto steersRightForPositiveTravel = direction > 0.0;

    // The bracket is **found rather than assumed**, and 0.15 m is past the end of this linkage. A
    // steering arm offset from the kingpin reaches further one way than the other — the Golf's front
    // corner solves to about 110 mm of rack in one direction and 90 in the other — so a fixed upper
    // bracket is a geometry question dressed as a constant, and it answers "the rack cannot reach
    // the lock the car states" for a car whose lock is at 70 mm and perfectly reachable.
    auto high = 0.15;
    while (high > 0.01 && !steerAt(high))
    {
        high *= 0.5;
    }

    const auto reach = steerAt(high);
    if (!reach)
    {
        return std::unexpected("the steering linkage will not solve at any usable rack travel: " + reach.error());
    }

    if (reach.value() < roadWheelAngle)
    {
        return std::unexpected("the steering linkage cannot reach the stated lock within a rack's travel");
    }

    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto steer = steerAt(middle);
        if (!steer)
        {
            return std::unexpected(steer.error());
        }

        if (steer.value() < roadWheelAngle)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }

    const auto magnitude = 0.5 * (low + high);

    return steersRightForPositiveTravel ? magnitude : -magnitude;
}

// The parallel-axis contribution of a point mass offset from the axis it is being measured about.
[[nodiscard]] glm::dmat3 offsetInertia(const double mass, const glm::dvec3& offset)
{
    auto contribution = glm::dmat3(0.0);
    contribution[0][0] = mass * (offset.y * offset.y + offset.z * offset.z);
    contribution[1][1] = mass * (offset.x * offset.x + offset.z * offset.z);
    contribution[2][2] = mass * (offset.x * offset.x + offset.y * offset.y);

    return contribution;
}

} // namespace

// --- the brakes, from the hardware -------------------------------------------------------------
//
// **`brakes.ini` is not consulted at all any more**, and that is the change. It stated
// `MAX_TORQUE=4200` and `FRONT_SHARE=0.75`; the first was verified at source on 2026-08-23, found to
// be identical across all four cars in the pack, found unable to lock this car's front wheels at any
// pedal position, and replaced with 5600 — **another chosen number**, marked as one at the time. The
// bound under it was arithmetic; everything above the bound was a feel question. What follows is
// `docs/brake-model-brief.md`'s answer to that: parts, each with a source, and no scalar.
//
// **Sources, one per part.** The car is a Mk7.5 GTI **Performance**, which is the variant the mod is,
// and the Performance Pack's brakes are shared with the Mk7 Golf R and the 8V Audi S3:
//
//   front  340 x 30 mm vented disc      Brembo 09.C306.11, catalogued against this car
//          175 x 70 x 20 mm pad         Brembo P 85 144, front axle, the pad for that disc
//          60 mm single piston          TRW/Lucas sliding caliper, 5Q0407253A/254A; the piston
//                                       diameter is quoted by the caliper vendors that sell the set
//   rear   310 x 22 mm vented disc      Brembo 09.A200.10/11, catalogued against this car
//          106 x 57 x 17 mm pad         Brembo P 85 073, rear axle
//          42 mm single piston          ATE sliding caliper, 3Q0615423F, quoted with the caliper
//
// Both calipers are **single-piston sliders**, which is what every factory Golf VII brake is: one
// piston, two pads, and the caliper body carries the reaction onto the outboard side. `pistons = 1`
// with `frictionFaces = 2` is that, and the pair is the thing to get right — a second piston here
// would double the car's brake torque on a fiction.
//
// **Two parts are estimates and are marked as such.** `padOuterClearance` at 5 mm is universal
// practice rather than a measurement of this pad, and it moves the front effective radius by 3.8%.
// The friction couple at 0.40 is the midpoint of SAE J866's F band, which is what an OE pad's edge
// code says and is the largest uncertainty in the whole derivation at +/-12.5%. Everything else is
// measured to the millimetre.
[[nodiscard]] BrakeHardware golfMk7FrontBrake()
{
    // **The disc's mass is published and it agrees with its own geometry.** CLP Automotive quote the
    // OE part verbatim while selling a lighter one — "STOCK 340x30mm disc - 10.7kg (per disc)" — and a
    // third party's 340 x 30 for this platform is listed at 10 kg. The swept ring alone, from 95 to
    // 170 mm of radius with two cheeks and vanes filling about a third of the gap, is 9.7 kg of grey
    // iron; a 50 mm hat is the rest. The agreement is the check, not the source.
    return BrakeHardware{.pistonBore = 0.060,
                         .pistons = 1,
                         .discDiameter = 0.340,
                         .discThickness = 0.030,
                         .discMass = 10.7,
                         .discVented = true,
                         // Brembo catalogue this part's height as 50 mm, which is the length of the
                         // neck stage 3's heat has to come down. Sourced, like the diameter.
                         .hatHeight = 0.050,
                         .padRadialHeight = 0.070,
                         .padOuterClearance = 0.005,
                         .frictionFaces = 2,
                         .couple = lowMetallicOnCastIron()};
}

[[nodiscard]] BrakeHardware golfMk7RearBrake()
{
    // **The rear's mass is derived rather than published**, and it is derived from the front's: the
    // front's published 10.7 kg implies that 0.719 of the disc's stated thickness is iron once the
    // vanes are taken out, and the same construction on a 310 x 22 with this pad's swept ring gives
    // 5.5 kg of ring and about 6.4 kg with its hat. Flagged, because nobody publishes it.
    return BrakeHardware{.pistonBore = 0.042,
                         .pistons = 1,
                         .discDiameter = 0.310,
                         .discThickness = 0.022,
                         .discMass = 6.4,
                         .discVented = true,
                         // **Not catalogued for this part and bounded**, on the same footing as the
                         // rear's mass: a rear hat runs a little shorter than a front's because the
                         // caliper behind it is smaller. 45 mm, and the whole rear path moves by
                         // about a tenth across the 40-50 mm it could be.
                         .hatHeight = 0.045,
                         .padRadialHeight = 0.057,
                         .padOuterClearance = 0.005,
                         .frictionFaces = 2,
                         .couple = lowMetallicOnCastIron()};
}

// The wheel the discs are bolted inside — **stage 3, and the only path this model has from a brake
// at 500 °C to a tread at 50**.
//
// One statement for all four corners, because it is one part number: this car wears the same 18 × 7.5
// wheel at each end and the discs behind them differ, which is why `wheelThermalOf` takes the brake
// as well and derives a different hat neck front and rear.
//
// **The mass is sourced and is a class figure.** A direct replacement for this car's own 18 × 7.5
// 5×112 57.1 mm wheel is quoted at 27 lb — 12.25 kg — and a reproduction usually runs a little
// heavier than the casting it replaces, so the bias is known and is toward a slower wheel. Same
// standing as the tyre's published mass, and flagged the same way.
//
// **The bolt circle is published as a PCD**: 5 × 112 mm gives a radius of 0.056 m. The hat wall at
// 7 mm is bounded rather than measured, 5 to 9 mm for a 340 mm hat, and it is the number the whole
// path turns on — far more than the bolted joint's, which is thirty times larger and is the one
// nobody publishes. docs/brake-thermal-brief.md.
[[nodiscard]] WheelHardware golfMk7Wheel()
{
    return WheelHardware{.mass = 12.25,
                         .diameter = 0.4572,
                         .emissivity = 0.85,
                         .hatWallThickness = 0.007,
                         .boltCircleRadius = 0.056,
                         .jointConductance = 60.0,
                         .toTyre = 4.0,
                         .discRadiationShare = 0.5};
}

// The hydraulics, which are what turn a pedal into the pressure both axles see.
//
// **The master cylinder is the only sourced part**: 23.81 mm, which is what Bosch, Delphi, LPR and
// TRW all state for a Golf VII cylinder. The pedal ratio and the servo are class-typical figures
// rather than this car's own — a pedal ratio of 3.2:1 to 4:1 is the published range for a boosted
// passenger car, a light-vehicle servo gain is 3 to 4, and a 10 inch diaphragm at 0.75 bar of
// depression is an ordinary one. **All four are marked guessed** under criterion 2 of the brief.
//
// **They are nonetheless checked, and the check is the reason to believe them.** This car reaches its
// own braking limit — the 0.945 g it can actually brake at, which needs 59 bar — at **236 N of pedal
// force**. A boosted passenger car is designed to reach a maximum-effort stop at 200 to 300 N, which
// is a figure with nothing to do with any of the parts above, and the chain lands in the middle of
// it. That is a stronger statement about the hydraulics than any one component's tolerance.
//
// A second one falls out for free, and it settles something the mod was internally inconsistent
// about. `brakes.ini` states `HANDBRAKE_TORQUE=1200` against a rear *service* brake of 1050 N.m — a
// cable out-braking the hydraulics it pulls on, which is not a thing. Off these calipers the rear
// makes **3357 N.m across the axle** before the proportioning valve has any say, and a handbrake
// cable pulls on the caliper directly rather than through the valve, so 1200 is suddenly an ordinary
// figure with room to spare. Nothing was fitted to make that happen.
[[nodiscard]] BrakeHydraulics golfMk7Hydraulics()
{
    return BrakeHydraulics{.masterCylinderBore = 0.02381,
                           .pedalRatio = 3.5,
                           .boostRatio = 4.0,
                           .boosterDiaphragm = 0.254,
                           .boosterVacuum = 75000.0,
                           .maxPedalForce = 500.0};
}

// The rear circuit's proportioning valve — **the other half of the brake bias, and the half the
// calipers cannot state**.
//
// The calipers alone put 0.686 of this car's brake torque on the front axle. That is a fixed number
// and the ideal one is not: load transfer moves it from 0.647 at 0.3 g to about 0.81 where this car
// locks, so a fixed 0.686 is roughly right in the middle of a stop and **locks the rear axle first**
// at the top of one — the unstable order, and the reason `brakes.ini`'s stated 0.75 looked better
// than the hardware it claimed to describe. A real Mk7 answers this with EBD inside its anti-lock
// unit; this model has no EBD, so it gets the component EBD replaced, which is a fixed valve.
//
// **Both numbers are measured against one stated criterion and neither is chosen**, and the criterion
// is the one every brake regulation is written around: *the front axle must lock first*. A car that
// locks its rears first is a car that swaps ends, and UN ECE R13-H's adhesion-utilisation annex
// exists to forbid exactly that. So the valve is the **most rear braking this car can carry while
// still locking its front axle first**, and that is a maximum rather than a preference.
//
// Swept and measured — `./EngineTests "[.brake-model]"`, the lock-pressure tables. With no valve at
// all, which is what the calipers alone describe:
//
//     the front axle locks at pedal 0.375 -> 58.95 bar
//     the rear axle locks at  pedal 0.250 -> 39.30 bar     <- first, and by a long way
//
// Then across knees of 24 to 32 bar and slopes of 0 to 0.30, **on dry tarmac and on mu 0.35**,
// because the ideal distribution moves with grip as well as with deceleration and a valve checked at
// one point of that range has been checked at one point. The largest admissible pair is 24 bar and
// 0.30: the rear then locks at pedal 0.450 against the front's 0.375 on dry and never before the
// front on mu 0.35, while 28 bar at the same slope already fails. It carries **730 N.m a side**
// against the 323 a plain limiter at the same knee would allow.
//
// What it buys is a bias that *moves*: 0.686 below the knee, where the calipers alone decide, rising
// to **0.834 at a fully applied pedal**. The ideal runs from 0.647 at 0.3 g to about 0.81 at the
// limit, so the valved car tracks the ideal band instead of crossing it — which is the whole of what
// a proportioning valve is for and the whole of what `FRONT_SHARE=0.75` was standing in for.
//
// The cross-check nothing was fitted to: production fixed valves are quoted with knees of 25 to 40
// bar and slopes of 0.3 to 0.5. The slope lands in that band and the knee sits a bar under it.
[[nodiscard]] ProportioningValve golfMk7RearProportioningValve()
{
    return ProportioningValve{.kneePressure = 24.0e5, .slope = 0.30};
}

[[nodiscard]] std::expected<VehicleSetup, std::string> golfGtiMk7()
{
    auto setup = VehicleSetup{};

    setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints = golfMk7FrontCorner(CornerSide::Left);
    setup.corners[static_cast<std::size_t>(Corner::FrontRight)].hardpoints = golfMk7FrontCorner(CornerSide::Right);
    setup.corners[static_cast<std::size_t>(Corner::RearLeft)].hardpoints = golfMk7RearCorner(CornerSide::Left);
    setup.corners[static_cast<std::size_t>(Corner::RearRight)].hardpoints = golfMk7RearCorner(CornerSide::Right);

    // --- the mass ledger ---
    //
    // TOTALMASS is the whole car and the corner masses are per corner, so the sprung mass is what is
    // left.
    //
    // **Everything below this line re-derives itself when the split changes, and that is the point.**
    // Correcting the unsprung mass was specified as a four-step job — set the corner figure,
    // recompute sprung mass to preserve the total, recompute the sprung centre of gravity to preserve
    // the *assembled* centre, recompute the inertia tensor from the corrected distribution — and only
    // the first step is an edit here. The other three were already solved rather than authored, for
    // the reason given at each: stating the same fact twice is how two statements of it come to
    // disagree. So `sprungMass` is a subtraction, `sprungCentre` is solved from the assembled centre
    // the data states, and `shell` is the assembled tensor less every parallel-axis term the assembly
    // puts back. Change a corner mass and all three follow, and `GolfGtiTests` asserts that the
    // assembled car still has the centre and the tensor the file states.
    const auto unsprung = 2.0 * golfFrontHubMass + 2.0 * golfRearHubMass;
    const auto sprungMass = golfTotalMass - unsprung;

    // Where the centre of gravity is. The height comes from car.ini's GRAPHICS_OFFSET, which places
    // the graphics origin relative to the physics body and so says how far the centre of gravity
    // sits above the road. That one is the mod's and is plausible: 0.572 m against a typical
    // C-segment hatch's 0.55.
    const auto centreHeight = 0.572;

    // **The longitudinal position is the third number taken away from the mod** (2026-08-22).
    //
    // suspensions.ini states `CG_LOCATION=0.53 ; Front Weight distribution in percentance` — the
    // file names its own convention, so `scripts/import-ac-car.py` reads it correctly and the source
    // is what disagrees. A transverse-engined front-drive hatchback does not sit at 53% front, and a
    // DSG Mk7 GTI measures **61.4/38.6**. At 1348 kg that is 114 kg per axle, 8.5% of the car.
    //
    // **It is a bigger change than the radius or the unsprung mass and it is a different kind.**
    // Those two corrected levers — a scale factor on speed and gearing, a split that preserved every
    // total. This moves the centre of gravity 222 mm forward, which moves the car's *balance*: front
    // tyre load, understeer, the aligning moment, and therefore where the steering limit falls in
    // rack torque. Nothing downstream is unaffected. It is here because the mod disagreed with the
    // published car three times out of three and Dominic's call was to source the chassis from the
    // specification rather than keep patching it.
    const auto frontFraction = 0.614;
    const auto centreStation = golfWheelbase * (frontFraction - 0.5);
    const auto centre = glm::dvec3(0.0, centreHeight, centreStation);

    // **And AC's own frame origin, which is not the same point any more.**
    //
    // Everything AC states positionally — the aero surfaces, the collider — it states relative to
    // *its* centre of gravity, and its centre of gravity is the one at 53%. A wing is bolted to the
    // bodywork: correcting where the car balances does not slide the radiator inlet 222 mm up the
    // chassis. So AC-relative positions are carried across from where AC put its origin, and only
    // the mass ledger uses the corrected centre.
    //
    // Getting this wrong is invisible in every total — the aero would still integrate to the same
    // drag — and would show up only as a pitching moment nobody authored, which is exactly the class
    // of fault this file keeps being bitten by.
    const auto acCentreStation = golfWheelbase * (0.53 - 0.5);
    const auto acOrigin = glm::dvec3(0.0, centreHeight, acCentreStation);

    // The unsprung masses ride at their own wheel centres, so the sprung centre is whatever puts the
    // *assembled* car's centre of gravity where the data says. Solved rather than authored, because
    // authoring it would be stating the same fact twice and letting the two disagree.
    const auto hubMass = [](const std::size_t index)
    {
        return index < 2 ? golfFrontHubMass : golfRearHubMass;
    };

    auto unsprungMoment = glm::dvec3(0.0);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        unsprungMoment += hubMass(index) * setup.corners[index].hardpoints.wheelCentre;
    }

    const auto sprungCentre = (golfTotalMass * centre - unsprungMoment) / sprungMass;

    // car.ini's INERTIA is the box whose inertia the whole car has, stated as its three dimensions —
    // 1.54 x 1.452 x 4.27 m, which are the car's track, its height and its length. That is the
    // *assembled* car's tensor about its own centre of gravity, so the shell carries it less every
    // parallel-axis term the assembly puts back: four unsprung masses at their wheel centres and the
    // sprung mass at its own offset. Stating the box on the shell instead would over-state the whole
    // car's yaw inertia by a third.
    auto shell = glm::dmat3(0.0);
    {
        const auto box = glm::dvec3(1.54, 1.452, 4.27);
        const auto twelfth = golfTotalMass / 12.0;

        shell[0][0] = twelfth * (box.y * box.y + box.z * box.z); // pitch, about the lateral axis
        shell[1][1] = twelfth * (box.x * box.x + box.z * box.z); // yaw
        shell[2][2] = twelfth * (box.x * box.x + box.y * box.y); // roll, about the longitudinal axis

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            shell -= offsetInertia(hubMass(index), setup.corners[index].hardpoints.wheelCentre - centre);
        }

        shell -= offsetInertia(sprungMass, sprungCentre - centre);
    }

    setup.sprung = {MassComponent{.mass = sprungMass, .centre = sprungCentre, .inertia = shell}};

    // tyres.ini WIDTH. The patch's length is not stated by AC and stays the model's own; the sample
    // counts stay at three by three, which is part of this car's configuration rather than free to
    // change under it — the penetration is a quadrature and converges rather than being exact.
    setup.sampling.width = 0.235;

    // --- aero, from aero.ini ---
    //
    // Each wing's area is its CHORD by its SPAN and its coefficient is its own lookup table read at
    // zero angle of attack, times the GAIN beside it — so the body carries all the drag and none of
    // the lift and the two ends carry lift and no drag, which is what a road car is. AC's sign
    // convention is that a positive CL is *downforce*, so the rear's -0.124 is lift and comes across
    // with its sign turned over.
    //
    // **Positions are AC's, relative to AC's origin** — see `acOrigin` above. They were relative to
    // `centre` until the weight distribution was corrected, and the two were the same point until
    // then, which is why this reads as a change to nothing.
    const auto wingArea = 1.03 * 2.07;
    const auto fromCentre = [&acOrigin](const double x, const double y, const double z)
    {
        return acOrigin + glm::dvec3(x, y, z);
    };

    setup.aero = {AeroSurface{.centre = fromCentre(0.0, 0.23, -0.40), .dragArea = 0.340 * wingArea, .liftArea = 0.0},
                  AeroSurface{.centre = fromCentre(0.0, -0.13, 1.10), .dragArea = 0.0, .liftArea = -0.00107 * wingArea},
                  AeroSurface{.centre = fromCentre(0.0, 0.86, -1.55), .dragArea = 0.0, .liftArea = 0.124 * wingArea}};

    // --- bodywork ---
    //
    // AC's only collider is a floor slab rather than a body box, so it gives the height the bodywork
    // starts at and nothing else: COLLIDER_0 is 0.1 m thick centred 0.20 m below the centre of
    // gravity, so its underside is the number taken. The length and height are the INERTIA box's,
    // which are the car's own; the width is the front track plus a tyre, since the track is measured
    // between the wheels' centre planes. Where along the car the box sits is not stated and it is
    // centred on the wheelbase.
    const auto floor = centreHeight - 0.20 - 0.05;
    const auto roof = 1.452;
    setup.body =
        CollisionBox{.centre = glm::dvec3(0.0, 0.5 * (floor + roof), 0.0),
                     .halfExtents = glm::dvec3(0.5 * (golfFrontTrack + 0.235), 0.5 * (roof - floor), 0.5 * 4.27)};

    // Where a ride height for this car is quoted from. **Not in AC's data**: its only collider is a
    // coarse body shell whose underside is 0.32 m up, which is a real surface of the car and not the
    // one anybody means. A Mk7 GTI is quoted at about 135 mm unladen, so that is the design figure
    // taken here — a placeholder in the same sense as every other number without a source.
    setup.rideHeightReference = 0.135;

    // **This car carries its load transfer through its wishbones as well as through its springs.**
    //
    // Off is the model every measured figure in docs/ was taken under, and it is still the model
    // default (`VehicleSetup::geometricLoadPath`), so a placeholder car is unchanged and this is the
    // only car that has moved. On, the tyre's in-plane forces reach the corner's own degree of
    // freedom through the linkage's Jacobian, which is roll centre, jacking, anti-dive, anti-squat
    // and anti-lift at once — not five features but one dot product that used to be missing four of
    // its terms. Roll falls about 12% and the roll gradient goes 2.77-2.95 to 2.40-2.59 deg/g, which
    // is the seat report of 2026-08-27 ("the body seems to pitch and roll excessively, however the
    // handling feels fine") addressed where it lives.
    //
    // **Dive rises, and that is this car's front geometry rather than a sign error.** The front
    // hardpoints put the side-view instant centre below ground, so the axle carries about -27%
    // anti-dive — it is pro-dive — and switching the path on lets that reach the springs. Measured,
    // localised and left alone: `[.load-path]` prints it, and moving a hardpoint on that evidence is
    // a separate decision with its own seat session. docs/suspension-load-path-brief.md.
    setup.geometricLoadPath = true;

    // **And the wheels' own spin reacts on the body.** On for this car since 2026-08-27, **driven and
    // kept on Dominic's instruction** — it was built, measured and shipped off that morning
    // precisely so the verdict could be his rather than a default's.
    //
    // The term is `chassis torque += -(net wheel spin torque)` about the spin axis, which is the
    // wheel's own `I·alpha` and is therefore **exactly zero whenever the spin is steady**. It is not
    // `-T_drive`, which would be wrong for braking and wrong at a constant speed, and it is not
    // torque roll: a transverse engine puts its crank and its differential on the *lateral* axis, so
    // the reaction is a **pitch** couple. Derived rather than argued — for a car accelerating at `a`,
    // angular momentum about the centre of gravity needs the load transfer to exceed `m·a·h/L` by
    // `sum(I·a/r)` over the four wheels, and this closes that balance. About 1.8% of the transfer.
    //
    // **What it costs, and it is knowingly paid.** 0-100 6.856 -> 6.889 s (+0.49%) and a full-pedal
    // stop from 30 m/s 54.18 -> 55.09 m (+1.69%), on a car whose braking is the open defect
    // (`docs/braking-chain-brief.md`: 41.74 m against a verified 35.5 m). What it buys is transient
    // pitch — peak squat over a launch 0.831 -> 0.963 deg, peak dive over a stop 1.688 -> 2.255 deg.
    // Those two are **mostly the fixture's step pedal**, where a real one takes a tenth of a second.
    //
    // `OSR_DRIVELINE_REACTION=off` is the way back and the control, and it is re-stamped on every
    // setup-sheet reload. `docs/suspension-fidelity-brief.md`, item 3.
    setup.drivelineReaction = true;

    // **The tyres carry a temperature, and the brakes carry one, and both were switched on together
    // on 2026-08-28 on Dominic's instruction: *"put on for good"*.**
    //
    // They went on as a pair and not one at a time, which was the plan from the moment the second was
    // built: brake heat through the rim is a term in the tread's own balance, so a car with a thermal
    // brake and an isothermal tyre is a car with half of one model. Each was driven and accepted on
    // its own first — the tyre as *"definitely noticable... could feel the understeer reduce"*, the
    // brake as fade whose magnitude matched Dominic's own experience of Bathurst — and stage 3, the
    // path between them, was measured before either default moved.
    //
    // **What this cost, and it was known before it was paid.** Every performance figure this project
    // had was measured on a tyre permanently at its best and a brake that could not fade. The
    // braking reference this car is furthest from — auto motor und sport's 35.5 m — is explicitly
    // *kalt*, so a cold tyre is what makes that comparison like-for-like for the first time, and it
    // makes the number worse before it makes it honest. `docs/known-red.md` carries whatever moved.
    //
    // `OSR_TYRE_THERMAL=off` and `OSR_BRAKE_THERMAL=off` are the way back and the controls, and both
    // are re-stamped on every setup-sheet reload. docs/tyre-state-brief.md,
    // docs/brake-thermal-brief.md.
    setup.tyreThermal = true;
    setup.brakeThermal = true;

    // car.ini CONTROLS: the steering wheel's lock in degrees each way, and the ratio between it and
    // the road wheel. The travel that produces that angle is solved off the linkage below.
    //
    // **The ratio's sign is not decoration.** This car's steering arm sits behind the kingpin, so a
    // rack moving to the right turns it left — the opposite of the placeholder's front-steer
    // geometry, and the opposite of what every consumer of `VehicleInput::steering` assumes. AC
    // states the ratio negative, and honouring that sign is what puts a right-hand turn back on
    // positive steering.
    const auto steerLock = 378.0;
    const auto steerRatio = -14.1;
    const auto roadWheelLock = std::abs(steerLock / steerRatio) * (3.14159265358979323846 / 180.0);

    // suspensions.ini: dampers with a knee and the anti-roll bars. The spring and damper are carried
    // onto the shaft by the motion ratio the linkage reports at design.
    //
    // **The spring rates are the tenth and eleventh numbers taken away from the mod** (2026-08-24).
    // `suspensions.ini` states `SPRING_RATE` 35000 front and 57000 rear, and this model read both as
    // **wheel** rates. Against measured figures for the car — 3.5 kg/mm front and 4.5 kg/mm rear on
    // motion ratios of 0.96 and 0.64 — the wheel rates are `spring x MR^2` = **31632 and 18076**.
    //
    // **The front being nearly right is what makes the rear credible as a defect.** At a front motion
    // ratio of 0.96 a spring rate and a wheel rate are the same number to 8%, so AC's 35000 is correct
    // whichever convention it is in — it lands within 11% of the wheel rate below. At the rear's 0.64
    // they differ by 1/0.64^2 = 2.4x, and AC's 57000 read as a wheel rate is **3.15 times too stiff**.
    // One end right and the other wrong is a bad number; both ends wrong the same way would have been
    // our own misreading.
    //
    // What it was doing: the rear ride frequency came out at **2.47 Hz against the front's 1.49**, a
    // 61% split where a road car runs the rear 10-20% above the front. These give 1.39 and 1.42 Hz.
    //
    // **Ride height is preserved and that is not luck**: `springFreeLengthForLoad` below solves the
    // rest length from *this* rate and the corner's static load, so a rate change moves stiffness and
    // not attitude. Scaling `springRate` on an already-built car does the opposite, which is how a
    // probe briefly credited this change with fixing the rear wheel lift that it does not fix.
    //
    // **HELD, NOT ADOPTED — and the reason is the whole argument for holding it.** Fitting the wheel
    // rates above breaks a criterion that was derived from the real car: the **skidpad falls to
    // 0.883 g against a 0.90-0.95 band** taken from a Mk7 GTI on OEM tyres, which is the measurement
    // this car's tyre grip was itself set against (`grip-set-without-circularity`).
    //
    // The mechanism is not grip, it is balance. Softening the rear by 68% while the anti-roll bars
    // stay put moves the roll-stiffness distribution from **49.7% front to 67.2%**, and a
    // front-drive car with two thirds of its roll couple on the front axle understeers into its
    // limit. The bars are AC's numbers too and would have to be re-derived alongside — which is a
    // second unverified figure to lean on, not a free move.
    //
    // **And the rear motion ratio is the number the whole thing turns on.** At the source's 0.64 the
    // rear ride frequency comes to 1.39 Hz; at 0.78 it would be 1.70, which is a sporty hatchback
    // and leaves the balance nearly alone. A 0.14 difference in an unverified motion ratio is the
    // difference between a car that passes its own skidpad and one that does not.
    //
    // So this stays on AC's figures until the rear motion ratio is confirmed, and the ride-frequency
    // anomaly is recorded rather than acted on. `docs/known-red.md`.
    const auto wheelRate = std::array{35000.0, 35000.0, 57000.0, 57000.0};
    const auto bumpRate = std::array{4600.0, 4600.0, 6200.0, 6200.0};
    const auto fastBumpRate = std::array{1834.0, 1834.0, 1842.0, 1842.0};
    const auto bumpKnee = std::array{0.070, 0.070, 0.100, 0.100};
    const auto reboundRate = std::array{5300.0, 5300.0, 6700.0, 6700.0};
    const auto fastReboundRate = std::array{2589.0, 2589.0, 2700.0, 2700.0};
    const auto reboundKnee = std::array{0.140, 0.140, 0.140, 0.140};
    const auto antiRoll = std::array{34000.0, 34000.0, 15000.0, 15000.0};

    // **The tenth and eleventh numbers this car has that AC's data does not carry**, and the first
    // two that come out of published measurements of *other* cars rather than of this one. Both are
    // stated here with their sources; `docs/suspension-fidelity-brief.md`, items 1 and 5, carry the
    // full argument and the bands.
    //
    // **Damper seal and rod friction, newtons at the shaft.** A `Curve` through the origin is a pure
    // viscous damper and has none; a real one carries a Coulomb term that dominates exactly where the
    // viscous term is smallest — small amplitudes and low velocities, which is straight-line running
    // on coarse tarmac, the first millimetre of a steering input and the settling after a kerb.
    //
    // Source: Deubel, Dittrich, Meinck and Prokop, *Experimental analysis and modelling of friction
    // in automotive shock absorbers operating under side forces*, Tribology International, 2025. The
    // specimen is a twin-tube damper from the **MacPherson front suspension of a VW Passat B8** —
    // MQB, the Mk7 Golf's own platform, which is as close a match as this project has ever had for a
    // borrowed number. Its pin-slider test reports approximately **117 N breakaway and 107 N
    // sliding**, with breakaway 2-14% above sliding (7% taken as representative) and the force
    // direction-independent to within 5% at moderate side force and velocity — which is what makes a
    // single symmetric constant a fair reduction of it.
    //
    // **The rear is the front's figure standing in for an unsourced one**, and is the weaker half of
    // this: that paper's whole point is that a MacPherson strut's friction is driven by the *side
    // force* the layout puts through the rod, and a rear damper that stands nearly upright carries
    // far less of it. So the rear is very likely lower and nobody has measured it. Marked here rather
    // than hidden in an average.
    const auto damperFriction = std::array{107.0, 107.0, 107.0, 107.0};

    // **Lateral-force compliance steer, radians per newton at the contact patch.** Every joint in
    // this model's linkage is an ideal pin or ball, so the rigid car takes no toe under load at all.
    // A real one takes a little, and it is *designed in* rather than tolerated.
    //
    // Source: Kawata, *Research on lateral force compliance steer settings for automobile front
    // suspensions*, Journal of the Institute of Industrial Applications Engineers 12(3), 2024,
    // Table 3 — six production cars measured on an **AB Dynamics K&C rig**, toe change of the outer
    // wheel per 1000 N applied laterally at both contact patches.
    //
    // **Which table applies is decided by this car's own hardpoints and not by assumption.** The
    // paper splits its samples by whether the steering gearbox sits ahead of the front axle or
    // behind it, and states that behind is the front-engine front-drive layout. The Golf's
    // `steeringRackOuter` is authored **222 mm behind the wheel centre**, so it is that group:
    // Table 3, whose four samples read **-0.004, -0.020, -0.022 and -0.032 deg/kN**. The **median,
    // -0.021 deg/kN**, is what is taken.
    //
    // **This is a class figure and not this car's**, which is the honest limit of it — the spread
    // across four production cars is eightfold, and the -0.004 sample is an outlier against a cluster
    // at -0.020 to -0.032. Negative is toe-out, which every sample in both of the paper's tables is,
    // because the purpose of the setting is to put the phase of the steering reaction force ahead of
    // the steering angle. The rear axle is left at zero: the paper is about front suspensions, this
    // car's rear is a multi-link stood in for by a double wishbone, and no figure was found.
    const auto lateralForceSteer =
        std::array{-0.021 * 0.017453292519943295 / 1000.0, -0.021 * 0.017453292519943295 / 1000.0, 0.0, 0.0};

    // The brakes, and **`brakes.ini` no longer appears in this line at all** (2026-08-23). Neither
    // `MAX_TORQUE` nor `FRONT_SHARE` is read: the peak is `peakBrakeTorque` on the hardware above and
    // the split is whatever the two calipers make of one line pressure. Sources per part and the two
    // estimates are at `golfMk7FrontBrake` above; the case for doing it at all, and what it
    // deliberately does not fix, are `docs/brake-model-brief.md`.
    //
    // **What it says, which was not what was expected.** 3666 N.m at each front corner and 1679 at
    // each rear — **10688 N.m in total against the 5600 it replaces**, and a front share of **0.686
    // against the mod's stated 0.75**. Both differences are the point rather than an error:
    //
    // - A road car *is* over-braked. Its brakes are sized for fade and for pedal effort, not for the
    //   grip limit, so full pressure is far past anything a tyre can use — 2.5 g here if grip were
    //   unlimited. What the size buys is where on the pedal the car reaches its limit: the fronts now
    //   lock at 0.47 of the pedal and the rears at 0.31, against 0.83 and 0.73 before. That is the
    //   whole feel change, and it is one a real car has.
    // - 0.75 was never a hardware figure. The calipers make 0.686, and the ideal share at this car's
    //   braking limit is 0.811 — so the raw hydraulics lock the **rear axle first**, by a wide margin.
    //   That is not a fault in the derivation; it is what a fixed split does on a car whose load
    //   transfer moves, and it is precisely what EBD exists to correct. A real Mk7 has EBD in the same
    //   unit as its ABS. This model does not, and the gap is now a **describable missing system**
    //   rather than a share somebody picked to hide it.
    //
    // The mod's 0.75 measured the shortest stop of everything swept, which is why it looked right. It
    // was right about the *distance* and wrong about the *hardware*, and those are separable now.
    setup.brakeHydraulics = golfMk7Hydraulics();
    setup.rearBrakeValve = golfMk7RearProportioningValve();

    const auto frontBrake = golfMk7FrontBrake();
    const auto rearBrake = golfMk7RearBrake();

    // Each corner's peak is its own circuit's pressure at a fully applied pedal — so the rear's is
    // taken after the valve, and the front/rear ratio at full pedal is *not* `frontBrakeShare`. That
    // is the point of the valve: the share moves with the pedal.
    const auto fullPressure = brakeCircuitPressures(setup, 1.0);
    const auto frontPeak = torquePerPressure(frontBrake) * fullPressure[static_cast<std::size_t>(Corner::FrontLeft)];
    const auto rearPeak = torquePerPressure(rearBrake) * fullPressure[static_cast<std::size_t>(Corner::RearLeft)];
    const auto brakeTorque = std::array{frontPeak, frontPeak, rearPeak, rearPeak};

    // And the thermal half of the same two parts. Derived from the hardware rather than stated
    // alongside it, so the disc that makes the torque above is the disc that carries the heat below —
    // there is no second statement of a mass, a diameter or a friction couple anywhere.
    //
    // **The wheel joins them since stage 3**, and it is derived from the same pair: `wheelThermalOf`
    // reads the disc's own hat geometry to work out the neck the heat has to come down, and
    // `brakeThermalOf`'s two-argument form tells the disc which share of its radiation now lands on
    // the wheel rather than on the sky.
    const auto wheel = golfMk7Wheel();
    const auto discs = std::array{brakeThermalOf(frontBrake, wheel), brakeThermalOf(frontBrake, wheel),
                                  brakeThermalOf(rearBrake, wheel), brakeThermalOf(rearBrake, wheel)};
    const auto wheels = std::array{wheelThermalOf(wheel, frontBrake), wheelThermalOf(wheel, frontBrake),
                                   wheelThermalOf(wheel, rearBrake), wheelThermalOf(wheel, rearBrake)};

    // Sprung load by statics about the other axle, from the sprung centre solved above.
    const auto frontSprung = sprungMass * standardGravity * (sprungCentre.z - golfRearAxle) / golfWheelbase;
    const auto rearSprung = sprungMass * standardGravity - frontSprung;
    const auto sprungLoad = std::array{frontSprung / 2.0, frontSprung / 2.0, rearSprung / 2.0, rearSprung / 2.0};

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& corner = setup.corners[index];

        // Each role's rate converts through its own ratio: the spring's through `SpringKinematics`
        // and the damper's through `DamperKinematics` — role-typed all the way into `kneedDamper`,
        // whose spring overload is deleted. On this car the spring is coaxial, so the two ratios
        // are the same bits and both conversions are the numbers they always were — the split is
        // the seam, not a change.
        const auto spring = solveSpringKinematics(corner.hardpoints, springElementOf(corner.hardpoints), 0.0, 0.0);
        const auto damper = solveDamperKinematics(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0, 0.0);
        if (!spring || !damper)
        {
            return std::unexpected("corner " + std::string(cornerAbbreviation(static_cast<Corner>(index))) + ": " +
                                   (spring ? damper.error() : spring.error()));
        }

        const auto springRatio = std::abs(spring->motionRatio);
        corner.springRate = wheelRate[index] / (springRatio * springRatio);
        corner.damper = kneedDamper(bumpRate[index], fastBumpRate[index], bumpKnee[index], reboundRate[index],
                                    fastReboundRate[index], reboundKnee[index], *damper);
        corner.antiRollRate = antiRoll[index];
        corner.brakeTorque = brakeTorque[index];
        corner.disc = discs[index];
        corner.wheel = wheels[index];
        corner.unsprungMass = hubMass(index);
        corner.damperFriction = damperFriction[index];
        corner.lateralForceSteer = lateralForceSteer[index];

        // tyres.ini, the Semislicks compound — AC's own default for this car, taken whole so the
        // grip and the carcass describe the same tyre: carcass rate and damping, angular inertia,
        // and rolling resistance. AC's ROLLING_RESISTANCE_0 of 12 is not a load fraction and the
        // file does not carry the formula it goes into; read as one part in a thousand it comes to
        // 0.012, and the speed-squared term beside it adds under a tenth of that at motorway speed
        // and is not modelled here.
        corner.tireVerticalRate = 298926.0;
        corner.tireVerticalDamping = 500.0;
        corner.wheelInertia = 1.45;
        corner.rollingResistance = 0.012;

        // The tyre. Reference load, both load-sensitivity exponents and both friction peaks are the
        // file's: DY_REF/DX_REF state the friction at FZ0, which is exactly this model's
        // peak-at-nominal-load convention, and AC states the force as going with load to the LS_EXP
        // power, so the exponent is that power less one with the sign turned over. 1.28 lateral sits
        // under the car's own rollover threshold — about 1.33 g for a 0.572 m centre of gravity on
        // this track width — so the handling cases still measure the tyre and not the wrong failure.
        //
        // **Both are stated now, and until 2026-08-23 only one was.** This model carried a single
        // exponent, the lateral one served both axes, and the comment that used to sit here said so
        // and called the file's longitudinal figure "close". It is not close in the place it is
        // asked: 0.1244 against 0.1926 is a 35% difference in how fast longitudinal grip falls away,
        // and the two references this car is measured against — a 100-0 and a 0-100 — sit at
        // opposite ends of the load curve, so an exponent that is wrong for the axis is wrong in
        // opposite directions at each end. Restoring the file's own number is a faithfulness fix and
        // not a tune; what it is worth in metres is measured in `[.in-gear]` and
        // `docs/tyre-grip-ratio-brief.md`, not assumed here.
        corner.tyre.nominalLoad = 2939.0;
        corner.tyre.lateralLoadSensitivity = 1.0 - 0.8074;
        corner.tyre.longitudinalLoadSensitivity = 1.0 - 0.8756;

        // **The fourth number taken away from the mod** (2026-08-22), after the wheel radius, the
        // unsprung mass and `CG_LOCATION`. The file states DY_REF 1.28 and DX_REF 1.30 for its
        // Semislicks — and **1.23/1.26 for a compound it calls "Street"**, which is not a street tyre:
        // a 225/40 R18 performance road tyre peaks nearer 1.0 to 1.15. Both of its compounds are
        // track-tyre numbers.
        //
        // Scaled by 0.87, which is **derived rather than chosen**, and the derivation is the point:
        //
        //   1. Criteria 6 and 7 were fixed from physical reasoning first, and they can be because
        //      neither is a grip figure — 6 is response shape and 7 is a ratio between two runs that
        //      share their grip. Measured across a 20% grip cut, 6's four channels move under 1.5%
        //      and 7's ratio moves 0.9%. They then hold at whatever grip is chosen.
        //   2. Criterion 5's threshold comes from the real car: a Mk7 GTI on OEM tyres skidpads at
        //      **0.90 to 0.95 g**. Grip is the free parameter that lands it, so nothing about the
        //      model may set it.
        //   3. 0.87 puts criterion 5 at **0.9230 g**, mid-band.
        //   4. **0-100 is then checked independently** on the launch fixture against the published
        //      6.5-6.6 s measured (MOTOR Australia, amS) — a different measurement against a
        //      different external reference, with no shared parameter but this one.
        //
        // Steps 3 and 4 agreeing is the validation. Neither number was used to set the other.
        //
        // Applied to the peaks and **not to `gripScale`**: that is the runtime seam a thermal,
        // pressure or wear model multiplies through, and spending it on a static data correction
        // would leave those models nowhere to act.
        //
        // **Driven and accepted, 2026-08-22** — Dominic's verdict was that it feels okay. That is a
        // pass on the car's character with 13% less grip than the mod stated, which is what the
        // derivation could not tell us.
        corner.tyre.lateralPeak = 1.114;
        corner.tyre.longitudinalPeak = 1.131;

        // **The longitudinal curve's shape, and these three are fitted to target behaviour rather
        // than sourced.** Labelled that way deliberately: `tyres.ini` carries `DX_REF` and nothing
        // whatever about fall-off, so claiming provenance for a shape factor would be inventing it.
        //
        // What was here were the model's defaults — C = 1.65, K = 20, E = -1.0 — and they were never
        // calibrated against anything. A Magic Formula's sliding force tends to `D * sin(C * pi/2)`
        // as slip grows, so C alone caps it: 1.65 gives sin(148.5 deg) = **52.2% of peak**, measured
        // at 54.8% by slip ratio 3. A standing start on this car reaches slip ratios of 2.5 to 3.8,
        // so it was spending the whole launch on half the grip it had. **The lateral curve's
        // C = 1.35 gives 85% and is healthy, which is why nothing caught it** — every validation this
        // project runs is lateral, and the longitudinal curve had never been checked against
        // anything at all.
        //
        // **Judged at a slip ratio of 1, not at 3**, and that correction matters more than the
        // numbers it produced. A tyre rig runs to about 0.2-0.3, and to 1.0 for locked-wheel work;
        // nobody measures one at 500% slip. Past that the formula is pure extrapolation and its
        // asymptote is a property of the functional form rather than of any tyre — which is how a
        // published C_x of 1.65 coexists with real tyres that plainly do not lose half their grip
        // when they spin. The 70-80%-of-peak figure this is aimed at is locked-wheel data, so slip 1
        // is where it is read: **77.7% here against 60.2% as shipped.**
        //
        // **K moves with C and E and is not free.** B is `K / (C * mu)`, so both shift where the peak
        // sits and K is what holds it at 0.120, where a semislick's belongs. 28 does that and stays
        // inside the 10-30 a road tyre runs. It is also what rules the alternatives out: E = +0.90 at
        // C = 1.50 wants K = 81, and C = 1.45 with E = 0.95 wants 120, against the low forties of a
        // race construction. A curve needing those is not this tyre.
        //
        // E = 0 rather than the +0.50 first tried. Holding the peak makes the two nearly equivalent
        // where the data lives — 0.777 against 0.765 at slip 1 — so the choice is near-peak breadth:
        // +0.50 makes 91.5% of peak at half the peak slip against this curve's 87.7%, and a semislick
        // should be the peakier of the two. Chosen on that rather than as a by-product of the K
        // constraint, and E = 0 needs no defending either way.
        //
        // Swept in `[.tyre-shape]`; **driven and accepted 2026-08-22**, in the same session as the
        // grip correction below — the two went to the seat together and cannot be separated by it.
        //
        // **No longer merely fitted: corroborated against two external references, 2026-08-24**
        // (`[.peak-to-tail]`, docs/tyre-peak-to-tail-brief.md). With the peak held, sweeping the
        // tail 0.44-0.91 spans 6.31-7.37 s of electronics-on 0-100 and only this curve lands in
        // the measured 6.5-6.6; its locked-to-peak ratio (0.804 through the chassis) sits inside
        // the published 0.75-0.90 dry band. The same sweep proved the anti-lock recovery law is
        // calibrated against this tail's shallowness — steep tails cost the car 4-6 m through the
        // controller alone — so changing these three is a controller co-design, not a data edit.
        corner.tyre.longitudinalShape = 1.50;
        corner.tyre.longitudinalCurvature = 0.0;
        corner.tyre.longitudinalStiffness = 28.0;

        // --- the tread's heat balance ---
        //
        // On, since 2026-08-28 — `setup.tyreThermal` above, on Dominic's instruction and after he
        // drove it. The sentence that stood here said the opposite and was left behind by that switch.
        //
        // **The geometry is the tyre's own size and is arithmetic, not data.** A 225/40 R18 on an 18
        // inch rim: rim radius 0.2286 m, section height 0.40 × 0.225 = 0.090 m, so the outer radius
        // is 0.3186 — which is `golfTyreRadius`, arrived at independently, and is the check that the
        // size and the model agree. The tread band's width is `tyres.ini`'s own WIDTH.
        corner.tyre.thermal.outerRadius = golfTyreRadius;
        corner.tyre.thermal.rimRadius = 0.2286;
        corner.tyre.thermal.treadWidth = 0.235;

        // **Groove depth and tyre mass are published, for a named performance tyre in exactly this
        // size**: a Michelin Pilot Sport 4S in 225/40 R18 states 9.5/32 inch of tread — 7.54 mm — and
        // weighs 22.7 lb, 10.30 kg. This car wears AC's "Semislicks" rather than a Pilot Sport 4S, so
        // it is a *class* figure and not this tyre's; what it buys is that the tread's mass comes out
        // of geometry and the carcass's out of a published total, instead of both being guessed.
        //
        // The undertread below the groove floor and the void fraction of the pattern are the two
        // numbers nobody publishes. Both are bounded rather than sourced — 1.5 to 3 mm of undertread
        // and 25 to 30% void on a performance summer pattern — and they matter only through the
        // tread's rubber volume, which they move by about a tenth either way.
        corner.tyre.thermal.grooveDepth = 0.00754;
        corner.tyre.thermal.underTread = 0.002;
        corner.tyre.thermal.voidFraction = 0.28;
        corner.tyre.thermal.tyreMass = 10.30;

        // **The share of patch friction power that heats the rubber rather than the road was to be
        // this model's one fitted number, and it is not fitted at all** — it is the effusivity
        // partition of the two materials already stated above, `661 / (661 + 1576)`. The derivation
        // and the reason it does not double-count the road conduction are on `TyreThermal`. Left as
        // the model's own default rather than restated here, because it is a property of rubber
        // against asphalt and not of this car.
        //
        // So there is **no fitted number in this tyre's thermal model**. What there is instead is
        // four *bounded* ones — the undertread, the pattern's void fraction, the belt package's
        // depth and still-air convection — each a textbook or construction band rather than a value
        // chosen to make an output land somewhere. `docs/tyre-state-brief.md`, section 4.

        // **The tread-road interface resists heat, and the figure is measured on this exact
        // interface**: 2.52 × 10⁴ W/(m²·K) for rubber on asphalt, NASA TN D-8161 (Miller, 1976),
        // stated as a lower limit — `TyreThermal::roadContactConductance` carries the source and the
        // series composition. Stated since 2026-08-29. What it buys is a fifth of the road path
        // (283.6 → 226.8 W/K at 100 km/h) and one to two degrees of tread core, below anything a
        // seat can resolve; both gates are 0 of 32400 with it stated and unstated, so stating it
        // moved no golden. `OSR_TYRE_CONTACT=perfect` is the control and the way back.
        // `docs/tyre-state-brief.md`.
        corner.tyre.thermal.roadContactConductance = 25200.0;

        // **The tyre's exterior radiates, and the figure is tyre thermography's own, stated twice at
        // the same value.** "The emissivity (ε) was set to 0.95, which is the standard characteristic
        // value for rubber tires and automotive paint" — Hu, Li, Ma, Cheng, Zheng & Zhang, *Deep
        // Learning-Based Real-Time Vehicle Tire and Tank Temperature Monitoring Using Thermal
        // Cameras*, Applied Sciences 16:2656, 2026 (doi 10.3390/app16062656). And "the both tire
        // surfaces emissivity has been set constant equal to 0.95" — Allouis, Farroni, Sakhnevych &
        // Timpone, *Tire Thermal Characterization: Test Procedure and Model Parameters Evaluation*,
        // WCE 2016, the same group whose Hilpert correlation and core-temperature finding this model
        // already carries. Carbon black is most of why: a filled tread is nearly a black body.
        //
        // What it is worth is a parked tyre — at a standstill radiation is ~7 W/(m²·K) against the
        // still-air floor's 5, which is what took the recorded ~35-minute parked time constant
        // towards the truth's ~20 — and at speed it hides inside the forced path. Stated 2026-08-29.
        //
        // **There is deliberately NO seat knob for this one**: a term whose effect only a
        // twenty-minute park can show is not a seat knob, and the way back is this line at 0.0.
        corner.tyre.thermal.emissivity = 0.95;

        // **AC's `tcurve_semis.lut`, slid 20 °C down its temperature axis on 2026-08-28 because the
        // window it came with belongs to a different tyre from the one this car wears.** The shape is
        // AC's, knot for knot and multiplier for multiplier; only where it sits has moved.
        //
        // **Why it moved.** The unslid plateau, 75 to 95 °C, was corroborated by Michelin's own
        // technical bulletin for the Pilot Sport Cup 2 R — 70 to 100 °C with 90 ideal — and that is a
        // **track** tyre's bulletin, while this car's tread depth and mass are a Pilot Sport **4S**'s,
        // a **road** tyre. Two published sources put a summer road tyre's design operating temperature
        // far below a racing tyre's:
        //
        //   Persson & Xu, *Rubber Friction: Theory, Mechanisms, and Challenges*, arXiv:2507.18782v3 —
        //   "The typical operating temperature for summer tires is around 50 °C... In contrast, the
        //   operating temperature for racing tires is ~100 °C or higher", with glass transition
        //   temperatures of −30 °C for a summer compound against −10 °C for a racing one.
        //
        //   Fortunato, Ciaravola and Furno (**Bridgestone Technical Center Europe**), Scaraggi, Lorenz
        //   and Persson, arXiv:1512.01359 — "for passenger car tires at typical operating temperatures
        //   it appears as if the friction usually decreases with increasing temperature while for
        //   special tires, e.g., motorsport tires, the friction may increase as the temperature
        //   increases up to rather high temperatures."
        //
        // **The shift is bounded rather than chosen, and the seat chose inside the bound.** The two
        // compounds' Tg differ by 20 °C, which is the smallest defensible shift; their published
        // operating temperatures differ by 40, which is the largest. **20 is what ships**, on Dominic's
        // instruction — *"I want the tyres to come in"* — because at 40 a lap's warm-up is worth
        // +0.47% of grip and the tyre is effectively always ready, while at 20 it is +2.56% and the
        // car still comes in. `OSR_TYRE_IDEAL=85` is the A/B and the way back to the track window.
        //
        // Its ends are another matter. 0.80 at the cold end and 0.60 at the hot one are nobody's
        // measurement in either position, and they are used because the alternative is a cliff at the
        // edge of the plateau. Flagged here exactly as `ARB FRONT 34000` is flagged, and for the same
        // reason: a borrowed number is usable and must never be mistaken for a measured one. **Sliding
        // a curve is not redrawing it** — nothing here reshapes a tail.
        //
        // The curve is read at the **tread core** and not at the surface. That is sourced, and it
        // falsified this brief's own first draft — Farroni et al., *TRT EVO*, Proc IMechE Part D
        // 233(1) 2019: grip correlates with the core layer, because a thin skin cannot change the
        // tread block's bulk viscoelastic state fast enough to matter.
        corner.tyre.thermal.grip = TemperatureCurve{
            .count = 13,
            .celsius = {{-20.0, 0.0, 20.0, 40.0, 55.0, 65.0, 75.0, 85.0, 120.0, 140.0, 180.0, 200.0, 230.0}},
            .multiplier = {{0.80, 0.92, 0.95, 0.98, 1.00, 1.00, 1.00, 0.97, 0.95, 0.88, 0.82, 0.80, 0.60}}};

        // --- the air inside it ---
        //
        // **`tyres.ini`'s own two numbers, and they are the only ones AC states about pressure that
        // carry a meaning this model can check.** `PRESSURE_STATIC 28` and `PRESSURE_IDEAL 34`, psi
        // gauge, converted to pascals here because every other pressure in this model is in pascals.
        //
        // **The reference temperature is a choice and is flagged as one.** AC states no temperature
        // for either pressure, so 20 °C is this project's convention — the temperature a pressure is
        // set at in a garage — and not a borrowed number. It matters: set the same 28 psi on a cold
        // morning and the tyre is a different tyre all day.
        //
        // **And the pair says something about the window this car now runs.** The gas law fixes what
        // 28 cold and 34 ideal mean together: the tyre reaches its ideal pressure when its air is at
        // 61.2 °C, which is inside the 55-75 °C road window above and 24 °C below the track window it
        // replaced. Two of AC's numbers now agree with a road tyre and none with a track one, and this
        // one was arrived at through the gas law rather than through anybody's curve.
        corner.tyre.pressure.coldPressure = 28.0 / psiPerPascal;
        corner.tyre.pressure.coldReferenceTemperature = 20.0;
        corner.tyre.pressure.idealPressure = 34.0 / psiPerPascal;

        // The cavity's volume, from this tyre's own geometry: an annulus between the rim's 0.2286 m
        // and the inner surface of the tread package, about 15 mm below the 0.3186 m outer radius,
        // across the section's width. 28 litres, and it is used for the gas's heat capacity alone.
        corner.tyre.pressure.cavityVolume = 0.0282;

        // The middle of that plateau, which is where every fixture starts. Persson's "around 50 °C"
        // for a summer tyre sits inside it; 65 is taken because it is the centre of the flat region
        // and therefore the seed with the most headroom either side before a capture starts moving.
        //
        // **`tyreDefaultTemperature` carries the same number and has to.** A seed left at the old
        // plateau's centre would put every fixture in the suite off the flat part of this curve and
        // move three to four per cent of skidpad, 0-100 and stopping distance with no physics changed.
        corner.tyre.thermal.idealTemperature = 65.0;

        // Stated on the shaft, and sized inside the travel the linkage has. The bump stops AC states
        // do not carry across: its front BUMPSTOP_UP of 0.80 m is not a travel any suspension has and
        // its rear BUMPSTOP_DN of 0 would have the droop stop touching at rest.
        //
        // **The bump gap is confirmed and the droop gap was wrong** (2026-08-24). Measurements of a
        // 2019 GTI Rabbit Edition DSG — same MQB platform — give a bare shock shaft of 73 mm at ride
        // height with **18 mm to the bump stop**, so the 20 mm here is right to two millimetres.
        //
        // Droop is a different thing and was being modelled as if it were the same thing. **On a strut
        // the rebound limit is the damper topping out**, not a bumper somebody specifies: the travel
        // is whatever stroke is left once ride height has used some. Placing 20 mm there invented a
        // constraint the car does not have, and it bound — measured, the rear reached its droop stop
        // at 23 mm of extension against **-4965 N**, and a rear tyre carrying 80 N is 0.27 mm off the
        // road. The wheel was leaving the ground under braking because its suspension had stopped
        // extending, which is why nothing at the front ever moved it.
        //
        // 40 mm is placed against the **linkage's** own limit rather than invented: the geometry
        // allows 49.5 mm of front shaft droop and 52.0 of rear, and the car only ever asks for 27.4.
        // That leaves the stop real travel to work in before the geometric clamp behind it, which has
        // no reaction force and is what `validateCornerSetup` exists to keep the car away from.
        // Measured: lowest rear axle load **79.8 N -> 402.1 N**, and the stop is then never touched at
        // all. It saturates by 30 mm, so this is not a number tuned to an outcome.
        //
        // Still placeholders, and still bounding everything behind them: `bumpAngle`/`droopAngle` give
        // only 55 mm of front bump travel where the real car measures **73**.
        corner.bumpStop = TravelStop{.gap = 0.020, .rate = 900000.0, .progression = 3.0, .damping = 40000.0};
        corner.droopStop = TravelStop{.gap = 0.040, .rate = 600000.0, .progression = 3.0, .damping = 30000.0};

        const auto restLength = springFreeLengthForLoad(corner, sprungLoad[index]);
        if (!restLength)
        {
            return std::unexpected(restLength.error());
        }

        corner.springFreeLength = restLength.value();

        if (const auto validated = validateCornerSetup(corner); !validated)
        {
            return std::unexpected("corner " + std::string(cornerAbbreviation(static_cast<Corner>(index))) + ": " +
                                   validated.error());
        }
    }

    // **The front left, and the side is load-bearing because of Ackermann.** A positive demand is a
    // right turn, so the left front is the *outer* wheel and Ackermann steers it less — this linkage
    // wants 70.0 mm of rack to put the outer wheel on the stated lock and 64.5 mm to put the inner
    // one there, an 8% difference in the car's whole steering ratio depending on which is asked.
    //
    // It reads the outer one because that is the wheel this car has always been derived from and
    // driven with: before the corner sides were un-mirrored (`outboardSign`) this line said
    // `corners[1]` and that index *was* the left front. Which of the two AC's single STEER_RATIO is
    // quoted against is a real question and it is not this change's to answer — un-mirroring the
    // labels must not quietly re-gear the car by eight percent.
    const auto rackTravel =
        rackTravelForSteer(setup.corners[static_cast<std::size_t>(Corner::FrontLeft)].hardpoints, roadWheelLock);
    if (!rackTravel)
    {
        return std::unexpected(rackTravel.error());
    }

    // Signed by the linkage, not by the ratio. `steerRatio` says how many degrees of rim make a
    // degree of road wheel and its sign is a convention about which the rack knows nothing.
    setup.rackTravelPerInput = rackTravel.value();

    return setup;
}

[[nodiscard]] DrivelineSetup golfGtiMk7Driveline()
{
    auto setup = DrivelineSetup{};

    const auto atRpm = [](const double rpm, const double torque)
    {
        return glm::dvec2(rpm * rpmToRadiansPerSecond, torque);
    };

    // Torque at a speed the manufacturer states as a *power*, which is the only honest way to write
    // the top of the curve: 180 kW from 5000 to 6200 rpm is one statement about three points, and
    // reading three torques off a dyno graph instead would be three inventions that happen to agree.
    const auto atPower = [&atRpm](const double rpm, const double watts)
    {
        return atRpm(rpm, watts / (rpm * rpmToRadiansPerSecond));
    };

    // **The seventh number taken away from the mod** (2026-08-24), after the wheel radius, the
    // unsprung mass, `CG_LOCATION`, the tyre peaks, `MAX_TORQUE` and `FRONT_SHARE`. Full account:
    // `docs/engine-curve-brief.md`.
    //
    // **The import is faithful and that is gated — the source data is wrong.** AC's turbo is
    // `boost = min(WASTEGATE, MAX_BOOST * (rpm/REFERENCE_RPM)^GAMMA)` and `[TURBO_0]` states 1.60,
    // 1.20, 2200 and 2.5; recomputed from those four numbers the imported curve matched at all
    // eighteen points, and the wastegate is on its stop at **1961 rpm**, so above two thousand revs
    // the imported curve was `power.lut x 2.2` and no turbo behaviour was involved in the shortfall
    // at all. It was entirely in `power.lut`, which would need a flat 168.2 N.m across the plateau
    // and carries 124 / 144 / 154 / 152 / 156. The mod's engine is a weaker engine than the car it
    // names, and `ui_car.json` claims a third figure again — 402 N.m, which is a Golf R's.
    //
    // **What is here is two published statements and nothing invented between them**: 370 N.m from
    // 1600 to 4300 rpm and 180 kW from 5000 to 6200, which is the whole of what a manufacturer
    // homologates and is exactly the shape a wastegated turbo makes. Torque between 4300 and 5000 is
    // the straight line joining them. Above 6200 it tapers to the mod's own figure at the limiter,
    // because nothing published says where it ends and the power up there was already right to 0.3%.
    //
    // **Below 1600 rpm nothing is published and the mod's own curve is kept.** That leaves a steeper
    // spool from 1000 to 1600 than the mod had — 146.7 to 370 rather than to 208 — which is what a
    // wastegated turbo physically does and is still not a measurement. It is the one stretch this
    // change makes a claim about that no in-gear pull exercises, so `CreepTests`, `AntiStallTests`
    // and `GradeTests` are what stand over it.
    //
    // **This model has no turbo, so the correction goes into the final curve.** AC's boost ramp is
    // therefore not a constraint on it, which matters: that ramp cannot reach full boost before
    // 1961 rpm and VW's plateau starts at 1600, so honouring the published figure means departing
    // from the mod's turbo shape. Where the mod and the real car disagree, the real car wins.
    //
    // What it is worth is the **spread across gears** and not the offset — a wrong scalar moves
    // three gears together and a wrong curve moves them apart. Measured through the reference car's
    // own gearing in `[.in-gear]`, the mod's curve ran -15.9% / -11.1% / +0.8% in 4th / 5th / 6th
    // against auto motor und sport's sheet, a spread of 16.7 points; this one runs flat.
    setup.engine.torque =
        Curve{.points = {atRpm(0.0, 95.0), atRpm(500.0, 129.92491), atRpm(1000.0, 146.745119), atRpm(1600.0, 370.0),
                         atRpm(4300.0, 370.0), atPower(5000.0, 180000.0), atPower(5500.0, 180000.0),
                         atPower(6200.0, 180000.0), atRpm(6500.0, 264.0), atRpm(6800.0, 248.16)}};

    setup.engine.inertia = 0.150;
    setup.engine.idleSpeed = 850.0 * rpmToRadiansPerSecond;
    setup.engine.limiterSpeed = 6800.0 * rpmToRadiansPerSecond;
    // engine.ini [COAST_REF]: what it absorbs at the limiter with the throttle shut.
    setup.engine.coastTorque = 75.0;

    // **The eighth number taken away from the mod** (2026-08-24). `drivetrain.ini` states seven
    // ratios and a **single** `FINAL` of 4.37; Volkswagen's own *2019 Golf GTI Technical
    // Specifications* state the 7-speed DSG as the ratios below and **two** final drives, 4.17 and
    // 3.13. Reverse as its magnitude — `reduction` puts the sign on for it, so carrying a negative
    // through would give a reverse gear that drives forward.
    //
    // **First gear was within 1.7% and that is why this lasted**: applying the low final to the whole
    // box left 2nd and 3rd 13-21% tall and 4th through 7th **35% to 47% short**, and nothing that
    // checks a standing start can see it. The sanity check that settles it is a cruise — the mod put
    // 7th at 2397 rpm at 100 km/h and 5912 at 250, which is near peak power in an overdrive; these
    // give 1691 and 4170.
    //
    // Which gear runs through which axle is **not stated** by VW and is inferred — see
    // `Gearbox::finalDrivePerGear` for the rule and for how it is validated against a box whose
    // mapping *is* published. It comes out `I, II, II, I, I, II, II`: gears alternating between the
    // two output shafts in pairs, which is what a dual-clutch box physically is and is not the
    // contiguous "low gears on the low axle" arrangement it is tempting to assume.
    //
    // The resulting overall reductions are 14.178 / 8.607 / 5.540 / 3.878 / 2.961 / 2.379 / 2.003,
    // whose steps rise all the way up the box — 0.607, 0.644, 0.700, 0.763, 0.803, 0.842 — which is
    // the shape every real gearbox has and is how the arrangement was chosen.
    // Full account: `docs/engine-curve-validation-brief.md`.
    setup.gearbox.ratios = {3.40, 2.75, 1.77, 0.93, 0.71, 0.76, 0.64};
    setup.gearbox.finalDrive = 4.17;
    setup.gearbox.finalDrivePerGear = {4.17, 3.13, 3.13, 4.17, 4.17, 3.13, 3.13};
    setup.gearbox.reverseRatio = 2.90;

    // **Derived on a different car, which is what makes it evidence rather than a tune.** No road
    // measurement can separate engine torque from driveline efficiency — they are a product in the
    // tractive force — so this is solved on auto motor und sport's 230 PS Mk7, whose engine, mass,
    // gearing, road load and in-gear times are every one of them separately published, leaving
    // efficiency the only unknown left in it. Three gears sampling three different rpm bands agree on
    // **0.879 to 0.945**, a spread of 7.3% of its own value, and this is the midpoint.
    //
    // It is emphatically **not** the 0.83 a first pass got by attributing the whole in-gear residual
    // to it: that residual is also carrying a 7.2% mass error and the gearing above.
    // `docs/engine-curve-validation-brief.md`, and `EngineCurveValidationProbe` re-derives it.
    setup.losses.efficiency = 0.912;

    // **Restated for this car's own final drive rather than inherited.** The default is derived
    // through 4.37, which is what `placeholderSedan`'s driveline has and what this car had until the
    // gearing was corrected; the same halfshafts through 4.17 are 25000/4.17^2 = 1438 N.m/rad at the
    // gearbox output. Leaving it on the default was a silent 7% softening of this car's driveline —
    // and it showed up as a *placeholder* car's shift test moving, because a default nobody restates
    // belongs to whoever is still using it.
    setup.compliance.stiffness = 1438.0;

    // **The launch programme this car has, shipped OFF** — like ABS, traction control and XDS, and for
    // the same reason: every measurement this project holds is of the car underneath its electronics,
    // and a default that quietly launched every fixture would be a suite that could no longer see the
    // car. A test or a setup sheet that wants it says so, once.
    //
    // What it reproduces is the procedure from the seat: brake and throttle together, the engine held
    // near 3000 rpm on a slipping clutch, and the car released when the brake comes off. The second
    // phase — what the controller does to the clutch *during* the take-up — is not modelled yet.
    setup.autoClutch.launch.enabled = false;
    setup.autoClutch.launch.targetSpeed = 3000.0 * rpmToRadiansPerSecond;

    setup.coupling.kind = DriveCouplingKind::FrictionClutch;

    // [TRACTION] TYPE=FWD, and [DIFFERENTIAL] POWER/COAST/PRELOAD. AC's two ramp numbers are the
    // fraction of the torque going through the diff that becomes locking torque on power and off it,
    // which is exactly what `clutchPackLsd` takes — so this fits the interface as it stands and needs
    // nothing built. A preload of zero does not make it open: it still locks with a quarter of
    // whatever is being put through it.
    setup.driven = DrivenAxle::Front;
    setup.differential = clutchPackLsd(0.0, 0.25, 0.25);

    return setup;
}

[[nodiscard]] AssistSetup golfGtiMk7Assists(const VehicleSetup& setup)
{
    auto assists = AssistSetup{};

    // What the hydraulics reach at a fully applied pedal, and what each corner's brake is worth per
    // pascal of it.
    //
    // **Divided out of the car rather than read off the hardware**, deliberately. `golfMk7FrontBrake`
    // is where these come from, so calling it here would be the shorter line — and it would make the
    // electronics deaf to a setup sheet that has moved `brake.front`. The ECU is calibrated against
    // the car that was just built, which is the whole claim in this function's comment, and a tuned
    // brake is exactly the case where a second copy of a number would diverge unnoticed.
    const auto fullPressure = brakeCircuitPressures(setup, 1.0);

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        assists.maximumWheelPressure[index] = fullPressure[index];
        assists.brakeTorquePerPressure[index] =
            fullPressure[index] > 0.0 ? setup.corners[index].brakeTorque / fullPressure[index] : 0.0;
    }

    // The radius the ECU converts wheel speed with. **The unloaded one on purpose**: the tyre's real
    // rolling radius is smaller than this under load, so the electronics read every wheel slightly
    // slow and think there is a little more slip than there is. That error is a real system's and is
    // not corrected here — see `ReferenceSpeedSetup::nominalRadius`.
    assists.reference.nominalRadius = golfTyreRadius;

    // Front-wheel drive, which is why traction control on this car is the easy case: the rear pair
    // is a road speed measurement that owes the engine nothing.
    const auto driven = std::array{true, true, false, false};
    assists.reference.driven = driven;
    assists.traction.driven = driven;

    // Both brake-based channels take the turn's kinematics off the undriven pair, so both need the
    // track widths as calibration. Stated twice rather than shared for the same reason `controlRate`
    // is: they are separate controllers in the same box and nothing makes them agree.
    assists.traction.frontTrack = golfFrontTrack;
    assists.traction.rearTrack = golfRearTrack;

    assists.cornering.frontTrack = golfFrontTrack;
    assists.cornering.rearTrack = golfRearTrack;

    // One ECU, one clock, as on the car.
    assists.antilock.controlRate = assists.controlRate;
    assists.traction.controlRate = assists.controlRate;
    assists.cornering.controlRate = assists.controlRate;

    return assists;
}

} // namespace raceengine
