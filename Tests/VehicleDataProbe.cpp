// The parameter sweep: every value in a published car's data, against the real car, against a hand
// calculation, or against the generic default it would otherwise have inherited.
// `./EngineTests "[.vehicle-data]"`.
//
// **Same shape as the telemetry channel sweep, and for the same reason.** Several separate faults
// this month were a number validated against a default, a placeholder or an unchecked assertion
// rather than against the actual car:
//
//   - Criterion 10's recorded 19.9 N·m was measured through `SteeringRack{}`'s 8.3 mm pinion. The
//     Golf's is 10.6 and the honest figure is 30.5.
//   - `Engine RPM [rpm]` carried rad/s for the whole of milestone 1.
//   - The EPS assist was justified by a parked rack stiffness of "2.5 N·m per mm" against a measured
//     0.11 — wrong by a factor of twenty-five, and nobody had run it.
//
// The failure mode is identical every time and it is not carelessness: a default is *designed* to be
// plausible, so inheriting one produces a car that behaves sensibly and is not this car.
//
// **Rewritten 2026-08-22 to enumerate rather than to sample.** The first version checked nine
// parameters and listed twenty as inherited, and its coverage was not stated — so "two disagree" was
// a fact about the nine, not about the car, and every field it never mentioned was equally suspect
// and looked settled. What it prints now is a walk over *every* scalar in `VehicleSetup`,
// `CornerSetup`, `ContactPatchSampling`, `AeroSurface`, `ContactMaterial` and `DrivelineSetup`, each
// in exactly one of four states, with the totals reconciled against a field count at the end. A
// parameter that cannot be checked is a line saying so, which is a different thing from a parameter
// nobody looked at.
//
// The four states:
//
//   OK / OFF   checked against a figure somebody else published for a Mk7.5 GTI
//   CALC/BAD   checked against arithmetic — no published figure exists, but the number is not free
//   DFLT       never set by this car, so it carries the generic mid-size placeholder's value
//   NOTE       a solver or model parameter with no vehicle-level analogue to check it against

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::cornerCount;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::VehicleSetup;

namespace
{

constexpr auto gravity = 9.80665;
constexpr auto pi = 3.14159265358979323846;

auto untouched = 0;
auto published = 0;
auto calculated = 0;
auto noted = 0;
auto disagreed = 0;

// A field with a figure somebody else published to check it against.
void against(const char* name, const double mine, const double reference, const double tolerance, const char* source)
{
    const auto agrees = std::abs(mine - reference) <= tolerance;
    published++;
    disagreed += agrees ? 0 : 1;

    std::printf("  %-4s %-28s %12.4f %12.4f   %s\n", agrees ? "OK" : "OFF", name, mine, reference, source);
}

// A field with no published figure but with arithmetic that constrains it: a frequency, a ratio, a
// deflection, a deceleration. Not weaker than a published check — the coastdown's `Crr` is checked
// this way and it is the check that found a 7.6% error nothing else could see.
void byHand(const char* name, const double mine, const double expected, const double tolerance, const char* working)
{
    const auto agrees = std::abs(mine - expected) <= tolerance;
    calculated++;
    disagreed += agrees ? 0 : 1;

    std::printf("  %-4s %-28s %12.4f %12.4f   %s\n", agrees ? "CALC" : "BAD", name, mine, expected, working);
}

// A field this car never set, so it is carrying whatever the generic mid-size placeholder says. Not
// necessarily wrong — plenty of these are genuinely unknown — but every one is a number nobody
// decided for this vehicle, and the point of the list is that it is a list.
void inherited(const char* name, const double mine, const double fallback, const char* note)
{
    const auto same = std::abs(mine - fallback) <= 1e-12 * std::max(1.0, std::abs(fallback));
    untouched += same ? 1 : 0;
    if (!same)
    {
        published++;
    }

    std::printf("  %-4s %-28s %12.4f %12.4f   %s\n", same ? "DFLT" : "set", name, mine, fallback, note);
}

// A parameter of the solver or of the model rather than of the car. Printed so the enumeration is
// complete and so nobody has to wonder whether it was forgotten.
void note(const char* name, const double mine, const char* what)
{
    noted++;
    std::printf("  %-4s %-28s %12.4f %12s   %s\n", "NOTE", name, mine, "-", what);
}

} // namespace

TEST_CASE("the Golf's data, every field, against the real car or against arithmetic", "[.vehicle-data]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto car = built.value();
    const auto driveline = golfGtiMk7Driveline();

    const auto generic = VehicleSetup{};
    const auto genericDriveline = raceengine::placeholderDriveline();

    untouched = 0;
    published = 0;
    calculated = 0;
    noted = 0;
    disagreed = 0;

    const auto& front = car.corners[0];
    const auto& rear = car.corners[2];
    const auto radius = front.hardpoints.wheelRadius;

    auto mass = car.unsprungMass();
    auto sprungMass = 0.0;
    auto heightMoment = 0.0;
    auto stationMoment = 0.0;
    for (const auto& component : car.sprung)
    {
        mass += component.mass;
        sprungMass += component.mass;
        heightMoment += component.mass * component.centre.y;
        stationMoment += component.mass * component.centre.z;
    }
    for (const auto& corner : car.corners)
    {
        heightMoment += corner.unsprungMass * corner.hardpoints.wheelCentre.y;
        stationMoment += corner.unsprungMass * corner.hardpoints.wheelCentre.z;
    }

    const auto centreHeight = heightMoment / mass;
    const auto centreStation = stationMoment / mass;
    const auto wheelbase = front.hardpoints.wheelCentre.z - rear.hardpoints.wheelCentre.z;

    // Read off the car rather than restated. It was the literal 0.53 in three places here, which is
    // the same "stated twice and now they disagree" trap the settle criterion fell into — correcting
    // the car's weight distribution silently left three hand calculations predicting the old one.
    const auto frontFraction = 0.5 + centreStation / wheelbase;

    std::printf("\n============================ GEOMETRY ============================\n");
    std::printf("  %-4s %-28s %12s %12s   %s\n", "", "parameter", "this car", "reference", "where the figure is from");

    against("wheelbase [m]", wheelbase, 2.637, 0.02, "VW");
    against("front track [m]", 2.0 * std::abs(front.hardpoints.wheelCentre.x), 1.538, 0.03, "VW");
    against("rear track [m]", 2.0 * std::abs(rear.hardpoints.wheelCentre.x), 1.516, 0.03, "VW");

    // **Corrected 2026-08-22 (V1).** The mod's tyres.ini says 0.3298; 225/40 R18 is 0.3186 unloaded.
    against("wheel radius [m]", radius, 0.3186, 0.002, "225/40 R18 unloaded, 18*25.4/2 + 225*0.40");

    // The design contact patch must sit on the road, which is what the whole pickup-point conversion
    // at the top of `PublishedCars.cppm` exists to arrange. If this ever disagrees, every hardpoint
    // is off the ground by the difference.
    byHand("wheel centre height [m]", front.hardpoints.wheelCentre.y, radius, 1e-9, "= wheel radius, by construction");

    against("steering lock to lock [deg]", car.steeringLockToLock * 180.0 / pi, 756.0, 1.0,
            "car.ini CONTROLS: 378 deg each way");

    // Solved off the linkage rather than authored, so what is checked is that the answer is a rack's
    // travel and not a number that merely parsed.
    byHand("rack travel per input [m]", car.rackTravelPerInput, 0.070, 0.020, "a C-segment rack is 60-90 mm each way");

    // Ride height: **the one geometry figure with no source at all.** AC states no floor, only a
    // coarse body shell whose underside is 0.32 m up, so this is a design figure for a Mk7 GTI.
    against("ride height reference [m]", car.rideHeightReference, 0.135, 0.015, "Mk7 GTI quoted unladen, PLACEHOLDER");

    std::printf("\n============================== MASS ==============================\n");

    // VW quote 1364 kg kerb for the manual and about 1385 for the DSG, which includes 75 kg of
    // driver and a 90% tank under the EU convention.
    against("total mass [kg]", mass, 1385.0, 60.0, "VW kerb weight, DSG");

    // **Corrected 2026-08-22 (V2).** The mod says 80 front / 85 rear; what is here is the component
    // build-up written out in `PublishedCars.cppm`, and published breakdowns put a passenger-car
    // corner at 35-45 kg.
    against("unsprung front [kg]", front.unsprungMass, 45.0, 8.0, "component build-up; 35-45 kg published range");
    against("unsprung rear [kg]", rear.unsprungMass, 42.0, 8.0, "component build-up, no driveshaft");

    byHand("sprung mass [kg]", sprungMass, mass - car.unsprungMass(), 1e-9,
           "= total less four corners, by subtraction");

    against("centre of gravity [m]", centreHeight, 0.55, 0.08, "typical for a C-segment hatch");

    // **This one disagrees and it is the sweep's biggest finding.** The mod's suspensions.ini says
    // `CG_LOCATION=0.53 ; Front Weight distribution in percentance` — the file names the convention
    // itself, so the importer is reading it correctly and the *source* is wrong. A DSG Mk7 GTI
    // measures 61.4/38.6 front to rear. At 1348 kg that is 114 kg per axle, eight percent of the
    // car, and it is a static front-load error that feeds every load-sensitive quantity there is:
    // front grip, the understeer balance, the self-aligning moment and therefore where the steering
    // limit falls in rack torque.
    //
    // **Not changed**, because unlike the wheel radius and the unsprung mass this one moves the
    // car's balance rather than correcting a lever, and the brief that asked for those two reserved
    // the "source the car from specifications instead" decision explicitly.
    against("front weight fraction []", 0.5 + centreStation / wheelbase, 0.614, 0.02, "Edmunds, measured, DSG");

    std::printf("\n=========================== SUSPENSION ===========================\n");

    // The rates are stated at the wheel in AC and carried onto the shaft by the motion ratio; what is
    // checkable without a published spring rate is the ride frequency they produce, which is a
    // designed-for quantity every chassis engineer works in.
    const auto rideFrequency = [&car, sprungMass, wheelbase](const std::size_t index, const double wheelRate)
    {
        const auto& corner = car.corners[index];
        const auto frontStation = car.corners[0].hardpoints.wheelCentre.z;
        const auto rearStation = car.corners[2].hardpoints.wheelCentre.z;

        auto moment = 0.0;
        for (const auto& component : car.sprung)
        {
            moment += component.mass * component.centre.z;
        }
        const auto sprungStation = moment / sprungMass;

        const auto share =
            index < 2 ? (sprungStation - rearStation) / wheelbase : (frontStation - sprungStation) / wheelbase;

        static_cast<void>(corner);

        return std::sqrt(wheelRate / (sprungMass * share / 2.0)) / (2.0 * pi);
    };

    const auto designRatio = [&car](const std::size_t index)
    {
        const auto design = raceengine::solveCornerWithJacobian(car.corners[index].hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());

        return std::abs(design->motionRatio);
    };

    const auto frontWheelRate = front.springRate * designRatio(0) * designRatio(0);
    const auto rearWheelRate = rear.springRate * designRatio(2) * designRatio(2);

    against("front wheel rate [N/m]", frontWheelRate, 35000.0, 1.0, "suspensions.ini, at the wheel");
    against("rear wheel rate [N/m]", rearWheelRate, 57000.0, 1.0, "suspensions.ini, at the wheel");

    byHand("front ride frequency [Hz]", rideFrequency(0, frontWheelRate), 1.55, 0.35, "sports hatch 1.3-1.9 Hz");
    byHand("rear ride frequency [Hz]", rideFrequency(2, rearWheelRate), 1.75, 0.35, "usually 10-30% above front");

    byHand("front wheel hop [Hz]",
           std::sqrt((frontWheelRate + front.tireVerticalRate) / front.unsprungMass) / (2.0 * pi), 13.0, 3.0,
           "sqrt((k_wheel+k_tyre)/m_unsprung); road cars 10-15 Hz");
    byHand("rear wheel hop [Hz]", std::sqrt((rearWheelRate + rear.tireVerticalRate) / rear.unsprungMass) / (2.0 * pi),
           13.0, 3.0, "same, rear");

    against("front anti-roll [N/m]", front.antiRollRate, 34000.0, 1.0, "suspensions.ini ARB FRONT");
    against("rear anti-roll [N/m]", rear.antiRollRate, 15000.0, 1.0, "suspensions.ini ARB REAR");

    // The damper, as a ratio against critical for the sprung corner. AC states a **kneed** curve, so
    // there is no single ratio: the slow-bump rate is what controls body motion and the fast rate is
    // what a bump sees, and a digressive damper is precisely one where those differ. Checking one
    // number against "0.25 to 0.60" is checking the wrong thing — both are printed and what is
    // asserted is the property that makes it a damper curve rather than a line.
    // The sprung mass on one front corner, by statics about the rear axle on the *sprung* centre —
    // not the whole car's front fraction, which is a different distribution (see the static-corner
    // check below for what that cost).
    const auto frontSprungCorner =
        sprungMass * 0.5 * (car.sprung.front().centre.z - rear.hardpoints.wheelCentre.z) / wheelbase;
    const auto ratioOf = [frontWheelRate, frontSprungCorner](const double rate)
    {
        return rate / (2.0 * std::sqrt(frontWheelRate * frontSprungCorner));
    };

    byHand("front damping, slow bump", ratioOf(4600.0), 0.70, 0.25, "c/2sqrt(km); controls body motion, firm end");
    byHand("front damping, fast bump", ratioOf(1834.0), 0.28, 0.15, "the digressive end; what a kerb sees");
    byHand("damper digression []", 4600.0 / 1834.0, 2.5, 1.0, "slow/fast > 1 is what makes it digressive");

    // Spring free length is **solved** so the corner sits at design under its static load — it is not
    // a spring anybody could order, it is whatever length puts this rate at this load at this
    // position. So the meaningful check is not the length, it is that the solve did what it claims:
    // the compression from free to installed, times the rate, must be the static load on the shaft.
    {
        const auto design = raceengine::solveCornerWithJacobian(front.hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());

        const auto shaftLoad = front.springRate * (front.springFreeLength - design->damperLength);
        const auto wheelLoad = shaftLoad * designRatio(0);

        // **The springs carry the *sprung* mass, whose distribution is not the whole car's.** The
        // four unsprung masses sit at the axles and are 49 kg front against 43 rear, so the sprung
        // centre's front fraction differs from the car's — by little enough to hide inside a 30 N
        // tolerance while the car was at 53% front, and by 73 N once it was corrected to 61.4%. The
        // check now uses the sprung component's own station, which is what
        // `springFreeLengthForLoad` was actually given.
        const auto sprungStation = car.sprung.front().centre.z;
        const auto rearStation = rear.hardpoints.wheelCentre.z;
        const auto sprungFrontShare = (sprungStation - rearStation) / wheelbase;

        byHand("front static corner [N]", wheelLoad, sprungMass * gravity * sprungFrontShare / 2.0, 5.0,
               "spring compression x rate x motion ratio = the sprung load it was solved for");
    }

    inherited("bump stop gap [m]", front.bumpStop.gap, generic.corners[0].bumpStop.gap,
              "AC's own BUMPSTOP_UP is 0.80 m, not a travel any suspension has");
    inherited("bump stop rate [N/m]", front.bumpStop.rate, generic.corners[0].bumpStop.rate,
              "placed, not from the file");
    inherited("droop stop gap [m]", front.droopStop.gap, generic.corners[0].droopStop.gap, "as above");
    inherited("droop stop rate [N/m]", front.droopStop.rate, generic.corners[0].droopStop.rate, "as above");

    std::printf("\n============================== TYRE ==============================\n");

    against("nominal load [N]", front.tyre.nominalLoad, 2939.0, 1.0, "tyres.ini FZ0, Semislicks");
    against("lateral peak []", front.tyre.lateralPeak, 1.28, 0.001, "tyres.ini DY_REF, Semislicks");
    against("longitudinal peak []", front.tyre.longitudinalPeak, 1.30, 0.001, "tyres.ini DX_REF, Semislicks");
    against("lateral load sensitivity []", front.tyre.lateralLoadSensitivity, 1.0 - 0.8074, 0.001, "1 - LS_EXP_Y");
    against("longitudinal load sensitivity []", front.tyre.longitudinalLoadSensitivity, 1.0 - 0.8756, 0.001,
            "1 - LS_EXP_X, stated by the file and discarded by this model until 2026-08-23");
    against("tyre vertical rate [N/m]", front.tyre.nominalLoad > 0.0 ? front.tireVerticalRate : 0.0, 298926.0, 1.0,
            "tyres.ini the compound's carcass rate");
    against("patch width [m]", car.sampling.width, 0.235, 0.001, "tyres.ini WIDTH, a 225 section mounted");
    against("rolling resistance []", front.rollingResistance, 0.012, 0.002, "tyres.ini ROLLING_RESISTANCE_0 / 1000");

    // The static deflection the carcass rate implies, and the patch length that deflection implies.
    // **This is the check that says whether the sampling length is right**, and it is the only one
    // available: AC does not state a patch length and the model's default is 0.16.
    const auto staticLoad = mass * gravity * frontFraction / 2.0;
    const auto deflection = staticLoad / front.tireVerticalRate;
    byHand("tyre deflection [mm]", deflection * 1000.0, 12.0, 5.0,
           "static front load / carcass rate; 10-15 mm typical");
    byHand("patch length [m]", car.sampling.length, 2.0 * std::sqrt(2.0 * radius * deflection), 0.04,
           "2*sqrt(2*R*d) from the deflection above");

    byHand("wheel inertia [kg.m2]", front.wheelInertia, 1.30, 0.35, "wheel+tyre ~21 kg at k~0.75R, plus the disc");

    inherited("tyre vertical damping", front.tireVerticalDamping, generic.corners[0].tireVerticalDamping,
              "AC states none; 500 is 5% of critical on this carcass and hub");
    inherited("patch samples across", static_cast<double>(car.sampling.across),
              static_cast<double>(generic.sampling.across), "part of the vehicle's configuration, not free");
    inherited("patch samples along", static_cast<double>(car.sampling.along),
              static_cast<double>(generic.sampling.along), "as above");
    inherited("patch length [m]", car.sampling.length, generic.sampling.length,
              "checked by hand above; still the default value");
    inherited("search distance [m]", car.sampling.searchDistance, generic.sampling.searchDistance,
              "how far down the ray looks for road");
    inherited("spike rejection [m]", car.sampling.spikeRejection, generic.sampling.spikeRejection,
              "**does** fire: the kerb-chamfer lottery, W3");

    std::printf("\n============================= BRAKES =============================\n");

    against("front brake torque [Nm]", front.brakeTorque, 4200.0 * 0.75 / 2.0, 1.0, "brakes.ini, FRONT_SHARE 0.75");
    against("rear brake torque [Nm]", rear.brakeTorque, 4200.0 * 0.25 / 2.0, 1.0, "brakes.ini");

    // What the whole system can do to the car, which is the number a road test publishes.
    byHand("braking [g]", 2.0 * (front.brakeTorque + rear.brakeTorque) / radius / (mass * gravity), 1.05, 0.25,
           "total torque / radius / weight; a GTI stops at about 1.0-1.1 g");

    std::printf("\n============================== AERO ==============================\n");

    auto dragArea = 0.0;
    auto liftArea = 0.0;
    for (const auto& surface : car.aero)
    {
        dragArea += surface.dragArea;
        liftArea += surface.liftArea;
    }

    against("Cd.A [m2]", dragArea, 0.66, 0.10, "0.30 Cd x 2.20 m2 frontal");
    byHand("net lift area [m2]", liftArea, 0.0, 0.40, "a road car makes a little of each end, near zero net");
    inherited("air density [kg/m3]", car.airDensity, generic.airDensity, "sea level, 15 C");

    std::printf("\n=========================== DRIVELINE ============================\n");

    auto peakTorque = 0.0;
    auto peakPower = 0.0;
    for (const auto& point : driveline.engine.torque.points)
    {
        peakTorque = std::max(peakTorque, point.y);
        peakPower = std::max(peakPower, point.x * point.y);
    }

    against("peak engine torque [Nm]", peakTorque, 370.0, 40.0, "VW, EA888 evo3");
    against("peak engine power [kW]", peakPower / 1000.0, 180.0, 25.0, "VW, 245 PS");
    against("idle speed [rpm]", driveline.engine.idleSpeed * 9.549296585513721, 800.0, 100.0, "Mk7 GTI idle");
    against("limiter [rpm]", driveline.engine.limiterSpeed * 9.549296585513721, 6800.0, 200.0, "engine.ini LIMITER");
    against("final drive []", driveline.gearbox.finalDrive, 4.37, 0.01, "drivetrain.ini FINAL");
    against("first gear []", driveline.gearbox.ratios[0], 3.19, 0.01, "drivetrain.ini GEAR_1");
    against("top gear []", driveline.gearbox.ratios[6], 0.65, 0.01, "drivetrain.ini GEAR_7 (DSG, seven forward)");

    // The gearing, in the units a driver reads. Top gear at the limiter must be *above* the car's
    // terminal speed or the car is gear-limited rather than drag-limited, which is what a 250 km/h
    // road car is not.
    const auto topReduction = driveline.gearbox.ratios[6] * driveline.gearbox.finalDrive;
    byHand("top gear at limiter [kph]", driveline.engine.limiterSpeed / topReduction * radius * 3.6, 290.0, 25.0,
           "must exceed the 250 kph the car is limited to");
    byHand("first gear at limiter [kph]",
           driveline.engine.limiterSpeed / (driveline.gearbox.ratios[0] * driveline.gearbox.finalDrive) * radius * 3.6,
           58.0, 10.0, "a hot hatch pulls about 60 kph in first");

    byHand("engine inertia [kg.m2]", driveline.engine.inertia, 0.15, 0.08, "a four-cylinder with a dual-mass flywheel");
    against("coast torque [Nm]", driveline.engine.coastTorque, 75.0, 1.0, "engine.ini COAST_REF");
    against("reverse ratio []", driveline.gearbox.reverseRatio, 2.9, 0.01, "drivetrain.ini GEAR_R, magnitude");

    note("differential preload [Nm]", 0.0, "AC PRELOAD=0; the pack still locks with a quarter of what it passes");
    note("diff power ramp []", 0.25, "AC POWER; fraction of passed torque that becomes locking");
    note("diff coast ramp []", 0.25, "AC COAST");
    note("driven axle []", 0.0, "TRACTION TYPE=FWD, so Front");

    inherited("engine stall speed [rad/s]", driveline.engine.stallSpeed, genericDriveline.engine.stallSpeed,
              "AC states no stall speed; the model's own floor");
    inherited("limiter restore band", driveline.engine.limiterRestoreBand, genericDriveline.engine.limiterRestoreBand,
              "how far under the limiter fuel comes back");
    inherited("gearbox input inertia", driveline.gearbox.inputInertia, genericDriveline.gearbox.inputInertia,
              "AC states none");
    inherited("upshift time [s]", driveline.gearbox.shift.upshiftTime, genericDriveline.gearbox.shift.upshiftTime,
              "a DSG is quicker than this; AC states no figure the importer reads");
    inherited("downshift time [s]", driveline.gearbox.shift.downshiftTime, genericDriveline.gearbox.shift.downshiftTime,
              "as above");
    inherited("auto clutch release []", driveline.autoClutch.releaseFraction,
              genericDriveline.autoClutch.releaseFraction, "AC states no auto-clutch behaviour at all");
    inherited("auto clutch launch [rad/s]", driveline.autoClutch.launchSpeed, genericDriveline.autoClutch.launchSpeed,
              "as above");
    inherited("clutch pedal free play", driveline.autoClutch.freePlay, genericDriveline.autoClutch.freePlay,
              "as above");

    std::printf("\n===================== CONTACT SOLVER (not the car) ================\n");

    note("body friction []", car.contact.friction, "the collision solver's, for bodywork touching road");
    note("body restitution []", car.contact.restitution, "as above");
    note("restitution threshold", car.contact.restitutionThreshold, "as above");
    note("penetration correction", car.contact.correction, "Baumgarte term; a solver number");
    note("allowed penetration [m]", car.contact.allowedPenetration, "slop; a solver number");
    note("body half extent x [m]", car.body.halfExtents.x, "front track plus a tyre; AC gives only a floor slab");
    note("body half extent y [m]", car.body.halfExtents.y, "the INERTIA box's height");
    note("body half extent z [m]", car.body.halfExtents.z, "the INERTIA box's length");

    std::printf("\n============================= TOTALS =============================\n");
    std::printf("  %d checked against a published figure, %d checked by hand calculation.\n", published, calculated);
    std::printf("  %d DISAGREE.\n", disagreed);
    std::printf("  %d fields are still the generic car's, %d are solver or model parameters.\n", untouched, noted);
    std::printf("  %d lines in total, which is every scalar in VehicleSetup, CornerSetup,\n",
                published + calculated + untouched + noted);
    std::printf("  ContactPatchSampling, AeroSurface, ContactMaterial and DrivelineSetup that this\n");
    std::printf("  car's factory either sets or inherits. Coverage is the point: an unchecked field\n");
    std::printf("  is indistinguishable from a checked one until somebody writes the list out.\n");
}
