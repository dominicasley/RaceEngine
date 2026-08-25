// In-gear acceleration, against a published data sheet. `./EngineTests "[.in-gear]"`.
//
// **Stage 3 of `docs/tyre-grip-ratio-brief.md`, and the brief calls it the step that actually decides
// the thing.** Everything in that brief hangs on 0-100 km/h being a trustworthy check on grip. It may
// not be: a real DSG without launch control has a protective, lazy take-up, this model's `AutoClutch`
// closes on the driver's request, and the rev-based launch regulator that used to stand in for one was
// deliberately removed on 2026-08-22. If our launch is optimistic then 0-100 landing inside its band
// is two errors cancelling and the whole tension is misattributed.
//
// An in-gear pull answers it because it has **no traction content at all** — 80 to 120 km/h at a fixed
// ratio asks the engine curve, the gearing, the mass and the drag, and nothing else. The driven-wheel
// slip is reported on every row precisely so that claim can be checked rather than assumed.
//
// **The reference**, sourced the way the caliper bores were: auto motor und sport's Supertest of the
// Golf VII GTI Performance. One sheet carries the gear ratios, both final drives, the test weight, the
// tyre size, the 0-100 and the elasticity rows, so nothing has to be reconciled across sources:
//
//   Getriebeübersetzungen  I 3.77  II 2.09  III 1.47  IV 1.15  V 1.17  VI 0.97
//   Achsantrieb            3.45:1 / 2.76:1   (I-IV on the first, V-VI on the second — which is why
//                                             V's ratio is numerically *above* IV's)
//   Leergewicht Testwagen  1406 kg           Reifen 225/40 R 18
//   0-100 km/h             6.5 s             100-0 km/h 35.5 m kalt
//   80-120 km/h            5.1 / 6.3 / 7.8 s (4th / 5th / 6th)
//
// **That car is the 230 PS manual and ours is the 245 PS DSG, and the difference runs the safe way.**
// The reference makes 350 N·m where ours is quoted at 370 and weighs 1406 kg where ours is 1348, so a
// faithful model of our car should beat every one of those three times. If it loses to a weaker,
// heavier car through that car's own gearing, the engine curve is the defect and no amount of tyre
// will fix it — which is the finding stage 3 exists to reach or to rule out.
//
// The gear-label reading of the elasticity row was checked rather than taken: at 1406 kg with a
// 350 N·m plateau and this gearing, hand arithmetic puts 4th at about 4.9 s, 5th at 6.1 and 6th at
// 7.6, against the sheet's 5.1 / 6.3 / 7.8. Three consecutive gears in the right order and within 4%
// is not a coincidence, so the row is 4./5./6. Gang.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::Curve;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::ShiftPhase;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto radiansPerSecondToRpm = 9.549296585513721;
constexpr std::array<double, cornerCount> noRoadTorque{};

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

// Long and narrow, and the car starts at one end. The generator lays the ground from z = 0 rather
// than centring it, so a car spawned at a negative station is a car dropped off the end of the world.
// A 120 km/h pull at 2 m/s^2 covers about 300 m; 12 km leaves room for sixth gear.
[[nodiscard]] ProvingGroundDescriptor straightGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 12000.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 4.0;
    descriptor.features = {};

    return descriptor;
}

[[nodiscard]] double designHeight(const VehicleSetup& setup)
{
    auto highest = 0.0;
    for (const auto& corner : setup.corners)
    {
        highest = std::max(highest, corner.hardpoints.wheelCentre.y + corner.hardpoints.wheelRadius);
    }

    return highest;
}

struct PullResult
{
    double time = 0.0;
    double startRpm = 0.0;
    double endRpm = 0.0;
    double meanEngineTorque = 0.0;
    double peakDrivenSlip = 0.0;
    double peakClutchSlip = 0.0;
    bool reached = false;
    bool limiter = false;
};

// One in-gear pull, at a fixed ratio and full throttle.
//
// **Timed from the crossing of the entry speed rather than from the throttle**, and the car is already
// flat out when it gets there. That removes the driveline's own step response — the compliant shaft
// winding up, the clutch settling — which is a property of the fixture and not of the car, and leaves
// the engine curve, the gearing, the mass and the drag, which are what is being asked about.
//
// It also means no turbo lag is included, and that is worth stating rather than hiding: this engine
// model has none to include. A magazine's elasticity figure is measured from a steady 80 with the
// throttle stepped, so it carries whatever the spool costs. Our number is therefore the optimistic
// one, which again runs the safe way for the conclusion this probe is here to reach.
[[nodiscard]] PullResult pullInGear(const VehicleSetup& setup, const DrivelineSetup& driveline,
                                    const PhysicsWorld& world, const int gear, const double fromSpeed,
                                    const double toSpeed)
{
    const auto radius = setup.corners.front().hardpoints.wheelRadius;
    const auto reduction = driveline.gearbox.reduction(gear);

    // Ten km/h of run-up, which is a second or so at these rates — long enough for the shaft and the
    // clutch to settle and short enough that the pull still starts inside the gear's useful range.
    const auto entry = fromSpeed - 10.0 / 3.6;

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);

    for (auto step = 0; step < 720; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, entry);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = entry / radius;
    }

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    // Put the box in the gear and the engine where that gear says it must be. Started at idle with the
    // clutch closed against wheels already turning at road speed, the coupling would spend the first
    // tenth of a second dragging the engine up through several thousand rpm — a cold-start transient
    // rather than a car in gear at speed, and the launch fixture's own header records what believing
    // one of those cost.
    drivelineState.gear = gear;
    drivelineState.targetGear = gear;
    drivelineState.shiftFrom = gear;
    drivelineState.engineSpeed = entry / radius * reduction;
    // The shaft sits between the gearbox output and the final drive, so which axle is downstream of
    // it — and therefore what speed it turns at — is a property of the gear on a two-final transaxle.
    drivelineState.shaftSpeed = entry / radius * driveline.gearbox.finalFor(gear);

    const auto inertias = wheelInertias(setup);
    auto road = noRoadTorque;

    auto input = VehicleInput{};
    input.gear = gear;
    input.throttle = 1.0;

    auto result = PullResult{};
    auto startTime = -1.0;
    auto samples = 0;
    auto torqueSum = 0.0;

    // The run-up. Flat out the whole way, so by the time the entry speed arrives everything transient
    // is over — which is exactly what the preconditions below then check rather than trust.
    for (auto step = 1; step <= 360 * 60; step++)
    {
        const auto torques = stepDriveline(driveline, drivelineState,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());

        const auto speed = state.chassis.linearVelocity.z;
        const auto now = static_cast<double>(step) * tick;

        if (startTime < 0.0 && speed >= fromSpeed)
        {
            // --- preconditions. Nothing this returns is worth quoting until these hold. ---
            //
            // In the gear that was asked for, with the shift finished; the clutch closed, so the
            // engine is geared to the road and not slipping against it; all four wheels on the ground;
            // and straight. A pull measured through a slipping clutch is a clutch measurement, and a
            // pull measured on a car with a wheel in the air is nothing at all.
            // Everything the four assertions below can fail on, captured together — a precondition
            // that reports only the quantity it tested makes the reader guess which of the chain's
            // links moved. `clutchSlip` and `clutchLocked` are here because the locked-chain check at
            // the bottom and the 5 rad/s slip bound above can disagree: 0.5% of engine speed is
            // 1.18 rad/s in fifth, so the tighter of the two binds and it is not the one named
            // "clutch".
            CAPTURE(gear, drivelineState.gear, drivelineState.clutchPedal, drivelineState.engineSpeed, speed,
                    torques->clutchSlip, torques->clutchLocked, torques->engine, torques->windUp,
                    drivelineState.shaftSpeed);

            REQUIRE(drivelineState.gear == gear);
            REQUIRE(drivelineState.shiftPhase == ShiftPhase::Engaged);
            REQUIRE(drivelineState.clutchPedal < 0.05);
            REQUIRE(std::abs(torques->clutchSlip) < 5.0);
            REQUIRE(std::abs(state.chassis.position.x) < 0.5);
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                REQUIRE(stepped->telemetry.wheels[index].inContact);
            }

            // And the engine really is geared to the gearbox *output*: from crank to there the chain
            // is rigid, so what turns is one body and the ratio is the whole of the relationship.
            //
            // Against the wheels rather than against the road is the distinction this used to make,
            // and it is still the right one — a driven tyre transmits force by slipping, so at full
            // throttle the hub runs a few percent ahead of the road however healthy the driveline is,
            // and folding that into a driveline check would make the check sensitive to grip.
            //
            // **But crank-to-hub is not rigid and this assertion used to say it was** (corrected
            // 2026-08-24, by the engine curve exposing it). There is a torsional element between the
            // gearbox output and the differential, on purpose, and while torque is *changing* it
            // twists — so the two ends of it genuinely turn at different speeds and no epsilon makes
            // that false. Measured on the build that exposed it: VW's 370 N.m wound the shaft
            // 0.285 rad, putting the gearbox output 1.12 rad/s ahead of the differential input. That
            // is 0.54%, it tripped a 0.5% bound, and it was the driveline working rather than
            // failing. The old form passed only because the mod's weaker curve wound the shaft less,
            // which is the worst reason for a check to pass.
            //
            // So the rigid claim is made where the chain is rigid, and the compliant element is
            // checked as a compliance below instead of being asserted out of existence.
            const auto shaftSpeed = drivelineState.shaftSpeed;
            REQUIRE(drivelineState.engineSpeed ==
                    Catch::Approx(shaftSpeed * driveline.gearbox.ratio(gear)).epsilon(0.005));

            // The shaft is wound and not coming apart. `maximumTwist` is the guard that keeps a
            // failed solve diagnosable rather than a NaN, so a pull sitting anywhere near it is not a
            // pull worth quoting — half of it is a wide berth and still far above the 0.285 rad a
            // full-torque plateau actually asks for.
            REQUIRE(std::abs(drivelineState.windUp) < 0.5 * driveline.compliance.maximumTwist);

            startTime = now;
            result.startRpm = drivelineState.engineSpeed * radiansPerSecondToRpm;
        }

        if (startTime >= 0.0)
        {
            samples++;
            torqueSum += torques->engine;
            result.peakClutchSlip = std::max(result.peakClutchSlip, std::abs(torques->clutchSlip));

            for (auto index = std::size_t{0}; index < 2; index++)
            {
                result.peakDrivenSlip =
                    std::max(result.peakDrivenSlip, std::abs(stepped->telemetry.wheels[index].slipRatio));
            }

            result.limiter = result.limiter || drivelineState.fuelCut;

            if (speed >= toSpeed)
            {
                result.time = now - startTime;
                result.endRpm = drivelineState.engineSpeed * radiansPerSecondToRpm;
                result.meanEngineTorque = torqueSum / static_cast<double>(samples);
                result.reached = true;
                break;
            }
        }
    }

    return result;
}

// The auto motor und sport test car's gearbox, folded onto one final drive. Its two axle ratios are
// 3.45 for gears I-IV and 2.76 for V-VI; this model carries a single final drive, so the second one is
// folded into the two gears that use it and the *overall* reduction — which is the only thing a road
// speed sees — is preserved exactly. 1.17 x 2.76 = 3.2292 = 0.9360 x 3.45, and 0.97 x 2.76 = 2.6772 =
// 0.7760 x 3.45.
[[nodiscard]] DrivelineSetup referenceManualDriveline()
{
    auto driveline = golfGtiMk7Driveline();

    driveline.gearbox.ratios = {3.77, 2.09, 1.47, 1.15, 0.9360, 0.7760};
    driveline.gearbox.finalDrive = 3.45;
    driveline.gearbox.reverseRatio = 4.55;

    return driveline;
}

// **The curve the mod stated, kept as history rather than as a proposal** — the roles here inverted
// on 2026-08-24, when VW's published curve was promoted into `golfGtiMk7Driveline()` itself
// (`docs/engine-curve-brief.md`). What this file used to hold was the *published* curve as a
// diagnostic against a car carrying the mod's; the car carries the published one now, so the only
// thing worth stating separately is what it replaced.
//
// These are AC's `power.lut` through its own turbo, which the import reproduces at all eighteen
// points: no torque plateau at all, a 349.8 N.m peak at 4500 rpm where the real engine is flat at 370
// from 1600, and a top end that was already right because power is what the mod got from a
// homologation figure.
//
// It is kept so the correction stays *measurable* rather than merely recorded. Delete it and the
// deficit table below becomes a comparison of the car against itself.
[[nodiscard]] Curve modEngineCurve()
{
    const auto atRpm = [](const double rpm, const double torque)
    {
        return glm::dvec2(rpm / radiansPerSecondToRpm, torque);
    };

    return Curve{.points = {atRpm(0.0, 95.0), atRpm(500.0, 129.92491), atRpm(1000.0, 146.745119),
                            atRpm(1500.0, 193.70088), atRpm(1768.0, 235.290486), atRpm(1904.0, 260.621688),
                            atRpm(2000.0, 272.8), atRpm(2500.0, 316.8), atRpm(2650.0, 325.6), atRpm(3000.0, 338.8),
                            atRpm(3500.0, 334.4), atRpm(4000.0, 334.4), atRpm(4500.0, 349.8), atRpm(5000.0, 343.2),
                            atRpm(5500.0, 314.6), atRpm(6300.0, 272.8), atRpm(6500.0, 264.0), atRpm(6800.0, 248.16)}};
}

// A constant driveline efficiency, applied where it is arithmetically identical to one: every newton
// metre the engine makes reaches the wheels multiplied by the ratio and by nothing else in this model,
// so scaling the curve scales the tractive force by exactly the same factor. It is not identical in
// the engine's *inertia* — a real loss does not slow the crank down — but that term is a couple of
// percent of the whole and this is a diagnostic, not a model.
[[nodiscard]] DrivelineSetup withEfficiency(DrivelineSetup driveline, const double efficiency)
{
    for (auto& point : driveline.engine.torque.points)
    {
        point.y *= efficiency;
    }

    return driveline;
}

void report(const char* label, const PullResult& run, const double published)
{
    if (!run.reached)
    {
        std::printf("  %-28s   did not reach the target speed%s\n", label, run.limiter ? " (rev limiter)" : "");
        return;
    }

    if (published > 0.0)
    {
        std::printf("  %-28s %7.2f s   %6.0f-%-6.0f rpm  %7.1f N.m  %7.3f  %6.2f s   %+6.1f%%%s\n", label, run.time,
                    run.startRpm, run.endRpm, run.meanEngineTorque, run.peakDrivenSlip, published,
                    100.0 * (run.time - published) / published, run.limiter ? "  (limiter)" : "");
        return;
    }

    std::printf("  %-28s %7.2f s   %6.0f-%-6.0f rpm  %7.1f N.m  %7.3f        -        -%s\n", label, run.time,
                run.startRpm, run.endRpm, run.meanEngineTorque, run.peakDrivenSlip, run.limiter ? "  (limiter)" : "");
}

} // namespace

TEST_CASE("what an in-gear pull says about the engine, with no traction in it", "[.in-gear]")
{
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    const auto eighty = 80.0 / 3.6;
    const auto onetwenty = 120.0 / 3.6;

    const auto sprung = raceengine::computeMassProperties(setup->sprung);
    REQUIRE(sprung.has_value());

    std::printf("\n=== 80-120 km/h in gear, against auto motor und sport's Supertest sheet ===\n");
    std::printf("  reference: Golf VII GTI Performance, 230 PS / 350 N.m, 6-speed manual, 1406 kg,\n");
    std::printf("             225/40 R 18 — 80-120 in 5.1 / 6.3 / 7.8 s in 4th / 5th / 6th\n");
    std::printf("  this car:  %.0f kg, VW's published curve since 2026-08-24 — 370 N.m 1600-4300,\n",
                sprung->mass + setup->unsprungMass());
    std::printf("             180 kW 5000-6200, quoted 245 PS\n");
    std::printf("  expected:  ours is lighter and stronger, so it should beat every row by about\n");
    std::printf("             (1348/1406)*(350/370) = 0.907, i.e. roughly -9%%\n");

    std::printf("\n--- our engine and mass, through the reference car's gearing ---\n");
    std::printf("  gear                            time      rpm band       mean T   pk slip  published    error\n");

    {
        const auto driveline = referenceManualDriveline();

        for (const auto gear : {4, 5, 6})
        {
            const auto published = gear == 4 ? 5.1 : (gear == 5 ? 6.3 : 7.8);
            const auto run = pullInGear(setup.value(), driveline, world.value(), gear, eighty, onetwenty);

            char label[64];
            std::snprintf(label, sizeof(label), "reference box, %d%s", gear,
                          gear == 4 ? "th" : (gear == 5 ? "th" : "th"));
            report(label, run, published);
        }
    }

    // **One of the two known omissions is closed and the other is not, so this block is now a
    // before-and-after rather than a pair of corrections.**
    //
    //   1. The engine curve matched the car's *power* and not its *torque* — right at the top of the
    //      range and increasingly wrong below it, which is precisely where an in-gear pull in a tall
    //      gear lives. **Closed 2026-08-24**, `docs/engine-curve-brief.md`. The mod's curve is kept in
    //      `modEngineCurve()` so the correction stays measurable, and it is the first pair of rows.
    //   2. There is still no driveline efficiency anywhere in this model. `stepDriveline` multiplies
    //      engine torque by the ratio and hands the product to the wheels; a real transverse
    //      front-drive transaxle loses about 8 to 14% of it in the gearset, the final drive and the
    //      joints. Scaling the curve is exactly equivalent to a constant efficiency for the tractive
    //      force, which is why that is how it is stood in for here — it is **not** equivalent for the
    //      engine's own inertia, so this remains a diagnostic and not a model.
    //
    // **Read the spread across gears, not the offset.** A wrong scalar — efficiency, mass, drag, or
    // the reference car itself — moves all three gears together; a wrong curve moves them apart,
    // because each gear samples a different part of the rev band. The spread is what the curve owns.
    std::printf("\n--- the same gearing: the curve that was replaced, and the one that replaced it ---\n");
    std::printf("  gear                            time      rpm band       mean T   pk slip  published    error\n");

    {
        auto legacy = referenceManualDriveline();
        legacy.engine.torque = modEngineCurve();
        const auto lossy = withEfficiency(legacy, 0.92);

        for (const auto gear : {4, 5, 6})
        {
            const auto published = gear == 4 ? 5.1 : (gear == 5 ? 6.3 : 7.8);
            const auto run = pullInGear(setup.value(), lossy, world.value(), gear, eighty, onetwenty);

            char label[64];
            std::snprintf(label, sizeof(label), "the mod's curve x 0.92, %dth", gear);
            report(label, run, published);
        }

        const auto both = withEfficiency(referenceManualDriveline(), 0.92);

        for (const auto gear : {4, 5, 6})
        {
            const auto reference = gear == 4 ? 5.1 : (gear == 5 ? 6.3 : 7.8);
            const auto run = pullInGear(setup.value(), both, world.value(), gear, eighty, onetwenty);

            char label[64];
            std::snprintf(label, sizeof(label), "shipped curve x 0.92, %dth", gear);
            report(label, run, reference);
        }
    }

    std::printf("\n--- what the car makes now against what the mod stated ---\n");
    std::printf("    rpm     this model     the mod's     correction\n");

    {
        const auto ours = golfGtiMk7Driveline().engine.torque;
        const auto legacy = modEngineCurve();

        for (const auto rpm : {1600.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4300.0, 5000.0, 5500.0, 6200.0})
        {
            const auto speed = rpm / radiansPerSecondToRpm;
            const auto mine = ours.at(speed);
            const auto theirs = legacy.at(speed);

            std::printf("   %5.0f    %8.1f N.m  %8.1f N.m   %+7.1f%%\n", rpm, mine, theirs,
                        100.0 * (mine - theirs) / theirs);
        }
    }

    std::printf("\n--- the car as it ships, in its own seven ratios ---\n");
    std::printf("  gear                            time      rpm band       mean T   pk slip  published    error\n");

    {
        const auto driveline = golfGtiMk7Driveline();

        for (auto gear = 1; gear <= driveline.gearbox.topGear(); gear++)
        {
            const auto run = pullInGear(setup.value(), driveline, world.value(), gear, eighty, onetwenty);

            char label[64];
            std::snprintf(label, sizeof(label), "shipped box, gear %d (x%.3f)", gear,
                          driveline.gearbox.reduction(gear));
            report(label, run, 0.0);
        }
    }
}

TEST_CASE("where each gearbox puts the engine at a given road speed", "[.in-gear]")
{
    // The gearing itself, before any pull. An in-gear time is the engine curve read through a ratio,
    // so two boxes that put the engine in different places at 120 km/h are asking the curve different
    // questions — and a comparison that does not print this is comparing gear *labels*.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto radius = setup->corners.front().hardpoints.wheelRadius;

    const auto table = [radius](const char* name, const DrivelineSetup& driveline)
    {
        std::printf("\n  %s\n", name);
        std::printf("    gear   overall   km/h per 1000 rpm   rpm at 80    rpm at 120\n");

        for (auto gear = 1; gear <= driveline.gearbox.topGear(); gear++)
        {
            const auto reduction = driveline.gearbox.reduction(gear);
            const auto perThousand = 1000.0 / radiansPerSecondToRpm / reduction * radius * 3.6;

            std::printf("    %2d    %7.3f   %14.2f      %7.0f       %7.0f\n", gear, reduction, perThousand,
                        80.0 / perThousand * 1000.0, 120.0 / perThousand * 1000.0);
        }
    };

    std::printf("\n=== what each ratio asks of the engine ===\n");
    table("the car as it ships (AC drivetrain.ini, one final drive of 4.37)", golfGtiMk7Driveline());
    table("auto motor und sport's test car (6-speed manual, 3.45 / 2.76)", referenceManualDriveline());
}

TEST_CASE("what the driveline's two omissions are worth to the number the brief rests on", "[.in-gear]")
{
    // The in-gear case above says the engine curve is wrong. **This says what that costs 0-100**,
    // which is the only reason anyone cares: `docs/tyre-grip-ratio-brief.md`'s whole tension is that a
    // longitudinal peak satisfying the 100-0 puts 0-100 under its published band, and that argument is
    // only as good as 0-100 being a measurement of grip rather than of the engine.
    //
    // Same launch three times, changing nothing but the driveline: as it ships, with a 92% efficiency,
    // and with VW's published torque curve behind that efficiency.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    const auto launch = [&](const char* label, const DrivelineSetup& driveline)
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, designHeight(setup.value()), 20.0);

        for (auto step = 0; step < 720; step++)
        {
            REQUIRE(stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        auto drivelineState = DrivelineState{};
        startEngine(driveline, drivelineState);

        const auto inertias = wheelInertias(setup.value());
        auto road = noRoadTorque;

        // Idling in gear on the brakes, which is what a car about to be launched has been doing. A
        // fixture that starts the engine and goes straight to full throttle spends its first quarter
        // second dumping torque through a clutch the automation is still crawling open, and every
        // conclusion drawn from that is about the fixture.
        {
            auto idling = VehicleInput{};
            idling.brake = 1.0;
            idling.gear = 1;

            for (auto step = 0; step < 360; step++)
            {
                const auto torques = stepDriveline(driveline, drivelineState,
                                                   {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                    state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                                   inertias, road, idling, tick);
                REQUIRE(torques.has_value());

                const auto stepped = stepVehicle(setup.value(), state, idling, torques->wheel, world.value(), tick);
                REQUIRE(stepped.has_value());
                road = roadTorques(stepped.value());
            }
        }

        // **The five preconditions a launch fixture has to assert**, and the reason they are here is
        // that their absence once cost an entire TCU build: stationary, in gear, on the brakes, at
        // idle, with the clutch open.
        CAPTURE(drivelineState.clutchPedal, drivelineState.engineSpeed, drivelineState.gear);
        REQUIRE(glm::length(state.chassis.linearVelocity) < 0.05);
        REQUIRE(drivelineState.gear == 1);
        REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
        REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);
        REQUIRE(drivelineState.clutchPedal > 0.85);
        for (const auto& corner : state.corners)
        {
            REQUIRE(std::abs(corner.wheelSpeed) < 0.5);
        }

        auto input = VehicleInput{};
        input.throttle = 1.0;
        input.gear = 1;

        // Road speed through the gear, at 0.93 of the limiter. Established in `[.vehicle-delta]`: a
        // schedule taken off the driven wheels holds first gear for ever once the fronts are spinning,
        // because the engine is at the limiter while the car is doing nine kilometres an hour.
        const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

        auto reached = -1.0;
        auto topGearUsed = 1;

        for (auto step = 1; step <= 360 * 20; step++)
        {
            const auto roadSideSpeed = std::abs(state.chassis.linearVelocity.z) /
                                       setup->corners.front().hardpoints.wheelRadius *
                                       driveline.gearbox.reduction(input.gear);

            if (roadSideSpeed > upshiftSpeed && input.gear < driveline.gearbox.topGear())
            {
                input.gear++;
                topGearUsed = std::max(topGearUsed, input.gear);
            }

            const auto torques = stepDriveline(driveline, drivelineState,
                                               {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                               inertias, road, input, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup.value(), state, input, torques->wheel, world.value(), tick);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());

            if (state.chassis.linearVelocity.z >= 100.0 / 3.6)
            {
                reached = static_cast<double>(step) * tick;
                break;
            }
        }

        if (reached < 0.0)
        {
            std::printf("  %-34s   never reached 100 km/h\n", label);
            return;
        }

        std::printf("  %-34s %7.3f s   through gear %d   %s\n", label, reached, topGearUsed,
                    reached < 6.5 ? "under the measured range" : (reached > 6.6 ? "over it" : "inside 6.5-6.6"));
    };

    std::printf("\n=== 0-100 km/h, against a MEASURED 6.5-6.6 s (n=2; see the validation brief) ===\n");
    launch("as it ships", golfGtiMk7Driveline());
    launch("with a 92% efficient driveline", withEfficiency(golfGtiMk7Driveline(), 0.92));

    {
        auto legacy = golfGtiMk7Driveline();
        legacy.engine.torque = modEngineCurve();
        launch("the mod's curve", legacy);
        launch("the mod's curve, 92% efficient", withEfficiency(legacy, 0.92));
    }
}
