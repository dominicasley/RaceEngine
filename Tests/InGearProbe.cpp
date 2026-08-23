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
    drivelineState.shaftSpeed = entry / radius * driveline.gearbox.finalDrive;

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
            CAPTURE(gear, drivelineState.gear, drivelineState.clutchPedal, drivelineState.engineSpeed, speed);

            REQUIRE(drivelineState.gear == gear);
            REQUIRE(drivelineState.shiftPhase == ShiftPhase::Engaged);
            REQUIRE(drivelineState.clutchPedal < 0.05);
            REQUIRE(std::abs(torques->clutchSlip) < 5.0);
            REQUIRE(std::abs(state.chassis.position.x) < 0.5);
            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                REQUIRE(stepped->telemetry.wheels[index].inContact);
            }

            // And the engine really is geared to the *wheels*: the chain from crank to hub is locked,
            // so what turns is one body and the ratio is the whole of the relationship.
            //
            // Against the wheels rather than against the road, which is the distinction that matters
            // here. A driven tyre transmits force by slipping, so at full throttle the hub runs a few
            // percent ahead of the road however healthy the driveline is — measured, 3.4% in fourth.
            // Asserting the engine against *road* speed would fold that slip into a driveline check
            // and would tighten or loosen with grip, which is the one thing this probe must not be
            // sensitive to.
            const auto driven = 0.5 * (state.corners[0].wheelSpeed + state.corners[1].wheelSpeed);
            REQUIRE(drivelineState.engineSpeed == Catch::Approx(driven * reduction).epsilon(0.005));

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

// The engine VW publishes, as a curve. **Two statements and nothing invented between them**: 370 N.m
// from 1600 to 4300 rpm, and 180 kW from 5000 to 6200 — which is the whole of what a manufacturer
// homologates and is exactly the shape a wastegated turbo makes. Torque between 4300 and 5000 is the
// straight line joining the two, and above 6200 it tapers to the mod's own figure at the limiter,
// because nothing published says where it ends.
//
// Below 1600 rpm nothing is stated and the mod's own curve is kept. No pull in this file goes below
// 1824 rpm, so that stretch decides nothing here and is not a place to be inventing data.
//
// **This is a diagnostic, not a proposal.** It exists to answer one question — how much of the gap to
// the published in-gear times is the torque curve — and replacing the car's engine is a different
// piece of work with its own brief.
[[nodiscard]] Curve publishedEngineCurve()
{
    const auto atRpm = [](const double rpm, const double torque)
    {
        return glm::dvec2(rpm / radiansPerSecondToRpm, torque);
    };

    const auto atPower = [&atRpm](const double rpm, const double watts)
    {
        return atRpm(rpm, watts / (rpm / radiansPerSecondToRpm));
    };

    return Curve{.points = {atRpm(0.0, 95.0), atRpm(500.0, 129.92491), atRpm(1000.0, 146.745119), atRpm(1600.0, 370.0),
                            atRpm(4300.0, 370.0), atPower(5000.0, 180000.0), atPower(5500.0, 180000.0),
                            atPower(6200.0, 180000.0), atRpm(6500.0, 264.0), atRpm(6800.0, 248.16)}};
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
    std::printf("  this car:  %.0f kg, engine curve peaking at 349.8 N.m, quoted 245 PS / 370 N.m\n",
                sprung->mass + setup->unsprungMass());

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

    // **Two things are known to be missing, and both are quantified here rather than argued about.**
    //
    //   1. There is no driveline efficiency anywhere in this model. `stepDriveline` multiplies engine
    //      torque by the ratio and hands the product to the wheels; a real transverse front-drive
    //      manual loses about 8 to 12% of it in the gearset, the final drive and the joints. Scaling
    //      the curve is exactly equivalent to a constant efficiency for the tractive force, which is
    //      why that is how it is measured.
    //   2. The engine curve matches the car's *power* and not its *torque*. It peaks at 349.8 N.m
    //      where VW states 370, and it reaches that peak at 4500 rpm where the real one is flat from
    //      1600 — so it is right at the top of the range and increasingly wrong below it, which is
    //      precisely where an in-gear pull in a tall gear lives.
    std::printf("\n--- the same gearing, with the two known omissions put back ---\n");
    std::printf("  gear                            time      rpm band       mean T   pk slip  published    error\n");

    {
        const auto lossy = withEfficiency(referenceManualDriveline(), 0.92);

        for (const auto gear : {4, 5, 6})
        {
            const auto published = gear == 4 ? 5.1 : (gear == 5 ? 6.3 : 7.8);
            const auto run = pullInGear(setup.value(), lossy, world.value(), gear, eighty, onetwenty);

            char label[64];
            std::snprintf(label, sizeof(label), "our curve x 0.92, %dth", gear);
            report(label, run, published);
        }

        auto published = referenceManualDriveline();
        published.engine.torque = publishedEngineCurve();
        const auto both = withEfficiency(published, 0.92);

        for (const auto gear : {4, 5, 6})
        {
            const auto reference = gear == 4 ? 5.1 : (gear == 5 ? 6.3 : 7.8);
            const auto run = pullInGear(setup.value(), both, world.value(), gear, eighty, onetwenty);

            char label[64];
            std::snprintf(label, sizeof(label), "VW's curve x 0.92, %dth", gear);
            report(label, run, reference);
        }
    }

    std::printf("\n--- what this car's engine makes against what VW says it makes ---\n");
    std::printf("    rpm     this model    VW published    deficit\n");

    {
        const auto ours = golfGtiMk7Driveline().engine.torque;
        const auto stated = publishedEngineCurve();

        for (const auto rpm : {1600.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4300.0, 5000.0, 5500.0, 6200.0})
        {
            const auto speed = rpm / radiansPerSecondToRpm;
            const auto mine = ours.at(speed);
            const auto theirs = stated.at(speed);

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
                                       setup->corners.front().hardpoints.wheelRadius * driveline.gearbox.finalDrive *
                                       driveline.gearbox.ratio(input.gear);

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
                    reached < 6.4 ? "under the band" : (reached > 6.7 ? "over the band" : "inside 6.4-6.7"));
    };

    std::printf("\n=== 0-100 km/h, against the published 6.4-6.7 s for a DSG without launch control ===\n");
    launch("as it ships", golfGtiMk7Driveline());
    launch("with a 92% efficient driveline", withEfficiency(golfGtiMk7Driveline(), 0.92));

    {
        auto published = golfGtiMk7Driveline();
        published.engine.torque = publishedEngineCurve();
        launch("VW's curve, 92% efficient", withEfficiency(published, 0.92));
    }
}
