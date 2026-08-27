// What a change to the car's *data* did to the car's *behaviour*. `./EngineTests "[.vehicle-delta]"`.
//
// The V1/V2 corrections — wheel radius from a published fitment, unsprung mass from a component
// build-up — are one-line edits with consequences everywhere, and the brief that asked for them is
// explicit that the consequences are the deliverable rather than the edits. So this prints the
// quantities the M1 criteria are *made of* rather than asserting they still hold: the criteria
// themselves are in `GolfGtiTests` and stay pass/fail, and this is the instrument that says by how
// much and in which direction.
//
// Every number here is printed and none is asserted, deliberately. A probe that failed would have to
// carry an expected value, and an expected value for "what should the wheel hop frequency be after
// halving the unsprung mass" is precisely the thing being measured.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::computeMassProperties;
using raceengine::cornerCount;
using raceengine::CornerSide;
using raceengine::DrivelineState;
using raceengine::evaluateTyre;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TyreSlip;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto gravity = 9.80665;
constexpr auto degrees = 57.29577951308232;

// What the road said last tick, before the tyre has answered. Named here for the same reason
// `DrivelineTests` names it: a bare `{}` at the call site reads as "no argument" rather than as
// "zero road torque".
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

ProvingGroundDescriptor plate(const double size = 400.0)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

// The design height is the car's own now rather than a constant this file states: V2 moves the
// sprung centre and V1 moves every wheel centre, so a fixture that spawned the car at a hard-coded
// 0.572 would be dropping it from a different height after each change and calling the difference a
// result.
[[nodiscard]] double designHeight(const VehicleSetup& setup)
{
    auto mass = setup.unsprungMass();
    auto moment = 0.0;

    for (const auto& component : setup.sprung)
    {
        mass += component.mass;
        moment += component.mass * component.centre.y;
    }

    for (const auto& corner : setup.corners)
    {
        moment += corner.unsprungMass * corner.hardpoints.wheelCentre.y;
    }

    return mass > 0.0 ? moment / mass : 0.0;
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ = 20.0)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        state.corners[index].wheelSpeed = speed / setup.corners[index].hardpoints.wheelRadius;
    }
}

struct SteadyState
{
    double lateralAcceleration = 0.0;
    double yawRate = 0.0;
    double speed = 0.0;
};

SteadyState hold(const VehicleSetup& setup, const PhysicsWorld& world, const double steering, const double speed)
{
    auto state = VehicleState{};
    settle(setup, state, world, speed);

    auto input = VehicleInput{};
    input.steering = steering;

    auto result = SteadyState{};
    auto samples = 0;

    for (auto step = 0; step < 1440; step++)
    {
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick);
        REQUIRE(stepped.has_value());

        if (step >= 1080)
        {
            constexpr auto toTheRight = outboardSign(CornerSide::Right);

            result.lateralAcceleration += stepped->telemetry.acceleration.x * toTheRight;
            result.yawRate += stepped->telemetry.yawRate * toTheRight;
            result.speed += glm::length(state.chassis.linearVelocity);
            samples++;
        }
    }

    result.lateralAcceleration /= static_cast<double>(samples) * gravity;
    result.yawRate /= static_cast<double>(samples);
    result.speed /= static_cast<double>(samples);

    return result;
}

} // namespace

TEST_CASE("what the corrected data did to the car", "[.vehicle-delta]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto setup = built.value();
    const auto driveline = golfGtiMk7Driveline();

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    // --- the ledger, as assembled ---

    std::printf("\n=== mass ledger ===\n");

    const auto properties = computeMassProperties(setup.sprung);
    REQUIRE(properties.has_value());

    const auto unsprung = setup.unsprungMass();
    const auto total = properties->mass + unsprung;

    std::printf("  total mass            %10.3f kg\n", total);
    std::printf("  unsprung, all four    %10.3f kg   (%.1f%% of the car)\n", unsprung, 100.0 * unsprung / total);
    std::printf("  unsprung front / rear %10.3f / %.3f kg per corner\n", setup.corners[0].unsprungMass,
                setup.corners[2].unsprungMass);
    std::printf("  sprung mass           %10.3f kg\n", properties->mass);
    std::printf("  sprung centre         %10.4f m up, %.4f m along\n", properties->centreOfMass.y,
                properties->centreOfMass.z);

    // The assembled car: the sprung component plus the four unsprung masses at their wheel centres,
    // which is how `stepVehicle` puts it together. This is the number that must not move when the
    // split changes, and printing it beside the sprung one is what makes that visible.
    {
        auto ledger = std::vector<raceengine::MassComponent>{setup.sprung.front()};
        for (const auto& corner : setup.corners)
        {
            ledger.push_back(raceengine::MassComponent{
                .mass = corner.unsprungMass, .centre = corner.hardpoints.wheelCentre, .inertia = glm::dmat3(0.0)});
        }

        const auto assembled = computeMassProperties(ledger);
        REQUIRE(assembled.has_value());

        std::printf("  ASSEMBLED centre      %10.4f m up, %.4f m along\n", assembled->centreOfMass.y,
                    assembled->centreOfMass.z);
        std::printf("  ASSEMBLED inertia     pitch %.1f  yaw %.1f  roll %.1f kg.m2\n", assembled->inertia[0][0],
                    assembled->inertia[1][1], assembled->inertia[2][2]);
        std::printf("  shell inertia         pitch %.1f  yaw %.1f  roll %.1f kg.m2\n", properties->inertia[0][0],
                    properties->inertia[1][1], properties->inertia[2][2]);
    }

    std::printf("  wheel radius          %10.4f m\n", setup.corners[0].hardpoints.wheelRadius);

    // --- the two frequencies the split sets ---
    //
    // Hand calculations rather than measurements, and stated as such: the ride frequency is the
    // sprung corner on its wheel rate, the hop frequency is the unsprung corner between the wheel
    // rate and the tyre's carcass rate. Both are the textbook two-mass quarter car, which is exactly
    // the model whose parameters just changed.

    std::printf("\n=== quarter-car frequencies, by hand ===\n");

    for (const auto axle : {std::size_t{0}, std::size_t{2}})
    {
        const auto& corner = setup.corners[axle];

        // Back to the wheel from the shaft, which is where `springRate` is stated — through the
        // spring element's own ratio.
        const auto spring =
            raceengine::solveSpringKinematics(corner.hardpoints, raceengine::springElementOf(corner.hardpoints), 0.0);
        REQUIRE(spring.has_value());
        const auto ratio = std::abs(spring->motionRatio);
        const auto wheelRate = corner.springRate * ratio * ratio;

        // Sprung mass on this corner, by statics about the other axle.
        const auto sprungHere =
            axle == 0
                ? properties->mass * (properties->centreOfMass.z - setup.corners[2].hardpoints.wheelCentre.z) /
                      (setup.corners[0].hardpoints.wheelCentre.z - setup.corners[2].hardpoints.wheelCentre.z) / 2.0
                : properties->mass * (setup.corners[0].hardpoints.wheelCentre.z - properties->centreOfMass.z) /
                      (setup.corners[0].hardpoints.wheelCentre.z - setup.corners[2].hardpoints.wheelCentre.z) / 2.0;

        const auto ride = std::sqrt(wheelRate / sprungHere) / (2.0 * 3.14159265358979323846);
        const auto hop =
            std::sqrt((wheelRate + corner.tireVerticalRate) / corner.unsprungMass) / (2.0 * 3.14159265358979323846);

        std::printf("  %-5s wheel rate %8.0f N/m, sprung corner %7.2f kg -> ride %.3f Hz\n",
                    axle == 0 ? "front" : "rear", wheelRate, sprungHere, ride);
        std::printf("        tyre rate  %8.0f N/m, unsprung      %7.2f kg -> hop  %.3f Hz\n", corner.tireVerticalRate,
                    corner.unsprungMass, hop);
    }

    // --- criterion 5: the skidpad ---

    std::printf("\n=== criterion 5, skidpad (coasting, 20 m/s entry) ===\n");
    std::printf("  %8s %10s %10s %10s\n", "steering", "lat g", "yaw rad/s", "speed m/s");

    const auto steerings = std::vector<double>{0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.22, 0.30, 0.45, 0.60};
    auto measured = std::vector<SteadyState>{};

    for (const auto steering : steerings)
    {
        measured.push_back(hold(setup, world.value(), steering, 20.0));
        std::printf("  %8.2f %10.4f %10.4f %10.3f\n", steering, measured.back().lateralAcceleration,
                    measured.back().yawRate, measured.back().speed);
    }

    auto peak = 0.0;
    for (const auto& sample : measured)
    {
        peak = std::max(peak, sample.lateralAcceleration);
    }
    std::printf("  peak lateral %.4f g, last sample %.4f g\n", peak, measured.back().lateralAcceleration);

    // --- criterion 7: the bar ---

    std::printf("\n=== criterion 7, rear bar against yaw rate at 0.11 steering ===\n");

    const auto withRearBar = [&world](const double rate)
    {
        auto rebuilt = golfGtiMk7();
        REQUIRE(rebuilt.has_value());

        rebuilt->corners[2].antiRollRate = rate;
        rebuilt->corners[3].antiRollRate = rate;

        return hold(rebuilt.value(), world.value(), 0.11, 20.0);
    };

    const auto soft = withRearBar(0.0);
    const auto medium = withRearBar(15000.0);
    const auto stiff = withRearBar(40000.0);

    std::printf("  no bar   yaw %.5f rad/s, lat %.4f g\n", soft.yawRate, soft.lateralAcceleration);
    std::printf("  15000    yaw %.5f rad/s, lat %.4f g\n", medium.yawRate, medium.lateralAcceleration);
    std::printf("  40000    yaw %.5f rad/s, lat %.4f g\n", stiff.yawRate, stiff.lateralAcceleration);
    std::printf("  stiff/soft %.4f  (the criterion wants > 1.05)\n", stiff.yawRate / soft.yawRate);

    // --- criterion 6: the step steer ---

    std::printf("\n=== criterion 6, step steer 0.06 at 25 m/s ===\n");
    {
        auto state = VehicleState{};
        settle(setup, state, world.value(), 25.0);

        auto input = VehicleInput{};
        input.steering = 0.06;

        auto history = std::vector<double>{};
        for (auto step = 0; step < 1080; step++)
        {
            const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());
            history.push_back(stepped->telemetry.yawRate * outboardSign(CornerSide::Right));
        }

        auto settledYaw = 0.0;
        for (auto index = history.size() - 180; index < history.size(); index++)
        {
            settledYaw += history[index];
        }
        settledYaw /= 180.0;

        auto riseTicks = std::size_t{0};
        while (riseTicks < history.size() && history[riseTicks] < settledYaw * 0.9)
        {
            riseTicks++;
        }

        const auto tail = std::vector<double>(history.begin() + 540, history.end());
        const auto overshoot = *std::max_element(history.begin(), history.end()) / settledYaw;
        const auto ripple =
            (*std::max_element(tail.begin(), tail.end()) - *std::min_element(tail.begin(), tail.end())) / settledYaw;

        std::printf("  settled yaw   %.5f rad/s\n", settledYaw);
        std::printf("  rise to 90%%   %.4f s   (criterion wants 0.05 to 0.60)\n",
                    static_cast<double>(riseTicks) * tick);
        std::printf("  overshoot     %.4f x   (criterion wants < 1.35)\n", overshoot);
        std::printf("  tail ripple   %.5f x   (criterion wants < 0.05)\n", ripple);
    }

    // --- criterion 9: the coastdown ---

    std::printf("\n=== criterion 9, coastdown from 27.78 m/s ===\n");
    {
        auto descriptor = plate(3000.0);
        descriptor.cellSize = 4.0;
        const auto straight = PhysicsWorld::create(generateProvingGround(descriptor).value());
        REQUIRE(straight.has_value());

        auto state = VehicleState{};
        settle(setup, state, straight.value(), 27.78);

        auto speeds = std::vector<double>{};
        auto decelerations = std::vector<double>{};
        auto previous = 27.78;

        for (auto step = 1; step <= 10800; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, straight.value(), tick).has_value());

            if (step % 360 == 0)
            {
                const auto speed = state.chassis.linearVelocity.z;
                speeds.push_back(0.5 * (previous + speed));
                decelerations.push_back(previous - speed);
                previous = speed;
            }
        }

        auto sumX = 0.0;
        auto sumY = 0.0;
        auto sumXX = 0.0;
        auto sumXY = 0.0;
        const auto count = static_cast<double>(speeds.size());

        for (auto index = std::size_t{0}; index < speeds.size(); index++)
        {
            const auto x = speeds[index] * speeds[index];
            sumX += x;
            sumY += decelerations[index];
            sumXX += x * x;
            sumXY += x * decelerations[index];
        }

        const auto drag = (count * sumXY - sumX * sumY) / (count * sumXX - sumX * sumX);
        const auto rolling = (sumY - drag * sumX) / count;

        auto rotational = 0.0;
        for (const auto& corner : setup.corners)
        {
            const auto radius = corner.hardpoints.wheelRadius;
            rotational += corner.wheelInertia / (radius * radius);
        }

        auto dragArea = 0.0;
        for (const auto& surface : setup.aero)
        {
            dragArea += surface.dragArea;
        }

        const auto effectiveMass = total + rotational;

        std::printf("  rotational equivalent %.2f kg  (wheel inertia / r^2, all four)\n", rotational);
        std::printf("  Crr  fitted %.5f  against %.5f specified\n", rolling * effectiveMass / (total * gravity),
                    setup.corners.front().rollingResistance);
        std::printf("  CdA  fitted %.4f  against %.4f specified\n", drag * effectiveMass / (0.5 * setup.airDensity),
                    dragArea);
    }

    // --- the driveline, end to end ---
    //
    // Full throttle from rest, shifting at the limiter, on a long straight. This is the fixture the
    // remediation brief asked for and did not have: a 3.5% error in the effective final drive is
    // directly visible in a 0-100 time and in a terminal speed, and in nothing else the suite runs.

    std::printf("\n=== acceleration and terminal speed, full throttle, shifting at the limiter ===\n");
    {
        // Long and narrow rather than a square, and the car starts at one end. The generator lays the
        // ground out from z = 0 to z = length — it is not centred on the origin — so a car spawned at
        // a negative station is a car dropped off the end of the world, which reads as a car that
        // will not accelerate.
        auto descriptor = plate(12000.0);
        descriptor.width = 40.0;
        descriptor.cellSize = 4.0;
        const auto straight = PhysicsWorld::create(generateProvingGround(descriptor).value());
        REQUIRE(straight.has_value());

        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);
        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, straight.value(), tick).has_value());
        }

        auto drivelineState = DrivelineState{};
        startEngine(driveline, drivelineState);

        const auto inertias = wheelInertias(setup);
        auto road = noRoadTorque;

        auto input = VehicleInput{};
        input.throttle = 1.0;
        input.gear = 1;

        // The shift schedule, and all three obvious sources for it are wrong on this car in
        // different ways. Worth writing down, because each failure looks like a result:
        //
        //   - **the engine alone** flares against the slipping auto-clutch while the car has not
        //     moved, so it shifts first to top within six ticks and the car never leaves the line;
        //   - **the driven wheels** are *spinning* — see the trace below — so it chases the spin and
        //     reaches fifth at 10 m/s;
        //   - **road speed through the gear** is immune to both and still fails, because with the
        //     fronts spinning and the engine on its limiter the road speed asymptotes at 15.9 m/s
        //     and never reaches the 16.3 the first-to-second point sits at. The car sits in first at
        //     the limiter for two minutes, which reads as a gearbox fault and is a tyre one.
        //
        // What ships is the engine, gated on the clutch being **engaged** — which is what a driver
        // reading a tacho actually does, and the one rule that distinguishes a launch flare (clutch
        // slipping, do not shift) from a car genuinely at the limiter (clutch locked, shift). It
        // honours wheelspin rather than hiding it: a real driver whose wheels are spinning at the
        // limiter changes up too.
        const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.995;

        auto hundred = -1.0;
        auto sixty = -1.0;
        auto peakSpeed = 0.0;
        auto lastGear = 1;

        constexpr auto seconds = 120;
        for (auto step = 1; step <= seconds * 360; step++)
        {
            const auto driven = 0.5 * (state.corners[0].wheelSpeed + state.corners[1].wheelSpeed);
            if (drivelineState.engineSpeed > upshiftSpeed && drivelineState.clutchPedal < 0.02 &&
                input.gear < driveline.gearbox.topGear())
            {
                input.gear++;
            }

            const auto torques = stepDriveline(driveline, drivelineState,
                                               {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                               inertias, road, input, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, input, torques->wheel, straight.value(), tick);
            REQUIRE(stepped.has_value());

            road = roadTorques(stepped.value());

            const auto speed = state.chassis.linearVelocity.z;
            peakSpeed = std::max(peakSpeed, speed);
            lastGear = input.gear;

            // The first four seconds, once a half second, with what the driven wheels are doing
            // beside the road speed. A front-drive hatch at full throttle from rest is
            // traction-limited and not power-limited, so a 0-100 that disagrees with the published
            // figure is answered here rather than by the headline number: if the fronts are turning
            // far faster than the road, the fixture is measuring wheelspin.
            if (step % 180 == 0 && step <= 360 * 4)
            {
                const auto wheelSurface = driven * setup.corners[0].hardpoints.wheelRadius;
                std::printf("    t=%4.1f s  gear %d  road %6.2f m/s  driven wheels %6.2f m/s  "
                            "slip x%.2f  engine %5.0f rpm\n",
                            static_cast<double>(step) * tick, input.gear, speed, wheelSurface,
                            speed > 0.5 ? wheelSurface / speed : 0.0, drivelineState.engineSpeed * 9.549296585513721);
            }

            const auto elapsed = static_cast<double>(step) * tick;
            if (sixty < 0.0 && speed >= 26.8224)
            {
                sixty = elapsed;
            }
            if (hundred < 0.0 && speed >= 27.7778)
            {
                hundred = elapsed;
            }
        }

        std::printf("  0-100 kph   %s\n", hundred < 0.0 ? "never reached" : (std::to_string(hundred) + " s").c_str());
        std::printf("  0-60 mph    %s\n", sixty < 0.0 ? "never reached" : (std::to_string(sixty) + " s").c_str());
        std::printf("  terminal    %.3f m/s = %.2f kph after %d s, in gear %d\n", peakSpeed, peakSpeed * 3.6, seconds,
                    lastGear);
        std::printf("  engine at terminal %.0f rpm\n", drivelineState.engineSpeed * 9.549296585513721);

        // What the gearing says the terminal speed *should* be if the engine were on the limiter in
        // top: a hand check on the same 3.5% the radius carries.
        const auto topRatio = driveline.gearbox.reduction(driveline.gearbox.topGear());
        std::printf("  geared speed at limiter in top %.2f kph  (radius %.4f m, top reduction %.3f)\n",
                    driveline.engine.limiterSpeed / topRatio * setup.corners[0].hardpoints.wheelRadius * 3.6,
                    setup.corners[0].hardpoints.wheelRadius, topRatio);

        // --- and the same thing rolling, which is the measurement that can see the gearing ---
        //
        // The standing start above is traction-limited, so it is a measurement of the tyre and of
        // the differential and says almost nothing about the final drive. A rolling pull in a fixed
        // gear is the opposite: the tyre is nowhere near its limit, so the whole of the answer is
        // engine torque against gearing, inertia and drag. **This is where a 3.5% error in effective
        // final drive is actually visible.**
        //
        // 80 to 120 km/h in a stated gear rather than a figure of this fixture's own invention,
        // because that is the elasticity figure European manufacturers and magazines publish and so
        // is the one with something to check against: AutoBild Sportscars measured this car at
        // **6.5 s in fifth and 8.4 s in sixth**.
        constexpr auto eighty = 22.2222;
        constexpr auto oneTwenty = 33.3333;

        for (const auto gear : {4, 5, 6})
        {
            auto rolling = VehicleState{};
            settle(setup, rolling, straight.value(), eighty, 20.0);

            auto rollingDriveline = DrivelineState{};
            startEngine(driveline, rollingDriveline);
            // Placed rather than left at idle: writing a speed into the state without also placing
            // the shaft is the fixture fault `placeDriveline` exists to prevent, and it shows up as
            // a five-figure torque on the first tick.
            raceengine::placeDriveline(driveline, rollingDriveline,
                                       eighty / setup.corners[0].hardpoints.wheelRadius *
                                           driveline.gearbox.reduction(gear));

            auto rollingRoad = noRoadTorque;

            auto pull = VehicleInput{};
            pull.throttle = 1.0;
            pull.gear = gear;

            auto reached = -1.0;
            for (auto step = 1; step <= 60 * 360; step++)
            {
                const auto torques = stepDriveline(driveline, rollingDriveline,
                                                   {rolling.corners[0].wheelSpeed, rolling.corners[1].wheelSpeed,
                                                    rolling.corners[2].wheelSpeed, rolling.corners[3].wheelSpeed},
                                                   inertias, rollingRoad, pull, tick);
                REQUIRE(torques.has_value());

                const auto stepped = stepVehicle(setup, rolling, pull, torques->wheel, straight.value(), tick);
                REQUIRE(stepped.has_value());

                rollingRoad = roadTorques(stepped.value());

                if (rolling.chassis.linearVelocity.z >= oneTwenty)
                {
                    reached = static_cast<double>(step) * tick;
                    break;
                }
            }

            const auto rpmAt = [&driveline, &setup, gear](const double roadSpeed)
            {
                return roadSpeed / setup.corners[0].hardpoints.wheelRadius * driveline.gearbox.reduction(gear) *
                       9.549296585513721;
            };

            std::printf("  rolling 80-120 kph in gear %d: %-12s (engine %4.0f -> %4.0f rpm)\n", gear,
                        reached < 0.0 ? "never reached" : (std::to_string(reached).substr(0, 5) + " s").c_str(),
                        rpmAt(eighty), rpmAt(oneTwenty));
        }
    }

    // --- the vertical criteria, which are the ones a mass-split change can actually reach ---
    //
    // Criteria 5, 6 and 7 are steady-state cornering on a flat plate, and the correction preserves
    // total mass, total centre of gravity and total yaw inertia by construction — so there is
    // nothing in them for it to move except where the car reaches its limit. What a sprung/unsprung
    // split changes is the *vertical* path: wheel hop, and load that transfers straight through the
    // tyre instead of through a spring. Those need vertical excitation, which means the kerb and the
    // jump.

    std::printf("\n=== criterion 10, kerb crossing (front left, 3600 ticks) ===\n");
    {
        // The criterion's own fixture, copied rather than re-invented: same ground, same speed, same
        // gentle steer onto the kerb. The kerb sits on +x, which is the car's *left*, so the wheel
        // that mounts it is the front left and reaching it is a negative demand.
        auto descriptor = ProvingGroundDescriptor{};
        descriptor.length = 200.0;
        descriptor.width = 30.0;
        descriptor.cellSize = 0.10;
        descriptor.kerbInnerEdge = 0.60;
        descriptor.features = {raceengine::Feature{.kind = raceengine::FeatureKind::Kerb, .from = 40.0, .to = 160.0}};

        const auto kerbed = PhysicsWorld::create(generateProvingGround(descriptor).value());
        REQUIRE(kerbed.has_value());

        auto state = VehicleState{};
        settle(setup, state, kerbed.value(), 8.0, 20.0);

        auto centreJumps = std::vector<double>{};
        auto loadJumps = std::vector<double>{};
        auto previousCentre = 0.0;
        auto previousLoad = 0.0;

        auto input = VehicleInput{};
        input.steering = 0.01 * outboardSign(CornerSide::Right);

        for (auto step = 0; step < 3600; step++)
        {
            const auto stepped = stepVehicle(setup, state, input, noDriveTorque, kerbed.value(), tick);
            REQUIRE(stepped.has_value());

            const auto& corner = stepped->corners[0];
            if (!corner.patch.inContact)
            {
                continue;
            }

            const auto offset = corner.patch.centre.x - corner.suspension.contactPatch.x;
            if (step > 60)
            {
                centreJumps.push_back(std::abs(offset - previousCentre));
                loadJumps.push_back(std::abs(corner.forces.tireVertical - previousLoad));
            }

            previousCentre = offset;
            previousLoad = corner.forces.tireVertical;
        }

        REQUIRE(centreJumps.size() > 1000);

        auto sortedCentre = centreJumps;
        auto sortedLoad = loadJumps;
        std::sort(sortedCentre.begin(), sortedCentre.end());
        std::sort(sortedLoad.begin(), sortedLoad.end());

        auto overEight = 0;
        for (const auto value : sortedCentre)
        {
            overEight += value > 0.008 ? 1 : 0;
        }

        std::printf("  contacting ticks %zu\n", centreJumps.size());
        std::printf("  patch centre jump: worst %.4f m, 99.9%% tail %.5f m, ticks over 8 mm %d\n", sortedCentre.back(),
                    sortedCentre[sortedCentre.size() * 999 / 1000], overEight);
        std::printf("  tyre load jump:    worst %.1f N, 99.9%% tail %.1f N\n", sortedLoad.back(),
                    sortedLoad[sortedLoad.size() * 999 / 1000]);
    }

    std::printf("\n=== criterion 11, off a ramp and back down ===\n");
    {
        auto descriptor = ProvingGroundDescriptor{};
        descriptor.length = 300.0;
        descriptor.width = 40.0;
        descriptor.cellSize = 0.5;
        descriptor.rampHeight = 0.85;
        descriptor.features = {raceengine::Feature{.kind = raceengine::FeatureKind::Ramp, .from = 60.0, .to = 75.0}};

        const auto ramped = PhysicsWorld::create(generateProvingGround(descriptor).value());
        REQUIRE(ramped.has_value());

        auto state = VehicleState{};
        settle(setup, state, ramped.value(), 22.0, 20.0);

        auto airborneTicks = 0;
        auto flightTicks = 0;
        auto landingSpeed = 0.0;
        auto peakAfterLanding = 0.0;
        auto hasLanded = false;
        auto previousVertical = 0.0;

        for (auto step = 0; step < 2400; step++)
        {
            const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, ramped.value(), tick);
            REQUIRE(stepped.has_value());

            auto anyContact = false;
            for (const auto& corner : stepped->corners)
            {
                anyContact = anyContact || corner.patch.inContact;
            }

            if (!anyContact)
            {
                airborneTicks++;
                flightTicks++;
            }
            else
            {
                if (!hasLanded && flightTicks > 60)
                {
                    hasLanded = true;
                    landingSpeed = previousVertical;
                }

                flightTicks = 0;
            }

            if (hasLanded)
            {
                peakAfterLanding = std::max(peakAfterLanding, state.chassis.linearVelocity.y);
            }

            previousVertical = state.chassis.linearVelocity.y;
        }

        std::printf("  airborne ticks %d, landing speed %.4f m/s\n", airborneTicks, landingSpeed);
        std::printf("  rebound %.4f m/s = %.3f of arrival   (criterion wants < 1.0)\n", peakAfterLanding,
                    peakAfterLanding / std::abs(landingSpeed));
        std::printf("  settled at t=6.67 s: height %.4f m, |vertical| %.4f m/s   (criterion wants < 0.25)\n",
                    state.chassis.position.y, std::abs(state.chassis.linearVelocity.y));
    }

    // --- peak mu by load band, straight off the tyre model ---
    //
    // The trace measures this as |Fy|/Fz binned by slip angle on a car that is also braking,
    // cambering and transferring load. This is the same quantity with all of that removed: pure
    // lateral slip, one load, the model's own answer. It is what load sensitivity *is*, and it moves
    // when the load the corner actually carries moves.

    std::printf("\n=== peak mu by load, pure lateral slip, off the tyre model ===\n");
    std::printf("  %10s %10s %12s\n", "Fz [N]", "peak mu", "at slip [deg]");

    for (const auto load : {2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0})
    {
        auto best = 0.0;
        auto bestSlip = 0.0;

        for (auto slipDegrees = 0.1; slipDegrees < 20.0; slipDegrees += 0.02)
        {
            const auto slip = TyreSlip{.slipRatio = 0.0, .slipAngle = slipDegrees / degrees};
            const auto forces = evaluateTyre(setup.corners[0].tyre, load, slip, 1.0);

            const auto mu = std::abs(forces.lateral) / load;
            if (mu > best)
            {
                best = mu;
                bestSlip = slipDegrees;
            }
        }

        std::printf("  %10.0f %10.4f %12.2f\n", load, best, bestSlip);
    }

    // And the two bands the brief asks for, as the load-weighted mean over the band rather than at
    // one point in it — which is what binning a trace by load actually produces.
    for (const auto band : {std::array{2000.0, 3000.0}, std::array{3000.0, 5000.0}})
    {
        auto sum = 0.0;
        auto samples = 0;

        for (auto load = band[0]; load <= band[1]; load += 25.0)
        {
            auto best = 0.0;
            for (auto slipDegrees = 0.1; slipDegrees < 20.0; slipDegrees += 0.02)
            {
                const auto slip = TyreSlip{.slipRatio = 0.0, .slipAngle = slipDegrees / degrees};
                best = std::max(best, std::abs(evaluateTyre(setup.corners[0].tyre, load, slip, 1.0).lateral) / load);
            }

            sum += best;
            samples++;
        }

        std::printf("  band %.0f-%.0f N: mean peak mu %.4f\n", band[0], band[1], sum / static_cast<double>(samples));
    }
}
