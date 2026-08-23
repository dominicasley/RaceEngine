#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::computeMassProperties;
using raceengine::Corner;
using raceengine::cornerCount;
using raceengine::CornerSide;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::frontBrakeShare;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::golfMk7FrontBrake;
using raceengine::golfMk7RearBrake;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::solveCornerWithJacobian;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::torquePerPressure;
using raceengine::validateCornerSetup;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

// The M1 acceptance criteria re-run against a car built entirely from published data, rather than
// against the placeholder they were calibrated on. What is being asked is not whether the numbers
// come out the same — they do not, and should not, since this is a 1348 kg front-wheel-drive hatch
// against a generic sedan — but whether every criterion still *holds*. Balance may move; correctness
// may not.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto gravity = 9.80665;
constexpr auto degrees = 57.29577951308232;

// The design ride height: the data puts the centre of gravity here and the contact patches on zero.
constexpr auto designHeight = 0.572;

// **Where the car balances, and it is the published figure rather than the mod's** (2026-08-22). The
// mod's suspensions.ini says `CG_LOCATION=0.53`; a DSG Mk7 GTI measures 61.4/38.6 front to rear, and
// a transverse-engined front-drive hatchback does not sit at 53%. Stated once here because two
// separate cases below check it — one that the ledger puts the centre of gravity where the fraction
// says, and one that the settled car's corner loads read it back.
constexpr auto frontWeightFraction = 0.614;
constexpr auto tyreRadius = 0.3186;

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

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ = 20.0)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
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
    // **Started with room in every direction.** The ground runs z from 0 to its length rather than
    // being centred on the origin, and a ten-second hold at 20 m/s is 200 m of arc on a circle of
    // about 54 m — so a car started at z = 20 curves straight off the negative-z end of the plate
    // partway round. It then loses support, and what the convergence check sees is that rather than
    // anything about the car. The callers hand this a plate long enough for the circle; this puts the
    // car in the middle of it.
    auto state = VehicleState{};
    settle(setup, state, world, speed, 400.0);

    auto input = VehicleInput{};
    input.steering = steering;

    // **The car is driven, and without that this is not a skidpad.** Stepping with `noDriveTorque`
    // the car coasts, and hard cornering scrubs it down — measured at 20 m/s falling to 5.9 by full
    // lock — so what came out was a spiral decaying to whatever equilibrium cornering drag left, at a
    // radius nobody chose. It also *inverted* the grip response: a lower-grip tyre scrubs less, keeps
    // more speed and therefore recorded more lateral acceleration, so cutting grip 15% appeared to
    // make the car corner harder. A real skidpad is driven, and the throttle it takes is part of the
    // test rather than a contamination of it.
    auto drive = std::array<double, cornerCount>{};
    auto integral = 0.0;

    auto result = SteadyState{};
    auto samples = 0;

    // Held apart so the preconditions below can be checked over the sample window rather than
    // averaged away: a mean speed on target says nothing about whether it was ever steady.
    auto slowest = std::numeric_limits<double>::max();
    auto fastest = 0.0;
    auto worstKinematic = 0.0;
    auto firstHalf = 0.0;
    auto secondHalf = 0.0;

    // Ten seconds, not four. At its limit this car needs about six to settle into a steady turn —
    // the convergence check below caught it still moving after four — and it circles inside a 40 m
    // radius, so a longer hold costs time and no more ground.
    for (auto step = 0; step < 3600; step++)
    {
        // Proportional on road speed, across whichever axle this car drives. It consumes grip exactly
        // as a real car's does holding a skidpad, which is honest rather than intrusive.
        const auto error = speed - glm::length(state.chassis.linearVelocity);
        // Proportional alone leaves a standing error — it held 18.4 of an asked-for 20, because a
        // constant drag needs a constant force and a P term only makes one from a constant error. The
        // integral removes it, so the fixture holds the speed it names rather than one near it.
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;
        drive = setup.corners.front().hardpoints.wheelRadius > 0.0
                    ? std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0}
                    : std::array<double, cornerCount>{};

        const auto stepped = stepVehicle(setup, state, input, drive, world, tick);
        REQUIRE(stepped.has_value());

        if (step >= 3240)
        {
            // Both stated **toward the car's own right** rather than toward +x, which is the car's
            // *left* (`outboardSign`). A positive demand is a right turn, so a positive answer here
            // is the car doing what it was asked — and positive yaw about +y swings the nose the
            // other way, which is the same relation ISO 8855 states.
            //
            // The acceleration needs no rotating: `TelemetryFrame::acceleration` is in the car's own
            // frame. It was in the *world's* until 2026-08-21, and this fixture did the rotation
            // itself — correctly, which is why these criteria were measuring the right quantity all
            // along even while the CSV column of the same name was not.
            constexpr auto toTheRight = outboardSign(CornerSide::Right);

            // **And it is still on the ground.** This is the precondition the plate-edge fault
            // violated, and it is the cheapest of the lot: a car that has driven off the end of the
            // world reports a beautifully converged nothing.
            //
            // Three wheels and not four, because **lifting the inside rear is real** — it is what a
            // stiff rear anti-roll bar does, and one of the criteria using this fixture exists to
            // stiffen that bar. Requiring all four asserted the absence of the thing being measured.
            // Leaving the world takes the front pair together and then the rest, so three still
            // catches it.
            auto supported = 0;
            for (const auto& wheel : stepped->telemetry.wheels)
            {
                supported += wheel.inContact ? 1 : 0;
            }
            REQUIRE(supported >= 3);

            const auto lateral = stepped->telemetry.acceleration.x * toTheRight;
            const auto yaw = stepped->telemetry.yawRate * toTheRight;
            const auto carried = glm::length(state.chassis.linearVelocity);

            result.lateralAcceleration += lateral;
            result.yawRate += yaw;
            result.speed += carried;
            samples++;

            slowest = std::min(slowest, carried);
            fastest = std::max(fastest, carried);
            // In a steady turn `a_y = v * yawRate`. Any disagreement means the car is not tracking a
            // circle — it is sliding, spinning or still slowing — and every number below then
            // describes a transient rather than a limit.
            worstKinematic = std::max(worstKinematic, std::abs(lateral - carried * yaw));
            (samples <= 180 ? firstHalf : secondHalf) += lateral;
        }
    }

    result.lateralAcceleration /= static_cast<double>(samples) * gravity;
    result.yawRate /= static_cast<double>(samples);
    result.speed /= static_cast<double>(samples);

    // **The fixture asserts that it did what its name says.** All four of the faults this suite has
    // turned up in a month were fixtures quietly measuring something else, and every one of them
    // would have been caught here rather than several conclusions later. These are cheap and they run
    // on every criterion that holds a steady state.
    CAPTURE(steering, speed, result.speed, slowest, fastest, worstKinematic);

    // **The speed check is the load-bearing one**, and it is the one the coasting fixture violated:
    // it was down to 5.9 m/s of an asked-for 20 by full lock. Tight, because holding the speed is the
    // whole difference between a skidpad and a spiral.
    REQUIRE(slowest > 0.9 * speed);
    REQUIRE(fastest < 1.1 * speed);

    // A gross-departure check, and deliberately loose. `a_y = v * yawRate` holds only while the
    // sideslip angle is *constant*, and the residual is `v * dbeta/dt` — so a car past its grip limit
    // legitimately fails it, because it is ploughing rather than tracking a circle. **This criterion
    // sweeps deliberately past the limit** to show the curve turn over, so a tight bound here would be
    // asserting the absence of the very thing the test exists to find. What is worth catching is a car
    // that has departed outright, where the two decouple completely.
    //
    // It is also worth being explicit that this would **not** have caught the coasting fault: that
    // fixture's kinematics agreed perfectly all the way down, because a decaying spiral is still a
    // spiral. It was the *speed* it got wrong, which is why that assertion above is the tight one and
    // this is the loose one.
    REQUIRE(worstKinematic < 0.5 * std::abs(result.lateralAcceleration) * gravity + 1.0);

    // And it had converged: the two halves of the sample window agree. Scaled to the lateral
    // acceleration for the same reason the departure check is — **past its grip limit the car ploughs
    // and its lateral acceleration genuinely wanders**, and these criteria sweep deliberately past the
    // limit to show the curve turn over. Verified still to catch what it was added for: the car
    // driving off the end of the plate read 3.7 against a threshold of 1.2 here.
    REQUIRE(std::abs(firstHalf / 180.0 - secondHalf / static_cast<double>(samples - 180)) <
            0.15 * std::abs(result.lateralAcceleration) * gravity + 0.2);

    return result;
}

} // namespace

TEST_CASE("the imported car weighs what the data says and is distributed as it says", "[physics][golf]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto properties = computeMassProperties(setup->sprung);
    REQUIRE(properties.has_value());

    // car.ini TOTALMASS, less the four corner masses. Those are **not** suspensions.ini HUB_MASS any
    // more: the file's 80 front / 85 rear is a quarter of the car as unsprung mass, and what is here
    // instead is the component build-up written out in `PublishedCars.cppm`. The total is what must
    // not move, and it does not — the sprung side is solved from it rather than stated beside it,
    // which is exactly what let the mod's figure hide for a milestone.
    REQUIRE(setup->unsprungMass() == Catch::Approx(184.0));
    REQUIRE(properties->mass == Catch::Approx(1164.0));
    REQUIRE(properties->mass + setup->unsprungMass() == Catch::Approx(1348.0));

    SECTION("and the assembled car carries the inertia the file states, not the shell")
    {
        // car.ini INERTIA is the box the *whole* car has the inertia of. The shell is that less every
        // parallel-axis term the assembly puts back, so the check is that assembling it recovers the
        // box — which is the arithmetic done backwards, and the one thing that would show a sign
        // error in it.
        auto ledger = std::vector<raceengine::MassComponent>{setup->sprung.front()};
        for (const auto& corner : setup->corners)
        {
            ledger.push_back(raceengine::MassComponent{
                .mass = corner.unsprungMass, .centre = corner.hardpoints.wheelCentre, .inertia = glm::dmat3(0.0)});
        }

        const auto assembled = computeMassProperties(ledger);
        REQUIRE(assembled.has_value());

        const auto twelfth = 1348.0 / 12.0;
        REQUIRE(assembled->inertia[0][0] == Catch::Approx(twelfth * (1.452 * 1.452 + 4.27 * 4.27)).epsilon(1e-9));
        REQUIRE(assembled->inertia[1][1] == Catch::Approx(twelfth * (1.54 * 1.54 + 4.27 * 4.27)).epsilon(1e-9));
        REQUIRE(assembled->inertia[2][2] == Catch::Approx(twelfth * (1.54 * 1.54 + 1.452 * 1.452)).epsilon(1e-9));

        // And the centre of gravity is where the two coordinates that could be sourced put it.
        REQUIRE(assembled->centreOfMass.y == Catch::Approx(designHeight).margin(1e-9));
        REQUIRE(assembled->centreOfMass.z == Catch::Approx(2.638 * (frontWeightFraction - 0.5)).margin(1e-9));
    }

    SECTION("the tyre is the one the file describes")
    {
        for (const auto& corner : setup->corners)
        {
            REQUIRE(corner.hardpoints.wheelRadius == Catch::Approx(tyreRadius));
            REQUIRE(corner.wheelInertia == Catch::Approx(1.45));
            REQUIRE(corner.tyre.nominalLoad == Catch::Approx(2939.0));
            // **No longer the Semislicks' DY_REF/DX_REF**, which were 1.28 and 1.30 — the fourth
            // figure taken away from the mod, after the wheel radius, the unsprung mass and
            // `CG_LOCATION`. Both of the file's compounds are track-tyre numbers, including the one it
            // calls "Street" at 1.23/1.26, where a 225/40 R18 performance road tyre peaks nearer 1.0
            // to 1.15.
            //
            // Scaled 0.87, derived rather than chosen: criteria 6 and 7 fixed first from physical
            // reasoning (both grip-independent — under 1.5% across a 20% grip cut), criterion 5's
            // threshold then set from the real car's 0.90-0.95 g skidpad, and grip moved to land it
            // mid-band at 0.9232 g. 0-100 then checked independently at 6.556 s against a published
            // 6.4-6.7 for a DSG without launch control. The full account is on the assignment in
            // `PublishedCars.cppm`.
            REQUIRE(corner.tyre.lateralPeak == Catch::Approx(1.114));
            REQUIRE(corner.tyre.longitudinalPeak == Catch::Approx(1.131));
        }

        REQUIRE(setup->sampling.width == Catch::Approx(0.235));
    }
}

TEST_CASE("every corner of the imported car validates", "[physics][golf]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup->corners[index];

        // The factory validates before it hands the car back, so this is a second statement of it —
        // and it is worth making, because the thing that would break it is somebody moving a hardpoint
        // and only running the geometry tests.
        REQUIRE(validateCornerSetup(corner).has_value());

        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());

        // AC states its spring rates *at the wheel*, and the model wants them along the damper. The
        // conversion is the motion ratio squared, so multiplying back must recover the file's own
        // number exactly — this is the check that the conversion is a conversion and not a guess.
        const auto wheelRate = corner.springRate * design->motionRatio * design->motionRatio;
        REQUIRE(wheelRate == Catch::Approx(index < 2 ? 35000.0 : 57000.0).epsilon(1e-9));

        // suspensions.ini [ARB], straight from the file.
        REQUIRE(corner.antiRollRate == Catch::Approx(index < 2 ? 34000.0 : 15000.0));

        // **Neither of the mod's two brake numbers survives** (2026-08-23, `docs/brake-model-brief.md`).
        // `MAX_TORQUE=4200` was verified at source, found identical across all four cars in the pack,
        // and could not lock this car's front wheels at any pedal position; it was corrected to a
        // chosen 5600 and is now not stated at all. `FRONT_SHARE=0.75` is gone with it, because a
        // brake split is not a setting: it is what two sets of calipers make of one line pressure,
        // and then what the rear circuit's proportioning valve makes of that.
        //
        // Asserted against the parts rather than against a literal, so this fails when a caliper or a
        // pad changes and not when the arithmetic is rounded differently.
        const auto hardware = index < 2 ? golfMk7FrontBrake() : golfMk7RearBrake();
        const auto pressure = brakeCircuitPressures(setup.value(), 1.0)[index];

        REQUIRE(corner.brakeTorque == Catch::Approx(torquePerPressure(hardware) * pressure));
    }

    SECTION("and the brake bias is not a number at all, because a proportioning valve moves it")
    {
        // The calipers alone put 0.6859 on the front axle. That is the whole bias only below the
        // valve's knee; above it the rear is limited and the share climbs. What it must not do is
        // stay put — a fixed share is the thing this replaced.
        const auto low = brakeCircuitPressures(setup.value(), 0.10);
        const auto high = brakeCircuitPressures(setup.value(), 1.00);

        const auto shareAt = [&](const std::array<double, cornerCount>& pressures)
        {
            const auto front = pressures[0] * torquePerPressure(golfMk7FrontBrake());
            const auto rear = pressures[2] * torquePerPressure(golfMk7RearBrake());

            return front / (front + rear);
        };

        // Below the knee the valve is transparent and the share is the calipers' own.
        REQUIRE(shareAt(low) == Catch::Approx(frontBrakeShare(golfMk7FrontBrake(), golfMk7RearBrake())));

        // And at a fully applied pedal it has moved a long way toward the ideal at the limit, which
        // this car's own axle loads put at about 0.81.
        CAPTURE(shareAt(low), shareAt(high));
        REQUIRE(shareAt(high) > shareAt(low) + 0.10);
        REQUIRE(shareAt(high) < 0.90);
    }

    SECTION("and the steering reaches the lock the car states, the way round the car states")
    {
        // 378 degrees of wheel over a ratio of 14.1 is 26.81 degrees at the road wheel, and the rack
        // travel that produces it is solved off the linkage rather than authored. The ratio's sign is
        // what puts a right-hand turn on positive steering: this car's steering arm sits behind the
        // kingpin, so a rack moving right turns it left.
        // Positive: a positive demand steers the car toward its own right, which is what every input
        // path produces for right and what `outboardSign` finally pins the meaning of.
        REQUIRE(setup->rackTravelPerInput > 0.0);

        const auto solved = solveCornerWithJacobian(setup->corners[1].hardpoints, 0.0, -setup->rackTravelPerInput);
        REQUIRE(solved.has_value());
        REQUIRE(std::abs(solved->toe) * degrees == Catch::Approx(378.0 / 14.1).margin(1e-6));
    }
}

TEST_CASE("the imported car settles on its springs", "[physics][golf][settle]")
{
    // Criterion 4, on real data. Dropped half a metre it must settle to a stable ride height with no
    // residual motion, and its corner loads must match a statics prediction made from the mass ledger
    // rather than from the same code path that produced them.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate(120.0)).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight + 0.5, 20.0);

    auto heights = std::vector<double>{};
    auto last = raceengine::VehicleStep{};

    for (auto step = 0; step < 2160; step++)
    {
        auto stepped = stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());

        last = stepped.value();
        heights.push_back(state.chassis.position.y);
    }

    SECTION("it comes to rest")
    {
        REQUIRE(std::abs(state.chassis.linearVelocity.y) < 1e-3);

        const auto tail = std::vector<double>(heights.end() - 180, heights.end());
        REQUIRE(*std::max_element(tail.begin(), tail.end()) - *std::min_element(tail.begin(), tail.end()) < 1e-4);
    }

    SECTION("a little below the design height, having sunk into its tyres")
    {
        // The springs are at their design length — `springFreeLengthForLoad` solved them to be — so
        // everything the car sinks is tyre. It is more than load over rate because the patch's
        // penetration is a quadrature over a curved tread rather than the depth at its centre.
        REQUIRE(state.chassis.position.y < designHeight);
        REQUIRE(state.chassis.position.y > designHeight - 0.025);

        for (const auto& corner : last.corners)
        {
            REQUIRE(corner.patch.inContact);
            REQUIRE(std::abs(corner.suspension.wheelTravel) < 1e-4);
            REQUIRE(corner.forces.bumpStop == Catch::Approx(0.0).margin(1.0));
            REQUIRE(corner.forces.droopStop == Catch::Approx(0.0).margin(1.0));
        }
    }

    SECTION("and the corner loads are the ones statics predicts")
    {
        const auto properties = computeMassProperties(setup->sprung);
        REQUIRE(properties.has_value());

        // Sprung load by moments about the far axle, plus the unsprung weight the tyre carries
        // directly. Computed here from the ledger, so it is a prediction rather than an echo.
        const auto sprungFront = properties->mass * gravity * (properties->centreOfMass.z + 1.319) / 2.638;
        const auto sprungRear = properties->mass * gravity - sprungFront;

        const auto load = [&last](const Corner corner)
        {
            return last.corners[static_cast<std::size_t>(corner)].forces.tireVertical;
        };

        // The unsprung weight comes off the car rather than being written here. It used to be the
        // literals 80 and 85, which is the mod's `HUB_MASS` restated in a second place — so
        // correcting the car's data broke this prediction *because the prediction was a copy*, which
        // is the same "stated twice and now they disagree" fault the ledger in `PublishedCars.cppm`
        // exists to avoid. Read from the setup it is predicting, it is a prediction again.
        const auto unsprungAt = [&setup](const Corner corner)
        {
            return setup->corners[static_cast<std::size_t>(corner)].unsprungMass * gravity;
        };

        REQUIRE(load(Corner::FrontLeft) ==
                Catch::Approx(sprungFront / 2.0 + unsprungAt(Corner::FrontLeft)).epsilon(1e-3));
        REQUIRE(load(Corner::RearLeft) == Catch::Approx(sprungRear / 2.0 + unsprungAt(Corner::RearLeft)).epsilon(1e-3));

        const auto front = load(Corner::FrontLeft) + load(Corner::FrontRight);
        const auto rear = load(Corner::RearLeft) + load(Corner::RearRight);

        REQUIRE(front + rear == Catch::Approx(1348.0 * gravity).epsilon(0.01));
        // The published weight distribution, read back off the car it was built into — the whole
        // chain from the mass ledger through the spring solve to four settled tyre loads.
        REQUIRE(front / (front + rear) == Catch::Approx(frontWeightFraction).epsilon(1e-3));

        REQUIRE(load(Corner::FrontLeft) == Catch::Approx(load(Corner::FrontRight)).epsilon(1e-4));
        REQUIRE(load(Corner::RearLeft) == Catch::Approx(load(Corner::RearRight)).epsilon(1e-4));
    }

    SECTION("it is sitting flat")
    {
        REQUIRE(last.telemetry.roll == Catch::Approx(0.0).margin(1e-6));
        REQUIRE(std::abs(last.telemetry.pitch) < 0.01);
    }
}

TEST_CASE("the imported car goes straight and keeps its speed", "[physics][golf]")
{
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate(1200.0)).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), 25.0);

    const auto startHeight = state.chassis.position.y;

    for (auto step = 0; step < 1800; step++)
    {
        REQUIRE(stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    REQUIRE(std::abs(state.chassis.position.x) < 0.01);
    REQUIRE(std::abs(state.chassis.position.y - startHeight) < 1e-3);
    REQUIRE(state.chassis.linearVelocity.z > 23.0);
    REQUIRE(state.chassis.linearVelocity.z < 24.5);
}

TEST_CASE("the imported car's skidpad has an understeer gradient and a limit", "[physics][golf][skidpad]")
{
    // Criterion 5. The steering values are this car's rather than the placeholder's: its rack travels
    // 70 mm for full lock against the placeholder's 55, so the same input is a much larger angle.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate(1200.0)).value());
    REQUIRE(world.has_value());

    // The sweep has to run **beyond** the peak for the last sample to show grip being given back, and
    // where the peak sits is a property of the car rather than of this list.
    //
    // It has been extended twice for that reason and the second time is worth recording, because the
    // failure looked like a limit that had disappeared. The corrected aligning moment moved the peak
    // far enough up that 0.45 was the peak itself, so 0.60 was added; then the **weight distribution
    // was corrected from the mod's 53% front to the published 61.4%** (2026-08-22) and 0.60 became
    // the peak in its turn — the last sample *was* the maximum, so "it does not gain grip
    // indefinitely" failed with the two sides exactly equal. A nose-heavy front-drive car understeers
    // more and needs more lock to reach its limit, which is the correct behaviour and not a lost
    // limit; what was wrong was a fixture whose range was cut to the old car.
    const auto steerings = std::vector<double>{0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.22, 0.30, 0.45, 0.60, 0.80, 1.00};
    auto measured = std::vector<SteadyState>{};

    for (const auto steering : steerings)
    {
        measured.push_back(hold(setup.value(), world.value(), steering, 20.0));
    }

    SECTION("more steering gives more grip, until it does not")
    {
        for (auto index = std::size_t{1}; index < 7; index++)
        {
            REQUIRE(measured[index].lateralAcceleration > measured[index - 1].lateralAcceleration);
        }
    }

    SECTION("the gradient falls away at every step, which is understeer")
    {
        // Per unit of steering rather than per sample, because the samples are not evenly spaced —
        // comparing raw differences across unequal steps measures the spacing as much as the car.
        auto previous = 1e9;
        for (auto index = std::size_t{1}; index < 7; index++)
        {
            const auto gradient = (measured[index].lateralAcceleration - measured[index - 1].lateralAcceleration) /
                                  (steerings[index] - steerings[index - 1]);

            REQUIRE(gradient < previous);
            previous = gradient;
        }
    }

    SECTION("and it does not gain grip indefinitely")
    {
        auto peak = 0.0;
        for (const auto& sample : measured)
        {
            peak = std::max(peak, sample.lateralAcceleration);
        }

        // 0.90 measured, and the semislicks' 1.28 barely moved it from the road compound's 0.93 —
        // deliberately understood, not a bug: this sweep coasts, and the grippier tyre pulls the car
        // into a tighter spiral that scrubs speed twice as hard (7.2 m/s left at 0.45 steering
        // against 13.0), so past 0.3 the sample is speed-limited rather than grip-limited. Where
        // speeds are comparable the semislick corners harder at every point (0.904 against 0.884 at
        // 0.3 steering). A powered skidpad would show the compound; this one shows the limit exists.
        // **The bound is the real car's, not the model's** (2026-08-22). A Mk7 GTI on OEM tyres
        // skidpads at 0.90 to 0.95 g, and that range is what this asserts — so this criterion is a
        // comparison against the car rather than a record of what the model last did.
        //
        // It is also the one criterion here that grip is free to move, which is why it had to be set
        // *after* 6 and 7 were fixed from reasoning: those two are grip-independent (measured, under
        // 1.5% across a 20% grip cut), so they could be pinned first and the tyre's peaks then chosen
        // to land this one. Setting a grip figure from a threshold that a grip figure had set would
        // have been circular.
        //
        // Reads 0.9232 g. The independent check is 0-100, which lands 6.556 s against a published
        // 6.4-6.7 for a DSG without launch control — a different measurement against a different
        // external reference, sharing only the tyre peaks.
        REQUIRE(peak > 0.90);
        REQUIRE(peak < 0.95);
        REQUIRE(measured.back().lateralAcceleration < peak);
    }

    SECTION("yaw rate rises with steering through the gripping range")
    {
        for (auto index = std::size_t{1}; index < 7; index++)
        {
            REQUIRE(measured[index].yawRate > measured[index - 1].yawRate);
        }
    }

    SECTION("and the fixture held its speed rather than the car scrubbing it off")
    {
        // **This section used to assert the opposite, and it was asserting a bug.** It required speed
        // at full lock to be *below* speed at the smallest steering — which was true only because
        // `hold()` coasted, so the car scrubbed itself from an asked-for 20 m/s down to 5.9 by full
        // lock. That decay is what made the whole criterion measure a spiral rather than a skidpad,
        // and this line was pinning it in place as though it were the physics.
        //
        // Driven, the fixture holds speed at every steering angle, which is what a skidpad is. Sliding
        // still costs the car energy; the throttle is what puts it back, exactly as on a real one.
        for (const auto& point : measured)
        {
            REQUIRE(point.speed > 19.0);
            REQUIRE(point.speed < 21.0);
        }
    }
}

TEST_CASE("a stiffer rear bar shifts the imported car toward oversteer", "[physics][golf][balance]")
{
    // Criterion 7, and the brief's own test of whether load sensitivity is real. This car's exponent
    // is the file's — one less LS_EXPY, or 0.1926 — rather than the placeholder's 0.15.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate(1200.0)).value());
    REQUIRE(world.has_value());

    const auto withRearBar = [&world](const double rate)
    {
        auto setup = golfGtiMk7();
        REQUIRE(setup.has_value());

        setup->corners[2].antiRollRate = rate;
        setup->corners[3].antiRollRate = rate;

        return hold(setup.value(), world.value(), 0.11, 20.0);
    };

    const auto soft = withRearBar(0.0);
    const auto medium = withRearBar(15000.0);
    const auto stiff = withRearBar(40000.0);

    // **Thresholds fixed from reasoning rather than from what the car happens to do**, 2026-08-22,
    // and this criterion is one of the two that can be: it is a *ratio* between two runs differing
    // only in a rear bar rate, so whatever grip they share divides out. Measured across a 20% grip
    // cut the ratio moves 1.1106 to 1.1010 — nine parts in a thousand — which is what makes it usable
    // as a check independent of the grip figure criterion 5 is used to set.
    //
    // The ordering is the physics: stiffening the rear bar moves lateral load transfer rearward, the
    // outside rear takes more load, load sensitivity costs it friction faster than the front gains,
    // and the balance shifts toward oversteer. Monotonic in bar rate because load transfer is.
    REQUIRE(medium.yawRate > soft.yawRate);
    REQUIRE(stiff.yawRate > medium.yawRate);

    // The magnitude floor asks that the effect be *unambiguously present* rather than that it be any
    // particular size — this model has never been calibrated against a measured bar sweep, so a tight
    // band would be asserting a number nobody has. 5% clears the fixture's own noise by a wide margin:
    // the runs are deterministic and repeat exactly, and the convergence precondition in `hold()`
    // bounds what survives averaging at well under a percent.
    REQUIRE(stiff.yawRate > soft.yawRate * 1.05);
}

TEST_CASE("the imported car's step steer settles rather than ringing", "[physics][golf][stepsteer]")
{
    // Criterion 6.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), 25.0);

    auto input = VehicleInput{};
    input.steering = 0.06;

    auto history = std::vector<double>{};
    for (auto step = 0; step < 1080; step++)
    {
        const auto stepped = stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());
        // Toward the car's own right, so a positive demand reads positive. See `outboardSign`.
        history.push_back(stepped->telemetry.yawRate * outboardSign(CornerSide::Right));
    }

    auto settled = 0.0;
    for (auto index = history.size() - 180; index < history.size(); index++)
    {
        settled += history[index];
    }
    settled /= 180.0;

    // **Thresholds fixed from reasoning rather than from what the car happens to do**, 2026-08-22.
    // Criterion 6 is response *shape*, and shape is set by yaw inertia, geometry and roll stiffness
    // rather than by friction: measured across a 20% grip cut, settled yaw moves 0.1789 to 0.1773 and
    // rise time 0.222 s to 0.219. That independence is what lets these bounds be fixed while grip is
    // still being decided.
    //
    // At 25 m/s and 0.06 of demand the road wheels are at about 1.6 degrees. Ackermann alone would put
    // the car on a 95 m radius and 0.26 rad/s of yaw; understeer widens that, so anything from a third
    // of the Ackermann figure upward is a car that is steering. Bounded above as well, because a yaw
    // rate *exceeding* the Ackermann value at this demand would mean the car is oversteering into the
    // corner, which this one does not do.
    REQUIRE(settled > 0.08);
    REQUIRE(settled < 0.26);

    SECTION("it rises in a plausible time")
    {
        auto riseTicks = std::size_t{0};
        while (riseTicks < history.size() && history[riseTicks] < settled * 0.9)
        {
            riseTicks++;
        }

        const auto rise = static_cast<double>(riseTicks) * tick;

        // A passenger car's yaw time constant at this speed is of the order 0.1 to 0.3 s, and rise to
        // 90% of a well-damped second-order response is a little over two of them. **The old lower
        // bound of 0.05 s was not a physical constraint** — no car of this mass and inertia responds
        // in fifty milliseconds, so it excluded nothing. 0.12 s is the fastest this car could
        // plausibly be; 0.60 s is a car that feels slow to the driver and is the upper limit worth
        // shipping.
        REQUIRE(rise > 0.12);
        REQUIRE(rise < 0.60);
    }

    SECTION("it overshoots by a bounded amount and does not ring")
    {
        // A road car tuned for stability overshoots its steady yaw by something between a few percent
        // and a fifth; past about a third it reads as nervous and is the "ringing" this excludes. The
        // bound is on the character the car is meant to have rather than on the 1.077 it happens to
        // make.
        REQUIRE(*std::max_element(history.begin(), history.end()) < settled * 1.35);

        const auto tail = std::vector<double>(history.begin() + 540, history.end());
        REQUIRE(*std::max_element(tail.begin(), tail.end()) - *std::min_element(tail.begin(), tail.end()) <
                settled * 0.05);
    }
}

TEST_CASE("the imported car is stable at walking pace and at a standstill", "[physics][golf][lowspeed]")
{
    // Criterion 8.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    SECTION("creeping at walking pace produces no spikes")
    {
        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), 1.2);

        auto input = VehicleInput{};
        input.steering = 0.3;

        auto worstJump = 0.0;
        auto previous = 0.0;

        for (auto step = 0; step < 3600; step++)
        {
            const auto stepped = stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());

            const auto force = stepped->corners[0].contact.tyre.lateral;
            REQUIRE(std::isfinite(force));
            worstJump = std::max(worstJump, std::abs(force - previous));
            previous = force;
        }

        REQUIRE(worstJump < 300.0);
        REQUIRE(state.chassis.position.y > 0.5);
    }

    SECTION("brought to a stop, it stays stopped")
    {
        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), 8.0);

        auto input = VehicleInput{};
        input.brake = 1.0;

        for (auto step = 0; step < 3600; step++)
        {
            REQUIRE(stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick).has_value());
        }

        const auto resting = state.chassis.position;

        for (auto step = 0; step < 1800; step++)
        {
            REQUIRE(stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick).has_value());
        }

        REQUIRE(glm::length(state.chassis.linearVelocity) < 0.02);
        REQUIRE(glm::distance(state.chassis.position, resting) < 0.02);
    }
}

TEST_CASE("the imported car holds a fifteen degree slope on its brakes", "[physics][golf][lowspeed]")
{
    // Criterion 2's second half.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto descriptor = plate(200.0);
    descriptor.slopeAngle = 0.26180;
    descriptor.features = {Feature{.kind = FeatureKind::Slope, .from = 0.0, .to = 200.0}};

    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight + 60.0 * std::tan(descriptor.slopeAngle), 60.0);
    state.chassis.orientation = glm::angleAxis(-descriptor.slopeAngle, glm::dvec3(1.0, 0.0, 0.0));

    auto input = VehicleInput{};
    input.brake = 1.0;

    for (auto step = 0; step < 3600; step++)
    {
        REQUIRE(stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick).has_value());
    }

    const auto resting = state.chassis.position;

    for (auto step = 0; step < 3600; step++)
    {
        REQUIRE(stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick).has_value());
    }

    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.05);
    REQUIRE(glm::distance(state.chassis.position, resting) < 0.05);
}

TEST_CASE("the imported car's coastdown recovers the coefficients it was given", "[physics][golf][coastdown]")
{
    // Criterion 9, and here the coefficients being recovered are the file's own: a rolling resistance
    // read off tyres.ini and a drag area that is aero.ini's body wing, its chord by its span times the
    // coefficient its lookup table gives at zero angle of attack.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto descriptor = plate(3000.0);
    descriptor.cellSize = 4.0;
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), 27.78);

    auto speeds = std::vector<double>{};
    auto decelerations = std::vector<double>{};
    auto previous = 27.78;

    for (auto step = 1; step <= 10800; step++)
    {
        REQUIRE(stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());

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
    for (const auto& corner : setup->corners)
    {
        const auto radius = corner.hardpoints.wheelRadius;
        rotational += corner.wheelInertia / (radius * radius);
    }

    // Three surfaces here rather than one, so the drag being fitted is their sum — the two lifting
    // ones contribute none of it, which is what a CD_GAIN of zero on both of them says.
    auto dragArea = 0.0;
    for (const auto& surface : setup->aero)
    {
        dragArea += surface.dragArea;
    }

    const auto effectiveMass = 1348.0 + rotational;

    REQUIRE(rolling * effectiveMass / (1348.0 * gravity) ==
            Catch::Approx(setup->corners.front().rollingResistance).epsilon(0.08));
    REQUIRE(drag * effectiveMass / (0.5 * setup->airDensity) == Catch::Approx(dragArea).epsilon(0.08));

    REQUIRE(drag > 0.0);
    REQUIRE(rolling > 0.0);
}

// **Known red, and deliberately marked so rather than loosened.** `[!shouldfail]` means Catch2 expects
// this to fail and will report it if it ever *passes* — which is exactly the notification wanted,
// because the thing that will make it pass is the tyre enveloping model landing.
//
// What fails is the patch-centre continuity bound: **25.9 mm in a single tick against a 12 mm bound**,
// past the 20 mm hard limit too. It is the W3 chamfer-edge defect — contacting samples drop out over a
// chamfer with the missed ones left in the divisor — and the tail is under 8 mm, so it is still the
// one-tick out-and-back that signature has always had, at a larger amplitude.
//
// **The bound has already been raised twice for this same signature**, once for the compound change
// and once for the unsprung-mass correction. A third raise, past a limit that reads as "never exceed
// two centimetres", would be the correction-that-hides-the-mechanism pattern this project tracks. It
// is not a tolerance question: a 26 mm single-tick jump in the patch centre is a force discontinuity
// through the steering on a real kerb.
//
// Owner: the E-series enveloping work, `docs/tyre-enveloping-and-dsg-brief.md`. Nothing before that
// lands will move it. Listed in `docs/known-red.md`.
TEST_CASE("the imported car crosses a kerb continuously", "[physics][golf][kerb]")
{
    // Criterion 10, and the assertion is deliberately not the placeholder's. Comparing the worst tick
    // against the 99th percentile assumes the wheel spends under a percent of the run on the kerb's
    // ramp; this car crosses at a shallower angle and spends longer there, so the 99th percentile sits
    // in the middle of the transition and the ratio measures the shape of the run rather than the
    // continuity of the load. What a discontinuity actually is — one tick out of line with the ticks
    // either side of it — shows in the *tail* of the distribution, so the tail is what it is compared
    // against. Measured, the worst five load jumps are 271, 265, 253, 240 and 235 N: a ramp, not a
    // step.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 30.0;
    descriptor.cellSize = 0.10;
    descriptor.kerbInnerEdge = 0.60;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 40.0, .to = 160.0}};

    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), 8.0, 20.0);

    // **The kerb is on the proving ground's +x side** (`kerbInnerEdge` is measured from the
    // centreline toward +x) and +x is the car's *left* (`outboardSign`), so the wheel that mounts it
    // is the front **left** and reaching it means steering to the left, which is a negative demand.
    //
    // Both of those were hard-coded the other way and the test still passed, because they were wrong
    // together: a car steered toward its labelled right drifted toward +x, and the wheel labelled
    // front right was the one at +x. Un-mirroring the corner sides separated them.
    auto input = VehicleInput{};
    input.steering = 0.01 * outboardSign(CornerSide::Right);

    auto loadJumps = std::vector<double>{};
    auto centreJumps = std::vector<double>{};
    auto previousLoad = 0.0;
    auto previousCentre = 0.0;
    auto sawKerb = false;
    auto blended = false;

    for (auto step = 0; step < 3600; step++)
    {
        const auto stepped = stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());

        const auto& corner = stepped->corners[static_cast<std::size_t>(Corner::FrontLeft)];

        if (corner.patch.inContact)
        {
            const auto offset = corner.patch.centre.x - corner.suspension.contactPatch.x;

            if (step > 60)
            {
                loadJumps.push_back(std::abs(corner.forces.tireVertical - previousLoad));
                centreJumps.push_back(std::abs(offset - previousCentre));
            }

            previousLoad = corner.forces.tireVertical;
            previousCentre = offset;

            if (corner.patch.centre.y > 0.01)
            {
                sawKerb = true;
            }
            if (corner.patch.gripMultiplier < 0.999 && corner.patch.gripMultiplier > 0.86)
            {
                blended = true;
            }
        }
    }

    REQUIRE(sawKerb);
    REQUIRE(blended);
    REQUIRE(loadJumps.size() > 2000);

    const auto worstAgainstTail = [](std::vector<double> values)
    {
        std::sort(values.begin(), values.end());

        return std::pair{values.back(), values[values.size() * 999 / 1000]};
    };

    const auto [worstLoad, tailLoad] = worstAgainstTail(loadJumps);
    const auto [worstCentre, tailCentre] = worstAgainstTail(centreJumps);

    // **Every bound here was loosened twice for a mechanism that turned out to be a lost raycast, and
    // all four are now tighter than they have ever been** (2026-08-22, E2).
    //
    // What this test failed on was a single ray of 31,851 coming back with no hit while the samples
    // either side of it in the same row reported 19.5 and 18.8 mm of road. One tick, out and straight
    // back: 25.9 mm of patch-centre movement and 665 N, then 22.6 mm and 713 N returning. Nothing else
    // in the crossing was ever over 2.5 mm.
    //
    // Two mechanisms were on record for it and **both were wrong**. `docs/known-red.md` said the W3
    // chamfer-edge defect — the bed of independent springs dropping samples with the divisor left
    // alone. The comments that used to sit here said the spike-rejection lottery, a sample admitted
    // for one tick because the median the ceiling is measured from moved. Measured over the whole
    // crossing: **zero of 3539 ticks reject a single sample**, and the load dip the first explanation
    // names cannot produce a step because a sample leaving contact leaves with zero weight.
    // `./EngineTests "[.kerb-discontinuity]"` is the dissection; the fault and its 35 micrometre dead
    // band are written up at the retry in `JoltBackend.cpp`.
    //
    // Measured after the repair, on the same crossing: worst load jump **275 N against a 239 N tail**
    // and worst centre jump **2.42 mm against a 2.42 mm tail** — the tail *is* the worst, so the
    // distribution has no outlier left in it at all. The floors below are set at roughly a 50% margin
    // over that, which is what makes them a gate again rather than a formality: the old 12 mm floor
    // would not have noticed a five-fold regression.
    REQUIRE(worstLoad < std::max(tailLoad * 1.5, 400.0));
    REQUIRE(worstCentre < std::max(tailCentre * 1.5, 0.004));

    REQUIRE(worstLoad < 800.0);
    REQUIRE(worstCentre < 0.006);
}

TEST_CASE("the imported car flies cleanly and lands without being thrown", "[physics][golf][airborne]")
{
    // Criterion 11. The rebound bound is the criterion itself — the landing must not return more than
    // it received — rather than the placeholder's 0.6, and the difference between the two is data. AC
    // states this tyre's vertical damping at 500 N.s/m where the placeholder assumed 1500, and 500 is
    // five per cent of critical against the carcass's own 304 kN/m and an 80 kg hub. Measured on this
    // landing, that one number moves the rebound from 0.60 of the arrival speed to 0.86; the stops
    // cannot make it up, because their damping is contributed explicitly rather than solved and
    // 40 kN.s/m is already over half the tick's stability limit.
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 300.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 0.5;
    descriptor.rampHeight = 0.85;
    descriptor.features = {Feature{.kind = FeatureKind::Ramp, .from = 60.0, .to = 75.0}};

    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), 22.0, 20.0);

    auto airborneTicks = 0;
    auto flightTicks = 0;
    auto peakHeight = 0.0;
    auto worstDroop = 0.0;
    auto landingSpeed = 0.0;
    auto peakAfterLanding = 0.0;
    auto hasLanded = false;
    auto previousVertical = 0.0;

    // **Ten seconds rather than six and two thirds** (2026-08-22). The criterion is that the car
    // settles; the run length is how long it is given to, and 2400 ticks was cut to a car whose
    // centre of gravity sat 222 mm further back. Correcting the weight distribution to the published
    // 61.4% front left it still decaying at the old cut — 0.347 m/s there, against 0.019 by 7.33 s
    // and 0.0002 by 8.0, dead still at y = 0.5537 from then on. A nose-heavy car takes longer to stop
    // pitching after a ramp landing, which is the expected direction; sampling it mid-decay and
    // calling that "does not settle" is the fixture's fault and not the car's.
    //
    // Worth knowing that this window was **already marginal before any of it**: the mod's own car
    // read 0.2487 m/s here against the 0.25 bound, half a percent of margin, and the wheel-radius
    // correction alone was enough to tip it over.
    for (auto step = 0; step < 3600; step++)
    {
        const auto stepped = stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());

        auto anyContact = false;
        for (const auto& corner : stepped->corners)
        {
            anyContact = anyContact || corner.patch.inContact;
            worstDroop = std::min(worstDroop, corner.suspension.wheelTravel);
        }

        if (!anyContact)
        {
            airborneTicks++;
            flightTicks++;
            peakHeight = std::max(peakHeight, state.chassis.position.y);
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

        REQUIRE(std::isfinite(state.chassis.position.y));
    }

    SECTION("it actually left the ground and came back")
    {
        REQUIRE(airborneTicks > 60);
        REQUIRE(hasLanded);
        REQUIRE(peakHeight > 1.0);
    }

    SECTION("the suspension runs out to its stops rather than for ever")
    {
        REQUIRE(worstDroop < -0.02);
        REQUIRE(worstDroop > -0.15);
    }

    SECTION("landing does not throw the car back up harder than it fell")
    {
        REQUIRE(landingSpeed < 0.0);
        REQUIRE(peakAfterLanding < std::abs(landingSpeed));
    }

    SECTION("and it ends up settled on the ground")
    {
        REQUIRE(state.chassis.position.y > 0.45);
        REQUIRE(state.chassis.position.y < 0.65);
        REQUIRE(std::abs(state.chassis.linearVelocity.y) < 0.25);
    }
}

TEST_CASE("the imported car ticks purely and deterministically", "[physics][golf][determinism]")
{
    // Criterion 12. Nothing about the setup is a source of state, so this must hold for a real car as
    // it does for the placeholder — and it is worth asserting on this one, because a factory that
    // solves for a rack travel and for four spring lengths by iteration is exactly the kind of thing
    // that could make two calls to it disagree.
    const JoltGuard jolt;

    const auto first = golfGtiMk7();
    const auto second = golfGtiMk7();
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    REQUIRE(first->rackTravelPerInput == second->rackTravelPerInput);
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(first->corners[index].springRate == second->corners[index].springRate);
        REQUIRE(first->corners[index].springFreeLength == second->corners[index].springFreeLength);
    }

    const auto world = PhysicsWorld::create(generateProvingGround(plate(120.0)).value());
    REQUIRE(world.has_value());

    const auto drive = [&](VehicleState& state, const int ticks)
    {
        for (auto step = 0; step < ticks; step++)
        {
            auto input = VehicleInput{};
            input.steering = std::sin(static_cast<double>(step) * 0.01) * 0.2;

            REQUIRE(stepVehicle(first.value(), state, input, noDriveTorque, world.value(), tick).has_value());
        }
    };

    auto one = VehicleState{};
    one.chassis.position = glm::dvec3(0.0, 0.7, 20.0);
    auto two = one;

    drive(one, 600);
    drive(two, 600);

    REQUIRE(std::memcmp(&one, &two, sizeof(VehicleState)) == 0);

    SECTION("and a state restored from its own bytes carries on identically")
    {
        auto bytes = std::vector<unsigned char>(sizeof(VehicleState));
        std::memcpy(bytes.data(), &one, sizeof(VehicleState));

        auto restored = VehicleState{};
        std::memcpy(&restored, bytes.data(), sizeof(VehicleState));

        drive(one, 200);
        drive(restored, 200);

        REQUIRE(std::memcmp(&one, &restored, sizeof(VehicleState)) == 0);
    }
}

TEST_CASE("the imported driveline is the one the file states", "[physics][golf][driveline]")
{
    const auto driveline = golfGtiMk7Driveline();

    SECTION("seven forward gears, a final drive, and a reverse that goes backwards")
    {
        REQUIRE(driveline.gearbox.ratios.size() == 7);
        REQUIRE(driveline.gearbox.ratios.back() == Catch::Approx(0.65));
        REQUIRE(driveline.gearbox.finalDrive == Catch::Approx(4.37));

        REQUIRE(driveline.gearbox.reduction(7) == Catch::Approx(0.65 * 4.37));
        // GEAR_R is stated as -2.9 and `reduction` puts the sign on itself, so carrying the file's own
        // sign through would give a reverse gear that drives the car forwards.
        REQUIRE(driveline.gearbox.reduction(-1) == Catch::Approx(-2.9 * 4.37));
        REQUIRE(driveline.gearbox.reduction(0) == 0.0);
    }

    SECTION("an engine that makes what the car is quoted as making")
    {
        // The peak of the *boosted* curve. power.lut alone is the naturally aspirated one, and reading
        // it as final reports 159 N.m for a car that makes 350.
        auto peakTorque = 0.0;
        auto peakPower = 0.0;
        auto peakPowerSpeed = 0.0;

        for (auto rpm = 1000.0; rpm <= 6800.0; rpm += 10.0)
        {
            const auto speed = rpm * 0.10471975511965977;
            const auto torque = driveline.engine.torque.at(speed);

            peakTorque = std::max(peakTorque, torque);
            if (torque * speed > peakPower)
            {
                peakPower = torque * speed;
                peakPowerSpeed = rpm;
            }
        }

        REQUIRE(peakTorque == Catch::Approx(349.8).margin(0.1));
        // 243.5 bhp, in watts.
        REQUIRE(peakPower == Catch::Approx(243.5 * 745.699872).epsilon(0.02));
        REQUIRE(peakPowerSpeed == Catch::Approx(5780.0).epsilon(0.05));

        REQUIRE(driveline.engine.limiterSpeed == Catch::Approx(6800.0 * 0.10471975511965977));
        REQUIRE(driveline.engine.idleSpeed == Catch::Approx(850.0 * 0.10471975511965977));
        REQUIRE(driveline.engine.inertia == Catch::Approx(0.150));
        REQUIRE(driveline.engine.coastTorque == Catch::Approx(75.0));
    }

    SECTION("front wheel drive through a ramped clutch pack")
    {
        REQUIRE(driveline.driven == raceengine::DrivenAxle::Front);

        // drivetrain.ini [DIFFERENTIAL] POWER/COAST/PRELOAD, which `clutchPackLsd` already expresses:
        // AC's two ramps are the fraction of the torque going through the diff that becomes locking
        // torque, on power and off it, and that is exactly what the struct's fields mean. Nothing had
        // to be built for it.
        REQUIRE(driveline.differential.preload == Catch::Approx(0.0));
        REQUIRE(driveline.differential.powerRamp == Catch::Approx(0.25));
        REQUIRE(driveline.differential.coastRamp == Catch::Approx(0.25));

        // A preload of zero does not make it open. Asked to put 400 N.m across a pair of wheels
        // turning at different speeds it holds a quarter of that rather than nothing.
        auto state = raceengine::DifferentialState{};
        const auto split = driveline.differential.split(
            state,
            raceengine::DifferentialSides{
                .leftSpeed = 100.0, .rightSpeed = 90.0, .leftInertia = 1.2, .rightInertia = 1.2, .input = 400.0},
            tick);

        REQUIRE(state.capacity == Catch::Approx(100.0));
        REQUIRE(split.left < split.right);
        REQUIRE(split.right - split.left == Catch::Approx(200.0));
    }
}

TEST_CASE("the car's own data steers it correctly, with no help from a setup sheet", "[physics][golf][convention]")
{
    // The Golf drove correctly for a day only because `assets/Setups/golf-gti-mk7.setup` carried
    // `steering.invert 0`, forcing the rack positive over a derivation that answered negative. That
    // is a rendering-frame fault being corrected two layers away, in the vehicle, by a file that
    // claims in its own header to change nothing as shipped.
    //
    // The derivation was wrong because it read toe — a mirrored quantity — off a corner whose side
    // label was itself mirrored. `outboardSign` states the frame's handedness in one place now, the
    // corner sides follow it, and the sheet's line is commented out. This pins that it can stay
    // that way: the car has to steer correctly on its own data.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    REQUIRE(setup->rackTravelPerInput > 0.0);
    REQUIRE(setup->rackTravelPerInput == Catch::Approx(0.0700).margin(5e-4));
}
