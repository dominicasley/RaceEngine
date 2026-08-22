// Gravity along a slope: whether a car on a hill knows it is on one.
//
// **It did not, until 2026-08-23.** `stepVehicle` applied the tyre's vertical load along the world's
// up rather than along the contact normal, and closed the tyre's spring on the wheel's *vertical*
// velocity. On a slope that force cancels vertical gravity exactly and leaves nothing along the
// surface, so the car was held up and held still: in neutral on a 45% slope with no brakes it moved
// **0.0000 m in five seconds**, and a DSG creeping in gear climbed one at walking pace on a quarter
// of the power that would take.
//
// What survived, and why it went unnoticed for two milestones, is that the *in-plane* tyre forces were
// always taken in the patch's own frame — `contact.forward` is the heading projected onto
// `patch.normal` — so banking and kerbs produced grip correctly and only the load was flat-world.
//
// These are the tests that would have caught it. There were none: `FeatureKind::Slope` exists in
// `ProvingGround.cppm` for exactly this and had only ever been used by
// `the imported car holds a fifteen degree slope on its brakes`, which passed because nothing was
// trying to move the car and would have passed identically with the brakes off. This file is the
// missing half of that criterion.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::sampleProvingGround;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;

// `Vehicle.cppm`'s own `earthGravity`, which is `inline constexpr` and not exported, so it is restated
// here rather than reached for. The one constant in that file that is not a placeholder.
constexpr auto standardGravity = 9.80665;

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

// The car starts in the middle, because a car on a hill goes *downhill* and a plate laid out for
// acceleration has no room in that direction.
constexpr auto startStation = 400.0;

[[nodiscard]] ProvingGroundDescriptor slopedGround(const double angle)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 900.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 2.0;
    descriptor.slopeAngle = angle;
    descriptor.features = {Feature{.kind = FeatureKind::Slope, .from = 0.0, .to = descriptor.length}};

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

// Put the car down *on* the slope and pitched to match it, then hold it still on the brakes until it
// has settled. A level car dropped onto a grade puts one axle into the ground and the other in the
// air, which settles as a bounce and reads as a car that will not hold station.
struct Parked
{
    VehicleSetup setup;
    VehicleState state;
};

[[nodiscard]] Parked park(const PhysicsWorld& world, const ProvingGroundDescriptor& ground, const double angle)
{
    auto parked = Parked{.setup = golfGtiMk7().value(), .state = VehicleState{}};

    const auto surface = sampleProvingGround(ground, 0.0, startStation).height;
    parked.state.chassis.position = glm::dvec3(0.0, surface + designHeight(parked.setup), startStation);
    parked.state.chassis.orientation = glm::angleAxis(-angle, glm::dvec3(1.0, 0.0, 0.0));

    auto holding = VehicleInput{};
    holding.brake = 1.0;

    for (auto step = 0; step < 1800; step++)
    {
        REQUIRE(stepVehicle(parked.setup, parked.state, holding, noDriveTorque, world, tick).has_value());
    }

    // The precondition every number below is taken against: it really is standing still, and it
    // really is on the plate.
    CAPTURE(glm::length(parked.state.chassis.linearVelocity), parked.state.chassis.position.z, angle);
    REQUIRE(glm::length(parked.state.chassis.linearVelocity) < 0.05);
    REQUIRE(parked.state.chassis.position.z > 5.0);
    REQUIRE(parked.state.chassis.position.z < ground.length - 5.0);

    return parked;
}

// Speed along the surface rather than along z, so up and down report the same quantity.
[[nodiscard]] double alongSlope(const VehicleState& state, const double angle)
{
    return state.chassis.linearVelocity.z * std::cos(angle) + state.chassis.linearVelocity.y * std::sin(angle);
}

[[nodiscard]] double gradeAngle(const double percent)
{
    return std::atan(percent / 100.0);
}

} // namespace

TEST_CASE("a car in neutral on a hill rolls down it", "[physics][vehicle][grade]")
{
    const JoltGuard jolt;

    // Signed, so a slope that rises with +z is climbed by driving forward and a car released on one
    // rolls backwards down it.
    const auto percent = GENERATE(5.0, 10.0, 20.0, 30.0);
    CAPTURE(percent);

    const auto angle = gradeAngle(percent);
    const auto ground = slopedGround(angle);
    const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
    REQUIRE(world.has_value());

    auto parked = park(world.value(), ground, angle);
    const auto from = parked.state.chassis.position;

    // Three seconds of nothing at all: no engine, no driveline, no brakes.
    for (auto step = 0; step < 1080; step++)
    {
        REQUIRE(stepVehicle(parked.setup, parked.state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    const auto moved = parked.state.chassis.position - from;
    const auto speed = alongSlope(parked.state, angle);

    // Downhill is −z here, and the whole point is that this used to be zero.
    CAPTURE(moved.z, moved.y, speed);
    REQUIRE(moved.z < -0.5);
    REQUIRE(speed < 0.0);

    SECTION("and it accelerates at g sin(theta) less a constant, which is what rolling resistance is")
    {
        // **Bounded as a difference and not as a ratio**, and that is the whole shape of the claim.
        // What separates the car from free-rolling is rolling resistance plus a little drag, and
        // rolling resistance goes with *load* rather than with grade — so it is very nearly the same
        // deceleration at every angle here, and as a fraction of `g sin(theta)` it therefore grows as
        // the slope gets shallower. Measured, the shortfall is 0.13 m/s² at 5% and the same 0.13 at
        // 20%; a ratio bound tight enough to be worth anything at 20% fails at 5% for no reason but
        // arithmetic. A first pass at this asserted 80% of free and failed at 5% on exactly that.
        //
        // Stated this way it is also a **stronger** test than the ratio was: a missing cosine, or a
        // load taken along the wrong axis, scales with the grade and would show up as a deficit that
        // grows with it. A constant deficit is the signature of a constant force, which is the only
        // thing that should be there.
        const auto free = standardGravity * std::sin(angle);
        const auto reached = std::abs(speed) / 3.0;
        const auto shortfall = free - reached;

        CAPTURE(free, reached, shortfall, reached / free);

        // Never faster than free-rolling: that would be gravity applied twice.
        REQUIRE(reached < free);

        // And the gap is a plausible constant retarding acceleration rather than a fraction of the
        // driving one. 0.13 m/s² on this car; the band is wide enough for drag to vary across the
        // speeds these grades reach and far too narrow for anything proportional to slip through.
        REQUIRE(shortfall > 0.02);
        REQUIRE(shortfall < 0.30);
    }

    SECTION("and it stays on the road while it does")
    {
        // dy/dz over the run is the slope's own tangent if the car is running along the surface
        // rather than sinking into it or climbing out. This is what separates "gravity reaches the
        // car" from "the contact solve has come apart".
        const auto ratio = moved.y / moved.z;

        CAPTURE(ratio, std::tan(angle));
        REQUIRE(ratio == Catch::Approx(std::tan(angle)).epsilon(0.02));
    }
}

TEST_CASE("the brakes are what hold a parked car on a slope", "[physics][vehicle][grade]")
{
    // **`the imported car holds a fifteen degree slope on its brakes` was passing because nothing was
    // trying to move the car.** It is a real test now, and this is its other half: the same slope with
    // the brakes *off* must not hold, or the first one is measuring nothing again.
    //
    // Fifteen degrees is criterion 2's own figure, and 26.8% is well past the ~10% a Mk7's DSG creep
    // torque can climb, so nothing in the transmission can quietly rescue it either.
    const JoltGuard jolt;

    const auto angle = 0.26180;
    const auto ground = slopedGround(angle);
    const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
    REQUIRE(world.has_value());

    auto held = park(world.value(), ground, angle);
    const auto heldFrom = held.state.chassis.position;

    auto braking = VehicleInput{};
    braking.brake = 1.0;

    for (auto step = 0; step < 1800; step++)
    {
        REQUIRE(stepVehicle(held.setup, held.state, braking, noDriveTorque, world.value(), tick).has_value());
    }

    const auto slid = glm::distance(held.state.chassis.position, heldFrom);
    CAPTURE(slid);
    REQUIRE(slid < 0.05);

    // What the brakes are holding *against*, stated so the test carries its own justification: a
    // 1348 kg car on a 15 degree slope is being pulled down it by about 3.4 kN.
    const auto pull = held.state.chassis.mass * standardGravity * std::sin(angle);
    CAPTURE(pull);
    REQUIRE(pull > 3000.0);

    SECTION("and with them off it does not hold")
    {
        auto free = park(world.value(), ground, angle);
        const auto freeFrom = free.state.chassis.position;

        for (auto step = 0; step < 1800; step++)
        {
            REQUIRE(
                stepVehicle(free.setup, free.state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
        }

        const auto rolled = glm::distance(free.state.chassis.position, freeFrom);
        CAPTURE(rolled);
        REQUIRE(rolled > 10.0);
    }
}

TEST_CASE("a car on a slope carries the load a slope gives it", "[physics][vehicle][grade]")
{
    // The load falls as the cosine of the grade, which is the arithmetic the fix turns on: the tyre's
    // spring compresses by the *perpendicular* distance to the road, and a vertical overlap against a
    // tilted plane is that distance divided by the cosine.
    //
    // Asserted on the total rather than per corner, because a slope also transfers load fore and aft
    // and that split is a different claim.
    const JoltGuard jolt;

    const auto percent = GENERATE(0.0, 20.0, 40.0);
    CAPTURE(percent);

    const auto angle = gradeAngle(percent);
    const auto ground = slopedGround(angle);
    const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
    REQUIRE(world.has_value());

    auto parked = park(world.value(), ground, angle);

    auto braking = VehicleInput{};
    braking.brake = 1.0;

    const auto stepped = stepVehicle(parked.setup, parked.state, braking, noDriveTorque, world.value(), tick);
    REQUIRE(stepped.has_value());

    auto total = 0.0;
    for (const auto& corner : stepped->corners)
    {
        total += corner.forces.tireVertical;
    }

    const auto weight = parked.state.chassis.mass * standardGravity;
    const auto expected = weight * std::cos(angle);

    CAPTURE(total, weight, expected, total / weight);

    // Five percent, which is loose enough for the unsprung masses sitting outside the sprung ledger
    // and tight enough that a missing cosine — 8% at 40% grade — could not pass.
    REQUIRE(total == Catch::Approx(expected).epsilon(0.05));
}

TEST_CASE("a creeping DSG answers a grade with its speed", "[physics][vehicle][grade][creep]")
{
    // **T2b criterion 2, and it was blocked on the flat-world load rather than on the transmission.**
    // Creep commands a *torque* and lets the resulting speed fall out of the load, which is why a DSG
    // creeps slower uphill, faster downhill, and gives up rather than climbing past a point. A speed
    // regulator gets all three wrong. None of it could be measured while no car rolled down a hill.
    //
    // The grade it gives up on is not a threshold anybody set: 30 N·m at the clutch through this car's
    // first gear and final drive is `30 * 3.19 * 4.37 / 0.3186` = 1313 N at the road, against a 1348 kg
    // car, which balances at sin(theta) = 0.0993 — **9.97%**. Measured, it holds station at 10% and
    // rolls back above it. That is a prediction from the commanded torque landing on the nose, and it
    // is the strongest evidence there is that creep is a torque command and not a speed one.
    const JoltGuard jolt;

    const auto driveline = raceengine::golfGtiMk7Driveline();

    // Downhill, flat, gently uphill, and past the point it can hold. Each is one world, so the list is
    // short deliberately — `[.creep-grade]` sweeps it finely and prints the table.
    struct Case
    {
        double percent;
        double atLeast;
        double atMost;
    };

    // km/h along the slope. Bounded both ways so the ordering is asserted rather than eyeballed, and
    // sourced from `[.creep-grade]`'s table with room either side.
    const auto sample = GENERATE(Case{-10.0, 14.0, 22.0}, Case{0.0, 6.5, 8.0}, Case{6.0, 6.0, 7.6},
                                 Case{15.0, -25.0, -8.0});
    CAPTURE(sample.percent);

    const auto angle = gradeAngle(sample.percent);
    const auto ground = slopedGround(angle);
    const auto world = PhysicsWorld::create(generateProvingGround(ground).value());
    REQUIRE(world.has_value());

    auto parked = park(world.value(), ground, angle);

    auto drive = raceengine::DrivelineState{};
    raceengine::startEngine(driveline, drive);

    const auto inertias = raceengine::wheelInertias(parked.setup);
    auto road = std::array<double, cornerCount>{};

    const auto step = [&](const VehicleInput& input)
    {
        const auto torques = raceengine::stepDriveline(
            driveline, drive,
            {parked.state.corners[0].wheelSpeed, parked.state.corners[1].wheelSpeed,
             parked.state.corners[2].wheelSpeed, parked.state.corners[3].wheelSpeed},
            inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(parked.setup, parked.state, input, torques->wheel, world.value(), tick);
        REQUIRE(stepped.has_value());
        road = raceengine::roadTorques(stepped.value());
    };

    // Idling in gear on the brakes first — the precondition every creep number in this project is
    // taken against, and the one a fixture fault once got wrong at the cost of a whole TCU build.
    auto idling = VehicleInput{};
    idling.brake = 1.0;
    idling.gear = 1;

    for (auto held = 0; held < 720; held++)
    {
        step(idling);
    }

    REQUIRE(drive.clutchPedal > 0.85);
    REQUIRE(drive.creepCommand == 0.0);

    auto creeping = VehicleInput{};
    creeping.gear = 1;

    for (auto going = 0; going < 16 * 360; going++)
    {
        step(creeping);
    }

    const auto settled = alongSlope(parked.state, angle) * 3.6;
    CAPTURE(settled, drive.creepCommand);

    REQUIRE(settled > sample.atLeast);
    REQUIRE(settled < sample.atMost);

    // And it is still creep doing it rather than something else having taken the clutch: the command
    // is the full creep torque throughout, on every one of these grades.
    REQUIRE(drive.creepCommand == Catch::Approx(driveline.autoClutch.creepTorque));
}
