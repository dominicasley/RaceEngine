// Idle creep and low-speed clutch control: what a dual clutch does in gear at walking pace, from
// nobody touching anything up to the driver asking for everything.
//
// A DSG crawls because its controller *commands* it to. There is nothing in a pair of dry plates that
// leaks torque the way a fluid coupling does at stall, so creep is a rule rather than a consequence —
// and until it existed this model's answer to "green light, foot off everything" was that the car sat
// inert with its clutch fully open. Measured on the fixture below, the car moved **0.1 mm in twenty
// seconds**.
//
// The rule is a *torque* command and not a speed one, which is the whole of why it behaves. A real TCU
// commands a clutch pressure and lets the speed fall out of the load; the settling speed is then not a
// parameter anybody chose but idle through first gear on this wheel, which is where a closed clutch
// puts the car whatever the controller wanted.
//
// The measurements these thresholds come from are in `CreepDiagnosticProbe.cpp` — `[.creep]` and its
// neighbours. **Criterion 2 of the brief, grade response, is not here and cannot be**: the vehicle
// model applies the tyre's vertical load along world up rather than along the contact normal, so a car
// in neutral on a forty-five percent slope does not roll down it at all. That is measured in
// `[.creep-gravity]` and written up in `docs/vehicle-physics.md`; it is not a transmission fault and
// it is not this file's to assert around.


#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::advanceCreep;
using raceengine::antiStallPedal;
using raceengine::AutoClutch;
using raceengine::autoClutchPedal;
using raceengine::bringUpJolt;
using raceengine::clutchCapacity;
using raceengine::clutchPedalForCapacity;
using raceengine::cornerCount;
using raceengine::creepPedal;
using raceengine::DriveCoupling;
using raceengine::DriveCouplingKind;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::EngineState;
using raceengine::FrictionClutch;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
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

[[nodiscard]] ProvingGroundDescriptor plate(const double size = 400.0)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = 40.0;
    descriptor.cellSize = 2.0;
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

struct Sample
{
    double time = 0.0;
    double speed = 0.0;
    double station = 0.0;
    double engineSpeed = 0.0;
    double clutchTorque = 0.0;
    double slipEnergy = 0.0;
    double creepCommand = 0.0;
    bool locked = false;
};

// What the driver does for the length of one run.
struct Script
{
    double brake = 0.0;
    double throttle = 0.0;
    double throttleFrom = 1e30;
    double throttleTo = 1e30;
};

// Stand the car up, start it, idle it in gear on the brakes, **assert the state it is about to be
// measured from**, and then run the script.
//
// The precondition block is the point. `DrivelineState::clutchPedal` defaults to 0.0, which is a
// fully *engaged* clutch, and a fixture that starts the engine and immediately measures is measuring
// a cold start — that exact fault produced three wrong conclusions and cost a whole TCU build in this
// project once already. None of these would have let it through.
[[nodiscard]] std::vector<Sample> runCreep(const VehicleSetup& setup, const DrivelineSetup& driveline,
                                           const PhysicsWorld& world, const double seconds, const Script& script)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        auto holding = VehicleInput{};
        holding.brake = 1.0;

        REQUIRE(stepVehicle(setup, state, holding, noDriveTorque, world, tick).has_value());
    }

    auto driveState = DrivelineState{};
    startEngine(driveline, driveState);

    const auto inertias = wheelInertias(setup);
    auto road = std::array<double, cornerCount>{};

    const auto step = [&](const VehicleInput& input)
    {
        const auto torques =
            stepDriveline(driveline, driveState,
                          {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed, state.corners[2].wheelSpeed,
                           state.corners[3].wheelSpeed},
                          inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());

        return torques.value();
    };

    {
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto held = 0; held < 720; held++)
        {
            static_cast<void>(step(idling));
        }
    }

    {
        const auto atRest = glm::length(state.chassis.linearVelocity);
        CAPTURE(atRest, driveState.clutchPedal, driveState.engineSpeed, driveState.gear, driveState.creepCommand);

        // Stationary, in first, engine alight at its own idle, and the clutch open — which is the one
        // a creep rule could plausibly break, because a rule that fired through a full brake
        // application would show up here rather than three conclusions downstream.
        REQUIRE(atRest < 0.05);
        REQUIRE(driveState.gear == 1);
        REQUIRE(driveState.engine == EngineState::Running);
        REQUIRE(driveState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
        REQUIRE(driveState.engineSpeed < 1.5 * driveline.engine.idleSpeed);
        REQUIRE(driveState.clutchPedal > 0.85);
        REQUIRE(driveState.creepCommand == 0.0);

        for (const auto& corner : state.corners)
        {
            REQUIRE(std::abs(corner.wheelSpeed) < 0.5);
        }
    }

    auto run = std::vector<Sample>{};
    const auto ticks = static_cast<int>(seconds * 360.0);
    run.reserve(static_cast<std::size_t>(ticks));

    for (auto index = 1; index <= ticks; index++)
    {
        const auto time = static_cast<double>(index) * tick;

        auto input = VehicleInput{};
        input.brake = script.brake;
        input.throttle = time >= script.throttleFrom && time < script.throttleTo ? script.throttle : 0.0;
        input.gear = 1;

        const auto torques = step(input);

        run.push_back(Sample{.time = time,
                             .speed = state.chassis.linearVelocity.z,
                             .station = state.chassis.position.z,
                             .engineSpeed = driveState.engineSpeed,
                             .clutchTorque = torques.clutch,
                             .slipEnergy = torques.slipEnergy,
                             .creepCommand = torques.creepCommand,
                             .locked = torques.clutchLocked});
    }

    // **The car stayed on the plate.** A creep test travels further than it looks — thirty seconds at
    // two metres a second is sixty — and a fixture in this project has already produced a fictional
    // finding by running its car off the end of its ground and losing support.
    CAPTURE(state.chassis.position.z);
    REQUIRE(state.chassis.position.z > 5.0);
    REQUIRE(state.chassis.position.z < 395.0);

    return run;
}

[[nodiscard]] double settledSpeed(const std::vector<Sample>& run, const double from)
{
    auto total = 0.0;
    auto count = 0;

    for (const auto& sample : run)
    {
        if (sample.time >= from)
        {
            total += sample.speed;
            count++;
        }
    }

    return count > 0 ? total / static_cast<double>(count) : 0.0;
}

[[nodiscard]] int lockTransitions(const std::vector<Sample>& run, const double from)
{
    auto transitions = 0;
    auto previous = false;
    auto started = false;

    for (const auto& sample : run)
    {
        if (sample.time < from)
        {
            continue;
        }

        transitions += started && sample.locked != previous ? 1 : 0;
        previous = sample.locked;
        started = true;
    }

    return transitions;
}

[[nodiscard]] double worstStep(const std::vector<Sample>& run, const double from, const double to)
{
    auto worst = 0.0;
    auto previous = 0.0;
    auto started = false;

    for (const auto& sample : run)
    {
        if (sample.time < from || sample.time > to)
        {
            continue;
        }

        worst = started ? std::max(worst, std::abs(sample.clutchTorque - previous)) : worst;
        previous = sample.clutchTorque;
        started = true;
    }

    return worst;
}

// Idle through first gear on this wheel: where a closed clutch puts the car, and therefore what creep
// converges on. **Not a chosen number.** The three figures it is made of are engine.ini's IDLE,
// drivetrain.ini's GEAR_1 and FINAL, and tyres.ini's RADIUS, so it moves when the car's own data does
// and there is no second statement of it anywhere to fall out of step.
[[nodiscard]] double couplingPointSpeed(const VehicleSetup& setup, const DrivelineSetup& driveline)
{
    return driveline.engine.idleSpeed / (driveline.gearbox.ratios.front() * driveline.gearbox.finalDrive) *
           setup.corners.front().hardpoints.wheelRadius;
}

} // namespace

TEST_CASE("a clutch's capacity reads backwards as well as forwards", "[physics][clutch][creep]")
{
    const auto clutch = FrictionClutch{};

    // The round trip, which is the property the creep rule leans on: ask for a torque, get a pedal,
    // and that pedal delivers the torque asked for.
    for (const auto wanted : {5.0, 30.0, 100.0, 240.0, 400.0, 479.0})
    {
        const auto pedal = clutchPedalForCapacity(clutch, wanted);
        CAPTURE(wanted, pedal);
        REQUIRE(clutchCapacity(clutch, pedal) == Catch::Approx(wanted).epsilon(1e-9));
    }

    SECTION("and both flat ends answer with their least engaged pedal")
    {
        // Commanding nothing must give a *fully released* pedal, not the first pedal at which the
        // clamp happens to reach zero. The two are the same in torque and different in the pedal
        // channel, which is what a launch fixture asserts its preconditions on — it requires the
        // pedal above 0.85, and the curve reaches zero clamp at 0.75.
        REQUIRE(clutchPedalForCapacity(clutch, 0.0) == 1.0);
        REQUIRE(clutchPedalForCapacity(clutch, -10.0) == 1.0);

        // And more clamp than the clutch has is the top of the curve rather than the floor of the
        // pedal, for the same reason in the other direction.
        REQUIRE(clutchPedalForCapacity(clutch, 480.0) == Catch::Approx(0.35));
        REQUIRE(clutchPedalForCapacity(clutch, 10000.0) == Catch::Approx(0.35));
    }

    SECTION("creep lives in the narrow band the pedal curve leaves for it")
    {
        // `roadClutchEngagement` gives zero clamp above pedal 0.75 and full clamp below 0.35, so
        // everything metered has to happen in between. A creep torque of 30 N.m sits at 0.713, which
        // is worth knowing because it is also where the previous TCU attempt found it had no
        // authority at all.
        REQUIRE(clutchPedalForCapacity(clutch, 30.0) == Catch::Approx(0.7131).epsilon(0.001));
    }
}

TEST_CASE("what cancels a creep command", "[physics][clutch][creep]")
{
    const auto assist = AutoClutch{};
    const auto held = assist.creepTorque;

    // Long enough that the rate limit never binds, so this is testing the decision and not the ramp.
    constexpr auto longEnough = 1.0;

    SECTION("in gear, engine alight, no pedals — it commands its creep torque")
    {
        REQUIRE(advanceCreep(assist, 0.0, 0.0, 0.0, true, true, longEnough) == Catch::Approx(held));
    }

    SECTION("and every one of the four cancels it outright")
    {
        // Out of gear there is nothing to creep against; a dead engine is not creeping anywhere; a
        // driver on the accelerator has taken the clutch back; a driver on the brake has asked the
        // car to stay where it is.
        REQUIRE(advanceCreep(assist, held, 0.0, 0.0, false, true, longEnough) == 0.0);
        REQUIRE(advanceCreep(assist, held, 0.0, 0.0, true, false, longEnough) == 0.0);
        REQUIRE(advanceCreep(assist, held, 0.0, 1.0, true, true, longEnough) == 0.0);
        REQUIRE(advanceCreep(assist, held, 1.0, 0.0, true, true, longEnough) == 0.0);

        auto off = assist;
        off.creep = false;
        REQUIRE(advanceCreep(off, held, 0.0, 0.0, true, true, longEnough) == 0.0);

        off = assist;
        off.enabled = false;
        REQUIRE(advanceCreep(off, held, 0.0, 0.0, true, true, longEnough) == 0.0);
    }

    SECTION("the accelerator has a dead band and the brake does not")
    {
        // A sensor resting a percent off zero must not hold creep off for ever, which is what the
        // dead band is for. The brake is a threshold with nothing under it because a brake-light
        // switch is exactly that, and because the alternative — a taper — leaves a band where the
        // brakes hold the car and the clutch is still slipping into them.
        REQUIRE(advanceCreep(assist, 0.0, 0.0, 0.5 * assist.creepThrottleLift, true, true, longEnough) ==
                Catch::Approx(held));
        REQUIRE(advanceCreep(assist, 0.0, 0.0, 2.0 * assist.creepThrottleLift, true, true, longEnough) == 0.0);

        REQUIRE(advanceCreep(assist, 0.0, 0.99 * assist.creepBrakeCut, 0.0, true, true, longEnough) ==
                Catch::Approx(held));
        REQUIRE(advanceCreep(assist, 0.0, assist.creepBrakeCut, 0.0, true, true, longEnough) == 0.0);
    }

    SECTION("and it is a ramp in both directions, at its own two rates")
    {
        const auto rising = advanceCreep(assist, 0.0, 0.0, 0.0, true, true, 0.01);
        REQUIRE(rising == Catch::Approx(0.01 * assist.creepApplyRate));

        const auto falling = advanceCreep(assist, held, 1.0, 0.0, true, true, 0.01);
        REQUIRE(falling == Catch::Approx(held - 0.01 * assist.creepReleaseRate));

        // It drops faster than it builds, which is what lets a brake threshold be a threshold.
        REQUIRE(assist.creepReleaseRate > assist.creepApplyRate);
    }
}

TEST_CASE("a converter is not given a creep rule as well", "[physics][clutch][creep]")
{
    // A fluid coupling at stall already passes torque — that is what a fluid coupling *is*, and it is
    // why an automatic creeps without anything deciding that it should. A commanded creep on top of
    // one would be the same behaviour modelled twice, and the second copy would be the one with a
    // number in it.
    auto coupling = DriveCoupling{};
    coupling.kind = DriveCouplingKind::TorqueConverter;

    REQUIRE(creepPedal(coupling, 30.0) == 1.0);
    REQUIRE(creepPedal(coupling, 500.0) == 1.0);

    coupling.kind = DriveCouplingKind::FrictionClutch;
    REQUIRE(creepPedal(coupling, 30.0) == Catch::Approx(0.7131).epsilon(0.001));
    REQUIRE(creepPedal(coupling, 0.0) == 1.0);
}

TEST_CASE("a car rolling backwards in a forward gear is not caught up with its gear",
          "[physics][clutch][creep][regression]")
{
    // **`autoClutchPedal` used to take `std::abs` of the clutch-side speed.** The sign of the gear is
    // already in that number — a reverse ratio is negative, so a car reversing in reverse arrives
    // positive exactly as a car driving forward in first does — so the magnitude said that a car
    // rolling *backwards* in a forward gear had caught up with it, and clamped the clutch shut in
    // proportion to how fast it was rolling away downhill.
    //
    // It was held back for a session because it moved the driving golden by 23,191 of 32,400 blocks,
    // and then stopped moving it at all: what it had been reaching was the cold-start ring the launch
    // regulator caused, and with that regulator gone the correction is byte-identical on both gates.
    const auto assist = AutoClutch{};
    const auto idle = 89.0;

    // Rolling forward at the coupling point in the gear that suits it: caught up, clutch closed.
    REQUIRE(autoClutchPedal(assist, idle, idle, idle, 0.0, true) < 0.3);

    // The same speed the other way, which is a car rolling away downhill. The clutch must stay open.
    REQUIRE(autoClutchPedal(assist, idle, -idle, idle, 0.0, true) == 1.0);

    // And it stays open however fast the rollback gets, which is the failure mode: the old term made
    // the clamp *grow* with it.
    REQUIRE(autoClutchPedal(assist, idle, -10.0 * idle, idle, 0.0, true) == 1.0);
}

TEST_CASE("a dual clutch creeps when the brakes come off", "[physics][clutch][creep]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    const auto run = runCreep(setup, driveline, world.value(), 16.0, Script{});

    const auto settled = settledSpeed(run, 12.0);
    const auto coupling = couplingPointSpeed(setup, driveline);
    CAPTURE(settled, coupling, settled * 3.6);

    // **The band is the brief's and the sharp number is the car's own.** A Mk7 DSG creeps at roughly
    // 5-8 km/h on the flat, and this model's answer is not a parameter anybody chose: a clutch that
    // has closed puts the engine's idle through first gear and the wheel's radius, and those three
    // figures are engine.ini's IDLE (850 rpm), drivetrain.ini's GEAR_1 x FINAL (3.19 x 4.37) and
    // tyres.ini's RADIUS (0.3186 m). They come to 2.034 m/s, which is 7.32 km/h and lands in the band
    // without anything being aimed at it.
    REQUIRE(settled * 3.6 > 5.0);
    REQUIRE(settled * 3.6 < 8.0);

    // And it settles just under the coupling point rather than at it, by the longitudinal slip the
    // tyre needs to carry the car's own rolling resistance. Bounded both ways: above the coupling
    // point would be a car being driven by something other than an idling engine.
    REQUIRE(settled < coupling);
    REQUIRE(settled > 0.97 * coupling);

    SECTION("and with the rule switched off it does not move at all")
    {
        // The same code, the same fixture, the same inputs. This is what the model said before the
        // rule existed, and it is why an acceptance test asserting "the engine is undisturbed at a
        // green light" was asserting the missing rule rather than the automation.
        auto without = driveline;
        without.autoClutch.creep = false;

        const auto inert = runCreep(setup, without, world.value(), 16.0, Script{});

        CAPTURE(inert.back().station);
        REQUIRE(std::abs(inert.back().station - 20.0) < 0.01);
        REQUIRE(std::abs(inert.back().speed) < 0.001);
        REQUIRE(inert.back().creepCommand == 0.0);
    }

    SECTION("the clutch turns the take-up into heat, on the channel a thermal model will read")
    {
        // Creep is the one operating state where a real DSG genuinely heats its clutch, so this is
        // the natural first consumer when thermal lands. Nothing reads it yet; it is asserted
        // non-zero and finite so that it cannot quietly stop being filled.
        const auto energy = run.back().slipEnergy;
        CAPTURE(energy);

        REQUIRE(energy > 100.0);
        REQUIRE(std::isfinite(energy));

        // It stops growing once the clutch has closed, which is the difference between a plate
        // finishing an engagement and a plate being cooked.
        const auto lateGrowth = run.back().slipEnergy - run[static_cast<std::size_t>(14.0 * 360.0)].slipEnergy;
        CAPTURE(lateGrowth);
        REQUIRE(lateGrowth < 5.0);
    }
}

TEST_CASE("asking for torque closes the clutch, rather than letting the engine run away from it",
          "[physics][clutch][creep][engagement]")
{
    // **The rule this pins replaced a launch regulator, and it was reported from the seat before it
    // was measured anywhere**: flooring the throttle from rest, hearing the engine flare, and going
    // nowhere. `autoClutchPedal` used to scale engagement by how far the engine had climbed above idle
    // toward `AutoClutch::launchSpeed` — 250 rad/s, near 2400 rpm — so at full throttle from rest the
    // clutch was deliberately held *open* until the revs arrived, and the clamp then came in against
    // them. No dual clutch does that. It closes the clutch when you ask it for torque.
    //
    // Engine speed does not appear in the engagement law at all now. What is left of the old rule is
    // the anti-stall, which is the only thing allowed to open a clutch the driver has asked to be
    // shut, and which now lives only under the creep band.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate(600.0)).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    auto script = Script{};
    script.throttle = 1.0;
    script.throttleFrom = 0.0;

    const auto run = runCreep(setup, driveline, world.value(), 3.0, script);

    // When the clutch first carries real torque, and where the engine was when it did.
    auto bit = -1.0;
    auto revsAtBite = 0.0;
    auto worstBeforeBite = 0.0;

    for (const auto& sample : run)
    {
        if (bit < 0.0 && std::abs(sample.clutchTorque) > 50.0)
        {
            bit = sample.time;
            revsAtBite = sample.engineSpeed;
        }

        if (bit < 0.0)
        {
            worstBeforeBite = std::max(worstBeforeBite, sample.engineSpeed);
        }
    }

    CAPTURE(bit, revsAtBite, worstBeforeBite, driveline.autoClutch.launchSpeed);

    // It bites, and it bites promptly. The pedal starts at its stop — the car has been idling on the
    // brakes — and comes off at `pedalRate`, so a fifth of a second is the mechanism's own limit and
    // not slack in the rule.
    REQUIRE(bit > 0.0);
    REQUIRE(bit < 0.20);

    // **And the engine has not run away first.** This is the assertion that would have caught the
    // launch regulator: it held the clutch open until the engine reached `launchSpeed` and then kept
    // it near there while slipping, so the revs before the bite were the *target* rather than an
    // incidental. Bounded well under that target, so a rule reintroducing it cannot pass.
    REQUIRE(worstBeforeBite < 0.85 * driveline.autoClutch.launchSpeed);

    SECTION("and the slip closes rather than being held open against the revs")
    {
        const auto locked = std::find_if(run.begin(), run.end(), [](const Sample& sample) { return sample.locked; });
        REQUIRE(locked != run.end());

        CAPTURE(locked->time);
        REQUIRE(locked->time < 0.5);
    }

    SECTION("and the car actually goes")
    {
        CAPTURE(run.back().speed);
        REQUIRE(run.back().speed > 8.0);
    }
}

TEST_CASE("the anti-stall exists only underneath the creep band", "[physics][clutch][creep][antistall]")
{
    // It is a *protective* rule and the band is where the thing it protects against lives. Above it
    // the car is turning the engine rather than the engine turning the car, and an engine being
    // lugged out there is a driver in the wrong gear — which a car is allowed to be, and which a
    // clutch quietly opening itself to prevent would be an assist nobody asked for. It was doing
    // exactly that everywhere, and it is the other half of why the car would not pull away.
    const auto assist = AutoClutch{};
    const auto idle = 89.0;
    const auto dying = 0.6 * idle;

    // Under the band, an engine being dragged down gets the clutch opened for it.
    REQUIRE(antiStallPedal(assist, idle, 0.0, dying) > 0.5);
    REQUIRE(antiStallPedal(assist, idle, 0.99 * assist.grabFraction * idle, dying) > 0.5);

    // Past it, nothing.
    REQUIRE(antiStallPedal(assist, idle, 1.01 * assist.grabFraction * idle, dying) == 0.0);
    REQUIRE(antiStallPedal(assist, idle, 20.0 * idle, dying) == 0.0);

    // A healthy engine asks for nothing wherever it is, which is what keeps the band from being a
    // behaviour of its own rather than a scope on one.
    REQUIRE(antiStallPedal(assist, idle, 0.0, idle) == 0.0);

    // And a car rolling *backwards* is under the band from the wrong side, which is a stall coming.
    REQUIRE(antiStallPedal(assist, idle, -2.0 * idle, dying) > 0.5);
}

TEST_CASE("a light brake holds the car against creep, and holds it quietly", "[physics][clutch][creep]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    auto script = Script{};
    script.brake = driveline.autoClutch.creepBrakeCut;

    const auto run = runCreep(setup, driveline, world.value(), 20.0, script);

    CAPTURE(run.back().station, run.back().slipEnergy);

    // Still, and still for the whole twenty seconds rather than creeping slowly.
    REQUIRE(std::abs(run.back().station - 20.0) < 0.05);
    REQUIRE(std::abs(run.back().speed) < 0.01);

    // **And quietly.** The row that must not exist is one where the car is held *and* the clutch is
    // slipping: a plate carrying creep torque into a brake for as long as a light is red. Measured
    // with the taper this replaced, a tenth of the pedal put 1.3 kW through the clutch indefinitely —
    // 40 kJ over thirty seconds. Here the only slip energy is the initial settle and it stops.
    REQUIRE(run.back().slipEnergy < 1000.0);
    REQUIRE(run.back().slipEnergy - run[static_cast<std::size_t>(10.0 * 360.0)].slipEnergy < 1.0);

    // No chatter in the lock/slip machine either.
    REQUIRE(lockTransitions(run, 1.0) == 0);

    SECTION("and a hair below the cut-off it creeps, which is what keeps that band empty")
    {
        // The cut-off has to sit *below* the brake application at which the brakes can hold the car
        // against full creep, or the cooked-plate band comes back. This car's brakes make 4200 N.m
        // and full creep puts 418 N.m through the front axle, so they cross at a tenth of the pedal
        // and the cut is at a twentieth.
        auto barely = Script{};
        barely.brake = 0.99 * driveline.autoClutch.creepBrakeCut;

        const auto crept = runCreep(setup, driveline, world.value(), 12.0, barely);

        CAPTURE(crept.back().station, crept.back().speed);
        REQUIRE(crept.back().station - 20.0 > 5.0);
    }
}

TEST_CASE("the handover between creep and the throttle has no step in it", "[physics][clutch][creep]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    // Creep for eight seconds — long enough to be settled — then a quarter throttle for four, then
    // back off. A quarter rather than full: the question is whether the *transition* steps, and a
    // launch would drown it in the answer to a different question.
    auto script = Script{};
    script.throttle = 0.25;
    script.throttleFrom = 8.0;
    script.throttleTo = 12.0;

    const auto run = runCreep(setup, driveline, world.value(), 16.0, script);

    const auto creeping = worstStep(run, 6.0, 8.0);
    const auto applying = worstStep(run, 8.0, 9.0);
    const auto driving = worstStep(run, 10.5, 12.0);
    const auto releasing = worstStep(run, 12.0, 13.0);

    CAPTURE(creeping, applying, driving, releasing);

    // Creep is a **floor on the clamp** rather than a mode, so there is no handover to get wrong:
    // whichever rule wants the clutch closed further gets it, and by the time the car is creeping the
    // catching-up term is already asking for more clamp than any creep pressure. Applying throttle
    // therefore does not release anything first — which is the failure a mode switch would have, and
    // it would show up here as a step.
    //
    // **Bounded against the settled trace either side of it rather than against a round number**, so
    // this catches a step appearing rather than pinning today's smoothness. Measured, the transitions
    // are *quieter* than the steady states they join: 3.14 N.m applying against 11.50 while creeping,
    // and 2.86 releasing against 10.96 while driving — the largest jumps in this trace belong to the
    // compliant driveline ringing under ordinary load, not to anything the controller did.
    REQUIRE(applying < creeping);
    REQUIRE(releasing < driving);

    // The car did accelerate, or the section above is asserting a transition that never happened.
    const auto before = settledSpeed(std::vector<Sample>(run.begin(), run.begin() + 8 * 360), 7.0);
    const auto during = settledSpeed(std::vector<Sample>(run.begin(), run.begin() + 12 * 360), 11.0);
    CAPTURE(before, during);
    REQUIRE(during > before + 2.0);
}

TEST_CASE("creep does not hunt", "[physics][clutch][creep]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate(600.0)).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    const auto run = runCreep(setup, driveline, world.value(), 40.0, Script{});

    // Once the take-up has finished, the lock/slip machine settles and stays settled. A controller
    // hunting between commanding torque and giving it back would show as transitions at the rate the
    // two thresholds are crossed, which over 13,000 ticks would be hundreds.
    CAPTURE(lockTransitions(run, 0.0), lockTransitions(run, 3.0));
    REQUIRE(lockTransitions(run, 3.0) == 0);
    REQUIRE(lockTransitions(run, 0.0) < 4);

    // And the engine is back on its governor, not sawing about on it.
    auto lowest = run.back().engineSpeed;
    auto highest = run.back().engineSpeed;
    for (const auto& sample : run)
    {
        if (sample.time < 10.0)
        {
            continue;
        }

        lowest = std::min(lowest, sample.engineSpeed);
        highest = std::max(highest, sample.engineSpeed);
    }

    CAPTURE(lowest, highest);
    REQUIRE(highest - lowest < 0.02 * driveline.engine.idleSpeed);
}

TEST_CASE("creep never reaches a launch", "[physics][clutch][creep][launch]")
{
    // **Criterion 7, and it is a proof rather than a stopwatch.** A launch is full throttle from a
    // car that was held on the brakes, and creep is cancelled outright by either — so the command is
    // exactly zero for every tick of one, `creepPedal` answers a fully released pedal, and
    // `std::min(automatic, 1.0)` is `automatic` to the bit. The 0-100 cannot have moved because the
    // arithmetic did not.
    //
    // Asserted on the command rather than on the time because a stopwatch would only say it did not
    // move *this* time; this says the path was never entered.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate(600.0)).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    auto driveState = DrivelineState{};
    startEngine(driveline, driveState);

    const auto inertias = wheelInertias(setup);
    auto road = std::array<double, cornerCount>{};

    const auto step = [&](const VehicleInput& input)
    {
        const auto torques =
            stepDriveline(driveline, driveState,
                          {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed, state.corners[2].wheelSpeed,
                           state.corners[3].wheelSpeed},
                          inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world.value(), tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());

        return torques.value();
    };

    // Idling in gear on the brakes, which is what a car in the game has been doing long before
    // anybody floors it.
    auto idling = VehicleInput{};
    idling.brake = 1.0;
    idling.gear = 1;

    for (auto held = 0; held < 360; held++)
    {
        REQUIRE(step(idling).creepCommand == 0.0);
    }

    REQUIRE(driveState.clutchPedal > 0.85);

    auto launching = VehicleInput{};
    launching.throttle = 1.0;
    launching.gear = 1;

    for (auto going = 0; going < 2160; going++)
    {
        REQUIRE(step(launching).creepCommand == 0.0);
    }

    // It really was a launch and not a car sitting still.
    CAPTURE(state.chassis.linearVelocity.z);
    REQUIRE(state.chassis.linearVelocity.z > 15.0);
}

TEST_CASE("creep does not close the clutch on a dead engine", "[physics][clutch][creep][antistall]")
{
    // Anti-stall is a floor applied *over* whatever the automation asked for, so a creep command on
    // an engine below its floor opens the clutch rather than closing it: creep does not defeat
    // anti-stall because it is not in a position to. Belt and braces, `advanceCreep` also refuses to
    // command anything at all while the engine is out, which is what keeps the ramp from winding up
    // against a car that is being pushed.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7().value();
    const auto driveline = golfGtiMk7Driveline();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world.value(), tick).has_value());
    }

    // In gear, brakes off, throttle off — the creep state exactly — with the key out.
    auto driveState = DrivelineState{};
    const auto inertias = wheelInertias(setup);
    auto road = std::array<double, cornerCount>{};

    auto input = VehicleInput{};
    input.gear = 1;

    for (auto step = 0; step < 1080; step++)
    {
        const auto torques =
            stepDriveline(driveline, driveState,
                          {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed, state.corners[2].wheelSpeed,
                           state.corners[3].wheelSpeed},
                          inertias, road, input, tick);
        REQUIRE(torques.has_value());

        REQUIRE(torques->creepCommand == 0.0);
        REQUIRE(std::abs(torques->clutch) < 1e-9);

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world.value(), tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());
    }

    REQUIRE(driveState.engine == EngineState::Stalled);

    // It did not creep. Bounded at the same 50 mm the brake case uses rather than tighter: a car
    // dropped onto its springs with no brakes on drifts about twelve millimetres while it settles,
    // and that drift is the fixture rather than the transmission — the exact assertions above, that
    // the command is zero and the clutch carries nothing to a part in a billion, are what actually
    // says creep stayed out of it.
    REQUIRE(std::abs(state.chassis.position.z - 20.0) < 0.05);
}
