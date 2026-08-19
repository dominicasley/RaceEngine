#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::advanceClutchPedal;
using raceengine::AutoClutch;
using raceengine::autoClutchPedal;
using raceengine::bringUpJolt;
using raceengine::clutchCapacity;
using raceengine::cornerCount;
using raceengine::CouplingSides;
using raceengine::CouplingState;
using raceengine::DriveCoupling;
using raceengine::DriveCouplingKind;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::EngineState;
using raceengine::fillDrivelineTelemetry;
using raceengine::FrictionClutch;
using raceengine::generateProvingGround;
using raceengine::idleBypass;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeholderDriveline;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveCoupling;
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

// No road under the wheels, for the cases that hold their wheel speeds by hand.
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

// One tick of driveline and vehicle together, which is the pairing the game's loop makes and the
// only place the two meet. Everything the measurements below read comes out of here, so that a
// number in a report and a number in an assertion are the same number.
struct Sample
{
    double engineSpeed = 0.0;
    double engineTorque = 0.0;
    double clutchTorque = 0.0;
    double clutchSlip = 0.0;
    double slipEnergy = 0.0;
    double pedal = 0.0;
    bool locked = false;
    bool stalled = false;
    double roadSpeed = 0.0;
};

// The two states the game's loop carries, plus the road's last answer — which the driveline reads
// and the vehicle tick writes, so it belongs beside them rather than being rebuilt at each call.
struct Car
{
    VehicleState state;
    DrivelineState engine;
    std::array<double, cornerCount> road{};
};

Sample advance(const VehicleSetup& vehicle, const DrivelineSetup& driveline, const PhysicsWorld& world, Car& car,
               const VehicleInput& input)
{
    auto& state = car.state;
    auto& engine = car.engine;

    const auto torques = stepDriveline(driveline, engine,
                                       {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                        state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                       wheelInertias(vehicle), car.road, input, tick);
    REQUIRE(torques.has_value());

    auto stepped = stepVehicle(vehicle, state, input, torques->wheel, world, tick);
    REQUIRE(stepped.has_value());

    fillDrivelineTelemetry(stepped->telemetry, engine, torques.value());
    car.road = roadTorques(stepped.value());

    return Sample{.engineSpeed = engine.engineSpeed,
                  .engineTorque = stepped->telemetry.engineTorque,
                  .clutchTorque = stepped->telemetry.clutchTorque,
                  .clutchSlip = stepped->telemetry.clutchSlip,
                  .slipEnergy = stepped->telemetry.clutchSlipEnergy,
                  .pedal = engine.clutchPedal,
                  .locked = torques->clutchLocked,
                  .stalled = engine.engine == EngineState::Stalled,
                  .roadSpeed = state.chassis.linearVelocity.z};
}

// Dropped, settled and stationary, with the engine running and its idle already held — so anything
// that moves afterwards moved because of the clutch, and any droop is the clutch's droop rather than
// the governor still finding its own feet.
void settle(const VehicleSetup& vehicle, const DrivelineSetup& driveline, const PhysicsWorld& world, Car& car)
{
    car = Car{};
    car.state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(vehicle, car.state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    startEngine(driveline, car.engine);

    auto idling = VehicleInput{};
    idling.clutch = 1.0;

    for (auto step = 0; step < 720; step++)
    {
        static_cast<void>(advance(vehicle, driveline, world, car, idling));
    }
}

std::size_t transitionsIn(const std::vector<Sample>& run)
{
    auto count = std::size_t{0};

    for (auto index = std::size_t{1}; index < run.size(); index++)
    {
        count += run[index].locked != run[index - 1].locked ? std::size_t{1} : std::size_t{0};
    }

    return count;
}

// The shortest stretch the mode was held for, ignoring the run's own two ends: a stretch that begins
// before the run or continues past it says nothing about how long it would have been.
std::size_t shortestRun(const std::vector<Sample>& run)
{
    auto shortest = run.size();
    auto length = std::size_t{0};
    auto seenTransition = false;

    for (auto index = std::size_t{1}; index < run.size(); index++)
    {
        if (run[index].locked == run[index - 1].locked)
        {
            length++;
            continue;
        }

        if (seenTransition)
        {
            shortest = std::min(shortest, length + 1);
        }

        seenTransition = true;
        length = 0;
    }

    return shortest;
}

} // namespace

TEST_CASE("a clutch's capacity is a clamp load through a pedal curve", "[physics][clutch]")
{
    const auto clutch = FrictionClutch{};

    // 8 kN of clamp, 0.30 of friction, a 0.100 m effective radius and the two faces one plate has.
    REQUIRE(clutchCapacity(clutch, 0.0) == Catch::Approx(480.0));
    REQUIRE(clutchCapacity(clutch, 1.0) == 0.0);

    SECTION("it falls to nothing well before the pedal reaches the floor, and holds full at the top")
    {
        // The travel past the plate being released does nothing, and neither does the free play at
        // the top. Both are real and both are why a pedal is not a proportional control over its
        // whole length.
        REQUIRE(clutchCapacity(clutch, 0.80) == 0.0);
        REQUIRE(clutchCapacity(clutch, 0.20) == Catch::Approx(480.0));
    }

    SECTION("and it is nowhere near linear, which is the whole reason it is data")
    {
        // A linear pedal would put half the clamp at half the travel. This one has all of it above
        // 0.35 and none of it below 0.75, so the entire control range is the band a driver launches
        // on — and moving that band is a change to a curve rather than to any code.
        REQUIRE(clutchCapacity(clutch, 0.5) > 0.5 * 480.0);
        REQUIRE(clutchCapacity(clutch, 0.7) < 0.2 * 480.0);

        auto previous = 481.0;
        for (auto step = 0; step <= 100; step++)
        {
            const auto capacity = clutchCapacity(clutch, static_cast<double>(step) / 100.0);
            REQUIRE(capacity <= previous);
            previous = capacity;
        }
    }
}

TEST_CASE("the coupling slot has two kinds, and one of them says it is not here yet", "[physics][clutch]")
{
    // The slot is an element rather than a clutch because the converter lands in it next. Nothing
    // downstream may know which is fitted, so the only thing that can tell them apart from out here
    // is that one of them refuses.
    auto coupling = DriveCoupling{};

    const auto sides = CouplingSides{.drivingSpeed = 0.0,
                                     .drivenSpeed = 0.0,
                                     .drivingInertia = 1.0,
                                     .drivenInertia = 1.0,
                                     .drivingTorque = 200.0,
                                     .drivenTorque = 0.0,
                                     .capacity = 0.0};

    SECTION("the friction clutch answers, and the pedal is what it answers through")
    {
        auto pressed = CouplingState{};
        const auto held = stepDriveCoupling(coupling, pressed, sides, 0.0, tick);
        REQUIRE(held.has_value());
        REQUIRE(held->torque == Catch::Approx(100.0));

        auto released = CouplingState{};
        const auto none = stepDriveCoupling(coupling, released, sides, 1.0, tick);
        REQUIRE(none.has_value());
        REQUIRE(none->torque == 0.0);
    }

    SECTION("and the converter refuses rather than answering for a model nobody has written")
    {
        coupling.kind = DriveCouplingKind::TorqueConverter;

        auto state = CouplingState{};
        const auto out = stepDriveCoupling(coupling, state, sides, 0.0, tick);

        REQUIRE_FALSE(out.has_value());
        REQUIRE(out.error().find("converter") != std::string::npos);
    }

    SECTION("and a driveline with one fitted fails its tick rather than quietly driving")
    {
        auto setup = placeholderDriveline();
        setup.coupling.kind = DriveCouplingKind::TorqueConverter;

        auto state = DrivelineState{};
        startEngine(setup, state);

        auto input = VehicleInput{};
        input.throttle = 1.0;
        input.gear = 1;

        REQUIRE_FALSE(
            stepDriveline(setup, state, {10.0, 10.0, 10.0, 10.0}, {1.2, 1.2, 1.2, 1.2}, noRoadTorque, input, tick)
                .has_value());
    }
}

TEST_CASE("the idle governor is a bypass valve and behaves like one", "[physics][clutch][idle]")
{
    const auto setup = placeholderDriveline();
    auto integral = 0.0;

    SECTION("above idle it asks for nothing, and asks for nothing however long it is left there")
    {
        for (auto step = 0; step < 3600; step++)
        {
            REQUIRE(idleBypass(setup.engine, 300.0, integral, tick) == 0.0);
        }

        // A valve can only add air. An integrator allowed to run negative here would owe the engine
        // a debt it then pays off by hanging after the next clutch release.
        REQUIRE(integral == 0.0);
    }

    SECTION("below idle it winds up, and stops at the range the output can use")
    {
        auto previous = 0.0;
        for (auto step = 0; step < 360; step++)
        {
            const auto demand = idleBypass(setup.engine, 40.0, integral, tick);
            REQUIRE(demand >= previous);
            REQUIRE(demand <= 1.0);
            previous = demand;
        }

        REQUIRE(previous == Catch::Approx(1.0));
        REQUIRE(integral <= setup.engine.governor.maximumBypass);
    }
}

TEST_CASE("the driver's pedal is not overridden by the automation", "[physics][clutch][autoclutch]")
{
    // The rule the rig depends on. A human slipping the clutch from rest is the best test this model
    // has, and an assist that took the pedal off them would make that test impossible to perform.
    const auto assist = AutoClutch{};

    SECTION("a foot past the free play wins outright, in both directions")
    {
        // The automation wanting the clutch further out than the driver does.
        REQUIRE(advanceClutchPedal(assist, 0.55, 0.55, 1.0, tick) == Catch::Approx(0.55));
        // And wanting it further in.
        REQUIRE(advanceClutchPedal(assist, 0.55, 0.55, 0.0, tick) == Catch::Approx(0.55));
        // Neither blend does this: the more-released of the two lets the automation let out a clutch
        // the driver is biting, and the more-clamped lets it clamp one the driver is slipping.
    }

    SECTION("and a foot off it hands the pedal back where the foot left it, at a foot's speed")
    {
        const auto step = advanceClutchPedal(assist, 0.55, 0.0, 1.0, tick);

        REQUIRE(step > 0.55);
        REQUIRE(step == Catch::Approx(0.55 + assist.pedalRate * tick));
    }

    SECTION("switched off, the pedal is the driver's and nothing else")
    {
        auto disabled = AutoClutch{};
        disabled.enabled = false;

        REQUIRE(advanceClutchPedal(disabled, 1.0, 0.0, 1.0, tick) == 0.0);
    }

    SECTION("the automation asks for the clutch out when the car cannot carry the gear")
    {
        const auto idle = 89.0;

        // Stopped, in gear, off the throttle: nothing to engage for.
        REQUIRE(autoClutchPedal(assist, idle, 0.0, idle, 0.0, true) == Catch::Approx(1.0));
        // Moving fast enough for the gear: the clutch is simply out.
        REQUIRE(autoClutchPedal(assist, idle, 4.0 * idle, 300.0, 0.0, true) == Catch::Approx(0.0));
        // Neutral is neither: there is nothing on the other side of it.
        REQUIRE(autoClutchPedal(assist, idle, 0.0, 300.0, 1.0, false) == Catch::Approx(1.0));
        // And revs the driver asked for are revs it will spend, which is what a launch is.
        REQUIRE(autoClutchPedal(assist, idle, 0.0, 250.0, 1.0, true) == Catch::Approx(0.0));
        REQUIRE(autoClutchPedal(assist, idle, 0.0, 250.0, 0.5, true) == Catch::Approx(0.5));
    }
}

TEST_CASE("dumping the clutch at idle kills the engine", "[physics][clutch][stall]")
{
    const JoltGuard jolt;

    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    auto driveline = placeholderDriveline();
    // The stall is a property of the clutch. The layer above exists to prevent it, so it is switched
    // off to ask the question at all — and switched back on in the section below to show it works.
    driveline.autoClutch.enabled = false;

    auto car = Car{};
    settle(vehicle.value(), driveline, world.value(), car);
    auto& engine = car.engine;

    const auto idled = engine.engineSpeed;
    REQUIRE(idled == Catch::Approx(driveline.engine.idleSpeed).epsilon(0.02));

    auto input = VehicleInput{};
    input.gear = 1;
    input.clutch = 0.0;

    auto run = std::vector<Sample>{};
    for (auto step = 0; step < 1080; step++)
    {
        run.push_back(advance(vehicle.value(), driveline, world.value(), car, input));
    }

    SECTION("it dies, and quickly")
    {
        REQUIRE(engine.engine == EngineState::Stalled);
        REQUIRE(engine.engineSpeed == 0.0);

        const auto died = std::find_if(run.begin(), run.end(), [](const Sample& sample) { return sample.stalled; });
        REQUIRE(died != run.end());

        // Measured at 0.150 s, and bracketed on both sides. Too slow is an engine the clutch cannot
        // drag down; too fast is a threshold doing the work rather than the physics, and the whole
        // fight — the wheels breaking away, spinning up past the engine and handing some of it back
        // before the clutch wins — takes the better part of fifty ticks to play out.
        const auto diedAt = static_cast<double>(std::distance(run.begin(), died) + 1) * tick;
        REQUIRE(diedAt > 0.02);
        REQUIRE(diedAt < 0.30);
    }

    SECTION("and on the way down it never turns backwards, which is what the floors used to do")
    {
        for (const auto& sample : run)
        {
            REQUIRE(sample.engineSpeed >= 0.0);
            REQUIRE(std::isfinite(sample.engineSpeed));
        }
    }

    SECTION("a stalled engine stays stalled until something starts it")
    {
        auto restart = VehicleInput{};
        restart.gear = 1;
        restart.clutch = 1.0;
        restart.throttle = 1.0;

        for (auto step = 0; step < 720; step++)
        {
            const auto sample = advance(vehicle.value(), driveline, world.value(), car, restart);
            REQUIRE(sample.stalled);
            REQUIRE(sample.engineSpeed == 0.0);
            // A dead engine makes no torque, whatever the throttle is doing.
            REQUIRE(sample.engineTorque == 0.0);
        }

        startEngine(driveline, engine);
        REQUIRE(engine.engine == EngineState::Running);
        REQUIRE(engine.engineSpeed == Catch::Approx(driveline.engine.idleSpeed));
    }

    SECTION("and the automation, given the same pedal, does not let it happen")
    {
        // The same model underneath and the same inputs. All that differs is who is on the pedal,
        // which is the point of the layer being a layer.
        driveline.autoClutch.enabled = true;

        auto assisted = Car{};
        settle(vehicle.value(), driveline, world.value(), assisted);

        for (auto step = 0; step < 1080; step++)
        {
            static_cast<void>(advance(vehicle.value(), driveline, world.value(), assisted, input));
        }

        REQUIRE(assisted.engine.engine == EngineState::Running);
        REQUIRE(assisted.engine.engineSpeed == Catch::Approx(driveline.engine.idleSpeed).epsilon(0.05));
    }
}

TEST_CASE("a clutch-slip launch from the pedal is repeatable", "[physics][clutch][launch]")
{
    const JoltGuard jolt;

    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate(600.0)).value());
    REQUIRE(world.has_value());

    // The automation is left *on*, so that the launch below is also the proof that a driver on the
    // pedal keeps it.
    const auto driveline = placeholderDriveline();

    // A driver's launch, from where a driver's foot actually is at the lights: just above the bite,
    // coming up over two seconds. Nothing here is a special case in the model — it is the same pedal,
    // the same curve and the same coupling the automation drives, and the release rate is the whole
    // of what makes this a slipped launch rather than a dumped one.
    //
    // Starting at the floor instead is a worse test and not a gentler one: with the clutch fully out
    // this engine reaches its limiter from idle in about a second, so a slow release from 1.0 arrives
    // at the bite point at 6400 rpm and the launch that follows is a dump with extra steps.
    const auto pedalAt = [](const int step)
    {
        constexpr auto aboveTheBite = 0.78;
        const auto time = static_cast<double>(step) * tick;

        return std::clamp(aboveTheBite * (1.0 - time / 2.0), 0.0, aboveTheBite);
    };

    const auto launch = [&](std::vector<Sample>& run)
    {
        auto car = Car{};
        settle(vehicle.value(), driveline, world.value(), car);

        for (auto step = 0; step < 2160; step++)
        {
            auto input = VehicleInput{};
            input.throttle = 0.4;
            input.gear = 1;
            input.clutch = pedalAt(step);

            run.push_back(advance(vehicle.value(), driveline, world.value(), car, input));
        }
    };

    auto run = std::vector<Sample>{};
    launch(run);

    // Slipping means carrying torque across a speed difference. A pedal on the floor also has a
    // speed difference across it and is not slipping — it is disengaged, and counting those ticks
    // would report the whole run as a slip whatever the driver did.
    const auto slipping = [](const Sample& sample)
    {
        return std::abs(sample.clutchSlip) > 1.0 && std::abs(sample.clutchTorque) > 1.0;
    };

    auto peakSlip = 0.0;
    auto slippingTicks = std::size_t{0};
    for (const auto& sample : run)
    {
        peakSlip = slipping(sample) ? std::max(peakSlip, std::abs(sample.clutchSlip)) : peakSlip;
        slippingTicks += slipping(sample) ? std::size_t{1} : std::size_t{0};
    }

    SECTION("the car goes, the engine lives, and the clutch is what carried it")
    {
        REQUIRE(run.back().stalled == false);
        REQUIRE(run.back().roadSpeed > 8.0);
        REQUIRE(run.back().engineSpeed > driveline.engine.idleSpeed);

        // Slip is visible in telemetry rather than inferred: the channel is filled every tick and it
        // is what says a launch was slipped rather than dropped.
        // Measured: 190 rad/s of peak slip over 0.69 s. Bracketed rather than floored, because a
        // launch that slips ten times as long is as wrong as one that does not slip at all.
        REQUIRE(peakSlip > 50.0);
        REQUIRE(peakSlip < 400.0);
        REQUIRE(static_cast<double>(slippingTicks) * tick > 0.3);
        REQUIRE(static_cast<double>(slippingTicks) * tick < 2.0);
    }

    SECTION("and the slip energy accumulates, which is the hook the heat model will read")
    {
        // Measured: 6.47 kJ, which is what a 1350 kg car being got moving on a friction plate costs.
        REQUIRE(run.front().slipEnergy >= 0.0);
        REQUIRE(run.back().slipEnergy > 1000.0);
        REQUIRE(run.back().slipEnergy < 50000.0);

        // Monotonic by construction — it is a running total of a magnitude — and that is the
        // property anything integrating it later depends on.
        for (auto index = std::size_t{1}; index < run.size(); index++)
        {
            REQUIRE(run[index].slipEnergy >= run[index - 1].slipEnergy);
        }
    }

    SECTION("and the same launch twice is the same launch")
    {
        auto again = std::vector<Sample>{};
        launch(again);

        REQUIRE(again.size() == run.size());
        for (auto index = std::size_t{0}; index < run.size(); index++)
        {
            REQUIRE(again[index].engineSpeed == run[index].engineSpeed);
            REQUIRE(again[index].clutchTorque == run[index].clutchTorque);
            REQUIRE(again[index].slipEnergy == run[index].slipEnergy);
        }
    }

    SECTION("and it does not hunt between locked and slipping")
    {
        // The failure the coupling's two thresholds and two dwells exist to avoid, asserted on the
        // sequence rather than on where the run ended up: a buzz at the timestep's frequency is
        // invisible in a final state and obvious in the trace.
        REQUIRE(transitionsIn(run) <= std::size_t{4});
        REQUIRE(static_cast<double>(shortestRun(run)) * tick >= driveline.coupling.clutch.coupling.lockDwell);

        // And it did settle: a launch that never locks has not hunted either, and would pass the two
        // lines above while being just as wrong.
        REQUIRE(run.back().locked);
    }
}

TEST_CASE("the idle holds against a clutch bite", "[physics][clutch][idle]")
{
    const JoltGuard jolt;

    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    auto driveline = placeholderDriveline();
    driveline.autoClutch.enabled = false;

    auto car = Car{};
    settle(vehicle.value(), driveline, world.value(), car);

    const auto idle = driveline.engine.idleSpeed;
    REQUIRE(car.engine.engineSpeed == Catch::Approx(idle).epsilon(0.02));

    // Off the throttle, in gear, and the pedal brought just into the bite and held there — which is
    // the load a driver puts on an idle every time they pull away from a junction.
    auto run = std::vector<Sample>{};
    for (auto step = 0; step < 1800; step++)
    {
        const auto time = static_cast<double>(step) * tick;

        auto input = VehicleInput{};
        input.gear = 1;
        input.clutch = std::max(0.72, 1.0 - time / 0.5);

        run.push_back(advance(vehicle.value(), driveline, world.value(), car, input));
    }

    const auto lowest = std::min_element(run.begin(), run.end(), [](const Sample& left, const Sample& right)
                                         { return left.engineSpeed < right.engineSpeed; });
    REQUIRE(lowest != run.end());

    const auto dip = idle - lowest->engineSpeed;
    const auto recovered =
        std::find_if(lowest, run.end(), [idle](const Sample& sample) { return sample.engineSpeed >= idle - 2.0; });

    SECTION("it dips, because a controller that never moves is not answering the load")
    {
        // Measured: 5.24 rad/s, which is 50 rpm off an 850 rpm idle — what a real car does when a
        // clutch is let out at a junction.
        REQUIRE(car.engine.engine == EngineState::Running);
        REQUIRE(dip > 2.0);
    }

    SECTION("and it recovers, because one that keeps dipping stalls the car at every junction")
    {
        // Measured: back inside 2 rad/s of idle 0.156 s after the bottom of the dip.
        REQUIRE(dip < 30.0);
        REQUIRE(recovered != run.end());
        REQUIRE(static_cast<double>(std::distance(lowest, recovered)) * tick < 1.5);
    }

    SECTION("and the car is moving, so the load was real")
    {
        REQUIRE(run.back().roadSpeed > 0.5);
    }
}
