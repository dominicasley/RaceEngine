#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::advanceLockup;
using raceengine::bringUpJolt;
using raceengine::converterFlow;
using raceengine::ConverterLockup;
using raceengine::cornerCount;
using raceengine::CouplingMode;
using raceengine::CouplingSides;
using raceengine::DriveCoupling;
using raceengine::DriveCouplingKind;
using raceengine::DriveCouplingState;
using raceengine::DrivelineSetup;
using raceengine::DrivelineState;
using raceengine::EngineState;
using raceengine::generateProvingGround;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeholderAutomatic;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveCoupling;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TorqueConverter;
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

ProvingGroundDescriptor plate(const double size)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 4.0;
    descriptor.features = {};

    return descriptor;
}

// The torque the converter multiplied by, which is the one number that says it is a converter and
// not an expensive fluid coupling.
double multiplicationAt(const TorqueConverter& converter, const double speedRatio)
{
    const auto flow = converterFlow(converter, 200.0, 200.0 * speedRatio);

    return flow.impeller > 0.0 ? flow.turbine / flow.impeller : 1.0;
}

struct Car
{
    VehicleState state;
    DrivelineState engine;
    std::array<double, cornerCount> road{};
};

double advance(const VehicleSetup& vehicle, const DrivelineSetup& driveline, const PhysicsWorld& world, Car& car,
               const VehicleInput& input)
{
    const auto torques = stepDriveline(driveline, car.engine,
                                       {car.state.corners[0].wheelSpeed, car.state.corners[1].wheelSpeed,
                                        car.state.corners[2].wheelSpeed, car.state.corners[3].wheelSpeed},
                                       wheelInertias(vehicle), car.road, input, tick);
    REQUIRE(torques.has_value());

    auto stepped = stepVehicle(vehicle, car.state, input, torques->wheel, world, tick);
    REQUIRE(stepped.has_value());

    car.road = roadTorques(stepped.value());

    return car.state.chassis.linearVelocity.z;
}

void settle(const VehicleSetup& vehicle, const DrivelineSetup& driveline, const PhysicsWorld& world, Car& car)
{
    car = Car{};
    car.state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(vehicle, car.state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    startEngine(driveline, car.engine);

    for (auto step = 0; step < 720; step++)
    {
        static_cast<void>(advance(vehicle, driveline, world, car, VehicleInput{}));
    }
}

std::size_t transitionsIn(const std::vector<bool>& locked)
{
    auto count = std::size_t{0};

    for (auto index = std::size_t{1}; index < locked.size(); index++)
    {
        count += locked[index] != locked[index - 1] ? std::size_t{1} : std::size_t{0};
    }

    return count;
}

// The shortest stretch the mode was held for, ignoring the run's own two ends.
std::size_t shortestRun(const std::vector<bool>& locked)
{
    auto shortest = locked.size();
    auto length = std::size_t{0};
    auto seenTransition = false;

    for (auto index = std::size_t{1}; index < locked.size(); index++)
    {
        if (locked[index] == locked[index - 1])
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

TEST_CASE("a torque converter multiplies torque, which is what makes it one", "[physics][converter]")
{
    const auto converter = TorqueConverter{};

    SECTION("at stall it puts out more than it takes in")
    {
        // Held turbine, engine turning: the whole of the input becomes heat and the stator still
        // reacts more torque out than went in. A model that cannot show this is a lossy coupling
        // wearing a converter's name.
        const auto stalled = converterFlow(converter, 250.0, 0.0);

        REQUIRE(stalled.impeller > 0.0);
        REQUIRE(stalled.turbine > stalled.impeller);
        REQUIRE(stalled.turbine / stalled.impeller == Catch::Approx(2.10));

        // Impeller torque goes as the square of impeller speed at a fixed speed ratio, which is the
        // whole content of a capacity coefficient.
        const auto doubled = converterFlow(converter, 500.0, 0.0);
        REQUIRE(doubled.impeller == Catch::Approx(4.0 * stalled.impeller));
    }

    SECTION("and the multiplication decays to one at the coupling point and stays there")
    {
        auto previous = 2.11;

        for (auto step = 0; step <= 100; step++)
        {
            const auto ratio = multiplicationAt(converter, static_cast<double>(step) / 100.0);

            REQUIRE(ratio <= previous + 1e-12);
            REQUIRE(ratio >= 1.0 - 1e-12);
            previous = ratio;
        }

        REQUIRE(multiplicationAt(converter, 0.5) == Catch::Approx(1.485));
        REQUIRE(multiplicationAt(converter, 0.88) == Catch::Approx(1.0));
        // Past the coupling point the stator has freewheeled and there is nothing left to react.
        REQUIRE(multiplicationAt(converter, 0.95) == Catch::Approx(1.0));
    }

    SECTION("and the efficiency it reaches by the coupling point is what published data reports")
    {
        // Efficiency is the product of the two ratios, and this shape gives 0.90 by a speed ratio of
        // 0.8 — the figure quoted for a three-element converter. Above the coupling point it keeps
        // creeping toward one, because slip is the only loss in this model: churning and pumping
        // losses, which are what make a real curve turn over, are not here and would be a third
        // curve. There is almost no torque left up there to be efficient about, which is why the
        // omission costs nothing and why a lockup clutch exists anyway.
        REQUIRE(multiplicationAt(converter, 0.80) * 0.80 == Catch::Approx(0.904));
        REQUIRE(multiplicationAt(converter, 0.50) * 0.50 == Catch::Approx(0.7425));
        REQUIRE(multiplicationAt(converter, 0.20) * 0.20 == Catch::Approx(0.37));
    }
}

TEST_CASE("the converter's degenerate cases are decided rather than stumbled into", "[physics][converter]")
{
    const auto converter = TorqueConverter{};

    SECTION("nothing turning is no torque, and neither side has to be the divisor")
    {
        REQUIRE(converterFlow(converter, 0.0, 0.0).impeller == 0.0);
        REQUIRE(converterFlow(converter, 0.0, 0.0).turbine == 0.0);

        // An impeller at rest with a turbine turning is not a division by zero here: the ratio is
        // always formed over the faster side, so this is simply the overrun case.
        const auto dragged = converterFlow(converter, 0.0, 100.0);
        REQUIRE(dragged.impeller < 0.0);
        REQUIRE(dragged.turbine < 0.0);
    }

    SECTION("in step, there is no flow and therefore no torque")
    {
        const auto coupled = converterFlow(converter, 300.0, 300.0);

        REQUIRE(coupled.impeller == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(coupled.turbine == Catch::Approx(0.0).margin(1e-12));
    }

    SECTION("on overrun the turbine drives, and it multiplies nothing")
    {
        // The stator's one-way clutch has let go, so there is no reaction member: both sides see the
        // same torque. This is why an automatic gives so little engine braking.
        const auto overrun = converterFlow(converter, 200.0, 260.0);

        REQUIRE(overrun.turbine < 0.0);
        REQUIRE(overrun.impeller == Catch::Approx(overrun.turbine));

        // And there is no step across the crossing to be discontinuous about: the capacity is zero
        // where the two speeds meet, so both branches arrive there carrying nothing and the sign
        // change happens at zero torque rather than through one.
        REQUIRE(std::abs(converterFlow(converter, 200.0, 199.999).turbine) < 0.01);
        REQUIRE(std::abs(converterFlow(converter, 200.0, 200.001).turbine) < 0.01);
    }

    SECTION("a turbine turning backwards is worse than stall and is read as stall")
    {
        // A car rolling back down a hill in drive. The converter pushes it forward at the full
        // multiplication, which is what one does.
        const auto backwards = converterFlow(converter, 200.0, -30.0);
        const auto stalled = converterFlow(converter, 200.0, 0.0);

        REQUIRE(backwards.turbine == Catch::Approx(stalled.turbine));
        REQUIRE(backwards.turbine > 0.0);
    }
}

TEST_CASE("stall speed falls out of the curves rather than being set", "[physics][converter][stall]")
{
    // The turbine held at rest and the throttle open. Nothing in the setup names a stall speed: it
    // is where the engine's torque curve crosses the converter's capacity, and it moves if either
    // of them does.
    const auto setup = placeholderAutomatic();

    auto state = DrivelineState{};
    startEngine(setup, state);

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 1;

    auto previous = 0.0;
    for (auto step = 0; step < 3600; step++)
    {
        previous = state.engineSpeed;
        REQUIRE(stepDriveline(setup, state, {0.0, 0.0, 0.0, 0.0}, {1.2, 1.2, 1.2, 1.2}, noRoadTorque, input, tick)
                    .has_value());
    }

    const auto stallRpm = state.engineSpeed * radiansPerSecondToRpm;

    SECTION("and it settles somewhere a road automatic actually stalls")
    {
        // Measured: 2151 rpm, from a stall capacity factor of 140 rpm/sqrt(lb.ft) against this
        // engine's curve. A road automatic sits between about 1800 and 2500; outside that the
        // converter is too loose or too tight and the fix is the curve, not a number.
        REQUIRE(stallRpm > 1800.0);
        REQUIRE(stallRpm < 2500.0);
        REQUIRE(state.engineSpeed == Catch::Approx(previous).epsilon(1e-6));
    }

    SECTION("and a tighter converter stalls lower, which is what says the curve is the lever")
    {
        auto tighter = setup;
        for (auto& point : tighter.coupling.converter.capacity.points)
        {
            point.y *= 1.5;
        }

        auto tight = DrivelineState{};
        startEngine(tighter, tight);

        for (auto step = 0; step < 3600; step++)
        {
            REQUIRE(stepDriveline(tighter, tight, {0.0, 0.0, 0.0, 0.0}, {1.2, 1.2, 1.2, 1.2}, noRoadTorque, input, tick)
                        .has_value());
        }

        REQUIRE(tight.engineSpeed < state.engineSpeed);
    }
}

TEST_CASE("the clutch pedal is ignored by a converter, visibly", "[physics][converter]")
{
    // Not "has no effect because nothing reads it" — the field arrives at the slot, the converter
    // case reads it and drops it, and the proof is that two runs differing in nothing but the
    // driver's left foot are identical to the bit.
    const auto setup = placeholderAutomatic();

    const auto drive = [&setup](const double pedal)
    {
        auto state = DrivelineState{};
        startEngine(setup, state);

        auto input = VehicleInput{};
        input.throttle = 0.6;
        input.gear = 1;
        input.clutch = pedal;

        auto torque = 0.0;
        for (auto step = 0; step < 720; step++)
        {
            const auto torques =
                stepDriveline(setup, state, {12.0, 12.0, 12.0, 12.0}, {1.2, 1.2, 1.2, 1.2}, noRoadTorque, input, tick);
            REQUIRE(torques.has_value());
            torque = torques->wheel[0];
        }

        return std::pair{state.engineSpeed, torque};
    };

    const auto out = drive(0.0);
    const auto floored = drive(1.0);

    REQUIRE(floored.first == out.first);
    REQUIRE(floored.second == out.second);
    REQUIRE(out.second > 0.0);

    SECTION("and the same two runs on a friction clutch are not the same run, so that is not a tautology")
    {
        auto manual = setup;
        manual.coupling.kind = DriveCouplingKind::FrictionClutch;
        manual.autoClutch.enabled = false;

        const auto driveManual = [&manual](const double pedal)
        {
            auto state = DrivelineState{};
            startEngine(manual, state);

            auto input = VehicleInput{};
            input.throttle = 0.6;
            input.gear = 1;
            input.clutch = pedal;

            auto torque = 0.0;
            for (auto step = 0; step < 720; step++)
            {
                const auto torques = stepDriveline(manual, state, {12.0, 12.0, 12.0, 12.0}, {1.2, 1.2, 1.2, 1.2},
                                                   noRoadTorque, input, tick);
                REQUIRE(torques.has_value());
                torque = torques->wheel[0];
            }

            return torque;
        };

        REQUIRE(driveManual(0.0) != driveManual(1.0));
        REQUIRE(driveManual(1.0) == 0.0);
    }
}

TEST_CASE("the lockup clutch engages and lets go without hunting", "[physics][converter][lockup]")
{
    // The third consumer of the one coupling state machine, and the case it was written for: this is
    // a friction clutch with a hydraulic apply in front of it rather than a pedal.
    auto coupling = DriveCoupling{};
    coupling.kind = DriveCouplingKind::TorqueConverter;

    const auto& lockup = coupling.converter.lockup;

    // A cruising driveline: three per cent of slip across the converter, the engine pulling and the
    // road resisting, and inertias referred as `stepDriveline` refers them in a high gear.
    const auto cruising = [](const double turbineSpeed)
    {
        return CouplingSides{.drivingSpeed = 1.03 * turbineSpeed,
                             .drivenSpeed = turbineSpeed,
                             .drivingInertia = 0.15,
                             .drivenInertia = 0.174,
                             .drivingTorque = 150.0,
                             .drivenTorque = -150.0,
                             .capacity = 0.0};
    };

    SECTION("it locks above its engage speed and releases below its release speed")
    {
        auto state = DriveCouplingState{};

        for (auto step = 0; step < 720; step++)
        {
            const auto solved =
                stepDriveCoupling(coupling, state, cruising(220.0), {.clutchPedal = 0.0, .gear = 4}, tick);
            REQUIRE(solved.has_value());
        }

        REQUIRE(state.lockupApply == Catch::Approx(1.0));

        const auto locked = stepDriveCoupling(coupling, state, cruising(220.0), {.clutchPedal = 0.0, .gear = 4}, tick);
        REQUIRE(locked.has_value());
        REQUIRE(locked->locked);
        // Locked, the fluid is carrying nothing worth naming and the plate has the lot.
        REQUIRE(std::abs(locked->drivenTorque - locked->drivingTorque) < 1.0);

        for (auto step = 0; step < 720; step++)
        {
            REQUIRE(
                stepDriveCoupling(coupling, state, cruising(100.0), {.clutchPedal = 0.0, .gear = 4}, tick).has_value());
        }

        REQUIRE(state.lockupApply == 0.0);
        const auto released =
            stepDriveCoupling(coupling, state, cruising(100.0), {.clutchPedal = 0.0, .gear = 4}, tick);
        REQUIRE(released.has_value());
        REQUIRE_FALSE(released->locked);
    }

    SECTION("a gear below its threshold keeps it out however fast the car is going")
    {
        auto state = DriveCouplingState{};

        for (auto step = 0; step < 720; step++)
        {
            REQUIRE(
                stepDriveCoupling(coupling, state, cruising(300.0), {.clutchPedal = 0.0, .gear = 1}, tick).has_value());
        }

        REQUIRE(state.lockupApply == 0.0);
    }

    SECTION("and a speed crossing the band over and over locks once per crossing")
    {
        auto state = DriveCouplingState{};
        auto locked = std::vector<bool>{};

        // A triangle between 100 and 240 rad/s with a two-second period, so it crosses the whole
        // band ten times each way over twenty seconds.
        for (auto step = 0; step < 7200; step++)
        {
            const auto phase = std::abs(std::fmod(static_cast<double>(step) * tick, 2.0) - 1.0);
            const auto solved = stepDriveCoupling(coupling, state, cruising(100.0 + 140.0 * phase),
                                                  {.clutchPedal = 0.0, .gear = 4}, tick);
            REQUIRE(solved.has_value());
            locked.push_back(solved->locked);
        }

        const auto changes = transitionsIn(locked);

        // Twenty crossings and one more because the run opens at the top of the triangle already
        // above the engage speed. Measured: 21. Anything past that is the plate buzzing somewhere
        // inside a crossing, which is the failure the whole arrangement exists to stop and is
        // invisible in any final state.
        REQUIRE(changes >= std::size_t{18});
        REQUIRE(changes <= std::size_t{21});

        // And it stays where it goes. Measured: 0.469 s, which is half a second rather than the
        // coupling's own 0.02 s dwell scraped past.
        REQUIRE(static_cast<double>(shortestRun(locked)) * tick > 0.3);
    }

    SECTION("and a speed dithering on the threshold every tick does not chatter")
    {
        // Two thresholds with the apply level held between them, which is what makes that level its
        // own latch. One threshold and this is a plate banging in and out at the tick rate.
        auto state = DriveCouplingState{};
        auto locked = std::vector<bool>{};

        for (auto step = 0; step < 3600; step++)
        {
            const auto speed = lockup.engageSpeed + (step % 2 == 0 ? 1.0 : -1.0);
            const auto solved =
                stepDriveCoupling(coupling, state, cruising(speed), {.clutchPedal = 0.0, .gear = 4}, tick);
            REQUIRE(solved.has_value());
            locked.push_back(solved->locked);
        }

        REQUIRE(transitionsIn(locked) == std::size_t{1});
        REQUIRE(locked.back());
    }
}

TEST_CASE("the lockup's apply level is its own latch", "[physics][converter][lockup]")
{
    const auto lockup = ConverterLockup{};

    SECTION("above the engage speed it builds, below the release speed it bleeds")
    {
        REQUIRE(advanceLockup(lockup, 0.0, 200.0, 4, tick) > 0.0);
        REQUIRE(advanceLockup(lockup, 1.0, 100.0, 4, tick) < 1.0);
    }

    SECTION("and between the two it is simply held, which is the whole of the hysteresis")
    {
        REQUIRE(advanceLockup(lockup, 0.4, 160.0, 4, tick) == 0.4);
        REQUIRE(advanceLockup(lockup, 0.0, 160.0, 4, tick) == 0.0);
        REQUIRE(advanceLockup(lockup, 1.0, 160.0, 4, tick) == 1.0);
    }

    SECTION("a gear below the threshold releases it wherever the speed is")
    {
        REQUIRE(advanceLockup(lockup, 1.0, 300.0, 2, tick) < 1.0);
        REQUIRE(advanceLockup(lockup, 1.0, 300.0, -1, tick) < 1.0);
    }

    SECTION("and switched off it never applies at all")
    {
        auto disabled = lockup;
        disabled.enabled = false;

        REQUIRE(advanceLockup(disabled, 0.0, 300.0, 6, tick) == 0.0);
    }
}

TEST_CASE("a converter's slip energy is the power it did not pass on", "[physics][converter][slip]")
{
    // Not torque times slip, which is what a plate dissipates. A converter's stator is grounded and
    // does no work, so the heat is the power in less the power out — and those two forms agree only
    // where the torque ratio is one. At stall they agree because nothing comes out; in between the
    // plate's form overstates it by the multiplication.
    const auto setup = placeholderAutomatic();

    auto state = DrivelineState{};
    startEngine(setup, state);

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 1;

    auto previous = 0.0;
    auto lastSlip = 0.0;
    auto lastTorque = 0.0;

    for (auto step = 0; step < 3600; step++)
    {
        const auto torques =
            stepDriveline(setup, state, {0.0, 0.0, 0.0, 0.0}, {1.2, 1.2, 1.2, 1.2}, noRoadTorque, input, tick);
        REQUIRE(torques.has_value());

        REQUIRE(torques->slipEnergy >= previous);
        previous = torques->slipEnergy;
        lastSlip = torques->clutchSlip;
        lastTorque = torques->clutchReaction;
    }

    SECTION("and away from stall the plate's arithmetic would overstate it by twice")
    {
        // The brief for this element asks for torque times slip, which is a plate's loss and is
        // wrong here by the amount the stator carries. At half a speed ratio the converter is
        // passing 42.96 kW out of 57.86 kW in, so 14.90 kW is heat — where the impeller torque
        // across the slip speed reads 28.93 kW and the turbine torque across it reads 42.96. Both
        // of those count work the turbine actually did as though it had been dissipated.
        const auto flow = converterFlow(TorqueConverter{}, 225.0, 112.5);
        const auto heat = flow.impeller * 225.0 - flow.turbine * 112.5;

        REQUIRE(heat == Catch::Approx(14900.0).epsilon(0.001));
        REQUIRE(std::abs(flow.impeller * 112.5) / heat == Catch::Approx(1.942).epsilon(0.001));
    }

    SECTION("stalled against the brakes it is the whole of what the engine is making")
    {
        // Everything the impeller absorbs becomes heat, because the turbine is not turning and
        // therefore takes no power out.
        REQUIRE(previous > 100000.0);
        REQUIRE(lastTorque * lastSlip == Catch::Approx(previous / 10.0).epsilon(0.02));
    }

    SECTION("and the reaction is the smaller of the two torques, which is where the difference goes")
    {
        const auto torques =
            stepDriveline(setup, state, {0.0, 0.0, 0.0, 0.0}, {1.2, 1.2, 1.2, 1.2}, noRoadTorque, input, tick);
        REQUIRE(torques.has_value());

        REQUIRE(torques->clutch > torques->clutchReaction);
        REQUIRE(torques->clutch / torques->clutchReaction == Catch::Approx(2.10));
    }
}

TEST_CASE("an automatic creeps at idle, and nothing scripted it", "[physics][converter][creep]")
{
    // In gear, off the throttle, off the brakes. The car crawls because a stalled converter at idle
    // still multiplies whatever the idle governor is holding the engine on, and that is all: there
    // is no creep torque, no launch assist and no minimum speed anywhere in the model.
    const JoltGuard jolt;

    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate(600.0)).value());
    REQUIRE(world.has_value());

    const auto driveline = placeholderAutomatic();

    auto car = Car{};
    settle(vehicle.value(), driveline, world.value(), car);

    REQUIRE(car.engine.engineSpeed == Catch::Approx(driveline.engine.idleSpeed).epsilon(0.05));
    REQUIRE(std::abs(car.state.chassis.linearVelocity.z) < 0.05);

    auto input = VehicleInput{};
    input.gear = 1;

    auto speed = 0.0;
    for (auto step = 0; step < 3600; step++)
    {
        speed = advance(vehicle.value(), driveline, world.value(), car, input);
    }

    SECTION("it crawls, at a speed a car park would recognise")
    {
        // Measured: 1.91 m/s, 6.9 km/h. It is the balance of the converter's output at an 850 rpm
        // idle against rolling resistance and drag — move any one of the three and this moves.
        REQUIRE(speed > 0.8);
        REQUIRE(speed < 4.0);
        REQUIRE(car.engine.engineSpeed > driveline.engine.stallSpeed);
    }

    SECTION("and the brakes hold it without the engine dying, which a plate could not do")
    {
        auto braked = input;
        braked.brake = 1.0;

        for (auto step = 0; step < 1800; step++)
        {
            speed = advance(vehicle.value(), driveline, world.value(), car, braked);
        }

        REQUIRE(std::abs(speed) < 0.05);
        REQUIRE(car.engine.engineSpeed > driveline.engine.stallSpeed);
    }
}

TEST_CASE("an automatic drives away and locks up", "[physics][converter][integration]")
{
    const JoltGuard jolt;

    const auto vehicle = placeholderSedan();
    REQUIRE(vehicle.has_value());
    const auto world = PhysicsWorld::create(generateProvingGround(plate(1600.0)).value());
    REQUIRE(world.has_value());

    const auto driveline = placeholderAutomatic();

    auto car = Car{};
    settle(vehicle.value(), driveline, world.value(), car);

    // Up through the gears by hand — a shift schedule is somebody else's milestone — and into the
    // gear the lockup is allowed to work in.
    auto speed = 0.0;
    for (auto step = 0; step < 5400; step++)
    {
        auto input = VehicleInput{};
        input.throttle = 1.0;
        input.gear = std::min(1 + step / 900, 4);

        speed = advance(vehicle.value(), driveline, world.value(), car, input);
    }

    // Measured: 162.6 km/h in fourth at 5568 rpm, with the plate closed since 5.32 s and 33.6 kJ of
    // heat in the oil from the launch.
    REQUIRE(speed > 20.0);
    REQUIRE(car.engine.engine == EngineState::Running);
    REQUIRE(car.engine.coupling.lockupApply == Catch::Approx(1.0));
    REQUIRE(car.engine.coupling.coupling.mode == CouplingMode::Locked);
    REQUIRE(car.engine.slipEnergy > 1000.0);
}
