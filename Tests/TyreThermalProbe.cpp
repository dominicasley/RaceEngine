// What the thermal tyre actually does. `./EngineTests "[.tyre-thermal]"`.
//
// **A probe and not a gate**: no acceptance threshold on any finding, every `REQUIRE` a fixture
// precondition. `diagnostic-not-a-calibration-target` is Dominic's own correction and it applies
// here more than anywhere — the one fitted number in this model is set from what this file prints,
// so a threshold in it would be a model fitted to its own instrument.
//
// It answers four questions, in order of how much they matter:
//
//   1. **What does the tread warm up to, and how fast?** That is what `frictionToTread` is fitted
//      against — a plausible warm-up, meaning a lap or two of ordinary driving to the plateau and no
//      running away in one corner — and the sweep is here so the fit can be read rather than
//      asserted.
//   2. **What does a cold tyre cost?** The skidpad, the anti-lock stop and the launch, each run cold
//      and hot against the same car with the model switched off. The braking one is the point of the
//      whole exercise: the verified reference is auto motor und sport's 35.5 m and it is explicitly
//      *kalt*, so until now the comparison was never like-for-like.
//   3. **Where does the heat go?** The conductances, printed, because the finding that the road is
//      the dominant path at speed is not obvious and decides what the model is sensitive to.
//   4. **Is the tyre where the sources say it should be?** 70-100 °C, Michelin, for a track tyre on
//      track. A road car on a cool morning has no business being in that window and this says so.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::ambientAt;
using raceengine::AmbientConditions;
using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::BrakeCommand;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::CornerSide;
using raceengine::DrivelineState;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::seedTyreTemperatures;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TractionMode;
using raceengine::tyreDefaultTemperature;
using raceengine::psiPerPascal;
using raceengine::TyreState;
using raceengine::seedTyreGasPressures;
using raceengine::tyreGasTemperatureAtIdealPressure;
using raceengine::Corner;
using raceengine::rearAxle;
using raceengine::tyrePressureAt;
using raceengine::tyrePressureGripScale;
using raceengine::tyrePressureRollingResistanceScale;
using raceengine::tyrePressureVerticalRateScale;
using raceengine::tyreThermalNodes;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto rideHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto gravity = 9.80665;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto startZ = 40.0;

constexpr auto noBrakePressure = std::array<double, cornerCount>{};

// **The weather every measurement here is taken in, stated once.** The circuit scene is set at half
// past four in the afternoon — a 19-degree sun — so 20 °C of air puts the tarmac at 31.5. That is an
// ordinary temperate afternoon and it is deliberately not a cold morning: what is being measured is
// what a tyre does on a normal day, and a colder one only moves every number the same way.
[[nodiscard]] AmbientConditions weather()
{
    return ambientAt(20.0, 19.0);
}

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

[[nodiscard]] PhysicsWorld plate(const double length, const double width, const double cell)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = cell;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    return std::move(world.value());
}

[[nodiscard]] double spawnHeight(const VehicleSetup& setup)
{
    auto highest = 0.0;
    for (const auto& corner : setup.corners)
    {
        highest = std::max(highest, corner.hardpoints.wheelCentre.y + corner.hardpoints.wheelRadius);
    }

    return highest;
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double z)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, rideHeight, z);

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

// The tyre's own thermal state, summarised across the four corners. The outside front is what a
// driver feels and what the criteria are about, so it is reported beside the mean.
struct Temperatures
{
    double surface = 0.0;
    double core = 0.0;
    double carcass = 0.0;
    double hottestCore = 0.0;
    // The fourth node, since stage 2. Read here rather than in a struct of its own because every
    // question about the air is a question about where it sits relative to the rubber.
    double gas = 0.0;

    [[nodiscard]] static Temperatures of(const VehicleState& state)
    {
        auto reading = Temperatures{};
        for (const auto& corner : state.corners)
        {
            reading.surface += 0.25 * corner.tyre.surfaceTemperature;
            reading.core += 0.25 * corner.tyre.coreTemperature;
            reading.carcass += 0.25 * corner.tyre.carcassTemperature;
            reading.gas += 0.25 * corner.tyre.gasTemperature;
            reading.hottestCore = std::max(reading.hottestCore, corner.tyre.coreTemperature);
        }

        return reading;
    }
};

// A car held at a speed and a steering angle by torque on its driven axle, which is the fixture
// `GolfGtiTests` holds a skidpad with. Proportional plus integral, because a constant drag needs a
// constant force and a P term alone leaves a standing error — this fixture held 18.4 of an asked-for
// 20 before the integral existed.
struct Held
{
    Temperatures temperatures;
    double lateralAcceleration = 0.0;
    double speed = 0.0;
    // Mean dissipation at one patch over the sample window, watts — the heaviest-loaded corner. It
    // is printed because the heat balance is unreadable without it: what separates a cruise from a
    // corner is not the temperature, it is this.
    double slipPower = 0.0;
    std::vector<std::pair<double, Temperatures>> trace;
};

[[nodiscard]] Held hold(const VehicleSetup& setup, const PhysicsWorld& world, const double steering, const double speed,
                        const double seconds, const double seedTemperature, const std::vector<double>& marks)
{
    auto state = VehicleState{};
    settle(setup, state, world, speed, 600.0);
    seedTyreTemperatures(state, seedTemperature);

    auto input = VehicleInput{};
    input.steering = steering;

    auto integral = 0.0;
    auto result = Held{};

    const auto steps = static_cast<int>(seconds / tick);
    const auto window = std::max(steps - 360, 0);
    auto samples = 0;
    auto slowest = std::numeric_limits<double>::max();
    auto fewestSupported = cornerCount;
    auto mark = std::size_t{0};

    for (auto step = 0; step < steps; step++)
    {
        const auto error = speed - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) * tyreRadius / 2.0;
        const auto drive = std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0};

        const auto stepped = stepVehicle(setup, state, input, drive, world, tick, {}, weather());
        REQUIRE(stepped.has_value());

        const auto now = static_cast<double>(step + 1) * tick;
        if (mark < marks.size() && now >= marks[mark])
        {
            result.trace.emplace_back(marks[mark], Temperatures::of(state));
            mark++;
        }

        if (step >= window)
        {
            constexpr auto toTheRight = outboardSign(CornerSide::Right);

            result.lateralAcceleration += stepped->telemetry.acceleration.x * toTheRight;
            result.speed += glm::length(state.chassis.linearVelocity);

            auto hardest = 0.0;
            for (const auto& solution : stepped->corners)
            {
                hardest = std::max(hardest, solution.contact.tyre.slipPower);
            }
            result.slipPower += hardest;

            samples++;
        }

        slowest = std::min(slowest, glm::length(state.chassis.linearVelocity));

        auto supported = std::size_t{0};
        for (const auto& wheel : stepped->telemetry.wheels)
        {
            supported += wheel.inContact ? 1 : 0;
        }
        fewestSupported = std::min(fewestSupported, supported);
    }

    // The fixture asserts that it did what its name says. A hold that lost its speed is a spiral and
    // every number off it describes a transient rather than a steady state.
    //
    // **And that it was still on a road**, which is the one this fixture was caught by. A four-minute
    // warm-up at 100 km/h covers seven kilometres, and the first plate it was given was 1400 m long:
    // the car ran off the end of the generated mesh at 36 seconds, lost every contact patch, and the
    // tyres then cooled towards the air with nothing to heat them. It read as a tyre that *cools*
    // while being driven, which is a plausible-looking wrong answer of exactly the kind this
    // project's own rule about fixture preconditions exists to catch.
    CAPTURE(steering, speed, slowest, fewestSupported);
    REQUIRE(slowest > 0.85 * speed);
    REQUIRE(fewestSupported >= 3);
    REQUIRE(samples > 0);

    result.lateralAcceleration /= static_cast<double>(samples) * gravity;
    result.speed /= static_cast<double>(samples);
    result.slipPower /= static_cast<double>(samples);
    result.temperatures = Temperatures::of(state);

    return result;
}

// --- braking, the anti-lock stop, from a stated temperature ---

struct Stop
{
    double distance = 0.0;
    double time = 0.0;
    Temperatures entry;
    Temperatures exit;

    [[nodiscard]] double meanDeceleration() const
    {
        return time > 0.0 ? hundred / time / gravity : 0.0;
    }
};

[[nodiscard]] Stop brake(const VehicleSetup& setup, const PhysicsWorld& world, const double seedTemperature)
{
    auto state = VehicleState{};
    settle(setup, state, world, hundred, 20.0);
    seedTyreTemperatures(state, seedTemperature);

    auto assists = golfGtiMk7Assists(setup);
    assists.antilock.enabled = true;
    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

        return sensors;
    };

    // Half a second of rolling to settle the suspension into the entry speed, and short enough that
    // the seeded temperature is still the temperature the stop starts at.
    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped =
            stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes, weather());
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    REQUIRE(std::abs(state.chassis.linearVelocity.z - hundred) < 0.5);

    auto result = Stop{};
    result.entry = Temperatures::of(state);

    const auto start = state.chassis.position.z;

    auto input = VehicleInput{};
    input.brake = 1.0;

    for (auto step = 0; step < 360 * 20; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                           brakeCircuitPressures(setup, 1.0), tick);
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes, weather());
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        result.time += tick;

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            break;
        }
    }

    result.distance = state.chassis.position.z - start;
    result.exit = Temperatures::of(state);

    return result;
}

// --- the launch, traction control in sport, from a stated temperature ---

struct Launch
{
    double toHundred = -1.0;
    Temperatures exit;
};

[[nodiscard]] Launch accelerate(const VehicleSetup& setup, const PhysicsWorld& world, const double seedTemperature)
{
    auto driveline = golfGtiMk7Driveline();
    auto assists = golfGtiMk7Assists(setup);
    assists.traction.mode = TractionMode::Sport;
    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    const auto inertias = wheelInertias(setup);

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, spawnHeight(setup), startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    auto road = std::array<double, cornerCount>{};
    const auto speeds = [&]
    {
        return std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                               state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};
    };

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;

        return sensors;
    };

    {
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto held = 0; held < 360; held++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup, 1.0), tick);
            const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped =
                stepVehicle(setup, state, idling, torques->wheel, world, tick, command.brakes, weather());
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());
            lastStep = stepped.value();
        }
    }

    // Seeded after the idle hold, so the temperature the launch starts at is the temperature this
    // asked for rather than whatever a second of idling left.
    seedTyreTemperatures(state, seedTemperature);

    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.2);
    REQUIRE(drivelineState.gear == 1);

    auto result = Launch{};
    auto gear = 1;
    const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

    for (auto step = 1; step <= 360 * 30; step++)
    {
        const auto now = static_cast<double>(step) * tick;

        const auto roadSideSpeed =
            std::abs(state.chassis.linearVelocity.z) / tyreRadius * driveline.gearbox.reduction(gear);
        if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
        {
            gear++;
        }

        const auto command = updateAssists(assists, assistState, sense(), {.brake = 0.0, .throttle = 1.0},
                                           brakeCircuitPressures(setup, 0.0), tick);

        auto input = VehicleInput{};
        input.throttle = command.throttleScale;
        input.gear = gear;

        const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick, command.brakes, weather());
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());
        lastStep = stepped.value();

        if (result.toHundred < 0.0 && state.chassis.linearVelocity.z >= hundred)
        {
            result.toHundred = now;
            break;
        }
    }

    REQUIRE(result.toHundred > 0.0);
    result.exit = Temperatures::of(state);

    return result;
}

[[nodiscard]] VehicleSetup golfWith(const bool thermal, const double frictionToTread)
{
    auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto setup = built.value();
    setup.tyreThermal = thermal;

    if (frictionToTread >= 0.0)
    {
        for (auto& corner : setup.corners)
        {
            corner.tyre.thermal.frictionToTread = frictionToTread;
        }
    }

    return setup;
}

// The same car with a stated tread-road contact conductance, W/(m2.K). Zero is perfect contact,
// which is the control; the shipped tyre states the measured 25200 since 2026-08-29.
[[nodiscard]] VehicleSetup golfWithContact(const double conductance)
{
    auto setup = golfWith(true, -1.0);

    for (auto& corner : setup.corners)
    {
        corner.tyre.thermal.roadContactConductance = conductance;
    }

    return setup;
}

// And with a stated road-conducting fraction of the patch. One is the gross patch and is what the
// shipped tyre states; 0.72 is `1 - voidFraction` and is what this tread's grooves leave touching.
[[nodiscard]] VehicleSetup golfWithArea(const double fraction)
{
    auto setup = golfWith(true, -1.0);

    for (auto& corner : setup.corners)
    {
        corner.tyre.thermal.roadAreaFraction = fraction;
    }

    return setup;
}

} // namespace

TEST_CASE("where the tread's heat goes, and how much of it there is to move", "[.tyre-thermal]")
{
    const auto setup = golfWith(true, -1.0);
    const auto& thermal = setup.corners.front().tyre.thermal;
    const auto nodes = tyreThermalNodes(thermal);

    std::printf("\n  The tread's nodes, out of a 225/40 R18's own size and a published tyre mass.\n\n");
    std::printf("    tread band          %.4f m2       sidewalls   %.4f m2\n", nodes.treadArea, nodes.sidewallArea);
    std::printf("    tread mass          %6.3f kg       carcass     %6.3f kg   (published tyre %.2f kg)\n",
                nodes.treadMass, nodes.carcassMass, thermal.tyreMass);
    std::printf("    capacity  surface   %8.1f J/K   core %8.1f J/K   carcass %9.1f J/K\n", nodes.surfaceCapacity,
                nodes.coreCapacity, nodes.carcassCapacity);
    std::printf("    conduction  surface-core %6.2f W/K    core-carcass %6.2f W/K\n", nodes.surfaceToCore,
                nodes.coreToCarcass);
    std::printf("\n    volumetric heat capacity  %.3e J/(m3.K)     diffusivity  %.3e m2/s\n",
                thermal.density * thermal.specificHeat,
                thermal.conductivity / (thermal.density * thermal.specificHeat));
    std::printf("    frictionToTread %.4f, which is the effusivity partition 661/(661+%.0f) and not a fit\n\n",
                thermal.frictionToTread, thermal.roadEffusivity);

    // And the two paths out, measured on a rolling car rather than reasoned about — which is the
    // only way to see that the **road** is the dominant one at speed. Both are computed here exactly
    // as `stepTyreThermal` computes them, from the patch the tick actually produced.
    const auto guard = JoltGuard{};
    const auto world = plate(600.0, 60.0, 4.0);

    auto state = VehicleState{};
    settle(setup, state, world, hundred, 20.0);

    auto last = VehicleStep{};
    for (auto step = 0; step < 180; step++)
    {
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, {}, weather());
        REQUIRE(stepped.has_value());
        last = stepped.value();
    }

    const auto& front = last.corners.front();
    const auto penetration = front.patch.penetration;
    const auto chord = 2.0 * std::sqrt(std::max(2.0 * tyreRadius * penetration - penetration * penetration, 0.0));
    const auto patchArea = chord * setup.sampling.width;
    const auto residence = std::clamp(chord / std::max(std::abs(front.contact.longitudinalVelocity), 1e-3), 1e-3, 1.0);
    const auto effusivity = std::sqrt(thermal.conductivity * thermal.density * thermal.specificHeat);
    const auto perfect = 2.0 * effusivity / std::sqrt(3.141592653589793 * residence) * thermal.roadEffusivity /
                         (effusivity + thermal.roadEffusivity);

    // The car states an interface conductance since 2026-08-29, so the road path printed here is the
    // one `stepTyreThermal` actually uses: the series composition, times the conducting fraction.
    const auto contact = thermal.roadContactConductance > 0.0
                             ? 1.0 / (1.0 / perfect + 1.0 / thermal.roadContactConductance)
                             : perfect;
    const auto roadPath = contact * patchArea * thermal.roadAreaFraction;

    std::printf("  Rolling at 100 km/h, front left, off the tick's own patch.\n\n");
    std::printf("    penetration %.4f m   chord %.4f m   patch %.4f m2   residence %.4f s\n", penetration, chord,
                patchArea, residence);
    std::printf("    tread effusivity %.0f J/(m2.K.s^0.5)   contact coefficient %.0f W/(m2.K) perfect, %.0f stated\n",
                effusivity, perfect, contact);
    std::printf("    surface -> road  %7.2f W/K\n", roadPath);
    std::printf("    surface -> core  %7.2f W/K\n", nodes.surfaceToCore);
    std::printf("    surface time constant, every path together  %.2f s\n\n",
                nodes.surfaceCapacity / (roadPath + nodes.surfaceToCore));
}

TEST_CASE("what the tread warms up to, driven, and what the fitted number is worth", "[.tyre-thermal]")
{
    const auto guard = JoltGuard{};

    // Two plates, because the two fixtures need different ground. Four minutes at 100 km/h is seven
    // kilometres of straight; a skidpad at 20 m/s and about 0.9 g circles inside a 45 m radius and
    // needs width instead.
    const auto straightWorld = plate(8000.0, 100.0, 8.0);
    const auto circleWorld = plate(1400.0, 900.0, 8.0);
    const auto conditions = weather();

    const auto marks = std::vector<double>{15.0, 30.0, 60.0, 120.0, 240.0};

    std::printf("\n  Air %.1f C, track %.1f C. Tyres seeded at the track's temperature and driven.\n",
                conditions.airTemperature, conditions.trackTemperature);
    std::printf("  Core temperature, degrees Celsius, mean of four corners.\n");
    std::printf("  The plateau this compound wants is 55-75 C, slid 20 C down from AC's track window: a\n"
                "  summer road tyre works near 50 C (Persson & Xu, arXiv:2507.18782v3) where a racing\n"
                "  tyre wants ~100.\n\n");

    for (const auto share : {0.15, 0.2956, 0.50, 0.70})
    {
        const auto setup = golfWith(true, share);

        // A straight-line cruise at 100 km/h. Nothing slides, so this is the strain-energy term
        // alone — rolling resistance, which is why a tyre warms on a straight — and the fitted share
        // does not enter it at all. That is the control on the corner below.
        const auto straight = hold(setup, straightWorld, 0.0, hundred, 260.0, conditions.trackTemperature, marks);

        // And a steady corner near this car's own peak, where the patch is sliding and the fitted
        // share is most of the heat.
        const auto cornering = hold(setup, circleWorld, 0.37, 20.0, 260.0, conditions.trackTemperature, marks);

        // **And a fast corner, because a slow one is not what a lap is made of.** Slip power at a
        // given lateral acceleration goes with speed, so a 72 km/h skidpad understates a circuit by
        // the ratio of the two speeds. This is the same car leaning on the same tyre at 144 km/h,
        // which is an ordinary Bathurst corner.
        const auto fast = hold(setup, circleWorld, 0.20, 40.0, 260.0, conditions.trackTemperature, marks);

        std::printf("    frictionToTread %.2f\n", share);
        std::printf("      straight 100 km/h ");
        for (const auto& [when, reading] : straight.trace)
        {
            std::printf(" %5.0fs %5.1f", when, reading.core);
        }
        std::printf("   surface %5.1f  carcass %5.1f  patch %6.0f W\n", straight.temperatures.surface,
                    straight.temperatures.carcass, straight.slipPower);

        std::printf("      corner 20 m/s %.2fg", cornering.lateralAcceleration);
        for (const auto& [when, reading] : cornering.trace)
        {
            std::printf(" %5.0fs %5.1f", when, reading.core);
        }
        std::printf("   surface %5.1f  hottest core %5.1f  patch %6.0f W\n", cornering.temperatures.surface,
                    cornering.temperatures.hottestCore, cornering.slipPower);

        std::printf("      corner 40 m/s %.2fg", fast.lateralAcceleration);
        for (const auto& [when, reading] : fast.trace)
        {
            std::printf(" %5.0fs %5.1f", when, reading.core);
        }
        std::printf("   surface %5.1f  hottest core %5.1f  patch %6.0f W\n", fast.temperatures.surface,
                    fast.temperatures.hottestCore, fast.slipPower);
    }

    std::printf("\n");
}

TEST_CASE("what a cold tyre costs, against the same car with the model switched off", "[.tyre-thermal]")
{
    const auto guard = JoltGuard{};
    const auto conditions = weather();

    const auto off = golfWith(false, -1.0);
    const auto on = golfWith(true, -1.0);

    std::printf("\n  Air %.1f C, track %.1f C.\n\n", conditions.airTemperature, conditions.trackTemperature);

    SECTION("the anti-lock stop from 100 km/h, which is the kalt comparison")
    {
        const auto world = plate(600.0, 60.0, 2.0);

        const auto control = brake(off, world, tyreDefaultTemperature);
        const auto hot = brake(on, world, tyreDefaultTemperature);
        const auto warm = brake(on, world, 60.0);
        const auto cold = brake(on, world, conditions.trackTemperature);

        std::printf("    100-0 with ABS, against a verified 35.5 m *kalt* (auto motor und sport)\n");
        std::printf("      model off, always at its best     %6.2f m   %.3f g\n", control.distance,
                    control.meanDeceleration());
        std::printf("      thermal, on the plateau           %6.2f m   %.3f g   core %5.1f -> %5.1f\n", hot.distance,
                    hot.meanDeceleration(), hot.entry.core, hot.exit.core);
        std::printf("      thermal, seeded 60 C              %6.2f m   %.3f g   core %5.1f -> %5.1f\n", warm.distance,
                    warm.meanDeceleration(), warm.entry.core, warm.exit.core);
        std::printf("      thermal, seeded at track temp     %6.2f m   %.3f g   core %5.1f -> %5.1f\n", cold.distance,
                    cold.meanDeceleration(), cold.entry.core, cold.exit.core);
        std::printf("      cold against the reference        %+6.1f%%\n\n", 100.0 * (cold.distance - 35.5) / 35.5);
    }

    SECTION("the skidpad peak, which is what the criterion band is about")
    {
        const auto world = plate(900.0, 900.0, 8.0);
        const auto marks = std::vector<double>{};

        const auto control = hold(off, world, 0.37, 20.0, 12.0, tyreDefaultTemperature, marks);
        const auto hot = hold(on, world, 0.37, 20.0, 12.0, tyreDefaultTemperature, marks);
        const auto cold = hold(on, world, 0.37, 20.0, 12.0, conditions.trackTemperature, marks);

        std::printf("    skidpad at 0.37 lock, against a real Mk7's 0.90-0.95 g\n");
        std::printf("      model off                         %.4f g\n", control.lateralAcceleration);
        std::printf("      thermal, on the plateau           %.4f g   core %5.1f\n", hot.lateralAcceleration,
                    hot.temperatures.core);
        std::printf("      thermal, seeded at track temp     %.4f g   core %5.1f\n", cold.lateralAcceleration,
                    cold.temperatures.core);
        std::printf("\n");
    }

    SECTION("and 0-100 with traction control in sport, against a published 6.5-6.6 s")
    {
        const auto world = plate(1000.0, 60.0, 4.0);

        const auto control = accelerate(off, world, tyreDefaultTemperature);
        const auto hot = accelerate(on, world, tyreDefaultTemperature);
        const auto cold = accelerate(on, world, conditions.trackTemperature);

        std::printf("    0-100 km/h, traction control in sport\n");
        std::printf("      model off                         %.3f s\n", control.toHundred);
        std::printf("      thermal, on the plateau           %.3f s   core %5.1f at the finish\n", hot.toHundred,
                    hot.exit.core);
        std::printf("      thermal, seeded at track temp     %.3f s   core %5.1f at the finish\n", cold.toHundred,
                    cold.exit.core);
        std::printf("\n");
    }
}

TEST_CASE("what the tread-road interface resistance is worth", "[.tyre-thermal]")
{
    // `TyreThermal::roadContactConductance`. The semi-infinite road term assumes the tread and the
    // road touch everywhere; rubber on rough asphalt does not. The figure is measured on this exact
    // interface — C. David Miller, NASA TN D-8161, 1976, **2.52e4 W/(m2.K) for rubber against asphalt
    // and stated as a lower limit** — and the band is his own 1.2e4-to-5.7e4 carried through the
    // same conversion.
    const auto guard = JoltGuard{};
    const auto conditions = weather();

    constexpr auto measured = 25200.0;
    constexpr auto low = 10080.0;
    constexpr auto high = 47880.0;

    SECTION("what it does to the road path, at the speed every figure here is quoted at")
    {
        const auto setup = golfWith(true, -1.0);
        const auto& thermal = setup.corners.front().tyre.thermal;
        const auto world = plate(600.0, 60.0, 4.0);

        auto state = VehicleState{};
        settle(setup, state, world, hundred, 20.0);

        auto last = VehicleStep{};
        for (auto step = 0; step < 180; step++)
        {
            const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, {}, weather());
            REQUIRE(stepped.has_value());
            last = stepped.value();
        }

        const auto& front = last.corners.front();
        const auto penetration = front.patch.penetration;
        const auto chord = 2.0 * std::sqrt(std::max(2.0 * tyreRadius * penetration - penetration * penetration, 0.0));
        const auto patchArea = chord * setup.sampling.width;
        const auto residence =
            std::clamp(chord / std::max(std::abs(front.contact.longitudinalVelocity), 1e-3), 1e-3, 1.0);
        const auto effusivity = std::sqrt(thermal.conductivity * thermal.density * thermal.specificHeat);
        const auto perfect = 2.0 * effusivity / std::sqrt(3.141592653589793 * residence) * thermal.roadEffusivity /
                             (effusivity + thermal.roadEffusivity);

        std::printf("\n  Rolling at 100 km/h, residence %.4f s, patch %.4f m2.\n\n", residence, patchArea);
        std::printf("    contact conductance      h_total        road path      of perfect\n");
        std::printf("    perfect (control)      %8.0f W/(m2.K)  %7.2f W/K      100.0%%\n", perfect,
                    perfect * patchArea);

        for (const auto stated : {low, measured, high})
        {
            const auto series = 1.0 / (1.0 / perfect + 1.0 / stated);
            std::printf("    %8.0f W/(m2.K)     %8.0f W/(m2.K)  %7.2f W/K      %5.1f%%\n", stated, series,
                        series * patchArea, 100.0 * series / perfect);
        }

        // And the approximation the series form is, measured rather than asserted. The exact
        // transient for two semi-infinite bodies joined by an interface conductance, averaged over
        // the residence time.
        const auto b = measured * (effusivity + thermal.roadEffusivity) / (effusivity * thermal.roadEffusivity);
        const auto tau = b * b * residence;
        const auto exact =
            measured * (std::exp(tau) * std::erfc(std::sqrt(tau)) - 1.0 + 2.0 * std::sqrt(tau / 3.141592653589793)) /
            tau;
        const auto series = 1.0 / (1.0 / perfect + 1.0 / measured);

        std::printf("\n    the exact transient at %.0f is %.0f against the series form's %.0f, so the series is\n",
                    measured, exact, series);
        std::printf("    %.1f%% pessimistic, which is %.1f%% of the road path\n\n", 100.0 * (exact - series) / series,
                    100.0 * (exact - series) * patchArea / (perfect * patchArea));
    }

    SECTION("and what it is worth on the tread, driven")
    {
        const auto straightWorld = plate(8000.0, 100.0, 8.0);
        const auto circleWorld = plate(1400.0, 900.0, 8.0);
        const auto marks = std::vector<double>{};

        std::printf("\n  Air %.1f C, track %.1f C, seeded at the track's temperature and driven four minutes.\n\n",
                    conditions.airTemperature, conditions.trackTemperature);
        std::printf("    fixture                    perfect contact      %.0f W/(m2.K)     gain\n", measured);

        const auto row = [&](const char* label, const PhysicsWorld& world, const double steering, const double speed,
                             const bool hottest) {
            const auto open = hold(golfWithContact(0.0), world, steering, speed, 260.0, conditions.trackTemperature,
                                   marks);
            const auto resisted =
                hold(golfWithContact(measured), world, steering, speed, 260.0, conditions.trackTemperature, marks);

            const auto before = hottest ? open.temperatures.hottestCore : open.temperatures.core;
            const auto after = hottest ? resisted.temperatures.hottestCore : resisted.temperatures.core;

            std::printf("    %-26s %8.1f C            %8.1f C      %+5.2f\n", label, before, after, after - before);
        };

        row("straight 100 km/h, core", straightWorld, 0.0, hundred, false);
        row("corner 20 m/s, hottest core", circleWorld, 0.37, 20.0, true);
        row("corner 40 m/s, mean core", circleWorld, 0.20, 40.0, false);
        row("corner 40 m/s, hottest core", circleWorld, 0.20, 40.0, true);

        std::printf("\n    The tread reaches this compound's own plateau, which starts at 55 C.\n\n");
    }

    SECTION("and what it is worth on a performance figure, which is predicted to be nothing")
    {
        const auto brakeWorld = plate(600.0, 60.0, 2.0);
        const auto padWorld = plate(900.0, 900.0, 8.0);
        const auto launchWorld = plate(1000.0, 60.0, 4.0);
        const auto marks = std::vector<double>{};

        const auto openStop = brake(golfWithContact(0.0), brakeWorld, conditions.trackTemperature);
        const auto resistedStop = brake(golfWithContact(measured), brakeWorld, conditions.trackTemperature);

        const auto openPad = hold(golfWithContact(0.0), padWorld, 0.37, 20.0, 12.0, conditions.trackTemperature, marks);
        const auto resistedPad =
            hold(golfWithContact(measured), padWorld, 0.37, 20.0, 12.0, conditions.trackTemperature, marks);

        const auto openLaunch = accelerate(golfWithContact(0.0), launchWorld, conditions.trackTemperature);
        const auto resistedLaunch = accelerate(golfWithContact(measured), launchWorld, conditions.trackTemperature);

        std::printf("\n  Cold start, air %.1f C, track %.1f C. The core has 6957 J/K and a stop is three seconds.\n\n",
                    conditions.airTemperature, conditions.trackTemperature);
        std::printf("    100-0 with ABS      %6.2f m  ->  %6.2f m   core %.2f -> %.2f against %.2f -> %.2f\n",
                    openStop.distance, resistedStop.distance, openStop.entry.core, openStop.exit.core,
                    resistedStop.entry.core, resistedStop.exit.core);
        std::printf("    skidpad             %.4f g  ->  %.4f g\n", openPad.lateralAcceleration,
                    resistedPad.lateralAcceleration);
        std::printf("    0-100, TC sport     %6.3f s  ->  %6.3f s\n\n", openLaunch.toHundred, resistedLaunch.toHundred);

        // **Whether that stop is grip or the anti-lock unit is the question, and this is what settles
        // it.** `braking-criteria-flip-on-tiny-plant-changes` says a controller calibrated against a
        // plant moves its cycling on a fraction of a per cent, so a distance is the worst instrument
        // there is for a small grip change. Two discriminators, both cheap:
        //
        //   1. a sweep across the sourced band — a *monotonic* march is grip, scatter is the
        //      controller;
        //   2. the same pair seeded on the compound's plateau, where the curve is flat at exactly
        //      1.00 and the core cannot leave it in three seconds. **Grip is then identical by
        //      construction, so any movement at all is not grip.**
        std::printf("    the same stop across the band, and seeded on the plateau where grip cannot move\n");

        for (const auto stated : {0.0, low, measured, high, 1.0e6})
        {
            const auto cold = brake(golfWithContact(stated), brakeWorld, conditions.trackTemperature);
            const auto flat = brake(golfWithContact(stated), brakeWorld, tyreDefaultTemperature);

            std::printf("      %8.0f W/(m2.K)   cold %6.2f m  %.3f g      plateau %6.2f m  %.3f g\n", stated,
                        cold.distance, cold.meanDeceleration(), flat.distance, flat.meanDeceleration());
        }

        std::printf("\n");
    }
}

TEST_CASE("what the groove's share of the patch is worth on the road path", "[.tyre-thermal]")
{
    // `TyreThermal::roadAreaFraction`. The road term multiplies the *gross* patch area and 28% of
    // this tread is groove; grooves do not touch the road, so the conducting area is 0.72 of the
    // geometric patch. A hole in the area rather than a second path — still air across a 7.5 mm
    // groove is about 3.5 W/(m2.K), a thousandth of the rubber's — and a different mechanism from
    // the interface conductance above, whose figure is measured on a smooth-tread tyre.
    const auto guard = JoltGuard{};
    const auto conditions = weather();

    constexpr auto grooved = 0.72;
    constexpr auto interface_ = 25200.0;

    SECTION("what it does to the road path, at the speed every figure here is quoted at")
    {
        const auto setup = golfWith(true, -1.0);
        const auto& thermal = setup.corners.front().tyre.thermal;
        const auto world = plate(600.0, 60.0, 4.0);

        auto state = VehicleState{};
        settle(setup, state, world, hundred, 20.0);

        auto last = VehicleStep{};
        for (auto step = 0; step < 180; step++)
        {
            const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, {}, weather());
            REQUIRE(stepped.has_value());
            last = stepped.value();
        }

        const auto& front = last.corners.front();
        const auto penetration = front.patch.penetration;
        const auto chord = 2.0 * std::sqrt(std::max(2.0 * tyreRadius * penetration - penetration * penetration, 0.0));
        const auto patchArea = chord * setup.sampling.width;
        const auto residence =
            std::clamp(chord / std::max(std::abs(front.contact.longitudinalVelocity), 1e-3), 1e-3, 1.0);
        const auto effusivity = std::sqrt(thermal.conductivity * thermal.density * thermal.specificHeat);
        const auto perfect = 2.0 * effusivity / std::sqrt(3.141592653589793 * residence) * thermal.roadEffusivity /
                             (effusivity + thermal.roadEffusivity);
        const auto series = 1.0 / (1.0 / perfect + 1.0 / interface_);

        std::printf("\n  Rolling at 100 km/h, residence %.4f s, gross patch %.4f m2.\n\n", residence, patchArea);
        std::printf("    area fraction                        road path      of gross\n");
        std::printf("    1.00 gross (shipped)                 %7.2f W/K      100.0%%\n", perfect * patchArea);
        std::printf("    %.2f rubber only                     %7.2f W/K      %5.1f%%\n", grooved,
                    perfect * patchArea * grooved, 100.0 * grooved);
        std::printf("    and stacked with the interface's %.0f W/(m2.K):\n", interface_);
        std::printf("    1.00 with the interface              %7.2f W/K      %5.1f%%\n", series * patchArea,
                    100.0 * series / perfect);
        std::printf("    %.2f with the interface              %7.2f W/K      %5.1f%%\n\n", grooved,
                    series * patchArea * grooved, 100.0 * grooved * series / perfect);
    }

    SECTION("and what it is worth on the tread, driven")
    {
        const auto straightWorld = plate(8000.0, 100.0, 8.0);
        const auto circleWorld = plate(1400.0, 900.0, 8.0);
        const auto marks = std::vector<double>{};

        std::printf("\n  Air %.1f C, track %.1f C, seeded at the track's temperature and driven four minutes.\n\n",
                    conditions.airTemperature, conditions.trackTemperature);
        std::printf("    fixture                    gross patch          %.2f of it       gain\n", grooved);

        const auto row = [&](const char* label, const PhysicsWorld& world, const double steering, const double speed,
                             const bool hottest) {
            const auto gross =
                hold(golfWithArea(1.0), world, steering, speed, 260.0, conditions.trackTemperature, marks);
            const auto rubber =
                hold(golfWithArea(grooved), world, steering, speed, 260.0, conditions.trackTemperature, marks);

            const auto before = hottest ? gross.temperatures.hottestCore : gross.temperatures.core;
            const auto after = hottest ? rubber.temperatures.hottestCore : rubber.temperatures.core;

            std::printf("    %-26s %8.1f C            %8.1f C      %+5.2f\n", label, before, after, after - before);
        };

        row("straight 100 km/h, core", straightWorld, 0.0, hundred, false);
        row("corner 20 m/s, hottest core", circleWorld, 0.37, 20.0, true);
        row("corner 40 m/s, hottest core", circleWorld, 0.20, 40.0, true);

        std::printf("\n    The hardest fixture here is the 40 m/s corner's hottest core, which is where the\n"
                    "    +2.8 C estimate in docs/tyre-state-brief.md was made.\n\n");
    }
}

TEST_CASE("what the tyre's air does, and what its pressure is worth", "[.tyre-pressure]")
{
    // **Stage 2's measurements live here and not in a test**, for the reason every measurement in
    // this file does: a threshold attached to an instrument becomes a calibration target, which is
    // Dominic's own correction (`diagnostic-not-a-calibration-target`).
    const auto guard = JoltGuard{};
    const auto conditions = weather();

    auto setup = golfWith(true, -1.0);
    setup.tyrePressure = true;
    const auto pressure = setup.corners.front().tyre.pressure;

    std::printf("\n  Air %.1f C, track %.1f C.\n", conditions.airTemperature, conditions.trackTemperature);
    std::printf("  Cold %.1f psi set at %.1f C; ideal %.1f psi, which this tyre reaches at %.1f C of air.\n\n",
                pressure.coldPressure * psiPerPascal, pressure.coldReferenceTemperature,
                pressure.idealPressure * psiPerPascal, tyreGasTemperatureAtIdealPressure(pressure));

    SECTION("what the gas law is worth across the temperatures this car actually sees")
    {
        std::printf("    gas temperature -> pressure, and what each coupling does with it\n");
        std::printf("      %6s  %9s  %14s  %14s\n", "C", "psi", "vertical rate", "rolling res");

        auto state = TyreState{};
        for (const auto celsius : {0.0, 20.0, 31.5, 40.0, 50.0, 61.2, 70.0, 85.0})
        {
            state.gasTemperature = celsius;
            std::printf("      %6.1f  %9.2f  %13.3f%%  %13.3f%%\n", celsius,
                        tyrePressureAt(pressure, state) * psiPerPascal,
                        100.0 * (tyrePressureVerticalRateScale(pressure, state) - 1.0),
                        100.0 * (tyrePressureRollingResistanceScale(pressure, state) - 1.0));
        }

        std::printf("\n");
    }

    SECTION("what a stint does to the air, driven from cold")
    {
        const auto circleWorld = plate(900.0, 900.0, 8.0);
        const auto marks = std::vector<double>{30.0, 60.0, 120.0, 240.0};

        const auto cornering = hold(setup, circleWorld, 0.37, 20.0, 260.0, conditions.trackTemperature, marks);

        auto gasState = TyreState{};
        gasState.gasTemperature = cornering.temperatures.gas;

        std::printf("    260 s of 0.37 lock at 20 m/s, every node from the track's temperature\n");
        std::printf("      surface %5.1f C   core %5.1f C   carcass %5.1f C   **gas %5.1f C**\n",
                    cornering.temperatures.surface, cornering.temperatures.core, cornering.temperatures.carcass,
                    cornering.temperatures.gas);
        std::printf("      pressure %5.2f psi against an ideal of %5.2f, and the air needs %.1f C to get there\n",
                    tyrePressureAt(pressure, gasState) * psiPerPascal, pressure.idealPressure * psiPerPascal,
                    tyreGasTemperatureAtIdealPressure(pressure));
        std::printf("\n");
    }

    SECTION("and what it costs, against the same car at its ideal pressure")
    {
        const auto world = plate(600.0, 60.0, 2.0);

        auto ideal = setup;
        const auto cold = brake(setup, world, conditions.trackTemperature);
        const auto warm = brake(ideal, world, tyreGasTemperatureAtIdealPressure(pressure));

        std::printf("    100-0 with ABS, the pressure model on\n");
        std::printf("      seeded at the ideal pressure     %6.2f m   %.3f g\n", warm.distance,
                    warm.meanDeceleration());
        std::printf("      seeded at the track temperature  %6.2f m   %.3f g\n", cold.distance,
                    cold.meanDeceleration());
        std::printf("\n");
    }

    SECTION("what a grip-against-pressure law would be worth, at the three published donor fits")
    {
        // **The form is sourced and the numbers are not**, and this section exists to say how much
        // that costs rather than to close it. MF-Tyre 6.1 multiplies a peak friction coefficient by
        // `1 + p3·dpi + p4·dpi²` — Besselink, Schmeitz & Pacejka, Vehicle System Dynamics
        // 48(sup1):337, 2010 — and the campaign those terms were fitted on, de Hoogh, TU/e
        // DCT 2005.46, measured five passenger car tyres at three pressures each. Fitting this
        // quadratic to that report's own figures at **this model's own 4000 N nominal load** gives
        // three pairs that disagree in size and in sign. The fourth is the closest relative this car
        // has in any published measurement — a Goodyear Eagle F1 Asymmetric 255/45 ZR19 on a
        // Flat-Trac at 33, 37 and 41 psi (Singh & Sivaramakrishnan, The Goodyear Tire & Rubber
        // Company, *Extended Pacejka Tire Model for Enhanced Vehicle Stability Control*,
        // arXiv:2305.18422) — and it is **lateral, monotonic and tiny**, with grip rising with
        // pressure and no optimum anywhere in the measured range. Nothing here is a candidate to ship.
        struct Donor
        {
            const char* name;
            double linear;
            double quadratic;
        };

        const auto donors = std::array<Donor, 4>{Donor{"185/60 R14, longitudinal", 0.173, -2.250},
                                                 Donor{"205/55 R16, longitudinal", 0.098, -1.615},
                                                 Donor{"225/55 R16, longitudinal", -0.018, 0.470},
                                                 Donor{"255/45 ZR19 summer, lateral", 0.049, 0.035}};

        // The pressures the driven lap finished at. **Stamped on rather than generated**, because
        // this fixture has no brakes and the front-to-rear split is made *by* the brakes — a skidpad
        // leaves all four wheels at ambient, which is the fixture that reported the rim's sign
        // backwards last night. `traces/rack-exit-20260828-pressure-seat.csv`.
        constexpr auto frontPsi = 33.10;
        constexpr auto rearPsi = 30.58;
        constexpr auto seed = 31.5;

        auto split = setup;
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            auto& air = split.corners[index].tyre.pressure;
            air.coldReferenceTemperature = seed;
            air.coldPressure = (rearAxle(static_cast<Corner>(index)) ? rearPsi : frontPsi) / psiPerPascal;
        }

        std::printf("    the Magic Formula's own pressure terms, at the lap's own %.2f / %.2f psi\n", frontPsi,
                    rearPsi);
        std::printf("      %-30s %8s %10s %9s %9s %10s\n", "donor tyre", "linear", "quadratic", "front", "rear",
                    "balance");

        for (const auto& donor : donors)
        {
            auto front = TyreState{};
            auto rear = TyreState{};
            front.gasTemperature = seed;
            rear.gasTemperature = seed;

            auto air = split.corners.front().tyre.pressure;
            air.gripPressureLinear = donor.linear;
            air.gripPressureQuadratic = donor.quadratic;
            const auto atFront = tyrePressureGripScale(air, front);

            air.coldPressure = rearPsi / psiPerPascal;
            const auto atRear = tyrePressureGripScale(air, rear);

            std::printf("      %-30s %8.3f %10.3f %8.2f%% %8.2f%% %9.2f%%\n", donor.name, donor.linear,
                        donor.quadratic, 100.0 * (atFront - 1.0), 100.0 * (atRear - 1.0),
                        100.0 * (atFront / atRear - 1.0));
        }

        // And what that is worth on the car. The control and each donor differ in **nothing but the
        // two coefficients**, so the delta is the grip law alone — the vertical rate and the rolling
        // resistance are identical across every row, both already carrying the same split.
        const auto world = plate(900.0, 900.0, 8.0);
        const auto marks = std::vector<double>{};
        const auto control = hold(split, world, 0.37, 20.0, 12.0, seed, marks);

        std::printf("\n    skidpad at 0.37 lock, the same car with only the two coefficients changed\n");
        std::printf("      none stated, which is the shipped car  %.4f g\n", control.lateralAcceleration);

        for (const auto& donor : donors)
        {
            auto stated = split;
            for (auto& corner : stated.corners)
            {
                corner.tyre.pressure.gripPressureLinear = donor.linear;
                corner.tyre.pressure.gripPressureQuadratic = donor.quadratic;
            }

            const auto held = hold(stated, world, 0.37, 20.0, 12.0, seed, marks);
            std::printf("      %-38s %.4f g   %+.2f%%\n", donor.name, held.lateralAcceleration,
                        100.0 * (held.lateralAcceleration / control.lateralAcceleration - 1.0));
        }

        std::printf("\n");
    }
}
