// What the thermal brake actually does. `./EngineTests "[.brake-thermal]"`.
//
// **A probe and not a gate**: no acceptance threshold on any finding, every `REQUIRE` a fixture
// precondition. It answers three questions:
//
//   1. **How hot does one stop make a disc, and how fast does it cool?** Those two decide everything
//      else, and both are derived rather than fitted — a published disc mass and Limpert's
//      correlation.
//   2. **What does a back-to-back brake test do?** Ten stops from 100 km/h with a stated interval
//      between them, which is the shape of SAE J2522's fade procedure and is the case an OE pad's
//      edge code stops covering.
//   3. **Is the first stop of the day unchanged?** It must be. Every braking figure in `docs/` was
//      measured on a cold brake making one stop, and this model has nothing to say about that case.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::discConvection;
using raceengine::Feature;
using raceengine::frictionAtTemperature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::seedDiscTemperatures;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto rideHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto startZ = 20.0;

constexpr auto noBrakePressure = std::array<double, cornerCount>{};

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

[[nodiscard]] PhysicsWorld plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 4000.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 4.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    return std::move(world.value());
}

[[nodiscard]] VehicleSetup golfWith(const bool thermal)
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto setup = built.value();
    setup.brakeThermal = thermal;

    return setup;
}

struct StopResult
{
    double distance = 0.0;
    double time = 0.0;
    double entryDisc = 0.0;
    double exitDisc = 0.0;
    double friction = 0.0;
};

// A back-to-back brake test: N stops from 100 km/h at a full pedal with the anti-lock unit on, each
// followed by a fixed interval rolling at 100 km/h.
//
// **The interval is the cooling and the fixture says so.** A real fade procedure drives back up to
// speed and that heats nothing, so coasting at the entry speed is the right stand-in for it — what it
// is not is a claim about how long a lap takes. Twenty seconds is a back-to-back test's own cadence.
[[nodiscard]] std::vector<StopResult> fadeTest(const VehicleSetup& setup, const PhysicsWorld& world, const int stops,
                                               const double interval, const double seedDisc)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, rideHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    seedDiscTemperatures(state, seedDisc);

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

    const auto toSpeed = [&]
    {
        state.chassis.position = glm::dvec3(0.0, state.chassis.position.y, startZ);
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, hundred);
        for (auto& corner : state.corners)
        {
            corner.wheelSpeed = hundred / tyreRadius;
        }
    };

    auto results = std::vector<StopResult>{};

    for (auto number = 0; number < stops; number++)
    {
        toSpeed();

        // Half a second of rolling to settle the suspension into the entry speed.
        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes, weather());
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        auto result = StopResult{};
        result.entryDisc = state.corners.front().discTemperature;
        result.friction = frictionAtTemperature(setup.corners.front().disc.couple, result.entryDisc);

        const auto start = state.chassis.position.z;

        auto input = VehicleInput{};
        input.brake = 1.0;

        for (auto step = 0; step < 360 * 20; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup, 1.0), tick);
            const auto stepped =
                stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes, weather());
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            result.time += tick;

            if (state.chassis.linearVelocity.z <= 0.0)
            {
                break;
            }
        }

        result.distance = state.chassis.position.z - start;
        result.exitDisc = state.corners.front().discTemperature;
        results.push_back(result);

        // And the interval, rolling at the entry speed with the brake off.
        toSpeed();
        for (auto step = 0; step < static_cast<int>(interval / tick); step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes, weather());
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }
    }

    return results;
}

} // namespace

TEST_CASE("what one disc is worth, and how fast it loses what it takes", "[.brake-thermal]")
{
    const auto setup = golfWith(true);
    const auto& front = setup.corners.front().disc;
    const auto& rear = setup.corners.back().disc;

    std::printf("\n  The discs, out of two catalogued parts and one published mass.\n\n");
    std::printf("    front  capacity %7.1f J/K   convection %.4f m2   radiation %.4f m2\n", front.heatCapacity,
                front.convectionArea, front.radiationArea);
    std::printf("    rear   capacity %7.1f J/K   convection %.4f m2   radiation %.4f m2\n", rear.heatCapacity,
                rear.convectionArea, rear.radiationArea);
    std::printf("    heat into the disc rather than the pad: %.3f (effusivity partition, published band 0.90-0.95)\n",
                front.heatToDisc);

    const auto energy = 0.5 * 1452.0 * hundred * hundred;
    std::printf("\n    a 100-0 stop is %.0f kJ; the front axle takes about 80%% of it, so each front disc\n",
                energy / 1000.0);
    std::printf("    gains about %.1f C per stop before any cooling.\n\n", 0.8 * energy / 2.0 / front.heatCapacity);

    // **The constant is printed against every path and not against convection alone**, which is the
    // trap here: at a standstill radiation is the larger of the two, so a convection-only figure
    // reads an hour where the answer is half of that.
    constexpr auto stefanBoltzmann = 5.670374419e-8;
    const auto radiative = [&](const double celsius, const double air)
    {
        const auto surface = celsius + 273.15;
        const auto ambient = air + 273.15;

        return front.emissivity * stefanBoltzmann * front.radiationArea * (surface * surface + ambient * ambient) *
               (surface + ambient);
    };

    std::printf("    Cooling, front disc, at 300 C against 20 C air:\n");
    for (const auto kph : {0.0, 30.0, 60.0, 100.0, 150.0, 200.0})
    {
        const auto h = discConvection(front, kph / 3.6, 300.0, 20.0);
        const auto convective = h * front.convectionArea;
        const auto total = convective + radiative(300.0, 20.0);
        std::printf("      %5.0f km/h   h %7.2f W/(m2.K)   convection %6.2f + radiation %5.2f = %6.2f W/K   "
                    "constant %6.1f s\n",
                    kph, h, convective, radiative(300.0, 20.0), total, front.heatCapacity / total);
    }
    std::printf("\n");
}

TEST_CASE("what the pad's friction does as the disc heats", "[.brake-thermal]")
{
    const auto couple = golfWith(true).corners.front().disc.couple;

    std::printf("\n  Pad friction against disc temperature. The plateau is SAE J866's `FF` rating\n");
    std::printf("  (0.35-0.45 across 93-343 C, both bands); the tail is borrowed from published\n");
    std::printf("  SAE J2522 Fade I runs and is nobody's measurement of this pad.\n\n");
    std::printf("      C     mu    of nominal\n");
    for (const auto celsius : {20.0, 100.0, 200.0, 343.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 750.0})
    {
        const auto mu = frictionAtTemperature(couple, celsius);
        std::printf("    %5.0f  %.4f    %5.1f%%\n", celsius, mu, 100.0 * mu / couple.coefficient);
    }
    std::printf("\n");
}

TEST_CASE("ten stops in a row, which is what an OE pad's rating stops covering", "[.brake-thermal]")
{
    const auto guard = JoltGuard{};
    const auto world = plate();
    const auto conditions = weather();

    const auto off = golfWith(false);
    const auto on = golfWith(true);

    std::printf("\n  Air %.1f C. Ten 100-0 stops at a full pedal with ABS, twenty seconds of rolling\n",
                conditions.airTemperature);
    std::printf("  at 100 km/h between them. Front left disc.\n\n");

    const auto control = fadeTest(off, world, 3, 20.0, conditions.airTemperature);
    const auto measured = fadeTest(on, world, 10, 20.0, conditions.airTemperature);

    std::printf("    model off:");
    for (const auto& stop : control)
    {
        std::printf("  %6.2f m", stop.distance);
    }
    std::printf("\n\n");

    std::printf("    stop   entry C   exit C    mu     distance   vs first\n");
    for (auto index = std::size_t{0}; index < measured.size(); index++)
    {
        const auto& stop = measured[index];
        std::printf("    %3zu    %7.1f   %6.1f   %.4f   %7.2f m   %+6.2f%%\n", index + 1, stop.entryDisc, stop.exitDisc,
                    stop.friction, stop.distance, 100.0 * (stop.distance / measured.front().distance - 1.0));
    }

    std::printf("\n    the first stop of the day, thermal against the control: %.2f m against %.2f m\n",
                measured.front().distance, control.front().distance);

    // **The spread with the friction held at 0.4000 is the fixture's own noise floor**, and it has to
    // be printed or the fade below is unreadable. Nothing about the disc reaches the tyre in this
    // stage, so every one of these stops is the same car; what differs is the state carried into it —
    // suspension attitude, tyre deflections and the anti-lock unit's own memory. That is the
    // chaotic-baseline pattern `docs/known-red.md` already documents for this controller.
    auto lowest = measured.front().distance;
    auto highest = measured.front().distance;
    for (const auto& stop : measured)
    {
        lowest = std::min(lowest, stop.distance);
        highest = std::max(highest, stop.distance);
    }
    std::printf("    spread across ten stops with mu held at 0.4000: %.2f to %.2f m, %.1f%% — the noise floor\n",
                lowest, highest, 100.0 * (highest / lowest - 1.0));
    std::printf("\n");
}

TEST_CASE("stage 3: what a wheel lets past, and what the unpublished number is worth", "[.brake-thermal]")
{
    const auto setup = golfWith(true);
    const auto& disc = setup.corners.front().disc;
    const auto& wheel = setup.corners.front().wheel;

    const auto hardware = raceengine::WheelHardware{.mass = 12.25,
                                                    .diameter = 0.4572,
                                                    .emissivity = 0.85,
                                                    .hatWallThickness = 0.007,
                                                    .boltCircleRadius = 0.056,
                                                    .jointConductance = 60.0,
                                                    .toTyre = 4.0,
                                                    .discRadiationShare = 0.5};

    const auto brake = raceengine::BrakeHardware{.discDiameter = 0.340,
                                                 .discThickness = 0.030,
                                                 .discMass = 10.7,
                                                 .discVented = true,
                                                 .hatHeight = 0.050,
                                                 .padRadialHeight = 0.070,
                                                 .padOuterClearance = 0.005};

    std::printf("\n  The wheel, out of a published mass and the disc's own catalogued hat.\n\n");
    std::printf("    capacity %8.1f J/K   convection %.4f m2   radiation %.4f m2\n", wheel.heatCapacity,
                wheel.convectionArea, wheel.radiationArea);
    std::printf("    conduction to the disc %.4f W/K      to the tyre's carcass %.4f W/K\n", wheel.toDisc,
                wheel.toTyre);

    // **The finding.** The brief called stage 3 blocked on the bolted joint's conductance, which
    // nobody publishes and which two published estimates disagree about by thirteen. It is in series
    // with the hat's own neck and the neck is thirty times the smaller.
    const auto neck = raceengine::hatConductance(brake, hardware);
    std::printf("\n    the hat's neck alone, out of geometry and iron's 50 W/(m.K):  %.4f W/K\n", neck);
    std::printf("    the bolted joint, which nobody publishes, swept across its bound:\n\n");
    std::printf("      joint W/K    whole path W/K    of the neck\n");
    for (const auto joint : {30.0, 60.0, 150.0, 400.0, 760.0, 5000.0})
    {
        auto swept = hardware;
        swept.jointConductance = joint;

        const auto path = raceengine::wheelThermalOf(swept, brake).toDisc;
        std::printf("      %8.0f     %10.4f        %6.2f%%\n", joint, path, 100.0 * path / neck);
    }
    std::printf("\n    60 to 760 W/K is a factor of 12.7 and moves the path by 3%%. The number the brief\n");
    std::printf("    called blocking is not the bottleneck; the hat the model already described is.\n\n");

    // And the radiation, which the brief's stated path leaves out altogether.
    std::printf("    Disc -> wheel radiation, which section 7 does not mention at all:\n\n");
    std::printf("      disc C   wheel C    radiation W/K   conduction W/K   whole coupling W/K\n");
    for (const auto pair : std::array{std::array{100.0, 40.0}, std::array{300.0, 70.0}, std::array{500.0, 110.0},
                                      std::array{650.0, 140.0}})
    {
        const auto radiation = raceengine::discToWheelRadiation(disc, wheel, pair[0], pair[1]);
        const auto coupling = raceengine::discToWheelCoupling(disc, wheel, pair[0], pair[1]);
        std::printf("      %6.0f   %7.0f    %13.4f   %14.4f   %18.4f\n", pair[0], pair[1], radiation, wheel.toDisc,
                    coupling);
    }
    std::printf("\n");

    // The steady balance, which is what says how little of it arrives.
    std::printf("    Where the disc's heat goes once the wheel is in the path — steady state, air 20 C,\n");
    std::printf("    the tyre's carcass held at 45 C:\n\n");
    std::printf("      km/h   disc C    wheel C   to the air W   to the tyre W   of what crossed\n");
    for (const auto kph : {50.0, 100.0, 200.0})
    {
        for (const auto discCelsius : {300.0, 500.0})
        {
            // Solved by iterating the wheel's own balance to convergence rather than by inverting it,
            // because the radiation term depends on the answer.
            auto wheelCelsius = 40.0;
            for (auto pass = 0; pass < 200; pass++)
            {
                const auto coupling = raceengine::discToWheelCoupling(disc, wheel, discCelsius, wheelCelsius);
                const auto air =
                    raceengine::wheelConvection(wheel, kph / 3.6, wheelCelsius, 20.0) * wheel.convectionArea;
                const auto total = coupling + air + wheel.toTyre;

                wheelCelsius = (coupling * discCelsius + air * 20.0 + wheel.toTyre * 45.0) / total;
            }

            const auto coupling = raceengine::discToWheelCoupling(disc, wheel, discCelsius, wheelCelsius);
            const auto air = raceengine::wheelConvection(wheel, kph / 3.6, wheelCelsius, 20.0) * wheel.convectionArea;
            const auto crossed = coupling * (discCelsius - wheelCelsius);
            const auto toTyre = wheel.toTyre * (wheelCelsius - 45.0);

            std::printf("      %4.0f   %6.0f   %8.1f   %12.1f   %13.1f   %13.1f%%\n", kph, discCelsius, wheelCelsius,
                        air * (wheelCelsius - 20.0), toTyre, 100.0 * toTyre / std::max(crossed, 1e-6));
        }
    }
    std::printf("\n    **Most of what crosses the joint leaves again to the air**, which is what section 7\n");
    std::printf("    predicted: a wheel is a large aluminium heat sink in a 200 km/h airstream.\n\n");
}

TEST_CASE("stage 3: what the path is worth to the tread, measured against itself switched off", "[.brake-thermal]")
{
    const auto guard = JoltGuard{};
    const auto world = plate();
    const auto conditions = weather();

    std::printf("\n  100-0 stops at a full pedal with the tyres carrying a temperature too, seeded at\n");
    std::printf("  the track's %.1f C. One variable: the wheel's conductance into the carcass.\n",
                conditions.trackTemperature);
    std::printf("  **Two duty cycles, because the first one answers the wrong question** — twenty seconds\n");
    std::printf("  of 100 km/h between stops is twenty seconds of maximum airflow over the wheel.\n");

    const auto measure = [&](const double toTyre, const int stops, const double interval, const bool wheel = true)
    {
        auto setup = golfWith(true);
        setup.tyreThermal = true;
        for (auto& corner : setup.corners)
        {
            if (wheel)
            {
                corner.wheel.toTyre = toTyre;
            }
            else
            {
                // The car stages 1 and 2 measured: no wheel node at all, so no conduction out of the
                // hat and the whole of the disc's radiation spent against the sky.
                corner.wheel = {};
                corner.disc.wheelRadiationShare = 0.0;
            }
        }

        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, rideHeight, startZ);
        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
        }

        seedDiscTemperatures(state, conditions.airTemperature);
        raceengine::seedTyreTemperatures(state, conditions.trackTemperature);

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

        const auto toSpeed = [&]
        {
            state.chassis.position = glm::dvec3(0.0, state.chassis.position.y, startZ);
            state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, hundred);
            for (auto& corner : state.corners)
            {
                corner.wheelSpeed = hundred / tyreRadius;
            }
        };

        for (auto number = 0; number < stops; number++)
        {
            toSpeed();

            auto input = VehicleInput{};
            input.brake = 1.0;

            for (auto step = 0; step < 360 * 20; step++)
            {
                const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                                   brakeCircuitPressures(setup, 1.0), tick);
                const auto stepped =
                    stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes, conditions);
                REQUIRE(stepped.has_value());
                lastStep = stepped.value();

                if (state.chassis.linearVelocity.z <= 0.0)
                {
                    break;
                }
            }

            toSpeed();
            for (auto step = 0; step < static_cast<int>(interval / tick); step++)
            {
                const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
                const auto stepped =
                    stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes, conditions);
                REQUIRE(stepped.has_value());
                lastStep = stepped.value();
            }
        }

        return state;
    };

    const auto report = [&](const char* what, const int stops, const double interval)
    {
        const auto isolated = measure(0.0, stops, interval);
        const auto coupled = measure(4.0, stops, interval);

        std::printf("\n    %s\n\n", what);
        std::printf("    corner        disc C          wheel C         carcass C         core C\n");
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            std::printf("    %5zu    %6.1f -> %6.1f   %6.1f -> %6.1f   %6.1f -> %6.1f   %6.1f -> %6.1f\n", index,
                        isolated.corners[index].discTemperature, coupled.corners[index].discTemperature,
                        isolated.corners[index].wheelTemperature, coupled.corners[index].wheelTemperature,
                        isolated.corners[index].tyre.carcassTemperature, coupled.corners[index].tyre.carcassTemperature,
                        isolated.corners[index].tyre.coreTemperature, coupled.corners[index].tyre.coreTemperature);
        }
        std::printf("\n    front core gain from the whole path: %+.2f C\n",
                    coupled.corners.front().tyre.coreTemperature - isolated.corners.front().tyre.coreTemperature);

        // **And what the wheel costs the disc**, which is the half a reader does not expect: a wheel
        // is a cooling path a disc did not have, so adding it makes the brake *colder*.
        const auto stage2 = measure(0.0, stops, interval, false);
        std::printf("    front disc, no wheel at all against the whole path: %.1f -> %.1f C, %+.1f%%\n",
                    stage2.corners.front().discTemperature, coupled.corners.front().discTemperature,
                    100.0 * (coupled.corners.front().discTemperature / stage2.corners.front().discTemperature - 1.0));
    };

    report("Ten stops, twenty seconds apart — the back-to-back brake test.", 10, 20.0);
    report("Twenty stops, three seconds apart — as near a circuit's duty cycle as a plate gets.", 20, 3.0);
    std::printf("\n");
}

TEST_CASE("stage 3: the ceiling, with the discs pinned and all the time in the world", "[.brake-thermal]")
{
    // **Both driven fixtures run out before the wheel does**, which is this project's own trap for
    // the third time in three days: the wheel's heat capacity is 11 kJ/K against a conductance of
    // about 47 W/K, so its time constant is nearly four minutes and a twenty-stop brake test is two.
    // A measurement that stops while its slowest node is still climbing is a measurement of the
    // fixture.
    //
    // So this one takes the brake out of the question entirely: the discs are **held** at a stated
    // temperature and the car rolls for fifteen minutes. Nothing about it is a lap — it is the
    // *ceiling*, the most this path can ever deliver at that disc temperature and that speed, and it
    // is the number to argue with.
    const auto guard = JoltGuard{};
    const auto world = plate();
    const auto conditions = weather();

    std::printf("\n  The discs pinned and fifteen minutes to converge. **Not a lap** — a ceiling.\n");
    std::printf("  Air %.1f C, track %.1f C, tyres seeded at the track's.\n\n", conditions.airTemperature,
                conditions.trackTemperature);

    const auto ceiling = [&](const double discCelsius, const double kph, const double toTyre)
    {
        auto setup = golfWith(true);
        setup.tyreThermal = true;
        for (auto& corner : setup.corners)
        {
            corner.wheel.toTyre = toTyre;
        }

        const auto speed = kph / 3.6;

        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, rideHeight, startZ);
        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
        }

        raceengine::seedTyreTemperatures(state, conditions.trackTemperature);
        seedDiscTemperatures(state, conditions.airTemperature);

        for (auto step = 0; step < 360 * 900; step++)
        {
            // Held on the plate and at speed: this fixture is about the heat path and not about
            // where the car ends up, and the plate is finite. The 2026-08-28 tyre stage paid for
            // exactly this by driving off the end of one.
            state.chassis.position.z = startZ;
            state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
            for (auto& corner : state.corners)
            {
                corner.wheelSpeed = speed / tyreRadius;
                corner.discTemperature = discCelsius;
            }

            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, {}, conditions).has_value());
        }

        return state;
    };

    std::printf("    disc C   km/h    wheel C    carcass C     core C     core gain\n");
    for (const auto discCelsius : {300.0, 450.0, 557.0})
    {
        for (const auto kph : {60.0, 100.0, 160.0})
        {
            const auto isolated = ceiling(discCelsius, kph, 0.0);
            const auto coupled = ceiling(discCelsius, kph, 4.0);

            std::printf("    %6.0f   %4.0f   %8.1f   %10.1f   %8.1f   %+9.2f C\n", discCelsius, kph,
                        coupled.corners.front().wheelTemperature, coupled.corners.front().tyre.carcassTemperature,
                        coupled.corners.front().tyre.coreTemperature,
                        coupled.corners.front().tyre.coreTemperature - isolated.corners.front().tyre.coreTemperature);
        }
    }

    std::printf("\n    And what it would take to matter: the tread is 20 C short of its window, and moving\n");
    std::printf("    the core that far needs the carcass roughly 40 C higher, which is about 1500 W into\n");
    std::printf("    a node losing 23.8 W/K to the air. At a wheel 20 C over the carcass that is a\n");
    std::printf("    conductance of 75 W/K against the 4.0 this derives — nineteen times.\n\n");
}

TEST_CASE("and fifteen stops with no time to cool, which is where the pad's rating runs out", "[.brake-thermal]")
{
    const auto guard = JoltGuard{};
    const auto world = plate();
    const auto conditions = weather();

    std::printf("\n  Air %.1f C. Fifteen 100-0 stops at a full pedal with ABS and **five** seconds between\n",
                conditions.airTemperature);
    std::printf("  them, which is the shape of SAE J2522's fade procedure rather than a lap.\n\n");

    const auto measured = fadeTest(golfWith(true), world, 15, 5.0, conditions.airTemperature);

    std::printf("    stop   entry C   exit C    mu     of nominal   distance   vs first\n");
    for (auto index = std::size_t{0}; index < measured.size(); index++)
    {
        const auto& stop = measured[index];
        std::printf("    %3zu    %7.1f   %6.1f   %.4f    %6.1f%%    %7.2f m   %+6.2f%%\n", index + 1, stop.entryDisc,
                    stop.exitDisc, stop.friction, 100.0 * stop.friction / 0.40, stop.distance,
                    100.0 * (stop.distance / measured.front().distance - 1.0));
    }
    std::printf("\n");
}
