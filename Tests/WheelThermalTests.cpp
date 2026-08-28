// Stage 3 of `docs/brake-thermal-brief.md`, gated: the heat path from disc to wheel to tyre.
//
// This is the stage with no flat part of a curve to hide in. The tyre seeded warm because its grip
// curve is flat there and the brake seeded cold because its fade curve is flat there; a conductance
// has no plateau, so the inertness proof here is the plain one — **a car that states no wheel is the
// car stages 1 and 2 shipped, expression for expression**, and the switch is the same
// `VehicleSetup::brakeThermal` it always was.
//
// Four groups:
//
//   1. **the arithmetic** — the wheel's capacity out of a published mass, the hat's neck out of the
//      disc's own catalogued geometry, and the series with the bolted joint;
//   2. **the finding**, which is that the one quantity nobody publishes does not decide anything: a
//      thirteen-fold uncertainty in the joint moves the path by one part in thirty, because the hat
//      is thirty times the smaller resistance;
//   3. **the inertness proof**, from a wheel of no mass;
//   4. **the mechanism** — a wheel warms behind a hot disc, cools to the air, and passes some of it
//      into the carcass and none of it straight to the tread.
//
// Nothing here pins a temperature. What the wheel reaches is a *measurement* and it lives in
// `BrakeThermalProbe` (`[.brake-thermal]`).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::AmbientConditions;
using raceengine::brakeDefaultTemperature;
using raceengine::BrakeHardware;
using raceengine::BrakeThermalInput;
using raceengine::brakeThermalOf;
using raceengine::bringUpJolt;
using raceengine::castAluminiumSpecificHeat;
using raceengine::castIronConductivity;
using raceengine::cornerCount;
using raceengine::discToWheelCoupling;
using raceengine::discToWheelRadiation;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::hatConductance;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::seedDiscTemperatures;
using raceengine::seedTyreTemperature;
using raceengine::seedTyreTemperatures;
using raceengine::stepBrakeThermal;
using raceengine::stepTyreThermal;
using raceengine::stepVehicle;
using raceengine::stepWheelThermal;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::WheelHardware;
using raceengine::WheelThermalInput;
using raceengine::wheelThermalOf;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto rideHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto startZ = 20.0;

// The Golf's own wheel, restated here rather than reached for through the car, because these cases
// are about the derivation and want to vary one part of it at a time.
[[nodiscard]] WheelHardware golfWheel()
{
    return WheelHardware{.mass = 12.25,
                         .diameter = 0.4572,
                         .emissivity = 0.85,
                         .hatWallThickness = 0.007,
                         .boltCircleRadius = 0.056,
                         .jointConductance = 60.0,
                         .toTyre = 4.0,
                         .discRadiationShare = 0.5};
}

[[nodiscard]] BrakeHardware golfFrontDisc()
{
    return BrakeHardware{.discDiameter = 0.340,
                         .discThickness = 0.030,
                         .discMass = 10.7,
                         .discVented = true,
                         .hatHeight = 0.050,
                         .padRadialHeight = 0.070,
                         .padOuterClearance = 0.005};
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

[[nodiscard]] PhysicsWorld flatPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 600.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 4.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    return std::move(world.value());
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, rideHeight, startZ);

    for (auto step = 0; step < 720; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
    }
}

[[nodiscard]] VehicleSetup golf()
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    return built.value();
}

} // namespace

TEST_CASE("the wheel's thermal half comes out of a published mass and the disc's own geometry",
          "[physics][brakes][thermal][wheel]")
{
    const auto brake = golfFrontDisc();
    const auto wheel = wheelThermalOf(golfWheel(), brake);

    SECTION("its heat capacity is the published mass times aluminium's specific heat")
    {
        // 12.25 kg — 27 lb, quoted for a direct replacement of this car's own 18 x 7.5 5x112 wheel —
        // times A356's 900 J/(kg·K). A class figure and flagged as one, exactly like the tyre's mass.
        REQUIRE(wheel.heatCapacity == Catch::Approx(12.25 * castAluminiumSpecificHeat).epsilon(1e-9));
        REQUIRE(wheel.heatCapacity == Catch::Approx(11025.0).epsilon(0.001));
    }

    SECTION("it convects over more area than it radiates from, because the disc has some of it")
    {
        // The patch the disc is already radiating onto is taken out, or it would be spent twice —
        // once against the disc and once against the sky.
        REQUIRE(wheel.convectionArea > wheel.radiationArea);
        REQUIRE(wheel.radiationArea > 0.0);
    }

    SECTION("and a wheel of no mass is no wheel at all")
    {
        // **The inertness proof's first half.** Every car in this project except the Golf states no
        // wheel, and what they get is a node every step treats as absent.
        const auto absent = wheelThermalOf(WheelHardware{}, brake);

        REQUIRE(absent.heatCapacity == 0.0);
        REQUIRE(absent.toDisc == 0.0);
        REQUIRE(absent.toTyre == 0.0);
        REQUIRE(absent.radiationArea == 0.0);
    }
}

TEST_CASE("the hat is the bottleneck and the bolted joint is not", "[physics][brakes][thermal][wheel]")
{
    // **This is the finding stage 3 turns on.** The brief that planned this work said stage 3 was
    // blocked on one quantity nobody publishes — the thermal conductance of the bolted joint between
    // a disc hat and an alloy wheel — and bounded it at 60 to 760 W/K from two published estimates
    // that disagree by thirteen. It does not decide anything, because it is in series with the hat's
    // own neck: a thin iron cylinder from the swept ring down to the mounting flange.
    const auto brake = golfFrontDisc();
    const auto hardware = golfWheel();

    const auto neck = hatConductance(brake, hardware);

    SECTION("the neck is a couple of watts per kelvin, out of geometry and iron's conductivity alone")
    {
        // Ring inner radius 0.095 m, a catalogued 50 mm hat height, a bolt circle at 0.056 m from a
        // published 5 x 112 PCD, and a 7 mm wall. Nothing here is fitted and nothing is a temperature.
        REQUIRE(neck == Catch::Approx(2.09).epsilon(0.05));
    }

    SECTION("a thirteen-fold uncertainty in the joint is worth one part in thirty of the path")
    {
        const auto pessimistic = wheelThermalOf(hardware, brake).toDisc;

        auto optimistic = hardware;
        optimistic.jointConductance = 760.0;

        const auto best = wheelThermalOf(optimistic, brake).toDisc;

        CAPTURE(neck, pessimistic, best);

        REQUIRE(best > pessimistic);
        REQUIRE(best / pessimistic < 1.05);

        // And both sit just under the neck, which is what "in series with something much larger"
        // means. **If this ever fails, the hat has stopped being the bottleneck and the unpublished
        // number has started deciding the answer** — which is the day it has to be sourced properly.
        REQUIRE(pessimistic < neck);
        REQUIRE(best < neck);
        REQUIRE(best / neck > 0.95);
    }

    SECTION("and a longer or thinner hat throttles it, which is the geometry doing the work")
    {
        auto thin = hardware;
        thin.hatWallThickness = 0.005;

        auto tall = golfFrontDisc();
        tall.hatHeight = 0.080;

        REQUIRE(wheelThermalOf(thin, brake).toDisc < wheelThermalOf(hardware, brake).toDisc);
        REQUIRE(wheelThermalOf(hardware, tall).toDisc < wheelThermalOf(hardware, brake).toDisc);
    }
}

TEST_CASE("the disc radiates onto the wheel, and it is the same order as the bolts",
          "[physics][brakes][thermal][wheel]")
{
    // **The path the brief did not predict.** Section 7 states it as "disc -> hat -> hub and bolts ->
    // wheel rim" and leaves out the largest half at high temperature: the disc's outboard face is a
    // few centimetres from the wheel's inner dish, and a painted alloy wheel is not a mirror.
    const auto brake = golfFrontDisc();
    const auto hardware = golfWheel();
    const auto disc = brakeThermalOf(brake, hardware);
    const auto wheel = wheelThermalOf(hardware, brake);

    SECTION("at a hot disc it is worth about as much as the whole bolted path")
    {
        const auto radiation = discToWheelRadiation(disc, wheel, 500.0, 120.0);

        CAPTURE(radiation, wheel.toDisc);
        REQUIRE(radiation > 0.5 * wheel.toDisc);
        REQUIRE(radiation < 2.0 * wheel.toDisc);
    }

    SECTION("and it collapses when the disc is cold, which is the fourth power")
    {
        const auto hot = discToWheelRadiation(disc, wheel, 500.0, 120.0);
        const auto warm = discToWheelRadiation(disc, wheel, 100.0, 60.0);

        REQUIRE(warm < 0.5 * hot);
        REQUIRE(warm > 0.0);
    }

    SECTION("a disc with no wheel radiates onto nothing")
    {
        // **The partition, not an addition.** A disc that states no wheel spends all of its radiating
        // area against the sky, which is what stages 1 and 2 measured.
        const auto alone = brakeThermalOf(brake);

        REQUIRE(alone.wheelRadiationShare == 0.0);
        REQUIRE(discToWheelRadiation(alone, wheel, 500.0, 120.0) == 0.0);
        REQUIRE(discToWheelCoupling(alone, wheelThermalOf(WheelHardware{}, brake), 500.0, 120.0) == 0.0);
    }

    SECTION("and a disc that states a share but is stepped with no wheel still sheds all of it")
    {
        // **A defect this caught, and it is the shape a partition fails in.** The share exists to stop
        // the outboard face being spent twice; applied on one side only it is spent *nowhere*, and the
        // disc runs hotter than the same disc with no wheel at all. It surfaced as a runaway fixture
        // reaching 2227 °C where it had reached under 2000.
        const auto stated = brakeThermalOf(brake, golfWheel());
        const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

        REQUIRE(stated.wheelRadiationShare > 0.0);

        const auto cool = [&](const raceengine::BrakeThermal& thermal)
        {
            auto celsius = 500.0;
            const auto input = BrakeThermalInput{.frictionPower = 0.0, .airSpeed = 0.0, .ambient = weather};

            for (auto step = 0; step < 360 * 120; step++)
            {
                stepBrakeThermal(thermal, celsius, input, tick);
            }

            return celsius;
        };

        // Parked, where radiation is the larger of the two paths and a lost half would show.
        REQUIRE(cool(stated) == cool(brakeThermalOf(brake)));
    }
}

TEST_CASE("what the disc loses through the wheel is what the wheel gains", "[physics][brakes][thermal][wheel]")
{
    // Energy, which a two-node model can lose silently. The coupling is computed once by the caller
    // and handed to both steps for exactly this reason, and this is the case that says so.
    const auto brake = golfFrontDisc();
    const auto hardware = golfWheel();
    const auto disc = brakeThermalOf(brake, hardware);
    const auto wheel = wheelThermalOf(hardware, brake);
    const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

    auto discCelsius = 500.0;
    auto wheelCelsius = 20.0;

    const auto coupling = discToWheelCoupling(disc, wheel, discCelsius, wheelCelsius);
    REQUIRE(coupling > 0.0);

    const auto before = disc.heatCapacity * discCelsius + wheel.heatCapacity * wheelCelsius;

    // One tick with **no air over either of them**, so the only path open is the one between them.
    // A standstill still leaves free convection and radiation to the sky, so this is not a closed
    // system and the assertion is one-sided: the wheel must gain a share of what the disc loses.
    stepBrakeThermal(disc, discCelsius,
                     BrakeThermalInput{.frictionPower = 0.0,
                                       .airSpeed = 0.0,
                                       .wheelTemperature = 20.0,
                                       .wheelConductance = coupling,
                                       .ambient = weather},
                     tick);

    stepWheelThermal(wheel, wheelCelsius,
                     WheelThermalInput{.discTemperature = 500.0,
                                       .discConductance = coupling,
                                       .tyreTemperature = 0.0,
                                       .tyreConductance = 0.0,
                                       .airSpeed = 0.0,
                                       .ambient = weather},
                     tick);

    const auto after = disc.heatCapacity * discCelsius + wheel.heatCapacity * wheelCelsius;

    CAPTURE(discCelsius, wheelCelsius, coupling, before, after);

    REQUIRE(discCelsius < 500.0);
    REQUIRE(wheelCelsius > 20.0);

    // What crossed, both ways, against the conductance times the difference. Same number to a part
    // in a thousand — which is what says neither node invented or dropped any of it.
    const auto lost = disc.heatCapacity * (500.0 - discCelsius) / tick;
    const auto gained = wheel.heatCapacity * (wheelCelsius - 20.0) / tick;
    const auto expected = coupling * (500.0 - 20.0);

    REQUIRE(gained == Catch::Approx(expected).epsilon(0.001));
    REQUIRE(lost > gained);
    REQUIRE(after < before);
}

TEST_CASE("a wheel warms behind a hot disc and cools to the air behind a cold one", "[physics][brakes][thermal][wheel]")
{
    const auto brake = golfFrontDisc();
    const auto hardware = golfWheel();
    const auto disc = brakeThermalOf(brake, hardware);
    const auto wheel = wheelThermalOf(hardware, brake);
    const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

    const auto hold = [&](const double discCelsius, const double airSpeed, const double seconds)
    {
        auto celsius = weather.airTemperature;

        for (auto step = 0; step < static_cast<int>(seconds * 360.0); step++)
        {
            const auto coupling = discToWheelCoupling(disc, wheel, discCelsius, celsius);

            stepWheelThermal(wheel, celsius,
                             WheelThermalInput{.discTemperature = discCelsius,
                                               .discConductance = coupling,
                                               .tyreTemperature = 0.0,
                                               .tyreConductance = 0.0,
                                               .airSpeed = airSpeed,
                                               .ambient = weather},
                             tick);
        }

        return celsius;
    };

    SECTION("a disc held at 500 C warms it, and to a temperature a wheel survives")
    {
        // **Nothing here is a prediction of what the car does** — this is a disc pinned at 500 °C for
        // four minutes, which no lap sustains. What is asserted is the direction and the order: a
        // wheel behind a very hot brake is warm rather than hot, because it is a large aluminium heat
        // sink in an airstream, and a model that put it near the disc's own temperature would be
        // saying a wheel melts its own tyre.
        const auto warmed = hold(500.0, hundred, 240.0);

        CAPTURE(warmed);
        REQUIRE(warmed > 40.0);
        REQUIRE(warmed < 200.0);
    }

    SECTION("and more airspeed keeps it cooler, which is the whole reason the path is worth little")
    {
        REQUIRE(hold(500.0, 200.0 / 3.6, 240.0) < hold(500.0, 50.0 / 3.6, 240.0));
    }

    SECTION("and behind a cold disc it stays at the air, having nothing to warm it")
    {
        REQUIRE(hold(weather.airTemperature, hundred, 240.0) == Catch::Approx(weather.airTemperature).margin(0.5));
    }
}

TEST_CASE("stage 3 is inert with the brake switch off and inert on a car with no wheel",
          "[physics][brakes][thermal][wheel][golf][parity]")
{
    // **The plain proof, because a conductance has no flat part to hide in.** With `brakeThermal`
    // off, nothing in the wheel path runs at all; with it on and a wheel of no mass, both steps
    // reproduce stage 2's arithmetic expression for expression.
    const auto guard = JoltGuard{};
    const auto world = flatPlate();

    const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

    const auto run = [&](const VehicleSetup& setup)
    {
        auto state = VehicleState{};
        settle(setup, state, world, hundred);
        seedDiscTemperatures(state, weather.airTemperature);

        auto input = VehicleInput{};
        input.brake = 0.35;

        for (auto step = 0; step < 360 * 4; step++)
        {
            REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick, {}, weather).has_value());
        }

        return state;
    };

    auto off = golf();
    off.brakeThermal = false;

    auto stated = golf();
    stated.brakeThermal = true;

    // The same car with its wheel taken away, which is every other car in this project.
    auto absent = stated;
    for (auto& corner : absent.corners)
    {
        corner.wheel = {};
        corner.disc.wheelRadiationShare = 0.0;
    }

    const auto quiet = run(off);
    const auto withWheel = run(stated);
    const auto without = run(absent);

    SECTION("with the switch off the wheels never move off their seed")
    {
        for (const auto& corner : quiet.corners)
        {
            REQUIRE(corner.wheelTemperature == brakeDefaultTemperature);
        }
    }

    SECTION("with no wheel stated the discs are the discs stage 2 measured, to the bit")
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            CAPTURE(index);
            REQUIRE(without.corners[index].wheelTemperature == brakeDefaultTemperature);
        }
    }

    SECTION("and the wheel does warm and does cool the disc, so neither proof is vacuous")
    {
        // The complement, which is this project's own rule. A wheel that never warmed would make
        // every assertion above true for the wrong reason.
        //
        // **The margin is hundredths of a degree and that is not a weak test, it is the finding.**
        // The wheel is 11 kJ/K behind a 2 W/K neck, so its time constant is nearly four minutes; four
        // seconds of braking moves it by about a fortieth of a degree. Asserting a whole degree here
        // would be asserting that a wheel warms faster than aluminium can.
        CAPTURE(withWheel.corners.front().wheelTemperature);
        REQUIRE(withWheel.corners.front().wheelTemperature > brakeDefaultTemperature);

        // And the disc is **cooler** for having it, which is stage 3's own prediction: the wheel is a
        // cooling path the disc did not have. Not warmer, which is what a reader expects of a term
        // called "heat into the wheel".
        CAPTURE(withWheel.corners.front().discTemperature, without.corners.front().discTemperature);
        REQUIRE(withWheel.corners.front().discTemperature < without.corners.front().discTemperature);
    }
}

TEST_CASE("the wheel reaches the tyre's carcass and never the tread directly",
          "[physics][brakes][thermal][wheel][golf]")
{
    // The path arrives at the **carcass**, which is why the answer is small: the carcass loses to the
    // air over both sidewalls before the core sees any of it. A model that fed the tread would make
    // brake heat worth several times what it is.
    const auto guard = JoltGuard{};
    const auto world = flatPlate();

    const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

    const auto run = [&](const double toTyre)
    {
        auto setup = golf();
        setup.brakeThermal = true;
        setup.tyreThermal = true;

        for (auto& corner : setup.corners)
        {
            corner.wheel.toTyre = toTyre;
        }

        auto state = VehicleState{};
        settle(setup, state, world, hundred);
        seedDiscTemperatures(state, weather.airTemperature);
        seedTyreTemperatures(state, weather.airTemperature);

        // A disc already at a circuit's temperature, so this measures the path rather than the
        // warm-up in front of it.
        for (auto& corner : state.corners)
        {
            corner.discTemperature = 450.0;
        }

        auto input = VehicleInput{};
        input.brake = 0.20;

        for (auto step = 0; step < 360 * 60; step++)
        {
            REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick, {}, weather).has_value());
        }

        return state;
    };

    const auto coupled = run(4.0);
    const auto isolated = run(0.0);

    SECTION("the carcass is warmer for it")
    {
        REQUIRE(coupled.corners.front().tyre.carcassTemperature > isolated.corners.front().tyre.carcassTemperature);
    }

    SECTION("and the core follows it, by less")
    {
        const auto carcass =
            coupled.corners.front().tyre.carcassTemperature - isolated.corners.front().tyre.carcassTemperature;
        const auto core = coupled.corners.front().tyre.coreTemperature - isolated.corners.front().tyre.coreTemperature;

        CAPTURE(carcass, core);
        REQUIRE(core > 0.0);
        REQUIRE(core < carcass);
    }

    SECTION("and a wheel with no path to the tyre leaves the tyre alone, to the bit")
    {
        // **Asserted on `stepTyreThermal` directly rather than through the car**, and the reason is
        // worth keeping: a car-level A/B cannot isolate this. Pinning a disc at 450 °C to make the
        // path worth measuring also puts the pad on its fade curve, so a run with the brake model on
        // brakes 10% weaker than one with it off and every tyre temperature downstream of that
        // differs for a reason that has nothing to do with the rim.
        //
        // The claim is exact and belongs at the function: with the conductance at zero, the carcass
        // must not be able to tell what the rim is doing.
        const auto thermal = golf().corners.front().tyre.thermal;

        const auto advance = [&](const double rim, const double conductance)
        {
            auto state = raceengine::TyreState{};
            seedTyreTemperature(state, 40.0);

            const auto input = raceengine::TyreThermalInput{.slipPower = 3000.0,
                                                            .verticalLoad = 4000.0,
                                                            .rollingResistance = 0.012,
                                                            .roadSpeed = hundred,
                                                            .airSpeed = hundred,
                                                            .patchLength = 0.19,
                                                            .patchWidth = 0.225,
                                                            .wheelTemperature = rim,
                                                            .wheelConductance = conductance,
                                                            .ambient = weather};

            for (auto step = 0; step < 360 * 60; step++)
            {
                stepTyreThermal(thermal, state, input, tick);
            }

            return state;
        };

        const auto cold = advance(20.0, 0.0);
        const auto blazing = advance(300.0, 0.0);

        REQUIRE(cold.coreTemperature == blazing.coreTemperature);
        REQUIRE(cold.carcassTemperature == blazing.carcassTemperature);
        REQUIRE(cold.surfaceTemperature == blazing.surfaceTemperature);

        // And it is not vacuous: give it a conductance and the same rim moves the carcass.
        REQUIRE(advance(300.0, 4.0).carcassTemperature > cold.carcassTemperature);
    }
}
