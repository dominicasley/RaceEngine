// The thermal brake and its fade, gated. `docs/brake-thermal-brief.md`.
//
// Same three groups as `TyreThermalTests`, and for the same reason — that file's header explains why
// an inert hook is worse than no hook:
//
//   1. the arithmetic — the disc's mass against its own geometry, the cooling correlation against a
//      published anchor, the effusivity split against a published band;
//   2. **the inertness proof**, which here comes from the *cold* end of the fade curve where the
//      tyre's came from the hot end of its grip curve. An OE pad is rated flat from 93 °C to 343, so
//      a disc seeded at ambient multiplies brake torque by exactly one;
//   3. the mechanism — a disc heats when the car brakes, does not heat when the wheel is locked,
//      cools on a straight, and fades when it is hot enough.
//
// Nothing here pins a temperature or a stopping distance to a figure. What the brakes reach is a
// *measurement* and it lives in `BrakeThermalProbe` (`[.brake-thermal]`).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
using raceengine::castIronSpecificHeat;
using raceengine::cornerCount;
using raceengine::discConvection;
using raceengine::Feature;
using raceengine::frictionAtTemperature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::lowMetallicOnCastIron;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::seedDiscTemperatures;
using raceengine::stepBrakeThermal;
using raceengine::stepVehicle;
using raceengine::sweptFaceArea;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto rideHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto startZ = 20.0;

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

TEST_CASE("an OE pad does not fade until the standard says it does", "[physics][brakes][thermal]")
{
    const auto couple = lowMetallicOnCastIron();

    SECTION("the plateau is the pad's own edge code and it is flat across both rated bands")
    {
        // SAE J866 rates a lining over 200-400 °F (93-204 °C) and again over 300-650 °F (149-343 °C),
        // and an OE pad's `FF` is 0.35-0.45 in **both**. **The assertion is the flatness**, not the
        // value: a pad marked with the same letter twice is a pad whose maker is stating that it does
        // not fade across that range, and that is a specification rather than a modelling choice.
        REQUIRE(frictionAtTemperature(couple, 20.0) == 0.40);
        REQUIRE(frictionAtTemperature(couple, 100.0) == 0.40);
        REQUIRE(frictionAtTemperature(couple, 204.0) == 0.40);
        REQUIRE(frictionAtTemperature(couple, 343.0) == 0.40);

        // And the seed sits on it, which is the whole of why switching the model on changes nothing.
        REQUIRE(frictionAtTemperature(couple, brakeDefaultTemperature) == 0.40);
    }

    SECTION("and above it the pad fades, into the band the fade test reports")
    {
        // SAE J2522's Fade I shows friction falling from 0.32-0.34 early to **0.24-0.28** late, at
        // disc temperatures in the high hundreds. The tail is borrowed and is asserted only as a
        // direction and a magnitude — this pad has never been on a dynamometer.
        REQUIRE(frictionAtTemperature(couple, 450.0) < 0.40);
        REQUIRE(frictionAtTemperature(couple, 550.0) < frictionAtTemperature(couple, 450.0));
        REQUIRE(frictionAtTemperature(couple, 650.0) >= 0.24);
        REQUIRE(frictionAtTemperature(couple, 650.0) <= 0.28);
    }

    SECTION("and a couple with no curve is the brake this model had before, exactly")
    {
        const auto plain = raceengine::FrictionCouple{.coefficient = 0.40, .fade = {}};

        REQUIRE(frictionAtTemperature(plain, 20.0) == 0.40);
        REQUIRE(frictionAtTemperature(plain, 700.0) == 0.40);
    }
}

TEST_CASE("the disc's thermal half comes out of the same part its torque does", "[physics][brakes][thermal][golf]")
{
    const auto setup = golf();
    const auto& front = setup.corners.front().disc;
    const auto& rear = setup.corners.back().disc;

    SECTION("the front's heat capacity is its published mass times iron's specific heat")
    {
        // 10.7 kg, quoted against the OE part, times 460 J/(kg·K). The number this whole model exists
        // to divide watts by.
        REQUIRE(front.heatCapacity == Catch::Approx(10.7 * castIronSpecificHeat).epsilon(1e-9));
        REQUIRE(front.heatCapacity == Catch::Approx(4922.0).epsilon(0.001));
    }

    SECTION("and it is worth about fifty degrees per stop, which is what makes it worth modelling")
    {
        // A 1452 kg car from 100 km/h is 561 kJ. The front axle takes roughly 80% of it and there are
        // two front discs, so each takes about 224 kJ. **That is the whole case for a brake
        // temperature**: one stop is 46 °C and ten in a row is fade territory, and a model with no
        // disc mass cannot tell the first from the tenth.
        const auto energy = 0.5 * 1452.0 * hundred * hundred;
        const auto perDisc = 0.8 * energy / 2.0;

        REQUIRE(perDisc / front.heatCapacity == Catch::Approx(45.6).margin(2.0));
    }

    SECTION("the rear is smaller than the front in every way that matters")
    {
        REQUIRE(rear.heatCapacity > 0.0);
        REQUIRE(rear.heatCapacity < front.heatCapacity);
        REQUIRE(rear.convectionArea < front.convectionArea);
        REQUIRE(rear.diameter < front.diameter);
    }

    SECTION("a vented disc convects over more area than it radiates from")
    {
        // The vanes are the reason. They roughly double the area air passes over and they radiate
        // almost entirely to each other, so they join one area and not the other.
        REQUIRE(front.convectionArea == Catch::Approx(2.0 * front.radiationArea).epsilon(1e-9));
        REQUIRE(front.radiationArea == Catch::Approx(2.0 * sweptFaceArea(BrakeHardware{})).epsilon(0.05));
    }

    SECTION("the heat split is the effusivity partition and lands under the published band")
    {
        // The tyre's own finding, reused: heat released at a plane between two bodies divides by their
        // thermal effusivities. Grey iron is `sqrt(50·7200·460)` = 12868 and an organic pad about
        // 1732, so the disc takes 0.881 — and brake literature quotes 90-95% into the disc, which is a
        // band this was not fitted to.
        const auto iron = std::sqrt(50.0 * 7200.0 * 460.0);
        const auto pad = std::sqrt(1.2 * 2500.0 * 1000.0);

        REQUIRE(iron == Catch::Approx(12868.0).epsilon(0.001));
        REQUIRE(front.heatToDisc == Catch::Approx(iron / (iron + pad)).epsilon(0.005));
        REQUIRE(front.heatToDisc < 0.95);
        REQUIRE(front.heatToDisc > 0.85);
    }
}

TEST_CASE("the disc's cooling is Limpert's correlation and lands where the literature does",
          "[physics][brakes][thermal][golf]")
{
    const auto thermal = golf().corners.front().disc;

    SECTION("it rises with speed and never falls to nothing")
    {
        const auto parked = discConvection(thermal, 0.0, 300.0, 20.0);
        const auto slow = discConvection(thermal, 60.0 / 3.6, 300.0, 20.0);
        const auto fast = discConvection(thermal, 200.0 / 3.6, 300.0, 20.0);

        REQUIRE(parked > 0.0);
        REQUIRE(slow > parked);
        REQUIRE(fast > slow);
    }

    SECTION("and at 60 km/h it is the order the published analyses give")
    {
        // An independent analysis of a passenger disc lands at 58-60 W/(m²·K) at 60 km/h against CFD,
        // and Limpert's correlation gives about 87 there — it is known to read high because it is a
        // whole-assembly figure carrying the vanes' own pumping. **The bound is deliberately wide and
        // one-sided in the direction the literature disagrees**, because what would be a defect is an
        // order of magnitude, not a factor of one and a half.
        const auto h = discConvection(thermal, 60.0 / 3.6, 300.0, 20.0);

        REQUIRE(h > 50.0);
        REQUIRE(h < 120.0);
    }

    SECTION("and a measured vented rotor's vane coefficients are the same order")
    {
        // 27.0, 52.7 and 78.3 W/(m²·K) at 342, 684 and 1025 rpm on a vented rotor, linear in speed.
        // 1025 rpm is 123 km/h on this car's tyre. The model's whole-disc figure is above the vanes'
        // own, which is the right way round.
        const auto h = discConvection(thermal, 123.0 / 3.6, 300.0, 20.0);

        REQUIRE(h > 78.3);
        REQUIRE(h < 300.0);
    }
}

TEST_CASE("a disc heats when the brake does work and cools when it does not", "[physics][brakes][thermal][golf]")
{
    const auto thermal = golf().corners.front().disc;
    const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

    SECTION("one stop's worth of energy raises it by one stop's worth of temperature")
    {
        auto celsius = 20.0;

        // 224 kJ delivered over four seconds, which is a 100-0 stop's front disc.
        const auto input = BrakeThermalInput{.frictionPower = 224000.0 / 4.0, .airSpeed = 14.0, .ambient = weather};

        for (auto step = 0; step < 360 * 4; step++)
        {
            stepBrakeThermal(thermal, celsius, input, tick);
        }

        // The effusivity split takes 88% of it and cooling takes a little back, so this lands just
        // under the 46 °C the arithmetic above gives for all of it.
        REQUIRE(celsius > 50.0);
        REQUIRE(celsius < 70.0);
    }

    SECTION("and a hot disc on a straight cools with a time constant of a couple of minutes")
    {
        // **The constant is the assertion, not the temperature**, because it is the physical claim: a
        // ten-kilogram lump of iron in a 144 km/h airstream is a slow thing, and a brake that cooled
        // in seconds would be the surprising answer. Measured here rather than asserted from the
        // arithmetic, so a change to either area or to the correlation moves it.
        auto celsius = 500.0;

        const auto input = BrakeThermalInput{.frictionPower = 0.0, .airSpeed = 40.0, .ambient = weather};

        for (auto step = 0; step < 360 * 60; step++)
        {
            stepBrakeThermal(thermal, celsius, input, tick);
        }

        const auto remaining = (celsius - weather.airTemperature) / (500.0 - weather.airTemperature);
        const auto constant = -60.0 / std::log(remaining);

        CAPTURE(celsius, constant);
        REQUIRE(celsius < 500.0);
        REQUIRE(constant > 60.0);
        REQUIRE(constant < 300.0);

        // And given long enough it does reach the weather, which is what catches a model that has
        // stopped losing heat at all.
        for (auto step = 0; step < 360 * 900; step++)
        {
            stepBrakeThermal(thermal, celsius, input, tick);
        }

        REQUIRE(celsius < weather.airTemperature + 5.0);
        REQUIRE(celsius > weather.airTemperature - 1.0);
    }

    SECTION("nothing runs away, however hard it is worked")
    {
        auto celsius = 20.0;

        const auto input = BrakeThermalInput{.frictionPower = 200000.0, .airSpeed = 60.0, .ambient = weather};

        for (auto step = 0; step < 360 * 120; step++)
        {
            stepBrakeThermal(thermal, celsius, input, tick);
        }

        REQUIRE(std::isfinite(celsius));
        REQUIRE(celsius < 2000.0);
    }
}

TEST_CASE("the thermal brake is inert with its switch off and inert while the pad is on its plateau",
          "[physics][brakes][thermal][golf][parity]")
{
    // **The proof this ships on, and it comes from the cold end of the curve.** The tyre had to be
    // seeded *warm* to sit on its grip plateau; a pad's rated friction is flat from 93 °C to 343, so
    // a disc seeded at ambient is on its own plateau and the fade multiplier is exactly 1.0 — which
    // in IEEE arithmetic leaves the brake torque alone rather than nearly alone.
    const auto guard = JoltGuard{};
    const auto world = flatPlate();

    auto off = golf();
    off.brakeThermal = false;

    auto on = golf();
    on.brakeThermal = true;

    const auto weather = AmbientConditions{.airTemperature = 20.0, .trackTemperature = 30.0};

    const auto stop = [&](const VehicleSetup& setup)
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

    const auto reference = stop(off);
    const auto thermal = stop(on);

    SECTION("the discs stayed on the pad's plateau, which is this proof's precondition")
    {
        for (const auto& corner : thermal.corners)
        {
            CAPTURE(corner.discTemperature);
            REQUIRE(corner.discTemperature < 343.0);
        }
    }

    SECTION("and the car ended up in exactly the same place, to the bit")
    {
        REQUIRE(thermal.chassis.position.z == reference.chassis.position.z);
        REQUIRE(thermal.chassis.linearVelocity.z == reference.chassis.linearVelocity.z);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            CAPTURE(index);
            REQUIRE(thermal.corners[index].wheelSpeed == reference.corners[index].wheelSpeed);
            REQUIRE(thermal.corners[index].wishboneAngle == reference.corners[index].wishboneAngle);
        }
    }

    SECTION("and the discs did heat, so the proof is not vacuous")
    {
        // The complement, which is this project's own rule: a criterion that passes because the
        // mechanism is absent has not been tested.
        REQUIRE(thermal.corners.front().discTemperature > weather.airTemperature + 20.0);

        for (const auto& corner : reference.corners)
        {
            REQUIRE(corner.discTemperature == brakeDefaultTemperature);
        }
    }

    SECTION("and the front discs took more heat than the rears, which is where the bias puts it")
    {
        REQUIRE(thermal.corners[0].discTemperature > thermal.corners[2].discTemperature);
    }
}

TEST_CASE("a locked wheel heats the road and not the disc", "[physics][brakes][thermal][golf]")
{
    // **The one place getting this wrong would be invisible and backwards.** A pad clamped to a disc
    // that is not turning dissipates nothing: the energy goes into the tyre and the road, which the
    // tyre's own thermal model already has. A model that used the *commanded* torque instead of the
    // speed it was applied at would heat the discs hardest during a lock-up.
    const auto guard = JoltGuard{};
    const auto world = flatPlate();

    auto setup = golf();
    setup.brakeThermal = true;

    auto state = VehicleState{};
    settle(setup, state, world, hundred);
    seedDiscTemperatures(state, 20.0);

    // Full pedal with no anti-lock unit, which locks this car's wheels.
    auto input = VehicleInput{};
    input.brake = 1.0;

    auto lockedTicks = 0;
    auto whileLocked = 0.0;
    auto rolling = 0.0;
    auto previous = state.corners.front().discTemperature;
    auto wasLocked = false;

    for (auto step = 0; step < 360 * 5; step++)
    {
        REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick).has_value());

        const auto now = state.corners.front().discTemperature;
        const auto locked = std::abs(state.corners.front().wheelSpeed) < 1e-6;
        const auto moving = state.chassis.linearVelocity.z > 5.0;

        // **Summed tick by tick and only across ticks where the wheel was already stopped**, which is
        // the difference between measuring what a locked wheel does and measuring what happened in
        // between two locked episodes. A wheel held on a full pedal does not sit still — the tyre's
        // own spring nudges it and the brake arrests it again — so the two must be separated or the
        // nudges are counted as heating while locked.
        if (moving)
        {
            (locked && wasLocked ? whileLocked : rolling) += now - previous;
            lockedTicks += locked ? 1 : 0;
        }

        previous = now;
        wasLocked = locked;
    }

    CAPTURE(lockedTicks, whileLocked, rolling);
    REQUIRE(lockedTicks > 60);
    REQUIRE(whileLocked < 0.05);
}
