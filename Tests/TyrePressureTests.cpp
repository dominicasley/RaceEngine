// The tyre's air, gated. `docs/tyre-state-brief.md`, section 7 and its Progress entries.
//
// **Stage 2 has one exact law and two derived couplings, and every one of them is checkable by
// hand.** The gas law has no constant to choose; the vertical rate is linear in gauge pressure
// because the load is carried by the contained air (Rhyne's constant-belt model); rolling resistance
// follows a bounded power of pressure. Nothing here is fitted, so nothing here is asserted against a
// number somebody chose.
//
// The cases are in the same three groups the thermal suite uses:
//
//   1. the arithmetic — the gas law forwards and backwards, and each coupling against a hand figure;
//   2. **the inertness proof**, which is what lets this ship: a car whose air sits at its own ideal
//      pressure is the old car to the bit, because every pressure-dependent number the car states is
//      quoted at that pressure;
//   3. the mechanism — a cold tyre is soft and drags, a warming one comes up in pressure, and the
//      gas follows the carcass rather than leading it.
//
// **Grip against pressure is here as a form with no numbers in it.** The Magic Formula's own
// pressure extension is the shape — `1 + p3·dpi + p4·dpi²` on a peak friction coefficient — and both
// coefficients are zero, because the campaign that fitted them concludes there is no tyre-independent
// relation and the three donor fits available disagree in sign. The cases below assert the zeros and
// pin the arithmetic a stated pair would run: an inert factor that reads as a feature is worse than
// no factor, which is `Tyre.cppm:158`'s own lesson.

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::ambientAt;
using raceengine::atmosphericPressure;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::psiPerPascal;
using raceengine::seedTyreGasAtIdealPressure;
using raceengine::seedTyreGasPressures;
using raceengine::seedTyreTemperatures;
using raceengine::stepTyreGas;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::tyreDefaultTemperature;
using raceengine::tyreGasTemperatureAtIdealPressure;
using raceengine::TyrePressure;
using raceengine::tyrePressureAt;
using raceengine::tyrePressureGripScale;
using raceengine::tyrePressureRollingResistanceScale;
using raceengine::tyrePressureVerticalRateScale;
using raceengine::TyreState;
using raceengine::tyreStateGrip;
using raceengine::TyreThermalInput;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto rideHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto rolling = 20.0;
constexpr auto startZ = 40.0;

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

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, rideHeight, startZ);

    for (auto step = 0; step < 720; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, rolling);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = rolling / tyreRadius;
    }
}

// The Golf's own pressure data, taken off the car rather than restated here.
[[nodiscard]] TyrePressure golfPressure()
{
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    return car->corners.front().tyre.pressure;
}

[[nodiscard]] double psi(const double pascals)
{
    return pascals * psiPerPascal;
}

// The gas temperature at which a tyre reads a stated **gauge** pressure: the model's own seed
// inverted, so that a case can ask for "ten per cent under the ideal" rather than for a temperature
// somebody worked out once and wrote down.
[[nodiscard]] double gasTemperatureAtGauge(const TyrePressure& pressure, const double gauge)
{
    constexpr auto absoluteZero = 273.15;

    const auto cold = pressure.coldPressure + atmosphericPressure;
    const auto wanted = gauge + atmosphericPressure;

    return (pressure.coldReferenceTemperature + absoluteZero) * wanted / cold - absoluteZero;
}

} // namespace

TEST_CASE("the gas law reads a tyre's pressure off its air temperature", "[physics][tyre][pressure]")
{
    const auto pressure = golfPressure();

    // The car's own two numbers, in the units the sources quote them in.
    REQUIRE(psi(pressure.coldPressure) == Catch::Approx(28.0).margin(0.01));
    REQUIRE(psi(pressure.idealPressure) == Catch::Approx(34.0).margin(0.01));
    REQUIRE(pressure.coldReferenceTemperature == 20.0);

    auto state = TyreState{};

    // At the temperature it was set at, a tyre reads what it was set to. Exactly, not nearly.
    state.gasTemperature = pressure.coldReferenceTemperature;
    REQUIRE(tyrePressureAt(pressure, state) == Catch::Approx(pressure.coldPressure).margin(1.0));

    // And hotter air is more pressure, in the ratio of the two **absolute** temperatures against the
    // two **absolute** pressures. 28 psi gauge at 20 °C is 42.696 psia; at 61.2 °C that is
    // 42.696 × 334.35/293.15 = 48.70 psia, which is 34.0 psi gauge.
    state.gasTemperature = 61.2;
    REQUIRE(psi(tyrePressureAt(pressure, state)) == Catch::Approx(34.0).margin(0.05));

    // A tyre that has cooled below where it was set reads below it, and the model does not floor it
    // at the cold pressure — a car left out on a cold night really is soft in the morning.
    state.gasTemperature = 0.0;
    REQUIRE(psi(tyrePressureAt(pressure, state)) < 28.0);
    REQUIRE(psi(tyrePressureAt(pressure, state)) == Catch::Approx(25.06).margin(0.05));
}

TEST_CASE("the ideal-pressure seed is derived from the car's own two pressures", "[physics][tyre][pressure]")
{
    const auto pressure = golfPressure();

    // **This is the corroboration the window work turned up and it is asserted here rather than
    // written down only in a brief.** AC's own 28 cold and 34 ideal say, through the gas law alone,
    // that this tyre is at its best when its air is at 61.2 °C — which is inside the 55-75 °C road
    // window this car now runs and 24 °C below the track window it replaced.
    const auto seed = tyreGasTemperatureAtIdealPressure(pressure);
    REQUIRE(seed == Catch::Approx(61.20).margin(0.05));

    // And the seed does what it says: a tyre put there reads exactly its ideal pressure.
    auto state = TyreState{};
    seedTyreGasAtIdealPressure(pressure, state);
    REQUIRE(state.gasTemperature == Catch::Approx(seed));
    REQUIRE(tyrePressureAt(pressure, state) == Catch::Approx(pressure.idealPressure).margin(1.0));

    // **It is deliberately not the tread's seed**, and the gap is the reason the gas needed its own.
    // A tyre whose air sat at the tread's plateau centre would be over its ideal pressure.
    REQUIRE(seed < tyreDefaultTemperature);
    state.gasTemperature = tyreDefaultTemperature;
    REQUIRE(tyrePressureAt(pressure, state) > pressure.idealPressure);
}

TEST_CASE("both couplings are exactly one at the ideal pressure and move the right way off it",
          "[physics][tyre][pressure]")
{
    const auto pressure = golfPressure();

    auto state = TyreState{};
    seedTyreGasAtIdealPressure(pressure, state);

    // **Exactly one, because that is what makes the switch inert.** The car's `tireVerticalRate` and
    // `rollingResistance` are both quoted at the ideal pressure, so a tyre sitting there has to
    // multiply them by a value that leaves them alone in IEEE arithmetic rather than nearly alone.
    REQUIRE(tyrePressureVerticalRateScale(pressure, state) == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(tyrePressureRollingResistanceScale(pressure, state) == Catch::Approx(1.0).margin(1e-9));

    // Cold, at the temperature this scene's track sits at. A soft tyre is a soft spring and a
    // draggier one, and the two move in opposite directions, which is the whole shape of the effect.
    state.gasTemperature = 31.5;
    const auto cold = psi(tyrePressureAt(pressure, state));
    REQUIRE(cold == Catch::Approx(29.67).margin(0.05));

    // Linear in gauge pressure: 29.10/34.0.
    REQUIRE(tyrePressureVerticalRateScale(pressure, state) == Catch::Approx(cold / 34.0).epsilon(0.001));
    REQUIRE(tyrePressureVerticalRateScale(pressure, state) < 1.0);

    // And the power law the other way, `(P/P_ideal)^-0.5`.
    REQUIRE(tyrePressureRollingResistanceScale(pressure, state) ==
            Catch::Approx(std::pow(cold / 34.0, -0.5)).epsilon(0.001));
    REQUIRE(tyrePressureRollingResistanceScale(pressure, state) > 1.0);

    // Over-inflated is the mirror of it, which is worth asserting because nothing else in this model
    // has ever been asked what happens above its reference.
    state.gasTemperature = 100.0;
    REQUIRE(tyrePressureVerticalRateScale(pressure, state) > 1.0);
    REQUIRE(tyrePressureRollingResistanceScale(pressure, state) < 1.0);
}

TEST_CASE("the vertical rate coupling agrees with AC's own spring gain", "[physics][tyre][pressure]")
{
    // **The corroboration that says this is not a borrowed law.** `tyres.ini` states
    // `PRESSURE_SPRING_GAIN 8890` N/m per psi and `RATE 298926` N/m. A line through the origin at
    // that gain reads 302,260 N/m at the ideal 34 psi — 1.1% above the stated rate. So AC's own two
    // numbers say the rate is proportional to pressure to within one per cent, which is Rhyne's
    // constant-belt result, and this model scales the car's own rate instead of importing either.
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    const auto& corner = car->corners.front();
    const auto pressure = corner.tyre.pressure;

    auto state = TyreState{};
    seedTyreGasAtIdealPressure(pressure, state);
    const auto atIdeal = corner.tireVerticalRate * tyrePressureVerticalRateScale(pressure, state);

    REQUIRE(atIdeal == Catch::Approx(corner.tireVerticalRate).epsilon(1e-9));
    REQUIRE(atIdeal == Catch::Approx(8890.0 * 34.0).epsilon(0.02));

    // At the cold pressure the two still agree, which is what says the *shape* matches and not just
    // the one point they were referenced at.
    state.gasTemperature = pressure.coldReferenceTemperature;
    const auto atCold = corner.tireVerticalRate * tyrePressureVerticalRateScale(pressure, state);
    REQUIRE(atCold == Catch::Approx(8890.0 * 28.0).epsilon(0.02));
}

TEST_CASE("grip is multiplied in one place and pressure contributes exactly nothing yet",
          "[physics][tyre][pressure]")
{
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    const auto& tyre = car->corners.front().tyre;

    auto state = TyreState{};
    seedTyreGasAtIdealPressure(tyre.pressure, state);

    // Both off: exactly one, which is what lets the caller multiply unconditionally.
    REQUIRE(tyreStateGrip(tyre, state, false, false) == 1.0);

    // Pressure on, alone: still exactly one. **This case exists to keep an inert factor honest.**
    // Both Magic Formula pressure coefficients are zero because no tyre-independent pair exists to
    // put there — the campaign they were fitted on gives a curvature of −2.25, −1.62 and **+0.47**
    // on three passenger car tyres at this model's own nominal load, changing sign on the widest of
    // them, and a Goodyear Flat-Trac study of a 255/45 ZR19 finds the lateral effect *"not very
    // dominant"* and publishes a peak-grip adaptation without it. The day this car gets a pair of its
    // own, this case fails and says so.
    REQUIRE(tyre.pressure.gripPressureLinear == 0.0);
    REQUIRE(tyre.pressure.gripPressureQuadratic == 0.0);
    REQUIRE(tyreStateGrip(tyre, state, false, true) == 1.0);

    // And **exactly one at every pressure, not only at the ideal**, which is the whole difference
    // between this coupling and the other two: a zero times a finite number is a zero, so there is
    // no round trip to lose a bit in. Cold, hot and absurd.
    for (const auto celsius : {-20.0, 0.0, 31.5, 61.2, 120.0})
    {
        state.gasTemperature = celsius;
        REQUIRE(tyrePressureGripScale(tyre.pressure, state) == 1.0);
        REQUIRE(tyreStateGrip(tyre, state, false, true) == 1.0);
    }

    // Thermal on: the curve at the core, and the two multiply rather than one winning.
    seedTyreGasAtIdealPressure(tyre.pressure, state);
    state.coreTemperature = 20.0;
    const auto thermalOnly = tyreStateGrip(tyre, state, true, false);
    REQUIRE(thermalOnly < 1.0);
    REQUIRE(tyreStateGrip(tyre, state, true, true) == Catch::Approx(thermalOnly));
}

TEST_CASE("a stated pressure-grip pair is the Magic Formula's own, and it is a bound rather than a model",
          "[physics][tyre][pressure]")
{
    // The form is sourced even though the numbers are not: Besselink, Schmeitz & Pacejka, *An improved
    // Magic Formula/Swift tyre model that can handle inflation pressure changes*, Vehicle System
    // Dynamics 48(sup1):337, 2010, multiplies a peak friction coefficient by `1 + p3·dpi + p4·dpi²`
    // against `dpi = (P − P_ideal)/P_ideal`. This case pins the arithmetic against a hand figure so
    // that a session which does state a pair knows what it is stating.
    auto pressure = golfPressure();
    pressure.gripPressureLinear = 0.098;
    pressure.gripPressureQuadratic = -1.615;

    auto state = TyreState{};
    seedTyreGasAtIdealPressure(pressure, state);

    // At the ideal, `dpi` is zero and both terms vanish however large they are.
    REQUIRE(tyrePressureGripScale(pressure, state) == Catch::Approx(1.0).epsilon(1e-9));

    // Ten per cent under it — which is where this car's rear tyres finish a lap — the 205/55 R16's
    // own fit is worth 2.6% of grip. Checked by hand: 1 + 0.098·(−0.1) − 1.615·0.01.
    state.gasTemperature = gasTemperatureAtGauge(pressure, 0.9 * pressure.idealPressure);
    REQUIRE(psi(tyrePressureAt(pressure, state)) == Catch::Approx(30.6).epsilon(1e-3));
    REQUIRE(tyrePressureGripScale(pressure, state) == Catch::Approx(1.0 + 0.098 * -0.1 - 1.615 * 0.01).epsilon(1e-6));

    // And the floor holds a nonsense pair off the bottom of the world rather than letting a
    // downward parabola hand the solver a negative or a zero grip.
    pressure.gripPressureQuadratic = -400.0;
    REQUIRE(tyrePressureGripScale(pressure, state) == 0.1);
}

TEST_CASE("the cavity air follows the carcass rather than leading it", "[physics][tyre][pressure]")
{
    const auto pressure = golfPressure();
    const auto ambient = ambientAt(20.0, 19.0);

    auto state = TyreState{};
    state.carcassTemperature = 60.0;
    state.gasTemperature = 20.0;

    // No brake model, so no rim term: the gas sees the liner alone.
    const auto input = TyreThermalInput{.wheelTemperature = 0.0, .wheelConductance = 0.0, .ambient = ambient};

    // One second first, which is where the time constant is legible.
    for (auto step = 0; step < 360; step++)
    {
        stepTyreGas(pressure, state, input, tick);
    }

    // **Seconds, not minutes, and not tenths of a second either.** The air's heat capacity is about
    // 70 J/K against a liner conductance of 9 W/K, so the time constant is close to eight seconds and
    // one second of a 40 °C gap is worth about an eighth of it. The first draft of this case asserted
    // a ten-second answer after one second of stepping and failed — the model was right and the
    // expectation was not.
    REQUIRE(state.gasTemperature > 23.0);
    REQUIRE(state.gasTemperature < 28.0);

    // Ten seconds is most of the way there, and it never overshoots its neighbour.
    for (auto step = 0; step < 3240; step++)
    {
        stepTyreGas(pressure, state, input, tick);
    }

    REQUIRE(state.gasTemperature > 45.0);
    REQUIRE(state.gasTemperature < 60.0);

    // And it cools the same way. A tyre parked after a stint gives its air back to the carcass.
    state.carcassTemperature = 20.0;
    for (auto step = 0; step < 7200; step++)
    {
        stepTyreGas(pressure, state, input, tick);
    }

    REQUIRE(state.gasTemperature < 25.0);
    REQUIRE(state.gasTemperature > 20.0);
}

TEST_CASE("the pressure model is inert with its switch off and at equilibrium on its ideal pressure",
          "[physics][tyre][pressure][golf][parity]")
{
    // **This proof is weaker than the thermal model's and the reason is worth stating, because the
    // first draft of this case assumed it would be the same.**
    //
    // The thermal model is inert *over a range*: its grip curve is flat at exactly 1.00 across a
    // 20 °C plateau, so a tread anywhere inside it multiplies by a literal 1.0 and the car is
    // byte-identical however the temperature wanders. **Pressure has no plateau.** Every coupling is
    // monotonic in pressure, so any drift in the gas temperature moves the car, and the multiplier
    // is a computed ratio rather than a tabulated 1.00 — a gas-law round trip through a stored
    // double is not bit-exact even at the seed.
    //
    // So the inert configuration is a car in **thermal equilibrium** at its ideal pressure: nothing
    // heating the carcass, no rim pulling the air, and the gas already sitting where its neighbours
    // are. That is a narrower claim than the thermal model's and it is the honest one.
    const auto guard = JoltGuard{};
    const auto world = flatPlate();

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto seedTemperature = tyreGasTemperatureAtIdealPressure(built->corners.front().tyre.pressure);

    auto off = built.value();
    off.tyreThermal = false;
    off.brakeThermal = false;
    off.tyrePressure = false;

    auto on = off;
    on.tyrePressure = true;

    const auto roll = [&](const VehicleSetup& setup, const double gasSeed)
    {
        auto state = VehicleState{};
        settle(setup, state, world);
        seedTyreTemperatures(state, gasSeed);

        auto input = VehicleInput{};
        input.throttle = 0.4;

        for (auto step = 0; step < 1200; step++)
        {
            REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick).has_value());
        }

        return state;
    };

    const auto reference = roll(off, seedTemperature);
    const auto equilibrium = roll(on, seedTemperature);

    SECTION("the air does not move off its seed, which is this proof's precondition")
    {
        const auto pressure = golfPressure();
        for (const auto& corner : equilibrium.corners)
        {
            CAPTURE(corner.tyre.gasTemperature);
            REQUIRE(corner.tyre.gasTemperature == Catch::Approx(seedTemperature).margin(1e-6));
            REQUIRE(tyrePressureAt(pressure, corner.tyre) == Catch::Approx(pressure.idealPressure).margin(1.0));
        }
    }

    SECTION("and the car is in the same place to a part in 1e10, which is as exact as this can be")
    {
        // **Not `==`, deliberately, and the tolerance is stated once here rather than discovered
        // later.** The multiplier is 1.0 to within a unit in the last place and not exactly 1.0,
        // because a gas-law round trip through a stored double cannot come back bit-exact. Twenty
        // seconds of a rolling contact amplifies that to about a part in a million — six hundredths
        // of a millimetre in a hundred metres. Asserting byte-identity here would be asserting
        // something the arithmetic cannot deliver, and loosening a threshold *after* it failed is
        // exactly how a gate stops being a gate.
        REQUIRE(equilibrium.chassis.position.z == Catch::Approx(reference.chassis.position.z).epsilon(1e-5));
        REQUIRE(equilibrium.chassis.position.y == Catch::Approx(reference.chassis.position.y).epsilon(1e-5));
        REQUIRE(equilibrium.chassis.linearVelocity.z ==
                Catch::Approx(reference.chassis.linearVelocity.z).epsilon(1e-5));
    }

    SECTION("and it is not vacuous: a cold car is genuinely a different car")
    {
        const auto cold = roll(on, 31.5);

        // A tyre on a 31.5 °C track is at 29.7 psi against an ideal of 34, so it is a **softer**
        // spring and a draggier one. It has to end up somewhere else, and by much more than the
        // hundredth of a micron above.
        REQUIRE(std::abs(cold.chassis.position.z - reference.chassis.position.z) > 1e-4);
    }
}

TEST_CASE("a cold car starts soft, and the rim holds its air below the carcass",
          "[physics][tyre][pressure][golf]")
{
    const auto guard = JoltGuard{};
    const auto world = flatPlate();

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto setup = built.value();
    setup.tyreThermal = true;
    setup.tyrePressure = true;

    const auto pressure = setup.corners.front().tyre.pressure;

    auto state = VehicleState{};
    settle(setup, state, world);

    // A car out of a garage on this scene's track: every node at the track's temperature, air
    // included. **That is what a cold car is** — the air has been sitting in the same tyre as the
    // rubber.
    seedTyreTemperatures(state, 31.5);

    const auto start = tyrePressureAt(pressure, state.corners.front().tyre);
    REQUIRE(psi(start) == Catch::Approx(29.67).margin(0.05));
    REQUIRE(start < pressure.idealPressure);

    auto input = VehicleInput{};
    input.throttle = 0.5;

    for (auto step = 0; step < 7200; step++)
    {
        REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick).has_value());
    }

    // **Twenty seconds does not warm a tyre's air, and the first draft of this case assumed it
    // would.** Two things hold it down and both are real: the carcass has barely moved in twenty
    // seconds, and the **rim is a cold sink on the other side of the cavity** — 5.4 W/K to a wheel
    // sitting at ambient against 12 W/K to the carcass, so the air equilibrates *below* the rubber
    // rather than with it. That is the same sign stage 3 of the brake work measured for the tread,
    // now for a second path and a second reason.
    const auto warmed = tyrePressureAt(pressure, state.corners.front().tyre);
    REQUIRE(state.corners.front().tyre.gasTemperature < state.corners.front().tyre.carcassTemperature);

    // Still well under its ideal, which is the finding this stage predicted before it was built: AC's
    // ideal pressure belongs to a tyre whose air is at 61 °C, and a road car's air is not.
    REQUIRE(warmed < pressure.idealPressure);

    // And the mechanism, asserted directly rather than waited for: hold the carcass hot and the air
    // follows it up, which is what makes this a pressure model rather than a constant.
    for (auto& corner : state.corners)
    {
        corner.tyre.carcassTemperature = 80.0;
    }

    for (auto step = 0; step < 7200; step++)
    {
        REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick).has_value());
    }

    REQUIRE(tyrePressureAt(pressure, state.corners.front().tyre) > warmed);
    REQUIRE(state.corners.front().tyre.gasTemperature > 31.5);
}
