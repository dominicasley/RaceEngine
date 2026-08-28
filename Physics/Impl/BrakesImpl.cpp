// Brake bodies. Declarations are in Api/Brakes.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces
// an object and no BMI, so nothing imports it and nothing rebuilds when it changes. Measurements and
// the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>

module raceengine.physics;

namespace
{

// Local rather than imported: this unit names nothing else from anywhere, and a global module
// fragment that reaches for a header to get one constant is the cost `docs/build-times.md` is about.
constexpr auto pi = 3.14159265358979323846;

// Radiation's two constants, at file scope because three bodies need them now: the disc's own
// radiation to the sky, the disc's exchange with the wheel, and the wheel's to the sky.
constexpr auto stefanBoltzmann = 5.670374419e-8;
constexpr auto absoluteZero = 273.15;

[[nodiscard]] double circleArea(const double diameter)
{
    return 0.25 * pi * diameter * diameter;
}

} // namespace

namespace raceengine
{

[[nodiscard]] FrictionCouple lowMetallicOnCastIron()
{
    // **The plateau is the pad's own specification and the tail is borrowed. The curve says so by its
    // shape, and this comment draws the line the code cannot.**
    //
    // SAE J866 marks a lining with two letters for two temperature bands — the first over 200-400 °F
    // (93-204 °C) and the second over 300-650 °F (149-343 °C) — and an OE passenger pad is marked
    // `FF`, which is 0.35-0.45 in *both*. That is a statement, by the standard the pad is sold
    // against, that it does not fade up to about 343 °C. Hence 1.00 flat to 350.
    //
    // Above it, fade is real and is what SAE J2522's Fade I procedure measures. Published runs of it
    // show friction falling from 0.32-0.34 early to **0.24-0.28** late, at disc temperatures in the
    // high hundreds — a loss of about a third. Against this couple's nominal 0.40 that is a
    // multiplier near 0.65, which is where the tail is put. **It is not this pad's measurement** and
    // it is the same standing as the tyre's temperature-curve tails: used, and flagged.
    //
    // The cold end is deliberately flat rather than reduced. J866 rates the pad from 93 °C and says
    // nothing below it, and an OE road pad is designed to work from cold — asserting a cold-friction
    // loss would be inventing one.
    return FrictionCouple{.coefficient = 0.40,
                          .fade = TemperatureCurve{.count = 8,
                                                   .celsius = {{0.0, 100.0, 200.0, 350.0, 450.0, 550.0, 650.0, 800.0}},
                                                   .multiplier = {{1.00, 1.00, 1.00, 1.00, 0.90, 0.72, 0.65, 0.60}}}};
}

[[nodiscard]] double frictionAtTemperature(const FrictionCouple& couple, const double celsius)
{
    return std::max(couple.coefficient, 0.0) * couple.fade.at(celsius);
}

[[nodiscard]] double sweptFaceArea(const BrakeHardware& brake)
{
    const auto outer = std::max(0.5 * brake.discDiameter - brake.padOuterClearance, 0.0);
    const auto inner = std::max(outer - brake.padRadialHeight, 0.0);

    return pi * (outer * outer - inner * inner);
}

[[nodiscard]] BrakeThermal brakeThermalOf(const BrakeHardware& brake)
{
    const auto face = sweptFaceArea(brake);

    // Two external faces convect and radiate. A vented disc's internal passages roughly double the
    // area air passes over — that is what the vanes are for — and they radiate almost entirely to
    // each other, so they join the convection area and not the radiation one.
    const auto faces = 2.0 * face;

    return BrakeThermal{.heatCapacity = std::max(brake.discMass, 0.0) * castIronSpecificHeat,
                        .convectionArea = brake.discVented ? 2.0 * faces : faces,
                        .radiationArea = faces,
                        .diameter = brake.discDiameter,
                        .couple = brake.couple};
}

[[nodiscard]] double discConvection(const BrakeThermal& thermal, const double airSpeed, const double celsius,
                                    const double airCelsius)
{
    const auto diameter = std::max(thermal.diameter, 1e-3);

    // At the film temperature, which is what the correlation asks for and is the mean of the surface
    // and the air. A disc at 500 °C is moving air whose properties are nothing like the ambient's.
    const auto film = 0.5 * (celsius + airCelsius);
    const auto conductivity = airConductivity(film);
    const auto viscosity = airKinematicViscosity(film);

    if (viscosity <= 0.0)
    {
        return 0.0;
    }

    const auto reynolds = std::abs(airSpeed) * diameter / viscosity;
    if (reynolds <= 0.0)
    {
        // A stationary car. Limpert's is a forced correlation and goes to zero with speed; still air
        // over a hot vertical plate is a textbook 5 to 10 W/(m²·K) and this is the low end, which is
        // the conservative choice for a cooling term and matches the tyre's own floor.
        return 5.0;
    }

    return std::max(0.04 * conductivity / diameter * std::pow(reynolds, 0.8), 5.0);
}

void stepBrakeThermal(const BrakeThermal& thermal, double& celsius, const BrakeThermalInput& input,
                      const double deltaTime)
{
    if (deltaTime <= 0.0 || thermal.heatCapacity <= 0.0)
    {
        return;
    }

    const auto air = input.ambient.airTemperature;

    const auto convection =
        discConvection(thermal, input.airSpeed, celsius, air) * std::max(thermal.convectionArea, 0.0);

    // **Radiation matters here and it did not on the tyre**, because it goes as the fourth power and
    // a disc runs at 300-600 °C where a tread runs at 50-90. Linearised about the current temperature
    // *within the tick* — `h_rad = eps·sigma·(Ts² + Ta²)(Ts + Ta)` is the exact secant of the fourth
    // power between the two, so this is not an approximation of the physics, only of how far the
    // temperature moves before it is re-evaluated. At 360 Hz that is a hundredth of a degree.
    const auto surface = celsius + absoluteZero;
    const auto ambient = air + absoluteZero;
    const auto radiation = thermal.emissivity * stefanBoltzmann * std::max(thermal.radiationArea, 0.0) *
                           (surface * surface + ambient * ambient) * (surface + ambient);

    // Everything the wheel takes: conduction through the hat and the joint, plus the radiation that
    // crosses the gap. The caller computed it once so that the two nodes cannot disagree.
    const auto toWheel = std::max(input.wheelConductance, 0.0);

    // **The radiation is partitioned, not duplicated** — whatever share of the disc's faces looks at
    // the wheel is spent there and not against the sky.
    //
    // **And the partition is conditional on the wheel actually being in the path, which is a defect
    // this cost.** A disc that stated a share and was then stepped with no coupling — a fixture
    // calling `stepBrakeThermal` directly, which is most of `BrakeThermalTests` — lost that share of
    // its radiation to nothing at all and ran hotter than a disc with no wheel. It showed up as a
    // runaway case reaching 2227 °C where it had reached under 2000. A partition applied on one side
    // only is not a partition; it is a deletion.
    const auto share = toWheel > 0.0 ? std::clamp(thermal.wheelRadiationShare, 0.0, 1.0) : 0.0;
    const auto toSky = radiation * (1.0 - share);

    const auto conductance = convection + toSky + toWheel;
    const auto generation = std::max(input.frictionPower, 0.0) * std::clamp(thermal.heatToDisc, 0.0, 1.0);

    if (conductance <= 1e-12)
    {
        celsius += generation * deltaTime / thermal.heatCapacity;
        return;
    }

    // The equilibrium this disc is falling towards is the conductance-weighted mean of the far ends
    // of every path leaving it, plus what it is making.
    //
    // **The no-wheel arm is written out rather than left to fall out of the general one**, and that
    // is not tidiness. The two are the same in exact arithmetic and not in IEEE arithmetic, and every
    // figure stages 1 and 2 measured — 237 °C over ten stops, 557 on a lap, a first stop identical to
    // the model switched off — was taken through `air + generation / conductance`. A disc with no
    // wheel still reaches it, expression for expression.
    const auto equilibrium =
        toWheel > 0.0 ? (generation + (convection + toSky) * air + toWheel * input.wheelTemperature) / conductance
                      : air + generation / conductance;

    celsius = equilibrium + (celsius - equilibrium) * std::exp(-conductance * deltaTime / thermal.heatCapacity);
}

// --- the wheel the disc is bolted inside -------------------------------------------------------

[[nodiscard]] double hatConductance(const BrakeHardware& brake, const WheelHardware& wheel)
{
    // The swept ring's inner radius, which is where the heat is made and is the same two numbers
    // `sweptFaceArea` and `effectiveRadius` are built from.
    const auto outer = std::max(0.5 * brake.discDiameter - brake.padOuterClearance, 0.0);
    const auto ring = std::max(outer - brake.padRadialHeight, 0.0);

    const auto bolt = std::max(wheel.boltCircleRadius, 1e-3);
    const auto wall = std::max(wheel.hatWallThickness, 1e-4);
    const auto height = std::max(brake.hatHeight, 0.0);

    if (ring <= bolt || castIronConductivity <= 0.0)
    {
        return 0.0;
    }

    // Two resistances in series and both are pure geometry. The cylindrical wall carries the heat
    // axially back from the ring; the flange then carries it radially inward to the bolt circle,
    // which is the standard `ln(r_o/r_i) / (2·pi·k·t)` for an annulus.
    const auto axial = height / (castIronConductivity * 2.0 * pi * ring * wall);
    const auto radial = std::log(ring / bolt) / (2.0 * pi * castIronConductivity * wall);

    const auto resistance = axial + radial;

    return resistance > 0.0 ? 1.0 / resistance : 0.0;
}

[[nodiscard]] WheelThermal wheelThermalOf(const WheelHardware& wheel, const BrakeHardware& brake)
{
    // **A wheel of no mass is a car that states no wheel**, and it returns a node every step treats
    // as absent — which is what makes the whole of stage 3 inert for every car but the Golf.
    if (wheel.mass <= 0.0)
    {
        return WheelThermal{};
    }

    const auto diameter = std::max(wheel.diameter, 1e-3);
    const auto faces = 2.0 * circleArea(diameter);

    // What the disc is already radiating onto, taken out so it is not spent twice.
    const auto shared = 2.0 * sweptFaceArea(brake) * std::clamp(wheel.discRadiationShare, 0.0, 1.0);

    // The hat's neck in series with the bolted joint. **The neck is the smaller by a factor of
    // thirty**, which is why the joint's own thirteen-fold uncertainty is worth 3% of the answer.
    const auto neck = hatConductance(brake, wheel);
    const auto joint = std::max(wheel.jointConductance, 0.0);
    const auto conduction = (neck > 0.0 && joint > 0.0) ? 1.0 / (1.0 / neck + 1.0 / joint) : 0.0;

    return WheelThermal{.heatCapacity = wheel.mass * castAluminiumSpecificHeat,
                        .convectionArea = faces,
                        .radiationArea = std::max(faces - shared, 0.0),
                        .diameter = diameter,
                        .emissivity = wheel.emissivity,
                        .toDisc = conduction,
                        .toTyre = std::max(wheel.toTyre, 0.0)};
}

[[nodiscard]] BrakeThermal brakeThermalOf(const BrakeHardware& brake, const WheelHardware& wheel)
{
    auto thermal = brakeThermalOf(brake);
    thermal.wheelRadiationShare = wheel.mass > 0.0 ? std::clamp(wheel.discRadiationShare, 0.0, 1.0) : 0.0;

    return thermal;
}

[[nodiscard]] double discToWheelRadiation(const BrakeThermal& disc, const WheelThermal& wheel, const double discCelsius,
                                          const double wheelCelsius)
{
    const auto area = std::max(disc.radiationArea, 0.0) * std::clamp(disc.wheelRadiationShare, 0.0, 1.0);
    if (area <= 0.0 || wheel.heatCapacity <= 0.0)
    {
        return 0.0;
    }

    // Two close, nearly parallel faces re-radiating at each other, which is not the same problem as
    // one body facing a sky: the series emissivity `1/(1/e1 + 1/e2 - 1)` is what accounts for the
    // radiation each face sends back. Oxidised iron at 0.8 against painted aluminium at 0.85 gives
    // 0.701, so the pair is worth about 12% less than the disc alone would suggest.
    const auto first = std::clamp(disc.emissivity, 1e-3, 1.0);
    const auto second = std::clamp(wheel.emissivity, 1e-3, 1.0);
    const auto effective = 1.0 / (1.0 / first + 1.0 / second - 1.0);

    const auto hot = discCelsius + absoluteZero;
    const auto cold = wheelCelsius + absoluteZero;

    return effective * stefanBoltzmann * area * (hot * hot + cold * cold) * (hot + cold);
}

[[nodiscard]] double discToWheelCoupling(const BrakeThermal& disc, const WheelThermal& wheel, const double discCelsius,
                                         const double wheelCelsius)
{
    if (wheel.heatCapacity <= 0.0)
    {
        return 0.0;
    }

    return std::max(wheel.toDisc, 0.0) + discToWheelRadiation(disc, wheel, discCelsius, wheelCelsius);
}

[[nodiscard]] double wheelConvection(const WheelThermal& thermal, const double airSpeed, const double celsius,
                                     const double airCelsius)
{
    const auto diameter = std::max(thermal.diameter, 1e-3);

    const auto film = 0.5 * (celsius + airCelsius);
    const auto conductivity = airConductivity(film);
    const auto viscosity = airKinematicViscosity(film);

    if (viscosity <= 0.0)
    {
        return 0.0;
    }

    const auto reynolds = std::abs(airSpeed) * diameter / viscosity;
    if (reynolds <= 0.0)
    {
        return 5.0;
    }

    return std::max(0.04 * conductivity / diameter * std::pow(reynolds, 0.8), 5.0);
}

void stepWheelThermal(const WheelThermal& thermal, double& celsius, const WheelThermalInput& input,
                      const double deltaTime)
{
    if (deltaTime <= 0.0 || thermal.heatCapacity <= 0.0)
    {
        return;
    }

    const auto air = input.ambient.airTemperature;

    const auto convection =
        wheelConvection(thermal, input.airSpeed, celsius, air) * std::max(thermal.convectionArea, 0.0);

    // **The wheel radiates too, and leaving it out is the mistake the tyre made.** A wheel at 90 °C
    // is worth about 7 W/(m²·K) of radiation against 22 to 39 of forced convection at speed — 6 to
    // 10% — and it is the *whole* of the cooling when the car stops, which is the case a soak
    // measures. Linearised about the current temperature exactly as the disc's is.
    const auto surface = celsius + absoluteZero;
    const auto ambient = air + absoluteZero;
    const auto radiation = std::clamp(thermal.emissivity, 0.0, 1.0) * stefanBoltzmann *
                           std::max(thermal.radiationArea, 0.0) * (surface * surface + ambient * ambient) *
                           (surface + ambient);

    const auto toDisc = std::max(input.discConductance, 0.0);
    const auto toTyre = std::max(input.tyreConductance, 0.0);

    const auto conductance = convection + radiation + toDisc + toTyre;
    if (conductance <= 1e-12)
    {
        return;
    }

    // Nothing is generated here: a wheel makes no heat of its own. It falls towards the
    // conductance-weighted mean of the three places it is connected to.
    const auto weighted =
        (convection + radiation) * air + toDisc * input.discTemperature + toTyre * input.tyreTemperature;
    const auto equilibrium = weighted / conductance;

    celsius = equilibrium + (celsius - equilibrium) * std::exp(-conductance * deltaTime / thermal.heatCapacity);
}

[[nodiscard]] double pistonArea(const BrakeHardware& brake)
{
    return static_cast<double>(brake.pistons) * circleArea(std::max(brake.pistonBore, 0.0));
}

[[nodiscard]] double effectiveRadius(const BrakeHardware& brake)
{
    const auto outer = std::max(0.5 * brake.discDiameter - brake.padOuterClearance, 0.0);
    const auto inner = std::max(outer - brake.padRadialHeight, 0.0);

    return 0.5 * (outer + inner);
}

[[nodiscard]] double torquePerPressure(const BrakeHardware& brake)
{
    return pistonArea(brake) * std::max(brake.couple.coefficient, 0.0) * effectiveRadius(brake) *
           static_cast<double>(brake.frictionFaces);
}

[[nodiscard]] double frontBrakeShare(const BrakeHardware& front, const BrakeHardware& rear)
{
    const auto frontTorque = torquePerPressure(front);
    const auto total = frontTorque + torquePerPressure(rear);

    return total > 0.0 ? frontTorque / total : 0.0;
}

[[nodiscard]] double proportionedPressure(const ProportioningValve& valve, const double inlet)
{
    const auto knee = std::max(valve.kneePressure, 0.0);
    const auto pressure = std::max(inlet, 0.0);

    if (pressure <= knee)
    {
        return pressure;
    }

    return knee + std::clamp(valve.slope, 0.0, 1.0) * (pressure - knee);
}

[[nodiscard]] double masterCylinderArea(const BrakeHydraulics& hydraulics)
{
    return circleArea(std::max(hydraulics.masterCylinderBore, 0.0));
}

[[nodiscard]] double boosterAssistLimit(const BrakeHydraulics& hydraulics)
{
    return std::max(hydraulics.boosterVacuum, 0.0) * circleArea(std::max(hydraulics.boosterDiaphragm, 0.0));
}

[[nodiscard]] double brakeLinePressure(const BrakeHydraulics& hydraulics, const double pedal)
{
    const auto area = masterCylinderArea(hydraulics);
    if (area <= 0.0)
    {
        return 0.0;
    }

    // The driver's foot, through the pedal's lever, is the input the servo works on.
    const auto input =
        std::clamp(pedal, 0.0, 1.0) * std::max(hydraulics.maxPedalForce, 0.0) * std::max(hydraulics.pedalRatio, 0.0);

    // And the servo adds what it can, up to what its diaphragm and its depression allow. Past that
    // the gain is 1: everything further is the driver's own leg, which is exactly what a car with a
    // dead servo is and why one still stops.
    const auto assist = std::min(std::max(hydraulics.boostRatio - 1.0, 0.0) * input, boosterAssistLimit(hydraulics));

    return (input + assist) / area;
}

[[nodiscard]] double boosterRunoutPedal(const BrakeHydraulics& hydraulics)
{
    const auto gain = std::max(hydraulics.boostRatio - 1.0, 0.0);
    const auto lever = std::max(hydraulics.maxPedalForce, 0.0) * std::max(hydraulics.pedalRatio, 0.0);

    if (gain <= 0.0 || lever <= 0.0)
    {
        return 0.0;
    }

    return boosterAssistLimit(hydraulics) / gain / lever;
}

[[nodiscard]] double brakeTorqueAtPedal(const BrakeHardware& brake, const BrakeHydraulics& hydraulics,
                                        const double pedal)
{
    return torquePerPressure(brake) * brakeLinePressure(hydraulics, pedal);
}

[[nodiscard]] double peakBrakeTorque(const BrakeHardware& brake, const BrakeHydraulics& hydraulics)
{
    return brakeTorqueAtPedal(brake, hydraulics, 1.0);
}

} // namespace raceengine
