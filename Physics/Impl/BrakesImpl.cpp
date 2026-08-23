// Brake bodies. Declarations are in Api/Brakes.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces
// an object and no BMI, so nothing imports it and nothing rebuilds when it changes. Measurements and
// the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cstdint>

module raceengine.physics;

namespace
{

// Local rather than imported: this unit names nothing else from anywhere, and a global module
// fragment that reaches for a header to get one constant is the cost `docs/build-times.md` is about.
constexpr auto pi = 3.14159265358979323846;

[[nodiscard]] double circleArea(const double diameter)
{
    return 0.25 * pi * diameter * diameter;
}

} // namespace

namespace raceengine
{

[[nodiscard]] FrictionCouple lowMetallicOnCastIron()
{
    return FrictionCouple{.coefficient = 0.40};
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
