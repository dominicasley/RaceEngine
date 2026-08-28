// The ambient conditions' bodies. Declarations are in Physics/Api/Ambient.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — so it produces
// an object and no BMI and nothing rebuilds when a correlation is edited. docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>

module raceengine.physics;

namespace raceengine
{

namespace
{

// Zero Celsius in kelvin, and the two textbook constants Sutherland's law needs.
constexpr auto absoluteZero = 273.15;
constexpr auto sutherlandCoefficient = 1.458e-6;
constexpr auto sutherlandTemperature = 110.4;
constexpr auto seaLevelPressure = 101325.0;
constexpr auto airGasConstant = 287.05;

} // namespace

[[nodiscard]] double TemperatureCurve::at(const double temperature) const
{
    if (count == 0)
    {
        return 1.0;
    }

    if (count == 1 || temperature <= celsius[0])
    {
        return multiplier[0];
    }

    const auto last = count - 1;
    if (temperature >= celsius[last])
    {
        return multiplier[last];
    }

    for (auto index = std::size_t{1}; index <= last; index++)
    {
        if (temperature <= celsius[index])
        {
            const auto span = celsius[index] - celsius[index - 1];

            return span > 0.0 ? multiplier[index - 1] + (multiplier[index] - multiplier[index - 1]) *
                                                            (temperature - celsius[index - 1]) / span
                              : multiplier[index - 1];
        }
    }

    return multiplier[last];
}

[[nodiscard]] double airConductivity(const double celsius)
{
    const auto kelvin = std::max(celsius + absoluteZero, 1.0);

    return 0.02624 * std::pow(kelvin / 300.0, 0.8646);
}

[[nodiscard]] double airKinematicViscosity(const double celsius)
{
    const auto kelvin = std::max(celsius + absoluteZero, 1.0);

    const auto dynamic = sutherlandCoefficient * std::pow(kelvin, 1.5) / (kelvin + sutherlandTemperature);
    const auto density = seaLevelPressure / (airGasConstant * kelvin);

    return dynamic / density;
}

[[nodiscard]] double trackTemperatureInSun(const double airCelsius, const double sunElevationDegrees)
{
    // Below the horizon the road is not being heated by anything, and it does not go *below* air
    // temperature here — a clear night genuinely does take tarmac below the air by radiating to the
    // sky, and modelling that needs a sky temperature nobody in this project has sourced. So the
    // floor is the air, and the omission is stated rather than hidden behind a negative number.
    const auto elevation = std::sin(std::max(sunElevationDegrees, 0.0) * raceengine::degreesToRadians);

    constexpr auto irradiance = 1000.0;
    constexpr auto albedo = 0.12;
    constexpr auto surfaceCoefficient = 25.0;

    return airCelsius + irradiance * (1.0 - albedo) * elevation / surfaceCoefficient;
}

[[nodiscard]] AmbientConditions ambientAt(const double airCelsius, const double sunElevationDegrees)
{
    return AmbientConditions{.airTemperature = airCelsius,
                             .trackTemperature = trackTemperatureInSun(airCelsius, sunElevationDegrees)};
}

} // namespace raceengine
