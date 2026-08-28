module;

#include <array>
#include <cstddef>

export module raceengine.physics:Ambient;

namespace raceengine
{

// What the air and the road are at, and the properties of the air that follow from it.
//
// **Nothing in this model knew the weather until 2026-08-28.** A tyre that carries a temperature has
// to have something to lose heat to, and the number it loses it to is a property of the *scene* and
// not of the car — the same argument that put the sun's elevation in one place and derived the sky,
// the probes and the fog from it. So this is one number with everything else derived from it, and
// `OSR_AIR_TEMP` is its override.
//
// Celsius throughout, deliberately, against the SI rule the rest of Physics keeps. Every source this
// model is calibrated against states tyre and track temperatures in degrees Celsius — Michelin's
// bulletin, AC's curve, every trackside reading a driver has ever quoted — and converting at the
// boundary would put a units conversion between the data and the number a test asserts. What the
// unit rule is *for* is the quantities that multiply each other; the two places absolute temperature
// is genuinely needed (air's conductivity and its viscosity) convert inside their own functions.
export struct AmbientConditions
{
    // The air the car is driving through.
    double airTemperature = 20.0;
    // The road surface. Warmer than the air whenever the sun is on it — see `trackTemperatureInSun`,
    // which is what a scene derives this with.
    double trackTemperature = 20.0;
};

// How many points a temperature curve may carry. Sixteen because the curves this project ships have
// thirteen points at most, and because a fixed extent is what keeps the structs that carry one
// copyable without an allocation — they are copied per wheel and per corner every tick.
export inline constexpr std::size_t temperatureCurvePoints = 16;

// A multiplier against temperature, as a piecewise-linear table in degrees Celsius.
//
// **It lives here rather than with either of its consumers because it has two.** The tyre reads grip
// against tread core temperature and the brake reads pad friction against disc temperature, and they
// are the same object: a table in Celsius returning a dimensionless multiplier, flat where a
// specification says the thing does not change and falling where it does. Putting it beside the
// weather is the least surprising home for a curve indexed by a temperature. Outside the stated
// range the end value is held rather than extrapolated, which is `Curve`'s rule and is right for the
// same reason: a tyre asked about a temperature past its last measured point should keep the last
// answer, not run off to zero or to infinity.
export struct TemperatureCurve
{
    std::size_t count = 0;
    std::array<double, temperatureCurvePoints> celsius{};
    std::array<double, temperatureCurvePoints> multiplier{};

    // 1.0 for an empty curve, which is the model each consumer had before it carried one and is what
    // makes a car that states no curve identical to the bit.
    [[nodiscard]] double at(const double temperature) const;
};

// Air's thermal conductivity, W/(m·K), at a stated temperature in degrees Celsius.
//
// The standard correlation `k = 0.02624 · (T/300)^0.8646` with T in kelvin, which reproduces the
// tabulated values across the whole range a track ever sees. It is wanted at the *film* temperature
// — the mean of the surface and the air — which is what the convection correlation asks for and is
// the caller's business to supply.
export [[nodiscard]] double airConductivity(const double celsius);

// Air's kinematic viscosity, m²/s, at a stated temperature in degrees Celsius and at sea level.
//
// Sutherland's law for the dynamic viscosity over the ideal-gas density, both textbook, so the
// Reynolds number the convection correlation is evaluated at has no fitted constant anywhere in it.
export [[nodiscard]] double airKinematicViscosity(const double celsius);

// What a tarmac surface sits at, given the air temperature and the sun's elevation in degrees.
//
// **A stated surface energy balance rather than an offset somebody picked.** Clear-sky irradiance at
// the surface is about 1000 W/m² normal to the beam, aged asphalt reflects about 12% of it, and a
// horizontal surface receives it times the sine of the sun's elevation. What leaves is convection to
// the air and long-wave radiation to the sky, whose combined coefficient for a horizontal surface in
// light wind is about 25 W/(m²·K). So the rise is `1000 · (1 − 0.12) · sin(elevation) / 25`, which is
// 35 °C at the zenith, 11 °C at this scene's own 19-degree sun, and exactly zero at night.
//
// **Every constant in it is a textbook band and none is fitted**: the irradiance is the standard
// clear-sky figure, the albedo's published range for asphalt is 0.05 (fresh) to 0.20 (weathered),
// and the combined surface coefficient's is 15 to 30. What it produces — a track 10 to 20 °C over
// air in ordinary sunshine — is what trackside measurements report, which is the check on it rather
// than the source of it.
export [[nodiscard]] double trackTemperatureInSun(const double airCelsius, const double sunElevationDegrees);

// The pair, from the one number a scene states and the sun it already has.
export [[nodiscard]] AmbientConditions ambientAt(const double airCelsius, const double sunElevationDegrees);

} // namespace raceengine
