// VolumetricFog bodies. Declarations are in Graphics/Api/VolumetricFog.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>

module raceengine.graphics;

import :VolumetricFog;
import raceengine.graphics.models;

namespace raceengine
{

namespace
{

// How far the density profile's exponent may run either side of the reference height before it is
// held. Past this the medium is either so thin that nothing survives rounding or so thick that
// nothing survives the medium, and exp() of it stops being representable.
constexpr auto maximumFogExponent = 80.0f;

} // namespace

float fogOpticalDepth(const Fog& fog, const float originHeight, const float directionY, const float distance)
{
    if (!fog.enabled || fog.density <= 0.0f || distance <= 0.0f || fog.scaleHeight <= 0.0f)
    {
        return 0.0f;
    }

    const auto falloff = 1.0f / fog.scaleHeight;

    // The profile's exponent at each end of the ray, in scale heights above the reference, and the
    // climb between them. Everything below is stated in the two *ends* rather than in one end and a
    // climb, and that is a correction rather than a preference: clamping the climb alone reads as
    // no fog at all on exactly the ray it was meant to protect. A camera high above a shallow layer
    // looking steeply down has a tiny density at the eye and an enormous one at the far end, and the
    // product of an underflow and a clamped overflow is zero — a ray that should be completely
    // opaque coming back completely clear. Clamping each end instead bounds the *medium*, which is
    // a statement about the fog rather than about the ray's geometry.
    const auto nearExponent = (originHeight - fog.baseHeight) * falloff;
    const auto farExponent = nearExponent + distance * directionY * falloff;
    const auto climb = farExponent - nearExponent;

    const auto nearDensity = std::exp(-std::clamp(nearExponent, -maximumFogExponent, maximumFogExponent));
    const auto farDensity = std::exp(-std::clamp(farExponent, -maximumFogExponent, maximumFogExponent));

    // The integral is `density * distance * (1 - exp(-climb)) / climb`, which is the difference of
    // the two densities over the climb. At climb = 0 that is 0/0 with a limit of 1, and near zero it
    // is the difference of two nearly equal numbers over a small one — most of a float's mantissa.
    // A level ray is exactly zero and a level ray is what a driver spends the lap looking along, so
    // the series below is the common case rather than an edge one.
    if (std::abs(climb) > 1.0e-2f)
    {
        return std::max(fog.density * distance * (nearDensity - farDensity) / climb, 0.0f);
    }

    return std::max(fog.density * distance * nearDensity * (1.0f - climb * 0.5f + climb * climb * (1.0f / 6.0f)),
                    0.0f);
}

float fogTransmittance(const Fog& fog, const float originHeight, const float directionY, const float distance)
{
    return std::exp(-fogOpticalDepth(fog, originHeight, directionY, distance));
}

float henyeyGreenstein(const float cosTheta, const float anisotropy)
{
    const auto g = std::clamp(anisotropy, -0.95f, 0.95f);
    const auto gg = g * g;

    // The 1/4pi the published form carries is deliberately absent — see the declaration. What is
    // left is 1 everywhere when g is 0, which is what makes the coefficient beside it read as an
    // albedo rather than as an albedo times a solid angle.
    const auto denominator = 1.0f + gg - 2.0f * g * std::clamp(cosTheta, -1.0f, 1.0f);

    return (1.0f - gg) / std::max(std::pow(std::max(denominator, 0.0f), 1.5f), 1.0e-4f);
}

} // namespace raceengine
