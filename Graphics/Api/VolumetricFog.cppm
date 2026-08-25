export module raceengine.graphics:VolumetricFog;

import raceengine.graphics.models;

namespace raceengine
{

// The arithmetic the fog is built out of, stated once and away from any device so it can be read
// and tested on its own — the same standing this module gives `evaluateToneCurve`. The two scene
// shaders and the sky are the *second* implementation of these three functions, and a difference
// between them is a difference between what a test says the frame will look like and what it does.
//
// There is no glm and no std header in this unit's purview on purpose (docs/build-times.md): the
// closed form depends on the ray's height and the vertical component of its direction, and on
// nothing else about its geometry, so it is spelled in floats.

// The optical depth of a medium whose density falls exponentially with height, integrated along a
// ray. This is Íñigo Quílez's height-fog integral (https://iquilezles.org/articles/fog/), which is
// the analytic value of
//
//     integral from 0 to t of  density * exp(-(y(s) - baseHeight) / scaleHeight)  ds
//
// for a ray climbing linearly. `directionY` is the y component of a **normalised** direction, so a
// level ray is 0 and a ray straight up is 1; `distance` is how far along it to integrate, in world
// units.
//
// The degenerate case is the common one and it is handled rather than divided by: a level ray has
// `directionY` 0, where the published form divides by it. The limit is the constant-density answer,
// `density * exp(...) * distance`, and the series about it is what keeps a nearly-level ray — every
// ray down a straight — from losing precision on the difference of two numbers that are nearly
// equal.
export [[nodiscard]] float fogOpticalDepth(const Fog& fog, float originHeight, float directionY, float distance);

// What survives the medium: `exp(-opticalDepth)`. One minus it is the fraction of the ray that
// scattered or was absorbed, and — because the density is the derivative of the optical depth — it
// is also the exact value of the in-scattering integral for a medium nothing shadows. That identity
// is what lets the marched half and the analytic half be added without counting light twice; the
// brief has the derivation.
export [[nodiscard]] float fogTransmittance(const Fog& fog, float originHeight, float directionY, float distance);

// Henyey-Greenstein, normalised so that isotropic scattering is exactly 1 rather than 1/4pi. The
// engine works in relative radiance throughout, so a phase function that integrated to one over the
// sphere would drag a factor of 4pi into every in-scattering term and out of nothing else; stated
// this way, `anisotropy` 0 is "the medium does not care which way the light was going" and the term
// beside it is the medium's albedo, which is the reading a level wants.
//
// `cosTheta` is `dot(viewRayDirection, directionTowardsTheLight)` — the second of those is what
// `Light::position` already holds, so nothing has to be negated at the call site. A camera looking
// straight at the sun reads 1 and gets the forward lobe, which is the peak; looking away from it
// reads -1 and gets the backward one. Stated as the two vectors rather than as the scattering angle
// because the scattering angle is measured between the light's *travel* direction and the outgoing
// one, and both negations cancel — writing it out is how that stops being a sign to get wrong.
export [[nodiscard]] float henyeyGreenstein(float cosTheta, float anisotropy);

} // namespace raceengine
