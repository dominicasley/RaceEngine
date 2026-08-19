module;

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

export module raceengine.graphics:PhysicalCamera;

import :RenderContract;
import raceengine.graphics.models;

namespace raceengine
{

// The exposure triangle and the meter that closes it, stated away from any device so the numbers a
// frame is exposed with can be read and tested without one — the same split PostProcessing makes
// for the curve those numbers are then fed into.
//
// The engine renders in *relative* radiance: a sun of 3.2 is a number a level chose, not a
// measurement in cd/m². So none of the units below are absolute, and what the model buys is not
// photometric truth but consistency — film speed, aperture and shutter combine the way they do on a
// camera, so changing one and holding the picture means moving another by the amount a photographer
// would expect, and the two stops between f/1.4 and f/2.8 are two stops here too.

// Reflected-light meter calibration, ISO 2720. 12.5 is what the major manufacturers calibrate to.
export inline constexpr float meterCalibration = 12.5f;
// Lens transmission and vignetting: the q of ISO 2721's saturation-based speed relation.
export inline constexpr float lensTransmission = 0.65f;
// The saturation constant of that same relation — the luminance that lands on the top of the range.
export inline constexpr float saturationSpeedConstant = 78.0f;
// The film speed an EV100 is quoted at, by definition of the hundred in the name.
export inline constexpr float referenceFilmSpeed = 100.0f;

// A denominator guard, not a tolerance: every relation below is a ratio, and nothing meaningful
// happens at zero exposure or zero film speed.
inline constexpr float exposureEpsilon = 1e-6f;

// What the meter's reduction measures per pixel, and the C++ statement of the dot product
// LuminanceFragmentShader takes the logarithm of.
export [[nodiscard]] inline float relativeLuminance(const glm::vec3& radiance)
{
    return glm::dot(glm::max(radiance, glm::vec3(0.0f)),
                    glm::vec3(luminanceWeightRed, luminanceWeightGreen, luminanceWeightBlue));
}

// The exposure value a reflected-light meter reads off an average scene luminance. This is the
// whole of what "metering" is: a luminance in, a number of stops out.
export [[nodiscard]] inline float meteredEv100(const float averageLuminance)
{
    return std::log2(std::max(averageLuminance, luminanceFloor) * referenceFilmSpeed / meterCalibration);
}

// Aperture priority: the photographer picks the aperture and loads the film, and the meter picks the
// shutter. That is the answer to Camera carrying an iso and an aperture and no shutter time — the
// shutter is the leg auto exposure *solves for*, which is exactly why it was the one missing.
export [[nodiscard]] inline float shutterTimeForEv100(const float ev100, const unsigned int filmSpeed,
                                                      const float aperture)
{
    return aperture * aperture * referenceFilmSpeed /
           (std::max(static_cast<float>(filmSpeed), exposureEpsilon) * std::exp2(ev100));
}

// The multiplier the tone map applies, from all three legs at once. Substituting the shutter above
// into this collapses the film speed and the aperture out of the result, which is not a bug and is
// the entire meaning of an exposure value: two cameras metering the same scene agree on the picture
// however they got there. What they do not agree on is which shutter it took, and that is the leg
// the range on AutoExposure can refuse.
export [[nodiscard]] inline float exposureFromTriangle(const unsigned int filmSpeed, const float aperture,
                                                       const float shutterTime)
{
    return lensTransmission * static_cast<float>(filmSpeed) * shutterTime /
           (saturationSpeedConstant * std::max(aperture * aperture, exposureEpsilon));
}

// The inverse of the line above: the shutter that realises a wanted multiplier at this film speed
// and aperture. setExposure back-solves through here, so a game stating the multiplier directly
// leaves a camera whose three legs still agree with each other.
export [[nodiscard]] inline float shutterTimeForExposure(const float exposure, const unsigned int filmSpeed,
                                                         const float aperture)
{
    return exposure * saturationSpeedConstant * aperture * aperture /
           (lensTransmission * std::max(static_cast<float>(filmSpeed), exposureEpsilon));
}

// The average luminance a meter would have had to read to arrive at this multiplier. It is the
// whole chain run backwards, and the film speed and aperture are absent for the reason
// exposureFromTriangle gives. Used to seed the adaptation from whatever setExposure left, so that
// the manual value is where auto exposure starts rather than something it throws away.
export [[nodiscard]] inline float luminanceForExposure(const float exposure)
{
    return meterCalibration * lensTransmission / (saturationSpeedConstant * std::max(exposure, exposureEpsilon));
}

// One step of the eye's adaptation, over `seconds` of *simulated* time. Exponential, so the step is
// a function of how much time passed and not of how it was divided up — which is what lets the
// caller hand it a whole frame's worth of ticks in one call and still get the same answer a
// per-tick loop would have given.
export [[nodiscard]] inline float adaptLuminance(const float current, const float measured, const float speed,
                                                 const float seconds)
{
    // A camera with no reading behind it has nothing to adapt *from*; the first reading is where
    // the curve starts, not something to approach.
    if (current <= 0.0f)
    {
        return measured;
    }

    const auto blend = 1.0f - std::exp(-std::max(seconds, 0.0f) * std::max(speed, 0.0f));

    return current + (measured - current) * blend;
}

// What an auto-exposed camera settles on: the three numbers together, because a caller that wanted
// the exposure alone would still have to recompute the shutter to store it.
export struct ExposureSetting
{
    float ev100 = 0.0f;
    float shutterTime = 0.0f;
    float exposure = 0.0f;
};

// The whole meter, in the order a camera does it: read the luminance, move it by the compensation
// dial, pick the shutter that lands there at this film speed and aperture, refuse a shutter the
// body does not have, and read the multiplier back out of the triangle the shutter is now part of.
//
// Compensation subtracts because the dial is stated in the picture's terms: +1 EV means "one stop
// brighter", which a camera reaches by *under*-stating the exposure value it metered.
export [[nodiscard]] inline ExposureSetting meterExposure(const float averageLuminance, const AutoExposure& meter,
                                                          const unsigned int filmSpeed, const float aperture)
{
    const auto ev100 = meteredEv100(averageLuminance) - meter.compensation;
    const auto shutterTime = std::clamp(shutterTimeForEv100(ev100, filmSpeed, aperture),
                                        std::min(meter.minShutterTime, meter.maxShutterTime),
                                        std::max(meter.minShutterTime, meter.maxShutterTime));

    return ExposureSetting{.ev100 = ev100,
                           .shutterTime = shutterTime,
                           .exposure = exposureFromTriangle(filmSpeed, aperture, shutterTime)};
}

} // namespace raceengine
