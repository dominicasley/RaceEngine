#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using Catch::Approx;
using raceengine::adaptLuminance;
using raceengine::AutoExposure;
using raceengine::exposureFromTriangle;
using raceengine::luminanceFloor;
using raceengine::luminanceForExposure;
using raceengine::luminanceWeightBlue;
using raceengine::luminanceWeightGreen;
using raceengine::luminanceWeightRed;
using raceengine::meterExposure;
using raceengine::meteredEv100;
using raceengine::shutterTimeForEv100;
using raceengine::shutterTimeForExposure;

namespace
{

// A meter with the shutter range opened wide enough that nothing below clips against it. The range
// is a real part of the model and it gets a test of its own; everywhere else it would only be a way
// for an unrelated assertion to fail.
constexpr auto unbounded = AutoExposure{.minShutterTime = 1e-9f, .maxShutterTime = 1e9f};

// The span of average luminances a scene can plausibly meter at, in relative units: from a frame
// that is nearly all shadow to one looking into the sun.
std::vector<float> luminanceSweep()
{
    std::vector<float> values;
    for (auto step = 0; step <= 120; step++)
    {
        values.push_back(std::pow(10.0f, -3.0f + static_cast<float>(step) * (5.0f / 120.0f)));
    }
    return values;
}

} // namespace

TEST_CASE("the exposure triangle and its inverse are the same relation read both ways",
          "[postfx][exposure][triangle]")
{
    // setExposure back-solves a shutter out of a multiplier and the renderer then reads the
    // multiplier back out of the triangle. If these two disagreed anywhere, stating an exposure by
    // hand would quietly change it.
    for (const auto filmSpeed : {100u, 400u, 6400u})
    {
        for (const auto aperture : {1.4f, 2.8f, 11.0f})
        {
            for (const auto exposure : {0.05f, 1.0f, 2.5f, 40.0f})
            {
                const auto shutter = shutterTimeForExposure(exposure, filmSpeed, aperture);

                REQUIRE(exposureFromTriangle(filmSpeed, aperture, shutter) == Approx(exposure).epsilon(1e-5));
            }
        }
    }
}

TEST_CASE("a stop is a stop on every leg of the triangle", "[postfx][exposure][triangle]")
{
    // What the physical model is for: the legs trade against each other the way a photographer
    // expects, so holding the picture while changing one means moving another by a known amount.
    const auto base = exposureFromTriangle(100u, 2.8f, 1.0f / 250.0f);

    // Doubling the film speed is one stop of extra sensitivity.
    REQUIRE(exposureFromTriangle(200u, 2.8f, 1.0f / 250.0f) == Approx(base * 2.0f).epsilon(1e-5));
    // Doubling the shutter time is one stop of extra light.
    REQUIRE(exposureFromTriangle(100u, 2.8f, 1.0f / 125.0f) == Approx(base * 2.0f).epsilon(1e-5));
    // f/1.4 to f/2.8 is two stops, and the halving of the f-number is a doubling of the aperture's
    // area — which is why this leg squares where the other two are linear.
    REQUIRE(exposureFromTriangle(100u, 1.4f, 1.0f / 250.0f) == Approx(base * 4.0f).epsilon(1e-5));
}

TEST_CASE("metering is logarithmic in the luminance it reads", "[postfx][exposure][meter]")
{
    // Twice the light is one more stop, which is the whole of what an exposure value is.
    for (const auto luminance : luminanceSweep())
    {
        REQUIRE(meteredEv100(luminance * 2.0f) == Approx(meteredEv100(luminance) + 1.0f).margin(1e-5));
    }
}

TEST_CASE("a black frame meters as a number rather than as minus infinity", "[postfx][exposure][meter]")
{
    // The floor the GPU reduction takes its logarithm above, stated on this side of the divide too:
    // a frame that is entirely shadow still has to produce a shutter the camera can be set to.
    const auto floored = meteredEv100(luminanceFloor);

    REQUIRE(meteredEv100(0.0f) == Approx(floored));
    REQUIRE(std::isfinite(floored));
    REQUIRE(std::isfinite(meterExposure(0.0f, unbounded, 100u, 2.8f).exposure));
}

TEST_CASE("two cameras metering the same scene agree on the picture", "[postfx][exposure][meter]")
{
    // Film speed and aperture cancel out of the multiplier once the shutter has been solved for
    // them, so what the meter fixes is the exposure and what it leaves open is which body reached
    // it. This is the property that lets a level move iso or aperture for the look of the numbers
    // without the frame changing brightness under it.
    const auto reference = meterExposure(0.08f, unbounded, 100u, 2.8f);

    REQUIRE(meterExposure(0.08f, unbounded, 6400u, 1.4f).exposure == Approx(reference.exposure).epsilon(1e-5));
    REQUIRE(meterExposure(0.08f, unbounded, 200u, 11.0f).exposure == Approx(reference.exposure).epsilon(1e-5));
    // What they do not agree on is the shutter it took to get there: six stops of film speed and
    // two of aperture have to come out of the one leg that is left.
    REQUIRE(meterExposure(0.08f, unbounded, 6400u, 1.4f).shutterTime <
            meterExposure(0.08f, unbounded, 100u, 2.8f).shutterTime);
}

TEST_CASE("compensation is a dial in the picture's terms, not the meter's", "[postfx][exposure][meter]")
{
    const auto neutral = meterExposure(0.08f, unbounded, 400u, 2.0f);
    const auto openedUp = meterExposure(0.08f, AutoExposure{.compensation = 1.0f,
                                                            .minShutterTime = unbounded.minShutterTime,
                                                            .maxShutterTime = unbounded.maxShutterTime},
                                        400u, 2.0f);

    // +1 EV means one stop brighter, which the camera reaches by understating the value it metered.
    REQUIRE(openedUp.ev100 == Approx(neutral.ev100 - 1.0f).margin(1e-5));
    REQUIRE(openedUp.exposure == Approx(neutral.exposure * 2.0f).epsilon(1e-5));
}

TEST_CASE("the shutter range is the leg that can refuse", "[postfx][exposure][meter]")
{
    // This scene meters at about the luminance the sandbox does, which a slow lens on slow film
    // cannot reach inside a quarter of a second. The clamp is what makes film speed and aperture
    // matter to the picture and not only to the arithmetic: with the shutter pinned, the exposure
    // is whatever that shutter gives at this iso and aperture, and opening up is the only way back
    // to the metered answer.
    constexpr auto body = AutoExposure{.minShutterTime = 1.0f / 8000.0f, .maxShutterTime = 1.0f / 4.0f};
    const auto slowLens = meterExposure(0.05f, body, 100u, 11.0f);

    REQUIRE(slowLens.shutterTime == Approx(body.maxShutterTime));
    REQUIRE(slowLens.exposure == Approx(exposureFromTriangle(100u, 11.0f, body.maxShutterTime)).epsilon(1e-5));
    REQUIRE(slowLens.shutterTime < shutterTimeForEv100(slowLens.ev100, 100u, 11.0f));

    // The same scene through the fast lens and fast film the sandbox's camera carries is inside the
    // range, so the meter's answer stands rather than being held back.
    const auto fastLens = meterExposure(0.05f, body, 6400u, 1.4f);

    REQUIRE(fastLens.shutterTime < body.maxShutterTime);
    REQUIRE(fastLens.exposure > slowLens.exposure);
}

TEST_CASE("the luminance a manual exposure implies is the one that meters back to it",
          "[postfx][exposure][meter]")
{
    // How setExposure seeds the adaptation: the level's hand-set multiplier is turned into the
    // reading that would have produced it, so metering starts where the level left off instead of
    // stepping to a different value on the first frame that reads.
    for (const auto exposure : {0.25f, 1.0f, 2.5f, 12.0f})
    {
        const auto implied = luminanceForExposure(exposure);

        REQUIRE(meterExposure(implied, unbounded, 100u, 2.8f).exposure == Approx(exposure).epsilon(1e-4));
        REQUIRE(meterExposure(implied, unbounded, 6400u, 1.4f).exposure == Approx(exposure).epsilon(1e-4));
    }
}

TEST_CASE("adaptation does not depend on how the elapsed time was divided up",
          "[postfx][exposure][adaptation]")
{
    // The engine hands a whole frame's worth of ticks to one call rather than looping. That is only
    // correct because the curve is exponential, and this is the statement of it: a frame that
    // simulated four ticks lands exactly where four single-tick calls would have.
    constexpr auto tick = 1.0f / 120.0f;
    constexpr auto speed = 6.0f;
    auto stepped = 0.5f;

    for (auto step = 0; step < 4; step++)
    {
        stepped = adaptLuminance(stepped, 0.08f, speed, tick);
    }

    REQUIRE(adaptLuminance(0.5f, 0.08f, speed, 4.0f * tick) == Approx(stepped).epsilon(1e-5));
}

TEST_CASE("adaptation closes on the reading without passing it", "[postfx][exposure][adaptation]")
{
    // Monotone and bounded in both directions: an adaptation that overshot would read as the
    // exposure hunting, which on a capture gate looks like nondeterminism rather than like a curve.
    auto rising = 0.02f;
    auto falling = 1.20f;

    for (auto step = 0; step < 600; step++)
    {
        const auto nextRising = adaptLuminance(rising, 0.30f, 6.0f, 1.0f / 120.0f);
        const auto nextFalling = adaptLuminance(falling, 0.30f, 3.0f, 1.0f / 120.0f);

        REQUIRE(nextRising >= rising);
        REQUIRE(nextRising <= 0.30f);
        REQUIRE(nextFalling <= falling);
        REQUIRE(nextFalling >= 0.30f);

        rising = nextRising;
        falling = nextFalling;
    }

    REQUIRE(rising == Approx(0.30f).epsilon(1e-3));
    REQUIRE(falling == Approx(0.30f).epsilon(1e-3));
}

TEST_CASE("a camera with nothing behind it adapts to the first reading it gets",
          "[postfx][exposure][adaptation]")
{
    // Zero is "no reading yet", not "black": approaching it exponentially would spend the opening
    // seconds of a level walking up from a scene that was never there.
    REQUIRE(adaptLuminance(0.0f, 0.42f, 6.0f, 1.0f / 120.0f) == Approx(0.42f));
    // And a frame that simulated no time at all holds what it had.
    REQUIRE(adaptLuminance(0.11f, 0.42f, 6.0f, 0.0f) == Approx(0.11f));
}

TEST_CASE("the meter's luminance is the weighting the reduction shader takes the log of",
          "[postfx][exposure][meter]")
{
    // The C++ statement of the dot product in LuminanceFragmentShader, pinned here because the two
    // are the same function written twice and only this side can be tested without a device.
    REQUIRE(raceengine::relativeLuminance(glm::vec3(1.0f)) ==
            Approx(luminanceWeightRed + luminanceWeightGreen + luminanceWeightBlue));
    REQUIRE(raceengine::relativeLuminance(glm::vec3(0.0f, 1.0f, 0.0f)) == Approx(luminanceWeightGreen));
    // Grey in, that same grey out: the weights are a partition of one, so a neutral frame meters at
    // its own radiance.
    REQUIRE(raceengine::relativeLuminance(glm::vec3(0.18f)) == Approx(0.18f).epsilon(1e-5));
    // A negative radiance is not light. It reaches here from filtering rather than from a material,
    // and clamping is what keeps it from cancelling light that is actually there.
    REQUIRE(raceengine::relativeLuminance(glm::vec3(-4.0f, 0.5f, -1.0f)) == Approx(0.5f * luminanceWeightGreen));
}
