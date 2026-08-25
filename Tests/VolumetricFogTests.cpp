#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using Catch::Approx;
using raceengine::Fog;
using raceengine::fogOpticalDepth;
using raceengine::fogTransmittance;
using raceengine::henyeyGreenstein;

namespace
{

// A medium with numbers a circuit would actually state: a world unit is a tenth of a metre, so this
// is an extinction of 1e-4 per unit at ground level falling by 1/e every thirty metres.
constexpr auto haze = Fog{.enabled = true, .density = 1.0e-4f, .scaleHeight = 300.0f, .baseHeight = 0.0f};

// The integral the closed form claims to be, evaluated the dumb way. This is the whole point of the
// file: the analytic expression is what ships, and a midpoint sum of the density it was derived
// from is an answer that shares none of its algebra.
float marchedOpticalDepth(const Fog& fog, const float originHeight, const float directionY, const float distance,
                          const int steps)
{
    const auto step = distance / static_cast<float>(steps);
    auto total = 0.0;

    for (auto index = 0; index < steps; index++)
    {
        const auto s = (static_cast<float>(index) + 0.5f) * step;
        const auto height = originHeight + s * directionY;
        total += static_cast<double>(fog.density) *
                 static_cast<double>(std::exp(-(height - fog.baseHeight) / fog.scaleHeight)) *
                 static_cast<double>(step);
    }

    return static_cast<float>(total);
}

} // namespace

TEST_CASE("a scene that states no fog has no fog in it", "[postfx][fog]")
{
    // The default, which is what every scene that never mentions fog carries. Nothing about the ray
    // can make a disabled medium absorb anything, which is what makes the parity gates provable:
    // the branch is on this and nothing else.
    const auto none = Fog{};

    for (const auto directionY : {-1.0f, -0.3f, 0.0f, 0.3f, 1.0f})
    {
        CHECK(fogOpticalDepth(none, 10.0f, directionY, 5000.0f) == 0.0f);
        CHECK(fogTransmittance(none, 10.0f, directionY, 5000.0f) == 1.0f);
    }

    // And an enabled fog of no density is the same statement said the other way, so a level fading
    // it in from nothing starts from exactly the frame it would have drawn without it.
    const auto empty = Fog{.enabled = true, .density = 0.0f};
    CHECK(fogTransmittance(empty, 10.0f, 0.0f, 5000.0f) == 1.0f);
}

TEST_CASE("a level ray at the reference height integrates a constant density", "[postfx][fog]")
{
    // The one case with an answer anybody can check by hand, and the case the published form divides
    // by zero on: optical depth is density times distance, exactly.
    CHECK(fogOpticalDepth(haze, 0.0f, 0.0f, 10000.0f) == Approx(1.0f).epsilon(1e-5));
    CHECK(fogTransmittance(haze, 0.0f, 0.0f, 10000.0f) == Approx(std::exp(-1.0f)).epsilon(1e-5));
}

TEST_CASE("the closed form is the integral it claims to be", "[postfx][fog]")
{
    // Every geometry that matters: level, climbing, descending, and both steeply. The march is the
    // reference and the closed form is what ships.
    const auto directions = std::vector<float>{-0.9f, -0.5f, -0.2f, -0.05f, 0.0f, 0.05f, 0.2f, 0.5f, 0.9f};
    const auto heights = std::vector<float>{-50.0f, 0.0f, 120.0f, 900.0f};

    for (const auto height : heights)
    {
        for (const auto directionY : directions)
        {
            const auto analytic = fogOpticalDepth(haze, height, directionY, 4000.0f);
            const auto marched = marchedOpticalDepth(haze, height, directionY, 4000.0f, 200000);

            CHECK(analytic == Approx(marched).epsilon(1e-3));
        }
    }
}

TEST_CASE("the nearly-level ray is continuous with the level one", "[postfx][fog]")
{
    // The series exists for this: the published form is a difference of two nearly equal numbers
    // over a small one, and a driver spends the lap looking along a ray whose y component is a few
    // thousandths. A discontinuity here would read as the fog changing thickness as the car crests.
    const auto level = fogOpticalDepth(haze, 30.0f, 0.0f, 6000.0f);

    for (const auto directionY : {-1.0e-7f, -1.0e-5f, -1.0e-3f, 1.0e-7f, 1.0e-5f, 1.0e-3f})
    {
        CHECK(fogOpticalDepth(haze, 30.0f, directionY, 6000.0f) == Approx(level).epsilon(2e-2));
    }

    // And it approaches the level answer rather than merely staying near it.
    CHECK(std::abs(fogOpticalDepth(haze, 30.0f, 1.0e-7f, 6000.0f) - level) <
          std::abs(fogOpticalDepth(haze, 30.0f, 1.0e-3f, 6000.0f) - level) + 1e-6f);
}

TEST_CASE("height is what makes fog a layer rather than a dimming", "[postfx][fog]")
{
    // A ridge standing out of the mist and a car sitting in it are one number apart, and that number
    // is the scale height. Ten scale heights up is four and a half orders of magnitude of density.
    const auto atGround = fogOpticalDepth(haze, 0.0f, 0.0f, 3000.0f);
    const auto atRidge = fogOpticalDepth(haze, 3000.0f, 0.0f, 3000.0f);

    CHECK(atRidge < atGround * 1.0e-4f);

    // A ray that climbs out of the layer accumulates less than one that stays in it, and one that
    // descends into it accumulates more. Both against the same distance.
    CHECK(fogOpticalDepth(haze, 0.0f, 0.7f, 3000.0f) < atGround);
    CHECK(fogOpticalDepth(haze, 300.0f, -0.7f, 3000.0f) > fogOpticalDepth(haze, 300.0f, 0.0f, 3000.0f));
}

TEST_CASE("optical depth rises with distance and with density", "[postfx][fog]")
{
    auto previous = 0.0f;
    for (const auto distance : {100.0f, 500.0f, 1000.0f, 5000.0f, 20000.0f})
    {
        const auto depth = fogOpticalDepth(haze, 5.0f, 0.1f, distance);
        CHECK(depth > previous);
        previous = depth;
    }

    auto thicker = haze;
    thicker.density = haze.density * 3.0f;
    CHECK(fogOpticalDepth(thicker, 5.0f, 0.1f, 2000.0f) == Approx(3.0f * fogOpticalDepth(haze, 5.0f, 0.1f, 2000.0f)));
}

TEST_CASE("a steeply descending ray stays finite rather than overflowing", "[postfx][fog]")
{
    // Physically ordinary and numerically hostile: straight down through a shallow layer, far. The
    // exponent is clamped, so what comes back is a fully opaque medium rather than an infinity that
    // would reach the frame as a NaN through the in-scattering it multiplies.
    auto shallow = haze;
    shallow.scaleHeight = 20.0f;

    const auto depth = fogOpticalDepth(shallow, 4000.0f, -1.0f, 20000.0f);

    CHECK(std::isfinite(depth));
    CHECK(fogTransmittance(shallow, 4000.0f, -1.0f, 20000.0f) == Approx(0.0f).margin(1e-6));
}

TEST_CASE("transmittance across a march telescopes to the analytic total", "[postfx][fog]")
{
    // The property the god rays are built on. Each step of the in-scattering march is weighted by the
    // *difference* in transmittance across it, which is the analytic value of the in-scattering
    // integral over that step; so with nothing shadowing anything the sum is exactly 1 - T however
    // few steps are taken. That is what lets the marched half and the closed-form half be added
    // without counting the same light twice, and what makes the step count a quality knob rather
    // than a correctness one.
    const auto origin = 12.0f;
    const auto directionY = -0.15f;
    const auto distance = 4000.0f;
    const auto total = fogTransmittance(haze, origin, directionY, distance);

    for (const auto steps : {1, 2, 4, 16, 64})
    {
        auto sum = 0.0f;
        auto previous = 1.0f;

        for (auto index = 1; index <= steps; index++)
        {
            const auto reach = distance * static_cast<float>(index) / static_cast<float>(steps);
            const auto here = fogTransmittance(haze, origin, directionY, reach);
            sum += previous - here;
            previous = here;
        }

        CHECK(sum == Approx(1.0f - total).epsilon(1e-5));
    }
}

TEST_CASE("an isotropic phase function is one in every direction", "[postfx][fog]")
{
    // Normalised so isotropic is 1 rather than 1/4pi — the coefficient beside it is then an albedo,
    // which is the reading a level wants and the reason no factor of 4pi appears anywhere else.
    for (const auto cosTheta : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
    {
        CHECK(henyeyGreenstein(cosTheta, 0.0f) == Approx(1.0f));
    }
}

TEST_CASE("a forward-scattering medium is brighter towards the light than away from it", "[postfx][fog]")
{
    // The asymmetry *is* the effect: a shaft is bright when the camera looks towards the sun and
    // faint when it looks away, and at g = 0 there is no shaft at all, only a wash.
    const auto forward = henyeyGreenstein(1.0f, 0.6f);
    const auto sideways = henyeyGreenstein(0.0f, 0.6f);
    const auto backward = henyeyGreenstein(-1.0f, 0.6f);

    CHECK(forward > sideways);
    CHECK(sideways > backward);
    CHECK(forward > 4.0f * backward);

    // A negative asymmetry is the same medium looked at from the other side.
    CHECK(henyeyGreenstein(-1.0f, -0.6f) == Approx(forward));
}

TEST_CASE("the phase function still integrates to one over the sphere", "[postfx][fog]")
{
    // Dropping the 1/4pi rescales it and must not break it: the mean over the sphere has to stay 1,
    // or the medium is inventing or destroying light as the asymmetry is turned. The solid angle
    // measure is 2*pi*d(cos theta), so the mean over the sphere is the mean over cos theta.
    for (const auto g : {-0.8f, -0.3f, 0.0f, 0.3f, 0.6f, 0.8f})
    {
        constexpr auto steps = 400000;
        auto total = 0.0;

        for (auto index = 0; index < steps; index++)
        {
            const auto cosTheta = -1.0f + 2.0f * (static_cast<float>(index) + 0.5f) / static_cast<float>(steps);
            total += static_cast<double>(henyeyGreenstein(cosTheta, g));
        }

        CHECK(static_cast<float>(total / steps) == Approx(1.0f).epsilon(2e-3));
    }
}

TEST_CASE("the asymmetry is clamped short of the degenerate mirror", "[postfx][fog]")
{
    // At exactly one the published function is a delta: infinite forwards and zero everywhere else.
    // The clamp is what keeps a level that states it from putting an infinity in the frame, and the
    // service refuses the value before it can get here in the first place.
    CHECK(std::isfinite(henyeyGreenstein(1.0f, 1.0f)));
    CHECK(std::isfinite(henyeyGreenstein(-1.0f, -1.0f)));
    CHECK(henyeyGreenstein(1.0f, 0.99f) == Approx(henyeyGreenstein(1.0f, 0.95f)));
}
