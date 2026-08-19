#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using Catch::Approx;
using raceengine::evaluateToneCurve;
using raceengine::mipLevelCount;
using raceengine::PostProcessBinding;
using raceengine::postProcessBindings;
using raceengine::postProcessInputCount;
using raceengine::ToneCurve;
using raceengine::toneGreyPivot;

namespace
{

// The filmic transfer the curve is built on, written out here rather than reached for so the tests
// state what "neutral" means independently of the code under test.
float acesReference(const float x)
{
    return std::clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

constexpr auto neutral = ToneCurve{.contrast = 1.0f, .toe = 0.0f, .shoulder = 0.0f};

// The range a rendered frame actually spans on either side: well under a hundredth of middle grey
// to well over the clip.
std::vector<float> radianceSweep()
{
    std::vector<float> values;
    for (auto step = 0; step <= 240; step++)
    {
        values.push_back(std::pow(10.0f, -4.0f + static_cast<float>(step) * (6.0f / 240.0f)));
    }
    return values;
}

} // namespace

TEST_CASE("the neutral tone curve is the plain filmic transfer", "[postfx][tonecurve]")
{
    for (const auto radiance : radianceSweep())
    {
        REQUIRE(evaluateToneCurve(radiance, neutral, 2.5f) == Approx(acesReference(radiance * 2.5f)).margin(1e-6));
    }
}

TEST_CASE("exposure is a linear multiplier applied before the curve", "[postfx][tonecurve]")
{
    // Doubling the exposure is exactly halving the radiance's distance from the curve: the same
    // display value comes back out, which is what makes exposure the thing auto exposure drives.
    const auto curve = ToneCurve{};

    for (const auto radiance : radianceSweep())
    {
        REQUIRE(evaluateToneCurve(radiance, curve, 5.0f) ==
                Approx(evaluateToneCurve(radiance * 2.0f, curve, 2.5f)).margin(1e-6));
    }
}

TEST_CASE("the tone curve is monotonic and never leaves the display range", "[postfx][tonecurve]")
{
    const std::array curves = {neutral, ToneCurve{}, ToneCurve{.contrast = 1.6f, .toe = 0.8f, .shoulder = 0.6f},
                               ToneCurve{.contrast = 0.7f, .toe = 0.0f, .shoulder = 0.0f}};

    for (const auto& curve : curves)
    {
        auto previous = -1.0f;
        for (const auto radiance : radianceSweep())
        {
            const auto mapped = evaluateToneCurve(radiance, curve, 2.5f);

            REQUIRE(mapped >= 0.0f);
            REQUIRE(mapped <= 1.0f);
            REQUIRE(mapped >= previous);
            previous = mapped;
        }
    }
}

TEST_CASE("black stays black whatever the curve does", "[postfx][tonecurve]")
{
    // The ends of the range are fixed points of every stage, which is what makes contrast, toe and
    // shoulder controls of shape rather than of level: none of them can lift a black or clip a
    // white that was not already there.
    REQUIRE(evaluateToneCurve(0.0f, ToneCurve{}, 2.5f) == 0.0f);
    REQUIRE(evaluateToneCurve(0.0f, neutral, 2.5f) == 0.0f);
    REQUIRE(evaluateToneCurve(-1.0f, ToneCurve{}, 2.5f) == 0.0f);
}

TEST_CASE("the toe darkens the bottom of the range and leaves the top where it was", "[postfx][tonecurve]")
{
    const auto toed = ToneCurve{.contrast = 1.0f, .toe = 0.5f, .shoulder = 0.0f};

    // Something that lands well down the curve moves a long way; something already near white does
    // not, because the toe's exponent has run back to one by then.
    const auto dark = 0.02f;
    const auto bright = 40.0f;

    REQUIRE(evaluateToneCurve(dark, toed, 2.5f) < evaluateToneCurve(dark, neutral, 2.5f) * 0.8f);
    REQUIRE(evaluateToneCurve(bright, toed, 2.5f) == Approx(evaluateToneCurve(bright, neutral, 2.5f)).margin(0.01));
}

TEST_CASE("the shoulder holds the top of the range back and leaves the bottom where it was", "[postfx][tonecurve]")
{
    const auto shouldered = ToneCurve{.contrast = 1.0f, .toe = 0.0f, .shoulder = 0.5f};

    const auto dark = 0.02f;
    const auto bright = 2.0f;

    REQUIRE(evaluateToneCurve(dark, shouldered, 2.5f) == Approx(evaluateToneCurve(dark, neutral, 2.5f)).margin(0.005));
    REQUIRE(evaluateToneCurve(bright, shouldered, 2.5f) < evaluateToneCurve(bright, neutral, 2.5f));
}

TEST_CASE("contrast pivots at middle grey", "[postfx][tonecurve]")
{
    const auto contrasty = ToneCurve{.contrast = 1.4f, .toe = 0.0f, .shoulder = 0.0f};
    const auto exposure = 2.5f;
    // The radiance that lands exactly on the pivot once exposed: the one value contrast cannot
    // move, which is what "pivots at" means.
    const auto atPivot = toneGreyPivot / exposure;

    REQUIRE(evaluateToneCurve(atPivot, contrasty, exposure) ==
            Approx(evaluateToneCurve(atPivot, neutral, exposure)).margin(1e-5));
    REQUIRE(evaluateToneCurve(atPivot * 0.25f, contrasty, exposure) <
            evaluateToneCurve(atPivot * 0.25f, neutral, exposure));
    REQUIRE(evaluateToneCurve(atPivot * 4.0f, contrasty, exposure) >
            evaluateToneCurve(atPivot * 4.0f, neutral, exposure));
}

TEST_CASE("the default tone curve is the dramatic one", "[postfx][tonecurve]")
{
    // A characterization of the look, not a derivation of it: the numbers were chosen against this
    // scene's own frame (see docs/vulkan-abi.md), and the property they were chosen for is that
    // shadow goes down further than midtone and highlight comes back up.
    const auto curve = ToneCurve{};

    REQUIRE(curve.contrast > 1.0f);
    REQUIRE(curve.toe > 0.0f);
    REQUIRE(curve.shoulder > 0.0f);

    const auto exposure = 2.5f;
    const auto shadow = toneGreyPivot / exposure * 0.125f;
    const auto midtone = toneGreyPivot / exposure;
    const auto highlight = toneGreyPivot / exposure * 8.0f;

    REQUIRE(evaluateToneCurve(shadow, curve, exposure) < evaluateToneCurve(shadow, neutral, exposure) * 0.5f);
    REQUIRE(evaluateToneCurve(midtone, curve, exposure) < evaluateToneCurve(midtone, neutral, exposure));
    REQUIRE(evaluateToneCurve(highlight, curve, exposure) > evaluateToneCurve(highlight, neutral, exposure));
}

TEST_CASE("the tone curve is a transfer function, not a colour transform", "[postfx][tonecurve]")
{
    // Every channel goes through the same one-dimensional curve and no channel reads another, which
    // is why a grey stays grey through it.
    const auto curve = ToneCurve{};
    const auto colour = glm::vec3(0.4f, 0.05f, 1.7f);
    const auto mapped = evaluateToneCurve(colour, curve, 2.5f);

    REQUIRE(mapped.r == Approx(evaluateToneCurve(colour.r, curve, 2.5f)));
    REQUIRE(mapped.g == Approx(evaluateToneCurve(colour.g, curve, 2.5f)));
    REQUIRE(mapped.b == Approx(evaluateToneCurve(colour.b, curve, 2.5f)));

    const auto grey = evaluateToneCurve(glm::vec3(0.3f), curve, 2.5f);
    REQUIRE(grey.r == Approx(grey.g));
    REQUIRE(grey.g == Approx(grey.b));
}

TEST_CASE("a mip chain runs from the full level down to a single texel", "[postfx][mipchain]")
{
    REQUIRE(mipLevelCount(1, 1) == 1u);
    REQUIRE(mipLevelCount(2, 2) == 2u);
    REQUIRE(mipLevelCount(1920, 1080) == 11u);
    // The longest side decides: halving stops when *both* have reached one texel, and a 1024x1
    // image still has eleven levels of width to give away.
    REQUIRE(mipLevelCount(1024, 1) == 11u);
    REQUIRE(mipLevelCount(1, 1024) == 11u);
    // Non-powers of two round down, which is what Vulkan's own mip extents do.
    REQUIRE(mipLevelCount(100, 100) == 7u);
}

TEST_CASE("a pass's inputs land on the fullscreen set in order", "[postfx][inputs]")
{
    const auto fallback = PostProcessBinding{.gpuResourceId = 99, .level = 0};
    const std::array declared = {PostProcessBinding{.gpuResourceId = 7, .level = 0},
                                 PostProcessBinding{.gpuResourceId = 8, .level = 3}};

    const auto bindings = postProcessBindings(std::span<const PostProcessBinding>(declared), fallback);

    REQUIRE(bindings[0] == declared[0]);
    REQUIRE(bindings[1] == declared[1]);
}

TEST_CASE("every binding a pass did not name is filled with the fallback", "[postfx][inputs]")
{
    // The rule the descriptor array depends on: an element left unwritten is what validation
    // reports when the pipeline statically uses the array, and it declares the array whole.
    const auto fallback = PostProcessBinding{.gpuResourceId = 99, .level = 0};
    const std::array declared = {PostProcessBinding{.gpuResourceId = 7, .level = 0}};

    const auto bindings = postProcessBindings(std::span<const PostProcessBinding>(declared), fallback);

    REQUIRE(bindings.size() == postProcessInputCount);
    for (auto slot = size_t{1}; slot < bindings.size(); slot++)
    {
        REQUIRE(bindings[slot] == fallback);
    }
}

TEST_CASE("a pass with no inputs at all still fills the whole array", "[postfx][inputs]")
{
    const auto fallback = PostProcessBinding{.gpuResourceId = 99, .level = 0};
    const auto bindings = postProcessBindings(std::span<const PostProcessBinding>(), fallback);

    REQUIRE(std::ranges::all_of(bindings, [&](const auto& binding) { return binding == fallback; }));
}

TEST_CASE("inputs past the width of the fullscreen set are dropped", "[postfx][inputs]")
{
    // Dropped rather than wrapped or reported from here: the caller counts the excess, because it
    // is a scene asking for more than the contract carries and not a condition of the frame.
    const auto fallback = PostProcessBinding{.gpuResourceId = 99, .level = 0};
    std::vector<PostProcessBinding> declared;
    for (auto index = 0u; index < postProcessInputCount + 3u; index++)
    {
        declared.push_back(PostProcessBinding{.gpuResourceId = index + 1u, .level = 0});
    }

    const auto bindings = postProcessBindings(std::span<const PostProcessBinding>(declared), fallback);

    REQUIRE(bindings.size() == postProcessInputCount);
    REQUIRE(bindings.back().gpuResourceId == postProcessInputCount);
    REQUIRE(std::ranges::none_of(bindings, [&](const auto& binding) { return binding == fallback; }));
}
