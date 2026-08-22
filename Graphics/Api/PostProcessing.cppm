module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>

export module raceengine.graphics:PostProcessing;

import :RenderContract;
import raceengine.graphics.models;

namespace raceengine
{

// The arithmetic the post chain is built out of, stated once and away from any device so it can be
// read and tested on its own: how long a mip chain is, which image lands on which element of the
// fullscreen sampler array, and what the tone curve does to a value. The backend is the only
// caller of the first two and the fragment shader is the second implementation of the third.

// How many mip levels an image of this size can carry, counting the full one. Halving stops at
// 1x1, so a 1920x1080 attachment has eleven and a 1x1 one has a single level.
export [[nodiscard]] constexpr uint32_t mipLevelCount(const uint32_t width, const uint32_t height)
{
    auto levels = 1u;
    for (auto extent = std::max(width, height); extent > 1; extent /= 2)
    {
        levels++;
    }
    return levels;
}

static_assert(mipLevelCount(1, 1) == 1);
static_assert(mipLevelCount(2, 1) == 2);
static_assert(mipLevelCount(1920, 1080) == 11);
static_assert(mipLevelCount(128, 128) == 8);

// One element of the fullscreen set's sampler array: a renderer image id and the mip level of it
// the descriptor names. Both, because the level is what picks the view, and the view is what
// states which layout the barrier before the pass has to leave that subresource in.
export struct PostProcessBinding
{
    unsigned int gpuResourceId = 0;
    unsigned int level = 0;

    [[nodiscard]] constexpr bool operator==(const PostProcessBinding&) const = default;
};

// The whole array the fullscreen set is written with, from the inputs a pass declared.
//
// Inputs land in order from element 0 and every element past the last of them is `fallback`. The
// padding is not tidiness: a descriptor a pipeline statically uses must be written, the shader
// declares the array whole, and an element left unwritten is what validation reports — the same
// rule that makes a view with no shadow cascades still bind a 1x1 depth image. Inputs past the end
// of the array are dropped, which the caller counts rather than this reports: it is a scene
// asking for more than the contract carries, not a condition of the frame.
export [[nodiscard]] constexpr std::array<PostProcessBinding, postProcessInputCount>
postProcessBindings(const std::span<const PostProcessBinding> inputs, const PostProcessBinding& fallback)
{
    std::array<PostProcessBinding, postProcessInputCount> bindings{};
    bindings.fill(fallback);

    const auto bound = std::min<size_t>(inputs.size(), postProcessInputCount);
    for (size_t slot = 0; slot < bound; slot++)
    {
        bindings[slot] = inputs[slot];
    }

    return bindings;
}

// Scene radiance to a display value, which is the function HdrFragmentShader.glsl also is. Four
// stages, each monotonic and each leaving the ends of the range alone:
//
//  - exposure, a linear multiplier, because the scene is rendered in relative units and something
//    has to say what they map to;
//  - contrast, a power law about middle grey in that exposed domain;
//  - the filmic transfer itself;
//  - a toe and a shoulder, one-sided gammas whose exponent runs from 1 + toe at black to 1 at
//    white and from 1 at black to 1 + shoulder at white. Both only ever darken, so raising either
//    deepens a region rather than shifting the whole picture.
export [[nodiscard]] float evaluateToneCurve(const float value, const ToneCurve& curve, const float exposure);

// Per channel, exactly as the fragment stage applies it: the curve is a transfer function and not
// a colour transform, so nothing here reads one channel to decide another.
export [[nodiscard]] glm::vec3 evaluateToneCurve(const glm::vec3& value, const ToneCurve& curve, const float exposure);

} // namespace raceengine
