module;

#include <optional>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.graphics.models:Fbo;

import raceengine.resource;
import :Shader;
import :Texture;

namespace raceengine
{

export enum class FboAttachmentType { Color, Depth, Stencil };

export enum class FboType { Planar, CubeMap };

// Hardware depth comparison: the sampler returns the result of comparing the reference value
// against the stored depth rather than the depth itself, filtered — GL's
// GL_TEXTURE_COMPARE_MODE/GL_TEXTURE_COMPARE_FUNC and a Vulkan sampler's compareEnable/compareOp.
// It is the sampler2DShadow path percentage-closer filtering needs, and it is stated on the
// attachment because the sampler belongs to the image on one backend and to the texture object on
// the other; the engine has no separate sampler model to hang it on. Meaningful only on a Depth
// attachment; both backends ignore it elsewhere.
export enum class DepthComparison { None, LessOrEqual };

export struct FboAttachment
{
    FboAttachmentType type;
    std::optional<unsigned int> gpuResourceId{};
    unsigned int width;
    unsigned int height;
    TextureFormat captureFormat;
    TextureFormat internalFormat;
    DepthComparison depthComparison = DepthComparison::None;
    // How many mip levels the attachment carries. One is a plain render target and is what every
    // camera and post-process buffer has always been; more than one makes it a chain, whose levels
    // a pass renders into and samples one at a time — the shape a luminance reduction or a bloom
    // pyramid is. The backend clamps it to the chain the size actually supports and gives every
    // level a view of its own; a level is not a separate attachment, because it is not separately
    // sized, formatted or released.
    unsigned int levels = 1;
    std::vector<unsigned char> data{};
};

export struct Fbo
{
    FboType type;
    std::optional<unsigned int> gpuResourceId{};
    std::vector<Resource<FboAttachment>> attachments;
};

// One image a fullscreen pass reads: an attachment and which of its mip levels this pass samples.
// The level is part of the input rather than something the shader picks with an explicit LOD
// because the descriptor names a view, and a view over one level is what states the layout that
// level is in — which is the whole difference between a pass that reads level 2 of a chain it is
// writing level 3 of and a pass that reads an image it is also rendering into.
export struct PostProcessInput
{
    // Defaulted rather than required, because an input naming a texture below leaves this unset and
    // a handle that names nothing is exactly what that means.
    Resource<FboAttachment> attachment{};
    // An image loaded from disk rather than produced by an earlier pass, and when it is set it is
    // what this input names — `attachment` is then not read at all.
    //
    // Both are images a fullscreen shader samples, and the split is only in where they came from: an
    // attachment is something this frame drew and therefore has a layout that has to be moved and a
    // level that has to be chosen, and a texture is a picture that was uploaded once and has neither.
    // The distinction matters to the backend and to nothing else, which is why it is two fields on
    // one input rather than two kinds of input. A lens dirt plate is the case it was added for: a
    // photograph of a dirty piece of glass is not something a render pass can produce.
    std::optional<Resource<Texture>> texture{};
    unsigned int level = 0;
};

export struct PostProcess
{
    Resource<Shader> shader;
    std::optional<Resource<Fbo>> output;
    // Bound in order to the fullscreen set's sampler array, from element 0. A shader reads the
    // elements it declares an interest in; the backend fills the rest so the descriptor array is
    // whole. More inputs than the set is wide are dropped and counted.
    std::vector<PostProcessInput> inputs{};
    std::vector<Resource<FboAttachment>> attachments{};
    // Which mip level of the output's colour attachment this pass renders into. Zero for a plain
    // post-process; a chain is several passes over one output, each naming its own level.
    unsigned int outputLevel = 0;
    // The four numbers this pass's own effect is tuned by, pushed to its shader as the `effect`
    // field of the fullscreen push constant. What they mean belongs to the shader reading them —
    // the occlusion gather takes a radius and a strength, the bloom threshold takes a threshold and
    // a knee — and a pass that reads none of them leaves this zero. They live on the pass rather
    // than on the camera because that is what they describe: two passes of one chain can want
    // different numbers, and a camera holding them would have to know which pass was recording.
    glm::vec4 parameters{0.0f};
    // Whether a window resize rebuilds this pass's output buffer at the new size, for the same
    // reason `Camera::tracksWindowSize` exists and with the same default. False for a pass that
    // renders into a buffer a *service* owns and sizes: the exposure meter's reduction is a fixed
    // 512 square whose last level has to be one texel, and rebuilding it at the window's size would
    // leave the meter reading a corner of the frame. Such a buffer is resized by whoever built it,
    // or not at all.
    bool tracksWindowSize = true;
};

export struct Presenter
{
    Resource<FboAttachment> output;
    Resource<Shader> shader;
    // The last pass's own numbers, pushed to its shader as the `effect` field of the fullscreen
    // push constant exactly as a PostProcess's are. This is the one pass that runs on a
    // display-referred image — everything before it works in radiance — so what belongs here is
    // what a lens and a grade do rather than what light does: x how far the colour channels are
    // pulled apart towards the corners, y contrast about mid grey, z saturation, w a split tone
    // (positive warms the highlights and cools the shadows).
    glm::vec4 parameters{0.0f, 1.0f, 0.0f, 0.0f};
    // The colour grade, as a three-dimensional lookup table: what a grading tool exports and the
    // whole reason a look is tunable by whoever is grading it rather than by whoever can rebuild
    // the shader. A presenter without one grades nothing — the engine binds the identity — and the
    // analytic knobs above are then all there is.
    std::optional<Resource<Texture>> lookupTable{};
};

} // namespace raceengine
