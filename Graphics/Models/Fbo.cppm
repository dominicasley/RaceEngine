module;

#include <optional>
#include <string>
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
    // An opt-in first fill: when set, the backend clears every level to this colour at image
    // creation and leaves the image sampleable. It exists for an attachment something reads before
    // anything renders into it — frame 0's probe captures run before the scene cameras' chains, so
    // a map written at the end of a chain would otherwise be sampled in undefined layout on the one
    // frame nobody has written it. Unset is every attachment that ever existed: the first pass over
    // it finds the image exactly as it always has.
    std::optional<glm::vec4> initialColour{};
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
    // What this pass is for, in words: the backend stamps it on the pass as a debug label and a
    // GPU profiler zone, so a capture reads "gtao gather" rather than the nth anonymous
    // fullscreen draw. Empty is legal and labels as "post-process".
    std::string debugName{};
    // Bound in order to the fullscreen set's sampler array, from element 0. A shader reads the
    // elements it declares an interest in; the backend fills the rest so the descriptor array is
    // whole. More inputs than the set is wide are dropped and counted.
    std::vector<PostProcessInput> inputs{};
    // Three-dimensional textures this pass samples, bound in order to the fullscreen set's two
    // sampler3D bindings beside the grade. A list of its own rather than more `inputs` because the
    // input array is sampler2D the whole way across, and a 3D image bound to a sampler2D slot is
    // undefined with no validation to say so — which is why addInput refuses a volume outright.
    // Slots this pass leaves empty are written the neutral table; more than the set carries are
    // dropped and counted.
    std::vector<Resource<Texture>> volumes{};
    std::vector<Resource<FboAttachment>> attachments{};
    // Which mip level of the output's colour attachment this pass renders into. Zero for a plain
    // post-process; a chain is several passes over one output, each naming its own level.
    unsigned int outputLevel = 0;
    // Whether the pass blends over its target by source alpha, which is what every fullscreen pass
    // has always done and what a shader writing alpha 1.0 never notices. False frees the alpha
    // channel to carry data — the fog march stores its transmittance there — over a target whose
    // previous contents are undefined and must not be read by a blend.
    bool blend = true;
    // The weight the blend above mixes this pass's output in at, when the pass wants one stated
    // rather than taken from its own alpha. Unset is source-alpha blending, which is every pass
    // that ever existed; set makes the factors the constant pair, so the target becomes
    // `weight * source + (1 - weight) * destination` in all four channels and 1.0 is a plain
    // overwrite.
    //
    // It exists because alpha cannot do two jobs at once: a pass whose alpha channel is *data* —
    // the cloud dome's transmittance — and which also wants to accumulate over frames has nothing
    // left to state the mix with. Read only when `blend` is true; a pass that does not blend has no
    // weight to state. See `loadColour`, which such a pass always needs with it.
    std::optional<float> blendWeight{};
    // Whether this pass's target begins the pass holding what it last held, rather than with
    // contents the backend is free to discard. False is every pass that overwrites its whole
    // target, which is what an oversized triangle writing opaque pixels does — and it is the
    // cheaper load op, so it stays the default.
    //
    // True is the two cases that read the destination: a pass that blends against what is already
    // there, and a pass that writes only part of its target. Without it both are reading contents
    // Vulkan says are undefined — which happens to work on a desktop driver that leaves the memory
    // alone, and is exactly the kind of accident that renders correctly until it does not.
    bool loadColour = false;
    // True when this pass's target already holds the right picture and the frame may skip
    // recording it. `Camera::contentsHeld`'s twin, written by whoever owns the caching decision and
    // only read by the record loop — a cloud map that is marched every few frames rather than every
    // frame is the case it was added for. A skipped pass leaves its target exactly as it was,
    // sampleable, because that is where the last pass over it left the layout.
    bool contentsHeld = false;
    // How many vertical strips of its target one full update of this pass is spread over, and
    // which of them this recording writes. One and zero is the whole target in one pass, which is
    // every pass that ever existed.
    //
    // It is the other way to make an expensive pass cheaper per frame, and it differs from
    // `contentsHeld` in *where the saving lands*: holding leaves the frames that do run costing the
    // whole pass, so the average falls and the worst frame does not, while a strip costs a fraction
    // every frame and flattens both. Measured on the cloud dome (2026-08-27), one full update every
    // eight frames took the mean frame 10.93 -> 7.30 ms by holding and left p95 at 11.34; the whole
    // point of slicing is that last number.
    //
    // **Strips and not bands, and that is about what is being sliced rather than about rectangles.**
    // The dome's map is lat-long, so a row is a *ring of constant elevation* and the rows just above
    // the horizon carry rays tens of kilometres long while the rows at the zenith cross the shell in
    // a few hundred metres. Split by rows and one band costs several times what another does, which
    // is the spike again with extra steps. Every column carries one of each elevation, so columns
    // divide the work evenly by construction.
    //
    // The strip restricts the render area and the scissor and leaves the viewport alone: the
    // oversized triangle still maps onto the whole target, so a fragment that survives the scissor
    // gets exactly the coordinates it would have had. Pixels outside a render area are untouched by
    // definition — which is also why a sliced pass must state `loadColour`, since the strip it does
    // write is a read-modify-write of what was there.
    unsigned int sliceCount = 1;
    unsigned int slice = 0;
    // The fraction of the window this pass's own output is sized at, as a divisor: 1 is the window,
    // 2 is a half-resolution buffer that must stay half-resolution through a resize. Read where
    // tracksWindowSize is honoured; a service-owned buffer keeps its own rules as before.
    unsigned int windowSizeDivisor = 1;
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
