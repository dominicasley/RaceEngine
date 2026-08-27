module;

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

// GPU profiler zones, profile builds only: the header needs the Vulkan types above it, and every
// macro below compiles away without RACEENGINE_HAS_TRACY exactly as RaceEngineProfile.hpp's do.
#if defined(RACEENGINE_HAS_TRACY)
#include <tracy/TracyVulkan.hpp>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <shaderc/shaderc.hpp>
#include <spdlog/logger.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// An implementation partition, not an interface one: naming it `export module` put the whole
// backend — and `vulkan/vulkan.h`, `vk_mem_alloc.h` and shaderc with it — into the interface
// closure of raceengine.graphics, and so into every translation unit that imports raceengine.
// Only :RenderBackendFactoryImpl imports this, and nothing outside this module can.
module raceengine.graphics:VulkanRenderer;

import :FrameDiagnostics;
import :Frustum;
import :IRenderBackend;
import :LookupTable;
import :PostProcessing;
import :RenderContract;
import :SphericalHarmonics;
import :RenderableEntityService;
import :ShadowCascades;
import :SceneManagerService;
import :Window;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

namespace
{

constexpr auto waitForever = std::numeric_limits<uint64_t>::max();
constexpr const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
// A draw is every primitive of every renderable, once per *view*: four cascades, the occlusion
// prepass and the shading pass are six of them before a probe face is recorded, so a scene of a
// few hundred primitives spends a few thousand slots a frame. That is what decides this number,
// and getting it wrong is not a soft failure — the ring is walked in record order, so the frame
// that runs out loses everything after the first cascade rather than losing the least important
// draws. DrawData is 272 bytes now the skinning palette has a ring of its own, so eight thousand
// slots is the same 4 MiB per frame in flight that five hundred used to cost.
constexpr uint32_t drawDataRingSlots = 16384;
// The skinning palette, sized by the skinned draws alone. 128 mat4s is 8 KiB a slot, so this is
// the ring that has to stay short — and it can, because a skinned draw is a character rather than
// a body panel and the two counts are not related.
constexpr uint32_t jointDataRingSlots = 128;
// Car paint, sized by the painted draws alone for the reason the palette above is: only the body
// panels of a car take one, and a grid of cars is still a few hundred draws a frame against the
// sixteen thousand the draw ring carries. Sixty-four bytes a slot makes this ring 64 KiB.
constexpr uint32_t paintDataRingSlots = 1024;
// FrameData is per view, not per frame: Engine::step records every camera into one command
// buffer, so a single slot would have the last camera's view matrix and position shading all
// of them. 16 views in a frame is far past a split-screen game's needs at 352 bytes each.
constexpr uint32_t frameDataRingSlots = 16;

// Pool sizing: fixed and generous beats grow-on-demand here — the game's realistic
// ceiling is a few hundred materials (7 descriptors each) plus per-frame frame/draw sets
// and a handful of fullscreen input sets; these counts dwarf that at negligible cost.
constexpr uint32_t descriptorPoolMaxSets = 1024;
constexpr uint32_t descriptorPoolUniformBuffers = 512;
constexpr uint32_t descriptorPoolDynamicUniformBuffers = 16;
// Raised from 4096 when the material set grew from 6 samplers to 11 for the blended-material
// feature: this scene's two models carry 281 materials between them, which at 11 apiece is 3091
// before the shadow, occlusion and fullscreen sets are counted. Exhaustion is graceful — it reports
// DescriptorSetUnavailable and drops the draw — which is exactly the kind of failure that reads as
// "some geometry is missing sometimes", so the headroom is worth more than the handful of bytes.
constexpr uint32_t descriptorPoolCombinedImageSamplers = 8192;

// Set 0 binding 0 (vulkan-abi.md); std140-compatible, so the C++ layout is the GPU layout.
// A std140 array of these has a 16-byte element alignment, which 64 bytes already satisfies.
struct LightUbo
{
    glm::vec4 position;
    glm::vec4 diffuse;
    glm::vec4 specular;
    glm::vec4 ambientAttenuation;
};

// Set 0 binding 0, the probe half (vulkan-abi.md). One of these per light probe the frame shades
// from, std140-compatible so the C++ layout is the GPU layout.
//
// The irradiance is nine coefficients and the volume is six planes, and both are here rather than
// in a buffer of their own because a probe is per *frame* on exactly the terms the lights are: the
// shading loop reads every one of them for every fragment, and a second binding would buy a second
// descriptor and nothing else. The prefiltered radiance cannot ride along — it is an image — so it
// is one cube-array binding beside this block, and `position.w` is the slice of it this probe owns.
struct ProbeUbo
{
    std::array<glm::vec4, shCoefficientCount> irradiance;
    // xyz the influence box's world minimum, w the width of the band inside each face over which
    // this probe's weight ramps up.
    glm::vec4 boxMin;
    // xyz the world maximum, w non-zero for the scene's global probe — the one with no bound,
    // which every fragment falls back on for whatever weight the local probes left unaccounted.
    glm::vec4 boxMax;
    // xyz the point the environment was captured from, which is also the centre the specular
    // reflection is parallax-corrected about; w the probe's slice of the array.
    glm::vec4 position;
};

static_assert(sizeof(ProbeUbo) == 192);

struct FrameDataUbo
{
    glm::mat4 viewMatrix;
    glm::vec4 cameraPosition;
    glm::ivec4 lightCount;
    std::array<LightUbo, maxLights> lights;
    // The cascades. Per view rather than per frame only because the block is: every view of a
    // frame is handed the same values, and a second binding would buy nothing but a second
    // descriptor. shadowMatrices take world space straight to a shadow-map lookup — the backend's
    // depth convention and the negative-viewport y flip are already folded in
    // (shadowLookupCorrection), so the GLSL holds no convention of its own.
    std::array<glm::mat4, shadowCascadeCount> shadowMatrices;
    // Per cascade, in cascade order: where it ends along the view axis, the world size of one of
    // its texels, and normalised depth per world unit along the light. The last two are the whole
    // bias budget, which is why the shader needs no per-cascade tuning.
    glm::vec4 shadowSplits;
    glm::vec4 shadowTexelWorldSize;
    glm::vec4 shadowDepthScale;
    // x = cascades in use (0 = the view shades lit), y = which light they shadow, z = non-zero when
    // this view is a light probe capture. The sky reads z to leave the solar disc out of the cube:
    // the sun's beam reaches a surface as the directional light, and a disc in the capture would
    // hand the same sun to that surface a second time through the probe. See recordProbeFace.
    glm::ivec4 shadowParams;
    // x = light probes in use (0 = the view has no image-based lighting at all).
    glm::ivec4 probeParams;
    std::array<ProbeUbo, maxIblProbes> probes;
    // The air, appended — and appended rather than inserted for the reason the material block's
    // fifth vec4 was: a uniform block may be a prefix of the buffer backing it, so a shader that
    // declares the block only as far as the field it reads keeps matching, and a field added
    // anywhere above would silently move every probe under every one of them. Three vec4s:
    //  - x extinction at the reference height per world unit, y the reciprocal of the scale height,
    //    z that reference height, w how far along a ray the medium is integrated;
    //  - xyz the single-scatter albedo, w Henyey-Greenstein's asymmetry;
    //  - xyz a tint on the derived ambient in-scatter, w a gain on the sun's half.
    // Zero density is the disabled scene, and every shader's fog is one branch on it. See Fog.
    glm::vec4 fogDensity;
    glm::vec4 fogScatter;
    glm::vec4 fogAmbient;
    // Appended after the air, under the same prefix rule: x the simulated instant in seconds
    // (ticks x the fixed step — the one clock anything temporal in a shader may read, so a
    // capture's frame N is the same image on every machine), y the scene's rain intensity,
    // z the ground speed in metres per second and w the airflow phase — the accumulated integral
    // of speed squared, Scene::rainMotion, which is what lets a drop's drift accumulate smoothly
    // through a changing speed without the shader holding any state. Zero rain is the dry scene,
    // and every shader's rain is one branch on it that leaves z and w unread.
    glm::vec4 timeRain;
    // xyz the car's forward direction in world space, a full unit vector and not a flattened
    // heading — Scene::rainForward, whose comment carries why. What a pane's rain does with the
    // airstream depends on whether the pane faces it, and the pane's own normal cannot answer that
    // alone: the windscreen and the rear window both put their normal along the car's axis, one
    // into the airflow and one hiding from it. w is unused.
    glm::vec4 rainWind;
    // The wipers, appended under the same prefix rule and stated as geometry plus a timing law: two
    // arcs (pivot uv, inner radius, outer radius), where each blade parks and how far it sweeps, and
    // the cycle. What is deliberately absent is the blade's current *angle* — that is a closed-form
    // function of the clock the frame already carries, and so is the question the shader actually
    // asks, which is when a given point of the glass was last swept. See Wipers.
    glm::vec4 wiperArcA;
    glm::vec4 wiperArcB;
    glm::vec4 wiperSweep;  // parkA, spanA, parkB, spanB, in radians
    glm::vec4 wiperTiming; // period, sweep seconds, cycle start, blade half width
    glm::vec4 wiperPane;   // x the pane's v-to-u aspect; see Wipers::paneAspect
    // xyz the car's up direction in world space, the companion to rainWind and the other half of
    // the body frame the rain is expressed in. Appended rather than packed beside the forward axis
    // because both are now full vectors: the flattened pair they replace could not state a body
    // axis without leaning against it whenever the car pitched, and a lean multiplied by an
    // accumulating displacement is the fault that put the water on a shear. w is unused.
    //
    // **Read in the vertex stage, uniquely among the rain fields**, which is why the pass-through
    // vertex shader declares the block this far: it turns both axes into the model space of the
    // primitive being drawn, where they are constant, and hands the fragment stage a frame that
    // owes nothing to the camera, to the car's attitude, or to a screen-space derivative.
    glm::vec4 rainBody;
    // The clouds, appended under the same prefix rule: x the effective coverage, y the
    // stratus-to-cumulus type blend, z and w reserved and written zero. Zero coverage is the clear
    // sky, and every shader's clouds are one branch on it — the contract the rain and the fog
    // already honour. Unlike both of those, a probe capture keeps this field: the clouded sky is
    // exactly what a probe must photograph, since the captures are how clouds become ambient light.
    glm::vec4 cloudParams;
};

// Four split distances, four texel sizes and four depth scales ride in one vec4 each, which is
// what bounds the cascade count rather than any GPU limit.
static_assert(shadowCascadeCount <= 4);
static_assert(sizeof(LightUbo) == 64);
static_assert(sizeof(FrameDataUbo) == 2416);
static_assert(offsetof(FrameDataUbo, lightCount) == 80);
static_assert(offsetof(FrameDataUbo, lights) == 96);
static_assert(offsetof(FrameDataUbo, shadowMatrices) == 352);
static_assert(offsetof(FrameDataUbo, shadowSplits) == 608);
static_assert(offsetof(FrameDataUbo, shadowTexelWorldSize) == 624);
static_assert(offsetof(FrameDataUbo, shadowDepthScale) == 640);
static_assert(offsetof(FrameDataUbo, shadowParams) == 656);
static_assert(offsetof(FrameDataUbo, probeParams) == 672);
static_assert(offsetof(FrameDataUbo, probes) == 688);
static_assert(offsetof(FrameDataUbo, fogDensity) == 2224);
static_assert(offsetof(FrameDataUbo, fogScatter) == 2240);
static_assert(offsetof(FrameDataUbo, fogAmbient) == 2256);
static_assert(offsetof(FrameDataUbo, timeRain) == 2272);
static_assert(offsetof(FrameDataUbo, rainWind) == 2288);
static_assert(offsetof(FrameDataUbo, wiperArcA) == 2304);
static_assert(offsetof(FrameDataUbo, wiperArcB) == 2320);
static_assert(offsetof(FrameDataUbo, wiperSweep) == 2336);
static_assert(offsetof(FrameDataUbo, wiperTiming) == 2352);
static_assert(offsetof(FrameDataUbo, wiperPane) == 2368);
static_assert(offsetof(FrameDataUbo, rainBody) == 2384);
static_assert(offsetof(FrameDataUbo, cloudParams) == 2400);

// The scene's air, into the block a shading view is handed. Disabled leaves the three fields at the
// zero the block was value-initialised to, which is the block this engine uploaded before they
// existed — so a scene that states no fog is byte-identical rather than nearly so.
//
// The reciprocal of the scale height is taken here rather than in the shader because it is a
// per-scene constant and the shader would be computing it per fragment; the floor under it is what
// keeps a level that states zero from handing a division by zero to sixteen marched steps.
void uploadFog(const Fog& fog, FrameDataUbo& frameData)
{
    if (!fog.enabled)
    {
        return;
    }

    frameData.fogDensity = glm::vec4(fog.density, 1.0f / std::max(fog.scaleHeight, 1.0e-3f), fog.baseHeight,
                                     fog.maximumDistance);
    frameData.fogScatter = glm::vec4(fog.scatteringAlbedo, fog.anisotropy);
    frameData.fogAmbient = glm::vec4(fog.ambientTint, fog.sunIntensity);
}

// Set 2 binding 0, dynamic-offset (vulkan-abi.md); ring-buffered per frame in flight.
struct DrawDataUbo
{
    glm::mat4 localToWorld;
    glm::mat4 localToView;
    glm::mat4 localToScreen;
    glm::mat4 normalMatrix;
    glm::ivec4 animated;
};

static_assert(sizeof(DrawDataUbo) == 272);
// The per-draw fill writes the two regions directly into the mapped ring slot; the
// offsets are asserted so the writes cannot drift away from the shader's block layout.
static_assert(offsetof(DrawDataUbo, localToWorld) == 0);
static_assert(offsetof(DrawDataUbo, animated) == 256);

// Set 2 binding JOINT_DATA_BINDING, dynamic-offset, on a ring of its own (vulkan-abi.md). Every
// draw binds it because the layout declares it; only a draw with `animated.x != 0` allocates a
// slot, and the rest read the zeroed slot 0 they never look at.
struct JointDataUbo
{
    std::array<glm::mat4, maxJoints> jointTransforms;
};

static_assert(sizeof(JointDataUbo) == 8192);
static_assert(offsetof(JointDataUbo, jointTransforms) == 0);

// Set 2 binding PAINT_DATA_BINDING, dynamic-offset, on a ring of its own (vulkan-abi.md). Every draw
// binds it because the layout declares it; only a draw whose renderable states `Paint::enabled`
// allocates a slot, and everything else binds offset zero — the same arrangement, and the same
// reason, as the skinning palette beside it.
//
// `Paint` is per *renderable* rather than per material, so this is the one per-draw block that says
// which car is being drawn rather than what it is made of. See Paint.
struct PaintDataUbo
{
    // xyz the base coat, w how much of the surface is flake.
    glm::vec4 colour;
    // xyz the flake's own colour, w how fine it is.
    glm::vec4 flake;
    // x clearcoat strength, y its roughness, z orange peel amount, w orange peel scale.
    glm::vec4 clearcoat;
    // x how much of the material's scratch map to apply, y the same for its dirt map, **z non-zero
    // when this renderable states paint at all**. That flag is not redundant with the numbers
    // beside it: every draw binds this block and an unpainted one binds offset zero, so a shader
    // reading the zeroed block has to be able to tell "no paint" from "black paint, no clearcoat"
    // — and without it a material tagged for this shader on a renderable nobody painted renders
    // black rather than falling back to what its own textures say.
    glm::vec4 wear;
};

static_assert(sizeof(PaintDataUbo) == 64);
static_assert(offsetof(PaintDataUbo, flake) == 16);
static_assert(offsetof(PaintDataUbo, clearcoat) == 32);
static_assert(offsetof(PaintDataUbo, wear) == 48);

// Set 1 binding 0 (vulkan-abi.md); std140-compatible, so the C++ layout is the GPU layout.
// textureTransform carries a 3x3 UV transform in a mat4 slot: std140 pads a mat3's columns to
// 16 bytes each, which glm::mat3 (three packed vec3s) does not, so a mat3 member would not map.
struct MaterialDataUbo
{
    glm::vec4 baseColour;
    glm::vec4 roughMetal;
    glm::ivec4 useTextures;
    glm::ivec4 useTextures2;
    glm::mat4 textureTransform;
    // Appended, and appended rather than inserted so that the four shaders which declare the block
    // without it keep matching: a uniform block may be a prefix of the buffer backing it, and the
    // range bound is this whole struct whatever any one stage reads. A field added anywhere above
    // would silently move `textureTransform` under every one of them.
    glm::vec4 blinnPhong;
    // The blended-material feature, appended for the same reason: x..w are the four detail layers'
    // tiling, in repeats per unit of the model's own space.
    glm::vec4 detailTiling;
    // x the blend strength, y non-zero when this material states detail layers at all. A shader
    // reads y rather than testing the samplers, because every slot is written whatever the material
    // carries — an unstated layer holds the 1x1 dummy, which is white and would blend as itself.
    glm::vec4 blend;
};

static_assert(sizeof(MaterialDataUbo) == 176);
static_assert(offsetof(MaterialDataUbo, textureTransform) == 64);
static_assert(offsetof(MaterialDataUbo, blinnPhong) == 128);
static_assert(offsetof(MaterialDataUbo, detailTiling) == 144);
static_assert(offsetof(MaterialDataUbo, blend) == 160);

// The fullscreen layout's push constant (vulkan-abi.md). Eight vec4s rather than the one this used
// to be: the tone curve has a shape as well as a brightness, a pass that walks a mip chain has to be
// told which level it is on, a pass that reads a depth buffer has to be told what the camera that
// filled it was looking through, and a pass that draws weather has to be told where that camera
// stands and what time it is. A push constant rather than a block because it changes per pass and
// fills exactly the 128 bytes every Vulkan implementation guarantees, and a fullscreen shader that
// declares only a prefix of it simply does not see the rest — the range belongs to the layout.
struct FullscreenPushConstants
{
    // x exposure, y contrast, z toe, w shoulder. See ToneCurve and evaluateToneCurve, which is the
    // same function in C++.
    glm::vec4 tone;
    // x the mip level of the target being written, y the number of levels the target carries,
    // z, w the target level's width and height in texels. A pass that samples one level and writes
    // the next derives its filter footprint from these rather than from a resolution it assumed.
    glm::vec4 pass;
    // x, y the tangents of the half field of view horizontally and vertically, z the near plane,
    // w the far plane. Together they are the inverse of the projection the view was rendered with,
    // stated as the four numbers a fullscreen pass actually needs to turn a pixel and a view depth
    // back into a position — which is less than a matrix and is the whole of what the occlusion
    // gather asks for.
    glm::vec4 view;
    // `PostProcess::parameters`, verbatim: the pass's own numbers, and the one field here whose
    // meaning belongs to the shader reading it rather than to the layout. It is a slot rather than a
    // second push constant range because a range is per pipeline layout and every fullscreen pass
    // shares one.
    glm::vec4 effect;
    // Where the view stands and which way it faces, as the three columns of the view-to-world
    // rotation with the camera's position riding the w components. `view` above is the projection
    // run backwards; these are the *view transform* run backwards, and together they let a
    // fullscreen pass turn a pixel and a view depth into a world position — which is what a pass
    // that draws weather into the frame needs and the occlusion gather, working in view space,
    // never did. Appended rather than a second range for the reason `effect` is one field: every
    // fullscreen pass shares one layout, and a shader that declares the block only as far as it
    // reads keeps matching. 128 bytes total, which is exactly the size every Vulkan implementation
    // guarantees.
    glm::vec4 viewRight; // xyz the view's +x in world space, w the camera's world x
    glm::vec4 viewUp;    // xyz the view's +y in world space, w the camera's world y
    glm::vec4 viewBack;  // xyz the view's +z in world space — the view looks down -z — w the world z
    // The scene state a weather pass reads: x the simulated instant in seconds (the same tick-driven
    // clock the frame block carries, so a capture's frame N is the same image on every machine),
    // y the scene's rain intensity, zw the camera's own world velocity — x and z, in world units
    // per second, derived from where this camera stood when it last recorded a shading view. The
    // vertical component is deliberately not carried: what reads this is a streak's motion-blur
    // tilt, which the horizontal motion dominates, and the two spare lanes are what there are.
    glm::vec4 weather;
};

static_assert(sizeof(FullscreenPushConstants) == 128);
static_assert(offsetof(FullscreenPushConstants, pass) == 16);
static_assert(offsetof(FullscreenPushConstants, view) == 32);
static_assert(offsetof(FullscreenPushConstants, effect) == 48);
static_assert(offsetof(FullscreenPushConstants, viewRight) == 64);
static_assert(offsetof(FullscreenPushConstants, weather) == 112);

[[nodiscard]] constexpr uint32_t channelCount(const TextureFormat format)
{
    switch (format)
    {
    case TextureFormat::R:
        return 1;
    case TextureFormat::RG:
        return 2;
    case TextureFormat::RGB:
        return 3;
    case TextureFormat::RGBA:
    case TextureFormat::RGBA16F:
    case TextureFormat::RGBA32F:
        return 4;
    default:
        return 0;
    }
}

[[nodiscard]] constexpr uint32_t pixelComponentBytes(const PixelDataType type)
{
    switch (type)
    {
    case PixelDataType::UnsignedShort:
        return 2;
    case PixelDataType::Float:
        return 4;
    default:
        return 1;
    }
}

// Sampled images are always uploaded four-channel: the three-component formats
// (R8G8B8/R16G16B16/R32G32B32) have no universal sampling support, so the source is expanded
// on the CPU at its own precision instead — which is also what keeps a 16-bit glTF texture's
// bits, where GL keeps them by asking for a 16-bit internal format.
//
// An _SRGB format moves the decode into the sampler, so it costs nothing and it also makes mip
// generation right: the blit chain filters after decoding rather than averaging encoded bytes.
// Only the eight-bit path can carry it — Vulkan has no sRGB variant at 16 bits, and a float
// texture holds radiance, which has no encoding to undo — so a colour map at either of those
// precisions samples as though linear. Nothing in this tree authors one; a 16-bit sRGB base
// colour would need a CPU decode on the expansion pass instead.
[[nodiscard]] constexpr VkFormat sampledImageFormat(const PixelDataType type, const ColourSpace colourSpace)
{
    switch (type)
    {
    case PixelDataType::UnsignedShort:
        return VK_FORMAT_R16G16B16A16_UNORM;
    case PixelDataType::Float:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:
        return colourSpace == ColourSpace::Srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    }
}

// FboAttachment::internalFormat is the model's request, exactly as GL passes it to
// glTexImage2D. captureFormat has no Vulkan analogue: it is GL's client-side format for an
// upload the attachment path never performs.
[[nodiscard]] constexpr std::optional<VkFormat> attachmentFormat(const TextureFormat format)
{
    switch (format)
    {
    case TextureFormat::R:
        return VK_FORMAT_R8_UNORM;
    case TextureFormat::RG:
        return VK_FORMAT_R8G8_UNORM;
    case TextureFormat::RGB:
        return VK_FORMAT_R8G8B8_UNORM;
    case TextureFormat::RGBA:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case TextureFormat::RGBA16F:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case TextureFormat::RGBA32F:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case TextureFormat::DepthComponent:
    case TextureFormat::DepthComponent32F:
        return VK_FORMAT_D32_SFLOAT;
    default:
        return std::nullopt;
    }
}

// What one texel of a colour attachment is made of, for the raw readback in captureBuffers: how
// many components it carries, how many bytes it occupies, and whether those components are halves,
// floats or unsigned bytes. Zero components means a format that readback does not decode, which it
// reports and skips rather than guessing a stride for.
enum class ReadbackComponent { UnsignedByte, Half, Float };

struct ReadbackLayout
{
    unsigned int components = 0;
    unsigned int texelBytes = 0;
    ReadbackComponent component = ReadbackComponent::UnsignedByte;
};

[[nodiscard]] constexpr ReadbackLayout readbackLayout(const VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8_UNORM:
        return {1, 1, ReadbackComponent::UnsignedByte};
    case VK_FORMAT_R8G8_UNORM:
        return {2, 2, ReadbackComponent::UnsignedByte};
    case VK_FORMAT_R8G8B8_UNORM:
        return {3, 3, ReadbackComponent::UnsignedByte};
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return {4, 4, ReadbackComponent::UnsignedByte};
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return {4, 8, ReadbackComponent::Half};
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return {4, 16, ReadbackComponent::Float};
    default:
        return {};
    }
}

// CameraService keeps the GL depth convention (z in -w..w); Vulkan clips against 0..w.
// Column-major glm: z' = 0.5z + 0.5w (vulkan-abi.md).
[[nodiscard]] glm::mat4 clipCorrection()
{
    auto correction = glm::mat4(1.0f);
    correction[2][2] = 0.5f;
    correction[3][2] = 0.5f;
    return correction;
}

void ensure(const VkResult result, const char* call)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(call) + " returned VkResult " + std::to_string(static_cast<int>(result)));
    }
}

[[nodiscard]] const char* describeDeviceType(const VkPhysicalDeviceType type)
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "CPU";
    default:
        return "other";
    }
}

[[nodiscard]] const char* describeFormat(const VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "B8G8R8A8_SRGB";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "R8G8B8A8_SRGB";
    default:
        return "other";
    }
}

[[nodiscard]] const char* describeColorSpace(const VkColorSpaceKHR colorSpace)
{
    return colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ? "SRGB_NONLINEAR" : "other";
}

// Fullscreen shaders read gl_VertexIndex only: no Input variable carries a Location
// decoration. Builtins (which are also Input storage) carry BuiltIn instead, so the scan
// keys on Location-decorated ids specifically. The returned locations are the ones the
// pipeline must supply: a declared-but-unfed vertex input is invalid, so primitives
// lacking an attribute get a constant dummy binding for it.
[[nodiscard]] std::vector<uint32_t> spirvVertexInputLocations(const std::vector<uint32_t>& spirv)
{
    constexpr uint32_t opDecorate = 71;
    constexpr uint32_t opVariable = 59;
    constexpr uint32_t decorationLocation = 30;
    constexpr uint32_t storageClassInput = 1;
    constexpr size_t headerWords = 5;

    std::vector<std::pair<uint32_t, uint32_t>> decoratedLocations;
    std::vector<uint32_t> locations;
    for (auto pass = 0; pass < 2; pass++)
    {
        auto word = headerWords;
        while (word < spirv.size())
        {
            const auto wordCount = static_cast<size_t>(spirv[word] >> 16u);
            const auto opcode = spirv[word] & 0xffffu;
            if (wordCount == 0 || word + wordCount > spirv.size())
            {
                break;
            }

            if (pass == 0 && opcode == opDecorate && wordCount >= 4 && spirv[word + 2] == decorationLocation)
            {
                decoratedLocations.emplace_back(spirv[word + 1], spirv[word + 3]);
            }

            if (pass == 1 && opcode == opVariable && wordCount >= 4 && spirv[word + 3] == storageClassInput)
            {
                const auto decorated =
                    std::ranges::find(decoratedLocations, spirv[word + 2], &std::pair<uint32_t, uint32_t>::first);
                if (decorated != decoratedLocations.end() &&
                    std::ranges::find(locations, decorated->second) == locations.end())
                {
                    locations.push_back(decorated->second);
                }
            }

            word += wordCount;
        }
    }

    return locations;
}

// GL component-type enums exactly as GLTFService stores them (tinygltf keeps GL values).
constexpr int glByte = 0x1400;
constexpr int glUnsignedByte = 0x1401;
constexpr int glShort = 0x1402;
constexpr int glUnsignedShort = 0x1403;
constexpr int glFloat = 0x1406;

[[nodiscard]] constexpr uint32_t vertexComponentByteSize(const int componentType)
{
    switch (componentType)
    {
    case glByte:
    case glUnsignedByte:
        return 1;
    case glShort:
    case glUnsignedShort:
        return 2;
    default:
        return 4;
    }
}

// GL feeds non-normalized integer data to float attributes as int-valued floats; the
// *_USCALED/*_SSCALED formats are Vulkan's equivalent (plain *_UINT would require int
// shader inputs, which PassThroughVertexShader does not declare).
[[nodiscard]] constexpr VkFormat vertexAttributeFormat(const int componentType, const int size, const bool normalized)
{
    const auto index = static_cast<size_t>(std::clamp(size, 1, 4) - 1);

    switch (componentType)
    {
    case glFloat:
    {
        constexpr std::array formats = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
                                        VK_FORMAT_R32G32B32A32_SFLOAT};
        return formats[index];
    }
    case glUnsignedByte:
    {
        constexpr std::array normalizedFormats = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8_UNORM,
                                                  VK_FORMAT_R8G8B8A8_UNORM};
        constexpr std::array scaledFormats = {VK_FORMAT_R8_USCALED, VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8B8_USCALED,
                                              VK_FORMAT_R8G8B8A8_USCALED};
        return normalized ? normalizedFormats[index] : scaledFormats[index];
    }
    case glByte:
    {
        constexpr std::array normalizedFormats = {VK_FORMAT_R8_SNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8B8_SNORM,
                                                  VK_FORMAT_R8G8B8A8_SNORM};
        constexpr std::array scaledFormats = {VK_FORMAT_R8_SSCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_R8G8B8_SSCALED,
                                              VK_FORMAT_R8G8B8A8_SSCALED};
        return normalized ? normalizedFormats[index] : scaledFormats[index];
    }
    case glUnsignedShort:
    {
        constexpr std::array normalizedFormats = {VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM,
                                                  VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16A16_UNORM};
        constexpr std::array scaledFormats = {VK_FORMAT_R16_USCALED, VK_FORMAT_R16G16_USCALED,
                                              VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16A16_USCALED};
        return normalized ? normalizedFormats[index] : scaledFormats[index];
    }
    case glShort:
    {
        constexpr std::array normalizedFormats = {VK_FORMAT_R16_SNORM, VK_FORMAT_R16G16_SNORM,
                                                  VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16A16_SNORM};
        constexpr std::array scaledFormats = {VK_FORMAT_R16_SSCALED, VK_FORMAT_R16G16_SSCALED,
                                              VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R16G16B16A16_SSCALED};
        return normalized ? normalizedFormats[index] : scaledFormats[index];
    }
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

struct VertexBufferBind
{
    int bufferIndex;
    VkDeviceSize byteOffset;
};

struct VertexInputDescription
{
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    // Per binding: which MeshBuffer to bind and at which byte offset (accessor.byteOffset
    // is applied at vkCmdBindVertexBuffers time, keeping attribute offsets at 0 and clear
    // of maxVertexInputAttributeOffset).
    std::vector<VertexBufferBind> bufferBinds;
};

// Pure translation of GL glVertexAttribPointer semantics: one binding per attribute,
// stride 0 means tightly packed, and a repeated location (TEXCOORD_0/1/2 all map to
// TextureCoordinate) keeps the LAST pointer exactly as sequential GL calls would —
// Vulkan rejects duplicate locations, so last-wins is made explicit.
[[nodiscard]] constexpr VertexInputDescription
translateVertexInput(const std::span<const MeshPrimitiveAttribute> attributes)
{
    std::vector<std::pair<uint32_t, const MeshPrimitiveAttribute*>> byLocation;
    for (const auto& attribute : attributes)
    {
        if (!attribute.attributeType.has_value())
        {
            continue; // GL parity: only attributes with a recognised semantic are enabled
        }

        const auto location = static_cast<uint32_t>(attribute.attributeType.value());
        const auto existing =
            std::ranges::find(byLocation, location, &std::pair<uint32_t, const MeshPrimitiveAttribute*>::first);
        if (existing != byLocation.end())
        {
            existing->second = &attribute;
        }
        else
        {
            byLocation.emplace_back(location, &attribute);
        }
    }

    VertexInputDescription description;
    for (const auto& [location, attribute] : byLocation)
    {
        const auto binding = static_cast<uint32_t>(description.bindings.size());
        const auto stride = attribute->stride > 0 ? static_cast<uint32_t>(attribute->stride)
                                                  : static_cast<uint32_t>(attribute->size) *
                                                        vertexComponentByteSize(attribute->componentType);

        description.bindings.push_back(VkVertexInputBindingDescription{binding, stride, VK_VERTEX_INPUT_RATE_VERTEX});
        description.attributes.push_back(VkVertexInputAttributeDescription{
            location, binding, vertexAttributeFormat(attribute->componentType, attribute->size, attribute->normalized),
            0});
        description.bufferBinds.push_back(
            VertexBufferBind{attribute->bufferIndex, static_cast<VkDeviceSize>(attribute->offset)});
    }

    return description;
}

// Compile-time walk of test.glb mesh "Cube" primitive 0 exactly as GLTFService emits it
// (tinygltf map order NORMAL, POSITION, TANGENT, TEXCOORD_0/1/2; one buffer view each,
// tight strides, accessor byteOffset 0): the three TEXCOORD sets collapse to location 1
// keeping the last view (4), formats and strides match the accessors.
static_assert(
    []
    {
        const std::array attributes = {
            MeshPrimitiveAttribute{.size = 3,
                                   .type = 3,
                                   .componentType = glFloat,
                                   .stride = 12,
                                   .bufferIndex = 1,
                                   .attributeType = PrimitiveAttributeType::Normal,
                                   .normalized = false,
                                   .offset = 0},
            MeshPrimitiveAttribute{.size = 3,
                                   .type = 3,
                                   .componentType = glFloat,
                                   .stride = 12,
                                   .bufferIndex = 0,
                                   .attributeType = PrimitiveAttributeType::Position,
                                   .normalized = false,
                                   .offset = 0},
            MeshPrimitiveAttribute{.size = 4,
                                   .type = 4,
                                   .componentType = glFloat,
                                   .stride = 16,
                                   .bufferIndex = 5,
                                   .attributeType = PrimitiveAttributeType::Tangent,
                                   .normalized = false,
                                   .offset = 0},
            MeshPrimitiveAttribute{.size = 2,
                                   .type = 2,
                                   .componentType = glFloat,
                                   .stride = 8,
                                   .bufferIndex = 2,
                                   .attributeType = PrimitiveAttributeType::TextureCoordinate,
                                   .normalized = false,
                                   .offset = 0},
            MeshPrimitiveAttribute{.size = 2,
                                   .type = 2,
                                   .componentType = glFloat,
                                   .stride = 8,
                                   .bufferIndex = 3,
                                   .attributeType = PrimitiveAttributeType::TextureCoordinate,
                                   .normalized = false,
                                   .offset = 0},
            MeshPrimitiveAttribute{.size = 2,
                                   .type = 2,
                                   .componentType = glFloat,
                                   .stride = 8,
                                   .bufferIndex = 4,
                                   .attributeType = PrimitiveAttributeType::TextureCoordinate,
                                   .normalized = false,
                                   .offset = 0},
        };
        const auto description = translateVertexInput(attributes);
        return description.attributes.size() == 4 && description.bindings.size() == 4 &&
               description.bufferBinds.size() == 4 && description.attributes[0].location == 2 &&
               description.attributes[0].format == VK_FORMAT_R32G32B32_SFLOAT &&
               description.attributes[1].location == 0 &&
               description.attributes[1].format == VK_FORMAT_R32G32B32_SFLOAT &&
               description.attributes[2].location == 3 &&
               description.attributes[2].format == VK_FORMAT_R32G32B32A32_SFLOAT &&
               description.attributes[3].location == 1 && description.attributes[3].format == VK_FORMAT_R32G32_SFLOAT &&
               description.bindings[0].stride == 12 && description.bindings[2].stride == 16 &&
               description.bindings[3].stride == 8 && description.bufferBinds[1].bufferIndex == 0 &&
               description.bufferBinds[3].bufferIndex == 4 && description.attributes[3].offset == 0;
    }(),
    "vertex input translation must match the test.glb Cube primitive layout");

// Skinned-mesh shape: unsigned-byte joints with glTF stride 0 (tight) become USCALED
// int-valued floats with a computed stride, and the accessor offset moves to bind time.
static_assert(
    []
    {
        const std::array attributes = {MeshPrimitiveAttribute{.size = 4,
                                                              .type = 4,
                                                              .componentType = glUnsignedByte,
                                                              .stride = 0,
                                                              .bufferIndex = 7,
                                                              .attributeType = PrimitiveAttributeType::Joint,
                                                              .normalized = false,
                                                              .offset = 96}};
        const auto description = translateVertexInput(attributes);
        return description.attributes.size() == 1 && description.attributes[0].location == 4 &&
               description.attributes[0].format == VK_FORMAT_R8G8B8A8_USCALED && description.bindings[0].stride == 4 &&
               description.bufferBinds[0].bufferIndex == 7 && description.bufferBinds[0].byteOffset == 96 &&
               description.attributes[0].offset == 0;
    }(),
    "vertex input translation must derive tight strides and USCALED joint formats");

// Index accessors keep their GL component-type enums; VK_INDEX_TYPE_UINT8 needs an
// extension this backend does not enable, so unsigned-byte indices are reported and
// skipped (no asset in the sandbox uses them).
constexpr int glUnsignedInt = 0x1405;

[[nodiscard]] constexpr std::optional<VkIndexType> indexType(const int componentType)
{
    switch (componentType)
    {
    case glUnsignedShort:
        return VK_INDEX_TYPE_UINT16;
    case glUnsignedInt:
        return VK_INDEX_TYPE_UINT32;
    default:
        return std::nullopt;
    }
}

// MeshPrimitive::mode carries the GL/glTF draw mode; GL_LINE_LOOP has no Vulkan topology.
[[nodiscard]] constexpr std::optional<VkPrimitiveTopology> primitiveTopology(const int mode)
{
    switch (mode)
    {
    case 0:
        return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 1:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 4:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 5:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 6:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    default:
        return std::nullopt;
    }
}

struct StageAccess
{
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 access;
};

// Layout is the single piece of state tracked per image, so the barrier's stage/access
// masks are derived from it: every layout this backend uses has exactly one producer or
// consumer kind.
[[nodiscard]] constexpr StageAccess stageAccessFor(const VkImageLayout layout)
{
    switch (layout)
    {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT};
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return {VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE};
    default:
        return {VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE};
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugMessageHandler(const VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         const VkDebugUtilsMessageTypeFlagsEXT type,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                         void* userData)
{
    auto& logger = *static_cast<spdlog::logger*>(userData);
    const char* message = (callbackData != nullptr && callbackData->pMessage != nullptr) ? callbackData->pMessage : "";

    switch (severity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        logger.debug("Vulkan debug message [type 0x{:x}]: {}", type, message);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        logger.warn("Vulkan debug message [type 0x{:x}]: {}", type, message);
        break;
    default:
        logger.error("Vulkan debug message [type 0x{:x}]: {}", type, message);
        break;
    }

    return VK_FALSE;
}

[[nodiscard]] VkDebugUtilsMessengerCreateInfoEXT makeDebugMessengerCreateInfo(spdlog::logger& logger)
{
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{};
    messengerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messengerInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messengerInfo.pfnUserCallback = vulkanDebugMessageHandler;
    messengerInfo.pUserData = &logger;
    return messengerInfo;
}

void transitionImage(const VkCommandBuffer commandBuffer, const VkImage image, const VkImageLayout oldLayout,
                     const VkImageLayout newLayout, const VkPipelineStageFlags2 srcStage,
                     const VkAccessFlags2 srcAccess, const VkPipelineStageFlags2 dstStage,
                     const VkAccessFlags2 dstAccess, const VkImageSubresourceRange& subresourceRange)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = subresourceRange;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void transitionImage(const VkCommandBuffer commandBuffer, const VkImage image, const VkImageLayout oldLayout,
                     const VkImageLayout newLayout, const VkPipelineStageFlags2 srcStage,
                     const VkAccessFlags2 srcAccess, const VkPipelineStageFlags2 dstStage,
                     const VkAccessFlags2 dstAccess)
{
    transitionImage(commandBuffer, image, oldLayout, newLayout, srcStage, srcAccess, dstStage, dstAccess,
                    VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
}

// Where compiled artefacts live between runs: XDG_CACHE_HOME (or ~/.cache) /raceengine/. Empty
// when neither variable resolves, in which case every cache is in-memory only for the run.
std::string engineCacheDirectory()
{
    const char* xdgCache = std::getenv("XDG_CACHE_HOME");
    std::string base = xdgCache != nullptr && *xdgCache != '\0' ? std::string(xdgCache) : std::string();
    if (base.empty())
    {
        const char* home = std::getenv("HOME");
        if (home == nullptr || *home == '\0')
        {
            return {};
        }
        base = std::string(home) + "/.cache";
    }

    const auto directory = base + "/raceengine";
    static_cast<void>(mkdir(base.c_str(), 0755));
    static_cast<void>(mkdir(directory.c_str(), 0755));

    return directory;
}

std::string pipelineCacheFilePath()
{
    const auto directory = engineCacheDirectory();
    return directory.empty() ? directory : directory + "/vulkan-pipeline.cache";
}

uint64_t fnv1a64(const std::string_view text)
{
    auto hash = 14695981039346656037ull;
    for (const auto character : text)
    {
        hash ^= static_cast<unsigned char>(character);
        hash *= 1099511628211ull;
    }

    return hash;
}

// The fallback name a view labels its pass with when its camera carries no debugName.
constexpr const char* describeCameraRole(const CameraRole role)
{
    switch (role)
    {
    case CameraRole::ShadowCascade:
        return "cascade";
    case CameraRole::ProbeFace:
        return "probe face";
    case CameraRole::DepthNormalPrepass:
        return "prepass";
    case CameraRole::Scene:
        break;
    }

    return "scene";
}

// A GPU profiler zone over the commands recorded in the enclosing scope, named at run time.
// Contained in this file on purpose: macros do not cross an import, and no other unit records.
#if defined(RACEENGINE_HAS_TRACY)
#define RACEENGINE_GPU_ZONE(varname, commandBufferArg, nameArg)                                                        \
    TracyVkZoneTransient(tracyGpuContext, varname, commandBufferArg, nameArg, true)
#define RACEENGINE_GPU_COLLECT(commandBufferArg) TracyVkCollect(tracyGpuContext, commandBufferArg)
#else
#define RACEENGINE_GPU_ZONE(varname, commandBufferArg, nameArg) static_cast<void>(nameArg)
#define RACEENGINE_GPU_COLLECT(commandBufferArg) static_cast<void>(commandBufferArg)
#endif

// The prefilter's own shaders, as source. Engine-owned rather than game assets: a game authors
// what the world looks like, and how a captured environment is reduced to a roughness chain is no
// more its business than the layout of the draw-data ring is. They go through the same shaderc
// path as everything else, so they receive the same contract macros.
constexpr const char* probePrefilterVertexSource = R"GLSL(#version 450

// The oversized triangle, from gl_VertexIndex: no vertex buffers, no bindings, nothing for a
// pipeline to have to be fed.
void main()
{
    const vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

constexpr const char* probePrefilterFragmentSource = R"GLSL(#version 450

layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform samplerCube radianceCube;

layout(push_constant) uniform PrefilterParams {
    // x roughness, y the face being filtered, z the target's edge length, w the source's.
    vec4 value;
} params;

const float PI = 3.141592653589793;
// Enough to converge at this resolution, given that each sample reads a mip chosen from its own
// footprint: the chain is what removes the variance, and more samples past this buy smoothness
// the mip has already provided.
const uint sampleCount = 64u;

// The hardware's own face-to-direction mapping, and it has to be: what this pass writes is read
// back by a sampler using it, and the C++ irradiance projection spells the same table again
// (see :SphericalHarmonics). Three copies of one convention, none of which may drift.
vec3 faceDirection(int face, vec2 st)
{
    if (face == 0) { return vec3(1.0, -st.y, -st.x); }
    if (face == 1) { return vec3(-1.0, -st.y, st.x); }
    if (face == 2) { return vec3(st.x, 1.0, st.y); }
    if (face == 3) { return vec3(st.x, -1.0, -st.y); }
    if (face == 4) { return vec3(st.x, -st.y, 1.0); }
    return vec3(-st.x, -st.y, -1.0);
}

float radicalInverse(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float distributionGGX(float NdH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// A half-vector drawn from the GGX distribution about `normal`, for the low-discrepancy point xi.
vec3 importanceSampleGGX(vec2 xi, float roughness, vec3 normal)
{
    float a = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    // Not named `half`: that is a reserved word in GLSL and the compile error it produces names
    // the line and not the reason.
    vec3 tangentSpace = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 reference = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(reference, normal));
    vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * tangentSpace.x + bitangent * tangentSpace.y + normal * tangentSpace.z);
}

void main()
{
    float roughness = params.value.x;
    int face = int(params.value.y);
    float targetSize = params.value.z;
    float sourceSize = params.value.w;

    // gl_FragCoord is at the texel centre, so this is the same (s, t) the C++ side derives from a
    // texel index — which is why the pass records a positive-height viewport, unlike every other
    // pass here: framebuffer row 0 has to be the face's first stored row.
    vec2 st = (gl_FragCoord.xy / targetSize) * 2.0 - 1.0;
    vec3 normal = normalize(faceDirection(face, st));

    // The split-sum approximation's one assumption: the view direction is the normal, so a
    // prefiltered level depends on roughness alone and not on where it is looked at from. It is
    // what costs grazing reflections their stretch, and what makes this a chain of six images
    // rather than a function of two angles.
    if (roughness <= 0.0)
    {
        fragColor = vec4(textureLod(radianceCube, normal, 0.0).rgb, 1.0);
        return;
    }

    // The solid angle one source texel covers, which the mip selection below is measured against.
    float texelSolidAngle = 4.0 * PI / (6.0 * sourceSize * sourceSize);

    vec3 total = vec3(0.0);
    float totalWeight = 0.0;

    for (uint index = 0u; index < sampleCount; index++)
    {
        vec2 xi = vec2(float(index) / float(sampleCount), radicalInverse(index));
        vec3 halfVector = importanceSampleGGX(xi, roughness, normal);
        vec3 light = normalize(2.0 * dot(normal, halfVector) * halfVector - normal);

        float NdL = dot(normal, light);
        if (NdL <= 0.0)
        {
            continue;
        }

        float NdH = max(dot(normal, halfVector), 0.0);
        // The density this sample was drawn at, and from it the solid angle it stands for. A
        // sample that stands for more than one source texel reads a coarser mip, which is what
        // stops a small bright feature surviving the filter as a firefly.
        //
        // The general density is D * NdH / (4 * dot(H, V)); with the view direction taken as the
        // normal, dot(H, V) is NdH and the two cancel to D / 4.
        float density = distributionGGX(NdH, roughness) * 0.25 + 0.0001;
        float sampleSolidAngle = 1.0 / (float(sampleCount) * density);
        float level = 0.5 * log2(sampleSolidAngle / texelSolidAngle);

        total += textureLod(radianceCube, light, max(level, 0.0)).rgb * NdL;
        totalWeight += NdL;
    }

    fragColor = vec4(total / max(totalWeight, 0.0001), 1.0);
}
)GLSL";

} // namespace

// How a fullscreen pass mixes with what is already in its target. Three states rather than the
// bool this replaces, because a pass whose alpha is carrying *data* still has a reason to blend:
// the cloud dome's alpha is a transmittance and its accumulation weight cannot ride in it, so the
// weight becomes the pipeline's blend constant instead. `SourceAlpha` is every pass that ever
// existed and `None` is the fog march's straight overwrite.
enum class FullscreenBlend : uint8_t
{
    None,
    SourceAlpha,
    Constant
};

class VulkanRenderer : public IRenderBackend
{
private:
    static constexpr uint32_t framesInFlight = 2;

    struct FrameInFlight
    {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkBuffer frameDataBuffer = VK_NULL_HANDLE;
        VmaAllocation frameDataAllocation = nullptr;
        void* frameDataMapped = nullptr;
        VkBuffer drawDataBuffer = VK_NULL_HANDLE;
        VmaAllocation drawDataAllocation = nullptr;
        void* drawDataMapped = nullptr;
        VkBuffer jointDataBuffer = VK_NULL_HANDLE;
        VmaAllocation jointDataAllocation = nullptr;
        void* jointDataMapped = nullptr;
        VkBuffer paintDataBuffer = VK_NULL_HANDLE;
        VmaAllocation paintDataAllocation = nullptr;
        void* paintDataMapped = nullptr;
        VkDescriptorSet frameDataSet = VK_NULL_HANDLE;
        VkDescriptorSet drawDataSet = VK_NULL_HANDLE;
        uint32_t drawDataSlotsUsed = 0;
        uint32_t jointDataSlotsUsed = 0;
        uint32_t paintDataSlotsUsed = 0;
        uint32_t frameDataSlotsUsed = 0;
        // Byte offset of the view currently being recorded; every draw in that view binds it.
        VkDeviceSize frameDataOffset = 0;
    };

    struct ShaderObject
    {
        VkShaderModule vertexModule = VK_NULL_HANDLE;
        VkShaderModule fragmentModule = VK_NULL_HANDLE;
        bool fullscreen = false;
        // Vertex input locations the vertex shader declares; the scene pipeline must feed
        // every one of them, dummy binding included.
        std::vector<uint32_t> vertexInputLocations;
        // Fullscreen pipelines exist per reachable target format (presenter writes the
        // swapchain, post-processes write RGBA16F); the draw path picks by actual target.
        // Scene pipelines are vertex-input-dependent and are built by the draw path.
        VkPipeline swapchainTargetPipeline = VK_NULL_HANDLE;
    };

    // Cube maps and FBO attachments share this shape; attachments use the shared
    // attachmentSampler and keep their own field VK_NULL_HANDLE.
    struct ImageResource
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t mipLevels = 1;
        uint32_t width = 0;
        uint32_t height = 0;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        // The tracked current layout, one entry per mip level: the draw path derives every barrier
        // from it. Per level rather than per image because a chain is written one level at a time
        // from the level above it, so mid-chain the image genuinely holds two layouts at once and a
        // single field would be a lie about half of it. The light probe's scratch cube was the
        // first place that happened and used to keep the bookkeeping by hand; it no longer does.
        // Always mipLevels long, so an empty vector is an image that was never created.
        std::vector<VkImageLayout> layouts;
        // One single-level 2D view per mip, for the levels a pass renders into or samples on its
        // own. Empty on a single-level image, whose `view` already is exactly that.
        std::vector<VkImageView> levelViews;
    };

    struct FboResource
    {
        std::vector<unsigned int> attachmentIds;
    };

    struct BufferResource
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
    };

    // One UBO + one descriptor set per (material, environment) pair, allocated on first
    // use and kept for the process lifetime: material contents are load-time constant.
    struct MaterialResource
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    // One pipeline a primitive has been drawn with, and the vertex buffers that pipeline is fed.
    //
    // The two belong together and may not be separated. A pipeline consumes exactly the locations
    // its vertex shader declares (see scenePipeline), so which of the primitive's buffers land at
    // which binding index — and how many dummy bindings follow them — is a property of the
    // *shader*, not of the primitive: a mesh whose attributes are authored NORMAL-first feeds
    // binding 0 from its position buffer under a depth-only shader and from its normal buffer
    // under a shading one. Holding one buffer list per primitive meant the last pipeline built for
    // it decided what every pipeline for it was fed, and a shadow cascade recorded after any
    // shading view then read its positions out of the normal buffer: unit-length vectors, so the
    // whole world collapsed onto one texel at the centre of the depth map and every fragment
    // sampled it as lit.
    struct ResolvedPipeline
    {
        unsigned int shaderId = 0;
        VkCullModeFlags cullMode = 0;
        VkFormat colorFormat = VK_FORMAT_UNDEFINED;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        // Whether the fragment's alpha is a coverage to blend with. True for every pass that writes
        // a colour, which was every pass there was; false for one that writes data.
        bool blend = true;
        // Whether the draw contributes to the depth buffer as well as reading it. False only for
        // blended geometry in a shading view: a transparent surface does not hide what is behind
        // it, so recording it as an occluder is a claim the picture does not make.
        bool depthWrite = true;
        // VK_NULL_HANDLE is a build that failed and is not retried.
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::vector<VkBuffer> boundBuffers;
        std::vector<VkDeviceSize> boundOffsets;
    };

    // The Vulkan counterpart of the GL path's per-primitive VAO: everything a draw needs
    // that does not change once the primitive is uploaded. The pipeline and the buffers it is
    // fed resolve on first use, because they also depend on the shader the *view* draws with.
    struct PrimitiveBinding
    {
        VertexInputDescription input;
        std::vector<VkBuffer> vertexBuffers;
        std::vector<VkDeviceSize> vertexOffsets;
        std::string signature;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        VkIndexType indexType = VK_INDEX_TYPE_UINT16;
        uint32_t indexCount = 0;
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        bool drawable = false;
        // One entry per (shader, cull mode, target formats) this primitive has been drawn with:
        // the whole of what a pipeline's identity adds to the primitive's own vertex input. A
        // vector rather than a map because a primitive is drawn through two or three of these in a
        // frame — its material's shader on screen and a cascade's depth shader — and a linear scan
        // over three entries costs less than hashing the tuple.
        std::vector<ResolvedPipeline> pipelines;
    };

    // What the presenter drew last: captureFrame replays exactly this pass onto a freshly
    // acquired image, because the presented one may not be touched again.
    struct PresentPass
    {
        unsigned int shaderId = 0;
        unsigned int attachmentImageId = 0;
        // The presenter's own numbers and its colour grade, kept with the pass because the capture
        // replays it: a frame dumped with a different grade from the one presented would be a
        // picture of a frame this engine never drew.
        glm::vec4 parameters{0.0f};
        unsigned int lookupTableImageId = 0;
    };

    // A GPU object the engine has finished with, waiting for the GPU to finish with it too.
    // Deferral is not an optimisation here: two frames are in flight, so at the moment a release
    // is asked for there are up to `framesInFlight` submissions still executing that may name the
    // image, buffer or descriptor set being released. Destroying it at the call is undefined
    // behaviour that a validation layer catches and a driver usually does not.
    //
    // readyAt is a value of submittedFrames (see below), not a fence: fences are per frame slot
    // and get reset, so they cannot say "later than this". The rule is in retire().
    struct RetiredResource
    {
        uint64_t readyAt = 0;
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation imageAllocation = nullptr;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation bufferAllocation = nullptr;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };

    // Declaration order is dependency order; the destructor tears down in reverse.
    spdlog::logger& logger;
    // Every skipped draw, view or pass is counted here instead of at a per-site boolean; see
    // FrameDiagnostics for why the sites do not throttle themselves any more.
    FrameDiagnostics& diagnostics;
    IWindow& window;
    // The window's Vulkan half, taken separately: the same object as `window`, through the seam
    // that carries surface creation. IWindow does not, because IWindow is what the game holds.
    IVulkanSurfaceSource& surfaceSource;
    MemoryStorageService& memoryStorageService;
    RenderableEntityService& renderableEntityService;
    SceneManagerService& sceneManagerService;
    bool validationLayerEnabled = false;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessenger = nullptr;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits deviceLimits{};
    bool samplerAnisotropySupported = false;
    uint32_t graphicsQueueFamily = 0;
    uint32_t presentQueueFamily = 0;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VmaAllocator allocator = nullptr;
    // Every vkCreateGraphicsPipelines call goes through this cache, and the cache is persisted
    // across runs — which is what turns the first frame's several seconds of driver compilation
    // into a warm start. VK_NULL_HANDLE only if creation itself failed; pipelines still build.
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    // Pass labels for captures and profilers. Loaded whenever the loader offers debug utils —
    // which is independent of the validation layer — and null otherwise, so labelling is a no-op
    // on a machine without it rather than a branch every call site carries.
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel = nullptr;
#if defined(RACEENGINE_HAS_TRACY)
    TracyVkCtx tracyGpuContext = nullptr;
#endif

    // Opens a debug label over the commands recorded in its scope and closes it on every exit
    // path. Inert when the extension is absent: `end` is captured only if `begin` fired.
    class GpuPassLabel
    {
        VkCommandBuffer commandBuffer;
        PFN_vkCmdEndDebugUtilsLabelEXT end;

    public:
        GpuPassLabel(const VulkanRenderer& renderer, const VkCommandBuffer commandBuffer, const char* name) :
            commandBuffer(commandBuffer),
            end(renderer.cmdBeginDebugUtilsLabel != nullptr ? renderer.cmdEndDebugUtilsLabel : nullptr)
        {
            if (renderer.cmdBeginDebugUtilsLabel != nullptr)
            {
                VkDebugUtilsLabelEXT label{};
                label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
                label.pLabelName = name;
                renderer.cmdBeginDebugUtilsLabel(commandBuffer, &label);
            }
        }
        GpuPassLabel(const GpuPassLabel&) = delete;
        GpuPassLabel& operator=(const GpuPassLabel&) = delete;
        ~GpuPassLabel()
        {
            if (end != nullptr)
            {
                end(commandBuffer);
            }
        }
    };
    VkDescriptorSetLayout frameDataSetLayout = VK_NULL_HANDLE;  // scene set 0
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;   // scene set 1
    VkDescriptorSetLayout drawDataSetLayout = VK_NULL_HANDLE;   // scene set 2, dynamic
    VkDescriptorSetLayout shadowSetLayout = VK_NULL_HANDLE;     // scene set 3
    VkDescriptorSetLayout fullscreenSetLayout = VK_NULL_HANDLE; // fullscreen set 0
    VkPipelineLayout scenePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout fullscreenPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkCommandPool uploadCommandPool = VK_NULL_HANDLE;
    // The open upload batch. Every beginUploadCommands returns this one command buffer and
    // finishUploadCommands leaves it open; flushUploadCommands submits it once — from endFrame,
    // ahead of the frame's own submit, which is the same execution order the old per-upload fence
    // gave without the hundreds of blocking round trips docs/renderer.md measured as most of the
    // circuit's load. Staging buffers retire into the list beside it because they must outlive
    // the copies they feed. Mutable because the upload path is const and the batch is bookkeeping.
    mutable VkCommandBuffer uploadBatchCommands = VK_NULL_HANDLE;
    mutable std::vector<std::pair<VkBuffer, VmaAllocation>> uploadBatchStaging;
    VkSampler attachmentSampler = VK_NULL_HANDLE;
    // The cloud dome map's own sampler: the map is lat-long, so its u axis is an azimuth that
    // wraps at 360 degrees and must REPEAT or the sky seams at due +z — the shared sampler above
    // clamps, which every other attachment read wants. v stays clamped: the poles do not wrap.
    VkSampler cloudMapSampler = VK_NULL_HANDLE;
    VkDeviceSize drawDataStride = 0;
    VkDeviceSize jointDataStride = 0;
    VkDeviceSize paintDataStride = 0;
    VkDeviceSize frameDataStride = 0;
    VkSurfaceFormatKHR surfaceFormat{};
    VkExtent2D swapchainExtent{};
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    // Present-wait semaphores are per swapchain image: a per-frame one could still be in
    // use by the presentation engine when its frame slot comes around again.
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::array<FrameInFlight, framesInFlight> frames{};
    size_t frameIndex = 0;
    VkExtent2D requestedExtent{};
    bool recreateNeeded = false;
    // One monotonic id space (from 1) across every registry: any logged id names exactly
    // one renderer resource, and a cross-kind stale lookup misses loudly.
    mutable unsigned int nextResourceId = 1;
    std::unordered_map<unsigned int, ShaderObject> shaderObjects;
    mutable std::unordered_map<unsigned int, ImageResource> imageResources;
    mutable std::unordered_map<unsigned int, FboResource> fboResources;
    std::unordered_map<unsigned int, BufferResource> bufferResources;
    std::unordered_map<unsigned int, PrimitiveBinding> primitiveBindings;
    // Key packs the material's Resource id with the environment cube map's image id: the
    // scene environment is the fallback binding, so it is part of the set's identity.
    std::unordered_map<uint64_t, MaterialResource> materialResources;
    // Fullscreen input sets, keyed by the whole array of (image, level) the set was written with
    // rather than by one image id: the set carries postProcessInputCount samplers now, so its
    // identity is the list. A vector scanned linearly, like shadowSets and for the same two
    // reasons — a running game holds one entry per post-process and per presenter, and `<map>` in
    // a second global module fragment breaks the sandbox link on clang-19 (see CLAUDE.md).
    //
    // The key has to be something stable, because the pool never frees a set that is still cached:
    // a pass's inputs are fixed when the scene is built, so the number of distinct keys is the
    // number of passes and not the number of frames.
    struct FullscreenSetKey
    {
        std::array<PostProcessBinding, postProcessInputCount> inputs;
        unsigned int lookupTable;
        std::array<unsigned int, postProcessVolumeCount> volumes;

        [[nodiscard]] bool operator==(const FullscreenSetKey&) const = default;
    };
    mutable std::vector<std::pair<FullscreenSetKey, VkDescriptorSet>> attachmentSets;
    // Cascade sets, keyed by the whole tuple of depth image ids and the three images bound beside
    // them — the occlusion, the behind copy and the cloud map. A vector rather than a map because
    // there are two live entries in practice — the shading view's and the fallback — and `<map>` in
    // a second global module fragment breaks the sandbox link on clang-19 (see CLAUDE.md).
    struct ShadowSetKey
    {
        std::array<unsigned int, shadowCascadeCount> cascades;
        unsigned int occlusion;
        unsigned int behind;
        unsigned int cloudMap;

        [[nodiscard]] bool operator==(const ShadowSetKey&) const = default;
    };
    mutable std::vector<std::pair<ShadowSetKey, VkDescriptorSet>> shadowSets;
    // The behind copy each shading camera's blended draws sample, keyed by the colour attachment it
    // copies: destroyImageResource retires the copy with its attachment, so a resize takes both.
    // Renderer-internal rather than an FboAttachment because nothing renders into it — it is
    // written by blit and owns nothing a framebuffer would.
    mutable std::unordered_map<unsigned int, unsigned int> sceneBehindImages;
    // Where each Scene camera stood when it last recorded a shading view, and on what simulated
    // instant: the difference is the camera's own velocity, which the weather push constant
    // carries so a streak can smear along the apparent fall. Keyed by the camera's address, which
    // is stable — Scene::cameras is an add-only deque — and Scene-role cameras only, because the
    // prepass and probe views are built on the stack and a recycled stack address is not a camera.
    // A function of tick-driven positions, so captures reproduce.
    struct CameraTrack
    {
        glm::vec3 position;
        double simulationTime;
    };
    std::unordered_map<const Camera*, CameraTrack> cameraTracks;
    std::unordered_map<std::string, VkPipeline> scenePipelines;
    // Fullscreen pipelines that render into an offscreen attachment, keyed by shader id and
    // the target's format: the format comes from FboAttachment::internalFormat, which is not
    // known when the shader object is built.
    std::unordered_map<uint64_t, VkPipeline> offscreenPipelines;
    // Light probes. Capture draws into one scratch cube — one, because the scheduler captures one
    // probe at a time and a per-probe scratch would be eight images held to hold nothing — and the
    // prefiltered result lands in one slice of a cube *array*, which is what lets the shading loop
    // pick a probe with a dynamic index instead of needing a descriptor per probe.
    unsigned int probeRadianceImageId = 0;
    unsigned int probeDepthImageId = 0;
    unsigned int probeSpecularImageId = 0;
    // The render targets: one 2D view per face of the scratch cube's top mip, and one per
    // (slice, mip, face) of the array. Rendering into a cube means rendering into a single-layer
    // 2D view of it, so every one of these has to exist before the pass that writes through it.
    std::array<VkImageView, 6> probeRadianceFaceViews{};
    std::vector<VkImageView> probeSpecularFaceViews;
    VkDescriptorSetLayout probePrefilterSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout probePrefilterPipelineLayout = VK_NULL_HANDLE;
    VkPipeline probePrefilterPipeline = VK_NULL_HANDLE;
    VkShaderModule probePrefilterVertexModule = VK_NULL_HANDLE;
    VkShaderModule probePrefilterFragmentModule = VK_NULL_HANDLE;
    VkDescriptorSet probeRadianceSet = VK_NULL_HANDLE;
    VkSampler probeSampler = VK_NULL_HANDLE;
    // Where the irradiance projection reads its radiance from. Host-visible and permanently
    // mapped: it is written by a GPU copy and read by the CPU, once per probe capture.
    BufferResource probeReadbackBuffer{};
    void* probeReadbackMapped = nullptr;
    unsigned int probeSlicesUsed = 0;
    // The irradiance the last completed readback produced, per slice. Held here rather than
    // written straight through to the LightProbe because the write happens frames after the copy
    // was recorded, and a pointer into a scene's deque held across those frames would be a
    // pointer into a scene that may have been destroyed. The probe collects it instead.
    std::array<ShIrradiance, maxIblProbes> probeIrradiance{};
    // Which submission has to complete before probeReadbackBuffer holds this capture's radiance.
    // Same clock and same rule as the retirement queue: see retire().
    uint64_t probeReadbackReadyAt = 0;
    // The single texel a camera's luminance chain reduces a frame into, on its way to the CPU.
    // Host-visible and permanently mapped, exactly as the probe's is, and eight bytes long: one
    // RGBA16F texel, of which the meter wrote the red channel.
    BufferResource luminanceReadbackBuffer{};
    void* luminanceReadbackMapped = nullptr;
    // Which image's chain the buffer is currently holding a copy of, and the submission that has
    // to complete before it holds it. One buffer and one pending copy for every camera that
    // meters, for the reason the probe path has one: the cameras are recorded in a fixed order, so
    // which of them gets the buffer on which frame is a function of the frame number, and a second
    // camera simply waits its turn rather than needing a second buffer to be deterministic in.
    std::optional<unsigned int> luminanceReadbackOwner{};
    uint64_t luminanceReadbackReadyAt = 0;
    unsigned int dummyTextureId = 0;
    unsigned int neutralLookupTableId = 0;
    unsigned int dummyCubeMapId = 0;
    unsigned int dummyShadowMapId = 0;
    mutable VkDescriptorSet dummyShadowSet = VK_NULL_HANDLE;
    BufferResource dummyVertexBuffer{};
    std::optional<unsigned int> sceneEnvironmentImageId;
    // Frame recording state. The frame belongs to Engine::step: beginFrame acquires the
    // image and opens the command buffer, recordView and recordPresent only record into it,
    // and endFrame is the one place that submits and presents.
    bool frameOpen = false;
    bool swapchainPassRecorded = false;
    // The simulated instant this frame renders, in seconds, as beginFrame was told it. Every view
    // uploads it into the frame block, which is what makes shader time per-frame and deterministic
    // under capture rather than a reading of any clock of this backend's own.
    double frameSimulationTime = 0.0;
    uint32_t currentImageIndex = 0;
    std::optional<PresentPass> lastPresentPass;
    // Submissions this backend has issued, ever. It is the clock the retirement queue is keyed
    // to, and it only advances in submitAndPresent — the one place that submits.
    uint64_t submittedFrames = 0;
    mutable std::vector<RetiredResource> retiredResources;
    // Not a diagnostic: the one-shot info line that says the backend reached a recorded scene
    // pass, which the smoke gate reads. Nothing is skipped when it fires.
    bool drawSummaryLogged = false;

public:
    explicit VulkanRenderer(spdlog::logger& logger, FrameDiagnostics& diagnostics, IWindow& window,
                            IVulkanSurfaceSource& surfaceSource, RenderableEntityService& renderableEntityService,
                            SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService);
    ~VulkanRenderer() override;

    [[nodiscard]] std::expected<void, std::string> init() override;
    void setViewport(int width, int height) override;

    [[nodiscard]] bool beginFrame(double simulationTime) override;
    void recordView(Scene& scene, Camera& camera, float delta) override;
    void recordProbeCapture(Scene& scene, LightProbe& probe) override;
    void recordAmbientOcclusion(Scene& scene, Camera& camera, float delta) override;
    void recordAutoExposure(Camera& camera) override;
    void recordPresent(const Presenter& presenter) override;
    void endFrame() override;

    [[nodiscard]] std::expected<unsigned int, std::string>
    createShaderObject(const ShaderDescriptor& shaderDescriptor) override;
    [[nodiscard]] std::expected<unsigned int, std::string> createCubeMap(const Texture& front, const Texture& back,
                                                                         const Texture& left, const Texture& right,
                                                                         const Texture& top,
                                                                         const Texture& bottom) override;
    [[nodiscard]] std::expected<unsigned int, std::string> createFbo(const Fbo& fbo) override;
    void deleteFbo(Fbo& fbo) override;
    void releaseGpuResource(GpuResourceKind kind, unsigned int gpuResourceId) override;
    void releaseMaterial(const Resource<Material>& material) override;
    [[nodiscard]] GpuResourceCensus gpuResourceCensus() const override;

    [[nodiscard]] std::expected<void, std::string> captureFrame(const std::string& path) override;
    [[nodiscard]] std::expected<void, std::string> captureBuffers(const std::string& pathPrefix) override;

private:
    void createInstance();
    void createDebugMessenger();
    void selectPhysicalDevice();
    void createDevice();
    void createAllocator();
    void createPipelineCache();
    void writePipelineCache() const;
    void createSwapchain();
    void destroySwapchain();
    void createFrameResources();
    void createDescriptorInfrastructure();
    void recreateSwapchainIfNeeded();
    // The body of endFrame, with the capture readback the debug seam needs threaded through:
    // captureFrame replays a frame of its own and has to copy the presented image out of it.
    // Returns whether a frame was actually submitted and presented.
    bool submitAndPresent(VkBuffer captureBuffer);
    void recordClearOnlySwapchainPass();
    void recordScenePass(Scene& scene, Camera& camera, float delta);
    // Everything a light probe needs that outlives one capture: the scratch cube, the array, the
    // views each pass renders through, and the prefilter pipeline. Built once, in init(), because
    // its size is a contract constant rather than anything a scene decides — and because the
    // frame descriptor set names the array, and that set is written once at bring-up.
    void createProbeResources();
    // The eight bytes a camera's exposure meter is read out of. Built once for the same reason the
    // probe's is: it holds one texel whatever the chain in front of it looks like, and a buffer
    // created on the first metering frame would be a buffer allocated inside a recorded frame.
    void createExposureResources();
    // One face of `probe`'s environment, from the probe's own position through a 90-degree
    // frustum, into the scratch cube. A scene pass like any other: it shades, it samples the
    // cascades, and it draws the sky — the difference is that it draws only what the scene marked
    // static, because a probe is captured once and shaded from for many frames.
    void recordProbeFace(Scene& scene, const LightProbe& probe, unsigned int face, float delta);
    // The scratch cube's six faces are drawn: filter them down the roughness chain into the
    // probe's slice of the array, and copy the mip the irradiance projection reads out to the
    // readback buffer. Returns false if the pass could not be recorded at all.
    bool recordProbePrefilter(const LightProbe& probe);
    // Projects the radiance the readback buffer holds onto the spherical harmonic basis. Valid
    // only once the submission that wrote it has completed, which probeReadbackReadyAt states.
    [[nodiscard]] ShIrradiance projectProbeIrradiance() const;
    // The scene's probes as the frame block carries them, written into `frameData`.
    void uploadProbes(const Scene& scene, FrameDataUbo& frameData) const;
    // Everything recordSceneBehindCopy needs to suspend the pass, copy the opaque colour into the
    // behind chain and resume: the two image ids, the rendering info to re-begin with, and the
    // attachment structs it names — whose load ops become LOAD, because the resumed pass continues
    // over what the first half drew. `depthStoreAfterCopy` is the store policy the first half
    // overrode: a split pass must store its depth for the blended draws to test against, but the
    // resumed half owes nothing beyond what the unsplit pass stored.
    struct SceneBehindCopy
    {
        unsigned int colorImageId;
        unsigned int behindImageId;
        VkRenderingInfo* renderingInfo;
        VkRenderingAttachmentInfo* colorAttachment;
        VkRenderingAttachmentInfo* depthAttachment;
        VkAttachmentStoreOp depthStoreAfterCopy;
    };
    // Ends the current rendering block, blits the opaque colour into the behind image and its mip
    // chain, and re-begins rendering loading what is already there. Called between the opaque and
    // blended draws of a shading view, and only when there are blended draws to order it against.
    void recordSceneBehindCopy(const SceneBehindCopy& copy);
    // The behind image for a colour attachment, created on first use at the attachment's size with
    // sceneBehindMipCount levels. Returns the dummy texture when the attachment is unknown, which
    // shades as "nothing behind" rather than failing the view.
    [[nodiscard]] unsigned int sceneBehindImage(unsigned int colorImageId);
    // The entity/mesh/primitive walk both a camera's view and a probe's face record. `staticOnly`
    // is what a probe capture adds: a car baked into the environment goes on lighting the street
    // from wherever it was parked when the capture ran. `behindCopy` is non-null only for a shading
    // camera's view, and is what splits the pass between the opaque draws and the blended ones.
    [[nodiscard]] unsigned int recordSceneDraws(Scene& scene, const Camera& camera, float delta, bool shading,
                                                bool staticOnly, unsigned int viewShaderId, VkFormat colorFormat,
                                                VkFormat depthFormat, VkDescriptorSet shadowDescriptors,
                                                const SceneBehindCopy* behindCopy);
    [[nodiscard]] unsigned int createProbeImage(uint32_t resolution, uint32_t mipLevels, uint32_t layers,
                                                VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                                bool cube) const;
    // A view over one mip of one layer, seen as a plain 2D image: what dynamic rendering takes to
    // render into a cube face or into a level of a chain, and what a descriptor names to sample one
    // level without the rest of the chain coming with it.
    [[nodiscard]] VkImageView createLevelView(unsigned int imageId, uint32_t mip, uint32_t layer) const;
    // The view a pass renders into or samples for one level of an image. Level 0 of a single-level
    // image is its whole-image view, which is why nothing allocates a second one for it.
    [[nodiscard]] VkImageView levelView(const ImageResource& resource, uint32_t level) const;
    // `clipCorrectedViewProjection` is `clipCorrection() * camera.modelViewProjectionMatrix`, which
    // is constant for a whole view and is passed rather than rebuilt because this runs thousands of
    // times a frame. It is a parameter and not a member for the same reason every other per-view
    // value here is: a member would be state two passes could disagree about.
    void recordDraw(const MeshPrimitive& primitive, const Resource<Material>& materialKey, const Material& material,
                    unsigned int shaderId, const glm::mat4& entityModelMatrix, const Camera& camera,
                    const glm::mat4& clipCorrectedViewProjection, const std::vector<glm::mat4>& joints,
                    VkDeviceSize paintOffset, VkFormat colorFormat, VkFormat depthFormat,
                    VkDescriptorSet shadowDescriptors, bool doubleSided, bool depthWrite);
    // One slot of the paint ring, filled from a renderable's own paint. Allocated **per renderable
    // per view** rather than per draw, which is the shape the data actually has: paint describes a
    // car, so all eight hundred of a car's primitives read one block. Per draw it was not merely
    // wasteful — one car exhausted a thousand-slot ring in a single frame and drew itself unpainted.
    [[nodiscard]] VkDeviceSize writePaintSlot(const Paint& paint);
    // One fullscreen pass: bind the inputs, push the parameters, draw the oversized triangle. The
    // inputs are expected to be in SHADER_READ_ONLY_OPTIMAL already — the caller moves them,
    // because it is the caller that knows whether one of them is a level of the image being
    // written and must not be moved wholesale.
    // `lookupTableImageId` is the colour grade this pass binds — the neutral identity for every
    // pass but the last one, which is the only pass whose input is a display-referred image.
    // `shadowDescriptors` is the shadow set the pass binds at the fullscreen layout's third slot —
    // the view's own for a pass recorded inside a view's chain, so the fog pass marches the same
    // cascades the view shaded with, and the fallback set for a caller that has none.
    // `volumeImageIds` are the two sampler3D slots beside the grade — the pass's own volumes, or
    // neutralVolumes() for every pass that names none, because the layout declares both bindings
    // whatever the shader reads.
    // `loadColour` is PostProcess::loadColour: true makes the target begin the pass holding what it
    // last held, which is what a pass that blends against its destination — or writes only part of
    // it — needs. `blendWeight` is the constant the `FullscreenBlend::Constant` pipelines mix at;
    // it is dynamic state, so one pipeline serves every weight. `slice`/`sliceCount` are
    // PostProcess's: the vertical strip of the target this recording writes, of how many. The
    // viewport stays the whole target whatever they say — see the definition.
    bool recordFullScreenPass(std::span<const PostProcessBinding> inputs, unsigned int lookupTableImageId,
                              const std::array<unsigned int, postProcessVolumeCount>& volumeImageIds,
                              VkImageView targetView, VkExtent2D targetExtent, VkPipeline pipeline,
                              const FullscreenPushConstants& parameters,
                              VkDescriptorSet shadowDescriptors = VK_NULL_HANDLE, bool loadColour = false,
                              float blendWeight = 1.0f, unsigned int slice = 0, unsigned int sliceCount = 1);
    bool recordPresentPass(unsigned int shaderId, unsigned int attachmentImageId, const glm::vec4& parameters,
                           unsigned int lookupTableImageId);
    // Moves the whole image to one layout, emitting one barrier per run of levels that share a
    // current layout — which is a single barrier over the whole chain in every case but a
    // half-written one.
    void transitionTracked(VkCommandBuffer commandBuffer, unsigned int imageId, VkImageLayout newLayout);
    // Moves one mip level. The rest of the image keeps whatever layout it was in, which is the
    // point: a chain pass reads one level of an image while rendering into another.
    void transitionTrackedLevel(VkCommandBuffer commandBuffer, unsigned int imageId, uint32_t level,
                                VkImageLayout newLayout);
    void upload(const Resource<Model>& modelKey);
    [[nodiscard]] BufferResource createDeviceLocalBuffer(const void* data, VkDeviceSize size) const;
    [[nodiscard]] unsigned int uploadMeshBuffer(const MeshBuffer& meshBuffer);
    [[nodiscard]] PrimitiveBinding makePrimitiveBinding(const Model& model, const MeshPrimitive& primitive);
    void uploadMaterialTextures(const Resource<Material>& materialKey);
    [[nodiscard]] std::optional<unsigned int> uploadTexture(const Resource<Texture>& textureKey);
    [[nodiscard]] VkDescriptorSet materialSet(const Resource<Material>& materialKey, unsigned int environmentImageId);
    // The fullscreen set a pass binds, given the images it reads in declaration order. Every
    // element of the sampler array is written — the ones the pass did not name get the 1x1 dummy,
    // exactly as a view with no cascades still binds a 1x1 depth image — because the shader
    // declares the array whole and a pipeline that statically uses a descriptor must find one.
    [[nodiscard]] VkDescriptorSet attachmentSet(std::span<const PostProcessBinding> inputs,
                                                unsigned int lookupTableImageId,
                                                const std::array<unsigned int, postProcessVolumeCount>& volumeImageIds);
    // The two volume slots filled with the neutral table: what every pass that names no volumes
    // binds, the grade's own fallback applied one binding over.
    [[nodiscard]] std::array<unsigned int, postProcessVolumeCount> neutralVolumes();
    // The set 3 the scene pipelines bind. Given the cascades' depth images it caches one set per
    // tuple; given none it hands back the fallback, because a sampler2DShadow the pipeline
    // statically uses must have a descriptor whether or not the shader's cascade count says to
    // read it.
    [[nodiscard]] VkDescriptorSet shadowSet(const std::array<unsigned int, shadowCascadeCount>& imageIds,
                                            unsigned int occlusionImageId, unsigned int behindImageId,
                                            unsigned int cloudMapImageId);
    // The image a shading view samples its occlusion from, or the 1x1 white one when it gathers
    // none. Not const: the fallback is created on first use, as every other dummy here is.
    [[nodiscard]] unsigned int ambientOcclusionImage(const Camera& camera);
    // The image the scene's cloud map attachment carries, or the 1x1 white one when the scene
    // states none. Resolved per view, and by the probe faces too — the captures are how clouds
    // become ambient light, so a probe binding the dummy while the scene has a real map would
    // photograph a clear sky under a clouded one.
    [[nodiscard]] unsigned int cloudMapImage(const Scene& scene);
    [[nodiscard]] VkDescriptorSet fallbackShadowSet();
    [[nodiscard]] unsigned int dummyShadowMap();
    // The cascades' depth images in cascade order, or nothing if any of them is missing: a
    // partially resolved set would leave one binding naming an image from a previous scene.
    [[nodiscard]] std::optional<std::array<unsigned int, shadowCascadeCount>>
    shadowCascadeImages(const Scene& scene) const;
    // The pipeline this primitive is drawn with in this view, together with the vertex buffers
    // that pipeline expects. Borrowed from the binding's own cache and valid until the next call
    // for the same primitive, which is one draw's worth — recordDraw uses it and lets it go.
    // `blend` is false for a pass whose fragment alpha is data rather than coverage; `depthWrite`
    // is false for blended geometry in a shading view. See the state in the definition.
    [[nodiscard]] const ResolvedPipeline* scenePipeline(unsigned int shaderId, PrimitiveBinding& binding,
                                                        VkCullModeFlags cullMode, VkFormat colorFormat,
                                                        VkFormat depthFormat, bool blend, bool depthWrite);
    [[nodiscard]] VkPipeline offscreenPipeline(unsigned int shaderId, VkFormat colorFormat, FullscreenBlend blend);
    [[nodiscard]] unsigned int dummyTexture();
    [[nodiscard]] unsigned int dummyCubeMap();
    // Reports shaderc's own message rather than logging it: createShaderObject already has an
    // error channel, and a source that will not compile is the whole reason it has nothing to
    // hand back.
    [[nodiscard]] std::expected<std::vector<uint32_t>, std::string>
    compileToSpirv(const std::string& source, shaderc_shader_kind kind, const char* stageName);
    [[nodiscard]] VkShaderModule createShaderModule(const std::vector<uint32_t>& spirv) const;
    [[nodiscard]] VkPipeline buildFullscreenPipeline(VkShaderModule vertexModule, VkShaderModule fragmentModule,
                                                     FullscreenBlend blend, VkFormat colorFormat) const;
    void createHostVisibleUniformBuffer(VkDeviceSize size, VkBuffer& buffer, VmaAllocation& allocation,
                                        void*& mapped) const;
    [[nodiscard]] VkCommandBuffer beginUploadCommands() const;
    void finishUploadCommands(VkCommandBuffer commandBuffer) const;
    void retireUploadStaging(VkBuffer buffer, VmaAllocation allocation) const;
    void flushUploadCommands() const;
    [[nodiscard]] unsigned int createSampledImage(std::span<const Texture* const> faces, bool cube) const;
    // A three-dimensional sampled image: one mip, linearly filtered, and different enough from a
    // picture's upload that sharing the path above would be two functions in a trench coat. The
    // format follows the payload — float is the colour grade, expanded to RGBA32F and clamped as
    // it always was; byte is baked volumetric noise, uploaded at its own channel count and
    // repeating, because noise tiles where a grade's edges are black and white.
    [[nodiscard]] unsigned int createVolumeImage(const Texture& texture) const;
    // The grade that changes nothing, built once and bound by every pass that has none of its own.
    [[nodiscard]] unsigned int neutralLookupTable();
    // The sampler an FboAttachment::depthComparison asks for. GL sets the same two pieces of
    // state on the texture object; here they belong to a sampler, so the image carries one of its
    // own rather than sharing `attachmentSampler` — which is what ImageResource::sampler is for
    // and what attachmentSet already prefers whenever it is set.
    //
    // Linear filtering is what turns a comparison fetch into a 2x2 percentage-closer one, and for
    // depth formats it is a capability rather than a guarantee, so the device is asked.
    [[nodiscard]] VkSampler createComparisonSampler(VkFormat format) const;
    void destroyImageResource(unsigned int id) const;
    // Hands one GPU object to the retirement queue with the submission count it becomes safe at.
    void retire(RetiredResource resource) const;
    // Destroys everything the GPU is provably finished with. Called once per frame, right after
    // the fence wait that is the proof.
    void collectRetiredResources() const;
    // Destroys the whole queue regardless of readyAt. Only valid immediately after a device idle.
    void drainRetiredResources() const;
    void releaseShaderObject(unsigned int shaderId);
    [[nodiscard]] std::optional<VkDeviceSize> allocateDrawDataSlot();
    [[nodiscard]] std::optional<VkDeviceSize> allocateJointDataSlot();
    [[nodiscard]] std::optional<VkDeviceSize> allocatePaintDataSlot();
    [[nodiscard]] std::optional<VkDeviceSize> allocateFrameDataSlot();
};

VulkanRenderer::VulkanRenderer(spdlog::logger& logger, FrameDiagnostics& diagnostics, IWindow& window,
                               IVulkanSurfaceSource& surfaceSource, RenderableEntityService& renderableEntityService,
                               SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService) :
    logger(logger),
    diagnostics(diagnostics),
    window(window),
    surfaceSource(surfaceSource),
    memoryStorageService(memoryStorageService),
    renderableEntityService(renderableEntityService),
    sceneManagerService(sceneManagerService)
{
}

VulkanRenderer::~VulkanRenderer()
{
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

#if defined(RACEENGINE_HAS_TRACY)
    if (tracyGpuContext != nullptr)
    {
        TracyVkDestroy(tracyGpuContext);
        tracyGpuContext = nullptr;
    }
#endif

    // An upload batch still open at teardown was never named by any submitted frame, so its
    // commands are abandoned rather than run; the staging it holds goes with it.
    if (uploadBatchCommands != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(device, uploadCommandPool, 1, &uploadBatchCommands);
        uploadBatchCommands = VK_NULL_HANDLE;
    }
    for (const auto& [buffer, allocation] : uploadBatchStaging)
    {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
    uploadBatchStaging.clear();

    // The queue holds objects whose readyAt has not arrived and never will, because no further
    // frame will be submitted. The device idle above is the stronger guarantee they were waiting
    // for, so they go now — before the descriptor pool and the allocator they belong to.
    drainRetiredResources();

    for (const auto& [id, shader] : shaderObjects)
    {
        if (shader.swapchainTargetPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, shader.swapchainTargetPipeline, nullptr);
        }

        if (shader.vertexModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device, shader.vertexModule, nullptr);
        }
        if (shader.fragmentModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device, shader.fragmentModule, nullptr);
        }
    }

    for (const auto& [key, pipeline] : scenePipelines)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
    }

    for (const auto& [key, pipeline] : offscreenPipelines)
    {
        vkDestroyPipeline(device, pipeline, nullptr);
    }

    for (const auto& [key, material] : materialResources)
    {
        if (material.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, material.buffer, material.allocation);
        }
    }

    for (const auto& [id, buffer] : bufferResources)
    {
        if (buffer.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
        }
    }

    if (dummyVertexBuffer.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(allocator, dummyVertexBuffer.buffer, dummyVertexBuffer.allocation);
    }

    // The probe render targets: one view per cube face of the scratch, and one per (slice, mip,
    // face) of the array. They are not ImageResource::view — that is the sampling view each image
    // carries — so the loop below does not reach them.
    for (const auto view : probeRadianceFaceViews)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, view, nullptr);
        }
    }
    for (const auto view : probeSpecularFaceViews)
    {
        vkDestroyImageView(device, view, nullptr);
    }
    if (probeReadbackBuffer.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(allocator, probeReadbackBuffer.buffer, probeReadbackBuffer.allocation);
    }
    if (luminanceReadbackBuffer.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(allocator, luminanceReadbackBuffer.buffer, luminanceReadbackBuffer.allocation);
    }
    if (probePrefilterPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, probePrefilterPipeline, nullptr);
    }
    for (const auto shaderModule : {probePrefilterVertexModule, probePrefilterFragmentModule})
    {
        if (shaderModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
    if (probePrefilterPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, probePrefilterPipelineLayout, nullptr);
    }
    if (probePrefilterSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, probePrefilterSetLayout, nullptr);
    }
    if (probeSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, probeSampler, nullptr);
    }

    for (const auto& [id, image] : imageResources)
    {
        if (image.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, image.sampler, nullptr);
        }
        for (const auto levelView : image.levelViews)
        {
            vkDestroyImageView(device, levelView, nullptr);
        }
        if (image.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, image.view, nullptr);
        }
        if (image.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator, image.image, image.allocation);
        }
    }

    for (auto& frame : frames)
    {
        if (frame.frameDataBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, frame.frameDataBuffer, frame.frameDataAllocation);
        }
        if (frame.drawDataBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, frame.drawDataBuffer, frame.drawDataAllocation);
        }
        if (frame.paintDataBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, frame.paintDataBuffer, frame.paintDataAllocation);
            frame.paintDataBuffer = VK_NULL_HANDLE;
        }

        if (frame.jointDataBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, frame.jointDataBuffer, frame.jointDataAllocation);
        }
        if (frame.inFlight != VK_NULL_HANDLE)
        {
            vkDestroyFence(device, frame.inFlight, nullptr);
        }
        if (frame.imageAvailable != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device, frame.imageAvailable, nullptr);
        }
        if (frame.commandPool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device, frame.commandPool, nullptr);
        }
    }

    if (attachmentSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, attachmentSampler, nullptr);
    }
    if (cloudMapSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, cloudMapSampler, nullptr);
    }
    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }
    if (scenePipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, scenePipelineLayout, nullptr);
    }
    if (fullscreenPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, fullscreenPipelineLayout, nullptr);
    }
    for (const auto layout :
         {frameDataSetLayout, materialSetLayout, drawDataSetLayout, shadowSetLayout, fullscreenSetLayout})
    {
        if (layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        }
    }
    if (uploadCommandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, uploadCommandPool, nullptr);
    }

    if (allocator != nullptr)
    {
        vmaDestroyAllocator(allocator);
    }

    destroySwapchain();

    if (pipelineCache != VK_NULL_HANDLE)
    {
        writePipelineCache();
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
    }

    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
    }

    if (debugMessenger != VK_NULL_HANDLE && destroyDebugUtilsMessenger != nullptr)
    {
        destroyDebugUtilsMessenger(instance, debugMessenger, nullptr);
    }

    if (surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
}

std::expected<void, std::string> VulkanRenderer::init()
{
    // The bring-up steps report through `ensure`, which throws: converting here is what turns
    // "this machine has no usable Vulkan device" into something the composition root decides
    // about, rather than an exception escaping a constructor body.
    try
    {
        requestedExtent = VkExtent2D{static_cast<uint32_t>(std::max(window.state().windowWidth, 0)),
                                     static_cast<uint32_t>(std::max(window.state().windowHeight, 0))};
        createInstance();
        createDebugMessenger();
        surface = surfaceSource.generateVulkanSurface(instance);
        selectPhysicalDevice();
        createDevice();
        createAllocator();
        createPipelineCache();
        createSwapchain();
        createFrameResources();
        createDescriptorInfrastructure();
        createProbeResources();
        createExposureResources();
#if defined(RACEENGINE_HAS_TRACY)
        // The context's constructor records its own calibration into this buffer, submits it and
        // waits; the pool is reset before the first frame records, so borrowing it here is safe.
        tracyGpuContext = TracyVkContext(physicalDevice, device, graphicsQueue, frames.front().commandBuffer);
#endif
        return {};
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

void VulkanRenderer::createInstance()
{
    uint32_t availableLayerCount = 0;
    ensure(vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr), "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> availableLayers(availableLayerCount);
    if (availableLayerCount > 0)
    {
        ensure(vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data()),
               "vkEnumerateInstanceLayerProperties");
    }

    validationLayerEnabled = std::ranges::any_of(availableLayers, [](const VkLayerProperties& layer)
                                                 { return std::strcmp(layer.layerName, validationLayerName) == 0; });

    const auto requiredWindowExtensions = surfaceSource.getRequiredVulkanWindowExtensions();
    // The extension list pointer is GLFW-owned static storage; copy before appending.
    std::vector<const char*> extensions(requiredWindowExtensions.extensions,
                                        requiredWindowExtensions.extensions + requiredWindowExtensions.count);
    std::vector<const char*> layers;

    // Debug utils rides whenever the loader offers it, not only under validation: it is what
    // names the passes in a capture, and a capture is most wanted on exactly the runs that do
    // not carry the validation layer.
    uint32_t instanceExtensionCount = 0;
    ensure(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr),
           "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> instanceExtensions(instanceExtensionCount);
    if (instanceExtensionCount > 0)
    {
        ensure(vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, instanceExtensions.data()),
               "vkEnumerateInstanceExtensionProperties");
    }
    const auto debugUtilsAvailable =
        std::ranges::any_of(instanceExtensions, [](const VkExtensionProperties& extension)
                            { return std::strcmp(extension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0; });

    if (debugUtilsAvailable)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    if (validationLayerEnabled)
    {
        layers.push_back(validationLayerName);
    }

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "OpenStreetRace";
    applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.pEngineName = "RaceEngine";
    applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    auto messengerInfo = makeDebugMessengerCreateInfo(logger);

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (validationLayerEnabled)
    {
        instanceCreateInfo.pNext = &messengerInfo;
    }
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instanceCreateInfo.ppEnabledLayerNames = layers.data();
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

    const auto createResult = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    if (createResult == VK_ERROR_LAYER_NOT_PRESENT && validationLayerEnabled)
    {
        // Manifest present but the layer library would not load (e.g. VK_LAYER_PATH set
        // without LD_LIBRARY_PATH covering its .so); run without validation instead.
        logger.warn("Vulkan validation layer would not load; continuing without it");
        validationLayerEnabled = false;
        layers.pop_back();
        instanceCreateInfo.pNext = nullptr;
        instanceCreateInfo.enabledLayerCount = 0;
        ensure(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "vkCreateInstance");
    }
    else
    {
        ensure(createResult, "vkCreateInstance");
    }

    if (debugUtilsAvailable)
    {
        cmdBeginDebugUtilsLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
        cmdEndDebugUtilsLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
        // Both or neither: a label opened by one function pointer must be closable by the other.
        if (cmdBeginDebugUtilsLabel == nullptr || cmdEndDebugUtilsLabel == nullptr)
        {
            cmdBeginDebugUtilsLabel = nullptr;
            cmdEndDebugUtilsLabel = nullptr;
        }
    }

    logger.info("Vulkan instance created (api 1.3, validation layer {})",
                validationLayerEnabled ? "enabled" : "not present");
}

void VulkanRenderer::createDebugMessenger()
{
    if (!validationLayerEnabled)
    {
        return;
    }

    const auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    destroyDebugUtilsMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    if (createMessenger == nullptr || destroyDebugUtilsMessenger == nullptr)
    {
        logger.warn("Vulkan debug utils entry points unavailable; continuing without a messenger");
        return;
    }

    const auto messengerInfo = makeDebugMessengerCreateInfo(logger);
    ensure(createMessenger(instance, &messengerInfo, nullptr, &debugMessenger), "vkCreateDebugUtilsMessengerEXT");
}

void VulkanRenderer::selectPhysicalDevice()
{
    uint32_t deviceCount = 0;
    ensure(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (deviceCount > 0)
    {
        ensure(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");
    }

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties chosenProperties{};
    VkBool32 chosenSamplerAnisotropy = VK_FALSE;
    uint32_t chosenGraphicsFamily = 0;
    uint32_t chosenPresentFamily = 0;

    for (const auto& candidate : devices)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(candidate, &properties);

        if (properties.apiVersion < VK_API_VERSION_1_3)
        {
            continue;
        }

        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features13;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);

        // shaderDemoteToHelperInvocation: SPIR-V 1.6 (vulkan1.3 shaderc target) lowers
        // GLSL discard to OpDemoteToHelperInvocation.
        //
        // imageCubeArray is a selection criterion rather than an optional extra, unlike
        // samplerAnisotropy below: the light probe path reads every probe's prefiltered radiance
        // out of one cube array, which is what lets the shading loop pick a probe by a dynamic
        // index instead of needing a descriptor per probe. A device without it cannot run the
        // scene shader at all, so there is no degraded mode to fall back to.
        if (features13.dynamicRendering == VK_FALSE || features13.synchronization2 == VK_FALSE ||
            features13.shaderDemoteToHelperInvocation == VK_FALSE || features2.features.imageCubeArray == VK_FALSE)
        {
            continue;
        }

        uint32_t extensionCount = 0;
        ensure(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr),
               "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> deviceExtensions(extensionCount);
        if (extensionCount > 0)
        {
            ensure(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, deviceExtensions.data()),
                   "vkEnumerateDeviceExtensionProperties");
        }

        const auto hasSwapchain =
            std::ranges::any_of(deviceExtensions, [](const VkExtensionProperties& extension)
                                { return std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0; });
        if (!hasSwapchain)
        {
            continue;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());

        std::optional<uint32_t> foundGraphicsFamily;
        std::optional<uint32_t> foundPresentFamily;
        for (uint32_t familyIndex = 0; familyIndex < queueFamilyCount; familyIndex++)
        {
            const auto supportsGraphics = (queueFamilies[familyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            VkBool32 supportsPresent = VK_FALSE;
            ensure(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, familyIndex, surface, &supportsPresent),
                   "vkGetPhysicalDeviceSurfaceSupportKHR");

            if (supportsGraphics && supportsPresent == VK_TRUE)
            {
                foundGraphicsFamily = familyIndex;
                foundPresentFamily = familyIndex;
                break;
            }
            if (supportsGraphics && !foundGraphicsFamily.has_value())
            {
                foundGraphicsFamily = familyIndex;
            }
            if (supportsPresent == VK_TRUE && !foundPresentFamily.has_value())
            {
                foundPresentFamily = familyIndex;
            }
        }

        if (!foundGraphicsFamily.has_value() || !foundPresentFamily.has_value())
        {
            continue;
        }

        const auto preferCandidate =
            chosen == VK_NULL_HANDLE || (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                                         chosenProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        if (preferCandidate)
        {
            chosen = candidate;
            chosenProperties = properties;
            // Optional, not a selection criterion: GL falls back to 1x anisotropy on a driver
            // that does not advertise the extension, and so does this backend.
            chosenSamplerAnisotropy = features2.features.samplerAnisotropy;
            chosenGraphicsFamily = foundGraphicsFamily.value();
            chosenPresentFamily = foundPresentFamily.value();
        }
    }

    if (chosen == VK_NULL_HANDLE)
    {
        throw std::runtime_error("no Vulkan 1.3 device with dynamicRendering, synchronization2, imageCubeArray, "
                                 "a swapchain, and graphics+present queues");
    }

    physicalDevice = chosen;
    deviceLimits = chosenProperties.limits;
    samplerAnisotropySupported = chosenSamplerAnisotropy == VK_TRUE;
    graphicsQueueFamily = chosenGraphicsFamily;
    presentQueueFamily = chosenPresentFamily;

    // GL falls back to 1x where the anisotropy limit query yields it; so does this.
    const auto anisotropy = samplerAnisotropySupported ? deviceLimits.maxSamplerAnisotropy : 1.0f;
    logger.info("Vulkan device selected: {} ({}, api {}.{}.{}, anisotropic filtering {:g}x)",
                std::string_view(chosenProperties.deviceName), describeDeviceType(chosenProperties.deviceType),
                VK_API_VERSION_MAJOR(chosenProperties.apiVersion), VK_API_VERSION_MINOR(chosenProperties.apiVersion),
                VK_API_VERSION_PATCH(chosenProperties.apiVersion), anisotropy);
}

void VulkanRenderer::createDevice()
{
    const float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (const auto familyIndex : std::array{graphicsQueueFamily, presentQueueFamily})
    {
        const auto alreadyQueued = std::ranges::any_of(queueCreateInfos, [&](const VkDeviceQueueCreateInfo& info)
                                                       { return info.queueFamilyIndex == familyIndex; });
        if (alreadyQueued)
        {
            continue;
        }

        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = familyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.shaderDemoteToHelperInvocation = VK_TRUE;

    const std::array deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // GL applies GL_TEXTURE_MAX_ANISOTROPY to every model texture it uploads; the sampler
    // path below can only ask for it if the feature is switched on here.
    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = samplerAnisotropySupported ? VK_TRUE : VK_FALSE;
    // Required, not optional: see selectPhysicalDevice. Every candidate that got this far has it.
    features.imageCubeArray = VK_TRUE;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &features13;
    deviceCreateInfo.pEnabledFeatures = &features;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    ensure(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device), "vkCreateDevice");

    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueFamily, 0, &presentQueue);
}

void VulkanRenderer::createAllocator()
{
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    ensure(vmaCreateAllocator(&allocatorInfo, &allocator), "vmaCreateAllocator");
}

void VulkanRenderer::createPipelineCache()
{
    std::vector<char> initialData;
    const auto path = pipelineCacheFilePath();
    if (!path.empty())
    {
        if (std::FILE* file = std::fopen(path.c_str(), "rb"); file != nullptr)
        {
            std::fseek(file, 0, SEEK_END);
            if (const auto size = std::ftell(file); size > 0)
            {
                initialData.resize(static_cast<size_t>(size));
                std::fseek(file, 0, SEEK_SET);
                if (std::fread(initialData.data(), 1, initialData.size(), file) != initialData.size())
                {
                    initialData.clear();
                }
            }
            std::fclose(file);
        }
    }

    // The blob's own header states which device wrote it. A foreign blob is dropped here rather
    // than handed over: the spec leaves a driver's validation of initial data open, and a stale
    // cache must cost a cold start, never a crash.
    constexpr size_t headerSize = 32;
    if (initialData.size() >= headerSize)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        uint32_t headerVendor = 0;
        uint32_t headerDevice = 0;
        std::memcpy(&headerVendor, initialData.data() + 8, sizeof(headerVendor));
        std::memcpy(&headerDevice, initialData.data() + 12, sizeof(headerDevice));
        if (headerVendor != properties.vendorID || headerDevice != properties.deviceID ||
            std::memcmp(initialData.data() + 16, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0)
        {
            initialData.clear();
        }
    }
    else
    {
        initialData.clear();
    }

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = initialData.size();
    cacheInfo.pInitialData = initialData.empty() ? nullptr : initialData.data();

    if (vkCreatePipelineCache(device, &cacheInfo, nullptr, &pipelineCache) != VK_SUCCESS)
    {
        cacheInfo.initialDataSize = 0;
        cacheInfo.pInitialData = nullptr;
        ensure(vkCreatePipelineCache(device, &cacheInfo, nullptr, &pipelineCache), "vkCreatePipelineCache");
    }

    logger.info("Vulkan pipeline cache ready ({} bytes primed from {})", initialData.size(),
                path.empty() ? "nowhere - no cache directory" : path);
}

void VulkanRenderer::writePipelineCache() const
{
    if (pipelineCache == VK_NULL_HANDLE)
    {
        return;
    }

    const auto path = pipelineCacheFilePath();
    if (path.empty())
    {
        return;
    }

    size_t size = 0;
    if (vkGetPipelineCacheData(device, pipelineCache, &size, nullptr) != VK_SUCCESS || size == 0)
    {
        return;
    }

    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device, pipelineCache, &size, data.data()) != VK_SUCCESS)
    {
        return;
    }

    if (std::FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr)
    {
        if (std::fwrite(data.data(), 1, size, file) != size)
        {
            // A truncated cache would fail the header check next run and cost one cold start;
            // removing it now keeps even that from happening.
            std::fclose(file);
            std::remove(path.c_str());

            return;
        }
        std::fclose(file);
    }
}

void VulkanRenderer::createSwapchain()
{
    VkSurfaceCapabilitiesKHR capabilities;
    ensure(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities),
           "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    uint32_t formatCount = 0;
    ensure(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr),
           "vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (formatCount == 0)
    {
        throw std::runtime_error("surface reports no formats");
    }
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    ensure(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data()),
           "vkGetPhysicalDeviceSurfaceFormatsKHR");

    surfaceFormat = formats.front();
    for (const auto& candidate : formats)
    {
        if (candidate.format == VK_FORMAT_B8G8R8A8_UNORM && candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            surfaceFormat = candidate;
            break;
        }
    }

    auto extent = capabilities.currentExtent;
    if (extent.width == std::numeric_limits<uint32_t>::max())
    {
        extent.width =
            std::clamp(requestedExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height =
            std::clamp(requestedExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    swapchainExtent = extent;
    if (extent.width == 0 || extent.height == 0)
    {
        return; // 0x0 while minimised: stay without a swapchain, frames are skipped.
    }

    auto imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0)
    {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }

    VkSwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.surface = surface;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = surfaceFormat.format;
    swapchainInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapchainInfo.imageExtent = extent;
    swapchainInfo.imageArrayLayers = 1;
    // TRANSFER_SRC keeps every swapchain image readable for captureFrame (see vulkan-abi.md).
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const std::array sharedQueueFamilies = {graphicsQueueFamily, presentQueueFamily};
    if (graphicsQueueFamily != presentQueueFamily)
    {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = static_cast<uint32_t>(sharedQueueFamilies.size());
        swapchainInfo.pQueueFamilyIndices = sharedQueueFamilies.data();
    }
    else
    {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    swapchainInfo.preTransform = capabilities.currentTransform;
    swapchainInfo.compositeAlpha = (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0
                                       ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                       : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    // No vsync by default: MAILBOX presents the newest complete frame uncapped and without
    // tearing, IMMEDIATE is the raw uncapped mode, and FIFO is the spec-guaranteed fallback and
    // the old vsynced behaviour. RACEENGINE_PRESENT_MODE=immediate|mailbox|fifo forces one;
    // a forced mode the surface does not offer falls back to FIFO with a warning.
    uint32_t presentModeCount = 0;
    ensure(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr),
           "vkGetPhysicalDeviceSurfacePresentModesKHR");
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    ensure(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data()),
           "vkGetPhysicalDeviceSurfacePresentModesKHR");

    const auto presentModeOffered = [&](const VkPresentModeKHR candidate)
    { return std::ranges::find(presentModes, candidate) != presentModes.end(); };

    std::optional<VkPresentModeKHR> forced;
    if (const char* requested = std::getenv("RACEENGINE_PRESENT_MODE"); requested != nullptr && *requested != '\0')
    {
        const std::string_view name(requested);
        if (name == "immediate")
        {
            forced = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
        else if (name == "mailbox")
        {
            forced = VK_PRESENT_MODE_MAILBOX_KHR;
        }
        else if (name == "fifo")
        {
            forced = VK_PRESENT_MODE_FIFO_KHR;
        }
        else
        {
            logger.warn("RACEENGINE_PRESENT_MODE '{}' is not immediate, mailbox or fifo; using the default order",
                        name);
        }

        if (forced.has_value() && !presentModeOffered(forced.value()))
        {
            logger.warn("RACEENGINE_PRESENT_MODE '{}' is not offered by this surface; using FIFO", name);
            forced = VK_PRESENT_MODE_FIFO_KHR;
        }
    }

    auto presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (forced.has_value())
    {
        presentMode = forced.value();
    }
    else
    {
        for (const auto candidate : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
        {
            if (presentModeOffered(candidate))
            {
                presentMode = candidate;
                break;
            }
        }
    }
    swapchainInfo.presentMode = presentMode;
    swapchainInfo.clipped = VK_FALSE; // captureFrame reads rendered images back

    ensure(vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapchain), "vkCreateSwapchainKHR");

    uint32_t actualImageCount = 0;
    ensure(vkGetSwapchainImagesKHR(device, swapchain, &actualImageCount, nullptr), "vkGetSwapchainImagesKHR");
    swapchainImages.resize(actualImageCount);
    ensure(vkGetSwapchainImagesKHR(device, swapchain, &actualImageCount, swapchainImages.data()),
           "vkGetSwapchainImagesKHR");

    swapchainImageViews.resize(actualImageCount);
    renderFinishedSemaphores.resize(actualImageCount);
    for (uint32_t imageIndex = 0; imageIndex < actualImageCount; imageIndex++)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages[imageIndex];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = surfaceFormat.format;
        viewInfo.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ensure(vkCreateImageView(device, &viewInfo, nullptr, &swapchainImageViews[imageIndex]), "vkCreateImageView");

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ensure(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[imageIndex]),
               "vkCreateSemaphore");
    }

    requestedExtent = extent;

    logger.info("Vulkan swapchain created: {}x{}, {} images, format {}, colour space {}, present mode {}",
                extent.width, extent.height, actualImageCount, describeFormat(surfaceFormat.format),
                describeColorSpace(surfaceFormat.colorSpace),
                presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE"
                : presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX"
                                                             : "FIFO");
}

void VulkanRenderer::destroySwapchain()
{
    for (const auto semaphore : renderFinishedSemaphores)
    {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    renderFinishedSemaphores.clear();

    for (const auto imageView : swapchainImageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }
    swapchainImageViews.clear();
    swapchainImages.clear();

    if (swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::createFrameResources()
{
    for (auto& frame : frames)
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = graphicsQueueFamily;
        ensure(vkCreateCommandPool(device, &poolInfo, nullptr, &frame.commandPool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = frame.commandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        ensure(vkAllocateCommandBuffers(device, &allocateInfo, &frame.commandBuffer), "vkAllocateCommandBuffers");

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ensure(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frame.imageAvailable), "vkCreateSemaphore");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        ensure(vkCreateFence(device, &fenceInfo, nullptr, &frame.inFlight), "vkCreateFence");
    }
}

void VulkanRenderer::createHostVisibleUniformBuffer(const VkDeviceSize size, VkBuffer& buffer,
                                                    VmaAllocation& allocation, void*& mapped) const
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VmaAllocationInfo allocationInfo{};
    ensure(vmaCreateBuffer(allocator, &bufferInfo, &allocationCreateInfo, &buffer, &allocation, &allocationInfo),
           "vmaCreateBuffer");
    mapped = allocationInfo.pMappedData;
    std::memset(mapped, 0, static_cast<size_t>(size));
}

void VulkanRenderer::createDescriptorInfrastructure()
{
    const auto makeSetLayout = [&](const std::span<const VkDescriptorSetLayoutBinding> bindings)
    {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        ensure(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout), "vkCreateDescriptorSetLayout");
        return layout;
    };

    // Scene set 0: FrameData UBO, read by both stages (vulkan-abi.md). Dynamic-offset like
    // set 2, because the frame holds one of these per view rather than one in total.
    // Binding 1 is every light probe's prefiltered radiance, as one cube array. It belongs beside
    // the frame block rather than in a set of its own because it is exactly as per-frame as the
    // block is, and because one image serves every probe: the descriptor is written once at
    // bring-up and a probe re-capturing rewrites its contents, never this.
    const std::array frameBindings = {
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{probeSpecularBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    frameDataSetLayout = makeSetLayout(frameBindings);

    // Scene set 1: MaterialData UBO at binding 0, then one sampler per material texture slot
    // at the binding RenderContract assigns it.
    std::array<VkDescriptorSetLayoutBinding, materialTextureSlotCount + 1> materialBindings{};
    materialBindings[0] =
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    for (uint32_t slot = 0; slot < materialTextureSlotCount; slot++)
    {
        const auto binding = textureBinding(static_cast<MaterialTextureSlot>(slot));
        materialBindings[slot + 1] = VkDescriptorSetLayoutBinding{binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                                  VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    }
    materialSetLayout = makeSetLayout(materialBindings);

    // Scene set 2: dynamic-offset DrawData UBO, ring-buffered per frame in flight, and beside it the
    // skinning palette on a ring of its own. Two bindings rather than one block because the palette
    // is thirty times the size of everything else a draw carries and only a skinned draw has one:
    // folded together, a scene's draw ring is sized in units of the largest thing any draw *might*
    // need rather than what its draws actually need.
    const std::array drawBindings = {VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                                                  VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                                     VkDescriptorSetLayoutBinding{jointDataBinding,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                                                  VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                                     // The fragment stage, unlike the two above: paint is shading
                                     // rather than transform, and no vertex stage reads it.
                                     VkDescriptorSetLayoutBinding{paintDataBinding,
                                                                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                                                  VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    drawDataSetLayout = makeSetLayout(drawBindings);

    // Scene set 3: one combined sampler per cascade, read by the fragment stage only. A set of
    // its own rather than a tail of set 1 because the cascades are per frame: writing them into
    // every material set would mean rewriting every material set when a cascade target is rebuilt.
    // One binding holding an array of shadowCascadeCount samplers, not that many bindings: the
    // shader declares `sampler2DShadow shadowMaps[SHADOW_CASCADES]`, and a resource declared as an
    // array must be backed by a binding whose descriptorCount covers its whole length.
    //
    // Binding 1 is the view's ambient occlusion, and it is here for the same reason: it is one image
    // per camera pass, produced earlier in this frame and read only by a view that shades. Set 0 is
    // where it would otherwise belong and cannot hold it — that set is one descriptor addressed by a
    // dynamic offset per view, and an image binding does not vary by offset.
    const std::array shadowBindings = {
        VkDescriptorSetLayoutBinding{shadowMapBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadowCascadeCount,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{ambientOcclusionBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // Binding 2 is the behind copy — the opaque scene as it stood when the blended draws
        // started — and it lives here for the occlusion image's reason word for word: per camera
        // pass, produced earlier in this frame, readable only by a view that shades.
        VkDescriptorSetLayoutBinding{sceneBehindBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // Binding 3 is the cloud dome map, here on the same terms once more: produced by a pass —
        // the previous frame's cloud march — and read only by a view that shades, the skybox
        // compositing it and the probe faces photographing it. A scene with none binds the 1x1
        // white dummy behind the frame block's coverage branch.
        VkDescriptorSetLayoutBinding{cloudMapBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    shadowSetLayout = makeSetLayout(shadowBindings);

    // Fullscreen passes use their own single-set layout: one binding holding an array of
    // postProcessInputCount combined samplers, not that many bindings, because the shader declares
    // `sampler2D inputs[POST_INPUTS]` and an array resource must be backed by a binding whose
    // descriptorCount covers its whole length — the same shape and the same reason as set 3's
    // cascades. A pass reads the elements it wants; every element is written regardless.
    const std::array fullscreenBindings = {
        VkDescriptorSetLayoutBinding{postProcessInputBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     postProcessInputCount, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // The colour grade, which is a sampler3D and therefore cannot be an element of the array
        // above: a descriptor array holds one type. Written for every pass — the neutral identity
        // when the pass has no grade — because the pipeline's static use of it does not care which
        // pass this is.
        VkDescriptorSetLayoutBinding{lookupTableBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // The baked cloud noise volumes, beside the grade and each a binding of its own for the
        // grade's reason: a sampler3D cannot be an element of the sampler2D array. Written for
        // every pass — the neutral table where a pass names no volume — because the pipeline's
        // static use of them does not care which pass this is.
        VkDescriptorSetLayoutBinding{cloudBaseNoiseBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{cloudDetailNoiseBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    fullscreenSetLayout = makeSetLayout(fullscreenBindings);

    const auto makePipelineLayout = [&](const std::span<const VkDescriptorSetLayout> setLayouts)
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
        layoutInfo.pSetLayouts = setLayouts.data();

        VkPipelineLayout layout = VK_NULL_HANDLE;
        ensure(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
        return layout;
    };

    const std::array sceneSetLayouts = {frameDataSetLayout, materialSetLayout, drawDataSetLayout, shadowSetLayout};
    scenePipelineLayout = makePipelineLayout(sceneSetLayouts);

    // The fullscreen layout carries one push constant range: the tone curve the post chain maps
    // through, and which level of a chain the pass is on. See FullscreenPushConstants. A push
    // constant rather than a member of some block because it changes per pass and is 32 bytes
    // against a guaranteed 128, and because a fullscreen shader that does not declare it costs
    // nothing — the range belongs to the layout, and writing it is valid whether or not the
    // pipeline reads it.
    const VkPushConstantRange fullscreenPushConstants{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FullscreenPushConstants)};
    // Three sets, not the one this was: the frame block first — the same layout, set and dynamic
    // offset the scene draws bind — then the pass's own inputs, then the shadow set, so a
    // fullscreen pass can read the sun, the probes and the medium and march the cascades. The fog
    // pass is why; every other fullscreen shader declares only the middle set and pays nothing.
    const std::array fullscreenSetLayouts = {frameDataSetLayout, fullscreenSetLayout, shadowSetLayout};
    VkPipelineLayoutCreateInfo fullscreenLayoutInfo{};
    fullscreenLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    fullscreenLayoutInfo.setLayoutCount = static_cast<uint32_t>(fullscreenSetLayouts.size());
    fullscreenLayoutInfo.pSetLayouts = fullscreenSetLayouts.data();
    fullscreenLayoutInfo.pushConstantRangeCount = 1;
    fullscreenLayoutInfo.pPushConstantRanges = &fullscreenPushConstants;
    ensure(vkCreatePipelineLayout(device, &fullscreenLayoutInfo, nullptr, &fullscreenPipelineLayout),
           "vkCreatePipelineLayout");

    const std::array poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorPoolUniformBuffers},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, descriptorPoolDynamicUniformBuffers},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorPoolCombinedImageSamplers},
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // FREE_DESCRIPTOR_SET lets resize paths recycle attachment sets instead of burning pool space.
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = descriptorPoolMaxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    ensure(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");

    VkCommandPoolCreateInfo uploadPoolInfo{};
    uploadPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    uploadPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    uploadPoolInfo.queueFamilyIndex = graphicsQueueFamily;
    ensure(vkCreateCommandPool(device, &uploadPoolInfo, nullptr, &uploadCommandPool), "vkCreateCommandPool");

    // One shared sampler serves every FBO attachment read (mirrors GL's per-attachment
    // CLAMP_TO_EDGE + LINEAR). The LOD clamp is open rather than the implicit 0 a zeroed
    // VkSamplerCreateInfo carries: a chain attachment is sampled level by level through a view of
    // that level, and a sampler that could only ever reach level 0 would silently read the top of
    // every chain. Nothing changes for the single-level attachments that are every other target —
    // there is no second level for the clamp to have been keeping them off.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &attachmentSampler), "vkCreateSampler");

    // The cloud dome map's sampler: identical but repeating in u, because the map's u is a
    // lat-long azimuth that wraps at 360 degrees and a clamp would seam the sky where the mapping
    // rejoins itself. See the member's own comment.
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &cloudMapSampler), "vkCreateSampler");

    const auto alignment = std::max<VkDeviceSize>(deviceLimits.minUniformBufferOffsetAlignment, 1);
    drawDataStride = (sizeof(DrawDataUbo) + alignment - 1) / alignment * alignment;
    jointDataStride = (sizeof(JointDataUbo) + alignment - 1) / alignment * alignment;
    paintDataStride = (sizeof(PaintDataUbo) + alignment - 1) / alignment * alignment;
    frameDataStride = (sizeof(FrameDataUbo) + alignment - 1) / alignment * alignment;

    for (auto& frame : frames)
    {
        createHostVisibleUniformBuffer(frameDataStride * frameDataRingSlots, frame.frameDataBuffer,
                                       frame.frameDataAllocation, frame.frameDataMapped);
        createHostVisibleUniformBuffer(drawDataStride * drawDataRingSlots, frame.drawDataBuffer,
                                       frame.drawDataAllocation, frame.drawDataMapped);
        // Zeroed at creation, which is what makes slot 0 a legitimate thing for an unskinned draw
        // to bind: its `animated.x` is zero, so the palette is never read, but the descriptor is
        // still one the pipeline may statically use and must therefore address real memory.
        createHostVisibleUniformBuffer(jointDataStride * jointDataRingSlots, frame.jointDataBuffer,
                                       frame.jointDataAllocation, frame.jointDataMapped);
        createHostVisibleUniformBuffer(paintDataStride * paintDataRingSlots, frame.paintDataBuffer,
                                       frame.paintDataAllocation, frame.paintDataMapped);

        const std::array setLayouts = {frameDataSetLayout, drawDataSetLayout};
        std::array<VkDescriptorSet, 2> sets{};
        VkDescriptorSetAllocateInfo setAllocateInfo{};
        setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocateInfo.descriptorPool = descriptorPool;
        setAllocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
        setAllocateInfo.pSetLayouts = setLayouts.data();
        ensure(vkAllocateDescriptorSets(device, &setAllocateInfo, sets.data()), "vkAllocateDescriptorSets");
        frame.frameDataSet = sets[0];
        frame.drawDataSet = sets[1];

        // Dynamic UBO range is one FrameData; the bound offset walks the ring per view.
        const VkDescriptorBufferInfo frameDataInfo{frame.frameDataBuffer, 0, sizeof(FrameDataUbo)};
        // Dynamic UBO range is one DrawData; the bound offset walks the ring per draw.
        const VkDescriptorBufferInfo drawDataInfo{frame.drawDataBuffer, 0, sizeof(DrawDataUbo)};
        // Likewise one palette, on its own ring: a skinned draw walks it and every other draw
        // binds offset zero.
        const VkDescriptorBufferInfo jointDataInfo{frame.jointDataBuffer, 0, sizeof(JointDataUbo)};
        // And one car's paint, on a ring of its own: a painted draw walks it and every other draw
        // binds offset zero.
        const VkDescriptorBufferInfo paintDataInfo{frame.paintDataBuffer, 0, sizeof(PaintDataUbo)};

        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = frame.frameDataSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[0].pBufferInfo = &frameDataInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = frame.drawDataSet;
        writes[1].dstBinding = 0;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[1].pBufferInfo = &drawDataInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = frame.drawDataSet;
        writes[2].dstBinding = jointDataBinding;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[2].pBufferInfo = &jointDataInfo;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = frame.drawDataSet;
        writes[3].dstBinding = paintDataBinding;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[3].pBufferInfo = &paintDataInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // GL leaves a disabled vertex attribute at the generic default (0, 0, 0, 1); Vulkan
    // demands a real binding for every declared input, so absent attributes read this
    // constant through a zero-stride binding.
    const std::array<float, 4> defaultVertexAttribute = {0.0f, 0.0f, 0.0f, 1.0f};
    dummyVertexBuffer = createDeviceLocalBuffer(defaultVertexAttribute.data(), sizeof(defaultVertexAttribute));

    logger.info("Vulkan descriptor machinery ready: scene sets 0-2 + fullscreen set, pool for {} sets, "
                "draw-data ring {} slots x {} bytes and joint-palette ring {} slots x {} bytes per frame in flight",
                descriptorPoolMaxSets, drawDataRingSlots, drawDataStride, jointDataRingSlots, jointDataStride);
}

void VulkanRenderer::recreateSwapchainIfNeeded()
{
    const auto extentChanged =
        requestedExtent.width != swapchainExtent.width || requestedExtent.height != swapchainExtent.height;

    if (!recreateNeeded && !extentChanged && swapchain != VK_NULL_HANDLE)
    {
        return;
    }

    ensure(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
    // The idle is a stronger guarantee than any readyAt the queue is holding out for, and a
    // resize is the one place a lot of images are retired at once.
    drainRetiredResources();
    destroySwapchain();
    createSwapchain();
    recreateNeeded = false;
}

bool VulkanRenderer::beginFrame(const double simulationTime)
{
    // Held for the frame: every view this frame records uploads the same instant, which is what
    // makes anything temporal in a shader per-frame rather than per-view.
    frameSimulationTime = simulationTime;

    recreateSwapchainIfNeeded();
    if (swapchain == VK_NULL_HANDLE)
    {
        return false;
    }

    auto& frame = frames[frameIndex];
    ensure(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, waitForever), "vkWaitForFences");

    // That fence is the only proof of completion this backend has, so it is where deferred
    // destruction happens. See retire() for what the arithmetic means.
    collectRetiredResources();

    // The fence guarantees the GPU is done with this slot's uniform rings; reset both for
    // the views and draws recorded this frame.
    frame.drawDataSlotsUsed = 0;
    frame.jointDataSlotsUsed = 0;
    frame.paintDataSlotsUsed = 0;
    frame.frameDataSlotsUsed = 0;
    frame.frameDataOffset = 0;

    const auto acquireResult =
        vkAcquireNextImageKHR(device, swapchain, waitForever, frame.imageAvailable, VK_NULL_HANDLE, &currentImageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateNeeded = true;
        return false;
    }
    if (acquireResult == VK_SUBOPTIMAL_KHR)
    {
        recreateNeeded = true;
    }
    else
    {
        ensure(acquireResult, "vkAcquireNextImageKHR");
    }

    ensure(vkResetFences(device, 1, &frame.inFlight), "vkResetFences");
    ensure(vkResetCommandPool(device, frame.commandPool, 0), "vkResetCommandPool");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ensure(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    frameOpen = true;
    swapchainPassRecorded = false;
    return true;
}

void VulkanRenderer::recordClearOnlySwapchainPass()
{
    // Fallback for a frame that produced no presenter pass (no presenter yet, or a scene
    // whose assets have not landed): the window still gets the GL clear colour.
    auto& frame = frames[frameIndex];
    const auto image = swapchainImages[currentImageIndex];

    transitionImage(frame.commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchainImageViews[currentImageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color =
        VkClearColorValue{{clearColour[0], clearColour[1], clearColour[2], clearColour[3]}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, swapchainExtent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);
    vkCmdEndRendering(frame.commandBuffer);

    swapchainPassRecorded = true;
}

void VulkanRenderer::endFrame()
{
    static_cast<void>(submitAndPresent(VK_NULL_HANDLE));
}

bool VulkanRenderer::submitAndPresent(const VkBuffer captureBuffer)
{
    if (!frameOpen)
    {
        return false;
    }

    auto& frame = frames[frameIndex];
    if (!swapchainPassRecorded)
    {
        recordClearOnlySwapchainPass();
    }

    const auto image = swapchainImages[currentImageIndex];
    if (captureBuffer != VK_NULL_HANDLE)
    {
        transitionImage(frame.commandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = VkExtent3D{swapchainExtent.width, swapchainExtent.height, 1};
        vkCmdCopyImageToBuffer(frame.commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, captureBuffer, 1,
                               &region);

        transitionImage(frame.commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);
    }
    else
    {
        transitionImage(frame.commandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                        VK_ACCESS_2_NONE);
    }

    RACEENGINE_GPU_COLLECT(frame.commandBuffer);
    ensure(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");

    // The frame's uploads go first: the frame command buffer samples what they wrote, and queue
    // submission order plus the batch's own barriers is exactly the ordering the per-upload
    // fence used to provide.
    flushUploadCommands();

    // No-ops on the coherent memory VMA picks here, but the uniform writes this frame made
    // must be flushed before the submit on any host-cached heap.
    ensure(vmaFlushAllocation(allocator, frame.frameDataAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
    ensure(vmaFlushAllocation(allocator, frame.drawDataAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
    ensure(vmaFlushAllocation(allocator, frame.jointDataAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
    ensure(vmaFlushAllocation(allocator, frame.paintDataAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

    VkSemaphoreSubmitInfo waitSemaphoreInfo{};
    waitSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemaphoreInfo.semaphore = frame.imageAvailable;
    waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signalSemaphoreInfo{};
    signalSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphoreInfo.semaphore = renderFinishedSemaphores[currentImageIndex];
    signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = frame.commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;

    ensure(vkQueueSubmit2(graphicsQueue, 1, &submitInfo, frame.inFlight), "vkQueueSubmit2");

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentImageIndex];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &currentImageIndex;

    const auto presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
    {
        recreateNeeded = true;
    }
    else
    {
        ensure(presentResult, "vkQueuePresentKHR");
    }

    frameOpen = false;
    submittedFrames++;
    frameIndex = (frameIndex + 1) % framesInFlight;
    return true;
}

// The queue's whole contract, in one place.
//
// Submissions are numbered by submittedFrames: the submission being recorded right now, if a
// frame is open, will be number `submittedFrames`, and the last one already issued was
// `submittedFrames - 1`. Submission number n signals frames[n % framesInFlight].inFlight, and
// beginFrame for submission number n + framesInFlight waits on that same fence before recording
// anything. So by the time submittedFrames has reached some value C, every submission numbered
// C - framesInFlight or lower has completed on the GPU.
//
// An object being retired can be named by any submission already issued, and by the open one if
// there is a frame open. It is therefore safe once submission `submittedFrames - 1` (or
// `submittedFrames`, with a frame open) has completed, which by the rule above has happened once
// submittedFrames reaches that number + framesInFlight. readyAt is set one higher than the strict
// bound, which costs one extra frame of residency and removes the off-by-one from the argument.
void VulkanRenderer::retire(RetiredResource resource) const
{
    resource.readyAt = submittedFrames + framesInFlight + (frameOpen ? 1 : 0);
    retiredResources.push_back(resource);
}

void VulkanRenderer::collectRetiredResources() const
{
    if (retiredResources.empty())
    {
        return;
    }

    auto stillInUse = std::vector<RetiredResource>();

    for (const auto& resource : retiredResources)
    {
        if (resource.readyAt > submittedFrames)
        {
            stillInUse.push_back(resource);
            continue;
        }

        if (resource.descriptorSet != VK_NULL_HANDLE)
        {
            vkFreeDescriptorSets(device, descriptorPool, 1, &resource.descriptorSet);
        }
        if (resource.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, resource.sampler, nullptr);
        }
        if (resource.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, resource.view, nullptr);
        }
        if (resource.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator, resource.image, resource.imageAllocation);
        }
        if (resource.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, resource.buffer, resource.bufferAllocation);
        }
        if (resource.pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, resource.pipeline, nullptr);
        }
        if (resource.shaderModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device, resource.shaderModule, nullptr);
        }
    }

    retiredResources = std::move(stillInUse);
}

void VulkanRenderer::drainRetiredResources() const
{
    for (auto& resource : retiredResources)
    {
        resource.readyAt = 0;
    }

    collectRetiredResources();
}

// One view into the open frame. Every view records into the same command buffer, so N
// cameras produce N scene passes and still reach the screen through the single present
// endFrame issues.
void VulkanRenderer::recordView(Scene& scene, Camera& camera, const float delta)
{
    recordScenePass(scene, camera, delta);
}

void VulkanRenderer::recordPresent(const Presenter& presenter)
{
    const auto* shader = memoryStorageService.shaders.find(presenter.shader);
    const auto* attachment = memoryStorageService.bufferAttachments.find(presenter.output);

    // The grade the presenter named, uploaded on the frame it is first presented with, or the
    // neutral one. A table that has gone missing grades nothing rather than skipping the present:
    // the frame reaching the screen matters more than the look reaching it.
    auto lookupTableImageId = neutralLookupTable();
    if (presenter.lookupTable.has_value())
    {
        if (const auto uploaded = uploadTexture(presenter.lookupTable.value()); uploaded.has_value())
        {
            lookupTableImageId = uploaded.value();
        }
        else
        {
            diagnostics.record(FrameDiagnostic::ColourGradeUnavailable,
                               [] { return std::string("the presenter's colour grade is no longer loaded"); });
        }
    }

    if (shader != nullptr && attachment != nullptr && attachment->gpuResourceId.has_value())
    {
        recordPresentPass(shader->gpuResourceId, attachment->gpuResourceId.value(), presenter.parameters,
                          lookupTableImageId);
    }
}

bool VulkanRenderer::recordPresentPass(const unsigned int shaderId, const unsigned int attachmentImageId,
                                       const glm::vec4& parameters, const unsigned int lookupTableImageId)
{
    const auto shader = shaderObjects.find(shaderId);
    if (shader == shaderObjects.end() || shader->second.swapchainTargetPipeline == VK_NULL_HANDLE ||
        !imageResources.contains(attachmentImageId))
    {
        diagnostics.record(FrameDiagnostic::PresentPassSkipped,
                           [&]
                           {
                               return "shader object " + std::to_string(shaderId) +
                                      " has no swapchain pipeline, or attachment image " +
                                      std::to_string(attachmentImageId) + " is unknown";
                           });

        return false;
    }

    // Resolved before the swapchain image is touched: a pass that cannot bind its source
    // must leave the image untouched so the clear fallback can still take it.
    RACEENGINE_GPU_ZONE(presenterZone, frames[frameIndex].commandBuffer, "presenter");
    const GpuPassLabel presenterLabel(*this, frames[frameIndex].commandBuffer, "presenter");
    const std::array presentInputs = {PostProcessBinding{.gpuResourceId = attachmentImageId, .level = 0}};
    if (attachmentSet(presentInputs, lookupTableImageId, neutralVolumes()) == VK_NULL_HANDLE)
    {
        diagnostics.record(
            FrameDiagnostic::PresentPassSkipped,
            [&] { return "attachment image " + std::to_string(attachmentImageId) + " has no descriptor set"; });

        return false;
    }

    auto& frame = frames[frameIndex];
    transitionImage(frame.commandBuffer, swapchainImages[currentImageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    transitionTracked(frame.commandBuffer, attachmentImageId, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Neutral tone, every field: the present pass copies an image the post chain has already
    // exposed and tone mapped, so any curve here would apply the whole of it twice. What is *not*
    // neutral is `effect` — the lens and the grade belong to this pass, because this is the one
    // pass whose input is a display-referred image.
    const FullscreenPushConstants pushConstants{.tone = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
                                                .pass = glm::vec4(0.0f, 1.0f, static_cast<float>(swapchainExtent.width),
                                                                  static_cast<float>(swapchainExtent.height)),
                                                .view = glm::vec4(0.0f),
                                                .effect = parameters,
                                                // Zero, like the projection above: the present pass
                                                // stands nowhere and draws no weather.
                                                .viewRight = glm::vec4(0.0f),
                                                .viewUp = glm::vec4(0.0f),
                                                .viewBack = glm::vec4(0.0f),
                                                .weather = glm::vec4(0.0f)};
    recordFullScreenPass(presentInputs, lookupTableImageId, neutralVolumes(), swapchainImageViews[currentImageIndex],
                         swapchainExtent, shader->second.swapchainTargetPipeline, pushConstants);

    swapchainPassRecorded = true;
    lastPresentPass = PresentPass{shaderId, attachmentImageId, parameters, lookupTableImageId};
    return true;
}

void VulkanRenderer::transitionTracked(const VkCommandBuffer commandBuffer, const unsigned int imageId,
                                       const VkImageLayout newLayout)
{
    const auto entry = imageResources.find(imageId);
    if (entry == imageResources.end())
    {
        return;
    }

    auto& resource = entry->second;
    const auto destination = stageAccessFor(newLayout);

    // One barrier per run of levels that share a current layout. A whole image in one layout —
    // which is every image but one caught mid-chain — is a single run, and the barrier it emits
    // covers VK_REMAINING_MIP_LEVELS exactly as this did before there were chains at all.
    //
    // Emitted even when the layout already matches: consecutive frames write the same attachment
    // images, and the barrier is what orders this frame's access against the previous
    // submission's.
    for (auto level = 0u; level < resource.layouts.size();)
    {
        const auto runLayout = resource.layouts[level];
        auto runEnd = level + 1;
        while (runEnd < resource.layouts.size() && resource.layouts[runEnd] == runLayout)
        {
            runEnd++;
        }

        const auto source = stageAccessFor(runLayout);
        const auto wholeImage = level == 0 && runEnd == resource.layouts.size();
        transitionImage(commandBuffer, resource.image, runLayout, newLayout, source.stage, source.access,
                        destination.stage, destination.access,
                        VkImageSubresourceRange{resource.aspect, level,
                                                wholeImage ? VK_REMAINING_MIP_LEVELS : runEnd - level, 0,
                                                VK_REMAINING_ARRAY_LAYERS});
        level = runEnd;
    }

    std::ranges::fill(resource.layouts, newLayout);
}

void VulkanRenderer::transitionTrackedLevel(const VkCommandBuffer commandBuffer, const unsigned int imageId,
                                            const uint32_t level, const VkImageLayout newLayout)
{
    const auto entry = imageResources.find(imageId);
    if (entry == imageResources.end() || level >= entry->second.layouts.size())
    {
        return;
    }

    auto& resource = entry->second;
    const auto source = stageAccessFor(resource.layouts[level]);
    const auto destination = stageAccessFor(newLayout);

    transitionImage(commandBuffer, resource.image, resource.layouts[level], newLayout, source.stage, source.access,
                    destination.stage, destination.access,
                    VkImageSubresourceRange{resource.aspect, level, 1, 0, VK_REMAINING_ARRAY_LAYERS});
    resource.layouts[level] = newLayout;
}

void VulkanRenderer::recordScenePass(Scene& scene, Camera& camera, const float delta)
{
    auto& frame = frames[frameIndex];

    sceneEnvironmentImageId.reset();
    if (scene.environment.has_value())
    {
        if (const auto* environment = memoryStorageService.cubeMaps.find(scene.environment.value());
            environment != nullptr && imageResources.contains(environment->gpuResourceId))
        {
            sceneEnvironmentImageId = environment->gpuResourceId;
        }
    }

    std::optional<unsigned int> colorImageId;
    std::optional<unsigned int> depthImageId;
    const auto* outputBuffer =
        camera.output.has_value() ? memoryStorageService.frameBuffers.find(camera.output.value()) : nullptr;
    if (outputBuffer != nullptr)
    {
        for (const auto& attachmentKey : outputBuffer->attachments)
        {
            const auto* attachment = memoryStorageService.bufferAttachments.find(attachmentKey);
            if (attachment == nullptr || !attachment->gpuResourceId.has_value())
            {
                continue;
            }

            if (attachment->type == FboAttachmentType::Color && !colorImageId.has_value())
            {
                colorImageId = attachment->gpuResourceId;
            }
            else if (attachment->type == FboAttachmentType::Depth && !depthImageId.has_value())
            {
                depthImageId = attachment->gpuResourceId;
            }
        }
    }

    // A camera renders into whatever its framebuffer carries. Colour and depth is the pair every
    // on-screen camera has; a depth-only target — a shadow cascade, a depth pre-pass — records a
    // rendering info with no colour attachment at all, which is something dynamic rendering
    // expresses directly and the pipeline has to be told (see scenePipeline). What is not a
    // camera is a framebuffer with neither.
    const auto hasColor = colorImageId.has_value() && imageResources.contains(colorImageId.value());
    const auto hasDepth = depthImageId.has_value() && imageResources.contains(depthImageId.value());

    if (!hasColor && !hasDepth)
    {
        diagnostics.record(FrameDiagnostic::MissingCameraOutput,
                           [] { return std::string("the scene pass records nothing"); });

        return;
    }

    // The view's own shader, if it has one, resolved once: it is constant for the whole view, and
    // a view that cannot get it must record nothing rather than fall back to the entities' own —
    // those write colour, and this target may have no colour attachment to write. Checked before
    // anything is recorded, so an abandoned view leaves no half-open rendering behind it.
    const auto* viewShader =
        camera.overrideShader.has_value() ? memoryStorageService.shaders.find(camera.overrideShader.value()) : nullptr;

    if (camera.overrideShader.has_value() && viewShader == nullptr)
    {
        diagnostics.record(FrameDiagnostic::ViewShaderUnavailable,
                           [&] { return "shader slot " + std::to_string(camera.overrideShader->index); });

        return;
    }

    // The view's name over everything it records — the scene draws and the post chain both — so a
    // capture and a GPU profile read "world" or "cascade 2" rather than an anonymous run of draws.
    const auto passName = camera.debugName.empty() ? std::string(describeCameraRole(camera.role)) : camera.debugName;
    RACEENGINE_GPU_ZONE(scenePassZone, frame.commandBuffer, passName.c_str());
    const GpuPassLabel passLabel(*this, frame.commandBuffer, passName.c_str());

    FrameDataUbo frameData{};
    frameData.viewMatrix = camera.modelViewMatrix;
    frameData.cameraPosition = glm::vec4(camera.position, 1.0f);

    // A scene with no lights uploads lightCount 0: both shader loops then contribute nothing
    // and the ambient floor is zero, leaving the image-based term as the only lighting. That
    // is a legitimate scene, not a failure, so nothing is logged.
    auto uploadedLights = 0u;
    auto declaredLights = 0u;
    for (const auto& light : scene.lights)
    {
        declaredLights++;

        if (uploadedLights >= maxLights)
        {
            continue;
        }

        frameData.lights[uploadedLights] =
            LightUbo{glm::vec4(light.position, 1.0f), glm::vec4(light.diffuse, 0.0f), glm::vec4(light.specular, 0.0f),
                     glm::vec4(light.ambient, light.attenuation)};
        uploadedLights++;
    }
    frameData.lightCount = glm::ivec4(static_cast<int>(uploadedLights), 0, 0, 0);

    uploadProbes(scene, frameData);
    uploadFog(scene.fog, frameData);
    // The rain through this view's own scale — a sheltered view (Camera::rainScale) shades its
    // surfaces dry while the weather push constant below stays the scene's, so the streaks outside
    // a dry cabin still fall.
    frameData.timeRain = glm::vec4(static_cast<float>(frameSimulationTime), scene.rain * camera.rainScale,
                                   scene.rainMotion.x, scene.rainMotion.y);
    frameData.rainWind = glm::vec4(scene.rainForward, 0.0f);
    frameData.rainBody = glm::vec4(scene.rainBodyUp, 0.0f);
    frameData.wiperArcA = scene.wipers.bladeA;
    frameData.wiperArcB = scene.wipers.bladeB;
    frameData.wiperSweep = glm::vec4(scene.wipers.parkAngle.x, scene.wipers.sweepAngle.x, scene.wipers.parkAngle.y,
                                     scene.wipers.sweepAngle.y);
    frameData.wiperTiming = glm::vec4(scene.wipers.cyclePeriod, scene.wipers.sweepSeconds, scene.wipers.cycleStart,
                                      scene.wipers.bladeHalfWidth);
    frameData.wiperPane = glm::vec4(scene.wipers.paneAspect, 0.0f, 0.0f, 0.0f);
    frameData.cloudParams = glm::vec4(scene.clouds.coverage, scene.clouds.type, 0.0f, 0.0f);

    // A cascade is a producer and samples nothing; only the views that shade read the maps, and a
    // cascade sampling its own attachment while rendering into it would be a feedback loop.
    const auto shading = camera.role == CameraRole::Scene;
    const auto cascadeImages = shading ? shadowCascadeImages(scene) : std::nullopt;
    // The occlusion this view samples rides in the same set, so it is resolved with the cascades and
    // falls back with them: a view that gathers none binds white, and white is no occlusion at all.
    const auto occlusionImage = shading ? ambientOcclusionImage(camera) : dummyTexture();
    // Likewise the behind copy, resolved up front so the one set this view binds already names it:
    // the image only gains its contents mid-pass, but a descriptor names a view, not contents, and
    // nothing samples it before the copy runs — the opaque shaders never declare the binding. An
    // opaque-only view can never split — the copy runs between the halves and it has only one — so
    // it binds the dummy rather than allocating a full-resolution chain nothing will ever fill.
    const auto behindImage =
        shading && hasColor && hasDepth && camera.partition != DrawPartition::OpaqueOnly
            ? sceneBehindImage(colorImageId.value())
            : dummyTexture();
    // Splitting is only worth anything when there is a real behind image to fill; a view whose
    // colour attachment could not be resolved shades over the white dummy exactly as a view with
    // no occlusion does.
    const auto splitForBehind = shading && hasColor && hasDepth && behindImage != dummyTexture();
    // The cloud map rides the same set and resolves with the same shape: the scene's own image for
    // a view that shades, the dummy otherwise. It needs no transition here — the pass that wrote
    // it left it SHADER_READ_ONLY, and the initial clear puts a never-written one there too.
    const auto cloudImage = shading ? cloudMapImage(scene) : dummyTexture();
    auto shadowDescriptors = cascadeImages.has_value()
                                 ? shadowSet(cascadeImages.value(), occlusionImage, behindImage, cloudImage)
                                 : VK_NULL_HANDLE;

    if (shadowDescriptors != VK_NULL_HANDLE)
    {
        const auto correction = shadowLookupCorrection();

        for (auto cascade = 0u; cascade < shadowCascadeCount; cascade++)
        {
            const auto& slice = scene.shadows.cascades[cascade];
            const auto index = static_cast<int>(cascade);
            frameData.shadowMatrices[cascade] = correction * slice.camera->modelViewProjectionMatrix;
            frameData.shadowSplits[index] = slice.splitDistance;
            frameData.shadowTexelWorldSize[index] = slice.texelWorldSize;
            frameData.shadowDepthScale[index] = slice.depthPerWorldUnit;
        }

        frameData.shadowParams =
            glm::ivec4(static_cast<int>(shadowCascadeCount), static_cast<int>(scene.shadows.lightIndex), 0, 0);
    }
    else
    {
        // Zero cascades is what tells the shader to shade lit; the set still has to be bound,
        // because the pipeline statically uses the samplers in it (see fallbackShadowSet).
        shadowDescriptors = fallbackShadowSet();

        if (shading && !scene.shadows.cascades.empty())
        {
            diagnostics.record(FrameDiagnostic::ShadowCascadeUnavailable,
                               [] { return std::string("no Vulkan depth image for one of the cascades"); });
        }
    }

    if (declaredLights > maxLights)
    {
        diagnostics.record(FrameDiagnostic::LightLimitExceeded,
                           [&]
                           {
                               return "Vulkan frame data carries " + std::to_string(maxLights) + " of the " +
                                      std::to_string(declaredLights) + " declared";
                           });
    }

    // Its own slot in the ring, kept until the frame is submitted: the next view records into
    // the same command buffer, and a shared slot would have its camera shade this one's draws.
    const auto frameDataSlot = allocateFrameDataSlot();
    if (!frameDataSlot.has_value())
    {
        return;
    }

    frame.frameDataOffset = frameDataSlot.value();
    std::memcpy(static_cast<char*>(frame.frameDataMapped) + frame.frameDataOffset, &frameData, sizeof(frameData));

    if (hasColor)
    {
        transitionTracked(frame.commandBuffer, colorImageId.value(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    if (hasDepth)
    {
        transitionTracked(frame.commandBuffer, depthImageId.value(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }

    // Copied, not referenced: the lazy uploads below insert into imageResources, and a
    // rehash would leave a reference into it dangling.
    const auto colorImage = hasColor ? imageResources.at(colorImageId.value()) : ImageResource{};
    const auto depthImage = hasDepth ? imageResources.at(depthImageId.value()) : ImageResource{};
    // The extent is the target's, whichever attachment the target has: a depth-only camera is
    // sized by its depth image exactly as a normal one is by its colour image.
    const auto& extentSource = hasColor ? colorImage : depthImage;
    const VkExtent2D extent{extentSource.width, extentSource.height};

    // A prepass clears to zero rather than to the scene's clear colour, and the difference matters:
    // what it writes is geometry, not colour, and every consumer of it reads a zero distance as "the
    // camera saw nothing here". Cleared to white, the sky would arrive at the gather as a surface
    // one unit in front of the eye — which is not a subtle error but it is a plausible-looking one.
    // A camera's own clearColour wins over both: a layer buffer clears to transparent black because
    // where nothing drew there must be nothing for the composite to lay over.
    const auto& contractClear = camera.role == CameraRole::DepthNormalPrepass ? prepassClearColour : clearColour;
    const auto clear =
        camera.clearColour.value_or(glm::vec4(contractClear[0], contractClear[1], contractClear[2], contractClear[3]));

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = colorImage.view;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // A camera that continues a frame begins from what the previous view left; every other camera
    // clears. The contents are defined because the view that filled them stored and the tracked
    // transitions in between preserve — only a transition out of UNDEFINED discards, and a loading
    // camera's target was written, and therefore tracked, earlier in this same frame.
    colorAttachment.loadOp = camera.loadColour ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = VkClearColorValue{{clear.x, clear.y, clear.z, clear.w}};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthImage.view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = camera.loadDepth ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Kept only where something reads it, which on this engine is exactly the shadow cascades. A
    // depth attachment is given a sampler when, and only when, its FboAttachment asked for a
    // comparison — so the handle being null *is* the statement that nothing samples this buffer,
    // and no second flag is needed to say it.
    //
    // The two that do not: the shading camera's depth, which the tone map binds as its second input
    // and never reads (HdrFragmentShader samples inputs[0] and inputs[2]), and the occlusion
    // prepass's, whose own comment in AmbientOcclusionService says it is never sampled. Storing
    // them writes two full-resolution D32 buffers a frame that nothing looks at, and — the larger
    // cost — forces a depth decompression when the image is moved to a read layout afterwards. The
    // move itself stays: the descriptor written for it promises that layout whether or not the
    // shader reads it, and a mismatch there is undefined even for a binding nothing samples.
    // A split view is the exception to the policy above, for its first half only: ending the
    // rendering block between the opaque and blended draws makes the store real — the resumed half
    // loads this depth and the blended draws test against it — so a view that may split stores
    // whether or not anything samples the buffer afterwards. recordSceneBehindCopy puts the policy
    // store op back for the resumed half, which is the half the comment above was written about.
    // `keepDepth` joins the sampler in the policy rather than the split exception: a layered
    // frame's depth is consumed by the *next view's* rasteriser, which no sampler handle can prove,
    // so the camera states it and it holds for the resumed half of a split too.
    const auto depthStorePolicy = depthImage.sampler != VK_NULL_HANDLE || camera.keepDepth
                                      ? VK_ATTACHMENT_STORE_OP_STORE
                                      : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.storeOp = splitForBehind ? VK_ATTACHMENT_STORE_OP_STORE : depthStorePolicy;
    depthAttachment.clearValue.depthStencil = VkClearDepthStencilValue{1.0f, 0};

    // A camera with no colour attachment renders colour-less and one with no depth attachment
    // renders depth-less; either way the pipeline has to be told, so both formats are what the
    // attachments actually bound report, and VK_FORMAT_UNDEFINED means "not bound".
    const auto colorFormat = hasColor ? colorImage.format : VK_FORMAT_UNDEFINED;
    const auto depthFormat = hasDepth ? depthImage.format : VK_FORMAT_UNDEFINED;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, extent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = hasColor ? 1u : 0u;
    renderingInfo.pColorAttachments = hasColor ? &colorAttachment : nullptr;
    if (hasDepth)
    {
        renderingInfo.pDepthAttachment = &depthAttachment;
    }

    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);

    // GL-convention Y-flip via negative viewport height (vulkan-abi.md): the scene lands
    // in the attachment the same way up as GL leaves it, so the fullscreen chain that
    // samples it stays an identity mapping.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(extent.height);
    viewport.width = static_cast<float>(extent.width);
    viewport.height = -static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);

    const VkRect2D scissor{VkOffset2D{0, 0}, extent};
    vkCmdSetScissor(frame.commandBuffer, 0, 1, &scissor);

    // Handed down rather than acted on here because only the draw walk knows whether this view has
    // blended draws at all: a view with none never splits, and the whole pass records exactly as it
    // did before the behind copy existed.
    const SceneBehindCopy behindCopy{.colorImageId = hasColor ? colorImageId.value() : 0u,
                                     .behindImageId = behindImage,
                                     .renderingInfo = &renderingInfo,
                                     .colorAttachment = &colorAttachment,
                                     .depthAttachment = &depthAttachment,
                                     .depthStoreAfterCopy = depthStorePolicy};

    const auto recordedDraws =
        recordSceneDraws(scene, camera, delta, shading, false, viewShader != nullptr ? viewShader->gpuResourceId : 0u,
                         colorFormat, depthFormat, shadowDescriptors, splitForBehind ? &behindCopy : nullptr);

    vkCmdEndRendering(frame.commandBuffer);

    // Every attachment this view wrote ends the pass readable. The colour one has always moved
    // here because the post chain and the presenter sample it; the depth one moves too, and that
    // move is the whole of what "Vulkan can sample a depth attachment" needed — the image already
    // carried SAMPLED usage, and the tracked layout is what emits the barrier and what puts it
    // back to DEPTH_ATTACHMENT_OPTIMAL at the top of the next view that draws into it. Nothing
    // samples the depth yet; the layout is what makes sampling it possible.
    if (hasColor)
    {
        transitionTracked(frame.commandBuffer, colorImageId.value(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (hasDepth)
    {
        transitionTracked(frame.commandBuffer, depthImageId.value(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // The camera's own velocity, from where it stood on the last shading view it recorded. Zero on
    // the first frame and on any camera that holds still, which is every gate capture's.
    auto cameraVelocity = glm::vec3(0.0f);
    if (camera.role == CameraRole::Scene)
    {
        const auto [track, firstVisit] =
            cameraTracks.try_emplace(&camera, CameraTrack{camera.position, frameSimulationTime});
        if (!firstVisit)
        {
            const auto elapsed = frameSimulationTime - track->second.simulationTime;
            if (elapsed > 0.0)
            {
                cameraVelocity = (camera.position - track->second.position) / static_cast<float>(elapsed);
            }

            track->second = CameraTrack{camera.position, frameSimulationTime};
        }
    }

    for (const auto& postProcessKey : camera.postProcesses)
    {
        const auto* postProcess = memoryStorageService.postProcesses.find(postProcessKey);
        const auto* postProcessShader =
            postProcess == nullptr ? nullptr : memoryStorageService.shaders.find(postProcess->shader);
        if (postProcess == nullptr || postProcessShader == nullptr)
        {
            diagnostics.record(FrameDiagnostic::PostProcessSkipped,
                               [&]
                               {
                                   return "post-process slot " + std::to_string(postProcessKey.index) +
                                          " or its shader is no longer loaded";
                               });

            continue;
        }

        // A held pass's target already holds the right picture, so the frame spends nothing on it —
        // no descriptor set, no barrier, no draw. The target keeps the layout the last pass over it
        // left, which is SHADER_READ_ONLY, so everything that samples it this frame still can; a
        // target no pass has ever written is in that layout too, because the only kind that is read
        // before it is written states an `initialColour` and the clear leaves it there.
        if (postProcess->contentsHeld)
        {
            continue;
        }

        // Every input the pass declared, in order, resolved to the image and level a descriptor
        // names. GL bound these to texture units 0..n and this backend used to bind only the first
        // of them; the fullscreen set is postProcessInputCount samplers wide now, and a shader
        // reads the elements it declared an interest in.
        std::vector<PostProcessBinding> inputs;
        inputs.reserve(postProcess->inputs.size());
        for (const auto& input : postProcess->inputs)
        {
            // A picture rather than something this frame drew — uploaded on first use exactly as the
            // colour grade is, and level zero because a plate has no chain worth walking. It is
            // resolved first because when it is set it *is* the input, and `attachment` is then a
            // default-constructed handle naming nothing.
            std::optional<unsigned int> imageId;
            if (input.texture.has_value())
            {
                imageId = uploadTexture(input.texture.value());
            }
            else if (const auto* attachment = memoryStorageService.bufferAttachments.find(input.attachment);
                     attachment != nullptr)
            {
                imageId = attachment->gpuResourceId;
            }

            if (!imageId.has_value())
            {
                continue;
            }

            const auto image = imageResources.find(imageId.value());
            if (image == imageResources.end())
            {
                continue;
            }

            // Clamped here rather than where the view is picked, so that the barrier and the
            // descriptor cannot end up naming different subresources: one of them falling back to
            // the whole image while the other moved a level is exactly the layout mismatch this
            // per-level bookkeeping exists to prevent.
            inputs.push_back(PostProcessBinding{.gpuResourceId = imageId.value(),
                                                .level = std::min(input.level, image->second.mipLevels - 1u)});
        }

        if (inputs.size() > postProcessInputCount)
        {
            diagnostics.record(FrameDiagnostic::PostProcessInputsExceeded,
                               [&]
                               {
                                   return "the fullscreen set carries " + std::to_string(postProcessInputCount) +
                                          " of the " + std::to_string(inputs.size()) + " declared";
                               });
        }

        // The pass's volumes, into the two sampler3D slots beside the grade. Neutral in any slot
        // the pass leaves empty — the layout declares both bindings whatever the shader reads —
        // and uploaded on first use exactly as the grade is.
        auto volumes = neutralVolumes();
        const auto boundVolumes = std::min<size_t>(postProcess->volumes.size(), postProcessVolumeCount);
        for (size_t slot = 0; slot < boundVolumes; slot++)
        {
            if (const auto volumeId = uploadTexture(postProcess->volumes[slot]); volumeId.has_value())
            {
                volumes[slot] = volumeId.value();
            }
            else
            {
                diagnostics.record(FrameDiagnostic::VolumeInputUnavailable,
                                   [&] { return "volume slot " + std::to_string(slot) + " names no live texture"; });
            }
        }

        if (postProcess->volumes.size() > postProcessVolumeCount)
        {
            diagnostics.record(FrameDiagnostic::PostProcessVolumesExceeded,
                               [&]
                               {
                                   return "the fullscreen set carries " + std::to_string(postProcessVolumeCount) +
                                          " of the " + std::to_string(postProcess->volumes.size()) + " declared";
                               });
        }

        std::optional<unsigned int> targetImageId;
        if (postProcess->output.has_value())
        {
            const auto* targetBuffer = memoryStorageService.frameBuffers.find(postProcess->output.value());
            if (targetBuffer != nullptr)
            {
                for (const auto& attachmentKey : targetBuffer->attachments)
                {
                    const auto* attachment = memoryStorageService.bufferAttachments.find(attachmentKey);
                    if (attachment != nullptr && attachment->type == FboAttachmentType::Color &&
                        attachment->gpuResourceId.has_value())
                    {
                        targetImageId = attachment->gpuResourceId;
                        break;
                    }
                }
            }
        }

        const auto shader = shaderObjects.find(postProcessShader->gpuResourceId);
        // A pass fed only by volume textures has no 2D input to name — the cloud dome marches
        // 3D noise and reads no attachment — so volumes satisfy the has-something-to-read test.
        const auto usable = (!inputs.empty() || !postProcess->volumes.empty()) && targetImageId.has_value() &&
                            imageResources.contains(targetImageId.value()) && shader != shaderObjects.end() &&
                            shader->second.fullscreen;
        // The pipeline is built against the target attachment's own format, so a post-process
        // buffer created with any FboAttachment::internalFormat renders rather than mismatching
        // a hardcoded one.
        const auto blendMode = !postProcess->blend                    ? FullscreenBlend::None
                               : postProcess->blendWeight.has_value() ? FullscreenBlend::Constant
                                                                      : FullscreenBlend::SourceAlpha;
        const auto pipeline = usable ? offscreenPipeline(postProcessShader->gpuResourceId,
                                                         imageResources.at(targetImageId.value()).format, blendMode)
                                     : VK_NULL_HANDLE;
        if (pipeline == VK_NULL_HANDLE)
        {
            diagnostics.record(FrameDiagnostic::PostProcessSkipped,
                               [&]
                               {
                                   return "shader object " + std::to_string(postProcessShader->gpuResourceId) +
                                          " has no fullscreen pipeline for this target";
                               });

            continue;
        }

        // A level past the end of the chain would leave the pass rendering through the whole-image
        // view, which dynamic rendering rejects; clamping means a mis-stated level writes the last
        // level of the chain rather than nothing at all. Read out rather than held by reference:
        // the descriptor set the pass binds may create the fallback image, and that inserts into
        // imageResources.
        const auto& target = imageResources.at(targetImageId.value());
        const auto targetLevel = std::min(postProcess->outputLevel, target.mipLevels - 1u);
        const auto targetLevels = target.mipLevels;
        const auto targetView = levelView(target, targetLevel);
        const VkExtent2D targetExtent{std::max(target.width >> targetLevel, 1u),
                                      std::max(target.height >> targetLevel, 1u)};

        // Inputs first, then the level being written: if one of the inputs is a level of the target
        // image — which is what a chain pass reading the level above the one it writes looks like —
        // the second move has to be the one that lands.
        for (const auto& input : inputs)
        {
            transitionTrackedLevel(frame.commandBuffer, input.gpuResourceId, input.level,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        transitionTrackedLevel(frame.commandBuffer, targetImageId.value(), targetLevel,
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // The projection this view was rendered with, as the pair of half-angle tangents a pass
        // reconstructing a position from a depth needs. Orthographic views have no such pair and no
        // pass that reads one; the field is what the perspective arm reports and zero otherwise.
        const auto tanHalfVertical = camera.projection == CameraProjection::Perspective
                                         ? std::tan(glm::radians(camera.fieldOfView) * 0.5f)
                                         : 0.0f;

        // The view transform run backwards, for the passes that work in the world: the rotation is
        // orthonormal, so its transpose is its inverse and nothing is inverted per pass.
        const auto viewToWorld = glm::transpose(glm::mat3(camera.modelViewMatrix));

        const FullscreenPushConstants parameters{
            .tone =
                glm::vec4(camera.exposure, camera.toneCurve.contrast, camera.toneCurve.toe, camera.toneCurve.shoulder),
            .pass = glm::vec4(static_cast<float>(targetLevel), static_cast<float>(targetLevels),
                              static_cast<float>(targetExtent.width), static_cast<float>(targetExtent.height)),
            .view = glm::vec4(tanHalfVertical * camera.aspectRatio, tanHalfVertical, camera.nearClippingPlane,
                              camera.farClippingPlane),
            .effect = postProcess->parameters,
            .viewRight = glm::vec4(viewToWorld[0], camera.position.x),
            .viewUp = glm::vec4(viewToWorld[1], camera.position.y),
            .viewBack = glm::vec4(viewToWorld[2], camera.position.z),
            .weather =
                glm::vec4(static_cast<float>(frameSimulationTime), scene.rain, cameraVelocity.x, cameraVelocity.z)};

        // Every pass before the present works in radiance, where a grade has no meaning.
        const auto postName = postProcess->debugName.empty() ? std::string("post-process") : postProcess->debugName;
        RACEENGINE_GPU_ZONE(postPassZone, frame.commandBuffer, postName.c_str());
        const GpuPassLabel postLabel(*this, frame.commandBuffer, postName.c_str());
        recordFullScreenPass(inputs, neutralLookupTable(), volumes, targetView, targetExtent, pipeline, parameters,
                             shadowDescriptors, postProcess->loadColour, postProcess->blendWeight.value_or(1.0f),
                             postProcess->slice, postProcess->sliceCount);

        transitionTrackedLevel(frame.commandBuffer, targetImageId.value(), targetLevel,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (!drawSummaryLogged && recordedDraws > 0)
    {
        const auto* attachments = hasColor ? (hasDepth ? "colour and depth" : "colour only") : "depth only";

        logger.info("Vulkan scene pass recorded: {} draw(s) into {}x{} ({}), {} post-process pass(es), {} scene "
                    "pipeline(s), {} material set(s)",
                    recordedDraws, extent.width, extent.height, attachments, camera.postProcesses.size(),
                    scenePipelines.size(), materialResources.size());
        drawSummaryLogged = true;
    }
}

// The entity/mesh/primitive walk, shared by a camera's view and a light probe's face.
//
// `viewShaderId` of zero means "each entity draws with its own"; `shading` is false only for a
// depth pass; `staticOnly` is what a probe capture adds. The two filters are separate flags
// because the sky is the case that differs between them: it must never cast a shadow, and it is
// the single most important thing a probe records.
//
// `staticOnly` doubles as the double-sided flag it hands recordDraw, for a reason that is a
// property of the pass rather than of the geometry: a cube face is rasterised with a positive
// viewport height where every other pass here uses a negative one, so the winding a
// back-face-culling pipeline was built against is inverted. Culling on inverted winding keeps the
// far side of a wall and discards the near one, which reads as a probe seeing straight through
// buildings. Culling nothing and letting the depth test decide is correct either way round.
unsigned int VulkanRenderer::recordSceneDraws(Scene& scene, const Camera& camera, const float delta, const bool shading,
                                              const bool staticOnly, const unsigned int viewShaderId,
                                              const VkFormat colorFormat, const VkFormat depthFormat,
                                              const VkDescriptorSet shadowDescriptors,
                                              const SceneBehindCopy* behindCopy)
{
    // Same hoisting rule the GL pass had: every handle is resolved at the outermost loop where it
    // is constant and read through for the nest below, so the pass costs
    // O(entities + meshes + primitives) storage lookups rather than one per field. A handle that
    // has gone stale — a model unloaded while its renderable is still in the scene — is skipped
    // here, not chased into a recycled slot.
    auto recordedDraws = 0u;

    // This view's own clip volume, extracted once. Every pass this function serves — the four
    // cascades, the occlusion prepass, the shading pass and a probe face — arrives here with its
    // matrix already built, so one extraction covers all of them and no pass needs a case of its
    // own (:Frustum states why, including why the cascade case is the one that looks like it needs
    // special handling and does not).
    const auto viewPlanes = frustumPlanes(camera.modelViewProjectionMatrix);

    // Likewise view-constant. `*` associates left to right, so folding the correction in here gives
    // every draw below the same `(clipCorrection() * mvp) * model` it computed for itself before.
    const auto clipCorrectedViewProjection = clipCorrection() * camera.modelViewProjectionMatrix;

    // Blended geometry does not draw where it is found. Alpha blending is not commutative — the
    // result depends on the order the fragments arrive in — so a transparent surface is only
    // correct once everything visible *through* it is already in the buffer. Recording it in glTF
    // node order means a windscreen drawn before the dashboard blends with the sky and then has
    // the dashboard drawn over the top of it, and the pane vanishes from wherever the interior
    // covers it. So the pass records every opaque draw first, then these, sorted back to front.
    //
    // The sort key is the view-space depth of the primitive's own bounding-box centre. Per
    // primitive rather than per entity or per node, because this asset's 79 blended primitives all
    // hang off nodes that resolve to the same origin: keyed on the node they would sort by 79
    // copies of one number, and the sort would be a no-op that looked like a fix.
    struct DeferredDraw
    {
        const MeshPrimitive* primitive;
        Resource<Material> materialKey;
        const std::vector<glm::mat4>* joints;
        // The slot this renderable's paint was written into, taken once for the whole renderable
        // before its primitives were walked.
        VkDeviceSize paintOffset;
        glm::mat4 entityModelMatrix;
        unsigned int shaderId;
        float viewDepth;
    };

    std::vector<DeferredDraw> blendedDraws;

    for (auto& entity : scene.models)
    {
        // The depth pass draws casters. A skybox in the map fills it at the near plane and puts
        // the whole world in shadow.
        if (!shading && !entity.castsShadow)
        {
            continue;
        }

        // A probe's capture draws the world, not the traffic on it. Anything that moves would be
        // baked into an environment shaded from for many frames afterwards, lighting the street
        // from wherever it happened to be when the capture ran.
        if (staticOnly && !entity.staticGeometry)
        {
            continue;
        }

        // A camera draws the layers its mask names, which is the whole of how a layered frame
        // splits: everything is born on layer 1 and the default mask draws every layer, so a scene
        // that never states layers is unchanged. The occlusion prepass arrives here with its mask
        // forced to every layer whatever its shading camera drew (recordAmbientOcclusion carries
        // why: the contact shadow under a car is exactly the pixel where two layers meet).
        if ((camera.layerMask & entity.layers) == 0u)
        {
            continue;
        }

        const auto* model = memoryStorageService.models.find(entity.model);
        const auto* instanceShader = memoryStorageService.shaders.find(entity.shader);
        if (model == nullptr || instanceShader == nullptr)
        {
            diagnostics.record(FrameDiagnostic::StaleModelHandle,
                               [&] { return "model slot " + std::to_string(entity.model.index); });

            continue;
        }

        // The view's shader where it has one, otherwise the instance's — not the material's:
        // materials live in shared storage, so two renderables built from one model would
        // otherwise restyle each other.
        const auto shaderId = viewShaderId != 0 ? viewShaderId : instanceShader->gpuResourceId;

        // This renderable's paint, written once here and read by every primitive it is about to
        // draw. Per renderable and not per draw, because that is the scope the data has: a car is
        // one colour, and eight hundred body panels asking the ring for eight hundred copies of it
        // exhausted a thousand slots in one frame and drew the car unpainted.
        const auto paintOffset = writePaintSlot(entity.paint);

        // Around the entity's submission, which on this backend is where its commands are
        // recorded: the same point in the frame GL calls them at.
        if (entity.beforeDraw.has_value())
        {
            (*entity.beforeDraw)();
        }

        for (auto& renderableMesh : entity.meshes)
        {
            const auto* mesh = memoryStorageService.meshes.find(renderableMesh.mesh);
            if (mesh == nullptr)
            {
                diagnostics.record(FrameDiagnostic::StaleMeshHandle,
                                   [&] { return "mesh slot " + std::to_string(renderableMesh.mesh.index); });

                continue;
            }

            // The entity's flag keeps a whole renderable out of the depth pass; this one keeps a
            // single mesh of it out. A track is the case that needs both: one model, and inside it
            // the decals — skid marks, kerb overlays, painted lines — that must be drawn and must
            // not cast, because each sits a millimetre above the surface it decorates and would
            // otherwise print a dark ghost of its own outline onto it.
            //
            // **The cascades only, and that is the whole of the distinction.** `castsShadow` is a
            // statement about casting, and the occlusion prepass is not a shadow: it records the
            // geometry a surface is *made of* so the gather can measure how much sky each pixel can
            // see. Tested against `!shading` this also emptied the prepass — on this circuit the
            // road itself is flagged non-casting along with its decals, so the tarmac was absent
            // from the occlusion buffer entirely while the posts and trunks standing on it were
            // not. The gather then had an occluder with no ground around it and printed the post's
            // own outline onto road it could not see, which is the "I can see the tree trunk
            // through the road" that was reported from the seat and took six diagnoses to find.
            // Measured on that view: the prepass went from 540 draws covering 36.5% of the frame to
            // 582 covering 79.0%, and its nearest recorded surface from 3.75 m to 0.33 m.
            if (camera.role == CameraRole::ShadowCascade && !mesh->castsShadow)
            {
                continue;
            }

            if (!mesh->gpuResourceId.has_value())
            {
                // upload() writes through mutate(), in place, so this borrow still names the same
                // element afterwards — carrying the ids the upload just produced.
                upload(entity.model);
            }

            if (!mesh->gpuResourceId.has_value())
            {
                diagnostics.record(FrameDiagnostic::MeshNotUploaded, [&] { return "mesh " + mesh->name; });

                continue;
            }

            const auto entityModelMatrix =
                sceneManagerService.modelMatrix(entity.node) * mesh->modelMatrix * renderableMesh.localTransform;
            // Bound by reference into the mesh's palette buffer; copying undoes the per-frame
            // allocation the service avoids.
            const auto& joints = renderableEntityService.joints(renderableMesh, delta);

            for (const auto& primitive : mesh->meshPrimitives)
            {
                if (!model->meshBuffers[static_cast<size_t>(primitive.meshBufferIndex)].gpuId.has_value())
                {
                    diagnostics.record(FrameDiagnostic::MeshBufferNotUploaded, [&] { return "mesh " + mesh->name; });

                    continue;
                }

                const auto* material = primitive.material.has_value()
                                           ? memoryStorageService.materials.find(primitive.material.value())
                                           : nullptr;

                if (material == nullptr || !material->shader.has_value())
                {
                    diagnostics.record(FrameDiagnostic::PrimitiveWithoutMaterial, [&] { return "mesh " + mesh->name; });

                    continue;
                }

                // A material that *declared* a shader is drawn with it, and this is the one place a
                // material may decide a pipeline. `Material::shader` still may not — it holds the
                // renderable's own choice written through as a default, and picking from it would let
                // one car restyle every other car built from the same model. A declared shader is the
                // asset's, so every renderable built from that model wants it equally.
                //
                // A view with an override still wins over both: a shadow cascade draws depth, and
                // what a surface would rather be shaded by does not come into it.
                auto primitiveShaderId = shaderId;
                if (viewShaderId == 0 && material->declaredShaderHandle.has_value())
                {
                    if (const auto* declared = memoryStorageService.shaders.find(material->declaredShaderHandle.value());
                        declared != nullptr)
                    {
                        primitiveShaderId = declared->gpuResourceId;
                    }
                }

                if (!primitive.gpuVao.has_value())
                {
                    diagnostics.record(FrameDiagnostic::PrimitiveNotUploaded, [&] { return "mesh " + mesh->name; });

                    continue;
                }

                // The occlusion prepass describes the geometry that shades, and a transparent
                // surface is not it: there is no single depth to stand for a pane you can see
                // through, and a car's glazing is always two near-coplanar shells. Written into
                // the prepass they fight — that pass writes depth for everything — and the gather
                // reads the fight as occlusion and prints it back onto the glass as hard-edged
                // wedges. Not deferred, dropped: a prepass has no blending to order, and the
                // surface does not belong in the buffer at all. `PbrFragmentShader` states the
                // other half of the same rule, and a transparent fragment reads no occlusion.
                if (camera.role == CameraRole::DepthNormalPrepass && !material->opaque)
                {
                    continue;
                }

                // Nothing this view can see, so nothing this view need record. The box is the one
                // the POSITION accessor declared, carried into world space by the same matrix the
                // vertex stage is about to be handed, and the planes are this view's own clip
                // volume — so a primitive wholly outside one of them is clipped by the rasteriser
                // before it can write a fragment, and skipping the draw records the same image.
                //
                // Two exemptions, both of which fail safe by drawing:
                //
                //   - a skinned draw, because the palette moves vertices outside the bind pose the
                //     accessor measured, so those bounds are not a bound at all. `joints` is the
                //     same vector recordDraw reads to decide whether this draw is animated.
                //   - a primitive whose accessor declared no bounds, which keeps a zero half-extent
                //     and would otherwise be culled for being unmeasured rather than for being
                //     invisible.
                //
                // The test sits below every guard above it on purpose: the mesh upload, the joint
                // palette and the entity's own draw callbacks are side effects a view owes whether
                // or not this primitive is visible, and skipping them would make what is on screen
                // depend on what was on screen last frame.
                if (joints.empty() && glm::all(glm::greaterThan(primitive.boundsHalfExtent, glm::vec3(0.0f))) &&
                    aabbOutsideFrustum(viewPlanes, entityModelMatrix, primitive.boundsCentre,
                                       primitive.boundsHalfExtent))
                {
                    continue;
                }

                // Depth-only and prepass views have no blending to order — they write data, and
                // their shaders carry no alpha at all — so only a shading view defers.
                if (shading && !material->opaque)
                {
                    // A view that draws only the opaque half of the frame leaves every blended draw
                    // to the view that finishes it, where the global back-to-front sort still holds
                    // across layers. See DrawPartition.
                    if (camera.partition == DrawPartition::OpaqueOnly)
                    {
                        continue;
                    }

                    const auto centreInView =
                        camera.modelViewMatrix * entityModelMatrix * glm::vec4(primitive.boundsCentre, 1.0f);

                    blendedDraws.push_back(DeferredDraw{.primitive = &primitive,
                                                        .materialKey = primitive.material.value(),
                                                        .joints = &joints,
                                                        .paintOffset = paintOffset,
                                                        .entityModelMatrix = entityModelMatrix,
                                                        .shaderId = primitiveShaderId,
                                                        // The view looks down -z, so the distance
                                                        // in front of the eye is -z (the same
                                                        // convention the shadow lookup reads).
                                                        .viewDepth = -centreInView.z});

                    continue;
                }

                // The other half of the same split: the view that finishes a layered frame records
                // the blended draws alone. It still walks the opaque primitives above for the side
                // effects a view owes — the upload, the palette, the hooks — and records none.
                if (shading && camera.partition == DrawPartition::BlendedOnly)
                {
                    continue;
                }

                recordDraw(primitive, primitive.material.value(), *material, primitiveShaderId, entityModelMatrix, camera,
                           clipCorrectedViewProjection, joints, paintOffset, colorFormat, depthFormat,
                           shadowDescriptors, staticOnly, true);
                recordedDraws++;
            }
        }

        if (entity.afterDraw.has_value())
        {
            (*entity.afterDraw)();
        }
    }

    // Furthest first. `stable_sort` so that two panels at the same depth — a car's outer and inner
    // glass share a bounding-box centre often enough — keep their authored order rather than
    // swapping between frames as the camera moves and the comparison flips.
    std::stable_sort(blendedDraws.begin(), blendedDraws.end(), [](const DeferredDraw& left, const DeferredDraw& right)
                     { return left.viewDepth > right.viewDepth; });

    // The seam the behind copy exists for: everything opaque is in the attachment and nothing has
    // blended over it yet, so what the copy captures is exactly what every blended draw below is
    // about to be composited onto. Skipped when this view has nothing blended — the copy would
    // record a pause, a blit and a resume that no draw ever reads.
    if (behindCopy != nullptr && !blendedDraws.empty())
    {
        recordSceneBehindCopy(*behindCopy);
    }

    for (const auto& deferred : blendedDraws)
    {
        // Re-resolved rather than carried: the borrow rule says a `find()` pointer is good until
        // its element is removed, and holding one across the whole entity walk is a longer window
        // than this pass needs to ask for. It is one lookup per transparent draw, which is a
        // rounding error against the sort.
        const auto* material = memoryStorageService.materials.find(deferred.materialKey);
        if (material == nullptr)
        {
            diagnostics.record(FrameDiagnostic::PrimitiveWithoutMaterial,
                               [&] { return "material slot " + std::to_string(deferred.materialKey.index); });

            continue;
        }

        // No depth write: these are the blended draws, and the sort above is what orders them.
        recordDraw(*deferred.primitive, deferred.materialKey, *material, deferred.shaderId, deferred.entityModelMatrix,
                   camera, clipCorrectedViewProjection, *deferred.joints, deferred.paintOffset, colorFormat,
                   depthFormat, shadowDescriptors, staticOnly, false);
        recordedDraws++;
    }

    return recordedDraws;
}

void VulkanRenderer::recordSceneBehindCopy(const SceneBehindCopy& copy)
{
    auto& frame = frames[frameIndex];
    const auto colour = imageResources.find(copy.colorImageId);
    const auto behind = imageResources.find(copy.behindImageId);
    if (colour == imageResources.end() || behind == imageResources.end())
    {
        return;
    }

    RACEENGINE_GPU_ZONE(behindCopyZone, frame.commandBuffer, "behind copy");
    const GpuPassLabel behindCopyLabel(*this, frame.commandBuffer, "behind copy");

    // Everything opaque is in the attachment; seal the block so the copy may read it.
    vkCmdEndRendering(frame.commandBuffer);

    transitionTracked(frame.commandBuffer, copy.colorImageId, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    transitionTracked(frame.commandBuffer, copy.behindImageId, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    const auto levelExtent = [](const uint32_t size, const uint32_t mip)
    { return static_cast<int32_t>(std::max(size >> mip, 1u)); };

    // Level 0 is an exact copy — identical extents, so the filter never runs and NEAREST says so.
    VkImageBlit blit{};
    blit.srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = VkOffset3D{levelExtent(colour->second.width, 0), levelExtent(colour->second.height, 0), 1};
    blit.dstSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = VkOffset3D{levelExtent(behind->second.width, 0), levelExtent(behind->second.height, 0), 1};
    vkCmdBlitImage(frame.commandBuffer, colour->second.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   behind->second.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

    // The chain, by blit, exactly as the probe prefilter halves its scratch cube: each level is
    // written as a destination and then read as the next one's source, which is what the per-mip
    // layout tracking exists to say honestly.
    for (auto mip = 1u; mip < behind->second.mipLevels; mip++)
    {
        transitionTrackedLevel(frame.commandBuffer, copy.behindImageId, mip - 1,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkImageBlit level{};
        level.srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 1};
        level.srcOffsets[1] =
            VkOffset3D{levelExtent(behind->second.width, mip - 1), levelExtent(behind->second.height, mip - 1), 1};
        level.dstSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
        level.dstOffsets[1] =
            VkOffset3D{levelExtent(behind->second.width, mip), levelExtent(behind->second.height, mip), 1};
        vkCmdBlitImage(frame.commandBuffer, behind->second.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       behind->second.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &level, VK_FILTER_LINEAR);
    }

    transitionTracked(frame.commandBuffer, copy.behindImageId, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transitionTracked(frame.commandBuffer, copy.colorImageId, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Resume over what the first half drew. Dynamic state — the viewport and scissor set at the
    // top of the pass — is command-buffer state and survives the boundary; pipelines and
    // descriptor sets are bound per draw regardless.
    copy.colorAttachment->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    copy.depthAttachment->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    copy.depthAttachment->storeOp = copy.depthStoreAfterCopy;
    vkCmdBeginRendering(frame.commandBuffer, copy.renderingInfo);
}

unsigned int VulkanRenderer::sceneBehindImage(const unsigned int colorImageId)
{
    if (const auto existing = sceneBehindImages.find(colorImageId); existing != sceneBehindImages.end())
    {
        return existing->second;
    }

    const auto source = imageResources.find(colorImageId);
    if (source == imageResources.end())
    {
        return dummyTexture();
    }

    const auto width = source->second.width;
    const auto height = source->second.height;
    const auto levels = std::clamp(sceneBehindMipCount, 1u, mipLevelCount(width, height));

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = source->second.format;
    imageInfo.extent = VkExtent3D{width, height, 1};
    imageInfo.mipLevels = levels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // Written only ever by blit — level 0 from the colour attachment, the chain from itself — and
    // read only ever by a sampler, so nothing here is a render target.
    imageInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    ensure(vmaCreateImage(allocator, &imageInfo, &allocationCreateInfo, &image, &allocation, nullptr),
           "vmaCreateImage");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = source->second.format;
    viewInfo.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1};

    VkImageView view = VK_NULL_HANDLE;
    ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

    // No sampler of its own: the shared attachment sampler is trilinear with an open LOD clamp,
    // which is exactly what a textureLod over the chain wants.
    const auto id = nextResourceId++;
    imageResources.emplace(id, ImageResource{.image = image,
                                             .allocation = allocation,
                                             .view = view,
                                             .sampler = VK_NULL_HANDLE,
                                             .format = source->second.format,
                                             .mipLevels = levels,
                                             .width = width,
                                             .height = height,
                                             .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                             .layouts = std::vector<VkImageLayout>(levels, VK_IMAGE_LAYOUT_UNDEFINED),
                                             .levelViews = {}});

    sceneBehindImages.emplace(colorImageId, id);

    logger.info("Vulkan scene-behind image {} ready: {}x{}, {} level(s), copying colour attachment {}", id, width,
                height, levels, colorImageId);
    return id;
}

// The material arrives already resolved — both the handle, which keys the descriptor set cache,
// and the element the caller borrowed to decide this draw was worth recording.
void VulkanRenderer::recordDraw(const MeshPrimitive& primitive, const Resource<Material>& materialKey,
                                const Material& material, const unsigned int shaderId,
                                const glm::mat4& entityModelMatrix, const Camera& camera,
                                const glm::mat4& clipCorrectedViewProjection, const std::vector<glm::mat4>& joints,
                                const VkDeviceSize paintOffset, const VkFormat colorFormat, const VkFormat depthFormat,
                                const VkDescriptorSet shadowDescriptors, const bool doubleSided, const bool depthWrite)
{
    const auto bound = primitiveBindings.find(primitive.gpuVao.value());
    if (bound == primitiveBindings.end() || !bound->second.drawable)
    {
        // makePrimitiveBinding already counted *why* the binding is not drawable (a topology or
        // index type this backend has no equivalent for, a buffer that never uploaded); this is
        // the draw that consequence costs.
        diagnostics.record(FrameDiagnostic::PrimitiveNotUploaded,
                           [&] { return "primitive binding " + std::to_string(primitive.gpuVao.value()); });

        return;
    }

    // The material's own doubleSided joins the probe capture's blanket one: a foliage card is
    // authored to be seen from both sides, and an alpha-tested material that got back-face culled
    // would lose half of every tree the moment MASK started counting as opaque.
    const VkCullModeFlags cullMode =
        (material.opaque && !doubleSided && !material.doubleSided) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;

    // A prepass writes geometry into its attachment — a normal and a distance — and neither is a
    // colour with a coverage in front of it. Blending them would read the distance as an opacity.
    const auto blend = camera.role != CameraRole::DepthNormalPrepass;

    // Borrowed, and used before anything else can resolve a pipeline for this primitive: the
    // buffers to feed it are part of the same entry, and nothing between here and the draw
    // resolves one.
    const auto* resolved =
        scenePipeline(shaderId, bound->second, cullMode, colorFormat, depthFormat, blend, depthWrite);
    if (resolved->pipeline == VK_NULL_HANDLE)
    {
        diagnostics.record(FrameDiagnostic::ScenePipelineUnavailable,
                           [&] { return "shader object " + std::to_string(shaderId); });

        return;
    }

    // GL's bindMaterial falls back to the scene environment when the material has none.
    auto environmentImageId = sceneEnvironmentImageId.value_or(0u);
    if (material.environment.has_value())
    {
        if (const auto* environment = memoryStorageService.cubeMaps.find(material.environment.value());
            environment != nullptr)
        {
            environmentImageId = environment->gpuResourceId;
        }
    }
    if (!imageResources.contains(environmentImageId))
    {
        environmentImageId = dummyCubeMap();
    }

    if (shadowDescriptors == VK_NULL_HANDLE)
    {
        // shadowSet/fallbackShadowSet counted the pool exhaustion; this draw is what it costs. The
        // set is not optional: the scene pipelines statically use the samplers in it.
        diagnostics.record(FrameDiagnostic::DescriptorSetUnavailable,
                           [] { return std::string("no shadow cascade set to bind for this view"); });

        return;
    }

    const auto set = materialSet(materialKey, environmentImageId);
    if (set == VK_NULL_HANDLE)
    {
        diagnostics.record(FrameDiagnostic::DescriptorSetUnavailable,
                           [&] { return "material slot " + std::to_string(materialKey.index); });

        return;
    }

    const auto slot = allocateDrawDataSlot();
    if (!slot.has_value())
    {
        // allocateDrawDataSlot counted the exhaustion; this draw is what it costs.
        return;
    }

    auto& frame = frames[frameIndex];
    auto* target = static_cast<unsigned char*>(frame.drawDataMapped) + slot.value();

    // Written out once and reused, rather than stated twice: `*` is left-associative, so this is
    // the same product with the same operands in the same order that both the view matrix slot and
    // the normal matrix below used to compute separately.
    const auto localToView = camera.modelViewMatrix * entityModelMatrix;

    const std::array<glm::mat4, 4> matrices = {
        entityModelMatrix, localToView, clipCorrectedViewProjection * entityModelMatrix,
        // The shader reads mat3(normalMatrix); the mat4 slot carries the upper 3x3.
        glm::mat4(glm::transpose(glm::inverse(glm::mat3(localToView))))};
    std::memcpy(target, matrices.data(), sizeof(matrices));

    auto jointCount = joints.size();
    if (jointCount > maxJoints)
    {
        diagnostics.record(FrameDiagnostic::JointLimitExceeded,
                           [&]
                           {
                               return "Vulkan draw data carries " + std::to_string(maxJoints) + " of the " +
                                      std::to_string(joints.size()) + " supplied";
                           });

        jointCount = maxJoints;
    }

    // Only a skinned draw takes a palette slot; everything else binds offset zero, which is the
    // whole point of the split. A skinned draw the palette ring has no room left for is drawn in
    // its bind pose rather than skipped: allocateJointDataSlot has counted the exhaustion, and a
    // character standing in its rest pose says so far more legibly than a hole in the frame.
    VkDeviceSize jointOffset = 0;
    if (jointCount > 0)
    {
        if (const auto jointSlot = allocateJointDataSlot())
        {
            jointOffset = jointSlot.value();
            std::memcpy(static_cast<unsigned char*>(frame.jointDataMapped) + jointOffset, joints.data(),
                        jointCount * sizeof(glm::mat4));
        }
        else
        {
            jointCount = 0;
        }
    }

    const auto animated = glm::ivec4(jointCount > 0 ? 1 : 0, 0, 0, 0);
    std::memcpy(target + offsetof(DrawDataUbo, animated), &animated, sizeof(animated));

    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, resolved->pipeline);

    // This pipeline's own buffer list, not the primitive's: which buffer lands at which binding
    // index follows the shader's declared locations (see ResolvedPipeline).
    if (!resolved->boundBuffers.empty())
    {
        vkCmdBindVertexBuffers(frame.commandBuffer, 0, static_cast<uint32_t>(resolved->boundBuffers.size()),
                               resolved->boundBuffers.data(), resolved->boundOffsets.data());
    }

    vkCmdBindIndexBuffer(frame.commandBuffer, bound->second.indexBuffer, bound->second.indexOffset,
                         bound->second.indexType);

    const std::array descriptorSets = {frame.frameDataSet, set, frame.drawDataSet, shadowDescriptors};
    // One offset per dynamic descriptor, in set-then-binding order: set 0's view slot, then set 2's
    // draw slot, its palette slot and its paint slot. The count is the *layout's* dynamic
    // descriptors, not the shader's — a stage that declares neither palette nor paint still binds
    // both.
    const std::array dynamicOffsets = {static_cast<uint32_t>(frame.frameDataOffset),
                                       static_cast<uint32_t>(slot.value()), static_cast<uint32_t>(jointOffset),
                                       static_cast<uint32_t>(paintOffset)};
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout, 0,
                            static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
                            static_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    vkCmdDrawIndexed(frame.commandBuffer, bound->second.indexCount, 1, 0, 0, 0);
}

// ---------------------------------------------------------------------------------------------
// Light probes.
//
// A probe is captured the way the frame is rendered — six scene passes from the probe's position,
// with the cascades bound, drawing the sky — and the result is reduced twice: down a GGX roughness
// chain for the specular term, and onto nine spherical harmonic coefficients for the diffuse one.
//
// The reduction is what the shading side actually reads, and it is the whole point. The env term
// this replaces sampled the sky cube directly along the reflection vector, at a fixed LOD, with
// nothing in front of it: no geometry, no shadow, no occlusion. A surface in a building's shadow
// got the same specular highlight as one in the open, because the only thing either sampled was
// the sky. A probe standing in that shadow records the building, so its irradiance and its
// prefiltered radiance are dark — the occlusion is in the data rather than applied to it.
// ---------------------------------------------------------------------------------------------

unsigned int VulkanRenderer::createProbeImage(const uint32_t resolution, const uint32_t mipLevels,
                                              const uint32_t layers, const VkFormat format,
                                              const VkImageUsageFlags usage, const VkImageAspectFlags aspect,
                                              const bool cube) const
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = VkExtent3D{resolution, resolution, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = layers;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    ensure(vmaCreateImage(allocator, &imageInfo, &allocationInfo, &image, &allocation, nullptr), "vmaCreateImage");

    // The *sampling* view: whole mip chain, whole array. Cube or cube array, because that is how
    // both the prefilter shader and the scene shader read these — the per-face 2D views the
    // passes render through are separate objects, created alongside.
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType =
        cube ? (layers > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = VkImageSubresourceRange{aspect, 0, mipLevels, 0, layers};

    VkImageView view = VK_NULL_HANDLE;
    ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

    const auto id = nextResourceId++;
    imageResources.emplace(id,
                           ImageResource{.image = image,
                                         .allocation = allocation,
                                         .view = view,
                                         .sampler = VK_NULL_HANDLE,
                                         .format = format,
                                         .mipLevels = mipLevels,
                                         .width = resolution,
                                         .height = resolution,
                                         .aspect = aspect,
                                         .layouts = std::vector<VkImageLayout>(mipLevels, VK_IMAGE_LAYOUT_UNDEFINED),
                                         .levelViews = {}});
    return id;
}

VkImageView VulkanRenderer::createLevelView(const unsigned int imageId, const uint32_t mip, const uint32_t layer) const
{
    const auto& resource = imageResources.at(imageId);

    // One mip, one layer, seen as a plain 2D image. Rendering into a cube face is exactly this and
    // so is rendering into one level of a chain: dynamic rendering takes an image view, and the
    // view is what says which subresource. A descriptor naming the same view is what makes the
    // layout of that one level the layout the sampler is promised.
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = resource.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = resource.format;
    viewInfo.subresourceRange = VkImageSubresourceRange{resource.aspect, mip, 1, layer, 1};

    VkImageView view = VK_NULL_HANDLE;
    ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");
    return view;
}

VkImageView VulkanRenderer::levelView(const ImageResource& resource, const uint32_t level) const
{
    return level < resource.levelViews.size() ? resource.levelViews[level] : resource.view;
}

void VulkanRenderer::createProbeResources()
{
    constexpr auto radianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr auto depthFormat = VK_FORMAT_D32_SFLOAT;

    // The scratch cube the capture draws into and the prefilter reads from. Its mip chain is
    // generated by blit and exists for one reason: the prefilter picks a source mip from the
    // solid angle each of its samples covers, which is what stops a bright, small feature — a sun
    // disc, a window — from becoming a firefly at high roughness.
    probeRadianceImageId = createProbeImage(probeCubeResolution, probeSpecularMipCount, 6, radianceFormat,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                            VK_IMAGE_ASPECT_COLOR_BIT, true);

    probeDepthImageId = createProbeImage(probeCubeResolution, 1, 1, depthFormat,
                                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, false);

    // TRANSFER_DST is for the initial clear alone: a slice no probe has captured yet is still
    // named by every frame's descriptor, and black is the only value that reads as "no probe here"
    // rather than as whatever the allocator handed over.
    probeSpecularImageId = createProbeImage(
        probeCubeResolution, probeSpecularMipCount, 6 * maxIblProbes, radianceFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, true);

    for (auto face = 0u; face < 6u; face++)
    {
        probeRadianceFaceViews[face] = createLevelView(probeRadianceImageId, 0, face);
    }

    probeSpecularFaceViews.reserve(static_cast<size_t>(maxIblProbes) * probeSpecularMipCount * 6);
    for (auto slice = 0u; slice < maxIblProbes; slice++)
    {
        for (auto mip = 0u; mip < probeSpecularMipCount; mip++)
        {
            for (auto face = 0u; face < 6u; face++)
            {
                probeSpecularFaceViews.push_back(createLevelView(probeSpecularImageId, mip, slice * 6u + face));
            }
        }
    }

    // Trilinear across the roughness chain: the shading side reads a fractional LOD, and a nearest
    // mip filter there is a visible step in the reflection as roughness varies across a surface.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = static_cast<float>(probeSpecularMipCount);
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &probeSampler), "vkCreateSampler");

    // The prefilter's own set layout and pipeline layout: one cube sampler, and a push constant
    // carrying the roughness and the face this invocation is filtering. Push constants rather
    // than a uniform buffer because there are six faces times six mips of them per capture and
    // each is four floats — a ring of thirty-six tiny UBO slots to say that would be absurd.
    const std::array prefilterBindings = {VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                                       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo prefilterSetInfo{};
    prefilterSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    prefilterSetInfo.bindingCount = static_cast<uint32_t>(prefilterBindings.size());
    prefilterSetInfo.pBindings = prefilterBindings.data();
    ensure(vkCreateDescriptorSetLayout(device, &prefilterSetInfo, nullptr, &probePrefilterSetLayout),
           "vkCreateDescriptorSetLayout");

    const VkPushConstantRange pushConstants{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4)};
    VkPipelineLayoutCreateInfo prefilterLayoutInfo{};
    prefilterLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    prefilterLayoutInfo.setLayoutCount = 1;
    prefilterLayoutInfo.pSetLayouts = &probePrefilterSetLayout;
    prefilterLayoutInfo.pushConstantRangeCount = 1;
    prefilterLayoutInfo.pPushConstantRanges = &pushConstants;
    ensure(vkCreatePipelineLayout(device, &prefilterLayoutInfo, nullptr, &probePrefilterPipelineLayout),
           "vkCreatePipelineLayout");

    // The prefilter's shaders are the backend's own, not a game asset: the game authors what the
    // world looks like, and how a probe is reduced to nine coefficients and a roughness chain is
    // no more the game's business than the layout of the draw-data ring is. Compiled through the
    // same shaderc path as everything else, so it gets the same contract macros.
    const auto vertexSpirv = compileToSpirv(probePrefilterVertexSource, shaderc_glsl_vertex_shader, "probe vertex");
    const auto fragmentSpirv =
        compileToSpirv(probePrefilterFragmentSource, shaderc_glsl_fragment_shader, "probe fragment");

    if (!vertexSpirv || !fragmentSpirv)
    {
        // Reported through the diagnostics rather than thrown: an engine that cannot prefilter is
        // an engine whose probes stay dark, which is a degraded picture and not a dead process.
        diagnostics.record(FrameDiagnostic::ProbeCaptureSkipped,
                           [&] { return vertexSpirv ? fragmentSpirv.error() : vertexSpirv.error(); });
        return;
    }

    probePrefilterVertexModule = createShaderModule(vertexSpirv.value());
    probePrefilterFragmentModule = createShaderModule(fragmentSpirv.value());

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = probePrefilterVertexModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = probePrefilterFragmentModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    // No blending: this pass *is* the result, and the frame's global source-alpha blend would
    // multiply a radiance the capture wrote with alpha 1 by an alpha that means nothing here.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    const std::array dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &radianceFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = probePrefilterPipelineLayout;

    ensure(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &probePrefilterPipeline),
           "vkCreateGraphicsPipelines");

    // The set the prefilter reads the scratch cube through. One, for the process: the cube is one
    // image and the prefilter is the only thing that samples it.
    VkDescriptorSetAllocateInfo setAllocateInfo{};
    setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocateInfo.descriptorPool = descriptorPool;
    setAllocateInfo.descriptorSetCount = 1;
    setAllocateInfo.pSetLayouts = &probePrefilterSetLayout;
    ensure(vkAllocateDescriptorSets(device, &setAllocateInfo, &probeRadianceSet), "vkAllocateDescriptorSets");

    const VkDescriptorImageInfo radianceInfo{probeSampler, imageResources.at(probeRadianceImageId).view,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet radianceWrite{};
    radianceWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    radianceWrite.dstSet = probeRadianceSet;
    radianceWrite.dstBinding = 0;
    radianceWrite.descriptorCount = 1;
    radianceWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    radianceWrite.pImageInfo = &radianceInfo;
    vkUpdateDescriptorSets(device, 1, &radianceWrite, 0, nullptr);

    // The probe array's descriptor, into every frame's set 0. Written once, here, and never
    // again: the image is created at bring-up at a size the contract fixes, so a probe capturing
    // rewrites its *contents* and never the descriptor that names it. That is the whole reason
    // this is one array rather than a cube per probe — a time-of-day change re-captures every
    // probe in the scene, and a descriptor rewrite per probe per change is what that would cost.
    const VkDescriptorImageInfo specularInfo{probeSampler, imageResources.at(probeSpecularImageId).view,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    for (const auto& frame : frames)
    {
        VkWriteDescriptorSet specularWrite{};
        specularWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        specularWrite.dstSet = frame.frameDataSet;
        specularWrite.dstBinding = probeSpecularBinding;
        specularWrite.descriptorCount = 1;
        specularWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        specularWrite.pImageInfo = &specularInfo;
        vkUpdateDescriptorSets(device, 1, &specularWrite, 0, nullptr);
    }

    // The staging buffer the irradiance is projected from. One mip's worth of one cube: at the
    // mip the contract names that is 16x16x6 texels, which is 12 KiB and is read once per capture.
    const auto readbackExtent = probeCubeResolution >> probeIrradianceSourceMip;
    const auto readbackBytes = static_cast<VkDeviceSize>(readbackExtent) * readbackExtent * 6 * 4 * sizeof(uint16_t);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = readbackBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo readbackAllocationInfo{};
    readbackAllocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    readbackAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VmaAllocationInfo mappedInfo{};
    ensure(vmaCreateBuffer(allocator, &bufferInfo, &readbackAllocationInfo, &probeReadbackBuffer.buffer,
                           &probeReadbackBuffer.allocation, &mappedInfo),
           "vmaCreateBuffer");
    probeReadbackMapped = mappedInfo.pMappedData;

    // Both images start black and readable. The specular array is named by every frame's set 0
    // from the first frame, and a scene shading through it before any probe has captured would be
    // sampling an image in an undefined layout — the shader's probe count says not to read it,
    // and "the pipeline statically uses this descriptor" does not care what the shader decides.
    const auto commandBuffer = beginUploadCommands();
    const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
    for (const auto imageId : {probeRadianceImageId, probeSpecularImageId})
    {
        transitionTracked(commandBuffer, imageId, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const auto& resource = imageResources.at(imageId);
        const VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                            VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearColorImage(commandBuffer, resource.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &whole);
        transitionTracked(commandBuffer, imageId, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    finishUploadCommands(commandBuffer);

    logger.info("Vulkan light probe machinery ready: {} slices of a {}x{} cube array, {} roughness levels, "
                "irradiance projected from mip {} ({}x{})",
                maxIblProbes, probeCubeResolution, probeCubeResolution, probeSpecularMipCount, probeIrradianceSourceMip,
                readbackExtent, readbackExtent);
}

void VulkanRenderer::createExposureResources()
{
    // One RGBA16F texel. The meter writes its reduction into the red channel and leaves the rest,
    // but a buffer image copy transfers whole texels, so the buffer is the texel's width.
    constexpr VkDeviceSize luminanceReadbackBytes = 4 * sizeof(uint16_t);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = luminanceReadbackBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VmaAllocationInfo mappedInfo{};
    ensure(vmaCreateBuffer(allocator, &bufferInfo, &allocationInfo, &luminanceReadbackBuffer.buffer,
                           &luminanceReadbackBuffer.allocation, &mappedInfo),
           "vmaCreateBuffer");
    luminanceReadbackMapped = mappedInfo.pMappedData;
}

// The image a shading view samples its occlusion from. A camera that gathers none — or one whose
// buffer has gone out from under it — reads the 1x1 white image, which is a visibility term of one
// everywhere and therefore no occlusion at all.
unsigned int VulkanRenderer::ambientOcclusionImage(const Camera& camera)
{
    if (!camera.ambientOcclusion.enabled || !camera.ambientOcclusion.occlusion.issued())
    {
        return dummyTexture();
    }

    const auto* attachment = memoryStorageService.bufferAttachments.find(camera.ambientOcclusion.occlusion);
    if (attachment == nullptr || !attachment->gpuResourceId.has_value() ||
        !imageResources.contains(attachment->gpuResourceId.value()))
    {
        diagnostics.record(FrameDiagnostic::AmbientOcclusionSkipped,
                           [] { return std::string("the occlusion attachment is no longer loaded"); });

        return dummyTexture();
    }

    return attachment->gpuResourceId.value();
}

unsigned int VulkanRenderer::cloudMapImage(const Scene& scene)
{
    if (!scene.cloudMap.has_value())
    {
        return dummyTexture();
    }

    const auto* attachment = memoryStorageService.bufferAttachments.find(scene.cloudMap.value());
    if (attachment == nullptr || !attachment->gpuResourceId.has_value() ||
        !imageResources.contains(attachment->gpuResourceId.value()))
    {
        diagnostics.record(FrameDiagnostic::CloudMapUnavailable,
                           [] { return std::string("the scene's cloud map attachment is no longer loaded"); });

        return dummyTexture();
    }

    return attachment->gpuResourceId.value();
}

// The view drawn a second time, for what its own geometry hides.
//
// A Camera on the stack rather than one the game keeps: every field of it except the target and the
// shader belongs to the camera this is a prepass for, and a second Camera a game had to hold in step
// would be a second answer to where the view is. What differs is exactly what the copy overwrites —
// it renders the prepass buffer, with the prepass shader, under a role that says it does not shade,
// and carries the gather and the blur as its post chain so recordScenePass runs them in order the
// moment the geometry is down.
void VulkanRenderer::recordAmbientOcclusion(Scene& scene, Camera& camera, const float delta)
{
    const auto& occlusion = camera.ambientOcclusion;

    if (!frameOpen || !occlusion.enabled || !occlusion.prepass.has_value() || !occlusion.prepassShader.has_value())
    {
        return;
    }

    Camera prepass = camera;
    prepass.role = CameraRole::DepthNormalPrepass;
    prepass.debugName = camera.debugName.empty() ? std::string("ao prepass") : camera.debugName + " prepass";
    prepass.output = occlusion.prepass.value();
    prepass.overrideShader = occlusion.prepassShader.value();
    // The gather and the blur, and not the tone map or the meter's reduction: those belong to the
    // view that shades, and this one has not shaded anything. The occlusion settings stay on the
    // copy because the gather reads its radius and strength out of them.
    prepass.postProcesses = occlusion.passes;
    // **Every layer, whatever this camera's own mask says.** The prepass records the geometry the
    // frame is made of so the gather can measure how much sky each pixel sees, and under a layered
    // frame one gather serves every layer's shading — so a prepass that drew only its own camera's
    // layer would erase the car from under its own contact shadow: the darkening under a tyre is
    // exactly the pixel where the two layers meet. This is the same class of decision as the
    // castsShadow guard in recordSceneDraws, and it is made here for the same reason it was made
    // there: the prepass is a statement about what exists, not about what this view shades. The
    // continuity fields reset with it — a prepass opens its own target and keeps nothing.
    prepass.layerMask = ~0u;
    prepass.partition = DrawPartition::All;
    prepass.loadColour = false;
    prepass.loadDepth = false;
    // The one continuity field the prepass does not reset to false: `shareDepth` says the views
    // recorded after this one load the depth it is about to write and test against it, which is
    // pre-Z out of geometry the frame has already rasterised. The game states it because the game
    // is what composed a shading camera over this buffer; the prepass camera itself is a copy
    // nobody outside this function holds.
    prepass.keepDepth = occlusion.shareDepth;
    prepass.clearColour.reset();

    recordScenePass(scene, prepass, delta);
}

// The meter's half of the deferred-readback pattern the probe path established: collect the copy
// the last call queued if the submission carrying it has completed, then queue the next one. The
// camera keeps the previous reading in between, so nothing here ever stalls the device and nothing
// downstream ever sees an unwritten buffer.
//
// The lag is `framesInFlight + 1` submissions and it is a lag in *submissions*, not in wall time —
// which is what makes it survive the frame gate: the schedule is a function of the frame number in
// exactly the way the light probe scheduler's is.
void VulkanRenderer::recordAutoExposure(Camera& camera)
{
    auto& meter = camera.autoExposure;

    if (!frameOpen || !meter.enabled || !meter.chain.issued() || luminanceReadbackMapped == nullptr)
    {
        return;
    }

    const auto* attachment = memoryStorageService.bufferAttachments.find(meter.chain);
    if (attachment == nullptr || !attachment->gpuResourceId.has_value())
    {
        diagnostics.record(FrameDiagnostic::ExposureMeterSkipped,
                           [] { return std::string("the chain attachment is no longer loaded"); });

        return;
    }

    const auto imageId = attachment->gpuResourceId.value();
    const auto image = imageResources.find(imageId);
    if (image == imageResources.end())
    {
        diagnostics.record(FrameDiagnostic::ExposureMeterSkipped,
                           [&] { return "chain attachment " + std::to_string(imageId) + " has no image"; });

        return;
    }

    if (luminanceReadbackOwner == imageId && luminanceReadbackReadyAt <= submittedFrames)
    {
        ensure(vmaInvalidateAllocation(allocator, luminanceReadbackBuffer.allocation, 0, VK_WHOLE_SIZE),
               "vmaInvalidateAllocation");

        // Two channels: the weighted sum of log2 luminance, and the weight it was summed with. The
        // ratio is the weighted mean of the logarithm, and undoing the logarithm on *that* is what
        // makes the reading the weighted geometric mean of the frame — the average an exposure
        // meter takes, and the one a handful of specular pinpricks cannot drag the whole picture
        // off. Both channels came down the same tree of averages, so the division is the whole of
        // what centre weighting costs on this side.
        //
        // A zero weight is a frame the reduction never wrote — the pass was skipped, or the chain
        // is a frame behind a resize — and holding the previous reading is what every other
        // not-yet-arrived case here does.
        const auto* halves = static_cast<const uint16_t*>(luminanceReadbackMapped);
        const auto weightedLogarithm = halfToFloat(halves[0]);
        const auto weight = halfToFloat(halves[1]);

        if (weight > 0.0f)
        {
            meter.measuredLuminance = std::exp2(weightedLogarithm / weight);
        }
        luminanceReadbackOwner.reset();
    }

    // Another camera's copy is still in flight. It will be collected on the frame it is ready, and
    // this camera queues its own on the one after — a fixed rotation, not a race.
    if (luminanceReadbackOwner.has_value())
    {
        return;
    }

    const auto vulkanImage = image->second.image;
    const auto level = std::min(meter.chainLevel, image->second.mipLevels - 1u);
    const auto& frame = frames[frameIndex];

    transitionTrackedLevel(frame.commandBuffer, imageId, level, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy readback{};
    readback.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
    readback.imageExtent = VkExtent3D{1, 1, 1};
    vkCmdCopyImageToBuffer(frame.commandBuffer, vulkanImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           luminanceReadbackBuffer.buffer, 1, &readback);

    // Back to where the chain's passes expect to find it, so the next frame's barrier is the one
    // the post-process path already emits rather than one derived from a transfer layout.
    transitionTrackedLevel(frame.commandBuffer, imageId, level, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Same arithmetic as retire(): the copy is a command in a frame that has not been submitted, so
    // the buffer holds this reading only once that submission has completed.
    luminanceReadbackOwner = imageId;
    luminanceReadbackReadyAt = submittedFrames + framesInFlight + 1;
}

void VulkanRenderer::recordProbeCapture(Scene& scene, LightProbe& probe)
{
    if (!frameOpen || probePrefilterPipeline == VK_NULL_HANDLE)
    {
        diagnostics.record(FrameDiagnostic::ProbeCaptureSkipped,
                           [] { return std::string("no open frame or no prefilter pipeline"); });

        // Ready with no slice contributes nothing and stops the scheduler retrying every frame;
        // the diagnostic is what says the probe is dark rather than merely uninteresting.
        probe.state = LightProbeState::Ready;
        return;
    }

    if (!probe.arraySlice.has_value())
    {
        if (probeSlicesUsed >= maxIblProbes)
        {
            diagnostics.record(FrameDiagnostic::ProbeLimitExceeded, [&] { return "probe '" + probe.name + "'"; });

            probe.state = LightProbeState::Ready;
            return;
        }

        probe.arraySlice = probeSlicesUsed++;
    }

    // Waiting on the readback that the last call queued. It is the *only* thing this call does,
    // because there is nothing else this probe needs: the specular chain is already in the array
    // and the frame is already shading from it. What is missing is the diffuse half.
    if (probe.state == LightProbeState::Projecting)
    {
        if (probeReadbackReadyAt > submittedFrames)
        {
            return;
        }

        probeIrradiance[probe.arraySlice.value()] = projectProbeIrradiance();
        probe.irradiance = probeIrradiance[probe.arraySlice.value()];
        probe.state = LightProbeState::Ready;

        // Once per probe rather than once per process: a probe's mean irradiance is the one number
        // that says the capture saw anything at all, and a scene where one probe is dark and the
        // rest are not is the case worth being able to read straight off the log. Re-captures are
        // silent — a time-of-day change would otherwise print the whole graph every few seconds.
        if (!probe.captureLogged)
        {
            // Upwards, because that is the direction most of a scene's surfaces face and so the
            // one whose value can be read straight off against what the sky looks like.
            const auto upward = evaluateShIrradiance(probe.irradiance, glm::vec3(0.0f, 1.0f, 0.0f));
            logger.info("Light probe '{}' captured into slice {}: six faces, {} roughness levels prefiltered, "
                        "irradiance facing up {:.3f} {:.3f} {:.3f}",
                        probe.name, probe.arraySlice.value(), probeSpecularMipCount, upward.r, upward.g, upward.b);
            probe.captureLogged = true;
        }

        return;
    }

    if (probe.captureFace < 6)
    {
        recordProbeFace(scene, probe, probe.captureFace, 0.0f);
        probe.captureFace++;
        probe.state = LightProbeState::Capturing;
        return;
    }

    if (!recordProbePrefilter(probe))
    {
        probe.state = LightProbeState::Ready;
        return;
    }

    // Same arithmetic as retire(): the copy is a command in a frame that has not been submitted
    // yet, so the buffer holds this capture's radiance only once that submission has completed.
    probeReadbackReadyAt = submittedFrames + framesInFlight + 1;
    probe.captureFace = 0;
    probe.state = LightProbeState::Projecting;
}

void VulkanRenderer::recordProbeFace(Scene& scene, const LightProbe& probe, const unsigned int face, const float delta)
{
    auto& frame = frames[frameIndex];

    // The six views, and the one place their handedness is decided.
    //
    // A cube map's faces are stored left-handed relative to the world — the legacy every graphics
    // API carries — so a face rendered with the scene pass's usual negative-viewport flip comes
    // out mirrored. The flip is dropped here instead: with Vulkan's native downward viewport, the
    // texel row a face stores first is the row clip y = -1 rasterises to, and these six
    // right-handed bases put the correct direction under it. The up vectors look wrong and are
    // not; they are what makes cross(forward, up) point where the face's +s axis has to.
    static constexpr std::array<glm::vec3, 6> faceForward = {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f),
                                                             glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f),
                                                             glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
    static constexpr std::array<glm::vec3, 6> faceUp = {glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f),
                                                        glm::vec3(0.0f, 0.0f, 1.0f),  glm::vec3(0.0f, 0.0f, -1.0f),
                                                        glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)};

    // A camera, on the stack, because that is all a view is to everything below: recordDraw reads
    // two matrices off it and the frame block reads a position. Building one here rather than
    // putting six per probe into Scene::cameras keeps them out of every walk over that container
    // — and out of a game's reach, which is right, since nothing about a cube face is authored.
    Camera camera{};
    camera.role = CameraRole::ProbeFace;
    camera.position = probe.position;
    camera.nearClippingPlane = probe.nearClippingPlane;
    camera.farClippingPlane = probe.farClippingPlane;
    camera.tracksWindowSize = false;
    camera.modelViewMatrix = glm::lookAt(probe.position, probe.position + faceForward[face], faceUp[face]);
    // Ninety degrees, square: six of them tile the sphere exactly, which is what a cube map is.
    camera.modelViewProjectionMatrix =
        glm::perspective(glm::radians(90.0f), 1.0f, probe.nearClippingPlane, probe.farClippingPlane) *
        camera.modelViewMatrix;

    FrameDataUbo frameData{};
    frameData.viewMatrix = camera.modelViewMatrix;
    frameData.cameraPosition = glm::vec4(camera.position, 1.0f);

    auto uploadedLights = 0u;
    for (const auto& light : scene.lights)
    {
        if (uploadedLights >= maxLights)
        {
            break;
        }

        frameData.lights[uploadedLights] =
            LightUbo{glm::vec4(light.position, 1.0f), glm::vec4(light.diffuse, 0.0f), glm::vec4(light.specular, 0.0f),
                     glm::vec4(light.ambient, light.attenuation)};
        uploadedLights++;
    }
    frameData.lightCount = glm::ivec4(static_cast<int>(uploadedLights), 0, 0, 0);

    // The clouds, alone among the weather, are kept in a capture: the fog and the rain are effects
    // between a surface and the eye, which a photograph must not bake in, but the clouded sky *is*
    // the thing being photographed — the captures are how clouds become the world's ambient light.
    frameData.cloudParams = glm::vec4(scene.clouds.coverage, scene.clouds.type, 0.0f, 0.0f);

    // No `uploadFog` here, and the omission is the rule rather than an oversight. A probe photographs
    // the world so that a surface can be given the light it cannot see directly; fog baked into that
    // photograph comes back as the surface's ambient light, and the surface then fogs a second time
    // on the way to the eye. It is the same double count the solar disc is kept out of a capture for,
    // and it is settled here — by uploading nothing — rather than as a flag each shader has to
    // remember to test. The fog's own ambient colour is read *from* these probes, so a capture that
    // contained it would also be feeding itself.

    // The capture is shaded, cascades and all: a probe that recorded the world unshadowed would
    // put the sun's full radiance into the very shadow the direct term just removed, which is the
    // defect this whole path exists to fix, reintroduced one level down.
    // White for the occlusion beside them: a probe records the world's indirect light, and screen
    // space has no answer for a face pointing somewhere the screen never looked. The cloud map is
    // the scene's real one, not the dummy — the skybox this face draws composites it, and a probe
    // that photographed a clear sky under a clouded one would light the world for the wrong day.
    const auto cascadeImages = shadowCascadeImages(scene);
    auto shadowDescriptors = cascadeImages.has_value() ? shadowSet(cascadeImages.value(), dummyTexture(),
                                                                   dummyTexture(), cloudMapImage(scene))
                                                       : VK_NULL_HANDLE;

    if (shadowDescriptors != VK_NULL_HANDLE)
    {
        const auto correction = shadowLookupCorrection();

        for (auto cascade = 0u; cascade < shadowCascadeCount; cascade++)
        {
            const auto& slice = scene.shadows.cascades[cascade];
            const auto index = static_cast<int>(cascade);
            frameData.shadowMatrices[cascade] = correction * slice.camera->modelViewProjectionMatrix;
            frameData.shadowSplits[index] = slice.splitDistance;
            frameData.shadowTexelWorldSize[index] = slice.texelWorldSize;
            frameData.shadowDepthScale[index] = slice.depthPerWorldUnit;
        }

        frameData.shadowParams =
            glm::ivec4(static_cast<int>(shadowCascadeCount), static_cast<int>(scene.shadows.lightIndex), 0, 0);
    }
    else
    {
        shadowDescriptors = fallbackShadowSet();
    }

    // Set after the branch above and not inside it, because it is a fact about this view rather
    // than about its cascades: a capture with no shadow maps still must not photograph the sun.
    //
    // What it turns off is the solar disc in the sky (see SkyboxFragmentShader). The disc is the
    // scene light stated as a radiance — the same sun the direct term already delivers — so a
    // capture that keeps it hands the surface its sun twice, and the aureole around it stays
    // because that glow is scattered light and genuinely belongs to the sky. The double count is
    // small in the mean and badly behaved in the tail: a face is 128 pixels across ninety degrees,
    // so a texel is 0.70° against the disc's 0.53°, and whether the sun lands in a texel centre is
    // a lottery that a capture would re-run every time the light moved.
    frameData.shadowParams.z = 1;

    // probeParams stays zero. A capture must not shade from probes — including from itself: a
    // probe that read the array it is about to write would feed its own previous answer back in
    // every capture, and the scene would brighten without bound.

    const auto frameDataSlot = allocateFrameDataSlot();
    if (!frameDataSlot.has_value())
    {
        return;
    }

    frame.frameDataOffset = frameDataSlot.value();
    std::memcpy(static_cast<char*>(frame.frameDataMapped) + frame.frameDataOffset, &frameData, sizeof(frameData));

    transitionTracked(frame.commandBuffer, probeRadianceImageId, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transitionTracked(frame.commandBuffer, probeDepthImageId, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    const VkExtent2D extent{probeCubeResolution, probeCubeResolution};

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = probeRadianceFaceViews[face];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color =
        VkClearColorValue{{clearColour[0], clearColour[1], clearColour[2], clearColour[3]}};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = imageResources.at(probeDepthImageId).view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = VkClearDepthStencilValue{1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, extent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);

    // Positive height, unlike every other pass this backend records — see the basis table above.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);

    const VkRect2D scissor{VkOffset2D{0, 0}, extent};
    vkCmdSetScissor(frame.commandBuffer, 0, 1, &scissor);

    static_cast<void>(recordSceneDraws(scene, camera, delta, true, true, 0,
                                       imageResources.at(probeRadianceImageId).format,
                                       imageResources.at(probeDepthImageId).format, shadowDescriptors, nullptr));

    vkCmdEndRendering(frame.commandBuffer);
}

bool VulkanRenderer::recordProbePrefilter(const LightProbe& probe)
{
    auto& frame = frames[frameIndex];
    const auto slice = probe.arraySlice.value();

    // The scratch cube's mip chain, by blit. The prefilter picks a source mip from the solid
    // angle each of its samples covers: without the chain every sample reads the top mip, and a
    // small bright feature — the sun, a window — survives 64 samples as a firefly instead of
    // spreading into the lobe that was supposed to blur it.
    //
    // The layout bookkeeping below is per *mip*. Each level is written as a blit destination and
    // then read as the next one's source, so mid-chain the image holds two layouts at once and one
    // tracked layout for the whole image would be a lie about half of it. This used to be done by
    // hand here, because it was the only place it happened; the tracking is per level now — see
    // ImageResource — so the chain says what it is doing through the same call everything else
    // does, and the fix-up that used to reunify the image afterwards is gone with it.
    const auto radianceImage = imageResources.at(probeRadianceImageId).image;

    transitionTracked(frame.commandBuffer, probeRadianceImageId, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    for (auto mip = 1u; mip < probeSpecularMipCount; mip++)
    {
        const auto sourceExtent = static_cast<int32_t>(probeCubeResolution >> (mip - 1));
        const auto targetExtent = static_cast<int32_t>(probeCubeResolution >> mip);

        transitionTrackedLevel(frame.commandBuffer, probeRadianceImageId, mip - 1,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkImageBlit blit{};
        blit.srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 6};
        blit.srcOffsets[1] = VkOffset3D{sourceExtent, sourceExtent, 1};
        blit.dstSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 6};
        blit.dstOffsets[1] = VkOffset3D{targetExtent, targetExtent, 1};

        vkCmdBlitImage(frame.commandBuffer, radianceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, radianceImage,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    }

    // The last level was written and never read, so it is the one still in the destination layout.
    // Moving it brings the whole image back to one layout, which is what makes the whole-image
    // transitions below a single barrier again.
    transitionTrackedLevel(frame.commandBuffer, probeRadianceImageId, probeSpecularMipCount - 1,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // The irradiance projection's source: one small mip of one cube, which the CPU projects onto
    // the harmonic basis a couple of frames from now. Its read is ordered against the blit that
    // wrote it by that level's own destination-to-source barrier above — which is why the mip has
    // to be one the chain reads from rather than the last one it writes.
    static_assert(probeIrradianceSourceMip + 1 < probeSpecularMipCount);
    const auto readbackExtent = probeCubeResolution >> probeIrradianceSourceMip;

    VkBufferImageCopy readback{};
    readback.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, probeIrradianceSourceMip, 0, 6};
    readback.imageExtent = VkExtent3D{readbackExtent, readbackExtent, 1};
    vkCmdCopyImageToBuffer(frame.commandBuffer, radianceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           probeReadbackBuffer.buffer, 1, &readback);

    transitionTracked(frame.commandBuffer, probeRadianceImageId, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transitionTracked(frame.commandBuffer, probeSpecularImageId, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, probePrefilterPipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, probePrefilterPipelineLayout, 0, 1,
                            &probeRadianceSet, 0, nullptr);

    for (auto mip = 0u; mip < probeSpecularMipCount; mip++)
    {
        const auto extent = probeCubeResolution >> mip;
        // Mip 0 is the mirror: roughness zero, which the shader's importance sampling collapses to
        // a single fetch along the reflection vector. The chain then walks to fully rough at the
        // last level, and the shading side inverts exactly this mapping to pick its LOD.
        const auto roughness = static_cast<float>(mip) / static_cast<float>(probeSpecularMipCount - 1);

        for (auto face = 0u; face < 6u; face++)
        {
            const auto viewIndex = (static_cast<size_t>(slice) * probeSpecularMipCount + mip) * 6u + face;

            VkRenderingAttachmentInfo colorAttachment{};
            colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAttachment.imageView = probeSpecularFaceViews[viewIndex];
            colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, VkExtent2D{extent, extent}};
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &colorAttachment;

            vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);

            // Positive height for the same reason the capture uses one: the shader derives its
            // direction from gl_FragCoord, so framebuffer row 0 has to be the face's first texel
            // row exactly as it was when the capture wrote it.
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(extent);
            viewport.height = static_cast<float>(extent);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);

            const VkRect2D scissor{VkOffset2D{0, 0}, VkExtent2D{extent, extent}};
            vkCmdSetScissor(frame.commandBuffer, 0, 1, &scissor);

            const auto pushConstants = glm::vec4(roughness, static_cast<float>(face), static_cast<float>(extent),
                                                 static_cast<float>(probeCubeResolution));
            vkCmdPushConstants(frame.commandBuffer, probePrefilterPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(pushConstants), &pushConstants);

            vkCmdDraw(frame.commandBuffer, 3, 1, 0, 0);
            vkCmdEndRendering(frame.commandBuffer);
        }
    }

    transitionTracked(frame.commandBuffer, probeSpecularImageId, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return true;
}

ShIrradiance VulkanRenderer::projectProbeIrradiance() const
{
    const auto extent = probeCubeResolution >> probeIrradianceSourceMip;
    const auto texelsPerFace = static_cast<size_t>(extent) * extent;

    if (probeReadbackMapped == nullptr)
    {
        return ShIrradiance{};
    }

    // R16G16B16A16_SFLOAT, tightly packed by vkCmdCopyImageToBuffer: face-major, four halves per
    // texel. Decoded rather than blitted into a float image because the decode is exact, testable
    // and about twenty lines, where the blit would be another image and another barrier.
    const auto* halves = static_cast<const uint16_t*>(probeReadbackMapped);

    std::array<std::vector<glm::vec3>, 6> storage;
    std::array<std::span<const glm::vec3>, 6> faces{};

    for (auto face = 0u; face < 6u; face++)
    {
        storage[face].reserve(texelsPerFace);

        for (size_t texel = 0; texel < texelsPerFace; texel++)
        {
            const auto base = (static_cast<size_t>(face) * texelsPerFace + texel) * 4;
            storage[face].emplace_back(halfToFloat(halves[base]), halfToFloat(halves[base + 1]),
                                       halfToFloat(halves[base + 2]));
        }

        faces[face] = std::span<const glm::vec3>(storage[face]);
    }

    return projectCubeToIrradiance(std::span<const std::span<const glm::vec3>, 6>(faces), extent);
}

void VulkanRenderer::uploadProbes(const Scene& scene, FrameDataUbo& frameData) const
{
    auto uploaded = 0u;

    for (const auto& probe : scene.probes)
    {
        // A probe with no slice never captured — the array was full — and one that has not
        // reached Ready holds either nothing or a partial environment. Neither may shade: a
        // half-captured probe lights the world from whichever faces happened to finish.
        if (uploaded >= maxIblProbes || !probe.arraySlice.has_value() || probe.state == LightProbeState::Capturing)
        {
            continue;
        }

        auto& target = frameData.probes[uploaded];
        target.irradiance = probe.irradiance;
        target.boxMin = glm::vec4(probe.position - probe.halfExtents, probe.blendDistance);
        target.boxMax = glm::vec4(probe.position + probe.halfExtents, probe.global ? 1.0f : 0.0f);
        target.position = glm::vec4(probe.position, static_cast<float>(probe.arraySlice.value()));
        uploaded++;
    }

    frameData.probeParams = glm::ivec4(static_cast<int>(uploaded), 0, 0, 0);
}

bool VulkanRenderer::recordFullScreenPass(const std::span<const PostProcessBinding> inputs,
                                          const unsigned int lookupTableImageId,
                                          const std::array<unsigned int, postProcessVolumeCount>& volumeImageIds,
                                          const VkImageView targetView, const VkExtent2D targetExtent,
                                          const VkPipeline pipeline, const FullscreenPushConstants& parameters,
                                          const VkDescriptorSet shadowDescriptors, const bool loadColour,
                                          const float blendWeight, const unsigned int slice,
                                          const unsigned int sliceCount)
{
    const auto set = attachmentSet(inputs, lookupTableImageId, volumeImageIds);
    if (set == VK_NULL_HANDLE)
    {
        // attachmentSet counted the pool exhaustion if that is what it was; an unknown image id
        // is the caller's, and both leave the target untouched.
        return false;
    }

    auto& frame = frames[frameIndex];

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = targetView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // Nothing to load, because nothing survives: the oversized triangle covers every pixel of the
    // render area, so whatever a clear wrote would be overwritten before it could be read. This was
    // a CLEAR because GL's drawFullScreenQuad cleared first, and the comment that stood here
    // conceded the point — "the quad covers the target anyway". Twenty-five of these run per frame.
    //
    // The one thing that makes the old contents observable is the blend, which is
    // SRC_ALPHA/ONE_MINUS_SRC_ALPHA on this layout: a pass writing alpha below one reads the
    // destination. Every fullscreen shader *but the cloud dome* writes 1.0, and the R8 targets have
    // no alpha component at all — which Vulkan reads as 1.0 — so for those the destination factor
    // is zero everywhere and the load is genuinely dead.
    //
    // The dome writes its transmittance there, so for years it blended against memory Vulkan says
    // is undefined and got away with it on a driver that leaves the image alone (2026-08-27; it is
    // the leading suspect for the recorded run-to-run sky flake). `loadColour` is that pass — and
    // any later one that reads its destination on purpose — saying so, which is the whole
    // difference between an accident and a contract.
    colorAttachment.loadOp = loadColour ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    // The strip this recording writes, in whole texels and tiling exactly: the boundaries are
    // `count * s / n`, so a width the count does not divide leaves no column unwritten and no
    // column written twice. A count of one is the whole target and is every pass but the sliced
    // one. A degenerate strip — a target narrower than its own slice count — is widened to a
    // single column rather than refused, because a render area of zero width is not a legal pass.
    const auto strips = std::max(sliceCount, 1u);
    const auto index = std::min(slice, strips - 1u);
    const auto left = static_cast<uint32_t>((static_cast<uint64_t>(targetExtent.width) * index) / strips);
    const auto right = static_cast<uint32_t>((static_cast<uint64_t>(targetExtent.width) * (index + 1u)) / strips);
    const VkRect2D strip{VkOffset2D{static_cast<int32_t>(left), 0},
                         VkExtent2D{std::max(right - left, 1u), targetExtent.height}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = strip;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);

    // Positive height here: the fullscreen shaders derive their texture coordinates from
    // gl_VertexIndex in Vulkan's own y-down clip space, so the scene attachment (already
    // stored the GL way up) maps through one-to-one.
    //
    // **The whole target, even when only a strip of it is being written.** The viewport is what
    // maps clip space onto the framebuffer, so shrinking it would squeeze the oversized triangle
    // into the strip and hand every fragment the coordinates of a different texel. The strip is the
    // render area and the scissor, which discard fragments outside it and leave what those texels
    // held alone — which is exactly the statement a sliced pass is making.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(targetExtent.width);
    viewport.height = static_cast<float>(targetExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);

    vkCmdSetScissor(frame.commandBuffer, 0, 1, &strip);

    // Only a FullscreenBlend::Constant pipeline reads these, and every fullscreen pipeline declares
    // them dynamic, so setting them unconditionally costs one command and removes a way to forget.
    // The alpha component is the one the constant factors name; the other three are written to
    // match so a future constant-colour factor cannot pick up a stale number.
    const std::array blendConstants = {blendWeight, blendWeight, blendWeight, blendWeight};
    vkCmdSetBlendConstants(frame.commandBuffer, blendConstants.data());

    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // All three of the layout's sets, every pass: the frame block at the view's own dynamic offset
    // — recordScenePass wrote it before this chain began, so a chain pass reads the view it rides —
    // the pass's inputs, and the shadow set (the caller's, or the fallback whose empty cascades
    // shade lit). A shader that declares only its inputs sees only its inputs; the binding is the
    // layout's to satisfy.
    const auto shadowSet = shadowDescriptors != VK_NULL_HANDLE ? shadowDescriptors : fallbackShadowSet();
    const std::array descriptorSets = {frame.frameDataSet, set, shadowSet};
    const auto frameOffset = static_cast<uint32_t>(frame.frameDataOffset);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fullscreenPipelineLayout, 0,
                            static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 1, &frameOffset);

    // Pushed for every fullscreen pass, read by the ones that tone map or walk a chain. A shader
    // that does not declare the block simply does not see it; the range belongs to the layout, not
    // the pipeline.
    vkCmdPushConstants(frame.commandBuffer, fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(parameters), &parameters);

    vkCmdDraw(frame.commandBuffer, 3, 1, 0, 0);

    vkCmdEndRendering(frame.commandBuffer);
    return true;
}

VulkanRenderer::BufferResource VulkanRenderer::createDeviceLocalBuffer(const void* data, const VkDeviceSize size) const
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    // glTF buffer views carry an optional target and GLTFService infers the rest, so every
    // uploaded buffer is tagged for both roles rather than trusting the inference twice.
    bufferInfo.usage =
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    BufferResource resource;
    ensure(
        vmaCreateBuffer(allocator, &bufferInfo, &allocationCreateInfo, &resource.buffer, &resource.allocation, nullptr),
        "vmaCreateBuffer");

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = size;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAllocationCreateInfo{};
    stagingAllocationCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = nullptr;
    VmaAllocationInfo stagingAllocationInfo{};
    ensure(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocationCreateInfo, &stagingBuffer, &stagingAllocation,
                           &stagingAllocationInfo),
           "vmaCreateBuffer");

    std::memset(stagingAllocationInfo.pMappedData, 0, static_cast<size_t>(size));
    if (data != nullptr)
    {
        std::memcpy(stagingAllocationInfo.pMappedData, data, static_cast<size_t>(size));
    }
    ensure(vmaFlushAllocation(allocator, stagingAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

    const auto commandBuffer = beginUploadCommands();

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, resource.buffer, 1, &region);

    // The fence in finishUploadCommands orders execution; the barrier is what makes the
    // copy visible to the vertex input stage of every later submission.
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = resource.buffer;
    barrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    finishUploadCommands(commandBuffer);
    retireUploadStaging(stagingBuffer, stagingAllocation);

    return resource;
}

unsigned int VulkanRenderer::uploadMeshBuffer(const MeshBuffer& meshBuffer)
{
    // A zero-length view still gets an allocation so its id exists and the primitives
    // referencing it stay in the same skip/draw logic as the GL path.
    const auto size = std::max<VkDeviceSize>(meshBuffer.data.size(), 4);
    const auto id = nextResourceId++;
    bufferResources.emplace(id,
                            createDeviceLocalBuffer(meshBuffer.data.empty() ? nullptr : meshBuffer.data.data(), size));
    return id;
}

VulkanRenderer::PrimitiveBinding VulkanRenderer::makePrimitiveBinding(const Model& model,
                                                                      const MeshPrimitive& primitive)
{
    PrimitiveBinding binding;
    binding.input = translateVertexInput(primitive.attributes);
    binding.indexCount = static_cast<uint32_t>(primitive.elementCount);
    binding.indexOffset = static_cast<VkDeviceSize>(primitive.byteOffset);

    const auto topology = primitiveTopology(primitive.mode);
    if (!topology.has_value())
    {
        diagnostics.record(FrameDiagnostic::UnsupportedTopology,
                           [&] { return "glTF draw mode " + std::to_string(primitive.mode); });

        return binding;
    }
    binding.topology = topology.value();

    const auto indices = indexType(primitive.componentType);
    if (!indices.has_value())
    {
        diagnostics.record(FrameDiagnostic::UnsupportedIndexType,
                           [&]
                           {
                               return "component type " + std::to_string(primitive.componentType) +
                                      ", outside this backend's uint16/uint32 support";
                           });

        return binding;
    }
    binding.indexType = indices.value();

    const auto bufferHandle = [&](const int bufferIndex) -> VkBuffer
    {
        if (bufferIndex < 0 || std::cmp_greater_equal(bufferIndex, model.meshBuffers.size()))
        {
            return VK_NULL_HANDLE;
        }

        const auto& gpuId = model.meshBuffers[static_cast<size_t>(bufferIndex)].gpuId;
        if (!gpuId.has_value())
        {
            return VK_NULL_HANDLE;
        }

        const auto resource = bufferResources.find(gpuId.value());
        return resource == bufferResources.end() ? VK_NULL_HANDLE : resource->second.buffer;
    };

    binding.indexBuffer = bufferHandle(primitive.meshBufferIndex);
    if (binding.indexBuffer == VK_NULL_HANDLE)
    {
        diagnostics.record(FrameDiagnostic::MeshBufferNotUploaded,
                           [&] { return "index buffer view " + std::to_string(primitive.meshBufferIndex); });

        return binding;
    }

    for (const auto& bufferBind : binding.input.bufferBinds)
    {
        const auto handle = bufferHandle(bufferBind.bufferIndex);
        if (handle == VK_NULL_HANDLE)
        {
            diagnostics.record(FrameDiagnostic::MeshBufferNotUploaded,
                               [&] { return "vertex buffer view " + std::to_string(bufferBind.bufferIndex); });

            return binding;
        }

        binding.vertexBuffers.push_back(handle);
        binding.vertexOffsets.push_back(bufferBind.byteOffset);
    }

    // Pipeline cache key part: the vertex input signature. Target formats and the shader
    // are the other two parts, added where the pipeline is built.
    for (const auto& attribute : binding.input.attributes)
    {
        binding.signature += std::to_string(attribute.location) + ":" + std::to_string(attribute.binding) + ":" +
                             std::to_string(static_cast<int>(attribute.format)) + ":" +
                             std::to_string(binding.input.bindings[attribute.binding].stride) + ";";
    }

    binding.drawable = true;
    return binding;
}

// The whole upload runs inside one mutate() of the model, with a nested mutate() per mesh: the
// only shape that writes the ids into the elements the draw path reads without copying a
// multi-megabyte model out and back, or const_casting the storage's own reference. Lock nesting
// is models -> meshes and models -> materials -> textures, never back into the same storage.
void VulkanRenderer::upload(const Resource<Model>& modelKey)
{
    auto uploadedBuffers = 0u;
    auto meshCount = size_t{0};
    auto materialCount = size_t{0};

    memoryStorageService.models.mutate(modelKey,
                                       [&](Model& model)
                                       {
                                           meshCount = model.meshes.size();
                                           materialCount = model.materials.size();

                                           for (const auto& meshKey : model.meshes)
                                           {
                                               memoryStorageService.meshes.mutate(
                                                   meshKey,
                                                   [&](Mesh& mesh)
                                                   {
                                                       if (mesh.gpuResourceId.has_value())
                                                       {
                                                           return;
                                                       }

                                                       for (auto& buffer : model.meshBuffers)
                                                       {
                                                           if (buffer.gpuId.has_value())
                                                           {
                                                               continue;
                                                           }

                                                           buffer.gpuId = uploadMeshBuffer(buffer);
                                                           buffer.data.clear();
                                                           buffer.data.shrink_to_fit();
                                                           uploadedBuffers++;
                                                       }

                                                       // draw()'s "already uploaded" sentinel. Nothing looks the id up
                                                       // — each primitive owns its binding in gpuVao — but it has to be
                                                       // assigned outside the loop below: GLTFService drops non-indexed
                                                       // primitives, so meshPrimitives can be empty, and an unset
                                                       // sentinel makes draw() re-enter upload() on every frame.
                                                       auto uploadedSentinel = 0u;

                                                       for (auto& primitive : mesh.meshPrimitives)
                                                       {
                                                           const auto id = nextResourceId++;
                                                           primitiveBindings.emplace(
                                                               id, makePrimitiveBinding(model, primitive));
                                                           primitive.gpuVao = id;
                                                           uploadedSentinel = id;
                                                       }

                                                       mesh.gpuResourceId = uploadedSentinel;
                                                   });
                                           }

                                           for (const auto& materialKey : model.materials)
                                           {
                                               uploadMaterialTextures(materialKey);
                                           }
                                       });

    logger.info("Vulkan model uploaded: {} buffer(s), {} mesh(es), {} material(s)", uploadedBuffers, meshCount,
                materialCount);
}

std::optional<unsigned int> VulkanRenderer::uploadTexture(const Resource<Texture>& textureKey)
{
    std::optional<unsigned int> uploadedId;

    // In place: the id is written and the pixels dropped inside the element, so a multi-megabyte
    // payload is never copied through the storage. The const_cast this needed is gone with
    // update() — mutate() is the API that says "write this element".
    memoryStorageService.textures.mutate(
        textureKey,
        [&](Texture& texture)
        {
            if (texture.gpuResourceId.has_value())
            {
                uploadedId = texture.gpuResourceId;
                return;
            }

            // Any channel count at any of the three source precisions uploads (createSampledImage
            // expands it the way GL's driver does); only a payload that cannot describe the image
            // is rejected.
            const auto texelCount = static_cast<size_t>(texture.width) * texture.height * std::max(texture.depth, 1u);
            const auto sourceBytes = texelCount * static_cast<size_t>(channelCount(texture.format)) *
                                     pixelComponentBytes(texture.pixelDataType);
            if (texelCount == 0 || channelCount(texture.format) == 0 || texture.data.size() < sourceBytes)
            {
                diagnostics.record(FrameDiagnostic::UnsupportedTextureLayout,
                                   [&]
                                   {
                                       return texture.name + " is " + std::to_string(texture.width) + "x" +
                                              std::to_string(texture.height) + ", needing " +
                                              std::to_string(sourceBytes) + " byte(s) and carrying " +
                                              std::to_string(texture.data.size());
                                   });

                return;
            }

            // A depth above one is a colour lookup table, and everything about its image differs
            // from a picture's: three dimensions, no mip chain (a grade is interpolated, never
            // minified) and a sampler that clamps rather than repeats, because the table's edges
            // *are* black and white and a wrapped sample there would return the opposite corner.
            const std::array faces = {&std::as_const(texture)};
            const auto id = texture.depth > 1 ? createVolumeImage(texture) : createSampledImage(faces, false);
            texture.gpuResourceId = id;
            uploadedId = id;

            // The GPU owns the texels now; shed the CPU copy the way upload() sheds mesh buffers.
            texture.data.clear();
            texture.data.shrink_to_fit();
        });

    return uploadedId;
}

void VulkanRenderer::uploadMaterialTextures(const Resource<Material>& materialKey)
{
    const auto* material = memoryStorageService.materials.find(materialKey);
    if (material == nullptr)
    {
        return;
    }

    // Every slot a material set will bind. This list is hand-written and that is a hazard worth
    // naming: a slot added to `Material` and forgotten here is never uploaded, so `textureImage`
    // finds no gpuResourceId, the descriptor takes the 1x1 white dummy, and the surface renders as
    // though the texture were pure white — no warning, no missing-texture magenta, just a wrong
    // picture. That is exactly what the blend layers did on their first run: the importer had them,
    // the material had them, and they arrived at the shader as four layers of white.
    for (const auto& textureKey : {material->albedo, material->normal, material->metallicRoughness, material->emissive,
                                   material->occlusion, material->blendMask})
    {
        if (textureKey.has_value())
        {
            static_cast<void>(uploadTexture(textureKey.value()));
        }
    }

    for (const auto& textureKey : material->detail)
    {
        if (textureKey.has_value())
        {
            static_cast<void>(uploadTexture(textureKey.value()));
        }
    }

    for (const auto& textureKey : material->textures)
    {
        static_cast<void>(uploadTexture(textureKey));
    }
}

unsigned int VulkanRenderer::dummyTexture()
{
    if (dummyTextureId != 0)
    {
        return dummyTextureId;
    }

    // GL binds texture 0 for a material slot with no image; Vulkan needs a real descriptor
    // even where the shader gates the sampler behind useTextures.
    const auto texture = Texture{.name = "vulkan dummy texture",
                                 .format = TextureFormat::RGBA,
                                 .pixelDataType = PixelDataType::UnsignedByte,
                                 .width = 1,
                                 .height = 1,
                                 .bitsPerPixel = 32,
                                 .data = {255, 255, 255, 255}};
    const std::array faces = {&texture};
    dummyTextureId = createSampledImage(faces, false);
    return dummyTextureId;
}

// The identity grade, as a two-cube: it interpolates to exactly the input everywhere, so a pass
// bound to it is a pass with no grade at all. Built here rather than shipped as a file because a
// neutral table is a definition, not an asset, and one the engine can state itself cannot go
// missing or arrive subtly wrong.
unsigned int VulkanRenderer::neutralLookupTable()
{
    if (neutralLookupTableId != 0)
    {
        return neutralLookupTableId;
    }

    const auto identity = identityLookupTable(neutralLookupTableSize);

    auto texture = Texture{.name = "vulkan neutral colour grade",
                           .format = TextureFormat::RGB,
                           .pixelDataType = PixelDataType::Float,
                           .width = identity.size,
                           .height = identity.size,
                           .depth = identity.size,
                           .bitsPerPixel = 96,
                           .data = {}};
    texture.data.resize(identity.entries.size() * sizeof(float));
    std::memcpy(texture.data.data(), identity.entries.data(), texture.data.size());

    neutralLookupTableId = createVolumeImage(texture);
    return neutralLookupTableId;
}

std::array<unsigned int, postProcessVolumeCount> VulkanRenderer::neutralVolumes()
{
    std::array<unsigned int, postProcessVolumeCount> volumes{};
    volumes.fill(neutralLookupTable());
    return volumes;
}

unsigned int VulkanRenderer::dummyCubeMap()
{
    if (dummyCubeMapId != 0)
    {
        return dummyCubeMapId;
    }

    const auto texture = Texture{.name = "vulkan dummy cube map",
                                 .format = TextureFormat::RGBA,
                                 .pixelDataType = PixelDataType::UnsignedByte,
                                 .width = 1,
                                 .height = 1,
                                 .bitsPerPixel = 32,
                                 .data = {0, 0, 0, 255}};
    const std::array faces = {&texture, &texture, &texture, &texture, &texture, &texture};
    dummyCubeMapId = createSampledImage(faces, true);
    return dummyCubeMapId;
}

// A 1x1 depth image holding the far value, with a comparison sampler.
//
// GL can leave a shadow unit unbound whenever the shader's cascade count says not to read it,
// because GL only faults on what a fragment actually samples. Vulkan does not: a descriptor a
// pipeline *statically* uses must be there whether the branch runs or not, so a scene with no
// cascades still needs something valid in set 3. Cleared to 1.0 so that a LESS_OR_EQUAL compare
// against it answers "nothing was in front of this" even if some future shader does read it.
unsigned int VulkanRenderer::dummyShadowMap()
{
    if (dummyShadowMapId != 0)
    {
        return dummyShadowMapId;
    }

    constexpr auto format = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = VkExtent3D{1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    ensure(vmaCreateImage(allocator, &imageInfo, &allocationCreateInfo, &image, &allocation, nullptr),
           "vmaCreateImage");

    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    const auto commandBuffer = beginUploadCommands();
    transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, range);
    const VkClearDepthStencilValue clearValue{1.0f, 0};
    vkCmdClearDepthStencilImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
    transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, range);
    finishUploadCommands(commandBuffer);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = range;

    VkImageView view = VK_NULL_HANDLE;
    ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

    dummyShadowMapId = nextResourceId++;
    imageResources.emplace(dummyShadowMapId, ImageResource{.image = image,
                                                           .allocation = allocation,
                                                           .view = view,
                                                           .sampler = createComparisonSampler(format),
                                                           .format = format,
                                                           .width = 1,
                                                           .height = 1,
                                                           .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                                                           .layouts = {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                                           .levelViews = {}});

    return dummyShadowMapId;
}

VkDescriptorSet VulkanRenderer::fallbackShadowSet()
{
    if (dummyShadowSet != VK_NULL_HANDLE)
    {
        return dummyShadowSet;
    }

    std::array<unsigned int, shadowCascadeCount> images{};
    images.fill(dummyShadowMap());
    dummyShadowSet = shadowSet(images, dummyTexture(), dummyTexture(), dummyTexture());

    return dummyShadowSet;
}

// Keyed by the whole tuple of depth image ids and the three images beside them, so a rebuilt
// cascade target — or a view that gathers occlusion sharing the frame with one that does not — gets
// a set of its own rather than a stale view. Linear scan over a list that holds one entry per view
// in a running game, and the one the fallback added.
VkDescriptorSet VulkanRenderer::shadowSet(const std::array<unsigned int, shadowCascadeCount>& imageIds,
                                          const unsigned int occlusionImageId, const unsigned int behindImageId,
                                          const unsigned int cloudMapImageId)
{
    const ShadowSetKey wanted{imageIds, occlusionImageId, behindImageId, cloudMapImageId};
    for (const auto& [key, set] : shadowSets)
    {
        if (key == wanted)
        {
            return set;
        }
    }

    std::array<VkDescriptorImageInfo, shadowCascadeCount> imageInfos{};
    for (auto cascade = 0u; cascade < shadowCascadeCount; cascade++)
    {
        const auto image = imageResources.find(imageIds[cascade]);
        if (image == imageResources.end() || image->second.sampler == VK_NULL_HANDLE)
        {
            // No comparison sampler means the attachment was not created with
            // DepthComparison::LessOrEqual, and a plain sampler bound where the shader declares a
            // sampler2DShadow is undefined — worse than shading lit.
            return VK_NULL_HANDLE;
        }

        imageInfos[cascade] =
            VkDescriptorImageInfo{image->second.sampler, image->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

    VkDescriptorSetAllocateInfo setAllocateInfo{};
    setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocateInfo.descriptorPool = descriptorPool;
    setAllocateInfo.descriptorSetCount = 1;
    setAllocateInfo.pSetLayouts = &shadowSetLayout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &setAllocateInfo, &set) != VK_SUCCESS)
    {
        diagnostics.record(FrameDiagnostic::DescriptorSetUnavailable,
                           [] { return std::string("the descriptor pool has no room for a shadow cascade set"); });

        return VK_NULL_HANDLE;
    }

    // The occlusion image beside them, which is a plain sampled image where the cascades are
    // comparison samplers — so it takes the shared attachment sampler, and a view that gathers no
    // occlusion arrives here with the 1x1 white one, which reads as "nothing is in the way".
    const auto occlusionImage = imageResources.find(occlusionImageId);
    if (occlusionImage == imageResources.end())
    {
        return VK_NULL_HANDLE;
    }

    const VkDescriptorImageInfo occlusionInfo{
        occlusionImage->second.sampler != VK_NULL_HANDLE ? occlusionImage->second.sampler : attachmentSampler,
        occlusionImage->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // The behind copy beside it, another plain sampled image on the shared attachment sampler —
    // which is trilinear with an open LOD clamp, so a reader's textureLod walks the copy's whole
    // chain. A view that takes no copy arrives here with the 1x1 white one.
    const auto behindImage = imageResources.find(behindImageId);
    if (behindImage == imageResources.end())
    {
        return VK_NULL_HANDLE;
    }

    const VkDescriptorImageInfo behindInfo{
        behindImage->second.sampler != VK_NULL_HANDLE ? behindImage->second.sampler : attachmentSampler,
        behindImage->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // The cloud map beside them, on the one sampler that repeats in u: the map is lat-long, its
    // u an azimuth that wraps at 360 degrees, and the shared attachment sampler's clamp would
    // seam the sky at due +z. A scene with none arrives here with the 1x1 white one, which keeps
    // its own sampler.
    const auto cloudImage = imageResources.find(cloudMapImageId);
    if (cloudImage == imageResources.end())
    {
        return VK_NULL_HANDLE;
    }

    const VkDescriptorImageInfo cloudInfo{
        cloudImage->second.sampler != VK_NULL_HANDLE ? cloudImage->second.sampler : cloudMapSampler,
        cloudImage->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // One write covering the cascade binding's whole array: they are consecutive elements of a
    // single binding, so imageInfos is handed over in one go rather than a write per cascade. The
    // occlusion, the behind copy and the cloud map are bindings of their own and therefore writes
    // of their own.
    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = shadowMapBinding;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = shadowCascadeCount;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = imageInfos.data();

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = ambientOcclusionBinding;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &occlusionInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = set;
    writes[2].dstBinding = sceneBehindBinding;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &behindInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = set;
    writes[3].dstBinding = cloudMapBinding;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &cloudInfo;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    shadowSets.emplace_back(wanted, set);

    return set;
}

std::optional<std::array<unsigned int, shadowCascadeCount>>
VulkanRenderer::shadowCascadeImages(const Scene& scene) const
{
    if (scene.shadows.cascades.size() != shadowCascadeCount)
    {
        return std::nullopt;
    }

    std::array<unsigned int, shadowCascadeCount> images{};

    for (auto cascade = 0u; cascade < shadowCascadeCount; cascade++)
    {
        const auto* cascadeCamera = scene.shadows.cascades[cascade].camera;
        const auto* output = cascadeCamera == nullptr || !cascadeCamera->output.has_value()
                                 ? nullptr
                                 : memoryStorageService.frameBuffers.find(cascadeCamera->output.value());
        if (output == nullptr)
        {
            return std::nullopt;
        }

        std::optional<unsigned int> depthImage;
        for (const auto& attachmentKey : output->attachments)
        {
            const auto* attachment = memoryStorageService.bufferAttachments.find(attachmentKey);
            if (attachment != nullptr && attachment->type == FboAttachmentType::Depth)
            {
                depthImage = attachment->gpuResourceId;
                break;
            }
        }

        if (!depthImage.has_value() || !imageResources.contains(depthImage.value()))
        {
            return std::nullopt;
        }

        images[cascade] = depthImage.value();
    }

    return images;
}

// Keyed by the material's slot index and the environment it is bound against. The index is
// enough because releaseMaterial() drops every entry for a slot when that slot is released, so a
// recycled index never finds the previous material's descriptors waiting for it.
VkDescriptorSet VulkanRenderer::materialSet(const Resource<Material>& materialKey,
                                            const unsigned int environmentImageId)
{
    const auto key = (static_cast<uint64_t>(materialKey.index) << 32u) | static_cast<uint64_t>(environmentImageId);
    const auto cached = materialResources.find(key);
    if (cached != materialResources.end())
    {
        return cached->second.set;
    }

    const auto* materialSlot = memoryStorageService.materials.find(materialKey);
    if (materialSlot == nullptr)
    {
        return VK_NULL_HANDLE;
    }

    const auto& material = *materialSlot;

    const auto textureImage = [&](const std::optional<Resource<Texture>>& textureKey) -> std::optional<unsigned int>
    {
        const auto* texture = textureKey.has_value() ? memoryStorageService.textures.find(textureKey.value()) : nullptr;

        if (texture == nullptr || !texture->gpuResourceId.has_value() ||
            !imageResources.contains(texture->gpuResourceId.value()))
        {
            return std::nullopt;
        }

        return texture->gpuResourceId;
    };

    const auto diffuse = textureImage(material.albedo);
    const auto normal = textureImage(material.normal);
    const auto specular = textureImage(material.metallicRoughness);
    const auto emissive = textureImage(material.emissive);
    const auto occlusion = textureImage(material.occlusion);

    // Resolved before any reference into imageResources is taken: creating a dummy inserts
    // into that map.
    const auto fallbackTexture = dummyTexture();
    const auto environment = imageResources.contains(environmentImageId) ? environmentImageId : dummyCubeMap();

    VkDescriptorSetAllocateInfo setAllocateInfo{};
    setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocateInfo.descriptorPool = descriptorPool;
    setAllocateInfo.descriptorSetCount = 1;
    setAllocateInfo.pSetLayouts = &materialSetLayout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &setAllocateInfo, &set) != VK_SUCCESS)
    {
        diagnostics.record(FrameDiagnostic::DescriptorSetUnavailable,
                           [] { return std::string("the descriptor pool has no room for another material set"); });

        return VK_NULL_HANDLE;
    }

    MaterialResource resource;
    resource.set = set;
    void* mapped = nullptr;
    createHostVisibleUniformBuffer(sizeof(MaterialDataUbo), resource.buffer, resource.allocation, mapped);

    MaterialDataUbo materialData{};
    materialData.baseColour = material.baseColour;
    materialData.roughMetal = glm::vec4(material.roughness, material.metalness, material.alphaCutoff, 0.0f);
    materialData.textureTransform = glm::mat4(material.transform);
    materialData.useTextures = glm::ivec4(diffuse.has_value() ? 1 : 0, normal.has_value() ? 1 : 0,
                                          specular.has_value() ? 1 : 0, emissive.has_value() ? 1 : 0);
    // y carries the material's opacity, because the shader has to know it: a transparent fragment
    // must not read the screen-space occlusion buffer, which holds one opaque surface per pixel and
    // not this one. Stated here rather than inferred from the fragment's alpha, which is a
    // threshold on a number that means something else.
    // z says the Blinn-Phong block below was *stated* by the asset rather than left at its
    // defaults, which is the difference between a shader shading from it and a shader falling back.
    materialData.useTextures2 =
        glm::ivec4(occlusion.has_value() ? 1 : 0, material.opaque ? 1 : 0, material.blinnPhong.has_value() ? 1 : 0, 0);
    const auto blinnPhong = material.blinnPhong.value_or(BlinnPhongShading{});
    materialData.blinnPhong =
        glm::vec4(blinnPhong.ambient, blinnPhong.diffuse, blinnPhong.specular, blinnPhong.specularExponent);

    const auto blend = material.blend.value_or(MaterialBlend{});
    materialData.detailTiling =
        glm::vec4(blend.layers[0].tiling, blend.layers[1].tiling, blend.layers[2].tiling, blend.layers[3].tiling);
    materialData.blend = glm::vec4(blend.strength, material.blend.has_value() ? 1.0f : 0.0f, 0.0f, 0.0f);
    std::memcpy(mapped, &materialData, sizeof(materialData));
    ensure(vmaFlushAllocation(allocator, resource.allocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

    // The blend mask and the detail layers. An absent one takes the white dummy like every other
    // slot, and `blend.y` above is what stops a shader reading it as a layer that is entirely white.
    const auto blendMask = textureImage(material.blendMask);
    const std::array sampledImages = {diffuse.value_or(fallbackTexture),
                                      normal.value_or(fallbackTexture),
                                      specular.value_or(fallbackTexture),
                                      emissive.value_or(fallbackTexture),
                                      occlusion.value_or(fallbackTexture),
                                      environment,
                                      blendMask.value_or(fallbackTexture),
                                      textureImage(material.detail[0]).value_or(fallbackTexture),
                                      textureImage(material.detail[1]).value_or(fallbackTexture),
                                      textureImage(material.detail[2]).value_or(fallbackTexture),
                                      textureImage(material.detail[3]).value_or(fallbackTexture)};
    static_assert(std::tuple_size_v<decltype(sampledImages)> == materialTextureSlotCount);

    const VkDescriptorBufferInfo bufferInfo{resource.buffer, 0, sizeof(MaterialDataUbo)};
    std::array<VkDescriptorImageInfo, materialTextureSlotCount> imageInfos{};
    std::array<VkWriteDescriptorSet, materialTextureSlotCount + 1> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &bufferInfo;

    // sampledImages is in MaterialTextureSlot order; the binding each lands on is the
    // contract's, not this loop's index.
    for (uint32_t slot = 0; slot < materialTextureSlotCount; slot++)
    {
        const auto& image = imageResources.at(sampledImages[slot]);
        imageInfos[slot] = VkDescriptorImageInfo{image.sampler, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        writes[slot + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[slot + 1].dstSet = set;
        writes[slot + 1].dstBinding = textureBinding(static_cast<MaterialTextureSlot>(slot));
        writes[slot + 1].descriptorCount = 1;
        writes[slot + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[slot + 1].pImageInfo = &imageInfos[slot];
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    materialResources.emplace(key, resource);
    return set;
}

VkDescriptorSet VulkanRenderer::attachmentSet(const std::span<const PostProcessBinding> inputs,
                                              const unsigned int lookupTableImageId,
                                              const std::array<unsigned int, postProcessVolumeCount>& volumeImageIds)
{
    // Resolved before any reference into imageResources is taken: creating the fallback inserts
    // into that map, and the padding below names it.
    const auto fallback = PostProcessBinding{.gpuResourceId = dummyTexture(), .level = 0};
    const auto bindings = postProcessBindings(inputs, fallback);
    const FullscreenSetKey wanted{bindings, lookupTableImageId, volumeImageIds};

    for (const auto& [key, set] : attachmentSets)
    {
        if (key == wanted)
        {
            return set;
        }
    }

    std::array<VkDescriptorImageInfo, postProcessInputCount> imageInfos{};
    for (auto slot = 0u; slot < postProcessInputCount; slot++)
    {
        const auto image = imageResources.find(bindings[slot].gpuResourceId);
        if (image == imageResources.end())
        {
            // A named image that is not there leaves an element of the array unwritten, and an
            // unwritten element is exactly what the pipeline's static use of the array would fault
            // on. The pass records nothing instead.
            return VK_NULL_HANDLE;
        }

        imageInfos[slot] = VkDescriptorImageInfo{
            image->second.sampler != VK_NULL_HANDLE ? image->second.sampler : attachmentSampler,
            levelView(image->second, bindings[slot].level), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

    VkDescriptorSetAllocateInfo setAllocateInfo{};
    setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocateInfo.descriptorPool = descriptorPool;
    setAllocateInfo.descriptorSetCount = 1;
    setAllocateInfo.pSetLayouts = &fullscreenSetLayout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &setAllocateInfo, &set) != VK_SUCCESS)
    {
        diagnostics.record(FrameDiagnostic::DescriptorSetUnavailable,
                           [] { return std::string("the descriptor pool has no room for another attachment set"); });

        return VK_NULL_HANDLE;
    }

    const auto grade = imageResources.find(lookupTableImageId);
    if (grade == imageResources.end())
    {
        return VK_NULL_HANDLE;
    }

    const VkDescriptorImageInfo gradeInfo{grade->second.sampler, grade->second.view,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    // The volumes beside it, each with the sampler its own image was created with — a volume image
    // always carries one. The caller filled unused slots with the neutral table already.
    std::array<VkDescriptorImageInfo, postProcessVolumeCount> volumeInfos{};
    for (auto slot = 0u; slot < postProcessVolumeCount; slot++)
    {
        const auto volume = imageResources.find(volumeImageIds[slot]);
        if (volume == imageResources.end())
        {
            return VK_NULL_HANDLE;
        }

        volumeInfos[slot] = VkDescriptorImageInfo{volume->second.sampler, volume->second.view,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

    // One write covering the input binding's whole array, as the cascades are written: the inputs
    // are consecutive elements of a single binding. The grade and the two volumes are bindings of
    // their own and therefore writes of their own.
    std::array<VkWriteDescriptorSet, 2 + postProcessVolumeCount> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = postProcessInputBinding;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = postProcessInputCount;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = imageInfos.data();

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = lookupTableBinding;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &gradeInfo;

    for (auto slot = 0u; slot < postProcessVolumeCount; slot++)
    {
        writes[2 + slot].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2 + slot].dstSet = set;
        writes[2 + slot].dstBinding = cloudBaseNoiseBinding + slot;
        writes[2 + slot].dstArrayElement = 0;
        writes[2 + slot].descriptorCount = 1;
        writes[2 + slot].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2 + slot].pImageInfo = &volumeInfos[slot];
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    attachmentSets.emplace_back(wanted, set);
    return set;
}

const VulkanRenderer::ResolvedPipeline*
VulkanRenderer::scenePipeline(const unsigned int shaderId, PrimitiveBinding& binding, const VkCullModeFlags cullMode,
                              const VkFormat colorFormat, const VkFormat depthFormat, const bool blend,
                              const bool depthWrite)
{
    // The whole of what a pipeline's identity adds to the primitive's own vertex input: the
    // shader, the cull mode its material asked for, the two formats of the target it is drawn
    // into, whether its alpha is a coverage, and whether it writes depth. The shader has to be in
    // the key because a view may override it — a shadow cascade draws the same primitives through a
    // depth-only shader — and the buffer list cached with the entry is the shader's, not the
    // primitive's. A failed build is not retried: the miss below caches a VK_NULL_HANDLE pipeline
    // too.
    const auto matches = [&](const ResolvedPipeline& candidate)
    {
        return candidate.shaderId == shaderId && candidate.cullMode == cullMode &&
               candidate.colorFormat == colorFormat && candidate.depthFormat == depthFormat &&
               candidate.blend == blend && candidate.depthWrite == depthWrite;
    };

    if (const auto resolved = std::ranges::find_if(binding.pipelines, matches); resolved != binding.pipelines.end())
    {
        return &*resolved;
    }

    ResolvedPipeline entry{.shaderId = shaderId,
                           .cullMode = cullMode,
                           .colorFormat = colorFormat,
                           .depthFormat = depthFormat,
                           .blend = blend,
                           .depthWrite = depthWrite,
                           .pipeline = VK_NULL_HANDLE,
                           .boundBuffers = {},
                           .boundOffsets = {}};

    const auto shader = shaderObjects.find(shaderId);
    if (shader == shaderObjects.end() || shader->second.fullscreen)
    {
        diagnostics.record(FrameDiagnostic::ScenePipelineUnavailable,
                           [&] { return "shader object " + std::to_string(shaderId) + " is not a scene shader"; });

        binding.pipelines.push_back(std::move(entry));

        return &binding.pipelines.back();
    }

    // Exactly the locations the vertex shader declares are fed, no more: a bound attribute
    // the shader never reads is work the driver has to discard, and Vulkan reports it.
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    for (const auto& attribute : binding.input.attributes)
    {
        const auto consumed = std::ranges::find(shader->second.vertexInputLocations, attribute.location) !=
                              shader->second.vertexInputLocations.end();
        if (!consumed)
        {
            continue;
        }

        const auto bindingIndex = static_cast<uint32_t>(bindings.size());
        auto pipelineBinding = binding.input.bindings[attribute.binding];
        pipelineBinding.binding = bindingIndex;
        bindings.push_back(pipelineBinding);

        auto pipelineAttribute = attribute;
        pipelineAttribute.binding = bindingIndex;
        attributes.push_back(pipelineAttribute);

        entry.boundBuffers.push_back(binding.vertexBuffers[attribute.binding]);
        entry.boundOffsets.push_back(binding.vertexOffsets[attribute.binding]);
    }

    // Vulkan rejects a vertex shader input that the pipeline does not feed; GL simply
    // supplied the generic default (0, 0, 0, 1), which is what the dummy binding holds.
    for (const auto location : shader->second.vertexInputLocations)
    {
        const auto present = std::ranges::any_of(attributes, [&](const VkVertexInputAttributeDescription& attribute)
                                                 { return attribute.location == location; });
        if (present)
        {
            continue;
        }

        const auto bindingIndex = static_cast<uint32_t>(bindings.size());
        bindings.push_back(VkVertexInputBindingDescription{bindingIndex, 0, VK_VERTEX_INPUT_RATE_VERTEX});
        attributes.push_back(
            VkVertexInputAttributeDescription{location, bindingIndex, VK_FORMAT_R32G32B32A32_SFLOAT, 0});
        entry.boundBuffers.push_back(dummyVertexBuffer.buffer);
        entry.boundOffsets.push_back(0);
    }

    // The pipeline's rendering formats are part of its identity, so they are part of the key:
    // the attachments they come from are the ones FboAttachment::internalFormat asked for.
    const auto key = std::to_string(shaderId) + "|" + std::to_string(cullMode) + "|" +
                     std::to_string(static_cast<int>(binding.topology)) + "|" +
                     std::to_string(static_cast<int>(colorFormat)) + "+" +
                     std::to_string(static_cast<int>(depthFormat)) + "|" + (blend ? "blend" : "opaque") + "|" +
                     (depthWrite ? "zwrite" : "zread") + "|" + binding.signature;
    if (const auto cachedPipeline = scenePipelines.find(key); cachedPipeline != scenePipelines.end())
    {
        entry.pipeline = cachedPipeline->second;
        binding.pipelines.push_back(std::move(entry));

        return &binding.pipelines.back();
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = shader->second.vertexModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = shader->second.fragmentModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
    vertexInput.pVertexBindingDescriptions = bindings.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = binding.topology;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    // GL runs glEnable(GL_CULL_FACE) + glCullFace(GL_BACK) and drops culling for materials
    // flagged non-opaque. GL's front face is the default CCW; the winding stays CCW here
    // because the negative viewport height and Vulkan's downward framebuffer y cancel out
    // (verified against the GL capture: CLOCKWISE culls the skybox interior and the
    // building's outer walls, leaving the scene inside-out).
    rasterization.cullMode = cullMode;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // GL enables GL_DEPTH_TEST globally and never touches the func or the write mask, so this was
    // GL's default LESS — with the write mask now a property of the draw rather than of the API.
    //
    // **LESS_OR_EQUAL since 2026-08-26, and pre-Z is what needs it.** A view that loads a depth
    // buffer another view already filled with its own geometry re-rasterises that geometry at
    // *exactly* the depth already stored — the prepass and the shading pair compute `gl_Position`
    // from the same expression over the same `localToScreen`, so the ties are exact rather than
    // near — and LESS rejects every one of them. The symptom is not subtle: a black world. What the
    // relaxation costs elsewhere is that two coplanar opaque surfaces at identical depth now let the
    // *later* draw win instead of the earlier, which is the ordinary convention and is measured by
    // both gates rather than argued about here.
    //
    // **A transparent surface tests depth and does not write it.** Writing is a claim that nothing
    // behind this fragment is visible, which is the one thing a pane of glass does not say. Two
    // near-coplanar panes — and a car's glazing is always two, an outer shell and an inner one, the
    // rear screen here overlapping through 0.12 cubic metres — then z-fight: whichever drew first
    // owns the depth and the second wins or loses per *triangle*, which is why the artefact comes
    // out as hard-edged wedges following the tessellation rather than as anything resembling glass.
    // Sorting back to front is what makes dropping the write safe: the far pane is already in the
    // buffer when the near one blends over it, so the depth test is not what was ordering them.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.maxDepthBounds = 1.0f;

    // GL runs with glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) globally enabled.
    //
    // Off for a pass that writes *data* rather than colour, which is what `blend` distinguishes.
    // The distinction had no reason to exist while every scene pass wrote a colour whose alpha was
    // a coverage between zero and one; the occlusion prepass writes a distance there, and blending
    // by it multiplies the normal by a few hundred and drives the stored alpha to plus and then
    // minus infinity as fragments land on top of each other. What that produces is not obviously
    // broken: it is a buffer that reads as geometry on the triangles one fragment covered and as
    // empty sky on the triangles two did.
    // **Premultiplied**: the scene pass hands over a colour that has already been scaled by its own
    // coverage, so the source factor is ONE rather than SRC_ALPHA. That is what lets a surface
    // decide *which* of its terms coverage applies to — glass transmits what is behind it and
    // still reflects the sky at full strength, and a blend state that multiplies the whole fragment
    // cannot express the difference (see PbrFragmentShader's `ads`). For an opaque fragment the two
    // forms are the same arithmetic, so nothing that writes alpha 1 moves by a bit.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = blend ? VK_TRUE : VK_FALSE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    // A depth-only target has no colour attachment to blend into, and one blend state per colour
    // attachment is the rule, so the count is zero for both this and the rendering info below.
    const auto colorAttachmentCount = colorFormat == VK_FORMAT_UNDEFINED ? 0u : 1u;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = colorAttachmentCount;
    colorBlend.pAttachments = colorAttachmentCount == 0 ? nullptr : &blendAttachment;

    const std::array dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = colorAttachmentCount;
    renderingInfo.pColorAttachmentFormats = colorAttachmentCount == 0 ? nullptr : &colorFormat;
    renderingInfo.depthAttachmentFormat = depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = scenePipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto createResult = vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline);
    if (createResult != VK_SUCCESS)
    {
        diagnostics.record(FrameDiagnostic::ScenePipelineUnavailable,
                           [&]
                           {
                               return "shader object " + std::to_string(shaderId) +
                                      ", vkCreateGraphicsPipelines returned VkResult " +
                                      std::to_string(static_cast<int>(createResult));
                           });

        binding.pipelines.push_back(std::move(entry));

        return &binding.pipelines.back();
    }

    scenePipelines.emplace(key, pipeline);
    entry.pipeline = pipeline;
    binding.pipelines.push_back(std::move(entry));

    return &binding.pipelines.back();
}

VkPipeline VulkanRenderer::offscreenPipeline(const unsigned int shaderId, const VkFormat colorFormat,
                                             const FullscreenBlend blend)
{
    // Shader ids are a small counter, so the top two bits above them are free to carry the blend
    // mode. The weight a constant-blend pipeline mixes at is dynamic state and deliberately not in
    // the key: one pipeline serves every weight, and a knob that moved between two runs would
    // otherwise build a pipeline per setting.
    const auto key = (static_cast<uint64_t>(blend) << 62u) | (static_cast<uint64_t>(shaderId) << 32u) |
                     static_cast<uint32_t>(colorFormat);
    const auto cached = offscreenPipelines.find(key);
    if (cached != offscreenPipelines.end())
    {
        return cached->second;
    }

    const auto shader = shaderObjects.find(shaderId);
    if (shader == shaderObjects.end() || !shader->second.fullscreen)
    {
        return VK_NULL_HANDLE;
    }

    const auto pipeline =
        buildFullscreenPipeline(shader->second.vertexModule, shader->second.fragmentModule, blend, colorFormat);
    if (pipeline != VK_NULL_HANDLE)
    {
        offscreenPipelines.emplace(key, pipeline);
    }

    return pipeline;
}

void VulkanRenderer::setViewport(const int width, const int height)
{
    // Recording the extent is all this does. It used to have to close an open frame first,
    // because the resize callback destroys the framebuffers a half-recorded command buffer
    // still names — but the callback fires from glfwPollEvents, which the window pumps after
    // Engine::step has already closed the frame it opened. Recreation stays lazy: the next
    // beginFrame rebuilds the swapchain if this extent no longer matches it.
    requestedExtent = VkExtent2D{static_cast<uint32_t>(std::max(width, 0)), static_cast<uint32_t>(std::max(height, 0))};
}

std::optional<VkDeviceSize> VulkanRenderer::allocateDrawDataSlot()
{
    auto& frame = frames[frameIndex];
    if (frame.drawDataSlotsUsed >= drawDataRingSlots)
    {
        diagnostics.record(FrameDiagnostic::DrawDataRingExhausted,
                           [&]
                           {
                               return "the ring holds " + std::to_string(drawDataRingSlots) + " slots of " +
                                      std::to_string(drawDataStride) + " bytes";
                           });

        return std::nullopt;
    }

    const auto offset = static_cast<VkDeviceSize>(frame.drawDataSlotsUsed) * drawDataStride;
    frame.drawDataSlotsUsed++;
    return offset;
}

VkDeviceSize VulkanRenderer::writePaintSlot(const Paint& paint)
{
    // Offset zero is the unpainted block every other draw binds, so a renderable that states no
    // paint — or one the ring has no room left for — reads zeros and the shader falls back to what
    // the material's own textures say. That is why the block carries an explicit painted flag:
    // without it, "no slot left" and "black paint" would be the same sixty-four bytes.
    if (!paint.enabled)
    {
        return 0;
    }

    const auto slot = allocatePaintDataSlot();
    if (!slot.has_value())
    {
        return 0;
    }

    auto& frame = frames[frameIndex];

    const PaintDataUbo block{
        .colour = glm::vec4(paint.colour, paint.flakeDensity),
        .flake = glm::vec4(paint.flakeColour, paint.flakeScale),
        .clearcoat = glm::vec4(paint.clearcoat, paint.clearcoatRoughness, paint.orangePeel, paint.orangePeelScale),
        .wear = glm::vec4(paint.scratch, paint.dirt, 1.0f, 0.0f)};

    std::memcpy(static_cast<unsigned char*>(frame.paintDataMapped) + slot.value(), &block, sizeof(block));

    return slot.value();
}

std::optional<VkDeviceSize> VulkanRenderer::allocatePaintDataSlot()
{
    auto& frame = frames[frameIndex];
    if (frame.paintDataSlotsUsed >= paintDataRingSlots)
    {
        diagnostics.record(FrameDiagnostic::PaintDataRingExhausted,
                           [&]
                           {
                               return "the ring holds " + std::to_string(paintDataRingSlots) + " slots of " +
                                      std::to_string(paintDataStride) + " bytes";
                           });

        return std::nullopt;
    }

    const auto offset = static_cast<VkDeviceSize>(frame.paintDataSlotsUsed) * paintDataStride;
    frame.paintDataSlotsUsed++;

    return offset;
}

std::optional<VkDeviceSize> VulkanRenderer::allocateJointDataSlot()
{
    auto& frame = frames[frameIndex];
    if (frame.jointDataSlotsUsed >= jointDataRingSlots)
    {
        diagnostics.record(FrameDiagnostic::JointDataRingExhausted,
                           [&]
                           {
                               return "the ring holds " + std::to_string(jointDataRingSlots) + " slots of " +
                                      std::to_string(jointDataStride) + " bytes";
                           });

        return std::nullopt;
    }

    const auto offset = static_cast<VkDeviceSize>(frame.jointDataSlotsUsed) * jointDataStride;
    frame.jointDataSlotsUsed++;
    return offset;
}

std::optional<VkDeviceSize> VulkanRenderer::allocateFrameDataSlot()
{
    auto& frame = frames[frameIndex];
    if (frame.frameDataSlotsUsed >= frameDataRingSlots)
    {
        diagnostics.record(FrameDiagnostic::FrameDataRingExhausted,
                           [&]
                           {
                               return "the ring holds " + std::to_string(frameDataRingSlots) + " slots of " +
                                      std::to_string(frameDataStride) + " bytes";
                           });

        return std::nullopt;
    }

    const auto offset = static_cast<VkDeviceSize>(frame.frameDataSlotsUsed) * frameDataStride;
    frame.frameDataSlotsUsed++;
    return offset;
}

std::expected<std::vector<uint32_t>, std::string>
VulkanRenderer::compileToSpirv(const std::string& source, const shaderc_shader_kind kind, const char* stageName)
{
    // The C++ RAII wrapper over the C API: results release themselves, and both stay
    // confined to this translation unit's global module fragment.
    const shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    // spirv-opt eliminates stage interface variables a shader declares but never reads, so
    // a Vulkan-dialect source must not declare any: an input dropped from the fragment
    // stage leaves the vertex stage writing an output nothing consumes, and validation
    // reports the mismatch on every pipeline built from the pair.
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    // Every source is compiled with the contract macros predefined, so no shader spells a set
    // index, a binding, an attribute location or an array bound that C++ also holds. The macro
    // values join the cache key below for the same reason they exist: a contract change must
    // recompile every shader that read it.
    std::string cacheKeyMaterial;
    cacheKeyMaterial.reserve(source.size() + 1024);
    cacheKeyMaterial += stageName;
    cacheKeyMaterial += '\x1f';
    cacheKeyMaterial += std::to_string(static_cast<int>(kind));
    cacheKeyMaterial += '\x1f';
    for (const auto& macro : shaderContractMacros())
    {
        auto value = std::to_string(macro.value);
        options.AddMacroDefinition(std::string(macro.name), value);
        cacheKeyMaterial += macro.name;
        cacheKeyMaterial += '=';
        cacheKeyMaterial += value;
        cacheKeyMaterial += ';';
    }

    // The same list for the contract's non-integer numbers. std::to_string writes six decimal
    // places, which is more than enough for a decimal literal to round back to the same float.
    for (const auto& macro : shaderContractFloatMacros())
    {
        auto value = std::to_string(macro.value);
        options.AddMacroDefinition(std::string(macro.name), value);
        cacheKeyMaterial += macro.name;
        cacheKeyMaterial += '=';
        cacheKeyMaterial += value;
        cacheKeyMaterial += ';';
    }
    cacheKeyMaterial += '\x1f';
    cacheKeyMaterial += source;

    // The compiled words, cached on disk keyed by everything the compilation depends on. The
    // measured first frame spent 3.8 s in shaderc across the shader set, and the driver's own
    // disk cache cannot reach any of it — it only sees the SPIR-V this cache is of.
    std::string cachePath;
    if (const auto directory = engineCacheDirectory(); !directory.empty())
    {
        std::array<char, 32> hashText{};
        std::snprintf(hashText.data(), hashText.size(), "%016llx",
                      static_cast<unsigned long long>(fnv1a64(cacheKeyMaterial)));
        cachePath = directory + "/spirv-" + hashText.data() + ".spv";

        if (std::FILE* file = std::fopen(cachePath.c_str(), "rb"); file != nullptr)
        {
            std::fseek(file, 0, SEEK_END);
            const auto size = std::ftell(file);
            std::fseek(file, 0, SEEK_SET);
            std::vector<uint32_t> words;
            auto usable = size > 0 && size % static_cast<long>(sizeof(uint32_t)) == 0;
            if (usable)
            {
                words.resize(static_cast<size_t>(size) / sizeof(uint32_t));
                usable = std::fread(words.data(), sizeof(uint32_t), words.size(), file) == words.size() &&
                         words.front() == 0x07230203u;
            }
            std::fclose(file);
            if (usable)
            {
                return words;
            }
        }
    }

    const auto result = compiler.CompileGlslToSpv(source, kind, stageName, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        // A Vulkan source that will not compile is a real failure on this backend, and shaderc's
        // message names the line — which is the whole value of it, and what was being logged and
        // then left out of what the caller reported.
        return std::unexpected(result.GetErrorMessage());
    }

    if (result.GetNumWarnings() > 0)
    {
        logger.warn("Vulkan {} shader compiled with {} warning(s)", stageName, result.GetNumWarnings());
    }

    std::vector<uint32_t> words(result.cbegin(), result.cend());

    if (!cachePath.empty() && !words.empty())
    {
        if (std::FILE* file = std::fopen(cachePath.c_str(), "wb"); file != nullptr)
        {
            if (std::fwrite(words.data(), sizeof(uint32_t), words.size(), file) != words.size())
            {
                // A truncated blob would fail the magic-and-size check next run, but removing it
                // now keeps even that cold start from happening.
                std::fclose(file);
                std::remove(cachePath.c_str());
            }
            else
            {
                std::fclose(file);
            }
        }
    }

    return words;
}

VkShaderModule VulkanRenderer::createShaderModule(const std::vector<uint32_t>& spirv) const
{
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = spirv.size() * sizeof(uint32_t);
    moduleInfo.pCode = spirv.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    ensure(vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule), "vkCreateShaderModule");
    return shaderModule;
}

VkPipeline VulkanRenderer::buildFullscreenPipeline(const VkShaderModule vertexModule,
                                                   const VkShaderModule fragmentModule, const FullscreenBlend blend,
                                                   const VkFormat colorFormat) const
{
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertexModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragmentModule;
    stages[1].pName = "main";

    // The oversized triangle comes from gl_VertexIndex: no vertex buffers at all.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    // The negative-viewport Y-flip inverts winding; culling a fullscreen triangle buys
    // nothing, so it stays off rather than depending on the flip.
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    // GL runs with glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) globally enabled. Straight
    // alpha, not the premultiplied form the scene pass uses: a fullscreen pass writes a whole
    // image over a target it does not read through, so there is no coverage here to split a
    // fragment's terms by. A pass whose alpha channel is data rather than coverage — the fog
    // march's transmittance — turns blending off instead.
    //
    // The third state is the constant pair, and it exists for the pass that is *both*: the cloud
    // dome's alpha is a transmittance the skybox reads, and the dome also accumulates over frames,
    // so the mix has to be stated somewhere other than in the fragment's alpha. It is set per pass
    // through vkCmdSetBlendConstants rather than baked here, so one pipeline serves every weight.
    //
    // Blending of either kind reads the destination, which means the pass must also load its
    // target — see PostProcess::loadColour, and note that the source-alpha arm ran for years over
    // a DONT_CARE load op without noticing, because every shader but this one writes alpha 1.0.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = blend == FullscreenBlend::None ? VK_FALSE : VK_TRUE;
    const auto sourceFactor =
        blend == FullscreenBlend::Constant ? VK_BLEND_FACTOR_CONSTANT_ALPHA : VK_BLEND_FACTOR_SRC_ALPHA;
    const auto destinationFactor = blend == FullscreenBlend::Constant ? VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA
                                                                      : VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.srcColorBlendFactor = sourceFactor;
    blendAttachment.dstColorBlendFactor = destinationFactor;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = sourceFactor;
    blendAttachment.dstAlphaBlendFactor = destinationFactor;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    // Blend constants are declared dynamic on every fullscreen pipeline and set on every fullscreen
    // pass, whether or not the factors above name them: a state nothing reads costs one command,
    // and one that is sometimes declared and sometimes not is a state somebody forgets to set.
    const std::array dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = fullscreenPipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    ensure(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline),
           "vkCreateGraphicsPipelines");
    return pipeline;
}

std::expected<unsigned int, std::string> VulkanRenderer::createShaderObject(const ShaderDescriptor& shaderDescriptor)
{
    try
    {
        if (shaderDescriptor.vertexShaderSource.empty() || shaderDescriptor.fragmentShaderSource.empty())
        {
            return std::unexpected("the shader descriptor carries no vertexShaderSource and/or "
                                   "fragmentShaderSource, which this backend cannot substitute");
        }

        const auto vertexSpirv =
            compileToSpirv(shaderDescriptor.vertexShaderSource, shaderc_glsl_vertex_shader, "vertex");
        if (!vertexSpirv)
        {
            return std::unexpected("the vertex source did not compile to SPIR-V: " + vertexSpirv.error());
        }

        const auto fragmentSpirv =
            compileToSpirv(shaderDescriptor.fragmentShaderSource, shaderc_glsl_fragment_shader, "fragment");
        if (!fragmentSpirv)
        {
            return std::unexpected("the fragment source did not compile to SPIR-V: " + fragmentSpirv.error());
        }

        ShaderObject shaderObject;
        shaderObject.vertexModule = createShaderModule(vertexSpirv.value());
        shaderObject.fragmentModule = createShaderModule(fragmentSpirv.value());
        shaderObject.vertexInputLocations = spirvVertexInputLocations(vertexSpirv.value());
        shaderObject.fullscreen = shaderObject.vertexInputLocations.empty();

        if (shaderObject.fullscreen)
        {
            shaderObject.swapchainTargetPipeline =
                buildFullscreenPipeline(shaderObject.vertexModule, shaderObject.fragmentModule,
                                        FullscreenBlend::SourceAlpha, surfaceFormat.format);
        }

        const auto id = nextResourceId++;
        shaderObjects.emplace(id, shaderObject);

        logger.info("Vulkan shader object {} ready: vertex {} + fragment {} SPIR-V words; {}", id, vertexSpirv->size(),
                    fragmentSpirv->size(),
                    shaderObject.fullscreen
                        ? "swapchain fullscreen pipeline built; offscreen targets build on first use"
                        : "scene pipeline awaits vertex input from the draw path");
        return id;
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

VkCommandBuffer VulkanRenderer::beginUploadCommands() const
{
    if (uploadBatchCommands != VK_NULL_HANDLE)
    {
        return uploadBatchCommands;
    }

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = uploadCommandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    ensure(vkAllocateCommandBuffers(device, &allocateInfo, &uploadBatchCommands), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ensure(vkBeginCommandBuffer(uploadBatchCommands, &beginInfo), "vkBeginCommandBuffer");
    return uploadBatchCommands;
}

void VulkanRenderer::finishUploadCommands(const VkCommandBuffer commandBuffer) const
{
    // The batch stays open: recording is complete for this upload, submission is
    // flushUploadCommands' — once per frame from endFrame, or on the spot for a reader that
    // needs the GPU's answer now. Execution order against the frame is unchanged, because the
    // flush submits ahead of the frame's own command buffer exactly as each fence-waited upload
    // used to complete ahead of it.
    static_cast<void>(commandBuffer);
}

void VulkanRenderer::retireUploadStaging(const VkBuffer buffer, const VmaAllocation allocation) const
{
    uploadBatchStaging.emplace_back(buffer, allocation);
}

void VulkanRenderer::flushUploadCommands() const
{
    if (uploadBatchCommands == VK_NULL_HANDLE)
    {
        return;
    }

    ensure(vkEndCommandBuffer(uploadBatchCommands), "vkEndCommandBuffer");

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = uploadBatchCommands;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence uploadFence = VK_NULL_HANDLE;
    ensure(vkCreateFence(device, &fenceInfo, nullptr, &uploadFence), "vkCreateFence");

    ensure(vkQueueSubmit2(graphicsQueue, 1, &submitInfo, uploadFence), "vkQueueSubmit2");
    ensure(vkWaitForFences(device, 1, &uploadFence, VK_TRUE, waitForever), "vkWaitForFences");

    vkDestroyFence(device, uploadFence, nullptr);
    vkFreeCommandBuffers(device, uploadCommandPool, 1, &uploadBatchCommands);
    uploadBatchCommands = VK_NULL_HANDLE;

    for (const auto& [buffer, allocation] : uploadBatchStaging)
    {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
    uploadBatchStaging.clear();
}

// A three-dimensional image: one mip (a minified grade is a different grade, and a noise volume's
// filtering is the march's own business), linearly filtered, and a format the payload decides.
// Float data is the colour grade, expanded to four float channels under GL's padding rule and
// clamped, because the table's edges hold black and white and a repeat would fetch the opposite
// corner of the cube. Byte data is baked volumetric noise, uploaded at its own channel count —
// sixteen bytes a texel would put 33 MB where 8 belong — and repeating, because the noise tiles
// and a clamp would freeze its last texel across every repeat. Three byte channels pad to four
// with an opaque alpha: R8G8B8_UNORM has no guaranteed sampled-image support.
unsigned int VulkanRenderer::createVolumeImage(const Texture& texture) const
{
    const auto texelCount = static_cast<size_t>(texture.width) * texture.height * texture.depth;
    const auto sourceChannels = static_cast<size_t>(channelCount(texture.format));
    const auto componentBytes = static_cast<size_t>(pixelComponentBytes(texture.pixelDataType));

    if (sourceChannels == 0 || texture.data.size() < texelCount * sourceChannels * componentBytes)
    {
        throw std::runtime_error("volume texture source has no usable payload");
    }

    const auto byteTexels = texture.pixelDataType == PixelDataType::UnsignedByte;
    if (!byteTexels && texture.pixelDataType != PixelDataType::Float)
    {
        throw std::runtime_error("a volume texture carries byte or float texels; nothing uploads the third kind");
    }

    const auto uploadChannels = byteTexels && sourceChannels != 3 ? sourceChannels : size_t{4};
    constexpr std::array<VkFormat, 4> byteFormats = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_UNDEFINED,
                                                     VK_FORMAT_R8G8B8A8_UNORM};
    const auto format = byteTexels ? byteFormats[uploadChannels - 1] : VK_FORMAT_R32G32B32A32_SFLOAT;
    const auto uploadBytes = static_cast<VkDeviceSize>(texelCount * uploadChannels * componentBytes);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = format;
    imageInfo.extent = VkExtent3D{texture.width, texture.height, texture.depth};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imageAllocationInfo{};
    imageAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation imageAllocation = nullptr;
    ensure(vmaCreateImage(allocator, &imageInfo, &imageAllocationInfo, &image, &imageAllocation, nullptr),
           "vmaCreateImage");

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = uploadBytes;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAllocationCreateInfo{};
    stagingAllocationCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = nullptr;
    VmaAllocationInfo stagingAllocationInfo{};
    ensure(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocationCreateInfo, &stagingBuffer, &stagingAllocation,
                           &stagingAllocationInfo),
           "vmaCreateBuffer");

    if (uploadChannels == sourceChannels)
    {
        // The straight copy every compact noise volume takes: the format above matches the payload
        // texel for texel.
        std::memcpy(stagingAllocationInfo.pMappedData, texture.data.data(), uploadBytes);
    }
    else if (byteTexels)
    {
        // Three byte channels into four, alpha opaque and read by nothing.
        auto* destination = static_cast<unsigned char*>(stagingAllocationInfo.pMappedData);
        const auto* source = texture.data.data();
        for (size_t texel = 0; texel < texelCount; texel++)
        {
            destination[texel * 4 + 0] = source[texel * 3 + 0];
            destination[texel * 4 + 1] = source[texel * 3 + 1];
            destination[texel * 4 + 2] = source[texel * 3 + 2];
            destination[texel * 4 + 3] = 255;
        }
    }
    else
    {
        // The grade's expansion, exactly as it always ran: a source with three components gets
        // zero in the fourth and this one gets one, which nothing reads.
        auto* destination = static_cast<float*>(stagingAllocationInfo.pMappedData);
        const auto* source = reinterpret_cast<const float*>(texture.data.data());
        for (size_t texel = 0; texel < texelCount; texel++)
        {
            for (size_t channel = 0; channel < 3; channel++)
            {
                destination[texel * 4 + channel] =
                    channel < sourceChannels ? source[texel * sourceChannels + channel] : 0.0f;
            }

            destination[texel * 4 + 3] = 1.0f;
        }
    }
    ensure(vmaFlushAllocation(allocator, stagingAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

    const auto commandBuffer = beginUploadCommands();
    const VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, whole);

    VkBufferImageCopy region{};
    region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = VkExtent3D{texture.width, texture.height, texture.depth};
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT, whole);

    finishUploadCommands(commandBuffer);
    retireUploadStaging(stagingBuffer, stagingAllocation);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = imageInfo.format;
    viewInfo.subresourceRange = whole;

    VkImageView view = VK_NULL_HANDLE;
    ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

    // The grade clamps because its edges are black and white; a noise volume repeats because it
    // tiles, and a clamp would smear its last texel across every repeat.
    const auto addressMode = byteTexels ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = addressMode;
    samplerInfo.addressModeV = addressMode;
    samplerInfo.addressModeW = addressMode;
    samplerInfo.maxLod = 0.0f;

    VkSampler sampler = VK_NULL_HANDLE;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");

    const auto id = nextResourceId++;
    imageResources.emplace(id, ImageResource{.image = image,
                                             .allocation = imageAllocation,
                                             .view = view,
                                             .sampler = sampler,
                                             .format = imageInfo.format,
                                             .width = texture.width,
                                             .height = texture.height,
                                             .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                             .layouts = {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                             .levelViews = {}});

    logger.info("Vulkan volume texture {} uploaded: {}x{}x{}", id, texture.width, texture.height, texture.depth);

    return id;
}

unsigned int VulkanRenderer::createSampledImage(const std::span<const Texture* const> faces, const bool cube) const
{
    const auto& first = *faces.front();
    const auto width = first.width;
    const auto height = first.height;
    const auto layerCount = static_cast<uint32_t>(faces.size());

    const auto sourceChannels = static_cast<size_t>(channelCount(first.format));
    const auto componentBytes = static_cast<size_t>(pixelComponentBytes(first.pixelDataType));
    if (sourceChannels == 0)
    {
        throw std::runtime_error("sampled image source has no known channel layout");
    }

    const auto format = sampledImageFormat(first.pixelDataType, first.colourSpace);
    const auto texelBytes = static_cast<VkDeviceSize>(componentBytes * 4);
    const auto texelCount = static_cast<size_t>(width) * height;
    const auto faceBytes = static_cast<VkDeviceSize>(texelCount) * texelBytes;
    const auto sourceFaceBytes = texelCount * sourceChannels * componentBytes;

    for (const auto* face : faces)
    {
        // Colour space joins the agreement check because one image carries every face: the
        // sampler decodes for all six or for none, so a cube whose faces disagree has no
        // representation rather than a merely inconvenient one.
        if (face->width != width || face->height != height || face->pixelDataType != first.pixelDataType ||
            face->format != first.format || face->colourSpace != first.colourSpace ||
            face->data.size() < sourceFaceBytes)
        {
            throw std::runtime_error("sampled image sources disagree on size, pixel type, channel layout or "
                                     "colour space");
        }
    }

    // GL generates cube map mips and the PBR shader samples explicit LODs (roughness *
    // 11), so mips are required for parity; blit-based generation needs linear-filter
    // blit support, which every desktop driver offers for these two formats.
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
    constexpr VkFormatFeatureFlags mipBlitFeatures = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    const auto blitCapable = (formatProperties.optimalTilingFeatures & mipBlitFeatures) == mipBlitFeatures;
    auto mipLevels = 1u;
    if (blitCapable)
    {
        mipLevels = mipLevelCount(width, height);
    }
    else
    {
        diagnostics.record(FrameDiagnostic::MipGenerationUnavailable,
                           [&] { return "VkFormat " + std::to_string(static_cast<int>(format)); });
    }

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = VkExtent3D{width, height, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = layerCount;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imageAllocationInfo{};
    imageAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation imageAllocation = nullptr;
    ensure(vmaCreateImage(allocator, &imageInfo, &imageAllocationInfo, &image, &imageAllocation, nullptr),
           "vmaCreateImage");

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = faceBytes * layerCount;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAllocationCreateInfo{};
    stagingAllocationCreateInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = nullptr;
    VmaAllocationInfo stagingAllocationInfo{};
    ensure(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocationCreateInfo, &stagingBuffer, &stagingAllocation,
                           &stagingAllocationInfo),
           "vmaCreateBuffer");

    // GL lets the driver expand a source with fewer than four components, filling the missing
    // colour components with 0 and a missing alpha with 1. Vulkan has no such conversion, so
    // the same rule is applied here, at the source's own precision.
    std::array<unsigned char, 4> opaqueComponent{};
    if (first.pixelDataType == PixelDataType::Float)
    {
        const auto one = 1.0f;
        std::memcpy(opaqueComponent.data(), &one, sizeof(one));
    }
    else
    {
        opaqueComponent.fill(0xFF);
    }

    for (size_t faceIndex = 0; faceIndex < faces.size(); faceIndex++)
    {
        const auto& face = *faces[faceIndex];
        auto* destination =
            static_cast<unsigned char*>(stagingAllocationInfo.pMappedData) + faceIndex * static_cast<size_t>(faceBytes);

        if (sourceChannels == 4)
        {
            std::memcpy(destination, face.data.data(), sourceFaceBytes);
            continue;
        }

        // Source rows are tightly packed (stbi and tinygltf both emit them that way).
        for (size_t texel = 0; texel < texelCount; texel++)
        {
            const auto* source = face.data.data() + texel * sourceChannels * componentBytes;
            auto* target = destination + texel * componentBytes * 4;

            std::memcpy(target, source, sourceChannels * componentBytes);
            std::memset(target + sourceChannels * componentBytes, 0, (3 - sourceChannels) * componentBytes);
            std::memcpy(target + 3 * componentBytes, opaqueComponent.data(), componentBytes);
        }
    }
    ensure(vmaFlushAllocation(allocator, stagingAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

    const auto commandBuffer = beginUploadCommands();
    const VkImageSubresourceRange allSubresources{VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, layerCount};

    transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, allSubresources);

    std::vector<VkBufferImageCopy> regions(faces.size());
    for (uint32_t faceIndex = 0; faceIndex < layerCount; faceIndex++)
    {
        regions[faceIndex].bufferOffset = faceBytes * faceIndex;
        regions[faceIndex].imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, faceIndex, 1};
        regions[faceIndex].imageExtent = VkExtent3D{width, height, 1};
    }
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()), regions.data());

    if (mipLevels > 1)
    {
        auto mipWidth = static_cast<int32_t>(width);
        auto mipHeight = static_cast<int32_t>(height);

        for (uint32_t level = 1; level < mipLevels; level++)
        {
            const VkImageSubresourceRange sourceLevel{VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, layerCount};
            transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_ACCESS_2_TRANSFER_READ_BIT, sourceLevel);

            const auto nextWidth = std::max(mipWidth / 2, 1);
            const auto nextHeight = std::max(mipHeight / 2, 1);

            VkImageBlit blit{};
            blit.srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, layerCount};
            blit.srcOffsets[1] = VkOffset3D{mipWidth, mipHeight, 1};
            blit.dstSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, level, 0, layerCount};
            blit.dstOffsets[1] = VkOffset3D{nextWidth, nextHeight, 1};
            vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            mipWidth = nextWidth;
            mipHeight = nextHeight;
        }

        const VkImageSubresourceRange blittedLevels{VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels - 1, 0, layerCount};
        transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, blittedLevels);

        const VkImageSubresourceRange lastLevel{VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, layerCount};
        transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, lastLevel);
    }
    else
    {
        transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, allSubresources);
    }

    finishUploadCommands(commandBuffer);
    retireUploadStaging(stagingBuffer, stagingAllocation);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = allSubresources;

    VkImageView view = VK_NULL_HANDLE;
    ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

    // LINEAR with trilinear mips, matching the GL texture parameters. Cube maps clamp so a
    // face never samples across a seam; 2D textures repeat, which is the glTF sampler default
    // and what KHR_texture_transform scaling relies on once a UV leaves 0..1. GL's anisotropy
    // is applied to 2D textures only, and this matches that.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    const auto addressMode = cube ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeU = addressMode;
    samplerInfo.addressModeV = addressMode;
    samplerInfo.addressModeW = addressMode;
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    if (!cube && samplerAnisotropySupported)
    {
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = deviceLimits.maxSamplerAnisotropy;
    }

    VkSampler sampler = VK_NULL_HANDLE;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");

    const auto id = nextResourceId++;
    imageResources.emplace(
        id, ImageResource{.image = image,
                          .allocation = imageAllocation,
                          .view = view,
                          .sampler = sampler,
                          .format = format,
                          .mipLevels = mipLevels,
                          .width = width,
                          .height = height,
                          .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                          .layouts = std::vector<VkImageLayout>(mipLevels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                          .levelViews = {}});
    return id;
}

std::expected<unsigned int, std::string> VulkanRenderer::createCubeMap(const Texture& front, const Texture& back,
                                                                       const Texture& left, const Texture& right,
                                                                       const Texture& top, const Texture& bottom)
{
    try
    {
        // GL uploads {right,left,bottom,top,front,back} onto +X..-Z with bottom deliberately
        // on POSITIVE_Y; Vulkan cube layers 0..5 are +X..-Z under the same sampling
        // convention, so the identical assignment preserves lookup parity (V4 adjudicates).
        const std::array faces = {&right, &left, &bottom, &top, &front, &back};
        const auto id = createSampledImage(faces, true);

        // The faces' pixel data is CubeMapService's to drop, through the handles it holds; this
        // used to const_cast the const Texture& it was handed and empty it here.
        logger.info("Vulkan cube map {} uploaded: 6 faces {}x{}, {} mip levels", id, right.width, right.height,
                    imageResources.at(id).mipLevels);
        return id;
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

// The image leaves the registry now — nothing may name it again — and is destroyed later, when
// the frames that could still be sampling it have completed. See retire().
VkSampler VulkanRenderer::createComparisonSampler(const VkFormat format) const
{
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
    const auto filter =
        (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0
            ? VK_FILTER_LINEAR
            : VK_FILTER_NEAREST;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = filter;
    samplerInfo.minFilter = filter;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Clamped to an opaque white border rather than to the edge, and GL's comparison attachments
    // are given the same border: a lookup outside the map must read "nothing was in front of
    // this", which for a depth comparison is the far value, and clamping to the edge would smear
    // whatever the border texel happens to hold across everything beyond the target.
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.compareEnable = VK_TRUE;
    samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkSampler sampler = VK_NULL_HANDLE;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");

    return sampler;
}

void VulkanRenderer::destroyImageResource(const unsigned int id) const
{
    const auto entry = imageResources.find(id);
    if (entry == imageResources.end())
    {
        return;
    }

    RetiredResource retired;

    // A cached fullscreen set names up to postProcessInputCount views, so one destroyed image
    // invalidates every set that mentions it — recycled with the image and on the same schedule,
    // because a descriptor set a submitted command buffer bound may not be freed either. The pool
    // carries FREE_DESCRIPTOR_SET for this.
    std::erase_if(attachmentSets,
                  [&](const auto& cached)
                  {
                      const auto names = std::ranges::any_of(cached.first.inputs, [&](const PostProcessBinding& binding)
                                                             { return binding.gpuResourceId == id; }) ||
                                         cached.first.lookupTable == id ||
                                         std::ranges::find(cached.first.volumes, id) != cached.first.volumes.end();
                      if (!names)
                      {
                          return false;
                      }

                      RetiredResource retiredSet;
                      retiredSet.descriptorSet = cached.second;
                      retire(retiredSet);
                      return true;
                  });

    // A cascade set names up to shadowCascadeCount views and the three images beside them — the
    // occlusion, the behind copy and the cloud map — so one destroyed image invalidates every set
    // that mentions it, on the same schedule and for the same reason as the fullscreen set above.
    // The next frame that asks for a set with a live tuple builds a new one.
    std::erase_if(shadowSets,
                  [&](const auto& entry)
                  {
                      if (std::ranges::find(entry.first.cascades, id) == entry.first.cascades.end() &&
                          entry.first.occlusion != id && entry.first.behind != id && entry.first.cloudMap != id)
                      {
                          return false;
                      }

                      RetiredResource retiredSet;
                      retiredSet.descriptorSet = entry.second;
                      retire(retiredSet);

                      if (entry.second == dummyShadowSet)
                      {
                          dummyShadowSet = VK_NULL_HANDLE;
                      }

                      return true;
                  });

    // The behind copy shadows its colour attachment's lifetime: an attachment rebuilt by a resize
    // arrives with a new id and takes a fresh copy, so one keyed by the old id would never be
    // asked for again and would hold a window-sized chain forever.
    if (const auto behindEntry = sceneBehindImages.find(id); behindEntry != sceneBehindImages.end())
    {
        const auto behindId = behindEntry->second;
        sceneBehindImages.erase(behindEntry);
        destroyImageResource(behindId);
    }

    const auto& resource = entry->second;

    // A chain's per-level views go on the same queue as everything else it owns: the frame that
    // rendered through one of them may still be executing, so an eager vkDestroyImageView here is
    // the same defect deferral exists to prevent. One entry each — RetiredResource carries a
    // single object of each kind, which is what keeps collectRetiredResources a flat sweep.
    for (const auto view : resource.levelViews)
    {
        RetiredResource retiredView;
        retiredView.view = view;
        retire(retiredView);
    }

    retired.sampler = resource.sampler;
    retired.view = resource.view;
    retired.image = resource.image;
    retired.imageAllocation = resource.allocation;

    imageResources.erase(entry);
    retire(retired);
}

std::expected<unsigned int, std::string> VulkanRenderer::createFbo(const Fbo& fbo)
{
    try
    {
        FboResource fboResource;
        auto loggedWidth = 0u;
        auto loggedHeight = 0u;

        for (const auto& attachmentKey : fbo.attachments)
        {
            const auto* attachmentSlot = memoryStorageService.bufferAttachments.find(attachmentKey);
            if (attachmentSlot == nullptr)
            {
                continue;
            }

            const auto& attachment = *attachmentSlot;

            VkImageUsageFlags usage;
            VkImageAspectFlags aspect;
            switch (attachment.type)
            {
            // TRANSFER_SRC beside the two that were always there: a colour attachment is also the
            // thing the CPU occasionally needs one texel of, which is what the exposure meter's
            // copy-back is. Stated for every colour target rather than only for a chain, because
            // the usage a level of an image carries is the image's and a meter is not the last
            // pass that will want to read one back.
            case FboAttachmentType::Color:
                usage =
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                break;
            case FboAttachmentType::Depth:
                usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                break;
            // Same reading as an internalFormat with no Vulkan equivalent below: a model this
            // backend cannot serve, not a runtime condition. Skipping it built a framebuffer
            // that was missing an attachment its owner had asked for and said so once.
            case FboAttachmentType::Stencil:
                throw std::runtime_error("Vulkan framebuffers carry Color and Depth attachments only; "
                                         "FboAttachmentType::Stencil has no equivalent here");
            }

            // The initial clear below is a transfer write, which the usage has to declare.
            if (attachment.type == FboAttachmentType::Color && attachment.initialColour.has_value())
            {
                usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }

            // GL honours FboAttachment::internalFormat; so does this. An internalFormat with no
            // Vulkan equivalent is a model the backend cannot serve, not a runtime condition.
            const auto requestedFormat = attachmentFormat(attachment.internalFormat);
            if (!requestedFormat.has_value())
            {
                throw std::runtime_error("FboAttachment::internalFormat " +
                                         std::to_string(static_cast<int>(attachment.internalFormat)) +
                                         " has no Vulkan format");
            }
            const auto format = requestedFormat.value();

            // A minimised window reports 0x0; a 1x1 image keeps the resource valid until the
            // next real resize arrives.
            const auto width = std::max(attachment.width, 1u);
            const auto height = std::max(attachment.height, 1u);
            loggedWidth = width;
            loggedHeight = height;

            // A chain no longer than the size supports, and at least one level: an attachment
            // asking for more levels than halving can produce is asking for a subresource that
            // cannot exist, and clamping is what the resize path needs anyway — a chain buffer
            // rebuilt at a smaller window is shorter than the one it replaces.
            const auto levels = std::clamp(attachment.levels, 1u, mipLevelCount(width, height));

            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = VkExtent3D{width, height, 1};
            imageInfo.mipLevels = levels;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = usage;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = nullptr;
            ensure(vmaCreateImage(allocator, &imageInfo, &allocationCreateInfo, &image, &allocation, nullptr),
                   "vmaCreateImage");

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange = VkImageSubresourceRange{aspect, 0, levels, 0, 1};

            VkImageView view = VK_NULL_HANDLE;
            ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

            // Only a depth attachment that asked for it: a comparison sampler bound where the
            // shader declares a plain sampler is undefined, so this is opt-in on both backends.
            VkSampler sampler = VK_NULL_HANDLE;
            if (attachment.type == FboAttachmentType::Depth &&
                attachment.depthComparison == DepthComparison::LessOrEqual)
            {
                sampler = createComparisonSampler(format);
            }

            const auto attachmentId = nextResourceId++;
            imageResources.emplace(
                attachmentId, ImageResource{.image = image,
                                            .allocation = allocation,
                                            .view = view,
                                            .sampler = sampler,
                                            .format = format,
                                            .mipLevels = levels,
                                            .width = width,
                                            .height = height,
                                            .aspect = aspect,
                                            .layouts = std::vector<VkImageLayout>(levels, VK_IMAGE_LAYOUT_UNDEFINED),
                                            .levelViews = {}});

            // A single-level attachment's whole-image view already is its level 0, so nothing is
            // allocated for the case every camera and post-process target is. A chain gets one view
            // per level, which is both what a pass renders through and what a descriptor names to
            // read that level alone.
            if (levels > 1)
            {
                auto& resource = imageResources.at(attachmentId);
                resource.levelViews.reserve(levels);
                for (auto level = 0u; level < levels; level++)
                {
                    resource.levelViews.push_back(createLevelView(attachmentId, level, 0));
                }
            }

            // The opt-in first fill (FboAttachment::initialColour): cleared through the synchronous
            // upload path because the image must be defined before any frame is — frame 0's probe
            // captures run before the chain that will write this, and would otherwise sample it in
            // UNDEFINED layout. Left in SHADER_READ_ONLY, and the tracked layouts say so, which is
            // what makes the first real pass's transition out of it truthful. A resize re-runs this
            // whole function over the same record, so a rebuilt attachment is re-cleared for free.
            if (attachment.type == FboAttachmentType::Color && attachment.initialColour.has_value())
            {
                const auto& colour = attachment.initialColour.value();
                const VkImageSubresourceRange whole{aspect, 0, levels, 0, 1};
                const auto commandBuffer = beginUploadCommands();
                transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT,
                                VK_ACCESS_2_TRANSFER_WRITE_BIT, whole);
                const VkClearColorValue clearValue{{colour.x, colour.y, colour.z, colour.w}};
                vkCmdClearColorImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1,
                                     &whole);
                transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT, whole);
                finishUploadCommands(commandBuffer);

                imageResources.at(attachmentId).layouts.assign(levels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }

            fboResource.attachmentIds.push_back(attachmentId);

            // Observable contract shared with GL: every attachment's id is written back
            // through its Resource.
            memoryStorageService.bufferAttachments.mutate(attachmentKey, [&](FboAttachment& target)
                                                          { target.gpuResourceId = attachmentId; });
        }

        const auto id = nextResourceId++;
        const auto attachmentCount = fboResource.attachmentIds.size();
        fboResources.emplace(id, std::move(fboResource));

        logger.info("Vulkan framebuffer {} ready: {} attachment(s), {}x{}", id, attachmentCount, loggedWidth,
                    loggedHeight);
        return id;
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

// No device idle any more. This used to stall the whole device on every resize, which was
// tolerable at resize rate and would not be at unload rate — an FBO is also what a camera owns,
// so unloading a scene deletes several. The retirement queue is the in-flight guard now, and it
// costs the images two more frames of residency instead of the GPU a full drain.
void VulkanRenderer::deleteFbo(Fbo& fbo)
{
    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    for (const auto& attachmentKey : fbo.attachments)
    {
        memoryStorageService.bufferAttachments.mutate(attachmentKey,
                                                      [&](FboAttachment& attachment)
                                                      {
                                                          if (!attachment.gpuResourceId.has_value())
                                                          {
                                                              return;
                                                          }

                                                          destroyImageResource(attachment.gpuResourceId.value());
                                                          attachment.gpuResourceId.reset();
                                                      });
    }

    if (fbo.gpuResourceId.has_value())
    {
        fboResources.erase(fbo.gpuResourceId.value());
        fbo.gpuResourceId.reset();
    }
}

void VulkanRenderer::releaseGpuResource(const GpuResourceKind kind, const unsigned int gpuResourceId)
{
    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    switch (kind)
    {
    case GpuResourceKind::Buffer:
    {
        const auto entry = bufferResources.find(gpuResourceId);
        if (entry == bufferResources.end())
        {
            return;
        }

        RetiredResource retired;
        retired.buffer = entry->second.buffer;
        retired.bufferAllocation = entry->second.allocation;
        bufferResources.erase(entry);
        retire(retired);
        break;
    }
    case GpuResourceKind::Texture:
    case GpuResourceKind::CubeMap:
        destroyImageResource(gpuResourceId);
        break;
    case GpuResourceKind::VertexArray:
        // The Vulkan counterpart of a VAO owns no Vulkan object: it is the resolved vertex input
        // plus copies of buffer handles the model owns and a pipeline the cache owns. Dropping the
        // entry is the whole release, and it has to happen before the buffers it names.
        primitiveBindings.erase(gpuResourceId);
        break;
    case GpuResourceKind::ShaderProgram:
        releaseShaderObject(gpuResourceId);
        break;
    case GpuResourceKind::FrameBuffer:
        fboResources.erase(gpuResourceId);
        break;
    }
}

// A shader owns its two modules and every pipeline built from them: the swapchain-target one, the
// offscreen ones keyed by target format, and the scene ones keyed by shader, cull mode, topology,
// formats and vertex-input signature — all of which start with this shader's id.
void VulkanRenderer::releaseShaderObject(const unsigned int shaderId)
{
    auto destroyedPipelines = std::vector<VkPipeline>();

    const auto entry = shaderObjects.find(shaderId);
    if (entry != shaderObjects.end())
    {
        if (entry->second.swapchainTargetPipeline != VK_NULL_HANDLE)
        {
            destroyedPipelines.push_back(entry->second.swapchainTargetPipeline);
        }

        RetiredResource vertexModule;
        vertexModule.shaderModule = entry->second.vertexModule;
        retire(vertexModule);

        RetiredResource fragmentModule;
        fragmentModule.shaderModule = entry->second.fragmentModule;
        retire(fragmentModule);

        shaderObjects.erase(entry);
    }

    std::erase_if(offscreenPipelines,
                  [&](const auto& pipeline)
                  {
                      if ((pipeline.first >> 32u) != shaderId)
                      {
                          return false;
                      }

                      destroyedPipelines.push_back(pipeline.second);

                      return true;
                  });

    const auto scenePrefix = std::to_string(shaderId) + "|";
    std::erase_if(scenePipelines,
                  [&](const auto& pipeline)
                  {
                      if (!pipeline.first.starts_with(scenePrefix))
                      {
                          return false;
                      }

                      destroyedPipelines.push_back(pipeline.second);

                      return true;
                  });

    // A primitive binding caches what it resolved to, one entry per shader and target shape it has
    // been drawn with. Any entry naming one of these pipelines has to resolve again — a model still
    // drawing with a released shader is a caller bug, but it must not be a dangling VkPipeline —
    // and so does any entry naming the released shader id, whose pipeline may never have built:
    // shader ids are handed out from a counter this release returns nothing to, but an entry that
    // outlived its shader is a cache of an answer nobody can check.
    for (auto& [id, binding] : primitiveBindings)
    {
        std::erase_if(binding.pipelines,
                      [&](const ResolvedPipeline& resolved)
                      {
                          return resolved.shaderId == shaderId ||
                                 std::ranges::find(destroyedPipelines, resolved.pipeline) != destroyedPipelines.end();
                      });
    }

    for (const auto pipeline : destroyedPipelines)
    {
        RetiredResource retired;
        retired.pipeline = pipeline;
        retire(retired);
    }
}

// Every descriptor set and uniform buffer this backend cached against the material's slot, for
// whatever environment it was bound with. The slot is about to be reused by a different material.
void VulkanRenderer::releaseMaterial(const Resource<Material>& material)
{
    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    const auto slot = static_cast<uint64_t>(material.index);

    std::erase_if(materialResources,
                  [&](const auto& entry)
                  {
                      if ((entry.first >> 32u) != slot)
                      {
                          return false;
                      }

                      RetiredResource retired;
                      retired.buffer = entry.second.buffer;
                      retired.bufferAllocation = entry.second.allocation;
                      retired.descriptorSet = entry.second.set;
                      retire(retired);

                      return true;
                  });
}

GpuResourceCensus VulkanRenderer::gpuResourceCensus() const
{
    return GpuResourceCensus{.buffers = static_cast<unsigned int>(bufferResources.size()),
                             .textures = static_cast<unsigned int>(imageResources.size()),
                             .vertexArrays = static_cast<unsigned int>(primitiveBindings.size()),
                             .shaderPrograms = static_cast<unsigned int>(shaderObjects.size()),
                             .frameBuffers = static_cast<unsigned int>(fboResources.size())};
}

std::expected<void, std::string> VulkanRenderer::captureFrame(const std::string& path)
{
    try
    {
        // Engine calls this after endFrame presented, so no frame is open here: the capture
        // opens one of its own below.
        recreateSwapchainIfNeeded();
        if (swapchain == VK_NULL_HANDLE)
        {
            return std::unexpected("there is no swapchain to read back from (window minimised?)");
        }

        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        VmaAllocationInfo allocationInfo{};
        auto width = 0u;
        auto height = 0u;
        VkDeviceSize byteCount = 0;

        // A presented swapchain image may not be touched until it is reacquired, so the
        // capture acquires a fresh image and replays the presenter's pass onto it: the HDR
        // attachment it samples is a persistent image that still holds the presented frame,
        // which makes the readback identical rather than merely similar.
        auto presented = false;
        for (auto attempt = 0; attempt < 2 && !presented; attempt++)
        {
            // The replay is the same simulated instant as the frame it replays.
            if (!beginFrame(frameSimulationTime))
            {
                continue;
            }

            if (buffer == VK_NULL_HANDLE)
            {
                width = swapchainExtent.width;
                height = swapchainExtent.height;
                byteCount = static_cast<VkDeviceSize>(width) * height * 4;

                VkBufferCreateInfo bufferInfo{};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = byteCount;
                bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VmaAllocationCreateInfo allocationCreateInfo{};
                allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
                allocationCreateInfo.flags =
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                ensure(vmaCreateBuffer(allocator, &bufferInfo, &allocationCreateInfo, &buffer, &allocation,
                                       &allocationInfo),
                       "vmaCreateBuffer");
            }

            if (lastPresentPass.has_value())
            {
                recordPresentPass(lastPresentPass.value().shaderId, lastPresentPass.value().attachmentImageId,
                                  lastPresentPass.value().parameters, lastPresentPass.value().lookupTableImageId);
            }

            presented = submitAndPresent(buffer);
        }
        ensure(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");

        if (!presented || buffer == VK_NULL_HANDLE)
        {
            if (buffer != VK_NULL_HANDLE)
            {
                vmaDestroyBuffer(allocator, buffer, allocation);
            }

            return std::unexpected("no swapchain image could be acquired to replay the present pass onto");
        }

        ensure(vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation");

        auto pixels = std::vector<unsigned char>(static_cast<size_t>(byteCount));
        std::memcpy(pixels.data(), allocationInfo.pMappedData, pixels.size());
        vmaDestroyBuffer(allocator, buffer, allocation);

        if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM || surfaceFormat.format == VK_FORMAT_B8G8R8A8_SRGB)
        {
            for (size_t pixelStart = 0; pixelStart < pixels.size(); pixelStart += 4)
            {
                std::swap(pixels[pixelStart], pixels[pixelStart + 2]);
            }
        }

        // Vulkan images are already top-down; rows go straight out.
        const auto rowBytes = static_cast<int>(width) * 4;
        if (stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 4, pixels.data(),
                           rowBytes) == 0)
        {
            return std::unexpected("stb_image_write could not write the frame dump to " + path);
        }

        logger.info("Frame dump written to {}", path);
        return {};
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

// Every colour attachment, as the numbers the pass wrote rather than as a picture.
//
// The file is a one-line ASCII header and then width*height*components little-endian float32, rows
// top-down, which is the order Vulkan already stores them in. Deliberately not a PNG: eight bits
// cannot hold a view depth in world units, and the moment a readback has to be decoded through the
// tone curve it stops being a measurement.
std::expected<void, std::string> VulkanRenderer::captureBuffers(const std::string& pathPrefix)
{
    try
    {
        if (device == VK_NULL_HANDLE)
        {
            return std::unexpected("there is no device to read attachments back from");
        }

        // The copies below read images this frame's passes wrote, so nothing may still be running.
        ensure(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");

        auto manifest = std::string("# framebuffer colour attachments, level 0\n"
                                    "# file: <prefix>-fbo<fbo>-att<attachment>.raw\n"
                                    "# each: ASCII header line 'RAWBUF <w> <h> <components>\\n' then float32 data\n");
        auto written = 0u;

        for (const auto& [fboId, fbo] : fboResources)
        {
            for (const auto attachmentId : fbo.attachmentIds)
            {
                const auto entry = imageResources.find(attachmentId);
                if (entry == imageResources.end())
                {
                    continue;
                }

                // Held by value: the transitions below write into imageResources, and a reference
                // into a container the loop mutates is the kind of borrow this codebase's storage
                // rules exist to stop.
                const auto image = entry->second;
                if (image.aspect != VK_IMAGE_ASPECT_COLOR_BIT || image.width == 0 || image.height == 0 ||
                    image.layouts.empty())
                {
                    continue;
                }

                const auto layout = readbackLayout(image.format);
                if (layout.components == 0)
                {
                    manifest += "fbo" + std::to_string(fboId) + "-att" + std::to_string(attachmentId) +
                                ": skipped, VkFormat " + std::to_string(static_cast<int>(image.format)) +
                                " is not decoded by this readback\n";

                    continue;
                }

                const auto texelCount = static_cast<size_t>(image.width) * image.height;
                const auto byteCount = static_cast<VkDeviceSize>(texelCount) * layout.texelBytes;

                VkBufferCreateInfo bufferInfo{};
                bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferInfo.size = byteCount;
                bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VmaAllocationCreateInfo allocationCreateInfo{};
                allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
                allocationCreateInfo.flags =
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                VkBuffer buffer = VK_NULL_HANDLE;
                VmaAllocation allocation = nullptr;
                VmaAllocationInfo allocationInfo{};
                ensure(vmaCreateBuffer(allocator, &bufferInfo, &allocationCreateInfo, &buffer, &allocation,
                                       &allocationInfo),
                       "vmaCreateBuffer");

                // Put the level back where the frame left it. Every colour attachment already
                // carries TRANSFER_SRC — the exposure meter's copy-back made that unconditional —
                // so nothing about image creation has to change for this. UNDEFINED is not a layout
                // a barrier may target, so a level that was never written stays where it lands.
                const auto previousLayout = image.layouts.front();
                const auto commandBuffer = beginUploadCommands();
                transitionTrackedLevel(commandBuffer, attachmentId, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

                VkBufferImageCopy region{};
                region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = VkExtent3D{image.width, image.height, 1};
                vkCmdCopyImageToBuffer(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                                       &region);

                if (previousLayout != VK_IMAGE_LAYOUT_UNDEFINED)
                {
                    transitionTrackedLevel(commandBuffer, attachmentId, 0, previousLayout);
                }

                finishUploadCommands(commandBuffer);
                // A readback: the CPU reads the mapping next, so the batch cannot stay lazy here.
                flushUploadCommands();
                ensure(vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation");

                auto values = std::vector<float>(texelCount * layout.components);
                const auto* bytes = static_cast<const unsigned char*>(allocationInfo.pMappedData);
                for (size_t index = 0; index < values.size(); index++)
                {
                    switch (layout.component)
                    {
                    case ReadbackComponent::UnsignedByte:
                        // Normalised on the way out, because UNORM is what the sampler would have
                        // given the shader and the shader is what this is being compared against.
                        values[index] = static_cast<float>(bytes[index]) / 255.0f;
                        break;
                    case ReadbackComponent::Half:
                    {
                        uint16_t half = 0;
                        std::memcpy(&half, bytes + index * sizeof(uint16_t), sizeof(half));
                        values[index] = halfToFloat(half);
                        break;
                    }
                    case ReadbackComponent::Float:
                        std::memcpy(&values[index], bytes + index * sizeof(float), sizeof(float));
                        break;
                    }
                }

                vmaDestroyBuffer(allocator, buffer, allocation);

                const auto path = pathPrefix + "-fbo" + std::to_string(fboId) + "-att" + std::to_string(attachmentId) +
                                  ".raw";
                auto* file = std::fopen(path.c_str(), "wb");
                if (file == nullptr)
                {
                    return std::unexpected("could not open " + path + " for writing");
                }

                const auto header = "RAWBUF " + std::to_string(image.width) + " " + std::to_string(image.height) +
                                    " " + std::to_string(layout.components) + "\n";
                const auto headerWritten = std::fwrite(header.data(), 1, header.size(), file) == header.size();
                const auto dataWritten =
                    std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size();
                std::fclose(file);

                if (!headerWritten || !dataWritten)
                {
                    return std::unexpected("could not write the whole of " + path);
                }

                manifest += "fbo" + std::to_string(fboId) + "-att" + std::to_string(attachmentId) + ": " +
                            std::to_string(image.width) + "x" + std::to_string(image.height) + ", " +
                            std::to_string(layout.components) + " component(s), VkFormat " +
                            std::to_string(static_cast<int>(image.format)) + "\n";
                written++;
            }
        }

        const auto manifestPath = pathPrefix + "-manifest.txt";
        auto* manifestFile = std::fopen(manifestPath.c_str(), "wb");
        if (manifestFile == nullptr)
        {
            return std::unexpected("could not open " + manifestPath + " for writing");
        }

        const auto manifestWritten = std::fwrite(manifest.data(), 1, manifest.size(), manifestFile) == manifest.size();
        std::fclose(manifestFile);

        if (!manifestWritten)
        {
            return std::unexpected("could not write the whole of " + manifestPath);
        }

        logger.info("Attachment dump: {} buffer(s) beside {}, listed in {}", written, pathPrefix, manifestPath);

        return {};
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

} // namespace raceengine
