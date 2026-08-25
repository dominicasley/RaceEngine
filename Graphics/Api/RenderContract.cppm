module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

export module raceengine.graphics:RenderContract;

import raceengine.graphics.models;

namespace raceengine
{

// Everything the backend and every shader must agree on. C++ is the single origin: the GLSL side
// never spells one of these numbers, it receives them as preprocessor macros (shaderContractMacros
// below, handed to shaderc as AddMacroDefinition), so a divergence cannot be authored.

export inline constexpr uint32_t maxJoints = 128;

// Forward shading loops every light per fragment and spends one vec3 interpolant per light
// on the direction. Four keeps the vertex stage inside the 60 varying components desktop GL
// guarantees (12 rows of 15 used) and FrameData at 352 bytes.
export inline constexpr uint32_t maxLights = 4;

// The clear colour every colour target of either backend is cleared to.
export inline constexpr std::array<float, 4> clearColour = {1.0f, 1.0f, 1.0f, 1.0f};

// What a depth-normal prepass clears to instead. Its attachment holds geometry rather than colour —
// a surface normal and a distance in front of the eye — and zero is the value every reader of it
// takes to mean "the camera saw nothing here". Cleared to the white above, the sky would reach the
// occlusion gather as a surface one unit from the eye.
export inline constexpr std::array<float, 4> prepassClearColour = {0.0f, 0.0f, 0.0f, 0.0f};

// How many slices the shadow-casting light's view frustum is split into. Four, because four is
// where the practical split scheme stops buying resolution for this camera's 1..2000 range — the
// nearest slice is already sub-metre per texel at 2048 — and because four split distances ride in
// one vec4, so the frame block gains no ragged tail. Raising it needs a wider split field, which
// the static_assert in the Vulkan backend states.
export inline constexpr uint32_t shadowCascadeCount = 4;

// Percentage-closer filtering: taps at (2r+1)^2 on a one-texel grid. r = 1 gives nine taps, and
// each tap is *already* a 2x2 percentage-closer average because the comparison sampler filters
// linearly — so nine taps cover a 4x4 texel neighbourhood with tent weighting. r = 2 would
// quadruple the cost for a softness that is not visible at this cascade resolution.
export inline constexpr uint32_t shadowPcfRadius = 1;

// The bias budget, in shadow-map texels of the cascade the fragment landed in. Stated here rather
// than in the two dialects because a shadow that acnes on one backend and peter-pans on the other
// is exactly the divergence a hand-copied constant produces. Both are texel counts, so they scale
// with the cascade automatically and no cascade needs its own tuning:
//  - the normal offset moves the sample point off the surface *sideways*, which is what removes
//    acne without detaching the contact point,
//  - the depth bias is the depth a surface at slope tan(theta) crosses over that many texels,
//    clamped so a grazing surface cannot ask for an unbounded one.
export inline constexpr uint32_t shadowNormalOffsetTexels = 3;
export inline constexpr uint32_t shadowSlopeBiasTexels = 3;
export inline constexpr uint32_t shadowConstantBiasTexels = 1;
export inline constexpr uint32_t shadowMaxSlope = 8;

// The last tenth of a cascade cross-fades into the next one, and the last tenth of the shadow
// distance fades to fully lit. Both are the same problem — a discontinuity in filter width and
// bias reads as a line drawn across the ground — and the second is why a fragment past the last
// cascade is lit rather than shadowed.
export inline constexpr uint32_t shadowCascadeBlendPercent = 10;
export inline constexpr uint32_t shadowDistanceFadePercent = 10;

// How many light probes one frame can shade from. Every probe is tested against every fragment —
// the same shape as the light loop, and for the same reason: a per-fragment cull needs a spatial
// structure the frame does not have and would not pay for at this count. Eight is what keeps the
// probe block inside a uniform buffer a driver guarantees the range of (the whole FrameData block
// is then under 2.5 KiB against a 16 KiB minimum) while being enough probes to give a street its
// open stretch, its shadowed side and a couple of interiors.
export inline constexpr uint32_t maxIblProbes = 8;

// The edge length of a probe's captured environment, and how many roughness levels its prefiltered
// chain holds. 128 is where a mirror reflection of a building still reads as a building; six mips
// take that down to 4x4, which is rough enough that the last level is indistinguishable from the
// irradiance the spherical harmonics already hold. The mip count is what maps roughness to a LOD,
// so the shader has to be told it rather than assuming textureQueryLevels.
export inline constexpr uint32_t probeCubeResolution = 128;
export inline constexpr uint32_t probeSpecularMipCount = 6;

// The mip of the captured radiance the irradiance projection reads back. Small on purpose:
// irradiance is bandlimited to the first three spherical harmonic bands, so projecting from a
// 16x16 face costs 1536 texels per probe instead of 98304 and loses nothing the basis could have
// represented. It is a mip index, so probeCubeResolution >> this is that 16.
export inline constexpr uint32_t probeIrradianceSourceMip = 3;

// Material texture slots in shader-declaration order.
//
// The five after Environment are the blended-material feature (Material.cppm, `MaterialBlend`): a
// mask that says where each layer applies, and one texture per layer. They are appended so that
// every existing slot keeps its binding — a slot's index *is* its binding, so inserting anywhere
// above would silently rebind every shader's samplers.
export enum class MaterialTextureSlot : uint32_t {
    Diffuse,
    Normal,
    Specular,
    Emissive,
    Occlusion,
    Environment,
    BlendMask,
    DetailR,
    DetailG,
    DetailB,
    DetailA,
};

export inline constexpr uint32_t materialTextureSlotCount = 6 + 1 + detailLayerCount;

static_assert(static_cast<uint32_t>(MaterialTextureSlot::DetailR) + detailLayerCount == materialTextureSlotCount);

// Set 1 carries the MaterialData UBO at binding 0, so slot N is binding N + 1.
export [[nodiscard]] constexpr uint32_t textureBinding(const MaterialTextureSlot slot)
{
    return static_cast<uint32_t>(slot) + 1u;
}

// Vertex attribute locations are PrimitiveAttributeType's enumerator order. Both backends
// derive the location from the enumerator and every vertex shader gets it as a macro, so
// reordering the enum can no longer silently unbind an attribute.
export [[nodiscard]] constexpr uint32_t attributeLocation(const PrimitiveAttributeType type)
{
    return static_cast<uint32_t>(type);
}

static_assert(attributeLocation(PrimitiveAttributeType::Position) == 0);
static_assert(attributeLocation(PrimitiveAttributeType::TextureCoordinate) == 1);
static_assert(attributeLocation(PrimitiveAttributeType::Normal) == 2);
static_assert(attributeLocation(PrimitiveAttributeType::Tangent) == 3);
static_assert(attributeLocation(PrimitiveAttributeType::Joint) == 4);
static_assert(attributeLocation(PrimitiveAttributeType::SkinWeight) == 5);

// Vulkan descriptor set indices (docs/vulkan-abi.md).
export inline constexpr uint32_t frameDescriptorSet = 0;
export inline constexpr uint32_t materialDescriptorSet = 1;
export inline constexpr uint32_t drawDescriptorSet = 2;

// The skinning palette, at its own binding in the per-draw set rather than as a tail of the draw
// block. Both are per draw and both ride a dynamic offset, so the split is not about *when* they
// change — it is about how big they are. The palette is `maxJoints` mat4s, thirty times the rest of
// the block put together, and a scene where one drawn thing in a thousand is skinned would size its
// whole draw ring on the one that is: a car exported as 808 separate primitives exhausted a
// 512-slot ring in the first cascade and drew nothing else all frame. Split, a draw slot is a
// couple of hundred bytes and the palette ring is sized by the skinned draws alone.
export inline constexpr uint32_t jointDataBinding = 1;
// Car paint, beside the palette in the per-draw set and there for the same reason: it is per draw,
// only some draws want it, and a slot in the draw block would make every draw in the frame pay for
// something a handful of body panels read. It rides the *renderable* rather than the material
// because materials are shared storage — see Paint — so this is the one binding that carries a fact
// about which car is being drawn rather than about what it is made of.
export inline constexpr uint32_t paintDataBinding = 2;
// The cascades get a set of their own rather than a tail of set 1: they are per *frame*, and
// writing them into every material set would mean rewriting every material set whenever a cascade
// target is rebuilt.
export inline constexpr uint32_t shadowDescriptorSet = 3;

// The cascades own set 3 outright, so their sampler array starts at binding 0 of it.
export inline constexpr uint32_t shadowMapBinding = 0;

// Screen-space ambient occlusion for the view being shaded, beside the cascades in set 3 because it
// is the same kind of thing: an image produced earlier in this frame, per camera pass, that only a
// shading view reads. Set 0 could not hold it — that set is one descriptor with a dynamic offset per
// view, and an image binding cannot vary by offset — which is the same reason the cascades are here.
// A view with no occlusion buffer binds a 1x1 white image, so the sample site is unconditional.
export inline constexpr uint32_t ambientOcclusionBinding = 1;

// What was already drawn behind the surface being drawn: a copy of the opaque scene colour, taken
// between the opaque draws and the blended ones, so a blended surface can sample what it is being
// composited over — which the render target itself cannot legally provide while it is being written.
// Beside the cascades and the occlusion for their reason exactly: one image per camera pass,
// produced earlier in this frame, read only by a view that shades. A view that takes no copy binds
// the 1x1 white image, so the sample site is unconditional.
export inline constexpr uint32_t sceneBehindBinding = 2;

// How many levels the behind copy halves down through. The chain is what turns the copy into a
// choice of scattering cone — level 0 is exact refraction, each level up reads a cone twice as
// wide — and seven reaches 1/64th of the frame, which is a halo the width a windscreen's grime
// spreads a low sun over. Past that a level costs a blit and buys a blur wider than the pane.
export inline constexpr uint32_t sceneBehindMipCount = 7;

// How the occlusion is gathered: directions through the pixel, and steps marched along each. Both
// sides count them — the shader marches them and the engine reports them once at bring-up — so they
// are contract numbers rather than literals in the shader. Three by four is the cheap end of GTAO
// and it is noisy on its own; the spatial jitter and the bilateral pass after it are what make that
// noise a texture rather than a pattern.
export inline constexpr uint32_t gtaoSliceCount = 3;
export inline constexpr uint32_t gtaoStepsPerSlice = 4;

// How many steps the volumetric fog's in-scattering march takes between the eye and the surface.
//
// It is a quality number and not a correctness one, which is a property of the quadrature rather
// than a hope: each step is weighted by the *difference* in transmittance across it, so the sum
// telescopes to exactly the analytic in-scattering total whenever the shadow term is one. Fewer
// steps therefore move where a shaft's edge falls and never how much light there is — and the march
// runs per shaded fragment in a forward pass, so this is also the one number that decides what the
// feature costs. Sixteen is where the dither stops reading as banding on this scene.
// docs/volumetric-fog-brief.md has the derivation.
export inline constexpr uint32_t fogMarchSteps = 8;

// How far a bloom chain is allowed to halve. Six levels from half of 1080p is a bottom level about
// thirty texels across, which is a spill the width of the screen; past that the chain is buying
// nothing but a pass per frame, and the levels are small enough that each one costs a fixed
// overhead rather than its area.
export inline constexpr uint32_t bloomMaxLevels = 6;

// Every probe's prefiltered radiance, as one cube-array image: layers 6i..6i+5 are probe i. It
// rides in set 0 beside the frame block rather than in a set of its own because it is exactly as
// per-frame as the block is, and because one image serves every probe — a per-probe binding would
// need a descriptor rewrite whenever a probe re-captured, which is the thing a time-of-day change
// does constantly.
export inline constexpr uint32_t probeSpecularBinding = 1;

// Fullscreen passes — every post-process and the presenter — bind a set of their own, and it
// carries the images they read. One binding holding an array of `postProcessInputCount` combined
// image samplers rather than that many bindings, for exactly the reason the cascades are one
// binding: the shader declares `sampler2D inputs[POST_INPUTS]`, and a resource declared as an
// array must be backed by a binding whose descriptorCount covers its whole length.
//
// Four is what the passes above this one need at once — a colour attachment, the depth beside it,
// a buffer of derived per-pixel data, and one result being composited back in. A pass reads the
// elements it wants and ignores the rest, but every element is written whatever the shader reads:
// a descriptor a pipeline statically uses must be there, and an unwritten element of a descriptor
// array is the same defect as an unbound sampler.
export inline constexpr uint32_t postProcessDescriptorSet = 0;
export inline constexpr uint32_t postProcessInputBinding = 0;
export inline constexpr uint32_t postProcessInputCount = 4;

// The colour grade, beside the inputs in the same set: one three-dimensional combined sampler. It is
// a binding of its own rather than an element of the array above because a descriptor array holds
// one type and this one is a sampler3D. Every fullscreen pass is written a table whether or not its
// shader reads one — the neutral identity when it has none — for the reason the array's unused
// elements get the white dummy: a descriptor a pipeline statically uses has to be there.
export inline constexpr uint32_t lookupTableBinding = 1;
// The identity table the engine builds for a pass with no grade of its own. Two is the smallest
// cube there is and interpolates to exactly the identity everywhere, so there is nothing to gain by
// making it larger.
export inline constexpr uint32_t neutralLookupTableSize = 2;

// Middle grey in the exposed scene-linear domain, and the point the tone curve's contrast term
// pivots about: below it a contrast above one darkens and above it brightens, and at one it does
// neither. Shared because the tone map is written twice — the fragment shader applies it and
// `evaluateToneCurve` states it in C++ — and a hand-copied pivot is how the two would stop being
// the same function.
export inline constexpr float toneGreyPivot = 0.18f;

// What "luminance" means to the exposure meter: Rec. 709's relative luminance weights, and the
// floor its logarithm is taken above. Shared because the reduction takes that logarithm on the GPU
// while `relativeLuminance` states the same weighting in C++ for the tests that pin the meter, and
// because a frame with black in it has to meter as a number rather than as minus infinity on both
// sides of the divide.
export inline constexpr float luminanceWeightRed = 0.2126f;
export inline constexpr float luminanceWeightGreen = 0.7152f;
export inline constexpr float luminanceWeightBlue = 0.0722f;
export inline constexpr float luminanceFloor = 0.0001f;

// The square the exposure meter reduces a frame into before halving it down to the single texel the
// CPU reads. Fixed rather than the window's size, for the reason the probe cube is fixed: a chain's
// length is decided when it is built, and a meter whose last level stopped being one texel because
// the window changed would be metering a corner of the frame. 512 is nine halvings to that texel,
// and its four taps per level still cover 1920x1080 densely at the top.
export inline constexpr uint32_t exposureMeterResolution = 512;

export struct ShaderMacro
{
    std::string_view name;
    uint32_t value;
};

// Contract numbers that are not counts. They go through a list of their own because
// ShaderMacro's value is an integer and there is no constexpr float-to-text in the standard
// library to fold them into it; the compile path formats both lists the same way.
export struct ShaderFloatMacro
{
    std::string_view name;
    float value;
};

export inline constexpr size_t shaderContractMacroCount = 50;

export inline constexpr size_t shaderContractFloatMacroCount = 5;

export [[nodiscard]] constexpr std::array<ShaderFloatMacro, shaderContractFloatMacroCount> shaderContractFloatMacros()
{
    return {{
        {"TONE_GREY_PIVOT", toneGreyPivot},
        {"LUMINANCE_R", luminanceWeightRed},
        {"LUMINANCE_G", luminanceWeightGreen},
        {"LUMINANCE_B", luminanceWeightBlue},
        {"LUMINANCE_FLOOR", luminanceFloor},
    }};
}

export [[nodiscard]] constexpr std::array<ShaderMacro, shaderContractMacroCount> shaderContractMacros()
{
    return {{
        {"MAX_JOINTS", maxJoints},
        {"MAX_LIGHTS", maxLights},
        {"ATTRIBUTE_POSITION", attributeLocation(PrimitiveAttributeType::Position)},
        {"ATTRIBUTE_TEXCOORD", attributeLocation(PrimitiveAttributeType::TextureCoordinate)},
        {"ATTRIBUTE_NORMAL", attributeLocation(PrimitiveAttributeType::Normal)},
        {"ATTRIBUTE_TANGENT", attributeLocation(PrimitiveAttributeType::Tangent)},
        {"ATTRIBUTE_JOINT", attributeLocation(PrimitiveAttributeType::Joint)},
        {"ATTRIBUTE_WEIGHT", attributeLocation(PrimitiveAttributeType::SkinWeight)},
        {"TEXTURE_DIFFUSE", textureBinding(MaterialTextureSlot::Diffuse)},
        {"TEXTURE_NORMAL", textureBinding(MaterialTextureSlot::Normal)},
        {"TEXTURE_SPECULAR", textureBinding(MaterialTextureSlot::Specular)},
        {"TEXTURE_EMISSIVE", textureBinding(MaterialTextureSlot::Emissive)},
        {"TEXTURE_OCCLUSION", textureBinding(MaterialTextureSlot::Occlusion)},
        {"TEXTURE_ENVIRONMENT", textureBinding(MaterialTextureSlot::Environment)},
        {"TEXTURE_BLEND_MASK", textureBinding(MaterialTextureSlot::BlendMask)},
        {"TEXTURE_DETAIL_R", textureBinding(MaterialTextureSlot::DetailR)},
        {"TEXTURE_DETAIL_G", textureBinding(MaterialTextureSlot::DetailG)},
        {"TEXTURE_DETAIL_B", textureBinding(MaterialTextureSlot::DetailB)},
        {"TEXTURE_DETAIL_A", textureBinding(MaterialTextureSlot::DetailA)},
        {"DETAIL_LAYERS", detailLayerCount},
        {"SET_FRAME", frameDescriptorSet},
        {"SET_MATERIAL", materialDescriptorSet},
        {"SET_DRAW", drawDescriptorSet},
        {"JOINT_DATA_BINDING", jointDataBinding},
        {"PAINT_DATA_BINDING", paintDataBinding},
        {"SET_SHADOW", shadowDescriptorSet},
        {"SHADOW_CASCADES", shadowCascadeCount},
        {"SHADOW_MAP_BINDING", shadowMapBinding},
        {"AO_MAP_BINDING", ambientOcclusionBinding},
        {"SCENE_BEHIND_BINDING", sceneBehindBinding},
        {"SCENE_BEHIND_MIPS", sceneBehindMipCount},
        {"GTAO_SLICES", gtaoSliceCount},
        {"GTAO_STEPS", gtaoStepsPerSlice},
        {"SHADOW_PCF_RADIUS", shadowPcfRadius},
        {"SHADOW_NORMAL_OFFSET_TEXELS", shadowNormalOffsetTexels},
        {"SHADOW_SLOPE_BIAS_TEXELS", shadowSlopeBiasTexels},
        {"SHADOW_CONSTANT_BIAS_TEXELS", shadowConstantBiasTexels},
        {"SHADOW_MAX_SLOPE", shadowMaxSlope},
        {"SHADOW_BLEND_PERCENT", shadowCascadeBlendPercent},
        {"SHADOW_FADE_PERCENT", shadowDistanceFadePercent},
        {"MAX_IBL_PROBES", maxIblProbes},
        {"SH_COEFFICIENTS", shCoefficientCount},
        {"PROBE_SPECULAR_BINDING", probeSpecularBinding},
        {"PROBE_SPECULAR_MIPS", probeSpecularMipCount},
        {"PROBE_CUBE_RESOLUTION", probeCubeResolution},
        {"SET_POST_PROCESS", postProcessDescriptorSet},
        {"POST_INPUT_BINDING", postProcessInputBinding},
        {"POST_INPUTS", postProcessInputCount},
        {"LUT_BINDING", lookupTableBinding},
        {"FOG_MARCH_STEPS", fogMarchSteps},
    }};
}

} // namespace raceengine
