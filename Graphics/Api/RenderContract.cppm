module;

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

export module raceengine.graphics:RenderContract;

import :GraphicsApi;
import raceengine.graphics.models;

namespace raceengine
{

// Everything both backends and every shader of both dialects must agree on. C++ is the
// single origin: the GLSL side never spells one of these numbers, it receives them as
// preprocessor macros (shaderContractMacros below), so a divergence cannot be authored.

export inline constexpr uint32_t maxJoints = 128;

// Forward shading loops every light per fragment and spends one vec3 interpolant per light
// on the direction. Four keeps the vertex stage inside the 60 varying components desktop GL
// guarantees (12 rows of 15 used) and FrameData at 352 bytes.
export inline constexpr uint32_t maxLights = 4;

// The clear colour every colour target of either backend is cleared to.
export inline constexpr std::array<float, 4> clearColour = {1.0f, 1.0f, 1.0f, 1.0f};

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

// Material texture slots in shader-declaration order.
export enum class MaterialTextureSlot : uint32_t {
    Diffuse,
    Normal,
    Specular,
    Emissive,
    Occlusion,
    Environment,
};

export inline constexpr uint32_t materialTextureSlotCount = 6;

// GL binds slot N to texture unit N. Vulkan's set 1 carries the MaterialData UBO at binding
// 0, so the same slot is binding N + 1 — the one-off that made the two hand-maintained lists
// drift is now the only place it exists.
export [[nodiscard]] constexpr uint32_t textureBinding(const MaterialTextureSlot slot, const GraphicsApi api)
{
    return static_cast<uint32_t>(slot) + (api == GraphicsApi::Vulkan ? 1u : 0u);
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
// The cascades get a set of their own rather than a tail of set 1: they are per *frame*, and
// writing them into every material set would mean rewriting every material set whenever a cascade
// target is rebuilt.
export inline constexpr uint32_t shadowDescriptorSet = 3;

// GL has one texture-unit namespace for the whole program, so the cascades sit past every material
// slot; Vulkan gives them a set to themselves, so they start at binding 0 there. Eight rather than
// materialTextureSlotCount (6) because bindMaterial hands Material::textures the units above the
// named slots — that vector is empty for every asset in this tree, and a material carrying three
// or more extra textures would be the collision this gap exists to make unlikely.
export inline constexpr uint32_t glShadowMapTextureUnit = 8;

export [[nodiscard]] constexpr uint32_t shadowMapBinding(const GraphicsApi api)
{
    return api == GraphicsApi::Vulkan ? 0u : glShadowMapTextureUnit;
}

export struct ShaderMacro
{
    std::string_view name;
    uint32_t value;
};

export inline constexpr size_t shaderContractMacroCount = 27;

export [[nodiscard]] constexpr std::array<ShaderMacro, shaderContractMacroCount>
shaderContractMacros(const GraphicsApi api)
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
        {"TEXTURE_DIFFUSE", textureBinding(MaterialTextureSlot::Diffuse, api)},
        {"TEXTURE_NORMAL", textureBinding(MaterialTextureSlot::Normal, api)},
        {"TEXTURE_SPECULAR", textureBinding(MaterialTextureSlot::Specular, api)},
        {"TEXTURE_EMISSIVE", textureBinding(MaterialTextureSlot::Emissive, api)},
        {"TEXTURE_OCCLUSION", textureBinding(MaterialTextureSlot::Occlusion, api)},
        {"TEXTURE_ENVIRONMENT", textureBinding(MaterialTextureSlot::Environment, api)},
        {"SET_FRAME", frameDescriptorSet},
        {"SET_MATERIAL", materialDescriptorSet},
        {"SET_DRAW", drawDescriptorSet},
        {"SET_SHADOW", shadowDescriptorSet},
        {"SHADOW_CASCADES", shadowCascadeCount},
        {"SHADOW_MAP_BINDING", shadowMapBinding(api)},
        {"SHADOW_PCF_RADIUS", shadowPcfRadius},
        {"SHADOW_NORMAL_OFFSET_TEXELS", shadowNormalOffsetTexels},
        {"SHADOW_SLOPE_BIAS_TEXELS", shadowSlopeBiasTexels},
        {"SHADOW_CONSTANT_BIAS_TEXELS", shadowConstantBiasTexels},
        {"SHADOW_MAX_SLOPE", shadowMaxSlope},
        {"SHADOW_BLEND_PERCENT", shadowCascadeBlendPercent},
        {"SHADOW_FADE_PERCENT", shadowDistanceFadePercent},
    }};
}

// GL compiles the source text exactly as handed over, so the contract macros are spliced in
// behind the #version directive that has to stay first. #line restores the authored
// numbering afterwards, otherwise every compile error would point one prologue further down
// than the line it came from. shaderc takes the same macros through AddMacroDefinition and
// needs no splice.
export [[nodiscard]] inline std::string withShaderContractMacros(const std::string& source, const GraphicsApi api)
{
    const auto versionStart = source.find("#version");
    auto insertAt = size_t{0};

    if (versionStart != std::string::npos)
    {
        const auto lineEnd = source.find('\n', versionStart);
        insertAt = lineEnd == std::string::npos ? source.size() : lineEnd + 1;
    }

    const auto precedingLines = std::ranges::count(std::string_view(source).substr(0, insertAt), '\n');

    std::string prologue;
    for (const auto& macro : shaderContractMacros(api))
    {
        prologue += "#define ";
        prologue += macro.name;
        prologue += ' ';
        prologue += std::to_string(macro.value);
        prologue += '\n';
    }
    prologue += "#line " + std::to_string(precedingLines + 1) + '\n';

    return source.substr(0, insertAt) + prologue + source.substr(insertAt);
}

} // namespace raceengine
