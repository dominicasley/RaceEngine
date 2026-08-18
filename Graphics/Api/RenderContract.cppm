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

export struct ShaderMacro
{
    std::string_view name;
    uint32_t value;
};

export inline constexpr size_t shaderContractMacroCount = 17;

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
