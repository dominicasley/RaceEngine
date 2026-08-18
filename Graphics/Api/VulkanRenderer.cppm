module;

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <shaderc/shaderc.hpp>
#include <spdlog/logger.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

export module raceengine.graphics:VulkanRenderer;

import :GraphicsApi;
import :IRenderBackend;
import :RenderContract;
import :RenderableEntityService;
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
// DrawData is ~8.5 KiB per draw; a few hundred draws per frame covers the sandbox scene
// with generous headroom while keeping the ring under 5 MiB per frame in flight.
constexpr uint32_t drawDataRingSlots = 512;
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
constexpr uint32_t descriptorPoolCombinedImageSamplers = 4096;

// Set 0 binding 0 (vulkan-abi.md); std140-compatible, so the C++ layout is the GPU layout.
// A std140 array of these has a 16-byte element alignment, which 64 bytes already satisfies.
struct LightUbo
{
    glm::vec4 position;
    glm::vec4 diffuse;
    glm::vec4 specular;
    glm::vec4 ambientAttenuation;
};

struct FrameDataUbo
{
    glm::mat4 viewMatrix;
    glm::vec4 cameraPosition;
    glm::ivec4 lightCount;
    std::array<LightUbo, maxLights> lights;
};

static_assert(sizeof(LightUbo) == 64);
static_assert(sizeof(FrameDataUbo) == 352);
static_assert(offsetof(FrameDataUbo, lightCount) == 80);
static_assert(offsetof(FrameDataUbo, lights) == 96);

// Set 2 binding 0, dynamic-offset (vulkan-abi.md); ring-buffered per frame in flight.
struct DrawDataUbo
{
    glm::mat4 localToWorld;
    glm::mat4 localToView;
    glm::mat4 localToScreen;
    glm::mat4 normalMatrix;
    glm::ivec4 animated;
    std::array<glm::mat4, maxJoints> jointTransforms;
};

static_assert(sizeof(DrawDataUbo) == 8464);
// The per-draw fill writes the three regions directly into the mapped ring slot; the
// offsets are asserted so the writes cannot drift away from the shader's block layout.
static_assert(offsetof(DrawDataUbo, localToWorld) == 0);
static_assert(offsetof(DrawDataUbo, animated) == 256);
static_assert(offsetof(DrawDataUbo, jointTransforms) == 272);

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
};

static_assert(sizeof(MaterialDataUbo) == 128);
static_assert(offsetof(MaterialDataUbo, textureTransform) == 64);

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
[[nodiscard]] constexpr VkFormat sampledImageFormat(const PixelDataType type)
{
    switch (type)
    {
    case PixelDataType::UnsignedShort:
        return VK_FORMAT_R16G16B16A16_UNORM;
    case PixelDataType::Float:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:
        return VK_FORMAT_R8G8B8A8_UNORM;
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
        return VK_FORMAT_D32_SFLOAT;
    default:
        return std::nullopt;
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

[[nodiscard]] uint32_t mipLevelCount(const uint32_t width, const uint32_t height)
{
    auto levels = 1u;
    for (auto extent = std::max(width, height); extent > 1; extent /= 2)
    {
        levels++;
    }
    return levels;
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

} // namespace

export class VulkanRenderer : public IRenderBackend
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
        VkDescriptorSet frameDataSet = VK_NULL_HANDLE;
        VkDescriptorSet drawDataSet = VK_NULL_HANDLE;
        uint32_t drawDataSlotsUsed = 0;
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
    // attachmentSampler and keep their own field VK_NULL_HANDLE. layout is the tracked
    // current layout: the draw path derives every barrier from it.
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
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
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

    // The Vulkan counterpart of the GL path's per-primitive VAO: everything a draw needs
    // that does not change once the primitive is uploaded. pipeline/boundBuffers resolve
    // on the first draw because they also depend on the material's shader.
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
        bool resolved = false;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::vector<VkBuffer> boundBuffers;
        std::vector<VkDeviceSize> boundOffsets;
    };

    // What the presenter drew last: captureFrame replays exactly this pass onto a freshly
    // acquired image, because the presented one may not be touched again.
    struct PresentPass
    {
        unsigned int shaderId = 0;
        unsigned int attachmentImageId = 0;
    };

    // Declaration order is dependency order; the destructor tears down in reverse.
    spdlog::logger& logger;
    IWindow& window;
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
    VkDescriptorSetLayout frameDataSetLayout = VK_NULL_HANDLE;  // scene set 0
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;   // scene set 1
    VkDescriptorSetLayout drawDataSetLayout = VK_NULL_HANDLE;   // scene set 2, dynamic
    VkDescriptorSetLayout fullscreenSetLayout = VK_NULL_HANDLE; // fullscreen set 0
    VkPipelineLayout scenePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout fullscreenPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkCommandPool uploadCommandPool = VK_NULL_HANDLE;
    VkSampler attachmentSampler = VK_NULL_HANDLE;
    VkDeviceSize drawDataStride = 0;
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
    mutable std::unordered_map<unsigned int, VkDescriptorSet> attachmentSets;
    std::unordered_map<std::string, VkPipeline> scenePipelines;
    // Fullscreen pipelines that render into an offscreen attachment, keyed by shader id and
    // the target's format: the format comes from FboAttachment::internalFormat, which is not
    // known when the shader object is built.
    std::unordered_map<uint64_t, VkPipeline> offscreenPipelines;
    unsigned int dummyTextureId = 0;
    unsigned int dummyCubeMapId = 0;
    BufferResource dummyVertexBuffer{};
    std::optional<unsigned int> sceneEnvironmentImageId;
    // Frame recording state. The frame belongs to Engine::step: beginFrame acquires the
    // image and opens the command buffer, recordView and recordPresent only record into it,
    // and endFrame is the one place that submits and presents.
    bool frameOpen = false;
    bool swapchainPassRecorded = false;
    uint32_t currentImageIndex = 0;
    std::optional<PresentPass> lastPresentPass;
    mutable bool mipGenerationUnavailableLogged = false;
    bool drawDataRingExhaustedLogged = false;
    bool frameDataRingExhaustedLogged = false;
    bool missingCameraOutputLogged = false;
    bool missingMeshUploadLogged = false;
    bool missingMaterialLogged = false;
    bool unsupportedIndexTypeLogged = false;
    bool unsupportedTopologyLogged = false;
    bool unsupportedTextureLayoutLogged = false;
    bool jointLimitLogged = false;
    bool lightLimitLogged = false;
    bool scenePipelineUnavailableLogged = false;
    bool fullscreenPipelineUnavailableLogged = false;
    bool descriptorSetUnavailableLogged = false;
    bool missingPostProcessTargetLogged = false;
    bool drawSummaryLogged = false;

public:
    explicit VulkanRenderer(spdlog::logger& logger, IWindow& window, RenderableEntityService& renderableEntityService,
                            SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService);
    ~VulkanRenderer() override;

    [[nodiscard]] std::expected<void, std::string> init() override;
    void setViewport(int width, int height) override;

    [[nodiscard]] bool beginFrame() override;
    void recordView(Scene& scene, Camera& camera, float delta) override;
    void recordPresent(const Resource<Shader>& shader, const Resource<FboAttachment>& attachment) override;
    void endFrame() override;

    [[nodiscard]] std::expected<unsigned int, std::string>
    createShaderObject(const ShaderDescriptor& shaderDescriptor) override;
    [[nodiscard]] std::expected<unsigned int, std::string> createCubeMap(const Texture& front, const Texture& back,
                                                                         const Texture& left, const Texture& right,
                                                                         const Texture& top,
                                                                         const Texture& bottom) override;
    [[nodiscard]] std::expected<unsigned int, std::string> createFbo(const Fbo& fbo) override;
    void deleteFbo(Fbo& fbo) override;

    [[nodiscard]] std::expected<void, std::string> captureFrame(const std::string& path) override;

private:
    void createInstance();
    void createDebugMessenger();
    void selectPhysicalDevice();
    void createDevice();
    void createAllocator();
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
    void recordDraw(const MeshPrimitive& primitive, const Resource<Shader>& shader, const glm::mat4& entityModelMatrix,
                    const Camera& camera, const std::vector<glm::mat4>& joints, VkFormat colorFormat,
                    VkFormat depthFormat);
    bool recordFullScreenPass(unsigned int sourceImageId, VkImageView targetView, VkExtent2D targetExtent,
                              VkPipeline pipeline);
    bool recordPresentPass(unsigned int shaderId, unsigned int attachmentImageId);
    void transitionTracked(VkCommandBuffer commandBuffer, unsigned int imageId, VkImageLayout newLayout);
    void upload(const Resource<Model>& modelKey);
    [[nodiscard]] BufferResource createDeviceLocalBuffer(const void* data, VkDeviceSize size) const;
    [[nodiscard]] unsigned int uploadMeshBuffer(const MeshBuffer& meshBuffer);
    [[nodiscard]] PrimitiveBinding makePrimitiveBinding(const Model& model, const MeshPrimitive& primitive);
    void uploadMaterialTextures(const Resource<Material>& materialKey);
    [[nodiscard]] std::optional<unsigned int> uploadTexture(const Resource<Texture>& textureKey);
    [[nodiscard]] VkDescriptorSet materialSet(const Resource<Material>& materialKey, unsigned int environmentImageId);
    [[nodiscard]] VkDescriptorSet attachmentSet(unsigned int imageId);
    [[nodiscard]] VkPipeline scenePipeline(unsigned int shaderId, PrimitiveBinding& binding, VkCullModeFlags cullMode,
                                           VkFormat colorFormat, VkFormat depthFormat);
    [[nodiscard]] VkPipeline offscreenPipeline(unsigned int shaderId, VkFormat colorFormat);
    [[nodiscard]] unsigned int dummyTexture();
    [[nodiscard]] unsigned int dummyCubeMap();
    [[nodiscard]] std::optional<std::vector<uint32_t>> compileToSpirv(const std::string& source,
                                                                      shaderc_shader_kind kind, const char* stageName);
    [[nodiscard]] VkShaderModule createShaderModule(const std::vector<uint32_t>& spirv) const;
    [[nodiscard]] VkPipeline buildFullscreenPipeline(VkShaderModule vertexModule, VkShaderModule fragmentModule,
                                                     VkFormat colorFormat) const;
    void createHostVisibleUniformBuffer(VkDeviceSize size, VkBuffer& buffer, VmaAllocation& allocation,
                                        void*& mapped) const;
    [[nodiscard]] VkCommandBuffer beginUploadCommands() const;
    void finishUploadCommands(VkCommandBuffer commandBuffer) const;
    [[nodiscard]] unsigned int createSampledImage(std::span<const Texture* const> faces, bool cube) const;
    void destroyImageResource(unsigned int id) const;
    [[nodiscard]] std::optional<VkDeviceSize> allocateDrawDataSlot();
    [[nodiscard]] std::optional<VkDeviceSize> allocateFrameDataSlot();
};

VulkanRenderer::VulkanRenderer(spdlog::logger& logger, IWindow& window,
                               RenderableEntityService& renderableEntityService,
                               SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService) :
    logger(logger),
    window(window),
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

    for (const auto& [id, image] : imageResources)
    {
        if (image.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, image.sampler, nullptr);
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
    for (const auto layout : {frameDataSetLayout, materialSetLayout, drawDataSetLayout, fullscreenSetLayout})
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
        surface = window.generateVulkanSurface(instance);
        selectPhysicalDevice();
        createDevice();
        createAllocator();
        createSwapchain();
        createFrameResources();
        createDescriptorInfrastructure();
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

    const auto requiredWindowExtensions = window.getRequiredVulkanWindowExtensions();
    // The extension list pointer is GLFW-owned static storage; copy before appending.
    std::vector<const char*> extensions(requiredWindowExtensions.extensions,
                                        requiredWindowExtensions.extensions + requiredWindowExtensions.count);
    std::vector<const char*> layers;
    if (validationLayerEnabled)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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
        extensions.pop_back();
        layers.pop_back();
        instanceCreateInfo.pNext = nullptr;
        instanceCreateInfo.enabledLayerCount = 0;
        instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ensure(vkCreateInstance(&instanceCreateInfo, nullptr, &instance), "vkCreateInstance");
    }
    else
    {
        ensure(createResult, "vkCreateInstance");
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
        if (features13.dynamicRendering == VK_FALSE || features13.synchronization2 == VK_FALSE ||
            features13.shaderDemoteToHelperInvocation == VK_FALSE)
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
        throw std::runtime_error("no Vulkan 1.3 device with dynamicRendering, synchronization2, a swapchain, "
                                 "and graphics+present queues");
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
    // FIFO for vsync parity with the GL path's glfwSwapInterval(1).
    swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
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

    logger.info("Vulkan swapchain created: {}x{}, {} images, format {}, colour space {}, present mode FIFO",
                extent.width, extent.height, actualImageCount, describeFormat(surfaceFormat.format),
                describeColorSpace(surfaceFormat.colorSpace));
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
    const std::array frameBindings = {
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    frameDataSetLayout = makeSetLayout(frameBindings);

    // Scene set 1: MaterialData UBO at binding 0, then one sampler per material texture slot
    // at the binding RenderContract assigns it.
    std::array<VkDescriptorSetLayoutBinding, materialTextureSlotCount + 1> materialBindings{};
    materialBindings[0] =
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    for (uint32_t slot = 0; slot < materialTextureSlotCount; slot++)
    {
        const auto binding = textureBinding(static_cast<MaterialTextureSlot>(slot), GraphicsApi::Vulkan);
        materialBindings[slot + 1] = VkDescriptorSetLayoutBinding{binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                                  VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    }
    materialSetLayout = makeSetLayout(materialBindings);

    // Scene set 2: dynamic-offset DrawData UBO, ring-buffered per frame in flight.
    const std::array drawBindings = {VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                                                  VK_SHADER_STAGE_VERTEX_BIT, nullptr}};
    drawDataSetLayout = makeSetLayout(drawBindings);

    // Fullscreen passes use their own single-set layout: one combined sampler.
    const std::array fullscreenBindings = {VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
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

    const std::array sceneSetLayouts = {frameDataSetLayout, materialSetLayout, drawDataSetLayout};
    scenePipelineLayout = makePipelineLayout(sceneSetLayouts);
    const std::array fullscreenSetLayouts = {fullscreenSetLayout};
    fullscreenPipelineLayout = makePipelineLayout(fullscreenSetLayouts);

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
    // CLAMP_TO_EDGE + LINEAR without mips).
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &attachmentSampler), "vkCreateSampler");

    const auto alignment = std::max<VkDeviceSize>(deviceLimits.minUniformBufferOffsetAlignment, 1);
    drawDataStride = (sizeof(DrawDataUbo) + alignment - 1) / alignment * alignment;
    frameDataStride = (sizeof(FrameDataUbo) + alignment - 1) / alignment * alignment;

    for (auto& frame : frames)
    {
        createHostVisibleUniformBuffer(frameDataStride * frameDataRingSlots, frame.frameDataBuffer,
                                       frame.frameDataAllocation, frame.frameDataMapped);
        createHostVisibleUniformBuffer(drawDataStride * drawDataRingSlots, frame.drawDataBuffer,
                                       frame.drawDataAllocation, frame.drawDataMapped);

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

        std::array<VkWriteDescriptorSet, 2> writes{};
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
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // GL leaves a disabled vertex attribute at the generic default (0, 0, 0, 1); Vulkan
    // demands a real binding for every declared input, so absent attributes read this
    // constant through a zero-stride binding.
    const std::array<float, 4> defaultVertexAttribute = {0.0f, 0.0f, 0.0f, 1.0f};
    dummyVertexBuffer = createDeviceLocalBuffer(defaultVertexAttribute.data(), sizeof(defaultVertexAttribute));

    logger.info("Vulkan descriptor machinery ready: scene sets 0-2 + fullscreen set, pool for {} sets, "
                "draw-data ring {} slots x {} bytes per frame in flight",
                descriptorPoolMaxSets, drawDataRingSlots, drawDataStride);
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
    destroySwapchain();
    createSwapchain();
    recreateNeeded = false;
}

bool VulkanRenderer::beginFrame()
{
    recreateSwapchainIfNeeded();
    if (swapchain == VK_NULL_HANDLE)
    {
        return false;
    }

    auto& frame = frames[frameIndex];
    ensure(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, waitForever), "vkWaitForFences");

    // The fence guarantees the GPU is done with this slot's uniform rings; reset both for
    // the views and draws recorded this frame.
    frame.drawDataSlotsUsed = 0;
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

    ensure(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");

    // No-ops on the coherent memory VMA picks here, but the uniform writes this frame made
    // must be flushed before the submit on any host-cached heap.
    ensure(vmaFlushAllocation(allocator, frame.frameDataAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
    ensure(vmaFlushAllocation(allocator, frame.drawDataAllocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

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
    frameIndex = (frameIndex + 1) % framesInFlight;
    return true;
}

// One view into the open frame. Every view records into the same command buffer, so N
// cameras produce N scene passes and still reach the screen through the single present
// endFrame issues.
void VulkanRenderer::recordView(Scene& scene, Camera& camera, const float delta)
{
    recordScenePass(scene, camera, delta);
}

void VulkanRenderer::recordPresent(const Resource<Shader>& shaderKey, const Resource<FboAttachment>& attachmentKey)
{
    const auto& attachment = memoryStorageService.bufferAttachments.get(attachmentKey);
    if (attachment.gpuResourceId.has_value())
    {
        recordPresentPass(shaderKey->gpuResourceId, attachment.gpuResourceId.value());
    }
}

bool VulkanRenderer::recordPresentPass(const unsigned int shaderId, const unsigned int attachmentImageId)
{
    const auto shader = shaderObjects.find(shaderId);
    if (shader == shaderObjects.end() || shader->second.swapchainTargetPipeline == VK_NULL_HANDLE ||
        !imageResources.contains(attachmentImageId))
    {
        if (!fullscreenPipelineUnavailableLogged)
        {
            logger.warn("Vulkan presenter pass skipped: shader {} has no swapchain pipeline or attachment image {} is "
                        "unknown; presenting the clear colour",
                        shaderId, attachmentImageId);
            fullscreenPipelineUnavailableLogged = true;
        }
        return false;
    }

    // Resolved before the swapchain image is touched: a pass that cannot bind its source
    // must leave the image untouched so the clear fallback can still take it.
    if (attachmentSet(attachmentImageId) == VK_NULL_HANDLE)
    {
        return false;
    }

    auto& frame = frames[frameIndex];
    transitionImage(frame.commandBuffer, swapchainImages[currentImageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    recordFullScreenPass(attachmentImageId, swapchainImageViews[currentImageIndex], swapchainExtent,
                         shader->second.swapchainTargetPipeline);

    swapchainPassRecorded = true;
    lastPresentPass = PresentPass{shaderId, attachmentImageId};
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
    const auto source = stageAccessFor(resource.layout);
    const auto destination = stageAccessFor(newLayout);

    // Emitted even when the layout already matches: consecutive frames write the same
    // attachment images, and the barrier is what orders this frame's access against the
    // previous submission's.
    transitionImage(commandBuffer, resource.image, resource.layout, newLayout, source.stage, source.access,
                    destination.stage, destination.access,
                    VkImageSubresourceRange{resource.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
    resource.layout = newLayout;
}

void VulkanRenderer::recordScenePass(Scene& scene, Camera& camera, const float delta)
{
    auto& frame = frames[frameIndex];

    sceneEnvironmentImageId.reset();
    if (scene.environment.has_value())
    {
        const auto environmentImageId = memoryStorageService.cubeMaps.get(scene.environment.value()).gpuResourceId;
        if (imageResources.contains(environmentImageId))
        {
            sceneEnvironmentImageId = environmentImageId;
        }
    }

    std::optional<unsigned int> colorImageId;
    std::optional<unsigned int> depthImageId;
    if (camera.output.has_value())
    {
        const auto& outputBuffer = memoryStorageService.frameBuffers.get(camera.output.value());
        for (const auto& attachmentKey : outputBuffer.attachments)
        {
            const auto& attachment = memoryStorageService.bufferAttachments.get(attachmentKey);
            if (!attachment.gpuResourceId.has_value())
            {
                continue;
            }

            if (attachment.type == FboAttachmentType::Color && !colorImageId.has_value())
            {
                colorImageId = attachment.gpuResourceId;
            }
            else if (attachment.type == FboAttachmentType::Depth && !depthImageId.has_value())
            {
                depthImageId = attachment.gpuResourceId;
            }
        }
    }

    if (!colorImageId.has_value() || !imageResources.contains(colorImageId.value()))
    {
        if (!missingCameraOutputLogged)
        {
            logger.warn("Camera output framebuffer has no colour image; the scene pass is skipped this session");
            missingCameraOutputLogged = true;
        }
        return;
    }

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

    if (declaredLights > maxLights && !lightLimitLogged)
    {
        logger.warn("Vulkan frame data carries at most {} lights; {} were declared and the rest are ignored", maxLights,
                    declaredLights);
        lightLimitLogged = true;
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

    transitionTracked(frame.commandBuffer, colorImageId.value(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (depthImageId.has_value())
    {
        transitionTracked(frame.commandBuffer, depthImageId.value(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }

    // Copied, not referenced: the lazy uploads below insert into imageResources, and a
    // rehash would leave a reference into it dangling.
    const auto colorImage = imageResources.at(colorImageId.value());
    const VkExtent2D extent{colorImage.width, colorImage.height};

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = colorImage.view;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color =
        VkClearColorValue{{clearColour[0], clearColour[1], clearColour[2], clearColour[3]}};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = VkClearDepthStencilValue{1.0f, 0};
    // A camera with no depth attachment renders depth-less, which the pipeline has to be told.
    auto depthFormat = VK_FORMAT_UNDEFINED;
    if (depthImageId.has_value())
    {
        const auto& depthImage = imageResources.at(depthImageId.value());
        depthAttachment.imageView = depthImage.view;
        depthFormat = depthImage.format;
    }

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, extent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    if (depthImageId.has_value())
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

    auto recordedDraws = 0u;
    for (auto& entity : scene.models)
    {
        auto& model = entity.model;

        // Around the entity's submission, which on this backend is where its commands are
        // recorded: the same point in the frame GL calls them at.
        if (entity.beforeDraw.has_value())
        {
            (*entity.beforeDraw)();
        }

        for (auto& mesh : entity.meshes)
        {
            if (!mesh.mesh->gpuResourceId.has_value())
            {
                upload(model);
            }

            if (!mesh.mesh->gpuResourceId.has_value())
            {
                if (!missingMeshUploadLogged)
                {
                    logger.warn("Skipping mesh without a GPU resource: {}", mesh.mesh->name);
                    missingMeshUploadLogged = true;
                }

                continue;
            }

            const auto entityModelMatrix = sceneManagerService.modelMatrix(entity.node) * mesh.mesh->modelMatrix;
            // Bound by reference into the mesh's palette buffer; copying undoes the per-frame
            // allocation the service avoids.
            const auto& joints = renderableEntityService.joints(mesh, delta);

            for (const auto& primitive : mesh.mesh->meshPrimitives)
            {
                if (!model->meshBuffers[static_cast<size_t>(primitive.meshBufferIndex)].gpuId.has_value())
                {
                    continue;
                }

                if (!primitive.material.has_value() || !primitive.material.value()->shader.has_value())
                {
                    if (!missingMaterialLogged)
                    {
                        logger.warn("Skipping primitive without a material and shader in mesh: {}", mesh.mesh->name);
                        missingMaterialLogged = true;
                    }

                    continue;
                }

                if (!primitive.gpuVao.has_value())
                {
                    continue;
                }

                recordDraw(primitive, entity.shader, entityModelMatrix, camera, joints, colorImage.format, depthFormat);
                recordedDraws++;
            }
        }

        if (entity.afterDraw.has_value())
        {
            (*entity.afterDraw)();
        }
    }

    vkCmdEndRendering(frame.commandBuffer);

    transitionTracked(frame.commandBuffer, colorImageId.value(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    for (const auto& postProcessKey : camera.postProcesses)
    {
        const auto& postProcess = memoryStorageService.postProcesses.get(postProcessKey);

        // GL binds every input to units 0..n; the Vulkan fullscreen layout carries exactly
        // one sampler (vulkan-abi.md), and the HDR shader declares only binding 0, so the
        // camera depth attachment GL also binds is deliberately left unbound.
        std::optional<unsigned int> sourceImageId;
        if (!postProcess.inputs.empty())
        {
            sourceImageId = memoryStorageService.bufferAttachments.get(postProcess.inputs.front()).gpuResourceId;
        }

        std::optional<unsigned int> targetImageId;
        if (postProcess.output.has_value())
        {
            const auto& targetBuffer = memoryStorageService.frameBuffers.get(postProcess.output.value());
            for (const auto& attachmentKey : targetBuffer.attachments)
            {
                const auto& attachment = memoryStorageService.bufferAttachments.get(attachmentKey);
                if (attachment.type == FboAttachmentType::Color && attachment.gpuResourceId.has_value())
                {
                    targetImageId = attachment.gpuResourceId;
                    break;
                }
            }
        }

        const auto shader = shaderObjects.find(postProcess.shader->gpuResourceId);
        const auto usable = sourceImageId.has_value() && targetImageId.has_value() &&
                            imageResources.contains(sourceImageId.value()) &&
                            imageResources.contains(targetImageId.value()) && shader != shaderObjects.end() &&
                            shader->second.fullscreen;
        // The pipeline is built against the target attachment's own format, so a post-process
        // buffer created with any FboAttachment::internalFormat renders rather than mismatching
        // a hardcoded one.
        const auto pipeline = usable ? offscreenPipeline(postProcess.shader->gpuResourceId,
                                                         imageResources.at(targetImageId.value()).format)
                                     : VK_NULL_HANDLE;
        if (pipeline == VK_NULL_HANDLE)
        {
            if (!missingPostProcessTargetLogged)
            {
                logger.warn("Vulkan post-process pass skipped: it needs an input attachment, a colour output "
                            "attachment and a fullscreen shader");
                missingPostProcessTargetLogged = true;
            }

            continue;
        }

        transitionTracked(frame.commandBuffer, targetImageId.value(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        const auto& targetImage = imageResources.at(targetImageId.value());
        recordFullScreenPass(sourceImageId.value(), targetImage.view, VkExtent2D{targetImage.width, targetImage.height},
                             pipeline);

        transitionTracked(frame.commandBuffer, targetImageId.value(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (!drawSummaryLogged && recordedDraws > 0)
    {
        logger.info("Vulkan scene pass recorded: {} draw(s) into {}x{} RGBA16F, {} post-process pass(es), {} scene "
                    "pipeline(s), {} material set(s)",
                    recordedDraws, extent.width, extent.height, camera.postProcesses.size(), scenePipelines.size(),
                    materialResources.size());
        drawSummaryLogged = true;
    }
}

void VulkanRenderer::recordDraw(const MeshPrimitive& primitive, const Resource<Shader>& shader,
                                const glm::mat4& entityModelMatrix, const Camera& camera,
                                const std::vector<glm::mat4>& joints, const VkFormat colorFormat,
                                const VkFormat depthFormat)
{
    const auto bound = primitiveBindings.find(primitive.gpuVao.value());
    if (bound == primitiveBindings.end() || !bound->second.drawable)
    {
        return;
    }

    const auto material = primitive.material.value();
    // The instance's shader, not the material's: materials live in shared storage, so two
    // renderables built from one model would otherwise restyle each other.
    const auto shaderId = shader->gpuResourceId;
    const VkCullModeFlags cullMode = material->opaque ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;

    const auto pipeline = scenePipeline(shaderId, bound->second, cullMode, colorFormat, depthFormat);
    if (pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    // GL's bindMaterial falls back to the scene environment when the material has none.
    auto environmentImageId = sceneEnvironmentImageId.value_or(0u);
    if (material->environment.has_value())
    {
        environmentImageId = memoryStorageService.cubeMaps.get(material->environment.value()).gpuResourceId;
    }
    if (!imageResources.contains(environmentImageId))
    {
        environmentImageId = dummyCubeMap();
    }

    const auto set = materialSet(material, environmentImageId);
    if (set == VK_NULL_HANDLE)
    {
        return;
    }

    const auto slot = allocateDrawDataSlot();
    if (!slot.has_value())
    {
        return;
    }

    auto& frame = frames[frameIndex];
    auto* target = static_cast<unsigned char*>(frame.drawDataMapped) + slot.value();

    const std::array<glm::mat4, 4> matrices = {
        entityModelMatrix, camera.modelViewMatrix * entityModelMatrix,
        clipCorrection() * camera.modelViewProjectionMatrix * entityModelMatrix,
        // The shader reads mat3(normalMatrix); the mat4 slot carries the upper 3x3.
        glm::mat4(glm::transpose(glm::inverse(glm::mat3(camera.modelViewMatrix * entityModelMatrix))))};
    std::memcpy(target, matrices.data(), sizeof(matrices));

    auto jointCount = joints.size();
    if (jointCount > maxJoints)
    {
        if (!jointLimitLogged)
        {
            logger.warn("Vulkan draw data carries at most {} joints; {} were supplied and the rest are ignored",
                        maxJoints, jointCount);
            jointLimitLogged = true;
        }

        jointCount = maxJoints;
    }

    const auto animated = glm::ivec4(joints.empty() ? 0 : 1, 0, 0, 0);
    std::memcpy(target + offsetof(DrawDataUbo, animated), &animated, sizeof(animated));
    if (jointCount > 0)
    {
        std::memcpy(target + offsetof(DrawDataUbo, jointTransforms), joints.data(), jointCount * sizeof(glm::mat4));
    }

    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    if (!bound->second.boundBuffers.empty())
    {
        vkCmdBindVertexBuffers(frame.commandBuffer, 0, static_cast<uint32_t>(bound->second.boundBuffers.size()),
                               bound->second.boundBuffers.data(), bound->second.boundOffsets.data());
    }

    vkCmdBindIndexBuffer(frame.commandBuffer, bound->second.indexBuffer, bound->second.indexOffset,
                         bound->second.indexType);

    const std::array descriptorSets = {frame.frameDataSet, set, frame.drawDataSet};
    // One offset per dynamic descriptor, in set-then-binding order: set 0's view slot, then
    // set 2's draw slot.
    const std::array dynamicOffsets = {static_cast<uint32_t>(frame.frameDataOffset),
                                       static_cast<uint32_t>(slot.value())};
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout, 0,
                            static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
                            static_cast<uint32_t>(dynamicOffsets.size()), dynamicOffsets.data());

    vkCmdDrawIndexed(frame.commandBuffer, bound->second.indexCount, 1, 0, 0, 0);
}

bool VulkanRenderer::recordFullScreenPass(const unsigned int sourceImageId, const VkImageView targetView,
                                          const VkExtent2D targetExtent, const VkPipeline pipeline)
{
    const auto set = attachmentSet(sourceImageId);
    if (set == VK_NULL_HANDLE)
    {
        return false;
    }

    auto& frame = frames[frameIndex];
    transitionTracked(frame.commandBuffer, sourceImageId, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = targetView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // GL's drawFullScreenQuad clears before the quad; the quad covers the target anyway.
    colorAttachment.clearValue.color =
        VkClearColorValue{{clearColour[0], clearColour[1], clearColour[2], clearColour[3]}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, targetExtent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);

    // Positive height here: the fullscreen shaders derive their texture coordinates from
    // gl_VertexIndex in Vulkan's own y-down clip space, so the scene attachment (already
    // stored the GL way up) maps through one-to-one.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(targetExtent.width);
    viewport.height = static_cast<float>(targetExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);

    const VkRect2D scissor{VkOffset2D{0, 0}, targetExtent};
    vkCmdSetScissor(frame.commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fullscreenPipelineLayout, 0, 1, &set,
                            0, nullptr);
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
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

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
        if (!unsupportedTopologyLogged)
        {
            logger.warn("Vulkan has no topology for glTF draw mode {}; primitives using it are not drawn",
                        primitive.mode);
            unsupportedTopologyLogged = true;
        }

        return binding;
    }
    binding.topology = topology.value();

    const auto indices = indexType(primitive.componentType);
    if (!indices.has_value())
    {
        if (!unsupportedIndexTypeLogged)
        {
            logger.warn("Vulkan index component type 0x{:x} is outside this backend's uint16/uint32 support; "
                        "primitives using it are not drawn",
                        primitive.componentType);
            unsupportedIndexTypeLogged = true;
        }

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
        return binding;
    }

    for (const auto& bufferBind : binding.input.bufferBinds)
    {
        const auto handle = bufferHandle(bufferBind.bufferIndex);
        if (handle == VK_NULL_HANDLE)
        {
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

void VulkanRenderer::upload(const Resource<Model>& modelKey)
{
    auto model = memoryStorageService.models.get(modelKey);
    auto uploadedBuffers = 0u;

    for (const auto& meshKey : model.meshes)
    {
        auto mesh = memoryStorageService.meshes.get(meshKey);

        if (mesh.gpuResourceId.has_value())
        {
            continue;
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

        memoryStorageService.models.update(modelKey, model);

        // draw()'s "already uploaded" sentinel. Nothing looks the id up — each primitive owns
        // its binding in gpuVao — but it has to be assigned outside the loop below: GLTFService
        // drops non-indexed primitives, so meshPrimitives can be empty, and an unset sentinel
        // makes draw() re-enter upload() on every frame.
        auto uploadedSentinel = 0u;

        for (auto& primitive : mesh.meshPrimitives)
        {
            const auto id = nextResourceId++;
            primitiveBindings.emplace(id, makePrimitiveBinding(model, primitive));
            primitive.gpuVao = id;
            uploadedSentinel = id;
        }

        mesh.gpuResourceId = uploadedSentinel;

        memoryStorageService.meshes.update(meshKey, mesh);
    }

    for (const auto& materialKey : model.materials)
    {
        uploadMaterialTextures(materialKey);
    }

    logger.info("Vulkan model uploaded: {} buffer(s), {} mesh(es), {} material(s)", uploadedBuffers,
                model.meshes.size(), model.materials.size());
}

std::optional<unsigned int> VulkanRenderer::uploadTexture(const Resource<Texture>& textureKey)
{
    const auto& texture = memoryStorageService.textures.get(textureKey);
    if (texture.gpuResourceId.has_value())
    {
        return texture.gpuResourceId;
    }

    // Any channel count at any of the three source precisions uploads (createSampledImage
    // expands it the way GL's driver does); only a payload that cannot describe the image is
    // rejected.
    const auto texelCount = static_cast<size_t>(texture.width) * texture.height;
    const auto sourceBytes =
        texelCount * static_cast<size_t>(channelCount(texture.format)) * pixelComponentBytes(texture.pixelDataType);
    if (texelCount == 0 || channelCount(texture.format) == 0 || texture.data.size() < sourceBytes)
    {
        if (!unsupportedTextureLayoutLogged)
        {
            logger.warn("Vulkan texture upload skipped for {}: {}x{} needs {} byte(s) for its declared layout and "
                        "carries {}",
                        texture.name, texture.width, texture.height, sourceBytes, texture.data.size());
            unsupportedTextureLayoutLogged = true;
        }

        return std::nullopt;
    }

    const std::array faces = {&texture};
    const auto id = createSampledImage(faces, false);

    // MemoryStorage hands out const references to non-const elements; writing the id back
    // in place keeps the GL-visible contract without copying a multi-megabyte payload
    // through get/update (same justification as createCubeMap's face clearing).
    auto& uploaded = const_cast<Texture&>(texture);
    uploaded.gpuResourceId = id;

    // The GPU owns the texels now; shed the CPU copy the way upload() sheds mesh buffers.
    uploaded.data.clear();
    uploaded.data.shrink_to_fit();

    return id;
}

void VulkanRenderer::uploadMaterialTextures(const Resource<Material>& materialKey)
{
    const auto& material = memoryStorageService.materials.get(materialKey);

    for (const auto& textureKey :
         {material.albedo, material.normal, material.metallicRoughness, material.emissive, material.occlusion})
    {
        if (textureKey.has_value() && memoryStorageService.textures.exists(textureKey.value()))
        {
            static_cast<void>(uploadTexture(textureKey.value()));
        }
    }

    for (const auto& textureKey : material.textures)
    {
        if (memoryStorageService.textures.exists(textureKey))
        {
            static_cast<void>(uploadTexture(textureKey));
        }
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

VkDescriptorSet VulkanRenderer::materialSet(const Resource<Material>& materialKey,
                                            const unsigned int environmentImageId)
{
    const auto key = (static_cast<uint64_t>(materialKey.id) << 32u) | static_cast<uint64_t>(environmentImageId);
    const auto cached = materialResources.find(key);
    if (cached != materialResources.end())
    {
        return cached->second.set;
    }

    const auto& material = memoryStorageService.materials.get(materialKey);

    const auto textureImage = [&](const std::optional<Resource<Texture>>& textureKey) -> std::optional<unsigned int>
    {
        if (!textureKey.has_value() || !textureKey.value()->gpuResourceId.has_value() ||
            !imageResources.contains(textureKey.value()->gpuResourceId.value()))
        {
            return std::nullopt;
        }

        return textureKey.value()->gpuResourceId;
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
        if (!descriptorSetUnavailableLogged)
        {
            logger.warn("Vulkan descriptor pool has no room for another material set; those primitives are not drawn");
            descriptorSetUnavailableLogged = true;
        }

        return VK_NULL_HANDLE;
    }

    MaterialResource resource;
    resource.set = set;
    void* mapped = nullptr;
    createHostVisibleUniformBuffer(sizeof(MaterialDataUbo), resource.buffer, resource.allocation, mapped);

    MaterialDataUbo materialData{};
    materialData.baseColour = material.baseColour;
    materialData.roughMetal = glm::vec4(material.roughness, material.metalness, 0.0f, 0.0f);
    materialData.textureTransform = glm::mat4(material.transform);
    materialData.useTextures = glm::ivec4(diffuse.has_value() ? 1 : 0, normal.has_value() ? 1 : 0,
                                          specular.has_value() ? 1 : 0, emissive.has_value() ? 1 : 0);
    materialData.useTextures2 = glm::ivec4(occlusion.has_value() ? 1 : 0, 0, 0, 0);
    std::memcpy(mapped, &materialData, sizeof(materialData));
    ensure(vmaFlushAllocation(allocator, resource.allocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

    const std::array sampledImages = {diffuse.value_or(fallbackTexture),   normal.value_or(fallbackTexture),
                                      specular.value_or(fallbackTexture),  emissive.value_or(fallbackTexture),
                                      occlusion.value_or(fallbackTexture), environment};

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
        writes[slot + 1].dstBinding = textureBinding(static_cast<MaterialTextureSlot>(slot), GraphicsApi::Vulkan);
        writes[slot + 1].descriptorCount = 1;
        writes[slot + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[slot + 1].pImageInfo = &imageInfos[slot];
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    materialResources.emplace(key, resource);
    return set;
}

VkDescriptorSet VulkanRenderer::attachmentSet(const unsigned int imageId)
{
    const auto cached = attachmentSets.find(imageId);
    if (cached != attachmentSets.end())
    {
        return cached->second;
    }

    const auto image = imageResources.find(imageId);
    if (image == imageResources.end())
    {
        return VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo setAllocateInfo{};
    setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocateInfo.descriptorPool = descriptorPool;
    setAllocateInfo.descriptorSetCount = 1;
    setAllocateInfo.pSetLayouts = &fullscreenSetLayout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &setAllocateInfo, &set) != VK_SUCCESS)
    {
        if (!descriptorSetUnavailableLogged)
        {
            logger.warn("Vulkan descriptor pool has no room for another attachment set; that pass is skipped");
            descriptorSetUnavailableLogged = true;
        }

        return VK_NULL_HANDLE;
    }

    const VkDescriptorImageInfo imageInfo{image->second.sampler != VK_NULL_HANDLE ? image->second.sampler
                                                                                  : attachmentSampler,
                                          image->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    attachmentSets.emplace(imageId, set);
    return set;
}

VkPipeline VulkanRenderer::scenePipeline(const unsigned int shaderId, PrimitiveBinding& binding,
                                         const VkCullModeFlags cullMode, const VkFormat colorFormat,
                                         const VkFormat depthFormat)
{
    // A primitive's material — and so its shader and cull mode — never changes after load,
    // so one resolution per primitive is enough and a failed build is not retried.
    if (binding.resolved)
    {
        return binding.pipeline;
    }
    binding.resolved = true;

    const auto shader = shaderObjects.find(shaderId);
    if (shader == shaderObjects.end() || shader->second.fullscreen)
    {
        if (!scenePipelineUnavailableLogged)
        {
            logger.warn("Vulkan shader object {} is not a scene shader; primitives using it are not drawn", shaderId);
            scenePipelineUnavailableLogged = true;
        }

        return VK_NULL_HANDLE;
    }

    // Exactly the locations the vertex shader declares are fed, no more: a bound attribute
    // the shader never reads is work the driver has to discard, and Vulkan reports it.
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    binding.boundBuffers.clear();
    binding.boundOffsets.clear();

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

        binding.boundBuffers.push_back(binding.vertexBuffers[attribute.binding]);
        binding.boundOffsets.push_back(binding.vertexOffsets[attribute.binding]);
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
        binding.boundBuffers.push_back(dummyVertexBuffer.buffer);
        binding.boundOffsets.push_back(0);
    }

    // The pipeline's rendering formats are part of its identity, so they are part of the key:
    // the attachments they come from are the ones FboAttachment::internalFormat asked for.
    const auto key = std::to_string(shaderId) + "|" + std::to_string(cullMode) + "|" +
                     std::to_string(static_cast<int>(binding.topology)) + "|" +
                     std::to_string(static_cast<int>(colorFormat)) + "+" +
                     std::to_string(static_cast<int>(depthFormat)) + "|" + binding.signature;
    const auto cachedPipeline = scenePipelines.find(key);
    if (cachedPipeline != scenePipelines.end())
    {
        binding.pipeline = cachedPipeline->second;
        return binding.pipeline;
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

    // GL enables GL_DEPTH_TEST globally and never touches the func or the write mask, so
    // this is GL's default LESS with writes on.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.maxDepthBounds = 1.0f;

    // GL runs with glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) globally enabled.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
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
    renderingInfo.pColorAttachmentFormats = &colorFormat;
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
    const auto createResult = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (createResult != VK_SUCCESS)
    {
        if (!scenePipelineUnavailableLogged)
        {
            logger.warn("Vulkan scene pipeline for shader {} was not created (VkResult {}); primitives using it are "
                        "not drawn",
                        shaderId, static_cast<int>(createResult));
            scenePipelineUnavailableLogged = true;
        }

        return VK_NULL_HANDLE;
    }

    scenePipelines.emplace(key, pipeline);
    binding.pipeline = pipeline;
    return pipeline;
}

VkPipeline VulkanRenderer::offscreenPipeline(const unsigned int shaderId, const VkFormat colorFormat)
{
    const auto key = (static_cast<uint64_t>(shaderId) << 32u) | static_cast<uint32_t>(colorFormat);
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
        buildFullscreenPipeline(shader->second.vertexModule, shader->second.fragmentModule, colorFormat);
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
        if (!drawDataRingExhaustedLogged)
        {
            logger.warn("Vulkan draw-data ring exhausted: {} slots of {} bytes; draws beyond the ring are skipped "
                        "for the rest of the frame",
                        drawDataRingSlots, drawDataStride);
            drawDataRingExhaustedLogged = true;
        }
        return std::nullopt;
    }

    const auto offset = static_cast<VkDeviceSize>(frame.drawDataSlotsUsed) * drawDataStride;
    frame.drawDataSlotsUsed++;
    return offset;
}

std::optional<VkDeviceSize> VulkanRenderer::allocateFrameDataSlot()
{
    auto& frame = frames[frameIndex];
    if (frame.frameDataSlotsUsed >= frameDataRingSlots)
    {
        if (!frameDataRingExhaustedLogged)
        {
            logger.warn("Vulkan frame-data ring exhausted: {} slots of {} bytes; views beyond the ring are skipped "
                        "for the rest of the frame",
                        frameDataRingSlots, frameDataStride);
            frameDataRingExhaustedLogged = true;
        }
        return std::nullopt;
    }

    const auto offset = static_cast<VkDeviceSize>(frame.frameDataSlotsUsed) * frameDataStride;
    frame.frameDataSlotsUsed++;
    return offset;
}

std::optional<std::vector<uint32_t>>
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

    // Every Vulkan source is compiled with the contract macros predefined, so no shader
    // spells a set index, a binding, an attribute location or an array bound that C++ also
    // holds. The GL path splices the same list in as text (withShaderContractMacros).
    for (const auto& macro : shaderContractMacros(GraphicsApi::Vulkan))
    {
        options.AddMacroDefinition(std::string(macro.name), std::to_string(macro.value));
    }

    const auto result = compiler.CompileGlslToSpv(source, kind, stageName, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        // A Vulkan source that will not compile is a real failure on this backend.
        logger.error("Vulkan {} shader compilation did not succeed: {}", stageName, result.GetErrorMessage());
        return std::nullopt;
    }

    if (result.GetNumWarnings() > 0)
    {
        logger.warn("Vulkan {} shader compiled with {} warning(s)", stageName, result.GetNumWarnings());
    }

    return std::vector<uint32_t>(result.cbegin(), result.cend());
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
                                                   const VkShaderModule fragmentModule,
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

    // GL runs with glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) globally enabled.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
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
    ensure(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
           "vkCreateGraphicsPipelines");
    return pipeline;
}

std::expected<unsigned int, std::string> VulkanRenderer::createShaderObject(const ShaderDescriptor& shaderDescriptor)
{
    try
    {
        if (shaderDescriptor.vulkanVertexShaderSource.empty() || shaderDescriptor.vulkanFragmentShaderSource.empty())
        {
            // Expected per vulkan-abi.md: a descriptor without the Vulkan dialect cannot
            // produce a shader object on this backend.
            return std::unexpected("the shader descriptor carries no vulkanVertexShaderSource and/or "
                                   "vulkanFragmentShaderSource, which this backend cannot substitute");
        }

        const auto vertexSpirv =
            compileToSpirv(shaderDescriptor.vulkanVertexShaderSource, shaderc_glsl_vertex_shader, "vertex");
        const auto fragmentSpirv =
            compileToSpirv(shaderDescriptor.vulkanFragmentShaderSource, shaderc_glsl_fragment_shader, "fragment");
        if (!vertexSpirv.has_value() || !fragmentSpirv.has_value())
        {
            return std::unexpected("the Vulkan-dialect sources did not compile to SPIR-V");
        }

        ShaderObject shaderObject;
        shaderObject.vertexModule = createShaderModule(vertexSpirv.value());
        shaderObject.fragmentModule = createShaderModule(fragmentSpirv.value());
        shaderObject.vertexInputLocations = spirvVertexInputLocations(vertexSpirv.value());
        shaderObject.fullscreen = shaderObject.vertexInputLocations.empty();

        if (shaderObject.fullscreen)
        {
            shaderObject.swapchainTargetPipeline =
                buildFullscreenPipeline(shaderObject.vertexModule, shaderObject.fragmentModule, surfaceFormat.format);
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
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = uploadCommandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    ensure(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ensure(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    return commandBuffer;
}

void VulkanRenderer::finishUploadCommands(const VkCommandBuffer commandBuffer) const
{
    ensure(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

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
    vkFreeCommandBuffers(device, uploadCommandPool, 1, &commandBuffer);
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

    const auto format = sampledImageFormat(first.pixelDataType);
    const auto texelBytes = static_cast<VkDeviceSize>(componentBytes * 4);
    const auto texelCount = static_cast<size_t>(width) * height;
    const auto faceBytes = static_cast<VkDeviceSize>(texelCount) * texelBytes;
    const auto sourceFaceBytes = texelCount * sourceChannels * componentBytes;

    for (const auto* face : faces)
    {
        if (face->width != width || face->height != height || face->pixelDataType != first.pixelDataType ||
            face->format != first.format || face->data.size() < sourceFaceBytes)
        {
            throw std::runtime_error("sampled image sources disagree on size, pixel type or channel layout");
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
    else if (!mipGenerationUnavailableLogged)
    {
        logger.warn("Vulkan mip generation unavailable for this format (no linear blit); sampling top level only");
        mipGenerationUnavailableLogged = true;
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
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

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
    imageResources.emplace(id, ImageResource{image, imageAllocation, view, sampler, format, mipLevels, width, height,
                                             VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
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

        // The GPU owns the texels now; shed the CPU copies the way the GL path sheds staged
        // mesh buffers. The storage deque's elements are non-const objects, so the cast is
        // defined behaviour.
        for (const auto* face : faces)
        {
            auto& texture = const_cast<Texture&>(*face);
            texture.data.clear();
            texture.data.shrink_to_fit();
        }

        logger.info("Vulkan cube map {} uploaded: 6 faces {}x{}, {} mip levels", id, right.width, right.height,
                    imageResources.at(id).mipLevels);
        return id;
    }
    catch (const std::exception& exception)
    {
        return std::unexpected(exception.what());
    }
}

void VulkanRenderer::destroyImageResource(const unsigned int id) const
{
    const auto entry = imageResources.find(id);
    if (entry == imageResources.end())
    {
        return;
    }

    // The cached fullscreen set names this image's view; recycle it with the image so a
    // resize does not leak pool space (the pool carries FREE_DESCRIPTOR_SET for this).
    const auto cachedSet = attachmentSets.find(id);
    if (cachedSet != attachmentSets.end())
    {
        vkFreeDescriptorSets(device, descriptorPool, 1, &cachedSet->second);
        attachmentSets.erase(cachedSet);
    }

    const auto& resource = entry->second;
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
        vmaDestroyImage(allocator, resource.image, resource.allocation);
    }
    imageResources.erase(entry);
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
            auto attachment = memoryStorageService.bufferAttachments.get(attachmentKey);

            VkImageUsageFlags usage;
            VkImageAspectFlags aspect;
            switch (attachment.type)
            {
            case FboAttachmentType::Color:
                usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                break;
            case FboAttachmentType::Depth:
                usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                break;
            default:
            {
                static auto warnedUnsupportedAttachment = false;
                if (!warnedUnsupportedAttachment)
                {
                    logger.warn("Vulkan framebuffers support Color and Depth attachments only; skipping type {}",
                                static_cast<int>(attachment.type));
                    warnedUnsupportedAttachment = true;
                }
                continue;
            }
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

            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = VkExtent3D{width, height, 1};
            imageInfo.mipLevels = 1;
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
            viewInfo.subresourceRange = VkImageSubresourceRange{aspect, 0, 1, 0, 1};

            VkImageView view = VK_NULL_HANDLE;
            ensure(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");

            const auto attachmentId = nextResourceId++;
            imageResources.emplace(attachmentId, ImageResource{image, allocation, view, VK_NULL_HANDLE, format, 1,
                                                               width, height, aspect, VK_IMAGE_LAYOUT_UNDEFINED});
            fboResource.attachmentIds.push_back(attachmentId);

            // Observable contract shared with GL: every attachment's id is written back
            // through its Resource.
            attachment.gpuResourceId = attachmentId;
            memoryStorageService.bufferAttachments.update(attachmentKey, attachment);
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

void VulkanRenderer::deleteFbo(Fbo& fbo)
{
    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    // Resize-rate operation: a full device idle is the in-flight guard until the draw
    // path gives a reason for deferred per-frame destruction (V3).
    ensure(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");

    for (const auto& attachmentKey : fbo.attachments)
    {
        const auto& attachment = memoryStorageService.bufferAttachments.get(attachmentKey);
        if (attachment.gpuResourceId.has_value())
        {
            destroyImageResource(attachment.gpuResourceId.value());
        }
    }

    if (fbo.gpuResourceId.has_value())
    {
        fboResources.erase(fbo.gpuResourceId.value());
    }
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
            if (!beginFrame())
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
                recordPresentPass(lastPresentPass.value().shaderId, lastPresentPass.value().attachmentImageId);
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

} // namespace raceengine
