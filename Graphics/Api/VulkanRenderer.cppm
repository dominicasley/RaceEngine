module;

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <shaderc/shaderc.hpp>
#include <spdlog/logger.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

import :IRenderer;
import :Window;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

namespace
{

constexpr auto waitForever = std::numeric_limits<uint64_t>::max();
constexpr const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
// Parity with the GL shaders' jointTransformationMatrixes[128] (vulkan-abi.md).
constexpr uint32_t maxJoints = 128;
// DrawData is ~8.5 KiB per draw; a few hundred draws per frame covers the sandbox scene
// with generous headroom while keeping the ring under 5 MiB per frame in flight.
constexpr uint32_t drawDataRingSlots = 512;

// Pool sizing: fixed and generous beats grow-on-demand here — the game's realistic
// ceiling is a few hundred materials (7 descriptors each) plus per-frame frame/draw sets
// and a handful of fullscreen input sets; these counts dwarf that at negligible cost.
constexpr uint32_t descriptorPoolMaxSets = 1024;
constexpr uint32_t descriptorPoolUniformBuffers = 512;
constexpr uint32_t descriptorPoolDynamicUniformBuffers = 16;
constexpr uint32_t descriptorPoolCombinedImageSamplers = 4096;

// Set 0 binding 0 (vulkan-abi.md); std140-compatible, so the C++ layout is the GPU layout.
struct FrameDataUbo
{
    glm::mat4 viewMatrix;
    glm::vec4 cameraPosition;
    glm::vec4 lightPosition;
    glm::vec4 lightDiffuse;
    glm::vec4 lightSpecular;
    glm::vec4 lightAmbientAttenuation;
};

static_assert(sizeof(FrameDataUbo) == 144);

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
// keys on Location-decorated ids specifically.
[[nodiscard]] bool spirvHasVertexInputLocations(const std::vector<uint32_t>& spirv)
{
    constexpr uint32_t opDecorate = 71;
    constexpr uint32_t opVariable = 59;
    constexpr uint32_t decorationLocation = 30;
    constexpr uint32_t storageClassInput = 1;
    constexpr size_t headerWords = 5;

    std::vector<uint32_t> locationDecoratedIds;
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

            if (pass == 0 && opcode == opDecorate && wordCount >= 3 && spirv[word + 2] == decorationLocation)
            {
                locationDecoratedIds.push_back(spirv[word + 1]);
            }

            if (pass == 1 && opcode == opVariable && wordCount >= 4 && spirv[word + 3] == storageClassInput &&
                std::ranges::find(locationDecoratedIds, spirv[word + 2]) != locationDecoratedIds.end())
            {
                return true;
            }

            word += wordCount;
        }
    }

    return false;
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

export class VulkanRenderer : public IRenderer
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
    };

    struct ShaderObject
    {
        VkShaderModule vertexModule = VK_NULL_HANDLE;
        VkShaderModule fragmentModule = VK_NULL_HANDLE;
        bool fullscreen = false;
        // Fullscreen pipelines exist per reachable target format (presenter writes the
        // swapchain, post-processes write RGBA16F); the draw path picks by actual target.
        // Scene pipelines are vertex-input-dependent and are built by the draw path.
        VkPipeline swapchainTargetPipeline = VK_NULL_HANDLE;
        VkPipeline offscreenTargetPipeline = VK_NULL_HANDLE;
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
    };

    struct FboResource
    {
        std::vector<unsigned int> attachmentIds;
    };

    // Declaration order is dependency order; the destructor tears down in reverse.
    spdlog::logger& logger;
    IWindow& window;
    MemoryStorageService& memoryStorageService;
    bool validationLayerEnabled = false;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessenger = nullptr;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits deviceLimits{};
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
    mutable bool mipGenerationUnavailableLogged = false;
    bool drawDataRingExhaustedLogged = false;

public:
    explicit VulkanRenderer(spdlog::logger& logger, IWindow& window, MemoryStorageService& memoryStorageService);
    ~VulkanRenderer() override;

    bool init() override;
    void draw(Scene& scene, Camera& camera, float delta) override;
    void drawFullScreenQuad(const Resource<Shader>& shader, const Resource<FboAttachment>& attachment) const override;
    void setViewport(int width, int height) override;
    std::optional<unsigned int> createShaderObject(const ShaderDescriptor& shaderDescriptor) override;
    [[nodiscard]] unsigned int createCubeMap(const Texture& front, const Texture& back, const Texture& left,
                                             const Texture& right, const Texture& top,
                                             const Texture& bottom) const override;
    [[nodiscard]] unsigned int createFbo(const Fbo& fbo) const override;
    void deleteFbo(Fbo& fbo) const override;
    void captureFrame(const std::string& path) override;

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
    void recordFrame(VkCommandBuffer commandBuffer, uint32_t imageIndex, VkBuffer captureBuffer) const;
    bool renderAndPresentFrame(VkBuffer captureBuffer);
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
};

VulkanRenderer::VulkanRenderer(spdlog::logger& logger, IWindow& window, MemoryStorageService& memoryStorageService) :
    logger(logger),
    window(window),
    memoryStorageService(memoryStorageService)
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
        if (shader.offscreenTargetPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, shader.offscreenTargetPipeline, nullptr);
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

bool VulkanRenderer::init()
{
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
        return true;
    }
    catch (const std::exception& exception)
    {
        // Engine ignores the return value, so an unusable backend must not limp on.
        logger.error("Vulkan renderer initialisation aborted: {}", exception.what());
        throw;
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
    graphicsQueueFamily = chosenGraphicsFamily;
    presentQueueFamily = chosenPresentFamily;

    logger.info("Vulkan device selected: {} ({}, api {}.{}.{})", std::string_view(chosenProperties.deviceName),
                describeDeviceType(chosenProperties.deviceType), VK_API_VERSION_MAJOR(chosenProperties.apiVersion),
                VK_API_VERSION_MINOR(chosenProperties.apiVersion), VK_API_VERSION_PATCH(chosenProperties.apiVersion));
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

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &features13;
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

    // Scene set 0: FrameData UBO, read by both stages (vulkan-abi.md).
    const std::array frameBindings = {VkDescriptorSetLayoutBinding{
        0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    frameDataSetLayout = makeSetLayout(frameBindings);

    // Scene set 1: MaterialData UBO + sampler2D bindings 1-5 + samplerCube binding 6.
    std::array<VkDescriptorSetLayoutBinding, 7> materialBindings{};
    materialBindings[0] =
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    for (uint32_t binding = 1; binding <= 6; binding++)
    {
        materialBindings[binding] = VkDescriptorSetLayoutBinding{binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
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

    for (auto& frame : frames)
    {
        createHostVisibleUniformBuffer(sizeof(FrameDataUbo), frame.frameDataBuffer, frame.frameDataAllocation,
                                       frame.frameDataMapped);
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

        const VkDescriptorBufferInfo frameDataInfo{frame.frameDataBuffer, 0, sizeof(FrameDataUbo)};
        // Dynamic UBO range is one DrawData; the bound offset walks the ring per draw.
        const VkDescriptorBufferInfo drawDataInfo{frame.drawDataBuffer, 0, sizeof(DrawDataUbo)};

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = frame.frameDataSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &frameDataInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = frame.drawDataSet;
        writes[1].dstBinding = 0;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[1].pBufferInfo = &drawDataInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

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

void VulkanRenderer::recordFrame(const VkCommandBuffer commandBuffer, const uint32_t imageIndex,
                                 const VkBuffer captureBuffer) const
{
    const auto image = swapchainImages[imageIndex];

    transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchainImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // Matches the GL path's glClearColor(1, 1, 1, 1).
    colorAttachment.clearValue.color = VkClearColorValue{{1.0f, 1.0f, 1.0f, 1.0f}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, swapchainExtent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    // GL-convention Y-flip via negative viewport height (vulkan-abi.md); established now
    // so the prebuilt pipelines inherit the state unchanged.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(swapchainExtent.height);
    viewport.width = static_cast<float>(swapchainExtent.width);
    viewport.height = -static_cast<float>(swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    const VkRect2D scissor{VkOffset2D{0, 0}, swapchainExtent};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdEndRendering(commandBuffer);

    if (captureBuffer != VK_NULL_HANDLE)
    {
        transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = VkExtent3D{swapchainExtent.width, swapchainExtent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, captureBuffer, 1, &region);

        transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);
    }
    else
    {
        transitionImage(commandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE);
    }
}

bool VulkanRenderer::renderAndPresentFrame(const VkBuffer captureBuffer)
{
    recreateSwapchainIfNeeded();
    if (swapchain == VK_NULL_HANDLE)
    {
        return false;
    }

    auto& frame = frames[frameIndex];
    ensure(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, waitForever), "vkWaitForFences");

    // The fence guarantees the GPU is done with this slot's DrawData ring; reset it for
    // the draws recorded this frame.
    frame.drawDataSlotsUsed = 0;

    uint32_t imageIndex = 0;
    const auto acquireResult =
        vkAcquireNextImageKHR(device, swapchain, waitForever, frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
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
    recordFrame(frame.commandBuffer, imageIndex, captureBuffer);
    ensure(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo waitSemaphoreInfo{};
    waitSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemaphoreInfo.semaphore = frame.imageAvailable;
    waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signalSemaphoreInfo{};
    signalSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphoreInfo.semaphore = renderFinishedSemaphores[imageIndex];
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
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const auto presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
    {
        recreateNeeded = true;
    }
    else
    {
        ensure(presentResult, "vkQueuePresentKHR");
    }

    frameIndex = (frameIndex + 1) % framesInFlight;
    return true;
}

void VulkanRenderer::draw(Scene&, Camera&, float)
{
    // One acquire->present cycle per call. With multiple cameras this would present once
    // per camera per step; a single scene+camera holds until V3 adds offscreen targets.
    renderAndPresentFrame(VK_NULL_HANDLE);
}

void VulkanRenderer::drawFullScreenQuad(const Resource<Shader>&, const Resource<FboAttachment>&) const
{
    static auto logged = false;
    if (!logged)
    {
        logger.info("Vulkan backend: fullscreen passes are stubs until V3; presenting the clear colour");
        logged = true;
    }
}

void VulkanRenderer::setViewport(const int width, const int height)
{
    // Resize callbacks fire inside glfwPollEvents mid-step; recreation happens lazily at
    // the next draw, so only the requested extent is recorded here.
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

std::optional<std::vector<uint32_t>>
VulkanRenderer::compileToSpirv(const std::string& source, const shaderc_shader_kind kind, const char* stageName)
{
    // The C++ RAII wrapper over the C API: results release themselves, and both stay
    // confined to this translation unit's global module fragment.
    const shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

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

std::optional<unsigned int> VulkanRenderer::createShaderObject(const ShaderDescriptor& shaderDescriptor)
{
    if (shaderDescriptor.vulkanVertexShaderSource.empty() || shaderDescriptor.vulkanFragmentShaderSource.empty())
    {
        // Expected per vulkan-abi.md: a descriptor without the Vulkan dialect cannot
        // produce a shader object on this backend.
        logger.error("Vulkan backend cannot create a shader object: the descriptor is missing "
                     "vulkanVertexShaderSource and/or vulkanFragmentShaderSource");
        return std::nullopt;
    }

    const auto vertexSpirv =
        compileToSpirv(shaderDescriptor.vulkanVertexShaderSource, shaderc_glsl_vertex_shader, "vertex");
    const auto fragmentSpirv =
        compileToSpirv(shaderDescriptor.vulkanFragmentShaderSource, shaderc_glsl_fragment_shader, "fragment");
    if (!vertexSpirv.has_value() || !fragmentSpirv.has_value())
    {
        return std::nullopt;
    }

    ShaderObject shaderObject;
    shaderObject.vertexModule = createShaderModule(vertexSpirv.value());
    shaderObject.fragmentModule = createShaderModule(fragmentSpirv.value());
    shaderObject.fullscreen = !spirvHasVertexInputLocations(vertexSpirv.value());

    if (shaderObject.fullscreen)
    {
        shaderObject.swapchainTargetPipeline =
            buildFullscreenPipeline(shaderObject.vertexModule, shaderObject.fragmentModule, surfaceFormat.format);
        shaderObject.offscreenTargetPipeline = buildFullscreenPipeline(
            shaderObject.vertexModule, shaderObject.fragmentModule, VK_FORMAT_R16G16B16A16_SFLOAT);
    }

    const auto id = nextResourceId++;
    shaderObjects.emplace(id, shaderObject);

    logger.info("Vulkan shader object {} ready: vertex {} + fragment {} SPIR-V words; {}", id, vertexSpirv->size(),
                fragmentSpirv->size(),
                shaderObject.fullscreen ? "fullscreen pipelines built for swapchain and RGBA16F targets"
                                        : "scene pipeline awaits vertex input from the draw path");
    return id;
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
    const auto isHdrFloat = first.pixelDataType == PixelDataType::Float;
    const auto layerCount = static_cast<uint32_t>(faces.size());

    // 96bpp float RGB pads to RGBA32F unconditionally (vulkan-abi.md: RGB32F sampling
    // support is not universal); the stbi_load path is RGBA8.
    const auto format = isHdrFloat ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    const auto texelBytes = static_cast<VkDeviceSize>(isHdrFloat ? 16 : 4);
    const auto texelCount = static_cast<size_t>(width) * height;
    const auto faceBytes = static_cast<VkDeviceSize>(texelCount) * texelBytes;
    const auto sourceFaceBytes = isHdrFloat ? texelCount * 3 * sizeof(float) : texelCount * 4;

    for (const auto* face : faces)
    {
        if (face->width != width || face->height != height ||
            (face->pixelDataType == PixelDataType::Float) != isHdrFloat || face->data.size() < sourceFaceBytes)
        {
            throw std::runtime_error("sampled image sources disagree on size or pixel type");
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

    for (size_t faceIndex = 0; faceIndex < faces.size(); faceIndex++)
    {
        const auto& face = *faces[faceIndex];
        auto* destination =
            static_cast<unsigned char*>(stagingAllocationInfo.pMappedData) + faceIndex * static_cast<size_t>(faceBytes);

        if (isHdrFloat)
        {
            // Source rows are tightly-packed RGB floats (stbi_loadf, no row padding).
            const auto* source = reinterpret_cast<const float*>(face.data.data());
            auto* padded = reinterpret_cast<float*>(destination);
            for (size_t texel = 0; texel < texelCount; texel++)
            {
                padded[texel * 4 + 0] = source[texel * 3 + 0];
                padded[texel * 4 + 1] = source[texel * 3 + 1];
                padded[texel * 4 + 2] = source[texel * 3 + 2];
                padded[texel * 4 + 3] = 1.0f;
            }
        }
        else
        {
            std::memcpy(destination, face.data.data(), sourceFaceBytes);
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

    // CLAMP_TO_EDGE + LINEAR with trilinear mips, matching the GL cube map parameters.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = static_cast<float>(mipLevels);

    VkSampler sampler = VK_NULL_HANDLE;
    ensure(vkCreateSampler(device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");

    const auto id = nextResourceId++;
    imageResources.emplace(id, ImageResource{image, imageAllocation, view, sampler, format, mipLevels});
    return id;
}

unsigned int VulkanRenderer::createCubeMap(const Texture& front, const Texture& back, const Texture& left,
                                           const Texture& right, const Texture& top, const Texture& bottom) const
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

void VulkanRenderer::destroyImageResource(const unsigned int id) const
{
    const auto entry = imageResources.find(id);
    if (entry == imageResources.end())
    {
        return;
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

unsigned int VulkanRenderer::createFbo(const Fbo& fbo) const
{
    FboResource fboResource;
    auto loggedWidth = 0u;
    auto loggedHeight = 0u;

    for (const auto& attachmentKey : fbo.attachments)
    {
        auto attachment = memoryStorageService.bufferAttachments.get(attachmentKey);

        VkFormat format;
        VkImageUsageFlags usage;
        VkImageAspectFlags aspect;
        switch (attachment.type)
        {
        case FboAttachmentType::Color:
            // The engine's colour attachment contract is RGBA capture into RGBA16F.
            format = VK_FORMAT_R16G16B16A16_SFLOAT;
            usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            break;
        case FboAttachmentType::Depth:
            format = VK_FORMAT_D32_SFLOAT;
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
        imageResources.emplace(attachmentId, ImageResource{image, allocation, view, VK_NULL_HANDLE, format, 1});
        fboResource.attachmentIds.push_back(attachmentId);

        // Observable contract shared with GL: every attachment's id is written back
        // through its Resource.
        attachment.gpuResourceId = attachmentId;
        memoryStorageService.bufferAttachments.update(attachmentKey, attachment);
    }

    const auto id = nextResourceId++;
    const auto attachmentCount = fboResource.attachmentIds.size();
    fboResources.emplace(id, std::move(fboResource));

    logger.info("Vulkan framebuffer {} ready: {} attachment(s), {}x{}", id, attachmentCount, loggedWidth, loggedHeight);
    return id;
}

void VulkanRenderer::deleteFbo(Fbo& fbo) const
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

void VulkanRenderer::captureFrame(const std::string& path)
{
    recreateSwapchainIfNeeded();
    if (swapchain == VK_NULL_HANDLE)
    {
        logger.warn("Vulkan frame capture skipped: no swapchain (window minimised?)");
        return;
    }

    const auto width = swapchainExtent.width;
    const auto height = swapchainExtent.height;
    const auto byteCount = static_cast<VkDeviceSize>(width) * height * 4;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteCount;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocationInfo{};
    ensure(vmaCreateBuffer(allocator, &bufferInfo, &allocationCreateInfo, &buffer, &allocation, &allocationInfo),
           "vmaCreateBuffer");

    // Presented swapchain images may not be touched until reacquired, so the capture
    // renders one more identical cleared frame with the readback copy appended.
    auto presented = false;
    for (auto attempt = 0; attempt < 2 && !presented; attempt++)
    {
        presented = renderAndPresentFrame(buffer);
    }
    ensure(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");

    if (!presented)
    {
        vmaDestroyBuffer(allocator, buffer, allocation);
        logger.warn("Vulkan frame capture skipped: no swapchain image available");
        return;
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
    if (stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 4, pixels.data(), rowBytes) ==
        0)
    {
        logger.error("Failed to write frame dump to {}", path);
        std::exit(1);
    }

    logger.info("Frame dump written to {}", path);
}

} // namespace raceengine
