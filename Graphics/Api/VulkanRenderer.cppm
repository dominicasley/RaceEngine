module;

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

#include <spdlog/logger.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module raceengine.graphics:VulkanRenderer;

import :IRenderer;
import :Window;
import raceengine.graphics.models;

namespace raceengine
{

namespace
{

constexpr auto waitForever = std::numeric_limits<uint64_t>::max();
constexpr const char* validationLayerName = "VK_LAYER_KHRONOS_validation";

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
                     const VkAccessFlags2 dstAccess)
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
    barrier.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
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
    };

    // Declaration order is dependency order; the destructor tears down in reverse.
    spdlog::logger& logger;
    IWindow& window;
    bool validationLayerEnabled = false;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessenger = nullptr;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    uint32_t presentQueueFamily = 0;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VmaAllocator allocator = nullptr;
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
    unsigned int nextShaderObjectId = 1;

public:
    explicit VulkanRenderer(spdlog::logger& logger, IWindow& window);
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
    void recreateSwapchainIfNeeded();
    void recordFrame(VkCommandBuffer commandBuffer, uint32_t imageIndex, VkBuffer captureBuffer) const;
    bool renderAndPresentFrame(VkBuffer captureBuffer);
};

VulkanRenderer::VulkanRenderer(spdlog::logger& logger, IWindow& window) :
    logger(logger),
    window(window)
{
}

VulkanRenderer::~VulkanRenderer()
{
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    for (auto& frame : frames)
    {
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

        if (features13.dynamicRendering == VK_FALSE || features13.synchronization2 == VK_FALSE)
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
    // so V2 pipelines inherit the state unchanged.
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

std::optional<unsigned int> VulkanRenderer::createShaderObject(const ShaderDescriptor&)
{
    // Placeholder ids instead of nullopt: the game unconditionally dereferences the
    // shader resources it creates, and V1 never consumes gpuResourceId.
    static auto logged = false;
    if (!logged)
    {
        logger.info("Vulkan backend: shader objects are stubs until V2; issuing placeholder ids");
        logged = true;
    }

    return nextShaderObjectId++;
}

unsigned int VulkanRenderer::createCubeMap(const Texture&, const Texture&, const Texture&, const Texture&,
                                           const Texture&, const Texture&) const
{
    static auto logged = false;
    if (!logged)
    {
        logger.info("Vulkan backend: cube maps are stubs until V2; issuing placeholder id 0");
        logged = true;
    }

    return 0;
}

unsigned int VulkanRenderer::createFbo(const Fbo&) const
{
    static auto logged = false;
    if (!logged)
    {
        logger.info("Vulkan backend: framebuffers are stubs until V3; issuing placeholder id 0");
        logged = true;
    }

    return 0;
}

void VulkanRenderer::deleteFbo(Fbo&) const
{
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
