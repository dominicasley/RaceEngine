#include <stdexcept>
#include <iostream>
#include "VulkanRenderer.h"

VulkanRenderer::VulkanRenderer(const IWindow& window) : window(const_cast<IWindow &>(window))
{
    init();
}

VulkanRenderer::~VulkanRenderer()
{
    vkDestroyDevice(vkDevice, nullptr);
    vkDestroySurfaceKHR(vkInstance, vkSurfaceKhr, nullptr);
    vkDestroyInstance(vkInstance, nullptr);
}

void VulkanRenderer::init()
{
    VkApplicationInfo vkApplicationInfo = {};
    vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    vkApplicationInfo.pApplicationName = "OSR";
    vkApplicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    vkApplicationInfo.pEngineName = "OSR Engine";
    vkApplicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    vkApplicationInfo.apiVersion = VK_API_VERSION_1_1;

    auto requiredWindowExtensions = window.getRequiredVulkanWindowExtensions();

    VkInstanceCreateInfo vkInstanceCreateInfo = {};
    vkInstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    vkInstanceCreateInfo.pApplicationInfo = &vkApplicationInfo;
    vkInstanceCreateInfo.enabledExtensionCount = requiredWindowExtensions.count;
    vkInstanceCreateInfo.ppEnabledExtensionNames = requiredWindowExtensions.extensions;
    vkInstanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    vkInstanceCreateInfo.ppEnabledLayerNames = validationLayers.data();

    auto result = vkCreateInstance(&vkInstanceCreateInfo, nullptr, &vkInstance);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Could not create instance\n");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(vkInstance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        throw std::runtime_error("No compatible devices found\n");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(vkInstance, &deviceCount, devices.data());

    for (const auto& device : devices)
    {
        VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
        VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures;

        vkGetPhysicalDeviceProperties(device, &vkPhysicalDeviceProperties);
        vkGetPhysicalDeviceFeatures(device, &vkPhysicalDeviceFeatures);

        if (vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            vkPhysicalDevice = device;
            break;
        }
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    auto i = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            availableQueueFamilies.graphicsFamily = i;
        }

        i++;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo vkDeviceQueueCreateInfo = {};
    vkDeviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    vkDeviceQueueCreateInfo.queueFamilyIndex = availableQueueFamilies.graphicsFamily;
    vkDeviceQueueCreateInfo.queueCount = 1;
    vkDeviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures = {};
    VkDeviceCreateInfo vkDeviceCreateInfo = {};
    vkDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    vkDeviceCreateInfo.pQueueCreateInfos = &vkDeviceQueueCreateInfo;
    vkDeviceCreateInfo.queueCreateInfoCount = 1;
    vkDeviceCreateInfo.pEnabledFeatures = &vkPhysicalDeviceFeatures;
    vkDeviceCreateInfo.enabledExtensionCount = 0;
    vkDeviceCreateInfo.enabledLayerCount = 0;

    const auto createDeviceResult = vkCreateDevice(vkPhysicalDevice, &vkDeviceCreateInfo, nullptr, &vkDevice);
    if (createDeviceResult != VK_SUCCESS)
    {
        throw std::runtime_error("No compatible devices found\n");
    }

    vkSurfaceKhr = window.generateVulkanSurface(vkInstance);

    i = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        VkBool32 presentationSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(vkPhysicalDevice, i, vkSurfaceKhr, &presentationSupport);

        if (presentationSupport)
        {
            availableQueueFamilies.presentationFamily = i;
        }

        i++;
    }

    window.makeContextCurrent();
}
