#pragma once

#include <vector>
#include <vulkan/vulkan.h>
#include "../System/IWindow.h"

struct QueueFamily
{
    uint32_t graphicsFamily;
    uint32_t presentationFamily;
};

class VulkanRenderer
{
private:
    VkInstance vkInstance;
    VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice vkDevice;
    VkSurfaceKHR vkSurfaceKhr;
    QueueFamily availableQueueFamilies;
    IWindow& window;

    const std::vector<const char *> validationLayers = {
            "VK_LAYER_KHRONOS_validation"
    };

    void init();

public:
    explicit VulkanRenderer(const IWindow& window);
    ~VulkanRenderer();
};
