#pragma once

#include <vulkan/vulkan.h>
#include "../Api/VulkanWindowRequiredExtensions.h"
#include "../../Async/Async.h"

class IWindow
{
public:
    virtual VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) = 0;
    virtual VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() = 0;
    virtual void makeContextCurrent() = 0;
    virtual void swapBuffers() const = 0;
    virtual bool shouldClose() const = 0;
    virtual bool getKeyPressed(int key) const = 0;

    Observable<std::tuple<int, int>> windowResize;
};
