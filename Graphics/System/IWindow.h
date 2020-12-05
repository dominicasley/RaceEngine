#pragma once

#include <vulkan/vulkan.h>
#include "../Api/VulkanWindowRequiredExtensions.h"
#include "../../Async/Async.h"
#include "WindowState.h"

class IWindow
{
public:
    virtual void makeContextCurrent() = 0;
    virtual void swapBuffers() const = 0;
    virtual void setMousePosition(int x, int y) = 0;
    [[nodiscard]] virtual VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) = 0;
    [[nodiscard]] virtual VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() = 0;
    [[nodiscard]] virtual bool shouldClose() const = 0;
    [[nodiscard]] virtual bool keyPressed(int key) const = 0;
    [[nodiscard]] virtual const WindowState& state() const = 0;
    [[nodiscard]] virtual std::tuple<double, double> mousePosition() = 0;
    [[nodiscard]] virtual float delta() const = 0;

    Observable<std::tuple<int, int>> windowResize;
};
