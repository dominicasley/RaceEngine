#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>
#include "../Api/VulkanWindowRequiredExtensions.h"
#include "WindowState.h"

class IWindow
{
public:
    virtual ~IWindow() = default;
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

    void onResize(std::function<void(int, int)> callback)
    {
        resizeCallbacks.push_back(std::move(callback));
    }

protected:
    std::vector<std::function<void(int, int)>> resizeCallbacks;
};
