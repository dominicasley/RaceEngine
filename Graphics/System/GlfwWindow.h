#pragma once

#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <spdlog/logger.h>

#include "IWindow.h"

class GLFWWindow : public IWindow
{
private:
    spdlog::logger& logger;
    GLFWwindow* window;
    static void windowResized(GLFWwindow* window, int width, int height);

public:
    explicit GLFWWindow(spdlog::logger& logger);
    ~GLFWWindow();
    [[nodiscard]] VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) override;
    [[nodiscard]] VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() override;
    void makeContextCurrent() override;
    void swapBuffers() const override;
    [[nodiscard]] bool shouldClose() const override;
    [[nodiscard]] bool getKeyPressed(int key) const override;
};
