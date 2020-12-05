#pragma once

#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <spdlog/logger.h>

#include "IWindow.h"
#include "WindowState.h"

class GLFWWindow : public IWindow
{
private:
    mutable double _delta;
    mutable double _frameTime;
    mutable double _avgFrameRate;
    mutable int _frameCount;

    WindowState windowState;
    spdlog::logger& logger;
    GLFWwindow* window;
    static void windowResized(GLFWwindow* window, int width, int height);
    static void cursorPositionChanged(GLFWwindow* window, double x, double y);

public:
    explicit GLFWWindow(spdlog::logger& logger);
    ~GLFWWindow();
    void makeContextCurrent() override;
    void swapBuffers() const override;
    void setMousePosition(int x, int y) override ;
    [[nodiscard]] VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) override;
    [[nodiscard]] VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() override;
    [[nodiscard]] bool shouldClose() const override;
    [[nodiscard]] bool keyPressed(int key) const override;
    [[nodiscard]] const WindowState& state() const override;
    [[nodiscard]] std::tuple<double, double> mousePosition() override;
    [[nodiscard]] float delta() const override;
};
