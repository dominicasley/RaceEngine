#include <stdexcept>
#include "GlfwWindow.h"
#include <Dwmapi.h>

GLFWWindow::GLFWWindow(spdlog::logger &logger) :
    logger(logger),
    _delta(0),
    _frameTime(0),
    _avgFrameRate(0),
    _frameCount(0),
    windowState({0, 0, 0, 0})
{
    if (!glfwInit())
    {
        logger.error("GLFW failed to initialize!");
        throw std::runtime_error("GLFW failed to initialize!");
    }

    if (glfwVulkanSupported())
    {
        logger.info("Vulkan support detected");
    }

    //glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    windowState.windowWidth = 2880;
    windowState.windowHeight = 1620;
    window = glfwCreateWindow(windowState.windowWidth,  windowState.windowHeight, "Quack!", nullptr, nullptr);
    glfwSetWindowPos(window, 150, 150);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, windowResized);
    glfwSetCursorPosCallback(window, cursorPositionChanged);

    if (!window)
    {
        glfwTerminate();
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
}

GLFWWindow::~GLFWWindow()
{
    glfwTerminate();
}

VkSurfaceKHR GLFWWindow::generateVulkanSurface(const VkInstance &vkInstance)
{
    VkSurfaceKHR vkSurfaceKhr;

    if (glfwCreateWindowSurface(vkInstance, window, nullptr, &vkSurfaceKhr) != VK_SUCCESS)
    {
        const char *description;
        int code = glfwGetError(&description);

        logger.error("{}, {}", code, description);

        throw std::runtime_error("failed to create window surface!");
    }

    return vkSurfaceKhr;
}

VulkanWindowRequiredExtensions GLFWWindow::getRequiredVulkanWindowExtensions()
{
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    return VulkanWindowRequiredExtensions{
        glfwExtensionCount,
        glfwExtensions
    };
}


bool GLFWWindow::shouldClose() const
{
    return glfwWindowShouldClose(window);
}

void GLFWWindow::swapBuffers() const
{
    const auto currentTime = glfwGetTime();
    _delta = (currentTime - _frameTime);
    _frameTime = currentTime;
    _frameCount++;

    if (currentTime - _avgFrameRate > 60)
    {
        logger.info("Last Frame Time: {}s", _delta);
        logger.info("Average FPS: {}", _frameCount / (currentTime - _avgFrameRate));
        _avgFrameRate = currentTime;
        _frameCount = 0;
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void GLFWWindow::makeContextCurrent()
{
    glfwMakeContextCurrent(window);
}

void GLFWWindow::windowResized(GLFWwindow *window, int width, int height)
{
    auto caller = reinterpret_cast<GLFWWindow *>(glfwGetWindowUserPointer(window));
    caller->windowState.windowWidth = width;
    caller->windowState.windowHeight = height;
    caller->windowResize.next(std::make_tuple(width, height));
}

bool GLFWWindow::keyPressed(int key) const
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}

void GLFWWindow::setMousePosition(int x, int y)
{
    glfwSetCursorPos(window, x, y);
}

const WindowState &GLFWWindow::state() const
{
    return windowState;
}

void GLFWWindow::cursorPositionChanged(GLFWwindow *window, double x, double y)
{
    auto caller = reinterpret_cast<GLFWWindow *>(glfwGetWindowUserPointer(window));
    caller->windowState.mouseX = x;
    caller->windowState.mouseY = y;
}

std::tuple<double, double> GLFWWindow::mousePosition()
{
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    windowState.mouseX = x;
    windowState.mouseY = y;

    return std::make_tuple(x, y);
}

float GLFWWindow::delta() const
{
    return static_cast<float>(_delta);
}

