#include <stdexcept>
#include "GlfwWindow.h"
#include <Dwmapi.h>

GLFWWindow::GLFWWindow(spdlog::logger& logger) : logger(logger)
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


    window = glfwCreateWindow(1920, 1080, "Quack!", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetWindowPos(window, 100, 100);
    glfwSetFramebufferSizeCallback(window, windowResized);

    if (!window)
    {
        glfwTerminate();
    }

    glfwMakeContextCurrent(window);
}

GLFWWindow::~GLFWWindow()
{
    glfwTerminate();
}

VkSurfaceKHR GLFWWindow::generateVulkanSurface(const VkInstance& vkInstance)
{
    VkSurfaceKHR vkSurfaceKhr;

    if (glfwCreateWindowSurface(vkInstance, window, nullptr, &vkSurfaceKhr) != VK_SUCCESS) {
        const char* description;
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

    return VulkanWindowRequiredExtensions {
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
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void GLFWWindow::makeContextCurrent()
{
    glfwMakeContextCurrent(window);
}

void GLFWWindow::windowResized(GLFWwindow *window, int width, int height)
{
    auto caller = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    caller->windowResize.next(std::make_tuple(width, height));
}

bool GLFWWindow::getKeyPressed(int key) const
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}

