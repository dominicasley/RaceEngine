module;

#define GLFW_INCLUDE_VULKAN
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif

#include <GLFW/glfw3.h>
#ifdef _WIN32
#include <GLFW/glfw3native.h>
#include <Dwmapi.h>
#endif
#include <vulkan/vulkan.h>
#include <spdlog/logger.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

export module raceengine.graphics:Window;

namespace raceengine
{

export struct WindowState {
    double mouseX;
    double mouseY;
    int windowWidth;
    int windowHeight;
};

export struct VulkanWindowRequiredExtensions {
    uint32_t count;
    const char** extensions;
};

export enum class Key : int {
    W = GLFW_KEY_W,
    A = GLFW_KEY_A,
    S = GLFW_KEY_S,
    D = GLFW_KEY_D,
    Escape = GLFW_KEY_ESCAPE,
    Space = GLFW_KEY_SPACE,
    LeftShift = GLFW_KEY_LEFT_SHIFT,
    LeftControl = GLFW_KEY_LEFT_CONTROL
};

export class IWindow
{
public:
    virtual ~IWindow() = default;
    virtual void makeContextCurrent() = 0;
    virtual void swapBuffers() const = 0;
    virtual void setMousePosition(int x, int y) = 0;
    [[nodiscard]] virtual VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) = 0;
    [[nodiscard]] virtual VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() = 0;
    [[nodiscard]] virtual bool shouldClose() const = 0;
    [[nodiscard]] virtual bool keyPressed(Key key) const = 0;
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

export class GLFWWindow : public IWindow
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
    [[nodiscard]] bool keyPressed(Key key) const override;
    [[nodiscard]] const WindowState& state() const override;
    [[nodiscard]] std::tuple<double, double> mousePosition() override;
    [[nodiscard]] float delta() const override;
};

GLFWWindow::GLFWWindow(spdlog::logger &logger) :
    _delta(0),
    _frameTime(0),
    _avgFrameRate(0),
    _frameCount(0),
    windowState({0, 0, 0, 0}),
    logger(logger)
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

    windowState.windowWidth = 1920;
    windowState.windowHeight = 1080;
    window = glfwCreateWindow(windowState.windowWidth,  windowState.windowHeight, "Quack!", nullptr, nullptr);

    if (!window)
    {
        logger.error("GLFW failed to create a window!");
        glfwTerminate();
        throw std::runtime_error("GLFW failed to create a window!");
    }

    glfwSetWindowPos(window, 150, 150);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, windowResized);
    glfwSetCursorPosCallback(window, cursorPositionChanged);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
}

GLFWWindow::~GLFWWindow()
{
    if (window)
    {
        glfwDestroyWindow(window);
    }

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

    if (currentTime - _avgFrameRate > 15)
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

    for (const auto& callback : caller->resizeCallbacks)
    {
        callback(width, height);
    }
}

bool GLFWWindow::keyPressed(Key key) const
{
    return glfwGetKey(window, static_cast<int>(key)) == GLFW_PRESS;
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

} // namespace raceengine
