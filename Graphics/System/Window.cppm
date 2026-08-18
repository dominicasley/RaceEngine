module;

#define GLFW_INCLUDE_VULKAN
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif

#include <GLFW/glfw3.h>
#ifdef _WIN32
#include <Dwmapi.h>
#include <GLFW/glfw3native.h>
#endif
#include <spdlog/logger.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

export module raceengine.graphics:Window;

import :GraphicsApi;

namespace raceengine
{

export struct WindowState
{
    double mouseX;
    double mouseY;
    int windowWidth;
    int windowHeight;
};

export struct VulkanWindowRequiredExtensions
{
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

// Captured hides the cursor and frees it from the screen's edges, which is what mouse-look
// needs: the platform reports unbounded motion instead of a position that stops at the
// desktop boundary. Warping the cursor to the window centre each frame is the alternative
// and it does not work — on X11 the warp is a request to the server, so a read-back in the
// same frame can report the requested centre while the pointer has not moved yet.
export enum class CursorMode { Normal, Captured };

export class IWindow
{
public:
    virtual ~IWindow() = default;
    virtual void makeContextCurrent() = 0;
    virtual void swapBuffers() const = 0;
    virtual void setMousePosition(int x, int y) = 0;
    virtual void setCursorMode(CursorMode mode) = 0;
    // Motion since the previous call, in pixels. Zero when the window is not focused, when
    // the run is unattended, and on the first call after either changes — a controller that
    // steers from this can never be handed a jump it did not earn.
    [[nodiscard]] virtual std::tuple<double, double> mouseDelta() = 0;
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
    GraphicsApi graphicsApi;
    // Unattended runs (smoke gates, frame captures) belong to no one at the keyboard: the
    // window never appears and every input path reports nothing, so an automated run
    // cannot steal focus, pull the cursor, or take keystrokes from the desktop session.
    // It also makes captures reproducible, since controllers see a cursor that never moves.
    bool unattended;
    mutable bool windowShown = false;
    CursorMode cursorMode = CursorMode::Normal;
    // Baseline for mouseDelta. resyncCursor forces the next call to re-baseline and report
    // no motion: the platform's cursor position jumps when capture is entered or left, and
    // that jump is not something the user did.
    double lastCursorX = 0.0;
    double lastCursorY = 0.0;
    bool resyncCursor = true;
    GLFWwindow* window;
    static void windowResized(GLFWwindow* window, int width, int height);
    static void cursorPositionChanged(GLFWwindow* window, double x, double y);
    [[nodiscard]] bool focused() const;

public:
    explicit GLFWWindow(spdlog::logger& logger, GraphicsApi graphicsApi);
    ~GLFWWindow();
    void makeContextCurrent() override;
    void swapBuffers() const override;
    void setMousePosition(int x, int y) override;
    void setCursorMode(CursorMode mode) override;
    [[nodiscard]] std::tuple<double, double> mouseDelta() override;
    [[nodiscard]] VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) override;
    [[nodiscard]] VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() override;
    [[nodiscard]] bool shouldClose() const override;
    [[nodiscard]] bool keyPressed(Key key) const override;
    [[nodiscard]] const WindowState& state() const override;
    [[nodiscard]] std::tuple<double, double> mousePosition() override;
    [[nodiscard]] float delta() const override;
};

GLFWWindow::GLFWWindow(spdlog::logger& logger, GraphicsApi graphicsApi) :
    _delta(0),
    _frameTime(0),
    _avgFrameRate(0),
    _frameCount(0),
    windowState({0, 0, 0, 0}),
    logger(logger),
    graphicsApi(graphicsApi),
    unattended(std::getenv("RACEENGINE_UNATTENDED") != nullptr || std::getenv("RACEENGINE_DUMP_FRAME") != nullptr)
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

    if (graphicsApi == GraphicsApi::Vulkan)
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }
    else
    {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }

    // Games block on asset loading before the first frame, and nothing pumps events until
    // then, so a window mapped here answers no window-manager pings and gets reported as
    // hung. It stays hidden until the first frame is on screen, and does not steal focus
    // when it appears.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);

    windowState.windowWidth = 1920;
    windowState.windowHeight = 1080;
    window = glfwCreateWindow(windowState.windowWidth, windowState.windowHeight, "Quack!", nullptr, nullptr);

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

    if (graphicsApi == GraphicsApi::OpenGL)
    {
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
    }
}

GLFWWindow::~GLFWWindow()
{
    if (window)
    {
        glfwDestroyWindow(window);
    }

    glfwTerminate();
}

VkSurfaceKHR GLFWWindow::generateVulkanSurface(const VkInstance& vkInstance)
{
    VkSurfaceKHR vkSurfaceKhr;

    if (glfwCreateWindowSurface(vkInstance, window, nullptr, &vkSurfaceKhr) != VK_SUCCESS)
    {
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

    return VulkanWindowRequiredExtensions{glfwExtensionCount, glfwExtensions};
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

    if (graphicsApi == GraphicsApi::OpenGL)
    {
        glfwSwapBuffers(window);
    }

    // Both backends have presented a frame by the time the loop reaches here, so the
    // window reveals rendered content rather than the blank fill it was created with.
    if (!windowShown && !unattended)
    {
        glfwShowWindow(window);
        windowShown = true;
    }

    glfwPollEvents();
}

void GLFWWindow::makeContextCurrent()
{
    glfwMakeContextCurrent(window);
}

void GLFWWindow::windowResized(GLFWwindow* window, int width, int height)
{
    auto caller = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    caller->windowState.windowWidth = width;
    caller->windowState.windowHeight = height;

    for (const auto& callback : caller->resizeCallbacks)
    {
        callback(width, height);
    }
}

bool GLFWWindow::keyPressed(Key key) const
{
    if (unattended || !focused())
    {
        return false;
    }

    return glfwGetKey(window, static_cast<int>(key)) == GLFW_PRESS;
}

void GLFWWindow::setMousePosition(int x, int y)
{
    // Warping the cursor of a window the user is not looking at pulls the pointer out of
    // whatever they are actually doing.
    if (unattended || !focused())
    {
        return;
    }

    glfwSetCursorPos(window, x, y);
}

void GLFWWindow::setCursorMode(const CursorMode mode)
{
    // An unattended run owns no cursor, so capturing one would take the pointer away from
    // whoever is actually using the machine.
    if (unattended)
    {
        return;
    }

    cursorMode = mode;
    glfwSetInputMode(window, GLFW_CURSOR, mode == CursorMode::Captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);

    // Raw motion skips the desktop's pointer acceleration, which is calibrated for a cursor
    // travelling to a target rather than for a camera. Only meaningful while captured.
    if (glfwRawMouseMotionSupported())
    {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, mode == CursorMode::Captured ? GLFW_TRUE : GLFW_FALSE);
    }

    resyncCursor = true;
}

std::tuple<double, double> GLFWWindow::mouseDelta()
{
    if (unattended || !focused())
    {
        // Losing focus also loses the capture, so the position on return is unrelated to the
        // one we left.
        resyncCursor = true;
        return std::make_tuple(0.0, 0.0);
    }

    double x, y;
    glfwGetCursorPos(window, &x, &y);

    if (resyncCursor)
    {
        lastCursorX = x;
        lastCursorY = y;
        resyncCursor = false;

        return std::make_tuple(0.0, 0.0);
    }

    const auto deltaX = x - lastCursorX;
    const auto deltaY = y - lastCursorY;
    lastCursorX = x;
    lastCursorY = y;

    return std::make_tuple(deltaX, deltaY);
}

bool GLFWWindow::focused() const
{
    return glfwGetWindowAttrib(window, GLFW_FOCUSED) == GLFW_TRUE;
}

const WindowState& GLFWWindow::state() const
{
    return windowState;
}

void GLFWWindow::cursorPositionChanged(GLFWwindow* window, double x, double y)
{
    auto caller = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));

    // The cursor still travels across an unattended or unfocused window; letting that
    // reach the state would hand controllers the movement mousePosition withholds.
    if (caller->unattended || !caller->focused())
    {
        return;
    }

    caller->windowState.mouseX = x;
    caller->windowState.mouseY = y;
}

std::tuple<double, double> GLFWWindow::mousePosition()
{
    double x, y;

    // Controllers steer from the change between samples, so repeating the previous sample
    // reads as no movement at all. Unattended runs render the spawn view however the mouse
    // moves, and an unfocused window never swings the camera with motion meant elsewhere.
    if (unattended)
    {
        x = windowState.windowWidth / 2.0;
        y = windowState.windowHeight / 2.0;
    }
    else if (!focused())
    {
        x = windowState.mouseX;
        y = windowState.mouseY;
    }
    else
    {
        glfwGetCursorPos(window, &x, &y);
    }

    windowState.mouseX = x;
    windowState.mouseY = y;

    return std::make_tuple(x, y);
}

float GLFWWindow::delta() const
{
    return static_cast<float>(_delta);
}

} // namespace raceengine
