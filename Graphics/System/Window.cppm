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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

export module raceengine.graphics:Window;

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
    virtual void swapBuffers() const = 0;
    virtual void setMousePosition(int x, int y) = 0;
    virtual void setCursorMode(CursorMode mode) = 0;
    // F11 is bound by the window itself rather than handed to the game as a key to poll, for
    // the reason the occlusion prepass builds its own camera: there is nothing here for a
    // level to keep in step. The platform reports a press once where keyPressed reports it
    // every frame it is held, so a polled binding would need a debounce whose only purpose is
    // to rediscover an edge the platform already had. A game wanting a different binding, or
    // a settings menu, calls this and never sees F11.
    virtual void setFullscreen(bool enable) = 0;
    // Motion since the previous call, in pixels. Zero when the window is not focused, when
    // the run is unattended, and on the first call after either changes — a controller that
    // steers from this can never be handed a jump it did not earn.
    [[nodiscard]] virtual std::tuple<double, double> mouseDelta() = 0;
    [[nodiscard]] virtual bool shouldClose() const = 0;
    [[nodiscard]] virtual bool fullscreen() const = 0;
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

// The window's Vulkan half, kept off IWindow for the same reason the renderer's three seams are
// not one interface: IWindow is what Engine::window() hands the game, and a game has no business
// being able to make a VkSurfaceKHR. Only the Vulkan backend takes this, and only the
// composition root — which owns the concrete window — is in a position to hand it over.
export class IVulkanSurfaceSource
{
public:
    IVulkanSurfaceSource() = default;
    IVulkanSurfaceSource(const IVulkanSurfaceSource&) = delete;
    IVulkanSurfaceSource(IVulkanSurfaceSource&&) = delete;
    IVulkanSurfaceSource& operator=(const IVulkanSurfaceSource&) = delete;
    IVulkanSurfaceSource& operator=(IVulkanSurfaceSource&&) = delete;
    virtual ~IVulkanSurfaceSource() = default;

    [[nodiscard]] virtual VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) = 0;
    [[nodiscard]] virtual VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() = 0;
};

export class GLFWWindow : public IWindow, public IVulkanSurfaceSource
{
private:
    mutable double _delta;
    mutable double _frameTime;
    mutable double _avgFrameRate;
    mutable int _frameCount;

    WindowState windowState;
    spdlog::logger& logger;
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
    bool inFullscreen = false;
    // Where to come back to. A fullscreen window reports the monitor's origin as its position
    // and the monitor's size as its size, so the placement to restore has to be taken before
    // the switch rather than asked for after it.
    int windowedX = 0;
    int windowedY = 0;
    int windowedWidth = 0;
    int windowedHeight = 0;
    GLFWwindow* window;
    static void windowResized(GLFWwindow* window, int width, int height);
    static void cursorPositionChanged(GLFWwindow* window, double x, double y);
    static void keyChanged(GLFWwindow* window, int key, int scancode, int action, int mods);
    // The monitor the window is most on, by area of overlap. GLFW has no such notion — a window
    // belongs to no monitor until it is fullscreen — and the primary one is the wrong answer on
    // any desk with two screens: it would throw the game onto the other one.
    [[nodiscard]] GLFWmonitor* coveringMonitor() const;
    [[nodiscard]] bool focused() const;

public:
    explicit GLFWWindow(spdlog::logger& logger);
    ~GLFWWindow();
    void swapBuffers() const override;
    void setMousePosition(int x, int y) override;
    void setCursorMode(CursorMode mode) override;
    void setFullscreen(bool enable) override;
    [[nodiscard]] std::tuple<double, double> mouseDelta() override;
    [[nodiscard]] VkSurfaceKHR generateVulkanSurface(const VkInstance& vkInstance) override;
    [[nodiscard]] VulkanWindowRequiredExtensions getRequiredVulkanWindowExtensions() override;
    [[nodiscard]] bool shouldClose() const override;
    [[nodiscard]] bool fullscreen() const override;
    [[nodiscard]] bool keyPressed(Key key) const override;
    [[nodiscard]] const WindowState& state() const override;
    [[nodiscard]] std::tuple<double, double> mousePosition() override;
    [[nodiscard]] float delta() const override;
};

GLFWWindow::GLFWWindow(spdlog::logger& logger) :
    _delta(0),
    _frameTime(0),
    _avgFrameRate(0),
    _frameCount(0),
    windowState({0, 0, 0, 0}),
    logger(logger),
    unattended(std::getenv("RACEENGINE_UNATTENDED") != nullptr || std::getenv("RACEENGINE_DUMP_FRAME") != nullptr)
{
    // Thrown rather than logged and thrown: the exception is the report, and main's boundary is
    // where it is read. Two channels for one problem reads as two problems.
    if (!glfwInit())
    {
        throw std::runtime_error("GLFW would not initialise");
    }

    if (glfwVulkanSupported())
    {
        logger.info("Vulkan support detected");
    }

    // The renderer owns the device outright; GLFW is asked for a window and nothing else.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // Games block on asset loading before the first frame, and nothing pumps events until
    // then, so a window mapped here answers no window-manager pings and gets reported as
    // hung. It stays hidden until the first frame is on screen, and does not steal focus
    // when it appears.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);

    // Fullscreen here adopts the monitor's current video mode rather than asking for one of
    // its own, so nothing was taken from the desktop that alt-tabbing has to give back. The
    // default — minimise on focus loss — would only cost a second screen its game.
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);

    windowState.windowWidth = 1920;
    windowState.windowHeight = 1080;
    window = glfwCreateWindow(windowState.windowWidth, windowState.windowHeight, "Quack!", nullptr, nullptr);

    if (!window)
    {
        const char* description = nullptr;
        const auto code = glfwGetError(&description);

        glfwTerminate();

        throw std::runtime_error(
            "GLFW would not create a window: " + std::string(description == nullptr ? "no description" : description) +
            " (code " + std::to_string(code) + ")");
    }

    glfwSetWindowPos(window, 150, 150);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, windowResized);
    glfwSetCursorPosCallback(window, cursorPositionChanged);
    glfwSetKeyCallback(window, keyChanged);
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

    // GLFW's own words travel with the exception rather than being logged beside a message that
    // dropped them: the caller (VulkanRenderer::init) turns this into the reported reason the
    // device would not come up, and "failed to create window surface!" said nothing about why.
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, &vkSurfaceKhr) != VK_SUCCESS)
    {
        const char* description = nullptr;
        const auto code = glfwGetError(&description);

        throw std::runtime_error("GLFW would not create a Vulkan window surface: " +
                                 std::string(description == nullptr ? "no description" : description) + " (code " +
                                 std::to_string(code) + ")");
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

    // The renderer has presented by the time the loop reaches here, so the window reveals
    // rendered content rather than the blank fill it was created with.
    if (!windowShown && !unattended)
    {
        glfwShowWindow(window);
        windowShown = true;
    }

    glfwPollEvents();
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

void GLFWWindow::keyChanged(GLFWwindow* window, const int key, int, const int action, int)
{
    auto caller = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));

    // GLFW_PRESS and not GLFW_REPEAT: a held F11 is one instruction, not a toggle per
    // auto-repeat interval. An unattended run owns no screen to take over.
    if (caller->unattended || action != GLFW_PRESS)
    {
        return;
    }

    if (key == GLFW_KEY_F11)
    {
        caller->setFullscreen(!caller->inFullscreen);
    }
}

GLFWmonitor* GLFWWindow::coveringMonitor() const
{
    auto windowX = 0;
    auto windowY = 0;
    auto width = 0;
    auto height = 0;
    glfwGetWindowPos(window, &windowX, &windowY);
    glfwGetWindowSize(window, &width, &height);

    auto monitorCount = 0;
    auto monitors = glfwGetMonitors(&monitorCount);

    auto best = glfwGetPrimaryMonitor();
    auto bestOverlap = 0;

    for (auto index = 0; index < monitorCount; index++)
    {
        auto monitorX = 0;
        auto monitorY = 0;
        glfwGetMonitorPos(monitors[index], &monitorX, &monitorY);

        const auto mode = glfwGetVideoMode(monitors[index]);
        if (mode == nullptr)
        {
            continue;
        }

        const auto overlapWidth =
            std::max(0, std::min(windowX + width, monitorX + mode->width) - std::max(windowX, monitorX));
        const auto overlapHeight =
            std::max(0, std::min(windowY + height, monitorY + mode->height) - std::max(windowY, monitorY));

        if (const auto overlap = overlapWidth * overlapHeight; overlap > bestOverlap)
        {
            bestOverlap = overlap;
            best = monitors[index];
        }
    }

    return best;
}

void GLFWWindow::setFullscreen(const bool enable)
{
    // An unattended run maps no window at all, so there is no screen for one to take.
    if (unattended || enable == inFullscreen)
    {
        return;
    }

    if (enable)
    {
        const auto monitor = coveringMonitor();
        const auto mode = monitor == nullptr ? nullptr : glfwGetVideoMode(monitor);

        // A headless session or a monitor that will not describe itself leaves the window
        // exactly as it was: reported, and windowed, rather than resized to a guess.
        if (mode == nullptr)
        {
            logger.warn("Fullscreen was not entered: no monitor could be measured");
            return;
        }

        glfwGetWindowPos(window, &windowedX, &windowedY);
        glfwGetWindowSize(window, &windowedWidth, &windowedHeight);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    }
    else
    {
        glfwSetWindowMonitor(window, nullptr, windowedX, windowedY, windowedWidth, windowedHeight, GLFW_DONT_CARE);
    }

    inFullscreen = enable;

    // The pointer is somewhere else entirely after the window has changed size and monitor,
    // and that jump is not motion the player asked the camera for.
    resyncCursor = true;
}

bool GLFWWindow::fullscreen() const
{
    return inFullscreen;
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
