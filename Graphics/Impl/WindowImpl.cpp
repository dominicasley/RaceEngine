// Window bodies. Declarations are in Graphics/System/Window.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
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

module raceengine.graphics;

import :Window;

namespace raceengine
{

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

    // The window's size, and it is overridable **because the parity goldens are a fixed size and the
    // game's window is not**.
    //
    // Both were 1920x1080 until 2026-08-21 23:49, when the window went to 2560x1440. That is a
    // perfectly good thing to want and it silently broke both frame gates: a capture is the
    // framebuffer, so every comparison started failing on a size mismatch against goldens that were
    // 1920x1080, and there was no way to run either gate without editing this line by hand. A gate
    // nobody can run is a gate that gets ignored.
    //
    // So the *game's* size stays whatever Dominic wants it to be, and `scripts/verify-parity.sh` asks
    // for the goldens' size through these. Deliberately not tied to `RACEENGINE_UNATTENDED`: an
    // unattended run is not necessarily a gate run, and a capture at a size nobody chose is how this
    // problem started.
    const auto sizeFrom = [](const char* name, const int fallback)
    {
        const auto* stated = std::getenv(name);
        if (stated == nullptr)
        {
            return fallback;
        }

        const auto value = std::atoi(stated);

        return value > 0 ? value : fallback;
    };

    windowState.windowWidth = sizeFrom("RACEENGINE_FRAME_WIDTH", 2560);
    windowState.windowHeight = sizeFrom("RACEENGINE_FRAME_HEIGHT", 1440);
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
