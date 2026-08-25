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
    // The telemetry toggle. A driver flagging an interesting moment must not have to leave the seat
    // to do it, which is the whole of why it is a key and not a command line switch.
    T = GLFW_KEY_T,
    // The free-camera toggle. A key rather than an environment variable for the telemetry key's
    // reason: deciding to go and look at something from outside the car is a thing you decide while
    // driving, and a knob that needs the process restarted is one nobody reaches for.
    C = GLFW_KEY_C,
    // Print where the camera is standing and which way it points. A key for the same reason again,
    // and for one more: a view is the hardest thing in this project to describe in words, so what a
    // driver saw can only be handed over by being reproduced, and this is what makes that possible.
    P = GLFW_KEY_P,
    // The wiper stalk, stepped one setting per press. A key for the reason the telemetry toggle is
    // one: it is a control a driver reaches for *because* of what they are looking at, and a wiper
    // setting chosen before the process started is a wiper setting chosen before the rain.
    V = GLFW_KEY_V,
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

} // namespace raceengine
