module;

#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

export module raceengine.tests.input;

import raceengine;

namespace raceengine::tests
{

// A device that is whatever the test says it is, including absent. A module unit rather than a
// header because the seam it implements is only reachable through `import`, and a header may not.
export class RecordingInputBackend final : public IInputBackend
{
    DeviceDescription description;
    DeviceCapabilities offered;
    std::atomic<bool> present{false};
    std::atomic<bool> isOpen{false};
    std::atomic<bool> vanishing{false};
    std::atomic<std::int32_t> steering{32273};
    std::atomic<std::int32_t> brake{65535};
    std::atomic<std::uint64_t> reports{0};
    std::atomic<double> lastTorque{0.0};

public:
    double rotationRangeAsked = 0.0;
    RecordingInputBackend()
    {
        offered.add(DeviceCapability::ConstantForce);
        offered.add(DeviceCapability::ReadRotationRange);

        description.identity = DeviceIdentity{.vendor = 0x0eb7, .product = 0x0004};
        description.name = "Recording Wheel";
        description.address = "nowhere";
        description.rotationDegrees = 900.0;
        description.capabilities = offered;

        for (auto index = std::size_t{0}; index < inputAxisCount; index++)
        {
            description.axes[index] = AxisBounds{.minimum = 0, .maximum = 65535, .present = true};
        }
    }

    void attach()
    {
        present.store(true);
    }

    // Switched off rather than unplugged: the next read reports the device gone, which is the path
    // a game has to survive without a diagnostic every two milliseconds.
    void vanish()
    {
        vanishing.store(true);
    }

    void turn(const std::int32_t raw)
    {
        steering.store(raw);
    }

    void press(const std::int32_t raw)
    {
        brake.store(raw);
    }

    [[nodiscard]] double torque() const
    {
        return lastTorque.load();
    }

    [[nodiscard]] std::string_view platform() const override
    {
        return "recording";
    }

    [[nodiscard]] std::vector<DeviceDescription> enumerate() override
    {
        if (!present.load())
        {
            return {};
        }

        return {description};
    }

    [[nodiscard]] std::expected<DeviceDescription, std::string> open(const DeviceIdentity identity) override
    {
        if (!present.load() || !(identity == description.identity))
        {
            return std::unexpected("nothing answering to that identity is present");
        }

        isOpen.store(true);
        vanishing.store(false);

        return description;
    }

    void close() override
    {
        isOpen.store(false);
    }

    [[nodiscard]] bool opened() const override
    {
        return isOpen.load();
    }

    [[nodiscard]] std::expected<DeviceSample, std::string> read(std::chrono::milliseconds timeout) override
    {
        std::ignore = timeout;

        if (!isOpen.load())
        {
            return std::unexpected("no device is open");
        }

        if (vanishing.load())
        {
            present.store(false);

            return std::unexpected("the device went away");
        }

        auto sample = DeviceSample{};
        sample.axes[axisIndex(InputAxis::Steering)] = steering.load();
        sample.axes[axisIndex(InputAxis::Throttle)] = 65535;
        sample.axes[axisIndex(InputAxis::Brake)] = brake.load();
        sample.axes[axisIndex(InputAxis::Clutch)] = 65535;
        sample.reports = reports.fetch_add(1) + 1;
        // A real backend stamps every report and the force feedback writer differences those stamps
        // to measure how fast the rim is turning. Left at zero this double skipped that estimator
        // entirely — so the damper read a rim speed of zero whatever the wheel did, and nothing in
        // the suite noticed because every damper case passes a rim speed to `mapRackTorque` by hand.
        sample.timestampNanos = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());

        return sample;
    }

    [[nodiscard]] std::expected<void, std::string> writeTorque(const double torqueFraction) override
    {
        if (!isOpen.load())
        {
            return std::unexpected("no device is open");
        }

        lastTorque.store(torqueFraction);

        return {};
    }

    [[nodiscard]] DeviceCapabilities capabilities() const override
    {
        return offered;
    }

    // The fake confirms whatever it is asked, which is what lets a service test assert the ask.
    [[nodiscard]] std::expected<double, std::string> setRotationRange(const double degrees) override
    {
        rotationRangeAsked = degrees;

        return degrees;
    }

    [[nodiscard]] UpdateRate updateRate() const override
    {
        return UpdateRate{.inputHz = 500.0, .outputHz = 500.0, .measured = true};
    }
};

// A window that reports nothing and does nothing, which is exactly what an unattended run's window
// reports. The keyboard source takes IWindow and this is the whole of what it needs from one.
export class SilentWindow final : public IWindow
{
    WindowState windowState{0.0, 0.0, 1920, 1080};
    bool pressed = false;

public:
    // What a key says. One switch for all of them, because what is under test is which source wins
    // and not which key does what.
    void hold(const bool held)
    {
        pressed = held;
    }

    void swapBuffers() const override
    {
    }

    void setMousePosition(int, int) override
    {
    }

    void setCursorMode(CursorMode) override
    {
    }

    void setFullscreen(bool) override
    {
    }

    [[nodiscard]] std::tuple<double, double> mouseDelta() override
    {
        return {0.0, 0.0};
    }

    [[nodiscard]] bool shouldClose() const override
    {
        return false;
    }

    [[nodiscard]] bool fullscreen() const override
    {
        return false;
    }

    [[nodiscard]] bool keyPressed(Key) const override
    {
        return pressed;
    }

    [[nodiscard]] const WindowState& state() const override
    {
        return windowState;
    }

    [[nodiscard]] std::tuple<double, double> mousePosition() override
    {
        return {0.0, 0.0};
    }

    [[nodiscard]] float delta() const override
    {
        return 0.0f;
    }
};

} // namespace raceengine::tests
