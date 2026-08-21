module;

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

export module raceengine.input:InputMapping;

import :DeviceProfile;
import :DriverInput;
import :InputBackend;

namespace raceengine
{

// A raw device report, a calibration and the car's own steering travel, to the one struct the
// physics takes. Pure, so the whole of what a wheel does to a car can be pinned by tests with no
// wheel: every failure this layer can have — an inverted pedal, a rack that pulls to one side, a
// brake that reaches full pressure at a third of the cell, a rim geared to the wrong lock — is a
// wrong number out of this function and is visible as one.
export [[nodiscard]] inline DriverInput deviceDriverInput(const DeviceProfile& profile,
                                                          const SteeringGeometry& geometry, const DeviceSample& sample)
{
    const auto raw = [&sample](const InputAxis axis)
    {
        return static_cast<double>(sample.axes[axisIndex(axis)]);
    };

    const auto held = [&profile, &sample](const DriverAction action)
    {
        const auto bit = profile.buttons[static_cast<std::size_t>(action)];

        return bit >= 0 && bit < 64 && (sample.buttons & (std::uint64_t{1} << static_cast<std::uint64_t>(bit))) != 0;
    };

    const auto rim = normaliseBipolar(profile.axes[axisIndex(InputAxis::Steering)], raw(InputAxis::Steering));
    const auto brakeTravel = normaliseUnipolar(profile.axes[axisIndex(InputAxis::Brake)], raw(InputAxis::Brake));

    return DriverInput{.steering = rackFromRim(geometry, rim),
                       .throttle =
                           normaliseUnipolar(profile.axes[axisIndex(InputAxis::Throttle)], raw(InputAxis::Throttle)),
                       // Force through the curve, not travel through a ramp. The one line where a
                       // load cell stops being an expensive potentiometer.
                       .brake = brakePressure(profile.brake, brakeTravel),
                       .clutch = normaliseUnipolar(profile.axes[axisIndex(InputAxis::Clutch)], raw(InputAxis::Clutch)),
                       .handbrake = held(DriverAction::Handbrake),
                       .upshift = held(DriverAction::Upshift),
                       .downshift = held(DriverAction::Downshift)};
}

// What the keys say, before anything has been made of it. Separated from the window so the mapping
// below is testable and so that a run with no window — every gate this repository has — reaches the
// same code with everything false.
export struct KeyboardDemand
{
    bool left = false;
    bool right = false;
    bool accelerate = false;
    bool brake = false;
    bool handbrake = false;
    bool upshift = false;
    bool downshift = false;
};

// Both keys down is neither, which is the only behaviour that does not surprise: the alternative —
// last one wins — makes a car that was going straight lurch when a key is released.
export [[nodiscard]] inline DriverInput keyboardDriverInput(const KeyboardDemand& demand)
{
    return DriverInput{.steering = (demand.right ? 1.0 : 0.0) - (demand.left ? 1.0 : 0.0),
                       .throttle = demand.accelerate ? 1.0 : 0.0,
                       .brake = demand.brake ? 1.0 : 0.0,
                       .clutch = 0.0,
                       .handbrake = demand.handbrake,
                       .upshift = demand.upshift,
                       .downshift = demand.downshift};
}

// The identity inside a joystick GUID of the kind SDL defines and GLFW reports, which is thirty-two
// hex digits with the bus, the vendor, the product and the version each little endian and each
// followed by a zeroed pair.
//
// Here because it is the only way to recognise a device through a window library that will not name
// one: nothing in this engine calls that API today — which is what keeps GLFW from opening a single
// input node, and is the whole of how the wheel is kept to one reader — and the day something does,
// this is what it filters against. A gamepad layer built on GLFW that could not tell the wheel from
// a pad would open the wheel as a second reader, and two readers on one device is intermittent
// dropouts that read as a physics fault.
export [[nodiscard]] inline std::optional<DeviceIdentity> identityFromJoystickGuid(const std::string_view guid)
{
    if (guid.size() < 20)
    {
        return std::nullopt;
    }

    const auto digit = [](const char character) -> std::optional<std::uint32_t>
    {
        if (character >= '0' && character <= '9')
        {
            return static_cast<std::uint32_t>(character - '0');
        }

        if (character >= 'a' && character <= 'f')
        {
            return static_cast<std::uint32_t>(character - 'a') + 10u;
        }

        if (character >= 'A' && character <= 'F')
        {
            return static_cast<std::uint32_t>(character - 'A') + 10u;
        }

        return std::nullopt;
    };

    const auto byteAt = [&guid, &digit](const std::size_t offset) -> std::optional<std::uint32_t>
    {
        const auto high = digit(guid[offset]);
        const auto low = digit(guid[offset + 1]);

        if (!high || !low)
        {
            return std::nullopt;
        }

        return (*high << 4u) | *low;
    };

    // Two bytes, little endian: the first pair of hex digits is the low half.
    const auto word = [&byteAt](const std::size_t offset) -> std::optional<std::uint16_t>
    {
        const auto low = byteAt(offset);
        const auto high = byteAt(offset + 2);

        if (!low || !high)
        {
            return std::nullopt;
        }

        return static_cast<std::uint16_t>(*low | (*high << 8u));
    };

    const auto vendor = word(8);
    const auto product = word(16);

    if (!vendor || !product)
    {
        return std::nullopt;
    }

    return DeviceIdentity{.vendor = *vendor, .product = *product};
}

} // namespace raceengine
