module;

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

export module raceengine.input:DirectInputContract;

import :InputBackend;

namespace raceengine
{

// The facts about DirectInput's ABI that are not COM calls, stated where a machine with no Windows
// on it can compile them and a test can check them.
//
// This is the half of "written twice from the start" that is actually verifiable here. The COM
// calls cannot be run on this machine and will not be until Windows ships; the conversions around
// them — a device's identity, which axis carries which role, what a torque of one means as a
// magnitude, what range the axes are configured to — are where the two platforms genuinely
// disagree, and every one of them is a pure function of numbers.

// DirectInput states an effect's magnitude over ±DI_FFNOMINALMAX, which is 10000 and is not a power
// of two. evdev's is ±32767. A force tuned against either and applied to the other is out by more
// than three to one, which is the sort of difference that gets called "the wheel feels different on
// Windows" and is never traced to a constant.
export inline constexpr double directInputNominalMaximum = 10000.0;

// The range every axis is configured to on open, so a `DeviceSample` means the same thing whichever
// backend produced it. DirectInput's own default varies by device; evdev reports whatever the
// report descriptor says, which on this wheel is 0..65535. Setting one explicitly is what stops a
// calibration being a statement about which backend read it.
export inline constexpr std::int32_t directInputAxisMinimum = 0;
export inline constexpr std::int32_t directInputAxisMaximum = 65535;

// Byte offsets into DIJOYSTATE2, spelled out so the role mapping can be stated and tested without
// <dinput.h> anywhere near it.
export inline constexpr std::uint32_t directInputOffsetX = 0;
export inline constexpr std::uint32_t directInputOffsetY = 4;
export inline constexpr std::uint32_t directInputOffsetZ = 8;
export inline constexpr std::uint32_t directInputOffsetRz = 20;

export [[nodiscard]] constexpr std::optional<InputAxis> directInputAxisRole(const std::uint32_t offset)
{
    switch (offset)
    {
    case directInputOffsetX:
        return InputAxis::Steering;
    case directInputOffsetY:
        return InputAxis::Throttle;
    case directInputOffsetZ:
        return InputAxis::Brake;
    case directInputOffsetRz:
        return InputAxis::Clutch;
    default:
        break;
    }

    return std::nullopt;
}

// DirectInput names a HID device with a product GUID whose first field packs the vendor and the
// product and whose last eight bytes spell PIDVID. Decoding it is how one profile finds one wheel
// on both platforms: the instance GUID beside it is explicitly not stable across a replug, which is
// exactly the mistake keying on an evdev node would be.
export [[nodiscard]] constexpr std::optional<DeviceIdentity>
identityFromProductGuid(const std::array<std::uint8_t, 16>& guid)
{
    constexpr auto signature = std::array<std::uint8_t, 8>{0x00, 0x00, 'P', 'I', 'D', 'V', 'I', 'D'};

    for (auto index = std::size_t{0}; index < signature.size(); index++)
    {
        if (guid[8 + index] != signature[index])
        {
            return std::nullopt;
        }
    }

    // Data1 is little endian here and on every machine this runs on: vendor in the low half,
    // product in the high one.
    const auto vendor =
        static_cast<std::uint16_t>(static_cast<std::uint32_t>(guid[0]) | (static_cast<std::uint32_t>(guid[1]) << 8u));
    const auto product =
        static_cast<std::uint16_t>(static_cast<std::uint32_t>(guid[2]) | (static_cast<std::uint32_t>(guid[3]) << 8u));

    return DeviceIdentity{.vendor = vendor, .product = product};
}

// A torque fraction to the magnitude a DICONSTANTFORCE carries. Rounded rather than truncated: a
// truncation is a dead band around zero that widens with the device's own quantisation, and a wheel
// with a dead band at centre is the first thing a driver notices.
export [[nodiscard]] constexpr std::int32_t directInputMagnitude(const double torqueFraction)
{
    const auto clamped = torqueFraction < -1.0 ? -1.0 : (torqueFraction > 1.0 ? 1.0 : torqueFraction);
    const auto scaled = clamped * directInputNominalMaximum;

    return static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
}

} // namespace raceengine
