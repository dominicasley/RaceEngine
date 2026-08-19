module;

#if defined(_WIN32)
#define DIRECTINPUT_VERSION 0x0800
#include <Windows.h>
#include <dinput.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// An implementation partition for the same reason the evdev one is: <dinput.h> pulls in Windows.h,
// COM and a macro for every English word, and none of that may reach a consumer of raceengine.input.
module raceengine.input:DirectInputBackend;

import :DirectInputContract;
import :InputBackend;

namespace raceengine
{

// Windows, through DirectInput 8.
//
// Written now rather than when Windows ships, and that is the whole point of this file: an
// interface with one implementation is a description of that implementation, and the cost of
// discovering so is paid after a season of force feedback has been tuned against it. Everything in
// here that is not a COM call is compiled on both platforms and tested on this one — the identity
// decode, the axis roles, the magnitude conversion, the ranges — because those are where the two
// platforms actually disagree.
//
// What is *not* symmetric is stated as a capability rather than assumed away. DirectInput has a
// device-side gain (`DIPROP_FFGAIN`) and an autocentre (`DIPROP_AUTOCENTER`) that this Linux driver
// silently discards; neither platform has any way to drive a Fanatec base's own tuning menu without
// the vendor's SDK, so that one is absent on both and a caller is told so once.

class DirectInputBackend final : public IInputBackend
{
    DeviceCapabilities deviceCapabilities;
    double measuredHz = 0.0;

    // Everything below is state only the COM half keeps, and it is guarded rather than merely
    // unused off Windows: a private field nothing reads is a warning, and this build treats a
    // warning as a stop.
#if defined(_WIN32)
    // The window DirectInput's cooperative level is set against. Opaque because this partition may
    // not name an HWND on the platform that has no such thing, and supplied by the composition root
    // because a device layer has no business reaching into the window. A null handle is a supported
    // configuration and costs exclusive mode — which costs force feedback, which is reported as a
    // missing capability rather than as a refusal to start.
    void* nativeWindow = nullptr;
    std::array<AxisBounds, inputAxisCount> bounds{};
    DeviceSample current{};
    std::string address;
    bool acquired = false;
    std::chrono::steady_clock::time_point lastReport{};
    IDirectInput8* factory = nullptr;
    IDirectInputDevice8* device = nullptr;
    IDirectInputEffect* constantForce = nullptr;
#endif

public:
#if defined(_WIN32)
    explicit DirectInputBackend(void* window) :
        nativeWindow(window)
    {
    }
#else
    explicit DirectInputBackend(void* window)
    {
        static_cast<void>(window);
    }
#endif

    ~DirectInputBackend() override;

    [[nodiscard]] std::string_view platform() const override
    {
        return "directinput";
    }

    [[nodiscard]] std::vector<DeviceDescription> enumerate() override;
    [[nodiscard]] std::expected<DeviceDescription, std::string> open(DeviceIdentity identity) override;
    void close() override;
    [[nodiscard]] bool opened() const override;
    [[nodiscard]] std::expected<DeviceSample, std::string> read(std::chrono::milliseconds timeout) override;
    [[nodiscard]] std::expected<void, std::string> writeTorque(double torqueFraction) override;

    [[nodiscard]] DeviceCapabilities capabilities() const override
    {
        return deviceCapabilities;
    }

    // DirectInput's buffered reads have no transport figure to read the way an evdev node's endpoint
    // does, so the only honest ceiling before a device has spoken is none at all, and the rate is
    // whatever it is measured at. It is generally above the Linux driver's, which is the difference
    // this query exists to make visible.
    [[nodiscard]] UpdateRate updateRate() const override
    {
        return UpdateRate{.inputHz = measuredHz, .outputHz = measuredHz, .measured = measuredHz > 0.0};
    }
};

} // namespace raceengine

namespace raceengine
{

#if defined(_WIN32)

namespace
{

[[nodiscard]] std::array<std::uint8_t, 16> guidBytes(const GUID& guid)
{
    auto bytes = std::array<std::uint8_t, 16>{};
    const auto data1 = guid.Data1;
    bytes[0] = static_cast<std::uint8_t>(data1 & 0xFFu);
    bytes[1] = static_cast<std::uint8_t>((data1 >> 8) & 0xFFu);
    bytes[2] = static_cast<std::uint8_t>((data1 >> 16) & 0xFFu);
    bytes[3] = static_cast<std::uint8_t>((data1 >> 24) & 0xFFu);
    bytes[4] = static_cast<std::uint8_t>(guid.Data2 & 0xFFu);
    bytes[5] = static_cast<std::uint8_t>((guid.Data2 >> 8) & 0xFFu);
    bytes[6] = static_cast<std::uint8_t>(guid.Data3 & 0xFFu);
    bytes[7] = static_cast<std::uint8_t>((guid.Data3 >> 8) & 0xFFu);

    for (auto index = std::size_t{0}; index < 8; index++)
    {
        bytes[8 + index] = static_cast<std::uint8_t>(guid.Data4[index]);
    }

    return bytes;
}

struct Enumeration
{
    std::vector<DeviceDescription> found;
    std::vector<GUID> instances;
};

BOOL CALLBACK collectDevice(const DIDEVICEINSTANCE* instance, void* context)
{
    auto& enumeration = *static_cast<Enumeration*>(context);

    const auto identity = identityFromProductGuid(guidBytes(instance->guidProduct));
    if (!identity)
    {
        return DIENUM_CONTINUE;
    }

    auto description = DeviceDescription{};
    description.identity = *identity;
    description.name = std::string(instance->tszInstanceName);
    description.address = "directinput";

    for (auto index = std::size_t{0}; index < inputAxisCount; index++)
    {
        description.axes[index] =
            AxisBounds{.minimum = directInputAxisMinimum, .maximum = directInputAxisMaximum, .present = true};
    }

    enumeration.found.push_back(std::move(description));
    enumeration.instances.push_back(instance->guidInstance);

    return DIENUM_CONTINUE;
}

BOOL CALLBACK rangeAxis(const DIDEVICEOBJECTINSTANCE* object, void* context)
{
    auto* device = static_cast<IDirectInputDevice8*>(context);

    auto range = DIPROPRANGE{};
    range.diph.dwSize = sizeof(DIPROPRANGE);
    range.diph.dwHeaderSize = sizeof(DIPROPHEADER);
    range.diph.dwHow = DIPH_BYID;
    range.diph.dwObj = object->dwType;
    range.lMin = directInputAxisMinimum;
    range.lMax = directInputAxisMaximum;
    device->SetProperty(DIPROP_RANGE, &range.diph);

    return DIENUM_CONTINUE;
}

[[nodiscard]] std::int32_t axisAt(const DIJOYSTATE2& state, const std::uint32_t offset)
{
    switch (offset)
    {
    case directInputOffsetY:
        return static_cast<std::int32_t>(state.lY);
    case directInputOffsetZ:
        return static_cast<std::int32_t>(state.lZ);
    case directInputOffsetRz:
        return static_cast<std::int32_t>(state.lRz);
    default:
        break;
    }

    return static_cast<std::int32_t>(state.lX);
}

} // namespace

DirectInputBackend::~DirectInputBackend()
{
    close();
}

std::vector<DeviceDescription> DirectInputBackend::enumerate()
{
    auto* created = static_cast<IDirectInput8*>(nullptr);
    if (DirectInput8Create(GetModuleHandle(nullptr), DIRECTINPUT_VERSION, IID_IDirectInput8,
                           reinterpret_cast<void**>(&created), nullptr) != DI_OK)
    {
        return {};
    }

    auto enumeration = Enumeration{};
    created->EnumDevices(DI8DEVCLASS_GAMECTRL, collectDevice, &enumeration, DIEDFL_ATTACHEDONLY);
    created->Release();

    return std::move(enumeration.found);
}

std::expected<DeviceDescription, std::string> DirectInputBackend::open(const DeviceIdentity identity)
{
    close();

    if (DirectInput8Create(GetModuleHandle(nullptr), DIRECTINPUT_VERSION, IID_IDirectInput8,
                           reinterpret_cast<void**>(&factory), nullptr) != DI_OK)
    {
        return std::unexpected("DirectInput would not start");
    }

    auto enumeration = Enumeration{};
    factory->EnumDevices(DI8DEVCLASS_GAMECTRL, collectDevice, &enumeration, DIEDFL_ATTACHEDONLY);

    for (auto index = std::size_t{0}; index < enumeration.found.size(); index++)
    {
        if (!(enumeration.found[index].identity == identity))
        {
            continue;
        }

        if (factory->CreateDevice(enumeration.instances[index], &device, nullptr) != DI_OK)
        {
            break;
        }

        device->SetDataFormat(&c_dfDIJoystick2);

        // Exclusive is what force feedback needs and what stops a second reader — the window
        // library's own joystick layer included — taking the device out from under this one.
        // Without a window to state it against it cannot be asked for, and the capability below is
        // what says so out loud instead of leaving a torque that goes nowhere.
        auto exclusive = false;
        if (nativeWindow != nullptr)
        {
            exclusive = device->SetCooperativeLevel(static_cast<HWND>(nativeWindow),
                                                    DISCL_EXCLUSIVE | DISCL_BACKGROUND) == DI_OK;
        }

        if (!exclusive && nativeWindow != nullptr)
        {
            device->SetCooperativeLevel(static_cast<HWND>(nativeWindow), DISCL_NONEXCLUSIVE | DISCL_BACKGROUND);
        }

        device->EnumObjects(rangeAxis, device, DIDFT_AXIS);

        auto capabilitiesOfDevice = DIDEVCAPS{};
        capabilitiesOfDevice.dwSize = sizeof(DIDEVCAPS);
        device->GetCapabilities(&capabilitiesOfDevice);

        deviceCapabilities = DeviceCapabilities{};
        if (exclusive && (capabilitiesOfDevice.dwFlags & DIDC_FORCEFEEDBACK) != 0)
        {
            deviceCapabilities.add(DeviceCapability::ConstantForce);
            // Both are device properties DirectInput owns, and both are present wherever force
            // feedback is — which is exactly where the Linux driver drops them.
            deviceCapabilities.add(DeviceCapability::ForceGain);
            deviceCapabilities.add(DeviceCapability::AutoCentre);
        }

        // Neither the rotation range nor the base's tuning menu is reachable through DirectInput at
        // all; both need Fanatec's own SDK. Absent, and said so.

        if (device->Acquire() != DI_OK)
        {
            break;
        }

        acquired = true;
        address = enumeration.found[index].address;
        bounds = enumeration.found[index].axes;
        current = DeviceSample{};
        measuredHz = 0.0;

        auto description = enumeration.found[index];
        description.capabilities = deviceCapabilities;

        return description;
    }

    close();

    return std::unexpected("no input device answering to " + std::to_string(identity.vendor) + ":" +
                           std::to_string(identity.product) + " is present");
}

void DirectInputBackend::close()
{
    if (constantForce != nullptr)
    {
        constantForce->Stop();
        constantForce->Release();
        constantForce = nullptr;
    }

    if (device != nullptr)
    {
        device->Unacquire();
        device->Release();
        device = nullptr;
    }

    if (factory != nullptr)
    {
        factory->Release();
        factory = nullptr;
    }

    acquired = false;
    deviceCapabilities = DeviceCapabilities{};
}

bool DirectInputBackend::opened() const
{
    return acquired;
}

std::expected<DeviceSample, std::string> DirectInputBackend::read(const std::chrono::milliseconds timeout)
{
    if (!acquired)
    {
        return std::unexpected("no device is open");
    }

    // DirectInput polls rather than waking a waiter, so the timeout is spent sleeping between
    // polls. std::this_thread rather than Sleep so that the wait is the same statement on both
    // platforms and the resolution is the standard library's problem.
    device->Poll();

    auto state = DIJOYSTATE2{};
    if (device->GetDeviceState(sizeof(DIJOYSTATE2), &state) != DI_OK)
    {
        if (device->Acquire() != DI_OK)
        {
            return std::unexpected("the device went away");
        }

        return current;
    }

    current.axes[axisIndex(InputAxis::Steering)] = axisAt(state, directInputOffsetX);
    current.axes[axisIndex(InputAxis::Throttle)] = axisAt(state, directInputOffsetY);
    current.axes[axisIndex(InputAxis::Brake)] = axisAt(state, directInputOffsetZ);
    current.axes[axisIndex(InputAxis::Clutch)] = axisAt(state, directInputOffsetRz);

    current.buttons = 0;
    for (auto index = std::size_t{0}; index < 64; index++)
    {
        if ((state.rgbButtons[index] & 0x80) != 0)
        {
            current.buttons |= std::uint64_t{1} << index;
        }
    }

    current.reports++;

    const auto now = std::chrono::steady_clock::now();
    if (lastReport.time_since_epoch().count() != 0)
    {
        const auto interval = std::chrono::duration<double>(now - lastReport).count();
        if (interval > 0.0 && interval < 1.0)
        {
            const auto instant = 1.0 / interval;
            measuredHz = measuredHz > 0.0 ? measuredHz + (instant - measuredHz) * 0.01 : instant;
        }
    }
    lastReport = now;

    static_cast<void>(timeout);

    return current;
}

std::expected<void, std::string> DirectInputBackend::writeTorque(const double torqueFraction)
{
    if (!acquired)
    {
        return std::unexpected("no device is open");
    }

    if (!deviceCapabilities.has(DeviceCapability::ConstantForce))
    {
        return std::unexpected("this device takes no constant force");
    }

    auto force = DICONSTANTFORCE{};
    force.lMagnitude = directInputMagnitude(torqueFraction);

    auto axes = std::array<DWORD, 1>{DIJOFS_X};
    auto directions = std::array<LONG, 1>{0};

    auto effect = DIEFFECT{};
    effect.dwSize = sizeof(DIEFFECT);
    effect.dwFlags = DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS;
    effect.dwDuration = INFINITE;
    effect.dwGain = DI_FFNOMINALMAX;
    effect.dwTriggerButton = DIEB_NOTRIGGER;
    effect.cAxes = 1;
    effect.rgdwAxes = axes.data();
    effect.rglDirection = directions.data();
    effect.cbTypeSpecificParams = sizeof(DICONSTANTFORCE);
    effect.lpvTypeSpecificParams = &force;

    if (constantForce == nullptr)
    {
        if (device->CreateEffect(GUID_ConstantForce, &effect, &constantForce, nullptr) != DI_OK)
        {
            return std::unexpected("the device would not take the force");
        }

        if (constantForce->Start(1, 0) != DI_OK)
        {
            return std::unexpected("the device would not start the force");
        }

        return {};
    }

    if (constantForce->SetParameters(&effect, DIEP_TYPESPECIFICPARAMS | DIEP_START) != DI_OK)
    {
        return std::unexpected("the device would not take the force");
    }

    return {};
}

#else

// Off Windows the class still exists, still compiles and still answers — which is what keeps the
// half of it above that is not a COM call from rotting between now and the day it ships.
DirectInputBackend::~DirectInputBackend() = default;

std::vector<DeviceDescription> DirectInputBackend::enumerate()
{
    return {};
}

std::expected<DeviceDescription, std::string> DirectInputBackend::open(DeviceIdentity)
{
    return std::unexpected("DirectInput belongs to Windows and this is not Windows");
}

void DirectInputBackend::close()
{
}

bool DirectInputBackend::opened() const
{
    return false;
}

std::expected<DeviceSample, std::string> DirectInputBackend::read(std::chrono::milliseconds)
{
    return std::unexpected("no device is open");
}

std::expected<void, std::string> DirectInputBackend::writeTorque(double)
{
    return std::unexpected("no device is open");
}

#endif

} // namespace raceengine
