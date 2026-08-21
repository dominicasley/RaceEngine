module;

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

export module raceengine.input:InputBackend;

namespace raceengine
{

// A device, named by what it *is* rather than by where the platform happened to put it. A profile
// is keyed on this and never on an evdev node or a DirectInput instance GUID: both change across a
// replug, and neither exists on the other platform, so a calibration keyed on one would be a
// calibration the same physical wheel loses by being unplugged.
export struct DeviceIdentity
{
    std::uint16_t vendor = 0;
    std::uint16_t product = 0;

    [[nodiscard]] friend constexpr bool operator==(DeviceIdentity, DeviceIdentity) = default;
};

static_assert(std::is_trivially_copyable_v<DeviceIdentity>);

// The roles the game asks for, which is what a calibration is keyed on. Which physical axis carries
// which role is the *backend's* knowledge — ABS_X here, a DIJOYSTATE member on Windows — because it
// is a property of the platform's report descriptor and not of the driver's leg. Calibration keyed
// on the role therefore travels between the two; a raw axis code would not.
export enum class InputAxis : std::uint8_t { Steering, Throttle, Brake, Clutch };

export inline constexpr std::size_t inputAxisCount = 4;

export [[nodiscard]] constexpr std::size_t axisIndex(const InputAxis axis)
{
    return static_cast<std::size_t>(axis);
}

export [[nodiscard]] constexpr const char* inputAxisName(const InputAxis axis)
{
    switch (axis)
    {
    case InputAxis::Steering:
        return "steering";
    case InputAxis::Throttle:
        return "throttle";
    case InputAxis::Brake:
        return "brake";
    case InputAxis::Clutch:
        break;
    }

    return "clutch";
}

export [[nodiscard]] constexpr std::optional<InputAxis> inputAxisFromName(const std::string_view name)
{
    if (name == "steering")
    {
        return InputAxis::Steering;
    }
    if (name == "throttle")
    {
        return InputAxis::Throttle;
    }
    if (name == "brake")
    {
        return InputAxis::Brake;
    }
    if (name == "clutch")
    {
        return InputAxis::Clutch;
    }

    return std::nullopt;
}

// What the device says an axis can read, in its own units. `present` is separate from a zero span
// because an absent axis and an axis stuck at one value are different things and only one of them
// is worth calibrating.
export struct AxisBounds
{
    std::int32_t minimum = 0;
    std::int32_t maximum = 0;
    bool present = false;
};

// One report's worth of device state, as the device stated it. Raw on purpose: calibration is a
// pure function applied above this, so what crosses the device thread is what the hardware said and
// the maths that shapes it can be tested with no hardware at all.
export struct DeviceSample
{
    std::array<std::int32_t, inputAxisCount> axes{};
    std::uint64_t buttons = 0;
    // Reports the device has produced since it was opened. Two ticks that read the same number saw
    // no report between them, which is what separates a still wheel from a stalled one.
    std::uint64_t reports = 0;
    // When the device said it, on the same clock `std::chrono::steady_clock` reads — CLOCK_MONOTONIC
    // on Linux, which is what `EVIOCSCLOCKID` asks the kernel for at open. Zero when the platform
    // will not say.
    //
    // It is the device's own stamp and not the moment this thread woke up, which is the entire
    // reason it is worth carrying: end-to-end force feedback latency is measured from the instant
    // the wheel reported its position to the instant a torque derived from it was written back, and
    // timing the wake-up instead measures the reader's poll interval.
    std::uint64_t timestampNanos = 0;
};

static_assert(std::is_trivially_copyable_v<DeviceSample>);

// Things one platform has and the other does not, stated rather than assumed.
//
// The list is not hypothetical. On the ClubSport V2.5 under `hid-fanatec` the base's tuning menu is
// one bit in the driver's device table and product 0x0004 does not have it, so no tuning node is
// created at all; gain and autocentre are advertised by neither the driver nor this device's
// effect list. None of the three has a Windows equivalent without Fanatec's own SDK. A caller that
// wanted one of them has to be told, and told *once*, rather than either refusing to start or
// quietly doing nothing.
export enum class DeviceCapability : std::uint8_t {
    // A signed torque can be written at all. Everything else here is a refinement of it.
    ConstantForce,
    // Device-side scaling of every effect. Linux drops `FF_GAIN` on this base with no error.
    ForceGain,
    // A device-side centring spring, which is a different thing from a torque the game computes.
    AutoCentre,
    // The lock-to-lock rotation range can be read back.
    ReadRotationRange,
    // ...and set. Reading without setting is the common case: the range is a property the driver or
    // the base's own menu owns, and a game that assumed it could set it would silently use a range
    // the wheel is not at.
    SetRotationRange,
    // The base's own tuning profile can be driven from the host.
    TuningMenu,
    Count
};

export [[nodiscard]] constexpr const char* deviceCapabilityName(const DeviceCapability capability)
{
    switch (capability)
    {
    case DeviceCapability::ConstantForce:
        return "constant force";
    case DeviceCapability::ForceGain:
        return "force gain";
    case DeviceCapability::AutoCentre:
        return "auto centre";
    case DeviceCapability::ReadRotationRange:
        return "rotation range readback";
    case DeviceCapability::SetRotationRange:
        return "rotation range control";
    case DeviceCapability::TuningMenu:
        return "tuning menu control";
    case DeviceCapability::Count:
        break;
    }

    return "an unnamed capability";
}

export class DeviceCapabilities
{
    std::uint32_t bits = 0;

public:
    constexpr void add(const DeviceCapability capability)
    {
        bits |= 1u << static_cast<std::uint32_t>(capability);
    }

    [[nodiscard]] constexpr bool has(const DeviceCapability capability) const
    {
        return (bits & (1u << static_cast<std::uint32_t>(capability))) != 0;
    }

    [[nodiscard]] constexpr bool any() const
    {
        return bits != 0;
    }

    [[nodiscard]] friend constexpr bool operator==(DeviceCapabilities, DeviceCapabilities) = default;
};

// What a caller wanted, what it wanted it for, and what it will do instead. Three strings rather
// than one because the sentence that comes out has to be actionable by somebody who did not write
// the calling code: "not supported" sends a reader to look for a permissions problem they do not
// have.
export struct CapabilityRequest
{
    DeviceCapability capability = DeviceCapability::ConstantForce;
    std::string_view purpose;
    std::string_view fallback;
};

// Nothing when the capability is there; otherwise the sentence to warn with. A shortfall is a
// warning by construction — this function cannot refuse, cannot throw and returns no error type —
// because degrading is the only correct answer: the alternative is a game that will not start on
// hardware that is merely different, or one that silently does nothing and reads as a bug in
// whatever was meant to have been tuned.
//
// The wording avoids the words the smoke gate greps for, which is not a formality: a capability
// this device was never going to have is not a fault of the run.
export [[nodiscard]] inline std::optional<std::string> capabilityShortfall(const DeviceCapabilities& capabilities,
                                                                           const CapabilityRequest& request)
{
    if (capabilities.has(request.capability))
    {
        return std::nullopt;
    }

    auto notice = std::string("This device has no ") + deviceCapabilityName(request.capability);

    if (!request.purpose.empty())
    {
        notice += ", so ";
        notice += request.purpose;
        notice += " is unavailable";
    }

    if (!request.fallback.empty())
    {
        notice += "; ";
        notice += request.fallback;
        notice += " instead";
    }

    return notice;
}

// How often the device answers, asked of the backend rather than written down anywhere.
//
// It is a query and not a constant because the two platforms differ and the difference is invisible
// in the picture: anything tuned against a number that was true on Linux — a filter's time
// constant, a force ramp per report, a rate limit — is tuned wrong on Windows, and the symptom is
// "the wheel feels different" with nothing to point at.
//
// `measured` says which of the two answers this is. A backend states the ceiling its transport
// allows the moment it opens the device, and replaces it with the rate the device is actually
// delivering once it has delivered anything: a wheel nobody is touching produces no reports at all,
// so an unmeasured rate is the honest answer rather than a placeholder.
export struct UpdateRate
{
    double inputHz = 0.0;
    double outputHz = 0.0;
    bool measured = false;
};

export struct DeviceDescription
{
    DeviceIdentity identity;
    std::string name;
    // What the platform calls this device today: an evdev node, a DirectInput instance. Diagnostics
    // only — it is deliberately not what anything is keyed on.
    std::string address;
    DeviceCapabilities capabilities;
    std::array<AxisBounds, inputAxisCount> axes{};
    // Lock to lock, in degrees, as the device is presently configured. Zero when the device will
    // not say, which is a real answer and not a missing one: a game must then use the range the
    // player states rather than a number it invented.
    double rotationDegrees = 0.0;
};

// The platform seam. Two implementations, written together rather than one now and one later,
// because the whole cost of this interface is paid on the day a second backend does not fit it.
export class IInputBackend
{
public:
    IInputBackend() = default;
    IInputBackend(const IInputBackend&) = delete;
    IInputBackend(IInputBackend&&) = delete;
    IInputBackend& operator=(const IInputBackend&) = delete;
    IInputBackend& operator=(IInputBackend&&) = delete;
    virtual ~IInputBackend() = default;

    // Which platform answered. Diagnostics and tests; nothing branches on it.
    [[nodiscard]] virtual std::string_view platform() const = 0;

    // Every device this backend can see. Empty is an ordinary answer — no wheel is switched on —
    // and not a condition to report, which is why it is a vector and not an `expected`.
    [[nodiscard]] virtual std::vector<DeviceDescription> enumerate() = 0;

    // By identity, never by address. Two of the same wheel on one desk is a case this deliberately
    // does not resolve: it takes the first, because the alternative is keying on a serial number
    // that Linux exposes and DirectInput does not.
    [[nodiscard]] virtual std::expected<DeviceDescription, std::string> open(DeviceIdentity identity) = 0;
    virtual void close() = 0;
    [[nodiscard]] virtual bool opened() const = 0;

    // Waits up to `timeout` for the device to say something, then answers with the state as it
    // stands whether or not it did. The timeout is what lets the device thread notice a stop
    // request promptly without spinning.
    [[nodiscard]] virtual std::expected<DeviceSample, std::string> read(std::chrono::milliseconds timeout) = 0;

    // Signed fraction of the device's own maximum torque, [-1, 1]. Newton metres belong one layer
    // up, with the profile that states what this device's maximum *is*: no backend can know it —
    // evdev has no such field and DirectInput's magnitude is a per-effect abstraction — so a
    // physical unit here would be a guess wearing a unit.
    //
    // Present before there is anything to feed it on purpose. Force feedback is a later piece of
    // work, and an interface grown to fit it after a season of tuning against one backend is
    // exactly the retrofit two implementations exist to prevent.
    //
    // **The sign is a contract, not a suggestion: positive turns the rim in the direction of
    // positive steering demand.** It is the same sense stage one reports its newton metres in, so
    // the whole pipeline carries one convention and each backend owns the translation to whatever
    // its platform means by positive — which is a platform fact, and the kind that is only settled
    // from the seat. Get it wrong and every restoring torque becomes the pull it was resisting.
    [[nodiscard]] virtual std::expected<void, std::string> writeTorque(double torqueFraction) = 0;

    [[nodiscard]] virtual DeviceCapabilities capabilities() const = 0;
    [[nodiscard]] virtual UpdateRate updateRate() const = 0;

    // Sets the device's own lock-to-lock, degrees, and answers with what the device confirmed.
    // The game asks for the *car's* travel, which is what a simulator does with a wheel: the rig's
    // physical stops land exactly on the car's lock, the axis calibration and the driver's hands
    // agree by construction, and every mapping between them collapses to one-to-one. Only a backend
    // whose capability list carries SetRotationRange can honour it; the rest refuse with a reason.
    [[nodiscard]] virtual std::expected<double, std::string> setRotationRange(double degrees) = 0;
};

} // namespace raceengine
