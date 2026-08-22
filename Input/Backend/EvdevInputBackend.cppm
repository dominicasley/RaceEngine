module;

#if defined(__linux__)
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <expected>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// An implementation partition, on the rule the Vulkan backend established: nothing outside
// raceengine.input can import this, so <linux/input.h> and the macros it brings stop here and the
// factory is the only way anybody obtains one.
module raceengine.input:EvdevInputBackend;

import :InputBackend;

namespace raceengine
{

// Linux, through evdev directly and not through GLFW: GLFW's joystick API is read-only — it has no
// force feedback at all — and its input functions may only be called from the thread that pumps
// the window, which is the thread the frame is recorded on. A wheel has to be read faster than a
// frame and written to at all, so it bypasses GLFW entirely and GLFW keeps the window, the
// keyboard and the mouse.
class EvdevInputBackend final : public IInputBackend
{
    static constexpr int noDevice = -1;

    int device = noDevice;
    DeviceCapabilities deviceCapabilities;
    double transportHz = 0.0;
    double measuredHz = 0.0;

    // State only the evdev half keeps. Guarded rather than merely unused off Linux, because a
    // private field nothing reads is a warning and a warning stops this build.
#if defined(__linux__)
    // What was actually obtained. A node the session owns read but not write is a wheel that can be
    // driven and cannot push back, which is a degradation and not a refusal.
    bool writable = false;
    std::string address;
    std::string sysfs;
    std::array<AxisBounds, inputAxisCount> bounds{};
    // Button codes in ascending order, so a profile's bit index is a stable name for a button on
    // this device rather than a raw evdev code the other platform has never heard of.
    std::vector<std::uint16_t> buttonCodes;
    DeviceSample current{};
    std::int16_t effect = -1;
    double lastReportSeconds = 0.0;
    // The pedals' own `rumble` attribute, if a set with motors is plugged in. **A different device
    // from the wheel base**, with its own USB identity, so it is searched for rather than derived
    // from the wheel's node the way `range` and the tuning menu are.
    std::filesystem::path pedalRumble;
    int pedalMotors = noDevice;
#endif

public:
    EvdevInputBackend() = default;
    ~EvdevInputBackend() override;

    [[nodiscard]] std::string_view platform() const override
    {
        return "evdev";
    }

    [[nodiscard]] std::vector<DeviceDescription> enumerate() override;
    [[nodiscard]] std::expected<DeviceDescription, std::string> open(DeviceIdentity identity) override;
    void close() override;

    [[nodiscard]] bool opened() const override
    {
        return device != noDevice;
    }

    [[nodiscard]] std::expected<DeviceSample, std::string> read(std::chrono::milliseconds timeout) override;
    [[nodiscard]] std::expected<void, std::string> writeTorque(double torqueFraction) override;
    [[nodiscard]] std::expected<void, std::string> writePedalMotors(std::uint8_t throttle, std::uint8_t brake) override;

    [[nodiscard]] DeviceCapabilities capabilities() const override
    {
        return deviceCapabilities;
    }

    [[nodiscard]] std::expected<double, std::string> setRotationRange(double degrees) override;

    [[nodiscard]] UpdateRate updateRate() const override
    {
        const auto ceiling = transportHz > 0.0 ? transportHz : 0.0;

        return UpdateRate{
            .inputHz = measuredHz > 0.0 ? measuredHz : ceiling, .outputHz = ceiling, .measured = measuredHz > 0.0};
    }
};

} // namespace raceengine

namespace raceengine
{

#if defined(__linux__)

namespace
{

// Every ioctl goes through here so that the request's type is converted once, explicitly, rather
// than at two dozen call sites where -Wsign-conversion would have to be reasoned about each time.
template <class T> [[nodiscard]] bool control(const int fileDescriptor, const unsigned long request, T* argument)
{
    return ::ioctl(fileDescriptor, request, argument) >= 0;
}

[[nodiscard]] bool bitSet(const std::uint8_t* bits, const std::size_t index)
{
    return ((bits[index / 8] >> (index % 8)) & 1u) != 0u;
}

// evdev names an axis by its own code; the game names it by what the driver does with it. On this
// base steering is ABS_X and the three pedals are the base's own 16-bit Y, Z and RZ — they are not
// a second USB device and there is no ABS_GAS or ABS_BRAKE anywhere on it.
//
// Which of the three is which is platform knowledge and it lives here, not in a profile: the
// profile carries calibration keyed on the *role*, so it travels to a platform that names the same
// physical axis something else. Getting the order wrong is one line here rather than a file
// everybody's wheel has to have rewritten.
//
// **Z is the throttle, RZ is the brake and Y is the clutch** on this rig, measured by pressing them.
// It is not the order the names suggest and it was wrong here twice before it was measured: the
// symptom is worth knowing, because it is not "the pedals are swapped". Every calibration prompt
// watches one axis, so asking for the throttle while the throttle pedal drives a different code reads
// as *no travel at all* — which was diagnosed first as a disconnected pedal and then as the USB
// drop-outs, and was neither.
//
// **This order should not be hardcoded here at all**, and the fact that it has been wrong twice is
// the argument: the calibration tool watches every axis while a pedal is pressed, so it already knows
// which one moved and could simply say so. Until the profile can carry that, this is a per-rig
// constant sitting in a place a rig cannot reach.
[[nodiscard]] bool axisRole(const std::uint16_t code, InputAxis& role)
{
    switch (code)
    {
    case ABS_X:
        role = InputAxis::Steering;
        return true;
    case ABS_Y:
        role = InputAxis::Clutch;
        return true;
    case ABS_Z:
        role = InputAxis::Throttle;
        return true;
    case ABS_RZ:
        role = InputAxis::Brake;
        return true;
    default:
        break;
    }

    return false;
}

[[nodiscard]] std::string readAttribute(const std::filesystem::path& path)
{
    auto file = std::ifstream(path);
    if (!file)
    {
        return {};
    }

    auto value = std::string();
    std::getline(file, value);

    return value;
}

[[nodiscard]] double readNumericAttribute(const std::filesystem::path& path)
{
    const auto text = readAttribute(path);
    if (text.empty())
    {
        return 0.0;
    }

    try
    {
        return std::stod(text);
    }
    catch (const std::exception&)
    {
        return 0.0;
    }
}

// Where a pedal set with vibration motors hangs its control, if one is attached.
//
// **Searched for across every HID device rather than derived from the wheel's**, because the pedals
// are not necessarily the wheel: a ClubSport V3 set on USB has its own identity (0EB7:183b) and
// `hid-fanatec` creates the attribute on *that* device. Plugged into a ClubSport V2.5 base's RJ12
// instead — which is the usual way, and how a rig arrives out of the box — the pedals are axes on
// the base, and the patched driver hangs a `rumble` on the *base* (0EB7:0004) that forwards to them
// with the pedal sub-report forced. An unpatched driver creates no attribute there, which is not a
// failure to report so much as a rig that has not been wired for this.
[[nodiscard]] std::filesystem::path findPedalRumble()
{
    // **The identity is checked and not merely the attribute's presence.** `hid-fanatec` creates a
    // `rumble` file for the CSL Elite *wheel base* as well, where it drives something that is not a
    // pedal — the driver sends a different sub-report for the two — so a search that took the first
    // node it found would, on that base, quietly write pedal cues into the wheel. The V2.5 base is
    // the one base whose rumble *is* the pedals (it has no motor of its own, and the patched driver
    // forces the pedal sub-report for it), and the pedals' own identity still wins over it so a set
    // rewired to USB keeps its direct path.
    constexpr auto clubsportPedalsV3 = std::string_view("0EB7:183B");
    constexpr auto clubsportV25Base = std::string_view("0EB7:0004");

    auto code = std::error_code();
    const auto devices = std::filesystem::path("/sys/bus/hid/devices");

    auto viaBase = std::filesystem::path();
    for (const auto& entry : std::filesystem::directory_iterator(devices, code))
    {
        const auto name = entry.path().filename().string();
        const auto ownIdentity = name.find(clubsportPedalsV3) != std::string::npos;
        if (!ownIdentity && name.find(clubsportV25Base) == std::string::npos)
        {
            continue;
        }

        auto candidate = entry.path() / "rumble";
        if (!std::filesystem::exists(candidate, code))
        {
            continue;
        }
        if (ownIdentity)
        {
            return candidate;
        }
        viaBase = candidate;
    }

    return viaBase;
}

// /sys/class/input/eventN/device/device is the HID device the node belongs to, which is where the
// Fanatec driver hangs `range`, `wheel_id` and — on the bases that have it — the tuning menu's
// attributes. Derived from the node rather than searched for, so it follows the device across a
// replug.
[[nodiscard]] std::filesystem::path sysfsForNode(const std::string& node)
{
    const auto leaf = std::filesystem::path(node).filename().string();
    auto code = std::error_code();
    auto path =
        std::filesystem::canonical(std::filesystem::path("/sys/class/input") / leaf / "device" / "device", code);

    return code ? std::filesystem::path() : path;
}

// The transport's own ceiling, from the interrupt endpoint's polling interval. A real runtime read
// and not a constant: on this base both endpoints state bInterval 1 at full speed, which is a
// millisecond frame and a thousand reports a second either way.
//
// It is a *ceiling* and deliberately named one. What the driver actually delivers can be lower —
// this one schedules its output on a two millisecond timer, so half of what the wire allows — and
// nothing in sysfs says so, which is why the rate a caller reads is replaced by a measurement the
// moment the device produces anything to measure.
[[nodiscard]] double transportCeiling(const std::filesystem::path& hidDevice)
{
    if (hidDevice.empty())
    {
        return 0.0;
    }

    auto best = 0.0;
    auto code = std::error_code();
    for (const auto& entry : std::filesystem::directory_iterator(hidDevice.parent_path(), code))
    {
        if (entry.path().filename().string().rfind("ep_", 0) != 0)
        {
            continue;
        }

        const auto interval = readNumericAttribute(entry.path() / "bInterval");
        if (interval > 0.0)
        {
            best = std::max(best, 1000.0 / interval);
        }
    }

    return best;
}

[[nodiscard]] DeviceCapabilities probeCapabilities(const int fileDescriptor, const std::filesystem::path& hidDevice,
                                                   const bool writable)
{
    auto found = DeviceCapabilities{};

    auto forceBits = std::array<std::uint8_t, (FF_MAX + 8) / 8>{};
    if (control(fileDescriptor, EVIOCGBIT(EV_FF, forceBits.size()), forceBits.data()))
    {
        // Writable as well as advertised: a node opened read-only will take no effect upload, and a
        // torque the game computes and cannot deliver is worse than one it knows it cannot.
        if (writable && bitSet(forceBits.data(), FF_CONSTANT))
        {
            found.add(DeviceCapability::ConstantForce);
        }

        // Neither of these is set on this base, and that is the driver rather than the device: it
        // installs no gain and no autocentre handler, so the kernel discards both silently. A
        // capability list is the only place that difference can be seen from above.
        if (writable && bitSet(forceBits.data(), FF_GAIN))
        {
            found.add(DeviceCapability::ForceGain);
        }

        if (writable && bitSet(forceBits.data(), FF_AUTOCENTER))
        {
            found.add(DeviceCapability::AutoCentre);
        }
    }

    if (hidDevice.empty())
    {
        return found;
    }

    auto code = std::error_code();

    if (std::filesystem::exists(hidDevice / "range", code))
    {
        found.add(DeviceCapability::ReadRotationRange);

        if (::access((hidDevice / "range").c_str(), W_OK) == 0)
        {
            found.add(DeviceCapability::SetRotationRange);
        }
    }

    // The tuning menu is one bit in the driver's device table. Where it is set the driver hangs the
    // menu's slots on the HID device; where it is not — product 0x0004 is one — there is nothing to
    // find, and no amount of permission changes it.
    if (std::filesystem::exists(hidDevice / "SLOT", code))
    {
        found.add(DeviceCapability::TuningMenu);
    }

    return found;
}

struct NodeDetail
{
    DeviceDescription description;
    std::vector<std::uint16_t> buttons;
    bool usable = false;
};

[[nodiscard]] NodeDetail describeNode(const int fileDescriptor, const std::string& node, const bool writable)
{
    auto detail = NodeDetail{};

    auto identifier = input_id{};
    if (!control(fileDescriptor, EVIOCGID, &identifier))
    {
        return detail;
    }

    auto absoluteBits = std::array<std::uint8_t, (ABS_MAX + 8) / 8>{};
    if (!control(fileDescriptor, EVIOCGBIT(EV_ABS, absoluteBits.size()), absoluteBits.data()))
    {
        return detail;
    }

    // A steering axis is what makes a node worth looking at. Keyboards and mice carry no ABS_X, and
    // a device that carries one is a joystick, a pad or a wheel — all three of which this backend
    // is willing to describe.
    if (!bitSet(absoluteBits.data(), ABS_X))
    {
        return detail;
    }

    auto name = std::array<char, 256>{};
    if (control(fileDescriptor, EVIOCGNAME(name.size()), name.data()))
    {
        detail.description.name = std::string(name.data(), ::strnlen(name.data(), name.size()));
    }

    detail.description.identity = DeviceIdentity{.vendor = identifier.vendor, .product = identifier.product};
    detail.description.address = node;

    for (auto code = std::uint16_t{0}; code <= ABS_MAX; code++)
    {
        auto role = InputAxis::Steering;
        if (!bitSet(absoluteBits.data(), code) || !axisRole(code, role))
        {
            continue;
        }

        auto information = input_absinfo{};
        if (!control(fileDescriptor, EVIOCGABS(code), &information))
        {
            continue;
        }

        detail.description.axes[axisIndex(role)] =
            AxisBounds{.minimum = information.minimum, .maximum = information.maximum, .present = true};
    }

    auto keyBits = std::array<std::uint8_t, (KEY_MAX + 8) / 8>{};
    if (control(fileDescriptor, EVIOCGBIT(EV_KEY, keyBits.size()), keyBits.data()))
    {
        for (auto code = std::size_t{BTN_MISC}; code <= KEY_MAX && detail.buttons.size() < 64; code++)
        {
            if (bitSet(keyBits.data(), code))
            {
                detail.buttons.push_back(static_cast<std::uint16_t>(code));
            }
        }
    }

    const auto hidDevice = sysfsForNode(node);
    detail.description.capabilities = probeCapabilities(fileDescriptor, hidDevice, writable);
    detail.description.rotationDegrees = readNumericAttribute(hidDevice / "range");
    detail.usable = true;

    return detail;
}

[[nodiscard]] std::vector<std::string> inputNodes()
{
    auto nodes = std::vector<std::string>();
    auto code = std::error_code();

    for (const auto& entry : std::filesystem::directory_iterator("/dev/input", code))
    {
        const auto leaf = entry.path().filename().string();
        if (leaf.rfind("event", 0) == 0)
        {
            nodes.push_back(entry.path().string());
        }
    }

    // Ascending, so which of two identical devices is taken first is a property of the machine and
    // not of the order the directory happened to be walked in.
    std::sort(nodes.begin(), nodes.end());

    return nodes;
}

} // namespace

EvdevInputBackend::~EvdevInputBackend()
{
    close();
}

std::vector<DeviceDescription> EvdevInputBackend::enumerate()
{
    auto found = std::vector<DeviceDescription>();

    for (const auto& node : inputNodes())
    {
        // Read-only to look: enumeration must not take a device away from anything, and the write
        // side is only wanted by whoever goes on to open it. Whether it *could* be had is asked
        // rather than assumed — a device this session may only read is a wheel that can be driven
        // and cannot push back, and enumerating it as having no force feedback at all would be
        // reporting the permission as a property of the hardware.
        const auto fileDescriptor = ::open(node.c_str(), O_RDONLY | O_NONBLOCK);
        if (fileDescriptor < 0)
        {
            continue;
        }

        if (auto detail = describeNode(fileDescriptor, node, ::access(node.c_str(), W_OK) == 0); detail.usable)
        {
            found.push_back(std::move(detail.description));
        }

        ::close(fileDescriptor);
    }

    return found;
}

std::expected<DeviceDescription, std::string> EvdevInputBackend::open(const DeviceIdentity identity)
{
    close();

    for (const auto& node : inputNodes())
    {
        auto fileDescriptor = ::open(node.c_str(), O_RDWR | O_NONBLOCK);
        auto writeable = fileDescriptor >= 0;

        if (!writeable)
        {
            fileDescriptor = ::open(node.c_str(), O_RDONLY | O_NONBLOCK);
        }

        if (fileDescriptor < 0)
        {
            continue;
        }

        auto detail = describeNode(fileDescriptor, node, writeable);
        if (!detail.usable || !(detail.description.identity == identity))
        {
            ::close(fileDescriptor);
            continue;
        }

        // Monotonic timestamps on the events themselves. The alternative is timing their arrival
        // here, which measures how often this thread woke up rather than how often the device
        // spoke, and the two differ by exactly the poll interval a rate estimate is trying to see
        // past.
        auto clockId = static_cast<int>(CLOCK_MONOTONIC);
        static_cast<void>(control(fileDescriptor, EVIOCSCLOCKID, &clockId));

        device = fileDescriptor;
        writable = writeable;
        address = node;
        deviceCapabilities = detail.description.capabilities;
        bounds = detail.description.axes;
        buttonCodes = std::move(detail.buttons);
        current = DeviceSample{};
        effect = -1;
        measuredHz = 0.0;
        lastReportSeconds = 0.0;
        sysfs = sysfsForNode(node).string();
        transportHz = transportCeiling(sysfs);

        // The pedals, if a set with motors in it is attached. **Opened here and not lazily**,
        // because whether the capability is there has to be settled before anything asks: a caller
        // that discovers it mid-session would start a cue in the middle of a corner.
        pedalRumble = findPedalRumble();
        pedalMotors = noDevice;

        if (!pedalRumble.empty())
        {
            pedalMotors = ::open(pedalRumble.c_str(), O_WRONLY | O_CLOEXEC);

            if (pedalMotors != noDevice)
            {
                deviceCapabilities.add(DeviceCapability::PedalMotors);
            }
        }

        // Whatever the axes are resting at is what the first sample says, so a profile seeded from
        // it is seeded from the device at rest rather than from zeroes.
        for (auto index = std::size_t{0}; index < inputAxisCount; index++)
        {
            if (!bounds[index].present)
            {
                continue;
            }

            auto information = input_absinfo{};
            const auto code = static_cast<std::uint16_t>(index == axisIndex(InputAxis::Steering)   ? ABS_X
                                                         : index == axisIndex(InputAxis::Throttle) ? ABS_Y
                                                         : index == axisIndex(InputAxis::Brake)    ? ABS_Z
                                                                                                   : ABS_RZ);
            if (control(device, EVIOCGABS(code), &information))
            {
                current.axes[index] = information.value;
            }
        }

        return detail.description;
    }

    return std::unexpected("no input device answering to " + std::to_string(identity.vendor) + ":" +
                           std::to_string(identity.product) + " is present");
}

void EvdevInputBackend::close()
{
    if (device == noDevice)
    {
        return;
    }

    if (effect >= 0)
    {
        auto identifier = static_cast<int>(effect);
        static_cast<void>(control(device, EVIOCRMFF, &identifier));
        effect = -1;
    }

    // **Silence before letting go, and this is not tidiness.** A torque settles when the writes stop
    // because the effect is removed with the device; an eccentric mass does not — it keeps turning
    // until something tells it nought. A pedal left buzzing after the game exits is a rig the driver
    // has to unplug.
    if (pedalMotors != noDevice)
    {
        static_cast<void>(::lseek(pedalMotors, 0, SEEK_SET));
        static_cast<void>(::write(pedalMotors, "0", 1));
        ::close(pedalMotors);
        pedalMotors = noDevice;
    }

    pedalRumble.clear();

    ::close(device);
    device = noDevice;
    writable = false;
    deviceCapabilities = DeviceCapabilities{};
    buttonCodes.clear();
}

std::expected<DeviceSample, std::string> EvdevInputBackend::read(const std::chrono::milliseconds timeout)
{
    if (device == noDevice)
    {
        return std::unexpected("no device is open");
    }

    auto waiting = pollfd{.fd = device, .events = POLLIN, .revents = 0};
    const auto ready = ::poll(&waiting, 1, static_cast<int>(timeout.count()));

    if (ready < 0)
    {
        return std::unexpected("the device stopped answering");
    }

    if (ready == 0)
    {
        return current;
    }

    if ((waiting.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
    {
        return std::unexpected("the device went away");
    }

    auto events = std::array<input_event, 64>{};

    while (true)
    {
        const auto taken = ::read(device, events.data(), sizeof(events));
        if (taken <= 0)
        {
            break;
        }

        const auto count = static_cast<std::size_t>(taken) / sizeof(input_event);

        for (auto index = std::size_t{0}; index < count; index++)
        {
            const auto& event = events[index];

            if (event.type == EV_ABS)
            {
                auto role = InputAxis::Steering;
                if (axisRole(event.code, role))
                {
                    current.axes[axisIndex(role)] = event.value;
                }

                continue;
            }

            if (event.type == EV_KEY)
            {
                const auto found = std::lower_bound(buttonCodes.begin(), buttonCodes.end(), event.code);
                if (found != buttonCodes.end() && *found == event.code)
                {
                    const auto bit = std::uint64_t{1} << static_cast<std::uint64_t>(found - buttonCodes.begin());
                    current.buttons = event.value != 0 ? current.buttons | bit : current.buttons & ~bit;
                }

                continue;
            }

            if (event.type != EV_SYN || event.code != SYN_REPORT)
            {
                continue;
            }

            current.reports++;
            current.timestampNanos = static_cast<std::uint64_t>(event.input_event_sec) * 1000000000ull +
                                     static_cast<std::uint64_t>(event.input_event_usec) * 1000ull;

            const auto seconds =
                static_cast<double>(event.input_event_sec) + static_cast<double>(event.input_event_usec) * 1e-6;

            if (lastReportSeconds > 0.0)
            {
                // One pole over about a second of reports. A single interval is whatever the
                // scheduler did; what is wanted is the rate the device is sustaining, and that is
                // the only figure a filter tuned against it can be tuned against.
                const auto interval = seconds - lastReportSeconds;
                if (interval > 0.0 && interval < 1.0)
                {
                    const auto instant = 1.0 / interval;
                    measuredHz = measuredHz > 0.0 ? measuredHz + (instant - measuredHz) * 0.01 : instant;
                }
            }

            lastReportSeconds = seconds;
        }

        if (count < events.size())
        {
            break;
        }
    }

    return current;
}

std::expected<double, std::string> EvdevInputBackend::setRotationRange(const double degrees)
{
    if (device == noDevice)
    {
        return std::unexpected("no device is open");
    }

    if (!deviceCapabilities.has(DeviceCapability::SetRotationRange))
    {
        return std::unexpected("this device's rotation range is not writable from here");
    }

    // The driver clamps to what the base accepts (90-900 on this family); asking inside that range
    // and reading back what stuck is what keeps the number honest rather than assumed. Writing it
    // at all is also what re-synchronises a base whose firmware has drifted from the file: this rig
    // arrived soft-locked at roughly 240 degrees while its range file read 900, and no calibration
    // on top of a lie could put the picture and the hands in the same place.
    const auto asked = static_cast<int>(std::lround(std::clamp(degrees, 90.0, 900.0)));

    {
        auto file = std::ofstream(std::filesystem::path(sysfs) / "range");
        if (!file)
        {
            return std::unexpected("the range file would not open for writing");
        }

        file << asked;
        if (!file.flush())
        {
            return std::unexpected("the range file did not take the write");
        }
    }

    const auto confirmed = readNumericAttribute(std::filesystem::path(sysfs) / "range");

    return confirmed > 0.0 ? confirmed : static_cast<double>(asked);
}

std::expected<void, std::string> EvdevInputBackend::writePedalMotors(const std::uint8_t throttle,
                                                                     const std::uint8_t brake)
{
    if (!deviceCapabilities.has(DeviceCapability::PedalMotors) || pedalMotors == noDevice)
    {
        return std::unexpected("no pedal motors are attached");
    }

    // Throttle in the high byte, brake in the middle, as the driver's own documentation states it.
    // Written in decimal because `ftec_rumble_store` parses with base 0, which accepts either, and
    // decimal cannot be misread.
    const auto word = (static_cast<std::uint32_t>(throttle) << 16) | (static_cast<std::uint32_t>(brake) << 8);
    const auto text = std::to_string(word);

    // A sysfs attribute is rewritten from the start every time rather than appended to.
    if (::lseek(pedalMotors, 0, SEEK_SET) < 0)
    {
        return std::unexpected("the pedals' control could not be rewound");
    }

    if (::write(pedalMotors, text.c_str(), text.size()) < 0)
    {
        // The pedals are a separate device and can be unplugged while the wheel keeps working, so
        // this is a normal thing to happen rather than an error to carry on through. Dropping the
        // capability is what stops the caller asking again every frame for the rest of the session.
        ::close(pedalMotors);
        pedalMotors = noDevice;
        deviceCapabilities.remove(DeviceCapability::PedalMotors);

        return std::unexpected("the pedals stopped taking vibration and have been let go");
    }

    return {};
}

std::expected<void, std::string> EvdevInputBackend::writeTorque(const double torqueFraction)
{
    if (device == noDevice)
    {
        return std::unexpected("no device is open");
    }

    if (!deviceCapabilities.has(DeviceCapability::ConstantForce))
    {
        return std::unexpected("this device takes no constant force");
    }

    const auto clamped = std::clamp(torqueFraction, -1.0, 1.0);

    auto shape = ff_effect{};
    shape.type = FF_CONSTANT;
    shape.id = effect;
    // 0x4000 is ninety degrees, which for a one-axis device is the axis itself. A direction of zero
    // means "away from the user" and is what a rumble pad wants; a wheel given it turns the sign of
    // every force into a property of the driver's interpretation rather than of the number.
    shape.direction = 0x4000;
    // **Negated, because evdev's positive at 0x4000 is the opposite of the seam's contract.** The
    // contract is that a positive fraction turns the rim toward positive steering demand — a right
    // turn on this rig — and the kernel's direction rose names 0x4000 "left": hid-fanatecff scales
    // the level by sin(direction) and passes the sign through, so a positive level pushes the rim
    // left. Written without the minus it inverted every torque this engine ever sent, and the two
    // places it showed were chosen by a second inversion upstream cancelling it at speed: the
    // aligning moment's sign fault in the tyre model made stage one slightly pro-steer while
    // driving — flipped here into a weak, plausible centring — and left the standstill, where stage
    // one was already correct, as the one regime a driver could feel the truth: ease the wheel off
    // centre with the car parked and it pulled itself further, the restoring torque delivered
    // backwards. Measured from the seat 2026-08-20.
    shape.u.constant.level = static_cast<std::int16_t>(std::lround(-clamped * 32767.0));
    // Zero is until it is replaced or removed. A length would make a torque that expires between
    // two writes, which is a stutter that looks like a physics fault.
    shape.replay.length = 0;
    shape.replay.delay = 0;

    if (!control(device, EVIOCSFF, &shape))
    {
        return std::unexpected("the device would not take the force");
    }

    const auto uploaded = effect < 0;
    effect = shape.id;

    if (!uploaded)
    {
        return {};
    }

    // Uploading an effect arms it; one play starts it, and every later upload against the same id
    // changes the force under a wheel that is already holding one.
    auto play = input_event{};
    play.type = EV_FF;
    play.code = static_cast<std::uint16_t>(effect);
    play.value = 1;

    if (::write(device, &play, sizeof(play)) != static_cast<::ssize_t>(sizeof(play)))
    {
        return std::unexpected("the device would not start the force");
    }

    return {};
}

#else

EvdevInputBackend::~EvdevInputBackend() = default;

std::vector<DeviceDescription> EvdevInputBackend::enumerate()
{
    return {};
}

std::expected<DeviceDescription, std::string> EvdevInputBackend::open(DeviceIdentity)
{
    return std::unexpected("evdev belongs to Linux and this is not Linux");
}

void EvdevInputBackend::close()
{
}

std::expected<DeviceSample, std::string> EvdevInputBackend::read(std::chrono::milliseconds)
{
    return std::unexpected("no device is open");
}

std::expected<void, std::string> EvdevInputBackend::writeTorque(double)
{
    return std::unexpected("no device is open");
}

std::expected<double, std::string> EvdevInputBackend::setRotationRange(double)
{
    return std::unexpected("no device is open");
}

#endif

} // namespace raceengine
