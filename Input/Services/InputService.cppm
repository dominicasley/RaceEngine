module;

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stop_token>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

export module raceengine.input:InputService;

import :DeviceProfile;
import :DriverInput;
import :InputBackend;
import :InputMapping;
import raceengine.graphics;

namespace raceengine
{

// The keyboard, which reaches the game through the window like every other key.
//
// It stays on GLFW deliberately: GLFW owns the window, the graphics surface, the keyboard and the
// mouse, and the only reason a wheel does not is that GLFW's joystick API is read-only and
// main-thread-only. Nothing about a keyboard needs either.
export class KeyboardInputSource final : public IInputSource
{
    const IWindow& window;

public:
    explicit KeyboardInputSource(const IWindow& window) :
        window(window)
    {
    }

    [[nodiscard]] InputSourceKind kind() const override
    {
        return InputSourceKind::Keyboard;
    }

    // Always. A keyboard is the floor every other source falls back to, and asking whether one is
    // attached is a question X11 will not answer honestly anyway.
    [[nodiscard]] bool available() const override
    {
        return true;
    }

    [[nodiscard]] DriverInput sample() override
    {
        // Unattended runs report every key up, so this is where a gate's car gets its hands off the
        // controls without a branch anywhere above.
        return keyboardDriverInput(KeyboardDemand{.left = window.keyPressed(Key::A),
                                                  .right = window.keyPressed(Key::D),
                                                  .accelerate = window.keyPressed(Key::W),
                                                  .brake = window.keyPressed(Key::S),
                                                  .handbrake = window.keyPressed(Key::Space),
                                                  .upshift = window.keyPressed(Key::LeftShift),
                                                  .downshift = window.keyPressed(Key::LeftControl)});
    }
};

// The tick's own copy of what the device thread published. A copy and not a view onto shared state:
// the whole point of the thread is that a device can stall and the simulation cannot, so what the
// tick reads is the last thing it managed to take.
export struct DeviceView
{
    DeviceSample sample{};
    DeviceProfile profile;
    SteeringGeometry geometry;
    UpdateRate rate;
    DeviceIdentity identity;
    bool connected = false;
    InputSourceKind kind = InputSourceKind::None;
};

export class DeviceInputSource final : public IInputSource
{
    const DeviceView& view;

public:
    explicit DeviceInputSource(const DeviceView& view) :
        view(view)
    {
    }

    [[nodiscard]] InputSourceKind kind() const override
    {
        return view.kind;
    }

    [[nodiscard]] bool available() const override
    {
        return view.connected;
    }

    [[nodiscard]] DriverInput sample() override
    {
        return deviceDriverInput(view.profile, view.geometry, view.sample);
    }
};

export struct InputOptions
{
    // Where per-device profiles live, keyed on identity. Empty reads nothing and writes nothing,
    // which is what a run with no home directory gets and what an unattended one is given.
    std::string profileDirectory;
    // What this car's own rim would turn, lock to lock, for the rack to reach its stops.
    double vehicleRotationDegrees = 900.0;
    // No device is opened, no thread is started and nothing is written. Gates run this way, and so
    // does any run on a machine somebody else is using.
    bool unattended = false;
    // How long between looks for a device that is not there. A wheel gets switched on after the
    // game more often than before it.
    std::chrono::milliseconds rediscoveryInterval{2000};
    // How long one read waits before answering with what it already had. It bounds how long a stop
    // request waits, and nothing else: a device with something to say wakes the wait immediately.
    std::chrono::milliseconds readTimeout{4};
    // What this game would use if the device had it. Stated by the composition root rather than
    // assumed here, because what is worth a warning depends on what the game was going to do.
    std::vector<CapabilityRequest> wanted;
};

// The device layer: one thread, one device, and a struct per tick.
//
// The thread exists because a wheel answers about five hundred times a second and a frame is drawn
// sixty, and because a device that stops answering must not be able to hold up a simulation. It
// never touches the physics and the physics never waits on it — the tick takes the published sample
// if it can have it without waiting and uses the previous one if it cannot, which at a hundred and
// twenty ticks a second is eight milliseconds of a wheel position in the very rare case anything
// contends at all.
export class InputService
{
    spdlog::logger& logger;
    IInputBackend& backend;
    InputOptions options;

    // Written by the device thread, read by the tick, and never held for longer than a copy.
    mutable std::mutex publication;
    DeviceSample publishedSample{};
    DeviceProfile publishedProfile;
    UpdateRate publishedRate;
    DeviceIdentity publishedIdentity;
    bool publishedConnected = false;
    InputSourceKind publishedKind = InputSourceKind::None;
    // Bumped whenever the profile behind the sample changes, which is once per connection. The tick
    // copies the profile only when this moves, so the per-tick cost is a fixed-size struct and never
    // an allocation.
    std::atomic<std::uint32_t> profileGeneration{0};

    // The tick's side. Touched by no other thread.
    DeviceView view;
    std::uint32_t takenGeneration = 0;
    InputSourceKind active = InputSourceKind::None;
    DeviceInputSource deviceSource;
    KeyboardInputSource keyboardSource;
    std::array<IInputSource*, 2> sources;

    // What was said last, so that a device that is not there is mentioned once rather than every
    // two seconds for the length of a session. Device thread only.
    bool announcedAbsence = false;

    std::mutex waiting;
    std::condition_variable_any wake;

    void pump(const std::stop_token& stopToken);
    void attach();
    void detach(const std::string& reason);
    void publish(const DeviceSample& sample);
    [[nodiscard]] DeviceProfile profileFor(const DeviceDescription& description, const DeviceSample& rest);
    void refresh();

    // Last, so it stops and joins before anything it touches on the way down is destroyed.
    std::jthread reader;

public:
    InputService(spdlog::logger& logger, IInputBackend& backend, const IWindow& window, InputOptions options);
    InputService(const InputService&) = delete;
    InputService(InputService&&) = delete;
    InputService& operator=(const InputService&) = delete;
    InputService& operator=(InputService&&) = delete;
    ~InputService();

    // Once per physics tick. Everything above this is a device and everything below it is a car.
    [[nodiscard]] DriverInput sample();

    [[nodiscard]] InputSourceKind activeKind() const
    {
        return active;
    }

    // What the device is answering at, as measured rather than as written down. Zero until
    // something has been read from it.
    [[nodiscard]] UpdateRate updateRate() const
    {
        return view.rate;
    }

    // The steering travel this car's own rim would have. The device's is the device's, and the
    // mapping between them is what `rackFromRim` states.
    void setVehicleRotation(const double degrees)
    {
        options.vehicleRotationDegrees = degrees;
        view.geometry.vehicleDegrees = degrees;
    }

    // What this service has taken, so that anything else that might enumerate devices — a joystick
    // layer inside the window library, most obviously — can leave it alone. Nothing does today,
    // which is why GLFW opens no input node at all in this engine.
    [[nodiscard]] DeviceIdentity claimedIdentity() const
    {
        return view.identity;
    }

    [[nodiscard]] bool connected() const
    {
        return view.connected;
    }
};

// Where profiles go when the game does not say. `XDG_CONFIG_HOME` then `HOME` on Linux, `APPDATA`
// on Windows; empty when neither is set, which is a run that remembers nothing rather than one that
// writes somewhere surprising.
export [[nodiscard]] std::string defaultProfileDirectory();

} // namespace raceengine

namespace raceengine
{

namespace
{

[[nodiscard]] std::string environment(const char* name)
{
    const auto* value = std::getenv(name);

    return value == nullptr ? std::string() : std::string(value);
}

} // namespace

std::string defaultProfileDirectory()
{
#if defined(_WIN32)
    if (const auto appData = environment("APPDATA"); !appData.empty())
    {
        return appData + "\\RaceEngine\\input";
    }
#else
    if (const auto configHome = environment("XDG_CONFIG_HOME"); !configHome.empty())
    {
        return configHome + "/raceengine/input";
    }

    if (const auto home = environment("HOME"); !home.empty())
    {
        return home + "/.config/raceengine/input";
    }
#endif

    return {};
}

InputService::InputService(spdlog::logger& logger, IInputBackend& backend, const IWindow& window,
                           InputOptions options) :
    logger(logger),
    backend(backend),
    options(std::move(options)),
    deviceSource(view),
    keyboardSource(window),
    sources{&deviceSource, &keyboardSource}
{
    view.geometry.vehicleDegrees = this->options.vehicleRotationDegrees;

    // An unattended run opens no device and starts no thread. That is the same rule every other
    // input path here already keeps — a gate owns no hands at the controls — and it is also what
    // makes a capture byte-identical with and without all of this: nothing is enumerated, nothing
    // is read, and no profile is written to somebody's home directory by a build server.
    if (this->options.unattended)
    {
        logger.info("Input devices are not opened on an unattended run");

        return;
    }

    reader = std::jthread([this](const std::stop_token& stopToken) { pump(stopToken); });
}

InputService::~InputService()
{
    reader.request_stop();
    wake.notify_all();
}

DriverInput InputService::sample()
{
    refresh();

    for (auto* source : sources)
    {
        if (!source->available())
        {
            continue;
        }

        active = source->kind();

        return source->sample();
    }

    active = InputSourceKind::None;

    return DriverInput{};
}

void InputService::refresh()
{
    // try_lock and not lock: this runs on the simulation's thread, and the one thing the device
    // must never be able to do is hold it up. A tick that cannot have the newest sample uses the
    // one it already has, which is a wheel position eight milliseconds old in the rare case the
    // device thread is mid-publish at exactly this instant.
    auto held = std::unique_lock<std::mutex>(publication, std::try_to_lock);
    if (!held.owns_lock())
    {
        return;
    }

    view.sample = publishedSample;
    view.rate = publishedRate;
    view.identity = publishedIdentity;
    view.connected = publishedConnected;
    view.kind = publishedKind;

    if (const auto generation = profileGeneration.load(std::memory_order_acquire); generation != takenGeneration)
    {
        view.profile = publishedProfile;
        view.geometry.deviceDegrees = publishedProfile.rotationDegrees;
        view.geometry.vehicleDegrees = options.vehicleRotationDegrees;
        takenGeneration = generation;
    }
}

void InputService::publish(const DeviceSample& sample)
{
    const auto rate = backend.updateRate();

    const auto guard = std::lock_guard<std::mutex>(publication);
    publishedSample = sample;
    publishedRate = rate;
}

DeviceProfile InputService::profileFor(const DeviceDescription& description, const DeviceSample& rest)
{
    auto seeded = seedDeviceProfile(description, rest);

    if (options.profileDirectory.empty())
    {
        return seeded;
    }

    const auto path = std::filesystem::path(options.profileDirectory) / deviceProfileFileName(description.identity);
    auto code = std::error_code();

    if (std::filesystem::exists(path, code))
    {
        auto file = std::ifstream(path, std::ios::binary);
        if (file)
        {
            const auto text = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

            if (auto parsed = parseDeviceProfile(text); parsed)
            {
                logger.info("Input profile read from {}", path.string());

                return std::move(parsed).value();
            }
            else
            {
                // Warned and left alone. Overwriting somebody's calibration because this engine
                // could not read one line of it is the one outcome nobody wants, and the device
                // still works from what it says about itself in the meantime.
                logger.warn("The input profile at {} was not usable and this device is running on what it says about "
                            "itself instead: {}",
                            path.string(), parsed.error());
            }
        }

        return seeded;
    }

    std::filesystem::create_directories(path.parent_path(), code);

    auto file = std::ofstream(path, std::ios::binary | std::ios::trunc);
    if (file)
    {
        file << deviceProfileToText(seeded);
        logger.info("Input profile written to {}", path.string());
    }

    return seeded;
}

void InputService::attach()
{
    const auto devices = backend.enumerate();
    if (devices.empty())
    {
        if (!announcedAbsence)
        {
            logger.info("No wheel or pad is connected; the keyboard is driving");
            announcedAbsence = true;
        }

        return;
    }

    const auto opened = backend.open(devices.front().identity);
    if (!opened)
    {
        if (!announcedAbsence)
        {
            logger.info("A device was seen and not taken, so the keyboard is driving: {}", opened.error());
            announcedAbsence = true;
        }

        return;
    }

    const auto& description = *opened;

    // Reading with no wait answers with the state the open already took off the device, which is
    // every axis where it rests. That is what a profile is seeded from, and it is why a pedal whose
    // polarity nobody can state in advance still comes out the right way up.
    const auto rest = backend.read(std::chrono::milliseconds{0});
    const auto resting = rest ? *rest : DeviceSample{};

    auto profile = profileFor(description, resting);

    // A wheel says what it is turned to and a pad does not, which is the one thing that separates
    // them without a table of product ids nobody can keep current.
    const auto kind =
        description.capabilities.has(DeviceCapability::ReadRotationRange) || description.rotationDegrees > 0.0
            ? InputSourceKind::Wheel
            : InputSourceKind::Gamepad;

    const auto rate = backend.updateRate();

    logger.info("Input device attached: {} ({:04x}:{:04x}) as a {} through {}, {} degrees lock to lock, reporting at "
                "up to {:.0f} Hz",
                description.name, description.identity.vendor, description.identity.product, inputSourceKindName(kind),
                backend.platform(), profile.rotationDegrees, rate.inputHz);

    // Degradation, once, at the moment there is a device to degrade against. Not a refusal: a base
    // whose tuning menu the host cannot reach is a perfectly good wheel, and a game that would not
    // start on one would be refusing hardware for being different.
    for (const auto& request : options.wanted)
    {
        if (const auto shortfall = capabilityShortfall(description.capabilities, request); shortfall)
        {
            logger.warn("{}", *shortfall);
        }
    }

    announcedAbsence = false;

    const auto guard = std::lock_guard<std::mutex>(publication);
    publishedIdentity = description.identity;
    publishedSample = resting;
    publishedProfile = std::move(profile);
    publishedRate = rate;
    publishedConnected = true;
    publishedKind = kind;
    profileGeneration.fetch_add(1, std::memory_order_release);
}

void InputService::detach(const std::string& reason)
{
    backend.close();

    logger.info("Input device released, so the keyboard is driving: {}", reason);

    const auto guard = std::lock_guard<std::mutex>(publication);
    publishedConnected = false;
    publishedKind = InputSourceKind::None;
    publishedIdentity = DeviceIdentity{};
    publishedSample = DeviceSample{};
}

void InputService::pump(const std::stop_token& stopToken)
{
    auto nextAttempt = std::chrono::steady_clock::now();

    while (!stopToken.stop_requested())
    {
        if (!backend.opened())
        {
            if (const auto now = std::chrono::steady_clock::now(); now >= nextAttempt)
            {
                nextAttempt = now + options.rediscoveryInterval;
                attach();
            }

            if (!backend.opened())
            {
                // Stoppable, so shutdown does not wait out a rediscovery interval. The predicate is
                // never true; the wait exists to be interrupted.
                auto held = std::unique_lock<std::mutex>(waiting);
                static_cast<void>(wake.wait_for(held, stopToken, options.rediscoveryInterval, [] { return false; }));
                continue;
            }
        }

        auto taken = backend.read(options.readTimeout);
        if (!taken)
        {
            detach(taken.error());
            nextAttempt = std::chrono::steady_clock::now() + options.rediscoveryInterval;
            continue;
        }

        publish(*taken);
    }

    backend.close();
}

} // namespace raceengine
