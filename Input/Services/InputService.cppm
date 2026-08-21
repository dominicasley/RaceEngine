module;

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <shared_mutex>
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

    // The device's newest report, mapped through its profile. Nothing else, and that is the change
    // (2026-08-21).
    //
    // This used to reconstruct the steering axis across the engine's catch-up burst, because
    // simulated time advanced in bursts and a steering wheel does not: the engine ran a frame's
    // worth of ticks back to back — three or four inside a quarter of a millisecond — and every one
    // of them read the *same* device sample, so the demand a fixed-step simulation saw was a
    // staircase climbing once per rendered frame. That mattered because the steering rack
    // differentiates the demand twice over, through its Coulomb friction and its viscous damping,
    // and the resulting impulse train at the frame rate was the buzz in the driver's hands.
    //
    // **The simulation has its own fixed-rate clock now and there is no burst to paper over.** Each
    // tick is a separate wake-up 2.78 ms after the last, and the device reports faster than that, so
    // consecutive ticks read consecutive samples of a wheel that is genuinely moving. Interpolating
    // across a burst of one was already the identity; with no bursts left the whole apparatus —
    // `CatchUpPhase`, the served/burst origin pair and the engine's `setCatchUpPhase` — was papering
    // over a scheduling artefact that has been removed instead. That it could go is the sign the
    // threading change is the right one.
    [[nodiscard]] DriverInput sample() override
    {
        return deviceDriverInput(view.profile, view.geometry, view.sample);
    }
};

// What the device thread knows about the device as it stands, rather than what the tick last
// managed to copy. Two consumers want two different things from the same device: the simulation
// wants a sample it can have without waiting, and the force feedback thread wants to know whether
// there is anything to write to and what a torque fraction of one would mean on it.
export struct DeviceLink
{
    bool connected = false;
    // Whether a signed torque can be written at all. A node this session may only read is a wheel
    // that can be driven and cannot push back, which is a degradation and not a failure.
    bool takesTorque = false;
    // N·m at the rim for a fraction of one, from the device's own profile. Zero when nothing is
    // connected, which is what stops a mapping being computed against a peak nobody stated.
    double peakTorque = 0.0;
    DeviceIdentity identity;
    // When the device last said something, on `steady_clock`'s own clock. What an end-to-end latency
    // is measured from.
    std::uint64_t sampleTimestampNanos = 0;
    // The physical rim's own angle, `rimDegrees()`'s answer restated on the thread-safe side: the
    // force feedback writer differences it against the sample timestamps above to measure how fast
    // the rim is actually turning, which is what the mapping's damper takes. False when what is
    // connected has no rim.
    bool hasRim = false;
    double rimDegrees = 0.0;
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
    bool publishedTakesTorque = false;
    InputSourceKind publishedKind = InputSourceKind::None;
    // Bumped whenever the profile behind the sample changes, which is once per connection. The tick
    // copies the profile only when this moves, so the per-tick cost is a fixed-size struct and never
    // an allocation.
    std::atomic<std::uint32_t> profileGeneration{0};

    // The keyboard's demand, taken on the **main** thread and read by the simulation's.
    //
    // The keyboard is the one source that cannot be sampled where the others are: it arrives through
    // GLFW, and `glfwGetKey` may only be called from the thread that made the window. So the split
    // is not a preference — the main thread polls it once per rendered frame and publishes the
    // struct, and the simulation thread reads its own copy. Nothing is lost by it: GLFW's key state
    // only changes inside `glfwPollEvents`, which runs once a frame, so polling it faster than the
    // frame rate would re-read the same answer.
    mutable std::mutex keyboardPublication;
    DriverInput keyboardDemand{};

    // The simulation thread's side. Touched by no other thread.
    DeviceView view;
    DriverInput takenKeyboard{};
    std::uint32_t takenGeneration = 0;
    InputSourceKind active = InputSourceKind::None;
    DeviceInputSource deviceSource;
    KeyboardInputSource keyboardSource;

    // What was said last, so that a device that is not there is mentioned once rather than every
    // two seconds for the length of a session. Device thread only.
    bool announcedAbsence = false;

    // The car's lock-to-lock, for the reader thread to push onto the base at attach. Atomic because
    // the tick side states it and the reader consumes it.
    std::atomic<double> wantedRotationDegrees{900.0};

    // The device's *existence*, not its state. Two threads sit on one file descriptor — this
    // service's reader and the force feedback writer — and reading and writing an evdev node
    // concurrently is fine, because they are different ioctls over disjoint fields. What is not
    // fine is the node being closed underneath a write in flight, so open and close take this
    // exclusively and a write takes it shared. Reads need no lock at all: they happen on the same
    // thread as the open and the close.
    mutable std::shared_mutex deviceLifetime;

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

    // Once per rendered frame, on the thread that owns the window. It reads the keyboard and
    // nothing else — see `keyboardDemand` for why that one source cannot be taken where the rest
    // are. A run with no window and no keys still calls it; every key reads up.
    void pollWindow();

    // Once per physics tick, on the thread that steps the simulation. Everything above this is a
    // device and everything below it is a car.
    //
    // **This and every accessor reading `view` belong to that one thread**, which is what lets the
    // copy be lock-free on the read side. Anything else asking about the device — the force feedback
    // writer, or a game wanting the rim angle for a rendered wheel — goes through `deviceLink()`.
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
        // The reader thread consumes this at the next attach, so it crosses threads as an atomic
        // snapshot rather than through the options struct the tick side owns.
        wantedRotationDegrees.store(degrees);
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

    // The physical rim's own angle in degrees, positive in the same sense as positive demand: the
    // raw steering axis through its calibration, times the device's half lock-to-lock. It exists
    // for a rendered wheel that mirrors the driver's hands — which is a different question from the
    // demand, because the demand clamps at the car's own lock while a 900-degree rig keeps turning.
    // Nothing when what is driving is not a wheel: a keyboard has no rim to mirror.
    [[nodiscard]] std::optional<double> rimDegrees() const
    {
        if (!view.connected || view.kind != InputSourceKind::Wheel)
        {
            return std::nullopt;
        }

        const auto steering = axisIndex(InputAxis::Steering);
        const auto rim = normaliseBipolar(view.profile.axes[steering], static_cast<double>(view.sample.axes[steering]));

        return rim * view.profile.rotationDegrees * 0.5;
    }

    // When the device said what this tick is driving on, on `steady_clock`'s clock. Zero when
    // nothing said it — a keyboard, or a scripted run. The tick's own copy, so it is the stamp
    // belonging to the sample the demand was actually made from rather than to whatever the device
    // has said since.
    [[nodiscard]] std::uint64_t sampleTimestampNanos() const
    {
        return view.sample.timestampNanos;
    }

    // The device thread's own answer, safe to ask from any thread. The tick uses the accessors
    // above, which read its private copy; anything else — the force feedback writer is the only
    // caller today — has to ask here, because `view` belongs to the simulation and is written
    // without a lock.
    [[nodiscard]] DeviceLink deviceLink() const;

    // A signed fraction of the device's own maximum torque. The only route to the device's write
    // side, and it exists here rather than on the backend so that one object owns the file
    // descriptor's lifetime: a writer holding the backend directly can be halfway through an
    // `EVIOCSFF` when the reader decides the device has gone and closes it.
    //
    // Newton metres are not this function's business and never will be — see `ForceMapping`, which
    // is where the device's peak, its code grid and every other hardware fact live.
    [[nodiscard]] std::expected<void, std::string> writeTorque(double torqueFraction);
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
    keyboardSource(window)
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

void InputService::pollWindow()
{
    // Sampled outside the lock: `keyPressed` reaches GLFW, and holding a lock the simulation thread
    // may be waiting on across a call into a third-party library is how a fixed-rate loop discovers
    // somebody else's mutex.
    const auto demand = keyboardSource.sample();

    const auto guard = std::lock_guard<std::mutex>(keyboardPublication);
    keyboardDemand = demand;
}

DriverInput InputService::sample()
{
    refresh();

    // A device first and the keyboard as the floor, which is the order the two sources were tried
    // in when they lived in one array. They are no longer symmetric — one is read here and the
    // other arrives from the main thread — so the priority is stated rather than iterated.
    if (deviceSource.available())
    {
        active = deviceSource.kind();

        return deviceSource.sample();
    }

    active = keyboardSource.kind();

    return takenKeyboard;
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

    // The keyboard's, on the same terms: taken if it can be had without waiting, and otherwise the
    // copy this thread already holds. A key held down does not change between one 360 Hz tick and
    // the next, and the main thread only writes this once a frame, so the copy is never stale by
    // anything a driver could feel.
    if (auto keys = std::unique_lock<std::mutex>(keyboardPublication, std::try_to_lock); keys.owns_lock())
    {
        takenKeyboard = keyboardDemand;
    }
}

DeviceLink InputService::deviceLink() const
{
    const auto guard = std::lock_guard<std::mutex>(publication);

    const auto wheel = publishedConnected && publishedKind == InputSourceKind::Wheel;
    const auto steering = axisIndex(InputAxis::Steering);
    const auto rim =
        wheel ? normaliseBipolar(publishedProfile.axes[steering], static_cast<double>(publishedSample.axes[steering])) *
                    publishedProfile.rotationDegrees * 0.5
              : 0.0;

    return DeviceLink{.connected = publishedConnected,
                      .takesTorque = publishedTakesTorque,
                      .peakTorque = publishedConnected ? publishedProfile.peakTorque : 0.0,
                      .identity = publishedIdentity,
                      .sampleTimestampNanos = publishedSample.timestampNanos,
                      .hasRim = wheel,
                      .rimDegrees = rim};
}

std::expected<void, std::string> InputService::writeTorque(const double torqueFraction)
{
    const auto guard = std::shared_lock<std::shared_mutex>(deviceLifetime);

    if (!backend.opened())
    {
        return std::unexpected("no device is open");
    }

    return backend.writeTorque(torqueFraction);
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

    auto description = *opened;

    // The base is *told* the car's lock-to-lock rather than asked for its own, which is what a
    // simulator does with a wheel: the rig's physical stops land exactly on the car's lock and
    // every mapping between hands, axis and rack collapses to one-to-one. It is also the only way
    // to be right at all on a base whose firmware has drifted from what its range file reads —
    // this one arrived soft-locked near 240 degrees while the file said 900, and a game that
    // trusts a report it could have replaced is calibrating against a lie.
    if (description.capabilities.has(DeviceCapability::SetRotationRange))
    {
        if (const auto set = backend.setRotationRange(wantedRotationDegrees.load()); set)
        {
            description.rotationDegrees = *set;
        }
        else
        {
            logger.info("The wheel's rotation range could not be set, so its own {:.0f} degrees stand: {}",
                        description.rotationDegrees, set.error());
        }
    }

    // Reading with no wait answers with the state the open already took off the device, which is
    // every axis where it rests. That is what a profile is seeded from, and it is why a pedal whose
    // polarity nobody can state in advance still comes out the right way up.
    const auto rest = backend.read(std::chrono::milliseconds{0});
    const auto resting = rest ? *rest : DeviceSample{};

    auto profile = profileFor(description, resting);

    // The range this session actually set beats whatever a saved profile remembers: the profile's
    // rotation describes the base as it was when the file was written, and the base was just told
    // otherwise.
    if (description.capabilities.has(DeviceCapability::SetRotationRange) && description.rotationDegrees > 0.0)
    {
        profile.rotationDegrees = description.rotationDegrees;
    }

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
    publishedTakesTorque = description.capabilities.has(DeviceCapability::ConstantForce);
    publishedKind = kind;
    profileGeneration.fetch_add(1, std::memory_order_release);
}

void InputService::detach(const std::string& reason)
{
    backend.close();

    logger.info("Input device released, so the keyboard is driving: {}", reason);

    const auto guard = std::lock_guard<std::mutex>(publication);
    publishedConnected = false;
    publishedTakesTorque = false;
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

                // Exclusive: nothing may be writing a torque to the node this is about to replace.
                const auto guard = std::lock_guard<std::shared_mutex>(deviceLifetime);
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

        // Unlocked deliberately: a read blocks for up to the timeout, and holding the lifetime lock
        // across it would stall every torque write by that long. It is safe because open and close
        // happen on this thread, so a read can never find the descriptor closed under it.
        auto taken = backend.read(options.readTimeout);
        if (!taken)
        {
            const auto guard = std::lock_guard<std::shared_mutex>(deviceLifetime);
            detach(taken.error());
            nextAttempt = std::chrono::steady_clock::now() + options.rediscoveryInterval;
            continue;
        }

        publish(*taken);
    }

    const auto guard = std::lock_guard<std::shared_mutex>(deviceLifetime);
    backend.close();
}

} // namespace raceengine
