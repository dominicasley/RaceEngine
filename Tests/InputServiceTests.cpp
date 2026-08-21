#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;
import raceengine.tests.input;
import raceengine.tests.log;

using Catch::Approx;
using raceengine::CapabilityRequest;
using raceengine::DeviceCapability;
using raceengine::InputOptions;
using raceengine::InputService;
using raceengine::InputSourceKind;
using raceengine::tests::CapturedLog;
using raceengine::tests::RecordingInputBackend;
using raceengine::tests::SilentWindow;

namespace
{

// Everything short so a state change lands inside the wait below rather than inside the two second
// interval a real session uses.
[[nodiscard]] InputOptions briskOptions()
{
    auto options = InputOptions{};
    options.rediscoveryInterval = std::chrono::milliseconds{1};
    options.readTimeout = std::chrono::milliseconds{1};
    options.vehicleRotationDegrees = 756.0;

    return options;
}

// The device layer is a thread, so a test that asserted immediately would be asserting on the
// scheduler. The wait is bounded and its expiry is the failure: either the service reached the
// state or the test says so, and neither outcome depends on how busy the machine is.
template <class Predicate> [[nodiscard]] bool settles(InputService& service, Predicate reached)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};

    while (std::chrono::steady_clock::now() < deadline)
    {
        // A tick is what moves the published sample into the service's own view, so the poll has to
        // be a tick and not a sleep.
        std::ignore = service.sample();

        if (reached())
        {
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    return false;
}

} // namespace

TEST_CASE("no device is a keyboard game rather than a refusal", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    const auto demand = service.sample();

    REQUIRE(service.activeKind() == InputSourceKind::Keyboard);
    REQUIRE(!service.connected());
    REQUIRE(demand.steering == Approx(0.0));
    REQUIRE(demand.throttle == Approx(0.0));
}

TEST_CASE("a device that is not there is mentioned once and not once per look", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    // A one millisecond rediscovery interval over a hundred milliseconds is a hundred looks. A
    // diagnostic storm is what the report-once rule exists to stop, and it is the only thing that
    // would make an absent wheel worse than no wheel at all.
    std::this_thread::sleep_for(std::chrono::milliseconds{120});
    std::ignore = service.sample();

    REQUIRE(log.occurrences("No wheel or pad is connected") == 1);
}

TEST_CASE("an unattended run opens nothing and starts no thread", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();

    backend.attach();

    auto options = briskOptions();
    options.unattended = true;

    auto service = InputService(log.sink(), backend, window, options);

    std::this_thread::sleep_for(std::chrono::milliseconds{50});

    const auto demand = service.sample();

    // The device is there and is deliberately not taken: this is what keeps a gate's capture
    // byte-identical whatever is plugged into the machine it runs on.
    REQUIRE(!backend.opened());
    REQUIRE(!service.connected());
    REQUIRE(service.activeKind() == InputSourceKind::Keyboard);
    REQUIRE(demand.steering == Approx(0.0));
    REQUIRE(demand.throttle == Approx(0.0));
}

TEST_CASE("a wheel switched on mid-session takes over from the keyboard", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    REQUIRE(service.sample().steering == Approx(0.0));
    REQUIRE(service.activeKind() == InputSourceKind::Keyboard);

    backend.attach();

    REQUIRE(settles(service, [&service] { return service.connected(); }));
    REQUIRE(service.activeKind() == InputSourceKind::Wheel);

    // The rim's own range against this car's, which is what a wheel means by full lock.
    backend.turn(65535);
    REQUIRE(settles(service, [&service] { return service.sample().steering > 0.99; }));

    // Half the load cell's travel against a 450 N maximum is full pressure.
    backend.press(32767);
    REQUIRE(settles(service, [&service] { return service.sample().brake > 0.99; }));
}

TEST_CASE("the device source hands out the newest report and carries no state", "[input][service]")
{
    // This replaces a test that pinned the opposite (2026-08-21), and the reason is worth keeping
    // because the pair of them is the whole argument for putting the simulation on its own thread.
    //
    // The demand used to be **reconstructed across the engine's catch-up burst**. Simulated time
    // advanced in bursts and a steering wheel does not: the engine ran a frame's worth of fixed
    // steps back to back — measured on the rig, three or four inside a quarter of a millisecond —
    // and every one of them read the same device report, so the demand a fixed-step simulation saw
    // was a staircase climbing once per rendered frame. That mattered because the steering rack
    // differentiates it twice, through Coulomb friction saturating above 10 mm/s and viscous damping
    // linear in the same velocity. On the rig's 300-second trace rack travel was unchanged between
    // 94.2% of consecutive ticks inside a burst, so both terms switched fully on and fully off at
    // the frame rate: 0.4254 N·m of torque step on a tick that moved against 0.0003 N·m on one that
    // did not. That was the buzz in the driver's hands, and interpolating the staircase was a
    // repair to a scheduling artefact rather than to a fault.
    //
    // **The simulation keeps its own fixed-rate clock now and there is no burst.** Ticks are 2.78 ms
    // apart and the device reports faster than that, so consecutive ticks read consecutive samples
    // of a wheel that is genuinely moving. What this pins is that the source is *stateless* — two
    // sources looking at one view agree, and asking twice without the device saying anything gives
    // the same answer twice — because the day any of it comes back, the burst has come back with it.
    auto view = raceengine::DeviceView{};

    view.connected = true;
    view.kind = InputSourceKind::Wheel;
    view.profile.rotationDegrees = 900.0;
    view.profile.axes[raceengine::axisIndex(raceengine::InputAxis::Steering)] =
        raceengine::AxisCalibration{.minimum = 0.0, .maximum = 65535.0, .centre = 32273.0};
    view.geometry = raceengine::SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 756.0};

    auto source = raceengine::DeviceInputSource(view);

    const auto turn = [&view](const std::int32_t raw)
    {
        view.sample.axes[raceengine::axisIndex(raceengine::InputAxis::Steering)] = raw;
    };

    turn(20000);
    const auto held = source.sample().steering;

    // A tick that arrives with nothing new to read gets the same answer, rather than one carried
    // part-way towards somewhere by a rule.
    REQUIRE(source.sample().steering == Approx(held).epsilon(1e-12));

    turn(45000);

    const auto moved = source.sample().steering;
    REQUIRE(moved > held);

    // The whole move, on the first tick that can see it. Nothing is spread across ticks to come.
    for (auto index = 0; index < 4; index++)
    {
        REQUIRE(source.sample().steering == Approx(moved).epsilon(1e-12));
    }

    // Stateless: a source that has watched the wheel travel from lock to lock and one that has just
    // been built answer identically off the same view.
    auto fresh = raceengine::DeviceInputSource(view);
    REQUIRE(source.sample().steering == Approx(fresh.sample().steering).epsilon(1e-12));
}

TEST_CASE("the keyboard is taken on the window's thread and the simulation reads a copy", "[input][service]")
{
    // The one source that cannot be sampled where the others are: it arrives through GLFW, and
    // `glfwGetKey` may only be called from the thread that made the window. So the demand is taken
    // by `pollWindow` on the main thread and read by `sample` on the simulation's.
    //
    // What this pins is that the split is not a leak — a key held does reach the car — and that it
    // takes a poll to do so. A `sample` that answered a key nobody had polled would mean the
    // keyboard was still being read from wherever the simulation happens to run.
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    window.hold(true);

    REQUIRE(service.sample().throttle == Approx(0.0));

    service.pollWindow();

    REQUIRE(service.activeKind() == InputSourceKind::Keyboard);
    REQUIRE(service.sample().throttle == Approx(1.0));

    window.hold(false);
    service.pollWindow();

    REQUIRE(service.sample().throttle == Approx(0.0));
}

TEST_CASE("a wheel switched off gives the car back to the keyboard", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    backend.attach();
    REQUIRE(settles(service, [&service] { return service.connected(); }));

    backend.turn(65535);
    REQUIRE(settles(service, [&service] { return service.sample().steering > 0.99; }));

    backend.vanish();

    REQUIRE(settles(service, [&service] { return !service.connected(); }));
    REQUIRE(service.activeKind() == InputSourceKind::Keyboard);
    // Not the last thing the wheel said. A device that went away leaves no demand behind, or a car
    // whose wheel was unplugged at full lock would go on turning.
    REQUIRE(service.sample().steering == Approx(0.0));
}

TEST_CASE("a device the game outranks still loses to nothing being held", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    backend.attach();
    REQUIRE(settles(service, [&service] { return service.connected(); }));

    // A key held while a wheel is connected does nothing: the highest available source answers
    // outright, because merging two steering demands is a car that fights whoever is holding it.
    // Polled, so that what is being shown is the wheel outranking a demand that genuinely arrived
    // rather than one that was never taken.
    window.hold(true);
    service.pollWindow();

    REQUIRE(service.activeKind() == InputSourceKind::Wheel);
    REQUIRE(service.sample().throttle == Approx(0.0));
}

TEST_CASE("a capability the device lacks is one warning and the game carries on", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();

    auto options = briskOptions();
    options.wanted = {CapabilityRequest{.capability = DeviceCapability::ConstantForce,
                                        .purpose = "force feedback",
                                        .fallback = "the wheel stays free"},
                      CapabilityRequest{.capability = DeviceCapability::TuningMenu,
                                        .purpose = "following the base's own tuning profile",
                                        .fallback = "the game's own settings stand alone"}};

    auto service = InputService(log.sink(), backend, window, options);

    backend.attach();
    REQUIRE(settles(service, [&service] { return service.connected(); }));

    // The one it has is silent; the one it does not have is said once, at the moment there was a
    // device to say it about.
    REQUIRE(log.occurrences("force feedback") == 0);
    REQUIRE(log.occurrences("following the base's own tuning profile") == 1);
    REQUIRE(service.activeKind() == InputSourceKind::Wheel);
}

TEST_CASE("the update rate is read off the backend rather than written down", "[input][service]")
{
    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, briskOptions());

    REQUIRE(service.updateRate().inputHz == Approx(0.0));

    backend.attach();
    REQUIRE(settles(service, [&service] { return service.connected(); }));

    REQUIRE(service.updateRate().inputHz == Approx(500.0));
    REQUIRE(service.updateRate().measured);
}

TEST_CASE("a torque reaches the device through the seam force feedback will use", "[input][service]")
{
    // Nothing calls this yet — force feedback is the next piece of work — and it is written and
    // exercised now because an interface grown to fit a feature after a season of tuning against
    // one backend is exactly the retrofit two implementations exist to prevent.
    auto backend = RecordingInputBackend();

    backend.attach();
    REQUIRE(backend.open(raceengine::DeviceIdentity{.vendor = 0x0eb7, .product = 0x0004}));
    REQUIRE(backend.writeTorque(-0.5));
    REQUIRE(backend.torque() == Approx(-0.5));

    backend.close();
    REQUIRE(!backend.writeTorque(0.5));
}

TEST_CASE("a profile is written once and read back on the next connection", "[input][service]")
{
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("raceengine-input-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    auto options = briskOptions();
    options.profileDirectory = directory.string();

    {
        auto log = CapturedLog();
        auto backend = RecordingInputBackend();
        auto window = SilentWindow();
        auto service = InputService(log.sink(), backend, window, options);

        backend.attach();
        REQUIRE(settles(service, [&service] { return service.connected(); }));
        REQUIRE(log.occurrences("Input profile written to") == 1);
    }

    REQUIRE(std::filesystem::exists(directory / "0eb7-0004.profile"));

    {
        auto log = CapturedLog();
        auto backend = RecordingInputBackend();
        auto window = SilentWindow();
        auto service = InputService(log.sink(), backend, window, options);

        backend.attach();
        REQUIRE(settles(service, [&service] { return service.connected(); }));

        // Read, not rewritten: a calibration somebody made survives the next session, which is the
        // whole reason it is keyed on the device and not on the run.
        REQUIRE(log.occurrences("Input profile read from") == 1);
        REQUIRE(log.occurrences("Input profile written to") == 0);
    }

    auto code = std::error_code();
    std::filesystem::remove_all(directory, code);
}

TEST_CASE("an unreadable profile is reported and left alone", "[input][service]")
{
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("raceengine-input-bad-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto code = std::error_code();
    std::filesystem::create_directories(directory, code);

    const auto path = directory / "0eb7-0004.profile";
    {
        auto file = std::ofstream(path);
        file << "version 1\nidentity 0eb7 0004\nwobble 3\n";
    }

    auto options = briskOptions();
    options.profileDirectory = directory.string();

    auto log = CapturedLog();
    auto backend = RecordingInputBackend();
    auto window = SilentWindow();
    auto service = InputService(log.sink(), backend, window, options);

    backend.attach();
    REQUIRE(settles(service, [&service] { return service.connected(); }));

    // Overwriting somebody's calibration because one line of it would not read is the one outcome
    // nobody wants; the device runs on what it says about itself in the meantime.
    REQUIRE(log.occurrences("was not usable") == 1);

    auto file = std::ifstream(path);
    const auto text = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    REQUIRE(text.find("wobble") != std::string::npos);

    std::filesystem::remove_all(directory, code);
}
