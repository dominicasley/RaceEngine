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

TEST_CASE("the steering demand is reconstructed across a catch-up burst", "[input][service]")
{
    // **Simulated time advances in bursts and a steering wheel does not.** The engine runs a
    // frame's worth of fixed steps back to back — measured on the rig, three or four inside a
    // quarter of a millisecond — and every one of them reads the device thread's newest sample,
    // which over that quarter millisecond is the same sample. So a 120 Hz simulation saw the demand
    // as a staircase climbing once per rendered frame, and the ticks in between saw a wheel that
    // was not moving at all.
    //
    // Nothing downstream can tell the difference except whatever differentiates the demand, and the
    // steering rack does: its Coulomb friction saturates above 10 mm/s and its viscous damping is
    // linear in the same velocity. On the rig's 300-second trace rack travel was unchanged between
    // 94.2% of consecutive ticks inside a burst, so both terms switched fully on and fully off at
    // the frame rate — 0.4254 N·m of torque step on a tick that moved against 0.0003 N·m on one
    // that did not. That was the buzz in the driver's hands.
    //
    // Driven against the source directly rather than through the service: no thread, no device and
    // no `refresh` racing a publication for a `try_lock`, so what is measured is the reconstruction
    // and not the scheduler. The view is written here exactly as the device thread would leave it.
    auto phase = raceengine::CatchUpPhase{};
    auto view = raceengine::DeviceView{};

    view.connected = true;
    view.kind = InputSourceKind::Wheel;
    view.profile.rotationDegrees = 900.0;
    view.profile.axes[raceengine::axisIndex(raceengine::InputAxis::Steering)] =
        raceengine::AxisCalibration{.minimum = 0.0, .maximum = 65535.0, .centre = 32273.0};
    view.geometry = raceengine::SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 756.0};

    auto source = raceengine::DeviceInputSource(view, phase);

    const auto turn = [&view](const std::int32_t raw)
    {
        view.sample.axes[raceengine::axisIndex(raceengine::InputAxis::Steering)] = raw;
    };

    turn(20000);
    phase = raceengine::CatchUpPhase{.index = 0, .steps = 1};
    const auto before = source.sample().steering;

    // One frame of catch-up: the wheel moved once, and four ticks are about to simulate the time it
    // moved in. Before this, all four read the same report — the first took the whole move and the
    // three after it saw a wheel standing still.
    turn(45000);

    auto served = std::vector<double>{};
    for (auto index = std::uint32_t{0}; index < 4; index++)
    {
        phase = raceengine::CatchUpPhase{.index = index, .steps = 4};
        served.push_back(source.sample().steering);
    }

    CAPTURE(before, served[0], served[1], served[2], served[3]);

    // Evenly, because the hands moved evenly as far as anything here can know.
    const auto stride = (served[3] - before) / 4.0;

    REQUIRE(stride > 0.0);

    for (auto index = std::size_t{0}; index < served.size(); index++)
    {
        REQUIRE(served[index] == Approx(before + stride * static_cast<double>(index + 1)).epsilon(1e-9));
    }

    // The last tick lands exactly on the device's own newest sample, so **nothing is delayed by any
    // of this**: the next burst, of one, returns the very same value.
    phase = raceengine::CatchUpPhase{.index = 0, .steps = 1};
    REQUIRE(source.sample().steering == Approx(served[3]).epsilon(1e-9));

    // And a burst of one is the identity, which is what makes all of this inert at frame rates at
    // or above the tick rate.
    turn(30000);
    const auto single = source.sample().steering;

    turn(52000);
    REQUIRE(source.sample().steering != Approx(single));

    phase = raceengine::CatchUpPhase{.index = 0, .steps = 1};
    turn(60000);

    auto direct = raceengine::DeviceInputSource(view, phase);
    std::ignore = direct.sample();

    REQUIRE(source.sample().steering == Approx(direct.sample().steering).epsilon(1e-9));
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
    window.hold(true);

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
