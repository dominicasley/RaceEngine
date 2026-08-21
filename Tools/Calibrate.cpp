// A guided calibration for a wheel, its pedals and its paddles.
//
// It exists because a profile nobody has calibrated is a profile full of the device's *declared*
// bounds, and a device's declared bounds are not its travel. This base states 0..65535 on all three
// pedal axes and rests at 65535 on two of them and 64700 on the third — so a seeded profile has the
// brake reading 1.3% applied with nobody's foot on it, and every paddle unbound because a button a
// device says nothing about cannot be guessed. Both of those read, from the driving seat, as "the
// wheel does not work".
//
// A console program and not a screen in the game, deliberately. Calibration is a thing done once per
// rig and it wants both hands, a prompt that waits, and somewhere to say "that did not look right,
// do it again" — none of which a car's dashboard is. It also means it can be run when the game will
// not start, which is exactly when a wrong profile is suspected.
//
// It talks to `IInputBackend` rather than to `InputService`, because what calibration needs is the
// raw device code and the service's whole job is to have shaped it already.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>

import raceengine;

namespace
{

using namespace std::chrono_literals;

// How long a prompt keeps sampling after the driver has said they are ready. Long enough to catch
// the extreme of a pedal that is being held, short enough that nobody waits for it.
constexpr auto settleWindow = 250ms;

// Below this many device counts an axis has not been swept, it has been sat still next to. A wheel
// or a pedal worth calibrating moves through tens of thousands.
constexpr auto usefulTravel = 1000;

struct Extremes
{
    std::int32_t lowest = 0;
    std::int32_t highest = 0;
    bool seen = false;

    void add(const std::int32_t value)
    {
        lowest = seen ? std::min(lowest, value) : value;
        highest = seen ? std::max(highest, value) : value;
        seen = true;
    }
};

// Read the device until the driver presses return, keeping the extreme of every axis and the union
// of every button seen. The reading has to continue *while* waiting, which is why this is one loop
// rather than a prompt followed by a sample: an axis that is only read after the key is pressed
// records where the pedal ended up rather than how far it went.
struct Watcher
{
    raceengine::IInputBackend& backend;
    raceengine::DeviceIdentity identity{};
    std::array<Extremes, raceengine::inputAxisCount> axes{};
    std::array<std::int32_t, raceengine::inputAxisCount> latest{};
    std::uint64_t buttons = 0;
    std::uint64_t buttonsEverHigh = 0;
    // How many times the device fell off the bus during this session, and when it last did.
    int drops = 0;
    std::chrono::steady_clock::time_point nextReopen{};

    // A read that fails is almost never a read that failed. On a marginal USB link — a base behind a
    // couple of cheap hubs is the case that prompted this — the kernel logs a real `USB disconnect`
    // and the node is recreated under a new number a second later, and every read against the old
    // one errors for ever. Reported and recovered from rather than left as silence, because silence
    // here reads as "your pedal is broken" and sends somebody to look at the wrong end of the rig.
    void pump()
    {
        const auto sample = backend.read(2ms);
        if (!sample)
        {
            recover();

            return;
        }

        for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
        {
            axes[index].add(sample->axes[index]);
            latest[index] = sample->axes[index];
        }

        buttons = sample->buttons;
        buttonsEverHigh |= sample->buttons;
    }

    void recover()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now < nextReopen)
        {
            return;
        }

        drops++;
        std::cout << "\n    [the wheel dropped off the USB bus; waiting for it to come back]\n" << std::flush;

        backend.close();

        // Re-enumeration takes about a second and the node comes back under a new number, so this
        // opens by *identity* and lets the backend find wherever it landed.
        const auto deadline = now + 15s;
        while (std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(200ms);

            if (backend.open(identity))
            {
                std::cout << "    [back]\n" << std::flush;
                nextReopen = std::chrono::steady_clock::now() + 1s;

                return;
            }
        }

        std::cout << "    [it did not come back within fifteen seconds]\n" << std::flush;
        nextReopen = std::chrono::steady_clock::now() + 5s;
    }

    void reset()
    {
        axes = {};
        buttons = 0;
        buttonsEverHigh = 0;
    }
};

// Wait for return, pumping the device and showing what it reads while it waits.
//
// The live display is not decoration. Without it the only evidence of what happened is the number
// printed afterwards, and when that number is wrong there is nothing to say whether the axis never
// moved, the tool never read it, or the device was not there — which is exactly the position this
// tool put its first user in. `axis` is which one to show, or `inputAxisCount` to show none.
void waitForReturn(Watcher& watcher, const std::size_t axis)
{
    // `std::atomic` rather than a plain bool: the spin below reads it while the reader thread writes
    // it, and a data race is a data race even when the loop happens to exit.
    auto pressed = std::atomic<bool>{false};
    auto reader = std::thread(
        [&pressed]
        {
            std::string line;
            std::getline(std::cin, line);
            pressed.store(true);
        });

    auto nextDraw = std::chrono::steady_clock::now();

    while (!pressed.load())
    {
        watcher.pump();

        if (axis >= raceengine::inputAxisCount)
        {
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < nextDraw)
        {
            continue;
        }

        nextDraw = now + 50ms;

        const auto& seen = watcher.axes[axis];
        std::cout << "\r    reading " << watcher.latest[axis] << "  (seen " << seen.lowest << ".." << seen.highest
                  << ")            " << std::flush;
    }

    reader.join();

    if (axis < raceengine::inputAxisCount)
    {
        std::cout << "\r                                                                      \r" << std::flush;
    }
}

// Everything the axis did over the wait, for the caller to take what it needs from.
[[nodiscard]] Extremes captureRange(Watcher& watcher, const raceengine::InputAxis axis, const std::string_view hold)
{
    std::cout << "  " << hold << ", then press return:\n" << std::flush;

    watcher.reset();
    waitForReturn(watcher, raceengine::axisIndex(axis));

    const auto deadline = std::chrono::steady_clock::now() + settleWindow;
    while (std::chrono::steady_clock::now() < deadline)
    {
        watcher.pump();
    }

    return watcher.axes[raceengine::axisIndex(axis)];
}

// The extreme of one named axis over the wait, which is what a pedal or a lock is.
[[nodiscard]] std::int32_t captureAxis(Watcher& watcher, const raceengine::InputAxis axis, const std::string_view hold,
                                       const bool wantHighest)
{
    std::cout << "  " << hold << ", then press return:\n" << std::flush;

    watcher.reset();
    waitForReturn(watcher, raceengine::axisIndex(axis));

    const auto deadline = std::chrono::steady_clock::now() + settleWindow;
    while (std::chrono::steady_clock::now() < deadline)
    {
        watcher.pump();
    }

    const auto& seen = watcher.axes[raceengine::axisIndex(axis)];
    if (!seen.seen)
    {
        std::cout << "    nothing was read from the device\n";

        return 0;
    }

    return wantHighest ? seen.highest : seen.lowest;
}

// Where an axis has come to rest, which is a different question from how far it went. The steering
// centre is the case: captured as an extreme it reports wherever the wheel passed through on the way
// back, and coming back from full right lock that is full right lock.
[[nodiscard]] std::int32_t captureResting(Watcher& watcher, const raceengine::InputAxis axis,
                                          const std::string_view hold)
{
    std::cout << "  " << hold << ", then press return:\n" << std::flush;

    watcher.reset();
    waitForReturn(watcher, raceengine::axisIndex(axis));

    const auto deadline = std::chrono::steady_clock::now() + settleWindow;
    while (std::chrono::steady_clock::now() < deadline)
    {
        watcher.pump();
    }

    if (!watcher.axes[raceengine::axisIndex(axis)].seen)
    {
        std::cout << "    nothing was read from the device\n";

        return 0;
    }

    return watcher.latest[raceengine::axisIndex(axis)];
}

// A button, by waiting for one to go high. No return key here: a paddle is pulled with the hand that
// would otherwise be on the keyboard, and asking for both at once is how a calibration ends up
// recording the return key's own timing.
[[nodiscard]] std::int32_t captureButton(Watcher& watcher, const std::string_view action)
{
    std::cout << "  " << action << " (or wait five seconds to leave it unbound): " << std::flush;

    watcher.reset();

    // Whatever is already held when the prompt opens is not what is being asked for.
    for (auto step = 0; step < 100; step++)
    {
        watcher.pump();
    }

    const auto already = watcher.buttons;
    const auto deadline = std::chrono::steady_clock::now() + 5s;

    while (std::chrono::steady_clock::now() < deadline)
    {
        watcher.pump();

        const auto fresh = watcher.buttons & ~already;
        if (fresh == 0)
        {
            continue;
        }

        for (auto bit = 0; bit < 64; bit++)
        {
            if ((fresh & (std::uint64_t{1} << static_cast<std::uint64_t>(bit))) == 0)
            {
                continue;
            }

            std::cout << "bit " << bit << "\n";

            // Let go before the next prompt opens, or the next one sees this one still held.
            const auto release = std::chrono::steady_clock::now() + 3s;
            while (std::chrono::steady_clock::now() < release && (watcher.buttons & fresh) != 0)
            {
                watcher.pump();
            }

            return bit;
        }
    }

    std::cout << "nothing pressed, left as it was\n";

    return -1;
}

// A window that is not there. `InputService` takes one because focus decides whether input is
// reported at all, and a console tool has no window to lose focus from — so it says it always has it.
class NoWindow final : public raceengine::IWindow
{
public:
    void swapBuffers() const override
    {
    }
    void setMousePosition(int, int) override
    {
    }
    void setCursorMode(raceengine::CursorMode) override
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
    [[nodiscard]] bool keyPressed(raceengine::Key) const override
    {
        return false;
    }
    [[nodiscard]] const raceengine::WindowState& state() const override
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

private:
    raceengine::WindowState windowState{};
};

// What the *game* sees, which is a different question from what the device sends and is the one that
// was actually in doubt. Everything between the two — the evdev code's role, the profile's ends, the
// brake's force curve, the rack geometry — is in this path and in none of the raw dumps.
int monitor()
{
    auto sink = spdlog::stdout_color_mt("calibrate");
    auto backend = raceengine::createInputBackend(nullptr);
    if (!backend)
    {
        std::cout << "no input backend on this platform\n";

        return 1;
    }

    auto window = NoWindow{};
    auto options = raceengine::InputOptions{};
    options.vehicleRotationDegrees = 756.0;

    auto service = raceengine::InputService(*sink, *backend, window, options);

    std::cout << "\nWhat the game reads. Press each pedal in turn; ^C to stop.\n\n";

    while (true)
    {
        const auto asked = service.sample();

        std::cout << "\r  steering " << std::setw(7) << std::fixed << std::setprecision(3) << asked.steering
                  << "   throttle " << std::setw(6) << asked.throttle << "   brake " << std::setw(6) << asked.brake
                  << "   clutch " << std::setw(6) << asked.clutch << "   up " << (asked.upshift ? '1' : '0') << " down "
                  << (asked.downshift ? '1' : '0') << "    " << std::flush;

        std::this_thread::sleep_for(50ms);
    }
}

} // namespace

int main(const int argc, const char** argv)
{
    if (argc > 1 && std::string_view(argv[1]) == "--monitor")
    {
        return monitor();
    }

    auto backend = raceengine::createInputBackend(nullptr);
    if (!backend)
    {
        std::cout << "no input backend on this platform\n";

        return 1;
    }

    const auto devices = backend->enumerate();
    if (devices.empty())
    {
        std::cout << "nothing is plugged in that this can calibrate.\n"
                     "A Fanatec base only appears on the bus when it is powered on.\n";

        return 1;
    }

    std::cout << "Devices:\n";
    for (auto index = std::size_t{0}; index < devices.size(); index++)
    {
        std::cout << "  [" << index << "] " << devices[index].name << " at " << devices[index].address << "\n";
    }

    auto chosen = std::size_t{0};
    if (argc > 1)
    {
        chosen = static_cast<std::size_t>(std::atoi(argv[1]));
    }

    if (chosen >= devices.size())
    {
        std::cout << "there is no device " << chosen << "\n";

        return 1;
    }

    const auto& description = devices[chosen];

    const auto opened = backend->open(description.identity);
    if (!opened)
    {
        std::cout << "could not open it: " << opened.error() << "\n";

        return 1;
    }

    auto watcher = Watcher{.backend = *backend, .identity = description.identity};

    // The rest sample, and it is taken before anything is asked for: what the axes read with nobody
    // touching them is the one measurement that cannot be got later, because by then somebody has.
    for (auto step = 0; step < 400; step++)
    {
        watcher.pump();
    }

    auto rest = raceengine::DeviceSample{};
    for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
    {
        rest.axes[index] = watcher.axes[index].seen ? watcher.axes[index].highest : 0;
    }

    // The existing profile if there is one, seeded from the device only when there is not. A
    // calibration replaces what it successfully measures and leaves the rest alone — so a rig with no
    // clutch pedal keeps its clutch entry, and a session that goes wrong halfway costs only the steps
    // it got to.
    auto profile = raceengine::seedDeviceProfile(description, rest);

    if (const auto directory = raceengine::defaultProfileDirectory(); !directory.empty())
    {
        const auto existing =
            std::filesystem::path(directory) / raceengine::deviceProfileFileName(description.identity);

        if (auto file = std::ifstream(existing))
        {
            const auto text = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            if (auto parsed = raceengine::parseDeviceProfile(text))
            {
                profile = std::move(parsed).value();
                std::cout << "Starting from the profile already at " << existing.string() << ".\n";
            }
        }
    }

    std::cout << "\nCalibrating " << description.name << ".\n"
              << "Hands off everything to start with; each step says what to hold.\n\n";

    auto skipped = 0;

    const auto pedal = [&watcher, &profile, &skipped](const raceengine::InputAxis axis, const std::string_view name)
    {
        std::cout << name << ":\n";

        const auto resting =
            captureRange(watcher, axis, std::string("let the ").append(name).append(" all the way up"));
        const auto pressed =
            captureAxis(watcher, axis, std::string("press the ").append(name).append(" all the way down"), false);

        // **The released end is the resting extreme nearest the pressed end, not the far one**, and
        // that is not a rounding choice. A pedal at rest is not one number: a load cell drifts with
        // temperature and preload, and a throttle sits somewhere in the last few hundred counts of
        // its stop. Taking the far end makes every one of those resting values read as *applied* —
        // measured here at 1.8% throttle and 3.7% brake with both feet on the floor, which in the car
        // is a permanent light throttle and a permanent brake drag. Taking the near end makes the
        // whole resting spread clamp to zero, which is what "released" means.
        const auto released =
            std::abs(resting.lowest - pressed) < std::abs(resting.highest - pressed) ? resting.highest : resting.lowest;

        if (resting.seen && resting.highest != resting.lowest)
        {
            std::cout << "    at rest it wandered " << (resting.highest - resting.lowest) << " counts ("
                      << resting.lowest << ".." << resting.highest << "), so " << released << " is the released end\n";
        }

        const auto travel = std::abs(pressed - released);
        std::cout << "    released " << released << ", pressed " << pressed << " (" << travel << " counts of travel)";

        if (travel < usefulTravel)
        {
            // Left alone rather than written. A pedal that did not move is not a measurement of a
            // pedal that does not move — it is the absence of a measurement, and overwriting a good
            // calibration with it is how one bad session costs a rig its setup.
            std::cout << "\n    <- almost no travel, so this axis is left as it was. Either the pedal is not"
                         "\n       reaching the base, or the wheel dropped off the bus during that step.\n\n";
            skipped++;

            return;
        }

        auto& calibration = profile.axes[raceengine::axisIndex(axis)];

        // Released is `minimum` and pressed is `maximum` whichever way round the codes run: an
        // inverted axis is stated by the ends rather than by a flag, which is what lets a pedal that
        // counts down and one that counts up be the same arithmetic.
        calibration.minimum = static_cast<double>(released);
        calibration.maximum = static_cast<double>(pressed);
        calibration.centre = static_cast<double>(released);

        std::cout << "\n\n";
    };

    pedal(raceengine::InputAxis::Throttle, "throttle");
    pedal(raceengine::InputAxis::Brake, "brake");
    pedal(raceengine::InputAxis::Clutch, "clutch");

    std::cout << "steering:\n";
    const auto left = captureAxis(watcher, raceengine::InputAxis::Steering, "turn the wheel to full left lock", false);
    const auto right = captureAxis(watcher, raceengine::InputAxis::Steering, "turn the wheel to full right lock", true);
    const auto centre = captureResting(watcher, raceengine::InputAxis::Steering, "let the wheel come back to centre");

    std::cout << "    left " << left << ", centre " << centre << ", right " << right << "\n";

    const auto lock = std::abs(right - left);
    const auto steeringUsable =
        lock >= usefulTravel && centre > std::min(left, right) && centre < std::max(left, right);

    if (steeringUsable)
    {
        auto& calibration = profile.axes[raceengine::axisIndex(raceengine::InputAxis::Steering)];
        calibration.minimum = static_cast<double>(left);
        calibration.maximum = static_cast<double>(right);
        calibration.centre = static_cast<double>(centre);
    }
    else
    {
        std::cout << "    <- that is not a calibration. The wheel has to reach both locks and come back to the\n"
                     "       middle, and the middle has to be between them.\n";
    }

    std::cout << "\npaddles and buttons:\n";

    // Skipped means "leave it as it was", not "unbind it". A driver who has no handbrake button and
    // waits out that prompt must not lose the two paddles they have just bound, and one who is only
    // re-doing the pedals must not lose all three.
    const auto bind = [&watcher, &profile](const raceengine::DriverAction action, const std::string_view prompt)
    {
        const auto found = captureButton(watcher, prompt);
        if (found >= 0)
        {
            profile.buttons[static_cast<std::size_t>(action)] = found;
        }
    };

    bind(raceengine::DriverAction::Upshift, "pull the UPSHIFT paddle");
    bind(raceengine::DriverAction::Downshift, "pull the DOWNSHIFT paddle");
    bind(raceengine::DriverAction::Handbrake, "press the HANDBRAKE button");

    backend->close();

    if (!steeringUsable)
    {
        std::cout << "\nNothing written: without a steering calibration there is no profile worth keeping, and\n"
                     "overwriting the one already there would cost you a good one. Run this again.\n";

        return 1;
    }

    auto directory = raceengine::defaultProfileDirectory();
    if (directory.empty())
    {
        std::cout << "\nNeither XDG_CONFIG_HOME nor HOME is set, so there is nowhere to write this.\n";
        std::cout << raceengine::deviceProfileToText(profile);

        return 1;
    }

    const auto path = std::filesystem::path(directory) / raceengine::deviceProfileFileName(description.identity);

    auto created = std::error_code{};
    std::filesystem::create_directories(directory, created);

    auto file = std::ofstream(path);
    if (!file)
    {
        std::cout << "\ncould not write " << path.string() << "\n";

        return 1;
    }

    file << raceengine::deviceProfileToText(profile);
    file.close();

    std::cout << "\nWritten to " << path.string() << ".\n"
              << "The game reads it when the device is next attached, so a restart is not needed.\n";

    if (watcher.drops > 0)
    {
        std::cout << "\nThe wheel fell off the USB bus " << watcher.drops
                  << " time(s) while calibrating, so some of the above may have been measured against a device that\n"
                     "was not there. That is a physical link fault rather than anything the game does: check\n"
                     "`sudo dmesg` for `USB disconnect`, and try the base in a port on the machine itself rather\n"
                     "than through a hub. Then run this again.\n";
    }

    return 0;
}
