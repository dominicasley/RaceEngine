#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>

import raceengine;

// The real device, through the real backend, on whatever machine this is.
//
// Hidden — the leading dot in the tag keeps it out of the default run and therefore out of ctest —
// because its result depends on what is plugged in, and a suite whose count changes when somebody
// switches a wheel off is a suite nobody trusts. Run it deliberately:
//
//     ./EngineTests "[device]" --success
//
// It asserts nothing about what it finds. It is a measurement, and the point of it is that the
// numbers a calibration is seeded from can be read off the hardware rather than assumed.
TEST_CASE("what this machine's input devices actually report", "[.device][input]")
{
    const auto backend = raceengine::createInputBackend(nullptr);

    std::printf("backend: %s\n", std::string(backend->platform()).c_str());

    const auto devices = backend->enumerate();
    std::printf("devices: %zu\n", devices.size());

    for (const auto& device : devices)
    {
        std::printf("  %04x:%04x  %s  at %s  %.0f degrees lock to lock\n", device.identity.vendor,
                    device.identity.product, device.name.c_str(), device.address.c_str(), device.rotationDegrees);

        for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
        {
            const auto& axis = device.axes[index];
            std::printf("    %-9s %s %d..%d\n", raceengine::inputAxisName(static_cast<raceengine::InputAxis>(index)),
                        axis.present ? "present" : "absent ", axis.minimum, axis.maximum);
        }

        for (auto index = std::size_t{0}; index < static_cast<std::size_t>(raceengine::DeviceCapability::Count);
             index++)
        {
            const auto capability = static_cast<raceengine::DeviceCapability>(index);
            std::printf("    %-28s %s\n", raceengine::deviceCapabilityName(capability),
                        device.capabilities.has(capability) ? "yes" : "no");
        }
    }

    if (devices.empty())
    {
        std::printf("nothing to open, which is a perfectly ordinary answer\n");

        return;
    }

    const auto opened = backend->open(devices.front().identity);
    REQUIRE(opened);

    const auto rate = backend->updateRate();
    std::printf("rate: input %.0f Hz, output %.0f Hz, %s\n", rate.inputHz, rate.outputHz,
                rate.measured ? "measured" : "the transport's ceiling, nothing having been read yet");

    // Two seconds of whatever the device is doing. Move the wheel and the pedals during it and the
    // extremes below are the calibration; leave it alone and they are the resting position, which
    // is what a seeded profile takes as centre and as released.
    auto lowest = std::array<std::int32_t, raceengine::inputAxisCount>{};
    auto highest = std::array<std::int32_t, raceengine::inputAxisCount>{};
    auto first = true;
    auto buttons = std::uint64_t{0};

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (std::chrono::steady_clock::now() < deadline)
    {
        const auto sample = backend->read(std::chrono::milliseconds{4});
        REQUIRE(sample);

        for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
        {
            const auto value = sample->axes[index];
            lowest[index] = first ? value : std::min(lowest[index], value);
            highest[index] = first ? value : std::max(highest[index], value);
        }

        buttons |= sample->buttons;
        first = false;
    }

    for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
    {
        std::printf("seen: %-9s %d..%d\n", raceengine::inputAxisName(static_cast<raceengine::InputAxis>(index)),
                    lowest[index], highest[index]);
    }

    std::printf("buttons seen: %016llx\n", static_cast<unsigned long long>(buttons));

    const auto measured = backend->updateRate();
    std::printf("rate after reading: input %.0f Hz, %s\n", measured.inputHz,
                measured.measured ? "measured" : "still the ceiling, the device having said nothing");
}
