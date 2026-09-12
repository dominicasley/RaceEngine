module;

#include <cstddef>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

export module raceengine.audio:AudioService;

import :AudioBackend;
import :CarAudio;
import :SoundBank;
import :TrafficAudio;

namespace raceengine
{

export struct AudioOptions
{
    // No device is opened and no bank is loaded. The rule every other device path here keeps: an
    // unattended run owns no speakers, and it is what makes a capture byte-identical with and without
    // any of this.
    bool unattended = false;

    // Ask for FMOD. False takes the silent backend without trying, which is what a machine with no
    // sound card wants; true tries FMOD and *says* when it cannot, rather than going quiet.
    bool useFmod = true;

    // Where the mix goes. Empty is the sound card; a path is a wav file and no device is opened at
    // all — which is what lets a run on a machine nobody is sitting at still be listened to.
    std::string capturePath;
};

// One body shape of the traffic fleet, as a game states it: a folder of the recordings the exporter
// wrote out of the car's bank, and the two engine speeds nothing in that folder states. A string
// rather than a path so a game unit can state one without `<filesystem>` in its fragment.
export struct TrafficFleetCar
{
    std::string name;
    std::string audioDirectory;
    double idleRpm = 800.0;
    double limiterRpm = 6500.0;
};

// What a game holds. One backend, one car's bank at a time, and a state written once per tick.
//
// It owns no thread. FMOD has its own and its `update` must be called from one thread only — this
// one — so the whole of the threading design is "call it where the car is stepped", which is also the
// only place the state exists.
export class AudioService
{
    spdlog::logger& logger;
    AudioOptions options;

    std::unique_ptr<IAudioBackend> backend;
    bool carLoaded = false;
    bool fleetLoaded = false;
    bool reportedUnmatched = false;

public:
    AudioService(spdlog::logger& logger, AudioOptions options);

    AudioService(const AudioService&) = delete;
    AudioService& operator=(const AudioService&) = delete;
    AudioService(AudioService&&) = delete;
    AudioService& operator=(AudioService&&) = delete;

    ~AudioService();

    // A car folder's `sfx`: the bank beside its GUID map. Fallible and says why — a car that makes no
    // sound is a thing somebody will spend an evening on, and every reason it can happen is nameable.
    [[nodiscard]] std::expected<void, std::string> loadCar(const std::filesystem::path& sfxDirectory, double idleRpm,
                                                           double limiterRpm);

    // Once per tick, from whoever stepped the car.
    void update(const CarAudioState& state);

    void unloadCar();

    // The traffic fleet's recordings, one folder per body shape **in the order the caller indexes
    // bodies by** — a car with no folder still takes its place in the order, silently. Fallible for
    // the reason `loadCar` is; a fleet with no recordings at all is a refusal and not a silence.
    [[nodiscard]] std::expected<void, std::string> loadTrafficFleet(std::span<const TrafficFleetCar> fleet,
                                                                    std::size_t voices);

    // Once per tick, from whoever knows where the ear and the cars are.
    void updateTraffic(const AudioListener& listener, std::span<const TrafficVoice> voices);

    void unloadTrafficFleet();

    // The patrol cars' siren, one file, after the fleet. A fleet with no siren is a fleet whose
    // police chase silently, said once.
    [[nodiscard]] std::expected<void, std::string> loadSiren(const std::filesystem::path& file);

    [[nodiscard]] bool active() const
    {
        return carLoaded;
    }

    [[nodiscard]] bool trafficActive() const
    {
        return fleetLoaded;
    }

    [[nodiscard]] std::string_view platform() const
    {
        return backend ? backend->platform() : "none";
    }
};

} // namespace raceengine

namespace raceengine
{

} // namespace raceengine
