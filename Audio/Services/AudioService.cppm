module;

#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

export module raceengine.audio:AudioService;

import :AudioBackend;
import :CarAudio;
import :SoundBank;

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

    [[nodiscard]] bool active() const
    {
        return carLoaded;
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
