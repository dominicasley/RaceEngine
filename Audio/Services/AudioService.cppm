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

AudioService::AudioService(spdlog::logger& logger, AudioOptions options) :
    logger(logger),
    options(std::move(options))
{
    if (this->options.unattended)
    {
        backend = createSilentAudioBackend();

        return;
    }

    if (this->options.useFmod)
    {
        auto fmod = createFmodAudioBackend(this->options.capturePath);
        if (fmod)
        {
            backend = std::move(fmod).value();

            if (this->options.capturePath.empty())
            {
                logger.info("Audio ready through FMOD");
            }
            else
            {
                logger.info("Audio ready through FMOD, writing to {}", this->options.capturePath);
            }

            return;
        }

        // Named once, at info rather than warning: a machine without FMOD is a supported
        // configuration and not a fault, and the sentence says exactly what would change it.
        logger.info("Audio is silent: {}", fmod.error());
    }

    backend = createSilentAudioBackend();
}

AudioService::~AudioService()
{
    unloadCar();
}

std::expected<void, std::string> AudioService::loadCar(const std::filesystem::path& sfxDirectory, const double idleRpm,
                                                       const double limiterRpm)
{
    if (!backend)
    {
        return std::unexpected("no audio backend");
    }

    unloadCar();

    auto error = std::error_code{};
    if (!std::filesystem::is_directory(sfxDirectory, error))
    {
        return std::unexpected("no sfx folder at " + sfxDirectory.string());
    }

    const auto guidPath = sfxDirectory / "GUIDs.txt";

    auto guidFile = std::ifstream(guidPath);
    if (!guidFile)
    {
        return std::unexpected("no GUIDs.txt beside the bank, so no event can be addressed: " + guidPath.string());
    }

    const auto text = std::string(std::istreambuf_iterator<char>(guidFile), std::istreambuf_iterator<char>());

    // The bank is whichever `.bank` is there. A car folder holds one; naming it in the GUID map would
    // be a second place for the same fact to be stated and a second place for it to be wrong.
    auto bankPath = std::filesystem::path{};
    for (const auto& entry : std::filesystem::directory_iterator(sfxDirectory, error))
    {
        if (entry.path().extension() == ".bank")
        {
            bankPath = entry.path();
            break;
        }
    }

    if (bankPath.empty())
    {
        return std::unexpected("no .bank in " + sfxDirectory.string());
    }

    auto map = parseSoundBankMap(text, bankPath.string());
    if (!map)
    {
        return std::unexpected(guidPath.string() + ": " + map.error());
    }

    if (const auto loaded = backend->loadCar(map.value(), idleRpm, limiterRpm); !loaded)
    {
        return std::unexpected(loaded.error());
    }

    carLoaded = true;
    reportedUnmatched = false;

    // What was found, so a car missing its engine event says so here rather than by being quiet.
    auto missing = std::vector<std::string>{};
    for (auto index = std::size_t{0}; index < static_cast<std::size_t>(CarEvent::Count); index++)
    {
        const auto which = static_cast<CarEvent>(index);
        if (!map->has(which))
        {
            missing.emplace_back(carEventName(which));
        }
    }

    if (missing.empty())
    {
        logger.info("Audio loaded {} through {}: every event present", bankPath.filename().string(), platform());
    }
    else
    {
        logger.info("Audio loaded {} through {}: {} event(s) this bank does not carry ({})",
                    bankPath.filename().string(), platform(), missing.size(),
                    [&missing]
                    {
                        auto joined = std::string{};
                        for (const auto& name : missing)
                        {
                            joined += (joined.empty() ? "" : ", ") + name;
                        }

                        return joined;
                    }());
    }

    // Which recordings became layers — the engine's and the tyres'. Said at load because it is the
    // one fact about a car's sound that no amount of reading this engine can establish: a bank names
    // its own samples and a car whose layers did not classify is silent with nothing anywhere to say
    // why.
    const auto declared = backend->declaredParameters();
    if (!declared.empty())
    {
        auto joined = std::string{};
        for (const auto& name : declared)
        {
            joined += (joined.empty() ? "" : ", ") + name;
        }

        logger.info("Audio layers ({}): {}", declared.size(), joined);
    }

    return {};
}

void AudioService::update(const CarAudioState& state)
{
    if (!backend || !carLoaded)
    {
        return;
    }

    backend->update(state);

    // Once, and only if something was refused. A bank whose parameters are named differently from the
    // ones being written is the single most likely reason a correctly loaded car is still silent, and
    // it is invisible from anywhere else — so it is said, once, with the names.
    if (reportedUnmatched)
    {
        return;
    }

    const auto unmatched = backend->unmatchedParameters();
    if (unmatched.empty())
    {
        return;
    }

    reportedUnmatched = true;

    auto joined = std::string{};
    for (const auto& name : unmatched)
    {
        joined += (joined.empty() ? "" : ", ") + name;
    }

    logger.info("Audio: this bank did not take {} — its events name their parameters differently", joined);
}

void AudioService::unloadCar()
{
    if (backend && carLoaded)
    {
        backend->unloadCar();
    }

    carLoaded = false;
}

} // namespace raceengine
