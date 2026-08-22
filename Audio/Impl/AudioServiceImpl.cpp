// AudioService bodies. Declarations are in Audio/Services/AudioService.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
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

module raceengine.audio;

import :AudioService;
import :AudioBackend;
import :CarAudio;
import :SoundBank;

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
