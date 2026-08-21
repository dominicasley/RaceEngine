module;

#if defined(RACEENGINE_HAS_FMOD)
#include <fmod.hpp>
#include <fmod_errors.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// An implementation partition for the reason the Vulkan and evdev backends are: `fmod.hpp` would
// otherwise land in every importer's closure, and it drags a platform layer that has no business in a
// module that only wants to know a car's engine speed.
module raceengine.audio:FmodAudioBackend;

import :AudioBackend;
import :CarAudio;
import :EngineLayers;
import :SoundBank;
import :TyreLayers;

namespace raceengine
{

// **Core rather than Studio, and that is the whole design.**
//
// An Assetto Corsa bank is FMOD Studio *1.x* and FMOD 2.x will not load it: the bank names the effect
// `FMOD Distance Filter`, which was a DSP in 1.x and is only a channel property in 2.x, so
// `loadBankFile` answers `FMOD_ERR_PLUGIN_MISSING` — an error that names a plugin and so does not
// read as a version mismatch at all.
//
// What is *inside* the bank is a plain FSB5 of PCM16 samples, and Core opens one of those directly.
// So the event graph is given up — that is the mod author's mix and losing it is the real cost of
// this route — and the crossfade becomes ours, driven from the driveline that already knows what the
// engine is doing. `:EngineLayers` is that crossfade and is where the interesting part lives.
// Everything in this file is transport.

#if defined(RACEENGINE_HAS_FMOD)

namespace
{

// Where the samples start. A `.bank` is a Studio container with the FSB5 embedded in it, so the
// offset is found rather than known: the header ahead of it carries the event graph this route is
// giving up, and its length is a function of how much of it there is.
[[nodiscard]] std::size_t findFsb5(const std::string& blob)
{
    return blob.find("FSB5");
}

} // namespace

// **How much of the frame the engine is allowed to take, and it is a headroom figure rather than a
// taste one.**
//
// The layer weights are built to sum to one *in power*, which is what stops the crossfade dipping —
// but two uncorrelated loops at 0.707 each still add to 1.414 in *amplitude*, and these recordings
// are mastered near full scale. Measured on a launch capture the engine alone peaked at 0.90 of full
// scale, so the first tyre squeal or backfire laid over it clips, and a clipped mix is heard as the
// engine being broken rather than as the mix being full.
//
// Six decibels leaves the engine the loudest thing in the car and still leaves room for everything
// that has to happen on top of it. It belongs on a group rather than on the layer gains because the
// balance between the engine and the rest of the car is the next thing to be given a knob, and that
// knob is this one.
constexpr float engineHeadroom = 0.5f;

// The tyres' own group, and the same kind of number: the skid recording is mastered as hot as the
// engine loops, so it takes the same six decibels down. What the two groups leave is measured on the
// launch capture — wheelspin over a wide-open engine is this game's loudest ordinary moment — and it
// has to come through under full scale.
constexpr float tyreHeadroom = 0.5f;

class FmodAudioBackend final : public IAudioBackend
{
    FMOD::System* system = nullptr;
    FMOD::ChannelGroup* engineGroup = nullptr;
    FMOD::ChannelGroup* tyreGroup = nullptr;
    FMOD::Sound* bank = nullptr;

    // The bank's bytes, held for as long as the sound is open. `FMOD_OPENMEMORY` copies, but the
    // pointer must stay valid across the call and this is also what makes the FSB5 offset meaningful.
    std::string blob;

    std::vector<EngineLayer> layers;
    std::vector<LayerMix> mix;
    std::vector<FMOD::Channel*> channels;
    std::vector<float> baseFrequency;
    std::vector<TyreLayer> tyreLayers;
    std::vector<LayerMix> tyreMix;
    std::vector<FMOD::Channel*> tyreChannels;
    std::vector<float> tyreBaseFrequency;
    std::vector<std::string> declared;

    // Held rather than borrowed: the writer takes its filename as a bare pointer.
    std::string capture;

    double idle = 0.0;

public:
    FmodAudioBackend() = default;

    ~FmodAudioBackend() override
    {
        unloadCar();

        if (engineGroup != nullptr)
        {
            engineGroup->release();
            engineGroup = nullptr;
        }

        if (tyreGroup != nullptr)
        {
            tyreGroup->release();
            tyreGroup = nullptr;
        }

        if (system != nullptr)
        {
            system->release();
        }
    }

    [[nodiscard]] std::expected<void, std::string> start(std::string capturePath)
    {
        capture = std::move(capturePath);

        if (const auto result = FMOD::System_Create(&system); result != FMOD_OK)
        {
            return std::unexpected(std::string("FMOD would not create a system: ") + FMOD_ErrorString(result));
        }

        // **The audio side of `RACEENGINE_DUMP_FRAME`, and it exists for the same reason.**
        //
        // Everything above this line can be reasoned about and unit-tested without a device — that is
        // what `:EngineLayers` is for. What cannot is whether any of it reaches a speaker, and the
        // only instrument for that is a person with ears, who is not always here. The writer turns
        // "the layers classified" into a file of the noise the car actually made.
        //
        // **Non-realtime**, not the plain writer: NRT mixes one block per `update` instead of
        // following the clock, so a capture run — whose tick count is a function of frame number
        // rather than of wall time — writes the same bytes on a busy machine as on an idle one. The
        // realtime writer would record however much of the run the mixer happened to get to.
        if (!capture.empty())
        {
            if (const auto result = system->setOutput(FMOD_OUTPUTTYPE_WAVWRITER_NRT); result != FMOD_OK)
            {
                return std::unexpected(std::string("FMOD would not write to a file: ") + FMOD_ErrorString(result));
            }
        }

        // A channel per engine layer with room for the one-shots that will follow. The writer takes
        // its filename as the driver data, which is why the path is held rather than borrowed.
        void* driverData = capture.empty() ? nullptr : static_cast<void*>(capture.data());

        if (const auto result = system->init(64, FMOD_INIT_NORMAL, driverData); result != FMOD_OK)
        {
            return std::unexpected(std::string("FMOD would not initialise: ") + FMOD_ErrorString(result));
        }

        if (const auto result = system->createChannelGroup("engine", &engineGroup); result != FMOD_OK)
        {
            return std::unexpected(std::string("FMOD would not create the engine group: ") + FMOD_ErrorString(result));
        }

        engineGroup->setVolume(engineHeadroom);

        if (const auto result = system->createChannelGroup("tyres", &tyreGroup); result != FMOD_OK)
        {
            return std::unexpected(std::string("FMOD would not create the tyre group: ") + FMOD_ErrorString(result));
        }

        tyreGroup->setVolume(tyreHeadroom);

        return {};
    }

    [[nodiscard]] std::string_view platform() const override
    {
        return "fmod-core";
    }

    [[nodiscard]] std::expected<void, std::string> loadCar(const SoundBankMap& map, const double idleRpm,
                                                           const double limiterRpm) override
    {
        unloadCar();

        if (system == nullptr)
        {
            return std::unexpected("no FMOD system");
        }

        idle = idleRpm;

        auto file = std::ifstream(map.bankPath, std::ios::binary);
        if (!file)
        {
            return std::unexpected("could not read " + map.bankPath);
        }

        blob.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

        const auto at = findFsb5(blob);
        if (at == std::string::npos)
        {
            return std::unexpected(map.bankPath + " has no FSB5 in it — is it an FMOD bank?");
        }

        auto info = FMOD_CREATESOUNDEXINFO{};
        info.cbsize = sizeof(info);
        info.length = static_cast<unsigned int>(blob.size() - at);

        const auto mode = FMOD_OPENMEMORY | FMOD_LOOP_NORMAL | FMOD_CREATESAMPLE | FMOD_2D;

        if (const auto result = system->createSound(blob.data() + at, mode, &info, &bank); result != FMOD_OK)
        {
            return std::unexpected("FMOD would not open the sound bank inside " + map.bankPath + ": " +
                                   FMOD_ErrorString(result));
        }

        auto count = 0;
        bank->getNumSubSounds(&count);

        if (count <= 0)
        {
            return std::unexpected("the bank inside " + map.bankPath + " holds no samples");
        }

        // The FSB5 carries its own name table, and that is what makes the layers classifiable at all:
        // a bank of anonymous samples could be played but never mixed, because nothing would say which
        // recording was the high one.
        auto names = std::vector<std::string>{};
        names.reserve(static_cast<std::size_t>(count));

        for (auto index = 0; index < count; index++)
        {
            FMOD::Sound* sub = nullptr;
            if (bank->getSubSound(index, &sub) != FMOD_OK || sub == nullptr)
            {
                names.emplace_back();
                continue;
            }

            auto buffer = std::array<char, 256>{};
            sub->getName(buffer.data(), static_cast<int>(buffer.size()));
            names.emplace_back(buffer.data());
        }

        layers = classifyEngineLayers(names, idleRpm, limiterRpm);

        if (layers.empty())
        {
            return std::unexpected("none of the " + std::to_string(count) + " samples in " + map.bankPath +
                                   " is named like an engine layer");
        }

        mix.assign(layers.size(), LayerMix{});
        channels.assign(layers.size(), nullptr);
        baseFrequency.assign(layers.size(), 0.0f);
        declared.clear();

        // Every layer starts now, looping, silent. Started once and never restarted: a loop restarted
        // mid-note is a click, and the whole point of mixing by gain is that the listener can move
        // between inside and outside the car without any of them stopping.
        for (auto index = std::size_t{0}; index < layers.size(); index++)
        {
            FMOD::Sound* sub = nullptr;
            if (bank->getSubSound(layers[index].sample, &sub) != FMOD_OK || sub == nullptr)
            {
                continue;
            }

            sub->setMode(FMOD_LOOP_NORMAL);

            if (system->playSound(sub, engineGroup, true, &channels[index]) != FMOD_OK || channels[index] == nullptr)
            {
                continue;
            }

            channels[index]->getFrequency(&baseFrequency[index]);
            channels[index]->setVolume(0.0f);
            channels[index]->setPaused(false);

            declared.push_back(layers[index].name);
        }

        // The tyres, by the same rule. None is not an error the way no engine is: a bank without
        // them is a quieter car, not a broken one.
        tyreLayers = classifyTyreLayers(names);
        tyreMix.assign(tyreLayers.size(), LayerMix{});
        tyreChannels.assign(tyreLayers.size(), nullptr);
        tyreBaseFrequency.assign(tyreLayers.size(), 0.0f);

        for (auto index = std::size_t{0}; index < tyreLayers.size(); index++)
        {
            FMOD::Sound* sub = nullptr;
            if (bank->getSubSound(tyreLayers[index].sample, &sub) != FMOD_OK || sub == nullptr)
            {
                continue;
            }

            sub->setMode(FMOD_LOOP_NORMAL);

            if (system->playSound(sub, tyreGroup, true, &tyreChannels[index]) != FMOD_OK ||
                tyreChannels[index] == nullptr)
            {
                continue;
            }

            tyreChannels[index]->getFrequency(&tyreBaseFrequency[index]);
            tyreChannels[index]->setVolume(0.0f);
            tyreChannels[index]->setPaused(false);

            declared.push_back(tyreLayers[index].name);
        }

        return {};
    }

    void update(const CarAudioState& state) override
    {
        if (system == nullptr || layers.empty())
        {
            return;
        }

        // Which side of the glass the listener is on. Not driven by the camera yet: the cockpit
        // camera exists and nothing tells this service which one is live, so it is stated here rather
        // than guessed, and it is one line to wire when the scene says.
        mixEngineLayers(layers, state, LayerPosition::Exterior, idle > 0.0 ? idle : state.idleRpm, mix);

        for (auto index = std::size_t{0}; index < layers.size(); index++)
        {
            if (channels[index] == nullptr)
            {
                continue;
            }

            channels[index]->setVolume(static_cast<float>(mix[index].gain));

            if (baseFrequency[index] > 0.0f)
            {
                channels[index]->setFrequency(baseFrequency[index] * static_cast<float>(mix[index].pitch));
            }
        }

        mixTyreLayers(tyreLayers, state, LayerPosition::Exterior, tyreMix);

        for (auto index = std::size_t{0}; index < tyreLayers.size(); index++)
        {
            if (tyreChannels[index] == nullptr)
            {
                continue;
            }

            tyreChannels[index]->setVolume(static_cast<float>(tyreMix[index].gain));

            if (tyreBaseFrequency[index] > 0.0f)
            {
                tyreChannels[index]->setFrequency(tyreBaseFrequency[index] * static_cast<float>(tyreMix[index].pitch));
            }
        }

        system->update();
    }

    void unloadCar() override
    {
        for (auto* channel : channels)
        {
            if (channel != nullptr)
            {
                channel->stop();
            }
        }

        for (auto* channel : tyreChannels)
        {
            if (channel != nullptr)
            {
                channel->stop();
            }
        }

        channels.clear();
        baseFrequency.clear();
        layers.clear();
        mix.clear();
        tyreChannels.clear();
        tyreBaseFrequency.clear();
        tyreLayers.clear();
        tyreMix.clear();

        if (bank != nullptr)
        {
            bank->release();
            bank = nullptr;
        }

        if (system != nullptr)
        {
            system->update();
        }

        blob.clear();
    }

    [[nodiscard]] std::vector<std::string> unmatchedParameters() const override
    {
        // Nothing to mismatch: this route writes gains onto channels it opened itself rather than
        // parameters onto events somebody else authored.
        return {};
    }

    [[nodiscard]] std::vector<std::string> declaredParameters() const override
    {
        return declared;
    }
};

std::expected<std::unique_ptr<IAudioBackend>, std::string> createFmodAudioBackend(std::string capturePath)
{
    auto backend = std::make_unique<FmodAudioBackend>();

    if (const auto started = backend->start(std::move(capturePath)); !started)
    {
        return std::unexpected(started.error());
    }

    return backend;
}

#else

// **Not absent, and that is the point.** The file compiles either way and the factory answers either
// way — with a sentence naming what is missing rather than with a link error or a target that
// silently is not built. A game asking for FMOD on a machine without it gets told so and falls back
// to silence, which is what the DirectInput backend does off Windows for exactly the same reason.
std::expected<std::unique_ptr<IAudioBackend>, std::string> createFmodAudioBackend(std::string)
{
    return std::unexpected("this build has no FMOD: configure with -DRACEENGINE_WITH_FMOD=ON and "
                           "-DFMOD_ROOT=<the FMOD Engine folder>");
}

#endif

} // namespace raceengine
