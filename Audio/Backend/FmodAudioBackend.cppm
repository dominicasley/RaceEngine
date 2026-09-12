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
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

// An implementation partition for the reason the Vulkan and evdev backends are: `fmod.hpp` would
// otherwise land in every importer's closure, and it drags a platform layer that has no business in a
// module that only wants to know a car's engine speed.
module raceengine.audio:FmodAudioBackend;

import :AudioBackend;
import :CarAudio;
import :EngineLayers;
import :SoundBank;
import :TrafficAudio;
import :TyreLayers;

namespace raceengine {
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

    namespace {
        // Where the samples start. A `.bank` is a Studio container with the FSB5 embedded in it, so the
        // offset is found rather than known: the header ahead of it carries the event graph this route is
        // giving up, and its length is a function of how much of it there is.
        [[nodiscard]] std::size_t findFsb5(const std::string &blob) {
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
    constexpr float engineHeadroom = 0.25f;

    // The tyres' own group, and the same kind of number: the skid recording is mastered as hot as the
    // engine loops, so it takes the same six decibels down. What the two groups leave is measured on the
    // launch capture — wheelspin over a wide-open engine is this game's loudest ordinary moment — and it
    // has to come through under full scale.
    constexpr float tyreHeadroom = 0.35f;

    // The other cars' group, and the same kind of number again: a fleet bank's loops are mastered as
    // hot as the Golf's, and a car passing at arm's length is at its full level on top of the player's
    // own engine. Six decibels down is the same headroom the engine took; the balance between the
    // street and the car is the knob this is.
    constexpr float trafficHeadroom = 0.5f;

    // Where a traffic car's level stops rising as it comes closer, and where it stops falling as it
    // goes away, metres. Inverse rolloff between the two — halving with every doubling of distance,
    // which is what a point source does in the open — so a car at fifty metres is twenty-two decibels
    // under one at four. Sound-designer numbers, meant to be moved by ear.
    constexpr float trafficNearMetres = 4.0f;
    constexpr float trafficFarMetres = 600.0f;

    // The siren is heard from much further than an engine, and it is loud: the inverse rolloff runs
    // between these, and the loop sits at this fraction of the traffic group's level.
    constexpr float sirenNearMetres = 6.0f;
    constexpr float sirenFarMetres = 2500.0f;
    constexpr float sirenLevel = 0.9f;

    // How many channels are mixed for real and how many may exist. The Golf is two dozen loops; eight
    // traffic voices on the fleet's largest bank are another hundred and forty, most of them at zero
    // gain — FMOD's virtual voices play the loudest of them and park the rest, which is what makes a
    // voice per layer affordable at all.
    constexpr int mixedChannels = 128;
    constexpr int virtualChannels = 512;

    [[nodiscard]] FMOD_VECTOR toFmod(const glm::dvec3 &vector) {
        return FMOD_VECTOR{static_cast<float>(vector.x), static_cast<float>(vector.y), static_cast<float>(vector.z)};
    }

    class FmodAudioBackend final : public IAudioBackend {
        FMOD::System *system = nullptr;
        FMOD::ChannelGroup *engineGroup = nullptr;
        FMOD::ChannelGroup *tyreGroup = nullptr;
        FMOD::ChannelGroup *trafficGroup = nullptr;
        FMOD::Sound *bank = nullptr;

        // The bank's bytes, held for as long as the sound is open. `FMOD_OPENMEMORY` copies, but the
        // pointer must stay valid across the call and this is also what makes the FSB5 offset meaningful.
        std::string blob;

        std::vector<EngineLayer> layers;
        std::vector<LayerMix> mix;
        std::vector<FMOD::Channel *> channels;
        std::vector<float> baseFrequency;
        std::vector<TyreLayer> tyreLayers;
        std::vector<LayerMix> tyreMix;
        std::vector<FMOD::Channel *> tyreChannels;
        std::vector<float> tyreBaseFrequency;
        std::vector<std::string> declared;

        // Held rather than borrowed: the writer takes its filename as a bare pointer.
        std::string capture;

        double idle = 0.0;

        // One body shape of the fleet: the loops it classified to, opened as 3D samples, and what each
        // was recorded at. Indexed by `TrafficVoice::body`, in the order the fleet was handed over, so a
        // bank with nothing in it still holds its place.
        struct FleetBank {
            std::string name;
            double idleRpm = 0.0;
            std::vector<EngineLayer> layers;
            std::vector<FMOD::Sound *> layerSounds = {};
            std::vector<float> layerBaseFrequency = {};
            std::vector<TyreLayer> tyres;
            std::vector<FMOD::Sound *> tyreSounds = {};
            std::vector<float> tyreBaseFrequency = {};
        };

        // One audible car: the channels it is playing on, and which car and bank they were started for,
        // so a slot the allocator hands to a different car is restarted on that car's bank.
        struct VoiceChannels {
            std::uint32_t car = noTrafficCar;
            std::uint8_t body = 0;
            std::vector<FMOD::Channel *> engine;
            std::vector<FMOD::Channel *> tyre;
            // The siren loop, started when the voice's car switches it on and stopped when it goes off.
            FMOD::Channel *siren = nullptr;
        };

        std::vector<FleetBank> fleet;
        std::vector<VoiceChannels> voices;
        // The one siren recording, for every patrol car, or null for a fleet that stated none.
        FMOD::Sound *sirenSound = nullptr;
        std::vector<LayerMix> voiceMix;
        std::vector<LayerMix> voiceTyreMix;

    public:
        FmodAudioBackend() = default;

        ~FmodAudioBackend() override {
            unloadTrafficFleet();
            unloadCar();

            if (engineGroup != nullptr) {
                engineGroup->release();
                engineGroup = nullptr;
            }

            if (tyreGroup != nullptr) {
                tyreGroup->release();
                tyreGroup = nullptr;
            }

            if (trafficGroup != nullptr) {
                trafficGroup->release();
                trafficGroup = nullptr;
            }

            if (system != nullptr) {
                system->release();
            }
        }

        [[nodiscard]] std::expected<void, std::string> start(std::string capturePath) {
            capture = std::move(capturePath);

            if (const auto result = FMOD::System_Create(&system); result != FMOD_OK) {
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
            if (!capture.empty()) {
                if (const auto result = system->setOutput(FMOD_OUTPUTTYPE_WAVWRITER_NRT); result != FMOD_OK) {
                    return std::unexpected(std::string("FMOD would not write to a file: ") + FMOD_ErrorString(result));
                }
            }

            // The writer takes its filename as the driver data, which is why the path is held rather
            // than borrowed.
            void *driverData = capture.empty() ? nullptr : static_cast<void *>(capture.data());

            system->setSoftwareChannels(mixedChannels);

            // **Right-handed, and it is a statement about which ear a car on the car's left lands in.**
            // The world is `+x` the car's left, `+y` up, `+z` forward (docs/vehicle-physics.md, *The
            // frame is left-handed and +x is the car's left*): screen-right is `cross(forward, up)` =
            // `cross(+z, +y)` = `-x`. FMOD's default derives the listener's right as `cross(up,
            // forward)`, which is `+x` here — the car's left — and would put a car passing on the left
            // in the right speaker. The right-handed flag derives it as `cross(forward, up)` instead,
            // which is the renderer's own rule, so the two agree about which side the street is on.
            if (const auto result = system->init(virtualChannels, FMOD_INIT_NORMAL | FMOD_INIT_3D_RIGHTHANDED,
                                                 driverData);
                result != FMOD_OK) {
                return std::unexpected(std::string("FMOD would not initialise: ") + FMOD_ErrorString(result));
            }

            if (const auto result = system->createChannelGroup("engine", &engineGroup); result != FMOD_OK) {
                return std::unexpected(
                    std::string("FMOD would not create the engine group: ") + FMOD_ErrorString(result));
            }

            engineGroup->setVolume(engineHeadroom);

            if (const auto result = system->createChannelGroup("tyres", &tyreGroup); result != FMOD_OK) {
                return std::unexpected(
                    std::string("FMOD would not create the tyre group: ") + FMOD_ErrorString(result));
            }

            tyreGroup->setVolume(tyreHeadroom);

            if (const auto result = system->createChannelGroup("traffic", &trafficGroup); result != FMOD_OK) {
                return std::unexpected(
                    std::string("FMOD would not create the traffic group: ") + FMOD_ErrorString(result));
            }

            trafficGroup->setVolume(trafficHeadroom);

            return {};
        }

        [[nodiscard]] std::string_view platform() const override {
            return "fmod-core";
        }

        [[nodiscard]] std::expected<void, std::string> loadCar(const SoundBankMap &map, const double idleRpm,
                                                               const double limiterRpm) override {
            unloadCar();

            if (system == nullptr) {
                return std::unexpected("no FMOD system");
            }

            idle = idleRpm;

            auto file = std::ifstream(map.bankPath, std::ios::binary);
            if (!file) {
                return std::unexpected("could not read " + map.bankPath);
            }

            blob.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

            const auto at = findFsb5(blob);
            if (at == std::string::npos) {
                return std::unexpected(map.bankPath + " has no FSB5 in it — is it an FMOD bank?");
            }

            auto info = FMOD_CREATESOUNDEXINFO{};
            info.cbsize = sizeof(info);
            info.length = static_cast<unsigned int>(blob.size() - at);

            const auto mode = FMOD_OPENMEMORY | FMOD_LOOP_NORMAL | FMOD_CREATESAMPLE | FMOD_2D;

            if (const auto result = system->createSound(blob.data() + at, mode, &info, &bank); result != FMOD_OK) {
                return std::unexpected("FMOD would not open the sound bank inside " + map.bankPath + ": " +
                                       FMOD_ErrorString(result));
            }

            auto count = 0;
            bank->getNumSubSounds(&count);

            if (count <= 0) {
                return std::unexpected("the bank inside " + map.bankPath + " holds no samples");
            }

            // The FSB5 carries its own name table, and that is what makes the layers classifiable at all:
            // a bank of anonymous samples could be played but never mixed, because nothing would say which
            // recording was the high one.
            auto names = std::vector<std::string>{};
            names.reserve(static_cast<std::size_t>(count));

            for (auto index = 0; index < count; index++) {
                FMOD::Sound *sub = nullptr;
                if (bank->getSubSound(index, &sub) != FMOD_OK || sub == nullptr) {
                    names.emplace_back();
                    continue;
                }

                auto buffer = std::array<char, 256>{};
                sub->getName(buffer.data(), static_cast<int>(buffer.size()));
                names.emplace_back(buffer.data());
            }

            layers = classifyEngineLayers(names, idleRpm, limiterRpm);

            if (layers.empty()) {
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
            for (auto index = std::size_t{0}; index < layers.size(); index++) {
                FMOD::Sound *sub = nullptr;
                if (bank->getSubSound(layers[index].sample, &sub) != FMOD_OK || sub == nullptr) {
                    continue;
                }

                sub->setMode(FMOD_LOOP_NORMAL);

                if (system->playSound(sub, engineGroup, true, &channels[index]) != FMOD_OK || channels[index] ==
                    nullptr) {
                    continue;
                }

                channels[index]->getFrequency(&baseFrequency[index]);
                channels[index]->setVolume(0.0f);
                // The player's own car is never the one parked when the street is busy: the traffic
                // voices are what the virtual voice system gets to choose between.
                channels[index]->setPriority(0);
                channels[index]->setPaused(false);

                declared.push_back(layers[index].name);
            }

            // The tyres, by the same rule. None is not an error the way no engine is: a bank without
            // them is a quieter car, not a broken one.
            tyreLayers = classifyTyreLayers(names);
            tyreMix.assign(tyreLayers.size(), LayerMix{});
            tyreChannels.assign(tyreLayers.size(), nullptr);
            tyreBaseFrequency.assign(tyreLayers.size(), 0.0f);

            for (auto index = std::size_t{0}; index < tyreLayers.size(); index++) {
                FMOD::Sound *sub = nullptr;
                if (bank->getSubSound(tyreLayers[index].sample, &sub) != FMOD_OK || sub == nullptr) {
                    continue;
                }

                sub->setMode(FMOD_LOOP_NORMAL);

                if (system->playSound(sub, tyreGroup, true, &tyreChannels[index]) != FMOD_OK ||
                    tyreChannels[index] == nullptr) {
                    continue;
                }

                tyreChannels[index]->getFrequency(&tyreBaseFrequency[index]);
                tyreChannels[index]->setVolume(0.0f);
                tyreChannels[index]->setPriority(0);
                tyreChannels[index]->setPaused(false);

                declared.push_back(tyreLayers[index].name);
            }

            return {};
        }

        void update(const CarAudioState &state) override {
            if (system == nullptr) {
                return;
            }

            // FMOD's own tick is at the end of this call whether or not the player's car has a bank,
            // because the traffic writes its channels between two of these and a city whose player
            // car is silent still has cars in it.
            if (layers.empty()) {
                system->update();

                return;
            }

            // Which side of the glass the listener is on. Not driven by the camera yet: the cockpit
            // camera exists and nothing tells this service which one is live, so it is stated here rather
            // than guessed, and it is one line to wire when the scene says.
            mixEngineLayers(layers, state, LayerPosition::Exterior, idle > 0.0 ? idle : state.idleRpm, mix);

            for (auto index = std::size_t{0}; index < layers.size(); index++) {
                if (channels[index] == nullptr) {
                    continue;
                }

                channels[index]->setVolume(static_cast<float>(mix[index].gain));

                if (baseFrequency[index] > 0.0f) {
                    channels[index]->setFrequency(baseFrequency[index] * static_cast<float>(mix[index].pitch));
                }
            }

            mixTyreLayers(tyreLayers, state, LayerPosition::Exterior, tyreMix);

            for (auto index = std::size_t{0}; index < tyreLayers.size(); index++) {
                if (tyreChannels[index] == nullptr) {
                    continue;
                }

                tyreChannels[index]->setVolume(static_cast<float>(tyreMix[index].gain));

                if (tyreBaseFrequency[index] > 0.0f) {
                    tyreChannels[index]->setFrequency(
                        tyreBaseFrequency[index] * static_cast<float>(tyreMix[index].pitch));
                }
            }

            system->update();
        }

        void unloadCar() override {
            for (auto *channel: channels) {
                if (channel != nullptr) {
                    channel->stop();
                }
            }

            for (auto *channel: tyreChannels) {
                if (channel != nullptr) {
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

            if (bank != nullptr) {
                bank->release();
                bank = nullptr;
            }

            if (system != nullptr) {
                system->update();
            }

            blob.clear();
        }

        [[nodiscard]] std::vector<std::string> unmatchedParameters() const override {
            // Nothing to mismatch: this route writes gains onto channels it opened itself rather than
            // parameters onto events somebody else authored.
            return {};
        }

        [[nodiscard]] std::vector<std::string> declaredParameters() const override {
            return declared;
        }

        // **A wav per loop rather than the car's bank, and it is a memory decision.** The Golf's route
        // decodes a whole bank into memory — sixty recordings, a hundred megabytes for a fleet of five
        // — for the two dozen it plays. The exporter has already written every recording out on its
        // own, so a fleet car opens only the loops that classified: a dozen files, a few megabytes.
        // What it gives up is the name table, and the file name carries the name.
        [[nodiscard]] std::expected<std::vector<std::string>, std::string>
        loadTrafficFleet(const std::span<const TrafficBank> banks, const std::size_t voiceCount) override {
            unloadTrafficFleet();

            if (system == nullptr) {
                return std::unexpected("no FMOD system");
            }

            auto report = std::vector<std::string>{};
            report.reserve(banks.size());

            for (const auto &traffic_bank: banks) {
                auto names = std::vector<std::string>{};
                names.reserve(traffic_bank.samples.size());

                for (const auto &sample: traffic_bank.samples) {
                    names.push_back(sample.name);
                }

                // 3D, world relative, inverse rolloff between the two distances; a sample in memory
                // rather than a stream, because eight voices may play the same loop at once and a
                // stream plays once. Looping, like every layer the mixers drive.
                const auto mode = FMOD_3D | FMOD_3D_WORLDRELATIVE | FMOD_3D_INVERSEROLLOFF | FMOD_LOOP_NORMAL |
                                  FMOD_CREATESAMPLE;

                const auto open = [&](const int sample, std::vector<FMOD::Sound *> &sounds,
                                      std::vector<float> &frequencies) -> std::expected<void, std::string> {
                    const auto &path = traffic_bank.samples[static_cast<std::size_t>(sample)].path;

                    FMOD::Sound *sound = nullptr;
                    if (const auto result = system->createSound(path.c_str(), mode, nullptr, &sound);
                        result != FMOD_OK) {
                        return std::unexpected("FMOD would not open " + path + ": " + FMOD_ErrorString(result));
                    }

                    sound->set3DMinMaxDistance(trafficNearMetres, trafficFarMetres);

                    auto frequency = 0.0f;
                    auto priority = 0;
                    sound->getDefaults(&frequency, &priority);

                    sounds.push_back(sound);
                    frequencies.push_back(frequency);

                    return {};
                };

                auto entry = FleetBank{
                    .name = traffic_bank.name, .idleRpm = traffic_bank.idleRpm,
                    .layers = classifyTrafficEngineLayers(names, traffic_bank.idleRpm, traffic_bank.limiterRpm),
                    .tyres = classifyTrafficTyreLayers(names),
                };

                for (const auto &layer: entry.layers) {
                    if (const auto opened = open(layer.sample, entry.layerSounds, entry.layerBaseFrequency); !opened) {
                        fleet.push_back(std::move(entry));
                        unloadTrafficFleet();

                        return std::unexpected(opened.error());
                    }
                }

                for (const auto &tyre: entry.tyres) {
                    if (const auto opened = open(tyre.sample, entry.tyreSounds, entry.tyreBaseFrequency); !opened) {
                        fleet.push_back(std::move(entry));
                        unloadTrafficFleet();

                        return std::unexpected(opened.error());
                    }
                }

                auto joined = std::string{};
                for (const auto &layer: entry.layers) {
                    joined += (joined.empty() ? "" : ", ") + layer.name;
                }

                report.push_back(traffic_bank.name + ": " + std::to_string(entry.layers.size()) + " engine layer(s)" +
                                 (entry.tyres.empty() ? ", no tread loop" : ", tread loop") +
                                 (joined.empty() ? std::string{} : " (" + joined + ")"));

                fleet.push_back(std::move(entry));
            }

            voices.assign(voiceCount, VoiceChannels{});

            return report;
        }

        void updateTraffic(const AudioListener &listener, const std::span<const TrafficVoice> placed) override {
            if (system == nullptr || fleet.empty()) {
                return;
            }

            // FMOD wants the listener's two axes unit length and at right angles, and a camera's are
            // very nearly that — made exactly so here, and the world's own axes when a camera has
            // nothing to say.
            auto forward = listener.forward;
            auto up = listener.up;

            if (glm::dot(forward, forward) < 1e-12) {
                forward = glm::dvec3(0.0, 0.0, 1.0);
            }

            forward = glm::normalize(forward);
            up -= forward * glm::dot(up, forward);

            if (glm::dot(up, up) < 1e-12) {
                up = glm::abs(forward.y) < 0.9 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0);
                up -= forward * glm::dot(up, forward);
            }

            up = glm::normalize(up);

            const auto listenerPosition = toFmod(listener.positionMetres);
            const auto listenerVelocity = toFmod(listener.velocityMetresPerSecond);
            const auto listenerForward = toFmod(forward);
            const auto listenerUp = toFmod(up);

            system->set3DListenerAttributes(0, &listenerPosition, &listenerVelocity, &listenerForward, &listenerUp);

            const auto count = std::min(placed.size(), voices.size());

            for (auto index = std::size_t{0}; index < count; index++) {
                const auto &voice = placed[index];
                auto &slot = voices[index];

                if (!voice.live || voice.body >= fleet.size()) {
                    stopVoice(slot);

                    continue;
                }

                if (slot.car != voice.car || slot.body != voice.body) {
                    stopVoice(slot);
                    startVoice(slot, voice);
                }

                const auto &fleet_bank = fleet[voice.body];
                const auto position = toFmod(voice.positionMetres);
                const auto velocity = toFmod(voice.velocityMetresPerSecond);
                const auto fade = static_cast<float>(std::clamp(voice.gain, 0.0, 1.0));

                // The siren: started the tick the car switches it on, stopped the tick it goes off, and
                // moved with the car in between so it drops through a pass like the engine does.
                if (voice.siren && slot.siren == nullptr && sirenSound != nullptr) {
                    if (system->playSound(sirenSound, trafficGroup, true, &slot.siren) != FMOD_OK ||
                        slot.siren == nullptr) {
                        slot.siren = nullptr;
                    } else {
                        slot.siren->set3DAttributes(&position, &velocity);
                        slot.siren->setVolume(0.0f);
                        slot.siren->setPaused(false);
                    }
                } else if (!voice.siren && slot.siren != nullptr) {
                    slot.siren->stop();
                    slot.siren = nullptr;
                }

                if (slot.siren != nullptr) {
                    slot.siren->set3DAttributes(&position, &velocity);
                    slot.siren->setVolume(sirenLevel * fade);
                }

                voiceMix.assign(fleet_bank.layers.size(), LayerMix{});
                mixTrafficEngineLayers(fleet_bank.layers, voice.state, fleet_bank.idleRpm, voiceMix);

                for (auto layer = std::size_t{0}; layer < slot.engine.size() && layer < voiceMix.size(); layer++) {
                    auto *channel = slot.engine[layer];
                    if (channel == nullptr) {
                        continue;
                    }

                    channel->set3DAttributes(&position, &velocity);
                    channel->setVolume(static_cast<float>(voiceMix[layer].gain) * fade);

                    if (layer < fleet_bank.layerBaseFrequency.size() && fleet_bank.layerBaseFrequency[layer] > 0.0f) {
                        channel->setFrequency(
                            fleet_bank.layerBaseFrequency[layer] * static_cast<float>(voiceMix[layer].pitch));
                    }
                }

                voiceTyreMix.assign(fleet_bank.tyres.size(), LayerMix{});
                mixTyreLayers(fleet_bank.tyres, voice.state, LayerPosition::Exterior, voiceTyreMix);

                for (auto layer = std::size_t{0}; layer < slot.tyre.size() && layer < voiceTyreMix.size(); layer++) {
                    auto *channel = slot.tyre[layer];
                    if (channel == nullptr) {
                        continue;
                    }

                    channel->set3DAttributes(&position, &velocity);
                    channel->setVolume(static_cast<float>(voiceTyreMix[layer].gain) * fade);

                    if (layer < fleet_bank.tyreBaseFrequency.size() && fleet_bank.tyreBaseFrequency[layer] > 0.0f) {
                        channel->setFrequency(fleet_bank.tyreBaseFrequency[layer] *
                                              static_cast<float>(voiceTyreMix[layer].pitch));
                    }
                }
            }

            for (auto index = count; index < voices.size(); index++) {
                stopVoice(voices[index]);
            }
        }

        [[nodiscard]] std::expected<void, std::string> loadSiren(const std::string &path) override {
            if (system == nullptr) {
                return std::unexpected("no FMOD system");
            }

            if (sirenSound != nullptr) {
                sirenSound->release();
                sirenSound = nullptr;
            }

            // The same 3D, world-relative, inverse-rolloff loop the engine layers are, in memory because
            // several patrol cars may run it at once; FMOD decodes the mp3 on the way in.
            const auto mode = FMOD_3D | FMOD_3D_WORLDRELATIVE | FMOD_3D_INVERSEROLLOFF | FMOD_LOOP_NORMAL |
                              FMOD_CREATESAMPLE;

            if (const auto result = system->createSound(path.c_str(), mode, nullptr, &sirenSound); result != FMOD_OK) {
                sirenSound = nullptr;

                return std::unexpected("FMOD would not open " + path + ": " + FMOD_ErrorString(result));
            }

            sirenSound->set3DMinMaxDistance(sirenNearMetres, sirenFarMetres);

            return {};
        }

        void unloadTrafficFleet() override {
            for (auto &slot: voices) {
                stopVoice(slot);
            }

            voices.clear();

            if (sirenSound != nullptr) {
                sirenSound->release();
                sirenSound = nullptr;
            }

            for (auto &fleet_bank: fleet) {
                for (auto *sound: fleet_bank.layerSounds) {
                    if (sound != nullptr) {
                        sound->release();
                    }
                }

                for (auto *sound: fleet_bank.tyreSounds) {
                    if (sound != nullptr) {
                        sound->release();
                    }
                }
            }

            fleet.clear();
            voiceMix.clear();
            voiceTyreMix.clear();

            if (system != nullptr) {
                system->update();
            }
        }

    private:
        // Every loop of the car's bank, started silent and placed where the car is. Silent rather than
        // at its gain, because the allocator's fade is what brings a voice in: a loop started at full
        // level mid-cycle is a click, and this is the one place a loop is started while it can be heard.
        void startVoice(VoiceChannels &slot, const TrafficVoice &voice) {
            const auto &fleet_bank = fleet[voice.body];
            const auto position = toFmod(voice.positionMetres);
            const auto velocity = toFmod(voice.velocityMetresPerSecond);

            const auto start = [&](const std::vector<FMOD::Sound *> &sounds, std::vector<FMOD::Channel *> &channel_vector) {
                channel_vector.assign(sounds.size(), nullptr);

                for (auto index = std::size_t{0}; index < sounds.size(); index++) {
                    if (sounds[index] == nullptr) {
                        continue;
                    }

                    if (system->playSound(sounds[index], trafficGroup, true, &channel_vector[index]) != FMOD_OK ||
                        channel_vector[index] == nullptr) {
                        channel_vector[index] = nullptr;

                        continue;
                    }

                    channel_vector[index]->set3DAttributes(&position, &velocity);
                    channel_vector[index]->setVolume(0.0f);
                    channel_vector[index]->setPaused(false);
                }
            };

            start(fleet_bank.layerSounds, slot.engine);
            start(fleet_bank.tyreSounds, slot.tyre);

            slot.car = voice.car;
            slot.body = voice.body;
        }

        void stopVoice(VoiceChannels &slot) {
            for (auto *channel: slot.engine) {
                if (channel != nullptr) {
                    channel->stop();
                }
            }

            for (auto *channel: slot.tyre) {
                if (channel != nullptr) {
                    channel->stop();
                }
            }

            if (slot.siren != nullptr) {
                slot.siren->stop();
                slot.siren = nullptr;
            }

            slot.engine.clear();
            slot.tyre.clear();
            slot.car = noTrafficCar;
            slot.body = 0;
        }
    };

    std::expected<std::unique_ptr<IAudioBackend>, std::string> createFmodAudioBackend(std::string capturePath) {
        auto backend = std::make_unique<FmodAudioBackend>();

        if (const auto started = backend->start(std::move(capturePath)); !started) {
            return std::unexpected(started.error());
        }

        return backend;
    }

#else

    // **Not absent, and that is the point.** The file compiles either way and the factory answers either
    // way — with a sentence naming what is missing rather than with a link error or a target that
    // silently is not built. A game asking for FMOD on a machine without it gets told so and falls back
    // to silence, which is what the DirectInput backend does off Windows for exactly the same reason.
    std::expected<std::unique_ptr<IAudioBackend>, std::string> createFmodAudioBackend(std::string) {
        return std::unexpected("this build has no FMOD: configure with -DRACEENGINE_WITH_FMOD=ON and "
            "-DFMOD_ROOT=<the FMOD Engine folder>");
    }

#endif
} // namespace raceengine
