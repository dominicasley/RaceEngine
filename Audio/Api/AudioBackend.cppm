module;

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

export module raceengine.audio:AudioBackend;

import :CarAudio;
import :SoundBank;
import :TrafficAudio;

namespace raceengine
{

// The seam, and it is one seam rather than three because audio has one job here: take a bank, take
// what the car is doing, and make a noise. Compare the renderer, which is split into a recorder, a
// resource factory and a capture because three different parts of the engine need three different
// halves of it — nothing here has that shape.
//
// What a backend must not be handed is anything above `CarAudioState`: no vehicle, no driveline, no
// physics module at all. That is the same rule stage one of the force feedback keeps and it is kept
// for the same reason — a backend that could see the car would eventually be *tuned* against the car,
// and the compensation would live where nobody would look for it.
export class IAudioBackend
{
public:
    virtual ~IAudioBackend() = default;

    IAudioBackend() = default;
    IAudioBackend(const IAudioBackend&) = delete;
    IAudioBackend& operator=(const IAudioBackend&) = delete;
    IAudioBackend(IAudioBackend&&) = delete;
    IAudioBackend& operator=(IAudioBackend&&) = delete;

    [[nodiscard]] virtual std::string_view platform() const = 0;

    // Load a car's bank and take hold of the events it names. Fallible and says why: a bank that will
    // not load is a car with no sound, and a game that carried on silently would leave nobody knowing
    // whether the file was missing, the wrong format, or simply had no engine event in it.
    [[nodiscard]] virtual std::expected<void, std::string> loadCar(const SoundBankMap& bank, double idleRpm,
                                                                   double limiterRpm) = 0;

    // One tick of the car, in the units the bank names. Never blocks: FMOD's own update runs on its
    // thread and this only writes parameters.
    virtual void update(const CarAudioState& state) = 0;

    // Everything stops. Called when a scene goes away and on the way out, and it is separate from the
    // destructor because a game may change cars without tearing the device down.
    virtual void unloadCar() = 0;

    // What the last update actually reached. Empty when every parameter the state carries was
    // accepted; otherwise the names the bank did not declare, which is the one diagnostic that says
    // "this car's events take different parameters from the ones being written".
    [[nodiscard]] virtual std::vector<std::string> unmatchedParameters() const = 0;

    // Every parameter the loaded events actually declare, as `event.parameter`. The one diagnostic
    // that cannot be reasoned out: a bank's parameter names are the author's and the only way to know
    // them is to ask. Without it, "the car is silent" and "this bank calls rpm something else" look
    // identical from every layer above.
    [[nodiscard]] virtual std::vector<std::string> declaredParameters() const = 0;

    // The other cars. One bank per body shape, in the order the caller indexes bodies by, and the
    // number of cars that may be heard at once. What comes back is one line per bank saying what
    // classified — the same fact `declaredParameters` states for the player's car, for the same
    // reason: a fleet car whose recordings did not classify drives past silently with nothing
    // anywhere to say why.
    [[nodiscard]] virtual std::expected<std::vector<std::string>, std::string>
    loadTrafficFleet(std::span<const TrafficBank> banks, std::size_t voices) = 0;

    // Once per tick: where the ear is and where each audible car is. A voice that is not live is
    // stopped; one that changed car is restarted on the other car's bank; the rest are moved and
    // mixed. Never blocks, for the reason `update` never does.
    virtual void updateTraffic(const AudioListener& listener, std::span<const TrafficVoice> voices) = 0;

    virtual void unloadTrafficFleet() = 0;

    // The siren: one recording for the whole fleet, looped and placed on every voice whose car has
    // its siren on, with the car's own velocity so it drops through a pass. Lives with the fleet —
    // after `loadTrafficFleet`, closed by `unloadTrafficFleet`.
    [[nodiscard]] virtual std::expected<void, std::string> loadSiren(const std::string& path) = 0;
};

// A backend that makes no sound and refuses nothing.
//
// Not a stub standing in for unfinished work: it is what an unattended run uses, the same way the
// gates run with no window and no wheel. A capture must be byte-identical with and without audio, and
// the only way to be sure of that is for the silent path to be a real object the game drives rather
// than a branch the game takes.
export [[nodiscard]] std::unique_ptr<IAudioBackend> createSilentAudioBackend();

// FMOD, when it is there. `std::unexpected` when it is not, naming what is missing rather than
// returning a null nobody checks — see the comment in the implementation for why this is not simply
// absent from the build.
//
// A non-empty path takes the run's sound to a wav file instead of to a device. That is a *backend*
// argument rather than a service one because it is a statement about where the mixer's output goes,
// which is the one thing above `CarAudioState` that no other layer here has any business knowing.
export [[nodiscard]] std::expected<std::unique_ptr<IAudioBackend>, std::string>
createFmodAudioBackend(std::string capturePath);

} // namespace raceengine
