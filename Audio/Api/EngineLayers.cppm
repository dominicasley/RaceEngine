module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

export module raceengine.audio:EngineLayers;

import :CarAudio;

namespace raceengine
{

// What makes a recording of an engine sound like an engine you are driving.
//
// A bank holds the same engine recorded at a handful of speeds, on and off the throttle, inside and
// out. None of them is the sound of the car: the sound is the *blend*, and the blend is a function of
// engine speed and load that has to be evaluated every tick. Two things happen at once here and both
// are needed —
//
//   **crossfade**, so the note moves between recordings rather than switching at a boundary, and
//   **pitch**, so between two recordings the note still rises with the engine instead of sitting flat
//   until the next layer takes over.
//
// Pitch alone is what a cheap engine sound is: one loop resampled across the whole range, which howls
// at the top and mumbles at the bottom because a real engine's harmonic content changes with speed
// and a resampled one's does not. Crossfade alone steps. Together they are convincing, and the whole
// of this file is those two.
//
// Pure, and deliberately: this is the part worth being sure of, and being sure of it requires no
// device, no bank and no sound card.

export enum class LayerPosition : std::uint8_t { Exterior, Interior };

// What the engine is doing, not how fast. `Idle` is its own case rather than the bottom of `On`
// because a bank records it separately and because an idling engine under no load is a different
// noise from the same speed being driven through.
export enum class LayerThrottle : std::uint8_t { On, Off, Idle };

export struct EngineLayer
{
    std::string name;
    // Index into whatever holds the samples. The mixer never opens a sound; it says which and how
    // loud, and the backend owns the rest.
    int sample = -1;

    LayerPosition position = LayerPosition::Exterior;
    LayerThrottle throttle = LayerThrottle::On;

    // Where this recording sits, in rpm. **Authored rather than measured**: a bank states a layer's
    // band by naming it `low` or `high` and nowhere states the speed the microphone heard, so this is
    // a sound designer's number and is meant to be moved by ear. What it must not be is a physics
    // claim — nothing downstream of the speaker reads it.
    double centreRpm = 0.0;
};

// One layer's contribution this tick.
export struct LayerMix
{
    int sample = -1;
    double gain = 0.0;
    // Multiplier on the recording's own rate. 1.0 plays it as recorded.
    double pitch = 1.0;
};

// The bands an Assetto Corsa bank names, lowest first. The words are the authors' and are consistent
// across every AC car I have looked at, which is what makes classifying by name work at all.
export inline constexpr std::array<std::string_view, 6> engineBands{"idle", "verylow", "low",
                                                                    "mid",  "high",    "veryhigh"};

// Where the bands sit for an engine that idles and revs to these speeds.
//
// Geometric rather than linear. An engine's note is heard in octaves — the interval from 1000 to 2000
// rpm is the same musical distance as 3000 to 6000 — so bands spread linearly would bunch every
// recording into the top of the range and leave the bottom to one loop stretched over an octave and a
// half, which is exactly what the cheap version sounds like.
export [[nodiscard]] inline double bandCentreRpm(const std::size_t band, const double idleRpm, const double limiterRpm)
{
    if (band == 0)
    {
        return idleRpm;
    }

    const auto low = std::max(idleRpm, 1.0);
    const auto high = std::max(limiterRpm, low * 2.0);
    const auto steps = static_cast<double>(engineBands.size() - 1);
    const auto through = static_cast<double>(band) / steps;

    return low * std::pow(high / low, through);
}

// A bank's sample names into layers.
//
// AC's convention is `<engine>_<position>_<state>_<band>[_suffix]`, as in `4c_ex_on_high` or
// `4c_in_off_verylow`. Anything that does not parse is not an engine layer — a horn, a door, a
// backfire — and is left out rather than guessed at, which is what keeps a bank with unusual extras
// from putting a door slam in the engine mix.
export [[nodiscard]] std::vector<EngineLayer> classifyEngineLayers(const std::span<const std::string> names,
                                                                   const double idleRpm, const double limiterRpm);

// The mix for this instant: which samples, how loud, and at what pitch.
//
// Every layer of the wrong position is silent rather than absent — a gain of zero on a running loop
// is what lets the camera move between inside and outside the car without restarting anything, and a
// loop restarted mid-note is a click.
export void mixEngineLayers(const std::span<const EngineLayer> layers, const CarAudioState& car,
                            const LayerPosition listening, const double idleRpm, const std::span<LayerMix> out);

} // namespace raceengine
