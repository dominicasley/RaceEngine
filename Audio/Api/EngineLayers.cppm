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
                                                                   const double idleRpm, const double limiterRpm)
{
    auto layers = std::vector<EngineLayer>{};

    for (auto index = std::size_t{0}; index < names.size(); index++)
    {
        const auto& name = names[index];

        auto parts = std::vector<std::string_view>{};
        auto view = std::string_view(name);

        while (!view.empty())
        {
            const auto at = view.find('_');
            parts.push_back(view.substr(0, at));
            view = at == std::string_view::npos ? std::string_view{} : view.substr(at + 1);
        }

        auto layer = EngineLayer{};
        auto sawPosition = false;
        auto sawThrottle = false;
        auto band = std::size_t{0};
        auto sawBand = false;

        for (const auto part : parts)
        {
            if (part == "ex")
            {
                layer.position = LayerPosition::Exterior;
                sawPosition = true;
            }
            else if (part == "in")
            {
                layer.position = LayerPosition::Interior;
                sawPosition = true;
            }
            else if (part == "on")
            {
                layer.throttle = LayerThrottle::On;
                sawThrottle = true;
            }
            else if (part == "off")
            {
                layer.throttle = LayerThrottle::Off;
                sawThrottle = true;
            }

            for (auto candidate = std::size_t{0}; candidate < engineBands.size(); candidate++)
            {
                if (part == engineBands[candidate])
                {
                    band = candidate;
                    sawBand = true;
                }
            }
        }

        // **`idle` is a band *and* a state, and it is the only word that is both** — so which it means
        // is decided by what else the name carries. `4c_ex_idle` names no throttle and is the engine
        // sitting there; `4c_ex_on_idle` is 28 seconds of the same engine being held against the
        // throttle at the bottom of its range, which is a completely different noise. Read as the
        // state unconditionally they collide at one band, the first one in the bank wins the tie, and
        // this car idles to a recording of itself being revved.
        if (band == 0 && !sawThrottle)
        {
            layer.throttle = LayerThrottle::Idle;
        }

        if (!sawPosition || !sawBand || !(sawThrottle || band == 0))
        {
            continue;
        }

        layer.name = name;
        layer.sample = static_cast<int>(index);
        layer.centreRpm = bandCentreRpm(band, idleRpm, limiterRpm);

        // **Alternate takes are dropped rather than opened and left silent.** A bank's `_2` and `_4`
        // are second recordings of a layer the author meant to be picked between at random, and
        // picking at random means restarting a loop — which this mixer never does, because moving
        // between inside and outside the car without a click is the whole reason it mixes by gain. A
        // take that can never be selected is a decoded sample and a channel doing nothing, and a load
        // log that names it is a load log that lies about what will be heard.
        const auto already = std::find_if(layers.begin(), layers.end(),
                                          [&layer](const EngineLayer& seen)
                                          {
                                              return seen.position == layer.position &&
                                                     seen.throttle == layer.throttle &&
                                                     seen.centreRpm == layer.centreRpm;
                                          });

        if (already != layers.end())
        {
            continue;
        }

        layers.push_back(std::move(layer));
    }

    return layers;
}

namespace
{

// How much of this layer's throttle state the car is in.
[[nodiscard]] double throttleWeight(const LayerThrottle state, const CarAudioState& car, const double idleRpm)
{
    // Idle fades out as the engine leaves it, over the octave above. Below that an idle recording is
    // the only honest thing to play; above it, it is a loop of an engine doing nothing under a car
    // that is doing something.
    const auto leavingIdle = std::clamp((car.engineRpm - idleRpm) / std::max(idleRpm, 1.0), 0.0, 1.0);

    switch (state)
    {
    case LayerThrottle::Idle:
        return 1.0 - leavingIdle;
    case LayerThrottle::On:
        return leavingIdle * car.load;
    case LayerThrottle::Off:
        return leavingIdle * (1.0 - car.load);
    }

    return 0.0;
}

} // namespace

// The mix for this instant: which samples, how loud, and at what pitch.
//
// Every layer of the wrong position is silent rather than absent — a gain of zero on a running loop
// is what lets the camera move between inside and outside the car without restarting anything, and a
// loop restarted mid-note is a click.
export void mixEngineLayers(const std::span<const EngineLayer> layers, const CarAudioState& car,
                            const LayerPosition listening, const double idleRpm, const std::span<LayerMix> out)
{
    const auto count = std::min(layers.size(), out.size());

    // Nearest neighbours in rpm, on each side, for each throttle state. A layer is only crossfaded
    // against one that means the same thing: blending an on-throttle recording into an overrun one
    // because they happen to be adjacent in speed is how an engine ends up sounding like it is doing
    // both at once.
    for (auto index = std::size_t{0}; index < count; index++)
    {
        out[index] = LayerMix{.sample = layers[index].sample, .gain = 0.0, .pitch = 1.0};
    }

    for (const auto state : {LayerThrottle::Idle, LayerThrottle::On, LayerThrottle::Off})
    {
        const auto stateWeight = throttleWeight(state, car, idleRpm);

        auto belowIndex = layers.size();
        auto aboveIndex = layers.size();

        for (auto index = std::size_t{0}; index < count; index++)
        {
            const auto& layer = layers[index];

            if (layer.throttle != state || layer.position != listening)
            {
                continue;
            }

            if (layer.centreRpm <= car.engineRpm &&
                (belowIndex == layers.size() || layer.centreRpm > layers[belowIndex].centreRpm))
            {
                belowIndex = index;
            }

            if (layer.centreRpm >= car.engineRpm &&
                (aboveIndex == layers.size() || layer.centreRpm < layers[aboveIndex].centreRpm))
            {
                aboveIndex = index;
            }
        }

        // Off both ends of the recorded range, the nearest layer carries it alone and the pitch does
        // the rest. That is what a bank's `_pr` loops are for: one long recording meant to be stretched
        // rather than crossfaded.
        if (belowIndex == layers.size())
        {
            belowIndex = aboveIndex;
        }

        if (aboveIndex == layers.size())
        {
            aboveIndex = belowIndex;
        }

        if (belowIndex == layers.size())
        {
            continue;
        }

        const auto& below = layers[belowIndex];
        const auto& above = layers[aboveIndex];

        // In the log domain, for the reason the bands are spread there.
        auto blend = 0.0;
        if (belowIndex != aboveIndex && below.centreRpm > 0.0 && above.centreRpm > below.centreRpm)
        {
            const auto span = std::log(above.centreRpm / below.centreRpm);
            blend = std::clamp(std::log(std::max(car.engineRpm, 1.0) / below.centreRpm) / span, 0.0, 1.0);
        }

        // Equal power rather than linear. Two loops of the same engine are not correlated, so their
        // powers add and a linear pair dips by three decibels in the middle of every crossfade —
        // audible as a soft spot at exactly the speeds a driver spends most time at.
        const auto lower = std::cos(blend * 1.57079632679489662);
        const auto upper = std::sin(blend * 1.57079632679489662);

        out[belowIndex].gain += stateWeight * lower;
        out[belowIndex].pitch = below.centreRpm > 0.0 ? std::max(car.engineRpm, 1.0) / below.centreRpm : 1.0;

        out[aboveIndex].gain += stateWeight * upper;
        out[aboveIndex].pitch = above.centreRpm > 0.0 ? std::max(car.engineRpm, 1.0) / above.centreRpm : 1.0;
    }

    for (auto index = std::size_t{0}; index < count; index++)
    {
        out[index].gain = std::clamp(out[index].gain, 0.0, 1.0);
        // A recording played at a wildly different speed from the one it was made at stops sounding
        // like the engine and starts sounding like the tape. Past this it is better to hear the
        // neighbouring layer even slightly out of tune.
        out[index].pitch = std::clamp(out[index].pitch, 0.5, 2.0);
    }
}

} // namespace raceengine
