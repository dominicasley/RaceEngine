// EngineLayers bodies. Declarations are in Audio/Api/EngineLayers.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
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

module raceengine.audio;

import :EngineLayers;
import :CarAudio;

namespace raceengine
{

[[nodiscard]] std::vector<EngineLayer> classifyEngineLayers(const std::span<const std::string> names,
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

void mixEngineLayers(const std::span<const EngineLayer> layers, const CarAudioState& car, const LayerPosition listening,
                     const double idleRpm, const std::span<LayerMix> out)
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
