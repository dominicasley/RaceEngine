// TyreLayers bodies. Declarations are in Audio/Api/TyreLayers.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

module raceengine.audio;

import :TyreLayers;
import :CarAudio;
import :EngineLayers;

namespace raceengine
{

[[nodiscard]] std::vector<TyreLayer> classifyTyreLayers(const std::span<const std::string> names)
{
    auto layers = std::vector<TyreLayer>{};

    for (auto index = std::size_t{0}; index < names.size(); index++)
    {
        auto lowered = names[index];
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });

        auto layer = TyreLayer{};

        if (lowered.find("rolling") != std::string::npos)
        {
            layer.noise = TyreNoise::Rolling;
        }
        else if (lowered.find("skid") != std::string::npos)
        {
            layer.noise = TyreNoise::Skid;

            // `skid_ext_mono` names its side; the bare `Skid` beside it is the other one, the same
            // way `4c_ex_*` beside `4c_in_*` pairs. A bare take with no `ext` sibling would still be
            // right to call interior-by-default here, because the listener this engine runs today is
            // the exterior and misassigning a side is a silent layer, not a wrong noise.
            if (lowered.find("ext") != std::string::npos)
            {
                layer.position = LayerPosition::Exterior;
                layer.positioned = true;
            }
            else if (lowered.find("int") != std::string::npos)
            {
                layer.position = LayerPosition::Interior;
                layer.positioned = true;
            }
            else
            {
                layer.position = LayerPosition::Interior;
                layer.positioned = true;
            }
        }
        else
        {
            continue;
        }

        layer.name = names[index];
        layer.sample = static_cast<int>(index);
        layers.push_back(std::move(layer));
    }

    return layers;
}

void mixTyreLayers(const std::span<const TyreLayer> layers, const CarAudioState& car, const LayerPosition listening,
                   const std::span<LayerMix> out)
{
    const auto count = std::min(layers.size(), out.size());

    const auto rolling = std::clamp(car.rollingSpeed / tyreRollingReferenceSpeed, 0.0, 1.0);

    // The square root because the ear wants the noise present early and growing, not arriving in the
    // last octave of speed: half the reference speed is well over half the loudness on the road too.
    const auto rollingGain = std::sqrt(rolling);
    // The recording is the reference speed; below it the hiss deepens. Gentler than proportional,
    // because tread noise rises in level far more than in pitch.
    const auto rollingPitch = std::clamp(0.75 + 0.5 * rolling, 0.5, 2.0);

    const auto slipping =
        std::clamp((car.wheelSlip - tyreSkidOnsetSlip) / (tyreSkidFullSlip - tyreSkidOnsetSlip), 0.0, 1.0);
    // The burnout case is why the tread's speed counts here too: at a standstill the road speed is
    // zero and the sliding is all the tread's.
    const auto carrying = std::clamp(std::max(car.roadSpeed, car.rollingSpeed) / tyreSkidAudibleSpeed, 0.0, 1.0);
    const auto skidGain = slipping * std::clamp(car.slipLoad, 0.0, 1.0) * carrying;

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto& layer = layers[index];
        out[index] = LayerMix{.sample = layer.sample, .gain = 0.0, .pitch = 1.0};

        if (layer.positioned && layer.position != listening)
        {
            continue;
        }

        if (layer.noise == TyreNoise::Rolling)
        {
            out[index].gain = rollingGain;
            out[index].pitch = rollingPitch;
        }
        else
        {
            out[index].gain = skidGain;
        }
    }
}

} // namespace raceengine
