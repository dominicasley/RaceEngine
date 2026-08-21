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

export module raceengine.audio:TyreLayers;

import :CarAudio;
import :EngineLayers;

namespace raceengine
{

// What the tyres add over the engine, and it is two noises with two different drivers.
//
// **Rolling** is the tread against the road and follows the *tread's* speed, not the car's. The
// distinction is inaudible in a straight line and is the entire sound of the two moments this game is
// made of: a locked wheel slides at road speed with its tread stopped — no rolling, all skid — and a
// burnout is the reverse. **Skid** is the tread sliding, and follows slip and the load under it: a
// locked front under braking is loud, the unloaded inside wheel of the same corner is nearly nothing.
//
// The same shape as `:EngineLayers` and pure for the same reason: the mixer says which samples and
// how loud, a backend owns the channels, and every number below is a sound designer's — authored,
// moved by ear, and read by nothing downstream of a speaker.

export enum class TyreNoise : std::uint8_t { Rolling, Skid };

export struct TyreLayer
{
    std::string name;
    int sample = -1;

    TyreNoise noise = TyreNoise::Rolling;

    // Which side of the glass, for the samples that say. `positioned` is false for the ones that do
    // not — the Golf's `tyre_rolling` is one recording for both sides — and an unpositioned layer is
    // audible to either listener rather than assigned a side it never named.
    LayerPosition position = LayerPosition::Exterior;
    bool positioned = false;
};

// Where the rolling loop reaches full gain, m/s. ~110 km/h: above it the recording is as loud as it
// gets and the pitch keeps carrying the speed.
export inline constexpr double tyreRollingReferenceSpeed = 30.0;

// The slip band the squeal lives across. A road tyre's force peak sits near 0.10 slip, so the onset
// is just past it — squealing under every ordinary braking effort would be a car that cries wolf —
// and full voice is well before the saturated end of the range.
export inline constexpr double tyreSkidOnsetSlip = 0.12;
export inline constexpr double tyreSkidFullSlip = 0.30;

// Below this neither the car nor the tread is moving fast enough for a squeal to carry, m/s. Slip
// angles at parking speed are numerically wild and physically almost silent, and this is the fade
// that keeps a three-point turn from screaming.
export inline constexpr double tyreSkidAudibleSpeed = 2.0;

// A bank's sample names into tyre layers.
//
// The convention across AC banks: `tyre_rolling` for the tread loop, `Skid` beside `skid_ext_mono`
// for the squeal. The rules key on `rolling` and `skid` so that `flat_tyre_mono` and
// `tyre_explosion` — tyre words for noises this engine does not make — are left out rather than
// looped, exactly as the engine classifier leaves out the horn.
export [[nodiscard]] std::vector<TyreLayer> classifyTyreLayers(const std::span<const std::string> names)
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

// The tyre mix for this instant. The two noises are independent sources rather than a crossfade —
// a sliding tyre still rolls and a rolling one can be slipping — so each gets its own gain and
// nothing sums to one. Layers on the wrong side of the glass are silent rather than absent, for the
// engine mixer's reason: a loop restarted mid-note is a click.
export void mixTyreLayers(const std::span<const TyreLayer> layers, const CarAudioState& car,
                          const LayerPosition listening, const std::span<LayerMix> out)
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
