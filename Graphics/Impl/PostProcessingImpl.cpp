// PostProcessing bodies. Declarations are in Graphics/Api/PostProcessing.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
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

#include <glm/glm.hpp>

module raceengine.graphics;

import :PostProcessing;
import :RenderContract;
import raceengine.graphics.models;

namespace raceengine
{

namespace
{

// Narkowicz's fit of the ACES reference tone curve. It is display-referred on the way out — the
// sRGB transfer is folded into the fit — which is why nothing encodes after it.
[[nodiscard]] float acesFilm(const float x)
{
    constexpr auto a = 2.51f;
    constexpr auto b = 0.03f;
    constexpr auto c = 2.43f;
    constexpr auto d = 0.59f;
    constexpr auto e = 0.14f;

    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

} // namespace

[[nodiscard]] float evaluateToneCurve(const float value, const ToneCurve& curve, const float exposure)
{
    const auto exposed = value * exposure;
    if (exposed <= 0.0f)
    {
        return 0.0f;
    }

    const auto shaped = toneGreyPivot * std::pow(exposed / toneGreyPivot, curve.contrast);
    auto mapped = acesFilm(shaped);
    mapped = std::pow(mapped, 1.0f + curve.toe * (1.0f - mapped));
    mapped = std::pow(mapped, 1.0f + curve.shoulder * mapped);

    return std::clamp(mapped, 0.0f, 1.0f);
}

[[nodiscard]] glm::vec3 evaluateToneCurve(const glm::vec3& value, const ToneCurve& curve, const float exposure)
{
    return glm::vec3(evaluateToneCurve(value.r, curve, exposure), evaluateToneCurve(value.g, curve, exposure),
                     evaluateToneCurve(value.b, curve, exposure));
}

} // namespace raceengine
