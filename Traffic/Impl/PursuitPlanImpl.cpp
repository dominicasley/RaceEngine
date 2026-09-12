// Pursuit planner bodies. Declarations are in Api/PursuitPlan.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export`.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.traffic;

namespace raceengine
{

namespace
{

constexpr auto standardGravity = 9.80665;

// Positive when v turns left of u in plan: with u along +z and v along +x, +x being the left.
[[nodiscard]] double cross2(const glm::dvec3& u, const glm::dvec3& v)
{
    return u.z * v.x - u.x * v.z;
}

[[nodiscard]] double interpolate(const std::vector<double>& at, const std::vector<double>& values,
                                 const double distance)
{
    if (at.empty() || at.size() != values.size())
    {
        return 0.0;
    }

    if (distance <= at.front())
    {
        return values.front();
    }

    if (distance >= at.back())
    {
        return values.back();
    }

    const auto after = std::ranges::upper_bound(at, distance);
    const auto index = static_cast<std::size_t>(after - at.begin());
    const auto span = at[index] - at[index - 1];
    const auto along = span > 1e-12 ? (distance - at[index - 1]) / span : 0.0;

    return values[index - 1] + (values[index] - values[index - 1]) * along;
}

} // namespace

double signedCurvature(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c)
{
    const auto u = glm::dvec3(b.x - a.x, 0.0, b.z - a.z);
    const auto v = glm::dvec3(c.x - b.x, 0.0, c.z - b.z);
    const auto w = glm::dvec3(c.x - a.x, 0.0, c.z - a.z);
    const auto lengths = glm::length(u) * glm::length(v) * glm::length(w);

    if (lengths < 1e-9)
    {
        return 0.0;
    }

    // Menger: twice the sine of the turn over the chord.
    return 2.0 * cross2(u, v) / lengths;
}

SpeedProfile planSpeedProfile(const std::span<const RoutePoint> points, const PlannerLimits& limits)
{
    RACEENGINE_ZONE_N("pursuit speed profile");

    auto profile = SpeedProfile{};
    const auto count = points.size();

    if (count == 0)
    {
        return profile;
    }

    profile.distanceMetres.resize(count);
    profile.curvature.assign(count, 0.0);
    profile.ceilingMetresPerSecond.resize(count);
    profile.speedMetresPerSecond.resize(count);

    const auto cap = std::max(limits.maximumSpeedMetresPerSecond, 0.0);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        profile.distanceMetres[index] = points[index].distanceMetres;
    }

    // 1. The curvature at every sample: the circle through the samples a baseline either side of it
    //    (clamped to the ends), then the median over the window about it (`PlannerLimits`).
    const auto baseline = std::max(limits.curvatureBaselineSamples, std::size_t{1});
    auto raw = std::vector<double>(count, 0.0);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto before = index >= baseline ? index - baseline : std::size_t{0};
        const auto after = std::min(index + baseline, count - 1);

        if (before == index || after == index)
        {
            continue;
        }

        raw[index] = signedCurvature(points[before].positionMetres, points[index].positionMetres,
                                     points[after].positionMetres);
    }

    const auto half = std::max(limits.curvatureMedianSamples, std::size_t{1}) / 2;
    auto window = std::vector<double>();
    window.reserve(2 * half + 1);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto from = index >= half ? index - half : std::size_t{0};
        const auto to = std::min(index + half, count - 1);

        window.assign(raw.begin() + static_cast<std::ptrdiff_t>(from), raw.begin() + static_cast<std::ptrdiff_t>(to) + 1);
        const auto middle = window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
        std::ranges::nth_element(window, middle);

        profile.curvature[index] = *middle;

        // A lone kink the median dropped, sharp enough to be a corner on its own: the circle through
        // the neighbouring samples reads sin(θ/2)/2 at the vertex, the driver's look-ahead arc
        // through θ has radius L / (2 sin(θ/2)), and at the tyre's limit that is a curvature of
        // (2 sin(θ/2))² / (T² a) — kept where it exceeds the median, with its own sign.
        if (index >= 1 && index + 1 < count)
        {
            const auto sharp = signedCurvature(points[index - 1].positionMetres, points[index].positionMetres,
                                               points[index + 1].positionMetres);
            const auto grip = points[index].ground ? limits.offRoadGripFactor : 1.0;
            const auto lateral =
                std::max(limits.corneringLimitG * standardGravity * limits.cornerMargin * grip, 1e-6);
            const auto seconds = std::max(limits.lookAheadSeconds, 1e-3);
            const auto chord = 4.0 * std::abs(sharp);
            const auto kink = chord * chord / (seconds * seconds * lateral);

            if (kink > std::abs(profile.curvature[index]))
            {
                profile.curvature[index] = std::copysign(kink, sharp);
            }
        }
    }

    // 2. The corner ceiling: the speed the stated share of the tyre holds through that curvature.
    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto grip = points[index].ground ? limits.offRoadGripFactor : 1.0;
        const auto lateral = std::max(limits.corneringLimitG * standardGravity * limits.cornerMargin * grip, 0.0);
        const auto bend = std::abs(profile.curvature[index]);
        const auto ceiling = bend > 1e-6 ? std::sqrt(lateral / bend) : cap;

        profile.ceilingMetresPerSecond[index] = std::min(ceiling, cap);
        profile.speedMetresPerSecond[index] = profile.ceilingMetresPerSecond[index];
    }

    auto& speed = profile.speedMetresPerSecond;
    const auto braking = std::max(limits.brakingMetresPerSecondSquared, 0.0);
    const auto acceleration = std::max(limits.accelerationMetresPerSecondSquared, 0.0);

    // 3. The braking pass, backward from the far end.
    speed[count - 1] = std::min(speed[count - 1], std::max(limits.exitSpeedMetresPerSecond, 0.0));

    for (auto index = count - 1; index > 0; index--)
    {
        const auto step = std::max(profile.distanceMetres[index] - profile.distanceMetres[index - 1], 0.0);
        const auto reachable = std::sqrt(speed[index] * speed[index] + 2.0 * braking * step);

        speed[index - 1] = std::min(speed[index - 1], reachable);
    }

    // 4. The acceleration pass, forward.
    speed[0] = std::min(speed[0], std::max(limits.entrySpeedMetresPerSecond, 0.0));

    for (auto index = std::size_t{1}; index < count; index++)
    {
        const auto step = std::max(profile.distanceMetres[index] - profile.distanceMetres[index - 1], 0.0);
        const auto reachable = std::sqrt(speed[index - 1] * speed[index - 1] + 2.0 * acceleration * step);

        speed[index] = std::min(speed[index], reachable);
    }

    return profile;
}

double profileSpeedAt(const SpeedProfile& profile, const double distanceMetres)
{
    return interpolate(profile.distanceMetres, profile.speedMetresPerSecond, distanceMetres);
}

double profileCurvatureAt(const SpeedProfile& profile, const double distanceMetres)
{
    return interpolate(profile.distanceMetres, profile.curvature, distanceMetres);
}

double profileBrakingPoint(const SpeedProfile& profile, const double fromMetres, const double speedMetresPerSecond)
{
    for (auto index = std::size_t{0}; index < profile.distanceMetres.size(); index++)
    {
        if (profile.distanceMetres[index] < fromMetres)
        {
            continue;
        }

        if (profile.speedMetresPerSecond[index] < speedMetresPerSecond)
        {
            return profile.distanceMetres[index];
        }
    }

    return profile.lengthMetres() + 1.0;
}

} // namespace raceengine
