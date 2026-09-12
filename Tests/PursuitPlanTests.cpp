// The pursuit planner: the speed profile along a route, its braking points and the curvature the
// turn-in reads (docs/pursuit-navigation-brief.md, stage 3). Pure over a polyline built in code.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::PlannerLimits;
using raceengine::planSpeedProfile;
using raceengine::profileBrakingPoint;
using raceengine::profileCurvatureAt;
using raceengine::profileSpeedAt;
using raceengine::RoutePoint;
using raceengine::signedCurvature;

using Catch::Matchers::WithinAbs;

namespace
{

constexpr auto gravity = 9.80665;

// A straight up +z, a left-hand quarter circle of the stated radius onto +x, and a straight out,
// sampled every four metres. The arc's samples are flagged as the caller says.
struct Corner
{
    std::vector<RoutePoint> points;
    double arcEntryMetres = 0.0;
    double arcExitMetres = 0.0;
};

[[nodiscard]] Corner cornerRoute(const double approach, const double radius, const double exit, const bool arcOnGround)
{
    auto corner = Corner{};
    auto distance = 0.0;
    auto last = glm::dvec3(0.0);

    const auto push = [&](const glm::dvec3& position, const bool ground)
    {
        if (!corner.points.empty())
        {
            distance += glm::length(position - last);
        }

        corner.points.push_back(RoutePoint{.positionMetres = position, .distanceMetres = distance, .ground = ground});
        last = position;
    };

    const auto approachSteps = static_cast<int>(approach / 4.0);
    for (auto step = 0; step <= approachSteps; step++)
    {
        push(glm::dvec3(0.0, 0.0, approach * static_cast<double>(step) / static_cast<double>(approachSteps)), false);
    }

    corner.arcEntryMetres = distance;

    // Centre at (radius, approach): the arc from angle pi (the entry) down to pi / 2 (the exit).
    const auto arcLength = 0.5 * std::numbers::pi * radius;
    const auto arcSteps = static_cast<int>(std::ceil(arcLength / 4.0));
    for (auto step = 1; step <= arcSteps; step++)
    {
        const auto angle =
            std::numbers::pi - 0.5 * std::numbers::pi * static_cast<double>(step) / static_cast<double>(arcSteps);
        push(glm::dvec3(radius + radius * std::cos(angle), 0.0, approach + radius * std::sin(angle)), arcOnGround);
    }

    corner.arcExitMetres = distance;

    const auto exitSteps = static_cast<int>(exit / 4.0);
    for (auto step = 1; step <= exitSteps; step++)
    {
        push(glm::dvec3(radius + exit * static_cast<double>(step) / static_cast<double>(exitSteps), 0.0,
                        approach + radius),
             false);
    }

    return corner;
}

} // namespace

TEST_CASE("three points give the circle's curvature, signed left positive, and nothing for a straight",
          "[police][police-plan]")
{
    // On a circle of radius 20 about the origin, three points a few degrees apart, going round
    // anticlockwise seen from above (+z toward +x is a left turn in this frame).
    const auto radius = 20.0;
    const auto on = [&](const double angle)
    {
        return glm::dvec3(radius * std::sin(angle), 0.0, radius * std::cos(angle));
    };

    REQUIRE_THAT(signedCurvature(on(0.0), on(0.1), on(0.2)), WithinAbs(1.0 / radius, 1e-9));
    REQUIRE_THAT(signedCurvature(on(0.2), on(0.1), on(0.0)), WithinAbs(-1.0 / radius, 1e-9));
    REQUIRE_THAT(signedCurvature(glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 4.0), glm::dvec3(0.0, 0.0, 8.0)),
                 WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(signedCurvature(glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(1.0, 0.0, 1.0)), WithinAbs(0.0, 1e-12));
}

TEST_CASE("the corner: the apex is planned at 12.2 m/s and the braking point from 30 m/s falls 47 m before it",
          "[police][police-plan]")
{
    // A 320 m approach: braking from the cap to the apex at 8 m/s² takes 231 m, so the first
    // ninety are at the cap and the rest the parabola.
    const auto corner = cornerRoute(320.0, 20.0, 100.0, false);
    const auto limits = PlannerLimits{};
    const auto profile = planSpeedProfile(corner.points, limits);

    REQUIRE(profile.distanceMetres.size() == corner.points.size());

    // The arc's curvature, read half way round, is the circle's, left positive.
    const auto apexMetres = 0.5 * (corner.arcEntryMetres + corner.arcExitMetres);
    REQUIRE_THAT(profileCurvatureAt(profile, apexMetres), WithinAbs(1.0 / 20.0, 0.002));
    REQUIRE_THAT(profileCurvatureAt(profile, 100.0), WithinAbs(0.0, 1e-9));

    // sqrt(0.85 g · 0.9 · 20) = 12.25 through the arc, and the cap on the straights far enough out.
    const auto apex = std::sqrt(0.85 * gravity * 0.9 * 20.0);
    REQUIRE_THAT(apex, WithinAbs(12.25, 0.01));
    REQUIRE_THAT(profileSpeedAt(profile, apexMetres), WithinAbs(apex, 0.05));
    REQUIRE_THAT(profileSpeedAt(profile, 20.0), WithinAbs(limits.maximumSpeedMetresPerSecond, 1e-9));

    // From 30 m/s at 8 m/s²: (30² − 12.25²) / 16 = 46.9 m before the speed has to be the apex's, which
    // is the first fully curved sample, one sample into the arc.
    const auto brakingPoint = profileBrakingPoint(profile, 0.0, 30.0);
    const auto constraint = corner.arcEntryMetres + 4.0;
    REQUIRE_THAT(constraint - brakingPoint, WithinAbs(46.9, 4.5));

    // The profile is monotonic into the corner and the braking pass is the stated rate: from the
    // braking point to the constraint, v² falls at 16 per metre.
    const auto atBraking = profileSpeedAt(profile, brakingPoint);
    const auto expected = std::sqrt(apex * apex + 2.0 * 8.0 * (constraint - brakingPoint));
    REQUIRE_THAT(atBraking, WithinAbs(expected, 0.6));

    // Out of the corner the forward pass allows 4 m/s² from the last fully curved sample, one short
    // of the exit vertex (whose three points span arc and straight): twenty-four metres on from it,
    // sqrt(12.25² + 2 · 4 · 24) = 18.5.
    REQUIRE_THAT(profileSpeedAt(profile, corner.arcExitMetres + 20.0),
                 WithinAbs(std::sqrt(apex * apex + 2.0 * 4.0 * 24.0), 0.6));

    // On the ground the arc is planned at sqrt(0.7) of the road's speed.
    const auto onGrass = planSpeedProfile(cornerRoute(320.0, 20.0, 100.0, true).points, limits);
    REQUIRE_THAT(profileSpeedAt(onGrass, apexMetres), WithinAbs(apex * std::sqrt(0.7), 0.05));

    // An exit speed of zero brakes the last forty metres: twenty from the end, sqrt(2 · 8 · 20) = 17.9.
    auto stopping = limits;
    stopping.exitSpeedMetresPerSecond = 0.0;
    const auto toStop = planSpeedProfile(corner.points, stopping);
    REQUIRE_THAT(toStop.speedMetresPerSecond.back(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(profileSpeedAt(toStop, toStop.lengthMetres() - 20.0), WithinAbs(std::sqrt(2.0 * 8.0 * 20.0), 0.6));

    // Nothing to plan is nothing.
    REQUIRE(planSpeedProfile({}, limits).empty());
}
