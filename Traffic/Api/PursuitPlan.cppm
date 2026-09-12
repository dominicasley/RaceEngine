module;

#include <cstddef>
#include <span>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.traffic:PursuitPlan;

import :PursuitRoute;

namespace raceengine
{

// The input plan for a route a unit will drive: the speed profile along it, from which the braking
// point before each corner and the speed through it fall out, and the curvature the turn-in reads
// (docs/pursuit-navigation-brief.md, stage 3).
//
// **Pure, and not a controller.** In: a polyline and a car's limits. Out: at every sample of the
// polyline, the curvature, the corner ceiling and the speed a car that brakes and accelerates at the
// stated rates can be doing there. The pedals stay the drivers': what the profile gives them is the
// speed to ask for, read a reaction time ahead of where the car is.

export struct PlannerLimits
{
    double maximumSpeedMetresPerSecond = 62.0;
    // The tyre's limit as a lateral acceleration in g, and the fraction of it a corner is planned at.
    double corneringLimitG = 0.85;
    double cornerMargin = 0.9;
    // On a ground leg the lateral limit is scaled by this: the grass is not the road.
    double offRoadGripFactor = 0.7;
    // The car's straight-line deceleration and the acceleration the forward pass allows.
    double brakingMetresPerSecondSquared = 8.0;
    double accelerationMetresPerSecondSquared = 4.0;
    // What the forward pass starts from at the first sample and what the backward pass ends at on
    // the last: the cap on both by default, so neither end constrains the middle.
    double entrySpeedMetresPerSecond = 62.0;
    double exitSpeedMetresPerSecond = 62.0;
    // How the curvature is read off the polyline (docs/police-driving-brief.md §9). The route is a
    // chord polyline of the lane's control points, eight metres apart on Grand City Parkway,
    // resampled every four: all of a bend's turning sits at the vertices, so the circle through
    // three points four metres apart reads **twice** the bend's curvature at a vertex and nothing
    // between — and reads every stray kink in the export as a corner. So the three points are this
    // many samples apart (two: eight metres, one control spacing, which reads a uniform bend's
    // curvature exactly on either kind of sample), and the estimate is the median over this many
    // samples about each, which drops a kink that stands alone and keeps a corner that does not.
    // A kink that stands alone is still a corner if it is sharp enough: the driver takes it as the
    // arc of its own look-ahead — this many seconds of speed — through the kink's angle, and the
    // speed at which that arc is at the tyre's limit is the ceiling. Read off the circle through the
    // three points a sample apart, whose curvature at a vertex is half the sine of half the kink.
    // Measured over the whole of Grand City Parkway: the old estimate let a unit plan 50 m/s on
    // 2.6 % of the road with a median of 33 m/s; this one on 19 % with a median of 42, reads a 40 m
    // and a 150 m bend exactly, holds a 12 m hairpin at 5.6 m/s (6.4 before), and leaves a lone 3°
    // kink uncapped where the old estimate held it to 34.
    std::size_t curvatureBaselineSamples = 2;
    std::size_t curvatureMedianSamples = 7;
    double lookAheadSeconds = 0.55;
};

export struct SpeedProfile
{
    // One entry per route point.
    std::vector<double> distanceMetres{};
    // Signed, left positive, from the three points about each sample; zero at the two ends.
    std::vector<double> curvature{};
    // The corner ceiling alone, before either pass.
    std::vector<double> ceilingMetresPerSecond{};
    // The speed after both passes: what the car can be doing here.
    std::vector<double> speedMetresPerSecond{};

    [[nodiscard]] bool empty() const
    {
        return distanceMetres.empty();
    }

    [[nodiscard]] double lengthMetres() const
    {
        return distanceMetres.empty() ? 0.0 : distanceMetres.back();
    }
};

// The curvature of the circle through three points in plan, signed left positive in this engine's
// frame (+x left, +z forward); zero where they are collinear or two coincide.
export [[nodiscard]] double signedCurvature(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c);

export [[nodiscard]] SpeedProfile planSpeedProfile(std::span<const RoutePoint> points, const PlannerLimits& limits);

// The profile read between its samples, held at its ends.
export [[nodiscard]] double profileSpeedAt(const SpeedProfile& profile, double distanceMetres);
export [[nodiscard]] double profileCurvatureAt(const SpeedProfile& profile, double distanceMetres);

// The braking point for a car at a speed: the distance along of the first sample at or past
// `fromMetres` where the profile is below that speed, or one past the profile's length where it
// never is.
export [[nodiscard]] double profileBrakingPoint(const SpeedProfile& profile, double fromMetres,
                                                double speedMetresPerSecond);

} // namespace raceengine
