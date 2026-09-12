// The corner's configuration-dependent inertia term (docs/nonlinear-geometry-brief.md, 2026-09-08
// latest of all (c)): the proofs.
//
// A corner's generalised inertia is `I(q) = m_u·|C'(q)|²` and it varies with travel. From the kinetic
// energy `T = ½·I(q)·q̇²`, Lagrange gives `I·q̈ + ½·I'(q)·q̇² = Q`, and `I' = 2·m_u·C'·C''`, so the term
// is the `−m_u·(C'·C'')·q̇²` of the unsprung mass's own acceleration along a curving, stretching
// coordinate — one term, one sign, no factor of two between the two derivations. Until this change
// the corner solved `I·q̈ = Q`: a mass coasting along its coordinate with no force gained or lost
// world speed wherever the coordinate's metric changed.
//
// The element proofs run the production velocity step (`solveCornerRate`, no damper, no stop) on
// mechanisms whose `C(q)` has a closed form — a straight line with a stretching parametrisation, where
// the exact motion is a constant world velocity; an arc, where `C'' ≠ 0` but `|C'|` is constant and
// the term must vanish; a uniformly parametrised line, the bit-for-bit control — and on the Golf's own
// corner through `cornerInertiaSlope`. `Scheme::Old` is the build before, written out beside the
// corrected step. Every table prints; `./EngineTests "[nonlinear-geometry]" -s` shows them.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::cornerGeneralisedInertia;
using raceengine::cornerInertiaSlope;
using raceengine::CornerRateStep;
using raceengine::CornerSetup;
using raceengine::golfGtiMk7;
using raceengine::inertiaSlopeStep;
using raceengine::solveCornerRate;
using raceengine::solveCornerWithJacobian;
using raceengine::VehicleSetup;

namespace
{

constexpr auto tick = 1.0 / 360.0;

// A one-degree-of-freedom mechanism: the point's world position along its coordinate, and the
// metric's closed forms where it has them.
struct Mechanism
{
    const char* name = "";
    double mass = 50.0;
    std::function<glm::dvec3(double)> position;
    std::function<glm::dvec3(double)> jacobian;
    std::function<glm::dvec3(double)> curvature; // C''
};

[[nodiscard]] double inertiaOf(const Mechanism& mechanism, const double q)
{
    const auto jacobian = mechanism.jacobian(q);
    return mechanism.mass * glm::dot(jacobian, jacobian);
}

// The Lagrangian's own slope, `2·m·C'·C''`, from the closed forms.
[[nodiscard]] double lagrangianSlope(const Mechanism& mechanism, const double q)
{
    return 2.0 * mechanism.mass * glm::dot(mechanism.jacobian(q), mechanism.curvature(q));
}

// And the slope the production helper's arithmetic takes: the central difference of `I` over the
// same step, so that the two can be compared on a mechanism where both are known.
[[nodiscard]] double differencedSlope(const Mechanism& mechanism, const double q)
{
    return (inertiaOf(mechanism, q + inertiaSlopeStep) - inertiaOf(mechanism, q - inertiaSlopeStep)) /
           (2.0 * inertiaSlopeStep);
}

// A straight line whose parametrisation stretches: `x = (q + a·q²)·e`. `C' = (1 + 2aq)·e`,
// `C'' = 2a·e`, `I = m·(1 + 2aq)²`, `I' = 4·m·a·(1 + 2aq)`. A free mass on it moves at a constant
// world velocity, which is the exact solution the discrete steps are held to.
[[nodiscard]] Mechanism stretchedLine(const double a)
{
    const auto e = glm::normalize(glm::dvec3(0.3, 0.9, 0.1));
    return Mechanism{.name = "stretched line",
                     .mass = 50.0,
                     .position = [=](const double q) { return (q + a * q * q) * e; },
                     .jacobian = [=](const double q) { return (1.0 + 2.0 * a * q) * e; },
                     .curvature = [=](const double) { return 2.0 * a * e; }};
}

// A uniformly parametrised line: `C'' = 0`, `I' = 0`. The control.
[[nodiscard]] Mechanism uniformLine()
{
    const auto e = glm::normalize(glm::dvec3(0.3, 0.9, 0.1));
    return Mechanism{.name = "uniform line",
                     .mass = 50.0,
                     .position = [=](const double q) { return q * e; },
                     .jacobian = [=](const double) { return e; },
                     .curvature = [=](const double) { return glm::dvec3(0.0); }};
}

// An arc of radius R: `C'' ≠ 0` (the tangent turns) but `|C'| = R` at every angle, so `I' = 0`
// exactly in exact arithmetic — the term is about the metric, not the curvature.
[[nodiscard]] Mechanism arc(const double radius)
{
    return Mechanism{.name = "arc",
                     .mass = 50.0,
                     .position = [=](const double q) { return radius * glm::dvec3(std::sin(q), 1.0 - std::cos(q), 0.0); },
                     .jacobian = [=](const double q) { return radius * glm::dvec3(std::cos(q), std::sin(q), 0.0); },
                     .curvature = [=](const double q) { return radius * glm::dvec3(-std::sin(q), std::cos(q), 0.0); }};
}

enum class Scheme
{
    Old,      // I·q̈ = Q, the build before
    Corrected // I·q̈ = Q − ½·I'(q₀)·q̇₀²
};

struct Sample
{
    double t = 0.0;
    double q = 0.0;
    double rate = 0.0;
    glm::dvec3 velocity{0.0}; // C'(q)·q̇, world
    double inertia = 0.0;
    double momentum = 0.0; // I·q̇
    double energy = 0.0;   // ½·I·q̇² = ½·m·|C'·q̇|²
};

// The production velocity step on a mechanism with nothing acting on it: `solveCornerRate` with no
// damper and no stop is `q̇₀ + Q·dt/I`, then the semi-implicit advance of the coordinate.
[[nodiscard]] std::vector<Sample> coast(const Mechanism& mechanism, const Scheme scheme, const double dt,
                                        const double seconds, const double q0, const double rate0,
                                        const std::function<double(double)>& slope)
{
    auto samples = std::vector<Sample>{};
    auto q = q0;
    auto rate = rate0;
    const auto ticks = static_cast<int>(std::lround(seconds / dt));

    const auto record = [&](const double t)
    {
        const auto inertia = inertiaOf(mechanism, q);
        samples.push_back(Sample{.t = t,
                                 .q = q,
                                 .rate = rate,
                                 .velocity = mechanism.jacobian(q) * rate,
                                 .inertia = inertia,
                                 .momentum = inertia * rate,
                                 .energy = 0.5 * inertia * rate * rate});
    };
    record(0.0);

    for (auto step = 1; step <= ticks; step++)
    {
        const auto inertia = inertiaOf(mechanism, q);
        const auto geometry = scheme == Scheme::Corrected ? -0.5 * slope(q) * rate * rate : 0.0;

        rate = solveCornerRate(CornerRateStep{.previousRate = rate,
                                              .generalisedForce = geometry,
                                              .generalisedInertia = inertia,
                                              .damperCoefficient = 0.0,
                                              .stopCoefficient = 0.0,
                                              .deltaTime = dt});
        q += rate * dt;
        record(static_cast<double>(step) * dt);
    }

    return samples;
}

// A fourth-order reference for `q̈ = −½·(I'/I)·q̇²` at a step a hundred times finer.
[[nodiscard]] std::vector<Sample> reference(const Mechanism& mechanism, const double dt, const double seconds,
                                            const double q0, const double rate0,
                                            const std::function<double(double)>& slope)
{
    auto samples = std::vector<Sample>{};
    auto q = q0;
    auto rate = rate0;
    const auto fine = dt / 100.0;
    const auto ticks = static_cast<int>(std::lround(seconds / dt));

    const auto acceleration = [&](const double at, const double v) { return -0.5 * slope(at) / inertiaOf(mechanism, at) * v * v; };
    const auto record = [&](const double t)
    {
        const auto inertia = inertiaOf(mechanism, q);
        samples.push_back(Sample{.t = t,
                                 .q = q,
                                 .rate = rate,
                                 .velocity = mechanism.jacobian(q) * rate,
                                 .inertia = inertia,
                                 .momentum = inertia * rate,
                                 .energy = 0.5 * inertia * rate * rate});
    };
    record(0.0);

    for (auto step = 1; step <= ticks; step++)
    {
        for (auto sub = 0; sub < 100; sub++)
        {
            const auto k1q = rate;
            const auto k1v = acceleration(q, rate);
            const auto k2q = rate + 0.5 * fine * k1v;
            const auto k2v = acceleration(q + 0.5 * fine * k1q, rate + 0.5 * fine * k1v);
            const auto k3q = rate + 0.5 * fine * k2v;
            const auto k3v = acceleration(q + 0.5 * fine * k2q, rate + 0.5 * fine * k2v);
            const auto k4q = rate + fine * k3v;
            const auto k4v = acceleration(q + fine * k3q, rate + fine * k3v);
            q += fine / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
            rate += fine / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }
        record(static_cast<double>(step) * dt);
    }

    return samples;
}

struct Errors
{
    double trajectory = 0.0; // max |q − q_ref|, rad
    double velocity = 0.0;   // max |v − v_exact| / |v_exact|
    double energy = 0.0;     // max |E − E₀| / E₀
    double momentum = 0.0;   // max |p − p₀| / |p₀|, informational: not an invariant of a changing metric
};

[[nodiscard]] Errors errorsOf(const std::vector<Sample>& run, const std::vector<Sample>& ref,
                              const glm::dvec3& exactVelocity)
{
    auto errors = Errors{};
    for (auto index = std::size_t{0}; index < run.size(); index++)
    {
        errors.trajectory = std::max(errors.trajectory, std::abs(run[index].q - ref[index].q));
        errors.velocity =
            std::max(errors.velocity, glm::length(run[index].velocity - exactVelocity) / glm::length(exactVelocity));
        errors.energy = std::max(errors.energy, std::abs(run[index].energy - run[0].energy) / run[0].energy);
        errors.momentum =
            std::max(errors.momentum, std::abs(run[index].momentum - run[0].momentum) / std::abs(run[0].momentum));
    }
    return errors;
}

} // namespace

// =================================================================================================
// The derivation, checked on the geometry
// =================================================================================================

TEST_CASE("the inertia's slope is the Lagrangian's: I' = 2·m·C'·C'' on the closed forms and on the Golf's own corners",
          "[physics][suspension][nonlinear-geometry]")
{
    std::printf("\n=== I'(q) three ways ===\n");

    // The closed forms: the helper's difference against the exact `2·m·C'·C''`.
    for (const auto& mechanism : {stretchedLine(0.5), arc(0.3), uniformLine()})
    {
        auto worst = 0.0;
        for (const auto q : {-0.6, -0.3, 0.0, 0.3, 0.6})
        {
            const auto exact = lagrangianSlope(mechanism, q);
            const auto differenced = differencedSlope(mechanism, q);
            worst = std::max(worst, std::abs(differenced - exact));
            if (q == 0.0 || q == 0.6)
            {
                std::printf("  %-15s q %+.1f: I %.6f kg·m², I' exact %+.9f, differenced %+.9f kg·m²/rad\n",
                            mechanism.name, q, inertiaOf(mechanism, q), exact, differenced);
            }
        }
        // The stretched line's `I` is a quadratic, so its central difference is exact to rounding;
        // the arc's and the line's slopes are zero.
        REQUIRE(worst < 1e-9);
    }

    // The Golf: the helper's slope against `2·m·C'·C''` with `C''` from an independent central
    // difference of the solve's Jacobian, at eleven angles across each corner's authored range.
    const auto golf = golfGtiMk7().value();
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = golf.corners[index];
        auto worst = 0.0;
        auto largest = 0.0;
        for (auto sample = 0; sample <= 10; sample++)
        {
            const auto q = corner.hardpoints.droopAngle +
                           (corner.hardpoints.bumpAngle - corner.hardpoints.droopAngle) * sample / 10.0;
            const auto helper = cornerInertiaSlope(corner, q, 0.0, true);

            const auto ahead = solveCornerWithJacobian(corner.hardpoints, q + inertiaSlopeStep, 0.0).value();
            const auto behind = solveCornerWithJacobian(corner.hardpoints, q - inertiaSlopeStep, 0.0).value();
            const auto here = solveCornerWithJacobian(corner.hardpoints, q, 0.0).value();
            const auto curvature = (ahead.wheelCentrePerAngle - behind.wheelCentrePerAngle) / (2.0 * inertiaSlopeStep);
            const auto lagrangian = 2.0 * corner.unsprungMass * glm::dot(here.wheelCentrePerAngle, curvature);

            worst = std::max(worst, std::abs(helper - lagrangian));
            largest = std::max(largest, std::abs(helper));
            if (sample == 0 || sample == 5 || sample == 10)
            {
                std::printf("  Golf %s q %+.4f: I %.5f kg·m², I' helper %+.6f, 2mC'·C'' %+.6f kg·m²/rad, I'/I %+.4f /rad\n",
                            cornerAbbreviation(static_cast<Corner>(index)), q,
                            cornerGeneralisedInertia(corner, q, 0.0, true).value(), helper, lagrangian,
                            helper / cornerGeneralisedInertia(corner, q, 0.0, true).value());
            }
        }
        // The two differ at second order in the step: a part in a million of a slope of order one.
        REQUIRE(worst < 1e-4 * std::max(largest, 1.0));
        REQUIRE(largest > 0.1);
    }

    // The Golf's front corner steered: the slope at a rack travel is the slope of that geometry.
    const auto steered = cornerInertiaSlope(golf.corners[0], 0.0, 0.02, true);
    const auto centred = cornerInertiaSlope(golf.corners[0], 0.0, 0.0, true);
    std::printf("  Golf FL I' centred %+.6f, at 20 mm of rack %+.6f kg·m²/rad\n", centred, steered);
    REQUIRE(std::isfinite(steered));
}

// =================================================================================================
// The primary proof: a free mass on a stretching coordinate
// =================================================================================================

TEST_CASE("a free mass on a stretched line keeps its world velocity with the term and not without, in both directions",
          "[physics][suspension][nonlinear-geometry]")
{
    const auto mechanism = stretchedLine(0.5);
    const auto slope = [&](const double q) { return lagrangianSlope(mechanism, q); };

    std::printf("\n=== a free mass on x = q + 0.5·q², 360 Hz ===\n");
    for (const auto speed : {1.0, -1.0})
    {
        // Positive: the parametrisation stretches ahead, so `q̇` must fall as the mass advances at a
        // constant world speed. Negative: it contracts, so `|q̇|` must grow. Both from `q = 0`, where
        // `C' = e` and `q̇₀ = speed`. The negative run stops short of the parametrisation's fold at
        // `q = −1`.
        const auto seconds = speed > 0.0 ? 1.0 : 0.4;
        const auto exactVelocity = speed * mechanism.jacobian(0.0);
        const auto ref = reference(mechanism, tick, seconds, 0.0, speed, slope);

        // The exact end state: x = speed·t on the line, q from the quadratic.
        const auto xEnd = speed * seconds;
        const auto qEnd = (-1.0 + std::sqrt(1.0 + 2.0 * xEnd)) / 1.0;
        const auto rateEnd = speed / (1.0 + qEnd);
        std::printf("  speed %+.0f m/s, %.1f s: exact end q %.6f, q̇ %.6f rad/s (reference q %.6f, q̇ %.6f)\n", speed,
                    seconds, qEnd, rateEnd, ref.back().q, ref.back().rate);
        REQUIRE(ref.back().q == Catch::Approx(qEnd).margin(1e-9));
        REQUIRE(ref.back().rate == Catch::Approx(rateEnd).margin(1e-9));

        for (const auto scheme : {Scheme::Old, Scheme::Corrected})
        {
            const auto run = coast(mechanism, scheme, tick, seconds, 0.0, speed, slope);
            const auto errors = errorsOf(run, ref, exactVelocity);
            const auto& end = run.back();
            std::printf("    %-9s end q %.6f q̇ %+.6f |v| %.6f m/s, I %.4f -> %.4f, p %.4f -> %.4f, E %.4f -> %.4f J | "
                        "max errors: q %.2e rad, |v| %.2e, E %.2e, p %.2e\n",
                        scheme == Scheme::Old ? "OLD" : "CORRECTED", end.q, end.rate, glm::length(end.velocity),
                        run.front().inertia, end.inertia, run.front().momentum, end.momentum, run.front().energy,
                        end.energy, errors.trajectory, errors.velocity, errors.energy, errors.momentum);

            if (scheme == Scheme::Old)
            {
                // The build before keeps `q̇` and lets the world speed follow the metric: ×2 forward, ×0.6 back.
                REQUIRE(end.rate == Catch::Approx(speed).margin(1e-12));
                REQUIRE(errors.velocity > 0.3);
                REQUIRE(errors.energy > 0.5);
            }
            else
            {
                // The corrected step follows the exact motion to the explicit step's first order —
                // `(I'/I)·q̇·dt` per tick, which is 0.3 % of the rate forward and, going back into the
                // contracting metric where `I'/I` and `|q̇|` are both five times larger by the end, 3 %.
                const auto bound = speed > 0.0 ? 3e-3 : 1.5e-2;
                REQUIRE(errors.velocity < bound);
                REQUIRE(errors.energy < 2.0 * bound);
                REQUIRE(errors.trajectory < 2e-3);
                // And the sign: forward, the rate falls toward the exact one; back, it grows.
                REQUIRE(end.rate < speed);
                REQUIRE(end.rate == Catch::Approx(rateEnd).epsilon(bound));
                // The generalised momentum is not an invariant here and must not be read as one.
                REQUIRE(errors.momentum > 0.1);
            }
        }
    }
}

// =================================================================================================
// The constant-Jacobian control
// =================================================================================================

TEST_CASE("where the metric does not change the term is exactly nothing: the uniform line to the bit, the arc to rounding",
          "[physics][suspension][nonlinear-geometry]")
{
    std::printf("\n=== the controls, 2 s at 360 Hz ===\n");

    // The uniform line: `I' = 0.0` exactly, the term is `−0.0`, and the two schemes' trajectories are
    // the same bits at every tick.
    {
        const auto mechanism = uniformLine();
        const auto slope = [&](const double q) { return differencedSlope(mechanism, q); };
        REQUIRE(slope(0.0) == 0.0);
        REQUIRE(slope(0.37) == 0.0);

        const auto old = coast(mechanism, Scheme::Old, tick, 2.0, 0.0, 1.3, slope);
        const auto corrected = coast(mechanism, Scheme::Corrected, tick, 2.0, 0.0, 1.3, slope);
        auto identical = true;
        for (auto index = std::size_t{0}; index < old.size(); index++)
        {
            identical = identical && old[index].q == corrected[index].q && old[index].rate == corrected[index].rate;
        }
        std::printf("  uniform line: I' %.1f, %zu ticks, OLD and CORRECTED bit-identical: %s\n", slope(0.0), old.size(),
                    identical ? "yes" : "NO");
        REQUIRE(identical);
        REQUIRE(corrected.back().rate == 1.3);
    }

    // The arc: the tangent turns (`C'' ≠ 0`) but the metric is constant, so the differenced slope is
    // rounding and the term is below anything the motion can see; both schemes keep the world speed.
    {
        const auto mechanism = arc(0.3);
        const auto slope = [&](const double q) { return differencedSlope(mechanism, q); };
        auto largestSlope = 0.0;
        for (const auto q : {0.0, 0.5, 1.0, 1.5})
        {
            largestSlope = std::max(largestSlope, std::abs(slope(q)));
            REQUIRE(glm::length(mechanism.curvature(q)) > 0.29);
        }

        const auto ref = reference(mechanism, tick, 2.0, 0.0, 2.0, slope);
        for (const auto scheme : {Scheme::Old, Scheme::Corrected})
        {
            const auto run = coast(mechanism, scheme, tick, 2.0, 0.0, 2.0, slope);
            auto speedError = 0.0;
            for (const auto& sample : run)
            {
                speedError = std::max(speedError, std::abs(glm::length(sample.velocity) - 0.6) / 0.6);
            }
            const auto errors = errorsOf(run, ref, run.front().velocity);
            std::printf("  arc: %-9s |I'| ≤ %.1e, world speed error %.1e, energy error %.1e, q end %.6f (reference %.6f)\n",
                        scheme == Scheme::Old ? "OLD" : "CORRECTED", largestSlope, speedError, errors.energy, run.back().q,
                        ref.back().q);
            REQUIRE(largestSlope < 1e-11);
            REQUIRE(speedError < 1e-12);
            REQUIRE(errors.energy < 1e-12);
        }
    }
}

// =================================================================================================
// The Golf's own corner as a mechanism
// =================================================================================================

TEST_CASE("the Golf's rear wheel coasting across its travel keeps its world speed with the term, and loses it without",
          "[physics][suspension][nonlinear-geometry]")
{
    // The corner's own `I(q)` and `I'(q)` through the production helpers, the chassis held, nothing
    // acting: the wheel launched at 2 rad/s from 5 mm short of its droop limit and left to coast to
    // 5 mm short of its bump limit. The invariant is the world speed `|C'(q)|·|q̇|`; the reference is
    // the fine fourth-order integration of the corrected equation.
    const auto golf = golfGtiMk7().value();

    std::printf("\n=== the Golf's corners coasting, chassis held, 360 Hz ===\n");
    for (auto index = std::size_t{0}; index < cornerCount; index += 2)
    {
        const auto& corner = golf.corners[index];
        const auto mechanism = Mechanism{
            .name = cornerAbbreviation(static_cast<Corner>(index)),
            .mass = corner.unsprungMass,
            .position = [&](const double q) { return solveCornerWithJacobian(corner.hardpoints, q, 0.0)->wheelCentre; },
            .jacobian = [&](const double q) { return solveCornerWithJacobian(corner.hardpoints, q, 0.0)->wheelCentrePerAngle; },
            .curvature = [&](const double q)
            {
                return (solveCornerWithJacobian(corner.hardpoints, q + inertiaSlopeStep, 0.0)->wheelCentrePerAngle -
                        solveCornerWithJacobian(corner.hardpoints, q - inertiaSlopeStep, 0.0)->wheelCentrePerAngle) /
                       (2.0 * inertiaSlopeStep);
            }};
        const auto slope = [&](const double q) { return cornerInertiaSlope(corner, q, 0.0, true); };

        const auto start = corner.hardpoints.droopAngle + 0.005;
        const auto stop = corner.hardpoints.bumpAngle - 0.005;
        const auto rate0 = 2.0;
        // Long enough to cross the range at about the launch rate, and no longer.
        const auto seconds = std::floor((stop - start) / rate0 * 360.0) / 360.0;

        const auto ref = reference(mechanism, tick, seconds, start, rate0, slope);
        const auto speed0 = glm::length(mechanism.jacobian(start)) * rate0;

        for (const auto scheme : {Scheme::Old, Scheme::Corrected})
        {
            const auto run = coast(mechanism, scheme, tick, seconds, start, rate0, slope);
            auto speedError = 0.0;
            for (const auto& sample : run)
            {
                speedError = std::max(speedError, std::abs(glm::length(sample.velocity) - speed0) / speed0);
            }
            auto trajectory = 0.0;
            for (auto at = std::size_t{0}; at < run.size(); at++)
            {
                trajectory = std::max(trajectory, std::abs(run[at].q - ref[at].q));
            }
            const auto energy = std::abs(run.back().energy - run.front().energy) / run.front().energy;
            std::printf("  %s %-9s q %+.4f -> %+.4f rad over %.3f s, I %.4f -> %.4f kg·m², world speed %.5f -> %.5f m/s "
                        "(error %.2e), energy error %.2e, |q − ref| %.2e rad\n",
                        mechanism.name, scheme == Scheme::Old ? "OLD" : "CORRECTED", run.front().q, run.back().q, seconds,
                        run.front().inertia, run.back().inertia, speed0, glm::length(run.back().velocity), speedError,
                        energy, trajectory);

            REQUIRE(run.back().q <= corner.hardpoints.bumpAngle);
            if (scheme == Scheme::Corrected)
            {
                REQUIRE(speedError < 1e-3);
                REQUIRE(energy < 2e-3);
                REQUIRE(trajectory < 1e-4);
            }
            else
            {
                // The build before: the world speed follows the metric across the travel.
                REQUIRE(speedError > 3e-3);
            }
        }
    }
}

// =================================================================================================
// Timestep convergence
// =================================================================================================

TEST_CASE("the corrected step converges to the exact motion with the timestep and the old one does not",
          "[physics][suspension][nonlinear-geometry]")
{
    const auto mechanism = stretchedLine(0.5);
    const auto slope = [&](const double q) { return lagrangianSlope(mechanism, q); };
    const auto exactVelocity = mechanism.jacobian(0.0);

    std::printf("\n=== a free mass on x = q + 0.5·q², 1 s, by timestep ===\n");
    auto lastCorrected = Errors{.trajectory = 1e300, .velocity = 1e300, .energy = 1e300};
    for (const auto hertz : {120.0, 360.0, 720.0, 2880.0})
    {
        const auto dt = 1.0 / hertz;
        const auto ref = reference(mechanism, dt, 1.0, 0.0, 1.0, slope);
        const auto old = errorsOf(coast(mechanism, Scheme::Old, dt, 1.0, 0.0, 1.0, slope), ref, exactVelocity);
        const auto corrected =
            errorsOf(coast(mechanism, Scheme::Corrected, dt, 1.0, 0.0, 1.0, slope), ref, exactVelocity);

        std::printf("  %6.0f Hz: OLD q %.2e rad, |v| %.2e, E %.2e | CORRECTED q %.2e rad, |v| %.2e, E %.2e\n", hertz,
                    old.trajectory, old.velocity, old.energy, corrected.trajectory, corrected.velocity, corrected.energy);

        REQUIRE(corrected.trajectory < lastCorrected.trajectory);
        REQUIRE(corrected.velocity < lastCorrected.velocity);
        REQUIRE(corrected.energy < lastCorrected.energy);
        REQUIRE(old.velocity > 0.5);
        lastCorrected = corrected;
    }
    // First order in the step, as an explicit term is: 5.6e-3 at 120 Hz to 2.3e-4 at 2880.
    REQUIRE(lastCorrected.velocity < 3e-4);
    REQUIRE(lastCorrected.energy < 6e-4);
}

// =================================================================================================
// The Golf's geometry, tabulated for the magnitude map
// =================================================================================================

TEST_CASE("the Golf's corner inertia and its slope across the travel", "[.nonlinear-geometry-table]")
{
    // `OSR_GEOMETRY_OUT=<file>`: one row per corner and angle, for scripts/nonlinear-geometry-analysis.py
    // to read the term off the characterisation's own (q, q̇) samples.
    const auto* env = std::getenv("OSR_GEOMETRY_OUT");
    auto* file = std::fopen(env != nullptr ? env : "geometry-table.csv", "w");
    REQUIRE(file != nullptr);
    std::fprintf(file, "corner,q,inertia,slope,travelPerAngle,curvature,curvatureAlong,jacobian,offsetY\n");

    const auto golf = golfGtiMk7().value();
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = golf.corners[index];
        const auto low = corner.hardpoints.droopAngle - 0.01;
        const auto high = corner.hardpoints.bumpAngle + 0.01;
        for (auto sample = 0; sample <= 400; sample++)
        {
            const auto q = low + (high - low) * sample / 400.0;
            const auto inertia = cornerGeneralisedInertia(corner, q, 0.0, true);
            if (!inertia)
            {
                continue;
            }
            // And, since 2026-09-08 latest of all (d), the Jacobian's own derivative, its magnitude, its
            // part along the Jacobian, the Jacobian's magnitude and the wheel's vertical offset from
            // design — what the chassis-side map reads the centripetal and offset terms from.
            const auto solved = solveCornerWithJacobian(corner.hardpoints, q, 0.0).value();
            const auto curvature = raceengine::cornerJacobianCurvature(corner, q, 0.0, true);
            std::fprintf(file, "%s,%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", cornerAbbreviation(static_cast<Corner>(index)), q,
                         inertia.value(), cornerInertiaSlope(corner, q, 0.0, true), solved.travelPerAngle,
                         glm::length(curvature), glm::dot(curvature, solved.wheelCentrePerAngle) / glm::length(solved.wheelCentrePerAngle),
                         glm::length(solved.wheelCentrePerAngle), solved.wheelCentre.y - corner.hardpoints.wheelCentre.y);
        }
    }
    std::fclose(file);
}
