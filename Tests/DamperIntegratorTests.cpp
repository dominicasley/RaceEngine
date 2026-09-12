#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::CornerRateStep;
using raceengine::CornerSetup;
using raceengine::Curve;
using raceengine::damperDampingCoefficient;
using raceengine::damperElementOf;
using raceengine::golfGtiMk7;
using raceengine::solveCornerRate;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperForce;
using raceengine::solveElement;
using raceengine::SuspensionState;

// The corner's velocity step against the ordinary damper (docs/damper-integrator-brief.md,
// 2026-09-08 later).
//
// The step this corner took until then carried the damper's whole force at the old rate in the
// explicit numerator and its local slope in the implicit divisor, so a stated slope acted about
// twice: for `F = c·v` the per-tick map was `(1 − β)/(1 + β)` where backward Euler's is `1/(1 + β)`.
// The corrected step is one Newton iteration of backward Euler from the old rate — the affine
// intercept `F(v₀) − F'(v₀)·v₀` in the numerator beside the slope in the divisor — and these cases
// are its element-level proof on `solveCornerRate`, the production arithmetic itself, in shaft
// coordinates: with a unit Jacobian the corner's rate is the negative of the shaft's compression
// velocity, its generalised inertia the shaft mass, and a force that pushes the shaft longer (as a
// damper's does under compression) is a positive generalised force. The old step is written out
// here, in the test only, as the control every table is read against.
//
// Every table prints; `./EngineTests "[damper-integrator]" -s` shows them.

namespace
{

// The step before 2026-09-08 later, character for character: the explicit force whole in the
// numerator, the slope in the divisor. The control.
[[nodiscard]] double oldCornerRate(const CornerRateStep& step)
{
    auto rate = step.previousRate + (step.generalisedForce / step.generalisedInertia) * step.deltaTime;
    rate /= 1.0 + ((step.damperCoefficient + step.stopCoefficient) / step.generalisedInertia) * step.deltaTime;

    return rate;
}

enum class Scheme
{
    Old,
    Corrected
};

// One tick on the shaft. `explicitForce` is everything acting at `v0` that pushes the shaft
// longer, the damper's own `F(v0)` included; `slope` is the damper's floored `F'(v0)`;
// `stopCoefficient` an engaged stop's viscous coefficient. Returns the compression velocity the
// step arrives at.
[[nodiscard]] double shaftStep(const Scheme scheme, const double v0, const double explicitForce, const double mass,
                               const double slope, const double stopCoefficient, const double dt)
{
    const auto step = CornerRateStep{.previousRate = -v0,
                                     .generalisedForce = explicitForce,
                                     .generalisedInertia = mass,
                                     .damperCoefficient = slope,
                                     .stopCoefficient = stopCoefficient,
                                     .deltaTime = dt};

    return -(scheme == Scheme::Corrected ? solveCornerRate(step) : oldCornerRate(step));
}

// One corner reduced to its damper shaft, as `StopElementTests.cpp` reduces it: the generalised
// inertia the force pass uses with the geometric load path on, brought onto the shaft through the
// damper element's design Jacobian.
struct Shaft
{
    const char* name = "";
    double mass = 0.0;
    double jacobian = 0.0;
    double springRate = 0.0;
    SuspensionState design;
};

[[nodiscard]] Shaft shaftOf(const CornerSetup& corner, const char* name)
{
    const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());

    const auto jacobian = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), 0.0).lengthPerAngle;
    const auto inertia = corner.unsprungMass * glm::dot(design->wheelCentrePerAngle, design->wheelCentrePerAngle);

    return Shaft{.name = name,
                 .mass = inertia / (jacobian * jacobian),
                 .jacobian = jacobian,
                 .springRate = corner.springRate,
                 .design = *design};
}

// The production element at a shaft velocity: its force and its floored slope, both on the shaft,
// read through `solveDamperForce` and `damperDampingCoefficient` exactly as the force pass reads
// them.
struct Element
{
    double force = 0.0;
    double slope = 0.0;
};

[[nodiscard]] Element elementAt(const CornerSetup& corner, const Shaft& shaft, const double velocity)
{
    const auto damper = solveDamperForce(corner, shaft.design, -velocity / shaft.jacobian);
    REQUIRE(damper.velocity == Catch::Approx(velocity).margin(1e-12));

    return Element{.force = damper.force,
                   .slope = damperDampingCoefficient(corner, damper) / (shaft.jacobian * shaft.jacobian)};
}

// A free decay of the shaft mass against the production element, tick by tick, with the work the
// recurrence's own momentum balance says the damper did: `m (v₁ − v₀) = −F_applied · dt` exactly,
// so the kinetic energy the damper took is `F_applied · ½ (v₀ + v₁) · dt`.
struct Decay
{
    double kineticIn = 0.0;
    double kineticOut = 0.0;
    double work = 0.0;
    double injected = 0.0; // the sum over ticks of `−F_applied · v₁ · dt` where that is positive
    int injectingTicks = 0;
    int signChanges = 0;
    int ticks = 0;
    double worstBalance = 0.0; // max |m (v₁ − v₀) + F_applied dt| / (m |v₀| + 1e-9)
    double lag = 0.0;          // the spring's half-step term, `½ k dt (x₀ v₀ − x_end v_end)`, exactly
};

[[nodiscard]] Decay decay(const Scheme scheme, const CornerSetup& corner, const Shaft& shaft, const double v0,
                          const double compression0, const bool spring, const double dt, const int ticks)
{
    auto result = Decay{};
    auto velocity = v0;
    auto compression = compression0;
    result.kineticIn = 0.5 * shaft.mass * v0 * v0;
    const auto potentialIn = spring ? 0.5 * shaft.springRate * compression0 * compression0 : 0.0;

    for (auto tick = 0; tick < ticks; tick++)
    {
        const auto element = elementAt(corner, shaft, velocity);
        const auto springForce = spring ? shaft.springRate * compression : 0.0;
        const auto next = shaftStep(scheme, velocity, springForce + element.force, shaft.mass, element.slope, 0.0, dt);

        // What the recurrence applied on the damper's account: the corrected step's linearised
        // force at the velocity it arrived at, and the old step's `F(v₀) + F'(v₀)·v₁` — which is the
        // double application written as a force.
        const auto applied = scheme == Scheme::Corrected ? element.force + element.slope * (next - velocity)
                                                         : element.force + element.slope * next;

        const auto balance = shaft.mass * (next - velocity) + (springForce + applied) * dt;
        result.worstBalance = std::max(result.worstBalance, std::abs(balance) / (shaft.mass * std::abs(velocity) + 1e-9));

        result.work += applied * 0.5 * (velocity + next) * dt;
        if (applied * next < 0.0)
        {
            result.injectingTicks++;
            result.injected += -applied * next * dt;
        }
        result.signChanges += (next > 0.0) != (velocity > 0.0) ? 1 : 0;

        velocity = next;
        compression += next * dt;
        result.ticks++;
    }

    result.kineticOut = 0.5 * shaft.mass * velocity * velocity;
    if (spring)
    {
        // With the spring in the loop the budget is the whole mechanical energy. The spring's force
        // is read at the old position and the position then moves with the new velocity, so its
        // work by the same trapezoid differs from its stored energy by `½ k dt (x₀ v₀ − x₁ v₁)` per
        // tick, which telescopes: the identity is `E_in − E_out = W + ½ k dt (x₀ v₀ − x_end v_end)`,
        // exactly, and `lag` carries that term.
        result.kineticIn += potentialIn;
        result.kineticOut += 0.5 * shaft.springRate * compression * compression;
        result.lag = 0.5 * shaft.springRate * dt * (compression0 * v0 - compression * velocity);
    }

    return result;
}

} // namespace

TEST_CASE("a linear damper decays at backward Euler's rate, and its slope acts once", "[physics][suspension][damper-integrator]")
{
    const auto masses = std::array{20.0, 44.17, 58.78, 120.0};
    const auto coefficients = std::array{1000.0, 3000.0, 6000.0, 12000.0, 16200.0};
    const auto rates = std::array{120.0, 360.0, 720.0, 2880.0};
    constexpr auto v0 = 0.5;

    for (const auto mass : masses)
    {
        for (const auto c : coefficients)
        {
            for (const auto rate : rates)
            {
                const auto dt = 1.0 / rate;
                const auto beta = c * dt / mass;
                INFO("m " << mass << " c " << c << " Hz " << rate << " beta " << beta);

                // One step: the corrected scheme is backward Euler on `c·v` to rounding, and the old
                // one is `(1 − β)/(1 + β)` to rounding — which past `β = 1` has changed sign.
                const auto corrected = shaftStep(Scheme::Corrected, v0, c * v0, mass, c, 0.0, dt);
                const auto old = shaftStep(Scheme::Old, v0, c * v0, mass, c, 0.0, dt);
                REQUIRE(corrected == Catch::Approx(v0 / (1.0 + beta)).epsilon(1e-12));
                REQUIRE(old == Catch::Approx(v0 * (1.0 - beta) / (1.0 + beta)).epsilon(1e-12).margin(1e-15));
                REQUIRE(corrected > 0.0);
                REQUIRE((old > 0.0) == (beta < 1.0));

                // Many steps: the discrete sequence `v₀ / (1 + β)ⁿ`, exactly.
                auto velocity = v0;
                for (auto tick = 1; tick <= 200; tick++)
                {
                    velocity = shaftStep(Scheme::Corrected, velocity, c * velocity, mass, c, 0.0, dt);
                    REQUIRE(velocity == Catch::Approx(v0 / std::pow(1.0 + beta, tick)).epsilon(1e-10));
                }
            }
        }
    }

    // The representative case, printed: the Golf's two shaft masses at 360 Hz, with the slopes the
    // corner actually sees — the slow bump branch, the slow branch plus the seal friction's slope at
    // zero, and the 3000 N·s/m the stop brief quoted. The old scheme's effective coefficient is the
    // one whose exponential map matches its per-tick factor, `(m/dt)·ln[(1 + β)/(1 − β)]`.
    std::printf("\n=== a linear damper at 360 Hz: the per-tick factor and the coefficient that acted ===\n");
    std::printf("  %-6s %8s %8s %8s %10s %10s %12s %12s %8s\n", "shaft", "m kg", "c N.s/m", "beta", "old v1/v0",
                "new v1/v0", "old c_eff", "exact e^-b", "old/c");
    constexpr auto dt = 1.0 / 360.0;
    for (const auto [name, mass] : std::array{std::pair{"front", 58.78}, std::pair{"rear", 44.17}})
    {
        for (const auto c : std::array{3000.0, 5489.0, 6365.0, 16189.0, 8865.0})
        {
            const auto beta = c * dt / mass;
            const auto old = shaftStep(Scheme::Old, 1.0, c, mass, c, 0.0, dt);
            const auto corrected = shaftStep(Scheme::Corrected, 1.0, c, mass, c, 0.0, dt);
            const auto effective = beta < 1.0 ? (mass / dt) * std::log((1.0 + beta) / (1.0 - beta)) : -1.0;
            std::printf("  %-6s %8.2f %8.0f %8.4f %10.5f %10.5f %12.0f %12.5f %8.3f\n", name, mass, c, beta, old,
                        corrected, effective, std::exp(-beta), effective / c);
        }
    }

    // The stop brief's figure, re-derived on the same arithmetic: 3000 N·s/m on the front shaft
    // acted as about 6041.
    {
        const auto beta = 3000.0 * dt / 58.78;
        const auto effective = (58.78 / dt) * std::log((1.0 + beta) / (1.0 - beta));
        REQUIRE(effective == Catch::Approx(6041.0).epsilon(0.002));
    }
}

TEST_CASE("the corrected step converges to the continuous decay and the old one does not", "[physics][suspension][damper-integrator]")
{
    // The same physical decay at every practical rate, against `v(t) = v₀ e^{−c t / m}`. Backward
    // Euler is first order, so the corrected error halves with the timestep; the old scheme's
    // per-tick map is `e^{−2β + O(β³)}`, so it converges — to the decay at twice the rate.
    constexpr auto mass = 58.78;
    const auto rates = std::array{120.0, 360.0, 720.0, 2880.0, 11520.0};

    std::printf("\n=== three time constants of decay on the front shaft, error against the continuous solution ===\n");
    std::printf("  %8s %8s %6s %12s %12s %12s %12s %12s\n", "c N.s/m", "T s", "Hz", "exact", "corrected", "err", "old",
                "err");

    for (const auto c : std::array{3000.0, 12000.0})
    {
        // Three time constants, rounded to a whole number of ticks at the coarsest rate so that
        // every finer rate divides it.
        const auto coarseTicks = std::max(1L, std::lround(3.0 * (mass / c) * 120.0));
        const auto horizon = static_cast<double>(coarseTicks) / 120.0;
        const auto exact = std::exp(-c * horizon / mass);
        auto previousError = 1e9;
        auto oldErrorAtFinest = 0.0;

        for (const auto rate : rates)
        {
            const auto dt = 1.0 / rate;
            const auto ticks = static_cast<int>(std::lround(horizon * rate));
            REQUIRE(std::abs(static_cast<double>(ticks) * dt - horizon) < 1e-12);

            auto corrected = 1.0;
            auto old = 1.0;
            for (auto tick = 0; tick < ticks; tick++)
            {
                corrected = shaftStep(Scheme::Corrected, corrected, c * corrected, mass, c, 0.0, dt);
                old = shaftStep(Scheme::Old, old, c * old, mass, c, 0.0, dt);
            }

            const auto error = std::abs(corrected - exact) / exact;
            const auto oldError = std::abs(old - exact) / exact;
            std::printf("  %8.0f %8.4f %6.0f %12.6f %12.6f %12.3e %12.6f %12.3e\n", c, horizon, rate, exact, corrected,
                        error, old, oldError);

            // First order: each halving-or-better of the timestep cuts the error by at least the
            // same ratio less a margin. And wherever the old scheme is a decay at all (β < 1 — past
            // it the old map alternates in sign) its decay *rate* is farther from the continuous
            // one: read in the log domain, because over three time constants both amplitudes are
            // small and a relative error of one says nothing about which rate was closer.
            REQUIRE(error < previousError);
            if (previousError < 1e8)
            {
                REQUIRE(error < 0.6 * previousError);
            }
            if (c * dt / mass < 1.0)
            {
                const auto rateError = std::abs(std::log(corrected) / std::log(exact) - 1.0);
                const auto oldRateError = std::abs(std::log(old) / std::log(exact) - 1.0);
                REQUIRE(oldRateError > rateError);
            }
            previousError = error;
            oldErrorAtFinest = oldError;
        }

        // At 11520 Hz the corrected decay is within 5 % of the continuous one over three time
        // constants — backward Euler's first-order error there is about `3β/2`, 0.7 % and 2.7 % for
        // the two coefficients — and the old scheme's error has settled at the gap between `e^{−x}`
        // and `e^{−2x}`.
        REQUIRE(previousError < 5e-2);
        REQUIRE(oldErrorAtFinest == Catch::Approx(1.0 - std::exp(-c * horizon / mass)).epsilon(0.02));
    }
}

TEST_CASE("an affine damper law applies its intercept once and its slope once", "[physics][suspension][damper-integrator]")
{
    // `F = a + c·v` with a non-zero `a`: backward Euler on the law is
    // `m (v₁ − v₀)/dt = −(a + c v₁)`, so `v₁ = (v₀ − a dt/m) / (1 + β)`. That is exactly the
    // `F(v₀) − F'(v₀)·v₀ = a` term doing its job; the old step gives `(v₀ (1 − β) − a dt/m)/(1 + β)`,
    // the intercept once and the slope twice.
    constexpr auto dt = 1.0 / 360.0;
    for (const auto mass : {44.17, 58.78})
    {
        for (const auto a : {-300.0, -20.0, 20.0, 150.0})
        {
            for (const auto c : {1500.0, 3000.0, 9000.0})
            {
                for (const auto v0 : {-0.3, -0.02, 0.005, 0.2})
                {
                    const auto beta = c * dt / mass;
                    const auto corrected = shaftStep(Scheme::Corrected, v0, a + c * v0, mass, c, 0.0, dt);
                    const auto old = shaftStep(Scheme::Old, v0, a + c * v0, mass, c, 0.0, dt);
                    INFO("m " << mass << " a " << a << " c " << c << " v0 " << v0);
                    REQUIRE(corrected == Catch::Approx((v0 - a * dt / mass) / (1.0 + beta)).epsilon(1e-12));
                    REQUIRE(old == Catch::Approx((v0 * (1.0 - beta) - a * dt / mass) / (1.0 + beta)).epsilon(1e-12));

                    // And the force the corrected step applied is the law at the velocity it arrived
                    // at — `a + c v₁` — which is what makes it backward Euler and not a scheme of its
                    // own.
                    const auto applied = (a + c * v0) + c * (corrected - v0);
                    REQUIRE(applied == Catch::Approx(a + c * corrected).epsilon(1e-12).margin(1e-9));
                }
            }
        }
    }
}

TEST_CASE("the Golf's own damper curves are linearised consistently at every branch", "[physics][suspension][damper-integrator]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    constexpr auto dt = 1.0 / 360.0;

    std::printf("\n=== the Golf's damper curves at 360 Hz: one free step from v0 on the shaft mass, friction off ===\n");
    std::printf("  F0, slope and intercept are the production element's at v0; v1 the corrected step; F_applied the\n");
    std::printf("  linearised force at v1, F(v1) the curve there; old v1 is the step before the change.\n");

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        // The curve alone: the friction is the next case's.
        auto corner = golf->corners[index];
        corner.damperFriction = 0.0;
        const auto shaft = shaftOf(corner, index == 0 ? "front" : "rear");

        // The knees at the shaft, read off the authored curve (the second and fourth points), and
        // the steepest branch, which bounds the linearisation error across a knee.
        REQUIRE(corner.damper.points.size() == 5);
        const auto reboundKnee = corner.damper.points[1].x;
        const auto bumpKnee = corner.damper.points[3].x;
        auto steepest = 0.0;
        for (auto point = std::size_t{1}; point < corner.damper.points.size(); point++)
        {
            const auto& low = corner.damper.points[point - 1];
            const auto& high = corner.damper.points[point];
            steepest = std::max(steepest, (high.y - low.y) / (high.x - low.x));
        }

        std::printf("\n  --- %s: shaft mass %.2f kg, knees %.4f (rebound) / %.4f (bump) m/s ---\n", shaft.name,
                    shaft.mass, reboundKnee, bumpKnee);
        std::printf("  %9s %9s %9s %10s %9s %10s %10s %9s %9s\n", "v0 m/s", "F0 N", "slope", "intercept", "v1 m/s",
                    "F_appl N", "F(v1) N", "old v1", "lin err");

        const auto velocities = std::array{-0.30,          -0.20, reboundKnee - 0.01, reboundKnee, reboundKnee + 0.01,
                                           -0.05,          -0.02, -0.005,             0.005,       0.02,
                                           0.05,           bumpKnee - 0.01, bumpKnee,  bumpKnee + 0.01, 0.10,
                                           0.20,           0.30};

        for (const auto v0 : velocities)
        {
            const auto element = elementAt(corner, shaft, v0);
            const auto intercept = element.force - element.slope * v0;
            const auto v1 = shaftStep(Scheme::Corrected, v0, element.force, shaft.mass, element.slope, 0.0, dt);
            const auto old = shaftStep(Scheme::Old, v0, element.force, shaft.mass, element.slope, 0.0, dt);
            const auto applied = element.force + element.slope * (v1 - v0);
            const auto atSolved = elementAt(corner, shaft, v1).force;

            std::printf("  %9.4f %9.1f %9.0f %10.2f %9.4f %10.1f %10.1f %9.4f %9.2f\n", v0, element.force,
                        element.slope, intercept, v1, applied, atSolved, old, applied - atSolved);

            INFO(shaft.name << " v0 " << v0);

            // The slope is the curve's own on whichever branch v0 sits, by construction of the
            // authored piecewise curve; the differencing width is 0.1 mm/s, so away from a knee it
            // is exact and at a knee it is the mean of the two.
            REQUIRE(element.slope > 0.0);

            // The step is the affine law's backward Euler, and the force it applied is the affine
            // law at the velocity it arrived at.
            const auto beta = element.slope * dt / shaft.mass;
            REQUIRE(v1 == Catch::Approx((v0 - intercept * dt / shaft.mass) / (1.0 + beta)).epsilon(1e-12));
            REQUIRE(applied == Catch::Approx(intercept + element.slope * v1).epsilon(1e-12));

            // A concave-in-|v| curve through the origin has an intercept of v0's sign, so the
            // applied force has the sign of the velocity it acted at: it resisted, on this tick.
            REQUIRE(intercept * v0 >= -1e-9);
            REQUIRE(applied * v1 >= 0.0);
            REQUIRE(std::abs(v1) < std::abs(v0));
            REQUIRE(v1 * v0 > 0.0);

            // The old step took more velocity out on the same tick — the slope acting twice — and
            // the linearisation error against the curve at v1 is bounded by the knee it may have
            // crossed: the difference between the tangent slope and the chord's, times the velocity
            // change, and the chord's slope is at most the steepest branch.
            REQUIRE(std::abs(old) < std::abs(v1));
            REQUIRE(std::abs(applied - atSolved) <= steepest * std::abs(v1 - v0) + 1e-9);
        }
    }
}

TEST_CASE("the seal friction's value and slope each act once", "[physics][suspension][damper-integrator]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    constexpr auto dt = 1.0 / 360.0;

    std::printf("\n=== the seal friction alone, f tanh(v/w), at 360 Hz on the shaft mass ===\n");
    std::printf("  exact = backward Euler on the tanh law itself, by bisection; the two linearised steps are read\n");
    std::printf("  against it. old's applied force is F0 + F'(v0) v1, the double application written as a force.\n");

    for (const auto index : {std::size_t{0}, std::size_t{2}})
    {
        // The friction alone: the curve is emptied, which `Curve::at` reads as zero force and zero
        // slope, so `solveDamperForce` and `damperDampingCoefficient` hand back the friction's own.
        auto corner = golf->corners[index];
        corner.damper = Curve{};
        REQUIRE(corner.damperFriction > 0.0);
        const auto shaft = shaftOf(corner, index == 0 ? "front" : "rear");
        const auto f = corner.damperFriction;
        const auto w = corner.damperFrictionSpeed;

        std::printf("\n  --- %s: f %.0f N, w %.4f m/s, shaft mass %.2f kg, slope at zero %.0f N.s/m (beta %.3f) ---\n",
                    shaft.name, f, w, shaft.mass, f / w, (f / w) * dt / shaft.mass);
        std::printf("  %9s %8s %8s %9s %9s %9s %9s %9s %9s %9s\n", "v0 m/s", "F0 N", "slope", "intercept",
                    "v1 exact", "v1 new", "err new", "v1 old", "err old", "F_appl");

        for (const auto multiple : {0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0})
        {
            const auto v0 = multiple * w;
            const auto u = v0 / w;
            const auto element = elementAt(corner, shaft, v0);

            // The production value and slope are the law's, analytically.
            const auto sech2 = 1.0 - std::tanh(u) * std::tanh(u);
            REQUIRE(element.force == Catch::Approx(f * std::tanh(u)).epsilon(1e-12));
            REQUIRE(element.slope == Catch::Approx((f / w) * sech2).epsilon(1e-9));

            const auto intercept = element.force - element.slope * v0;
            REQUIRE(intercept == Catch::Approx(f * (std::tanh(u) - u * sech2)).epsilon(1e-9).margin(1e-12));

            // Backward Euler on the tanh law itself: `m (v₁ − v₀) = −f tanh(v₁/w) dt`, monotone in
            // v₁, so a bisection finds it.
            auto low = 0.0;
            auto high = v0;
            for (auto iteration = 0; iteration < 200; iteration++)
            {
                const auto middle = 0.5 * (low + high);
                (shaft.mass * (middle - v0) + f * std::tanh(middle / w) * dt < 0.0 ? low : high) = middle;
            }
            const auto exact = 0.5 * (low + high);

            const auto corrected = shaftStep(Scheme::Corrected, v0, element.force, shaft.mass, element.slope, 0.0, dt);
            const auto old = shaftStep(Scheme::Old, v0, element.force, shaft.mass, element.slope, 0.0, dt);
            const auto applied = element.force + element.slope * (corrected - v0);

            std::printf("  %9.5f %8.2f %8.0f %9.3f %9.5f %9.5f %9.2e %9.5f %9.2e %9.2f\n", v0, element.force,
                        element.slope, intercept, exact, corrected, std::abs(corrected - exact) / v0, old,
                        std::abs(old - exact) / v0, applied);

            INFO(shaft.name << " v0 " << v0);

            // The corrected step is the affine law's backward Euler, its force the affine law at v1,
            // resisting; and it sits closer to the law's own backward Euler than the old step at
            // every velocity — with equality only where the law is linear and both are exact.
            const auto beta = element.slope * dt / shaft.mass;
            REQUIRE(corrected == Catch::Approx((v0 - intercept * dt / shaft.mass) / (1.0 + beta)).epsilon(1e-12));
            REQUIRE(applied == Catch::Approx(intercept + element.slope * corrected).epsilon(1e-12));
            REQUIRE(applied * corrected > 0.0);
            REQUIRE(corrected > 0.0);
            REQUIRE(corrected < v0);
            REQUIRE(std::abs(corrected - exact) <= std::abs(old - exact) + 1e-15);
            if (element.slope * dt / shaft.mass > 1e-9)
            {
                REQUIRE(old < corrected);
            }
            else
            {
                // Thirty widths out the slope has underflowed: with no coefficient the two steps are
                // the same arithmetic and the same bits.
                REQUIRE(old == corrected);
            }
        }
    }
}

TEST_CASE("the corrected step takes energy out and never puts it in", "[physics][suspension][damper-integrator]")
{
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());

    constexpr auto dt = 1.0 / 360.0;

    std::printf("\n=== free decays on the shaft at 360 Hz: energy in, energy out, the damper's work ===\n");
    std::printf("  identity: work == in − out to rounding, from the recurrence's own momentum balance;\n");
    std::printf("  injecting = ticks on which the applied force pushed along the velocity it arrived at.\n");
    std::printf("  %-28s %-9s %7s %10s %10s %10s %9s %5s %5s %9s\n", "case", "scheme", "v0 m/s", "E in J",
                "E out J", "work J", "inject J", "tks", "sign", "balance");

    struct Case
    {
        const char* name = "";
        std::size_t corner = 0;
        bool curve = true;
        bool friction = true;
        bool spring = false;
    };

    const auto cases = std::array{Case{.name = "front curve, linear branch", .corner = 0, .friction = false},
                                  Case{.name = "front curve + friction", .corner = 0},
                                  Case{.name = "rear curve + friction", .corner = 2},
                                  Case{.name = "front friction alone", .corner = 0, .curve = false},
                                  Case{.name = "front element on its spring", .corner = 0, .spring = true},
                                  Case{.name = "rear element on its spring", .corner = 2, .spring = true}};

    for (const auto& item : cases)
    {
        auto corner = golf->corners[item.corner];
        if (!item.curve)
        {
            corner.damper = Curve{};
        }
        if (!item.friction)
        {
            corner.damperFriction = 0.0;
        }
        const auto shaft = shaftOf(corner, item.corner == 0 ? "front" : "rear");

        for (const auto v0 : {0.5, -0.5, 0.05})
        {
            // On the spring: start displaced 20 mm as well, so the ring-down crosses zero many
            // times and the friction's held force meets a velocity of the other sign.
            const auto compression0 = item.spring ? 0.020 : 0.0;
            const auto ticks = item.spring ? 4 * 360 : 2 * 360;

            for (const auto scheme : {Scheme::Old, Scheme::Corrected})
            {
                const auto result = decay(scheme, corner, shaft, v0, compression0, item.spring, dt, ticks);
                std::printf("  %-28s %-9s %7.2f %10.4f %10.4f %10.4f %9.2e %5d %5d %9.2e\n", item.name,
                            scheme == Scheme::Old ? "old" : "corrected", v0, result.kineticIn, result.kineticOut,
                            result.work, result.injected, result.injectingTicks, result.signChanges,
                            result.worstBalance);

                if (scheme != Scheme::Corrected)
                {
                    continue;
                }

                INFO(item.name << " v0 " << v0);

                // The momentum balance is the recurrence's own, to rounding.
                REQUIRE(result.worstBalance < 1e-9);

                if (!item.spring)
                {
                    // Free decay: the damper's work is the kinetic energy it took, exactly; no
                    // tick injected; the velocity never changed sign.
                    REQUIRE(result.work == Catch::Approx(result.kineticIn - result.kineticOut).epsilon(1e-9));
                    REQUIRE(result.injectingTicks == 0);
                    REQUIRE(result.injected == 0.0);
                    REQUIRE(result.signChanges == 0);
                    REQUIRE(result.kineticOut < 1e-3 * result.kineticIn);
                }
                else
                {
                    // On the spring the zero crossings are where a held friction force can push
                    // along the new velocity for one tick. The budget: what the damper took is the
                    // mechanical energy lost, less the spring's own telescoped half-step term, to
                    // rounding; it is positive; and whatever was injected on a crossing tick is a
                    // small fraction of what was dissipated.
                    REQUIRE(result.work > 0.0);
                    REQUIRE(result.work ==
                            Catch::Approx(result.kineticIn - result.kineticOut - result.lag).epsilon(1e-9));
                    REQUIRE(result.injected < 0.005 * result.work);
                    REQUIRE(result.injectingTicks <= result.signChanges);
                    REQUIRE(result.kineticOut < 0.05 * result.kineticIn);
                }
            }
        }
    }
}

TEST_CASE("the damper's affine term and an engaged stop's coefficient coexist in one divisor", "[physics][suspension][damper-integrator]")
{
    // With a stop engaged the stop's viscous coefficient sits in the divisor with no intercept
    // (docs/stop-element-brief.md); the damper's intercept joins the numerator. For a linear damper
    // and a stop together the step is backward Euler on both: `v₁ = v₀ / (1 + β + β_s)`.
    constexpr auto dt = 1.0 / 360.0;
    for (const auto mass : {44.17, 58.78})
    {
        for (const auto c : {3000.0, 9000.0})
        {
            for (const auto stop : {0.0, 2000.0, 40000.0})
            {
                for (const auto v0 : {-0.3, 0.05, 0.7})
                {
                    const auto beta = c * dt / mass;
                    const auto betaStop = stop * dt / mass;
                    const auto corrected = shaftStep(Scheme::Corrected, v0, c * v0, mass, c, stop, dt);
                    INFO("m " << mass << " c " << c << " stop " << stop << " v0 " << v0);
                    REQUIRE(corrected == Catch::Approx(v0 / (1.0 + beta + betaStop)).epsilon(1e-12));

                    // And with the stop's explicit (elastic) part as a force: the affine closed form,
                    // the elastic part once, the damper's slope once, the stop's coefficient once.
                    const auto elastic = 800.0;
                    const auto withElastic = shaftStep(Scheme::Corrected, v0, c * v0 + elastic, mass, c, stop, dt);
                    REQUIRE(withElastic ==
                            Catch::Approx((v0 - elastic * dt / mass) / (1.0 + beta + betaStop)).epsilon(1e-12));
                }
            }
        }
    }
}

TEST_CASE("with no damper coefficient the step is the old arithmetic to the bit", "[physics][suspension][damper-integrator]")
{
    // The byte-inert control at the element: a corner with no damper slope — no curve, no
    // friction — runs the expression it always ran, character for character, whether or not a
    // stop is engaged.
    for (const auto previous : {-2.0, -0.3, 0.0, 1e-7, 0.25, 3.0})
    {
        for (const auto force : {-5000.0, -12.5, 0.0, 300.0, 9000.0})
        {
            for (const auto inertia : {1e-6, 0.5, 5.9, 7.3})
            {
                for (const auto stop : {0.0, 150.0, 3200.0})
                {
                    for (const auto dt : {1.0 / 120.0, 1.0 / 360.0, 1.0 / 2880.0})
                    {
                        const auto step = CornerRateStep{.previousRate = previous,
                                                         .generalisedForce = force,
                                                         .generalisedInertia = inertia,
                                                         .damperCoefficient = 0.0,
                                                         .stopCoefficient = stop,
                                                         .deltaTime = dt};
                        REQUIRE(solveCornerRate(step) == oldCornerRate(step));
                    }
                }
            }
        }
    }
}

TEST_CASE("the whole car applies the damper force its step solved, and the chassis reads that acceleration",
          "[physics][suspension][damper-integrator]")
{
    // The production bookkeeping, on the car: after a tick, `forces.damper` is the linearised
    // force at the velocity the step arrived at — `F(v₀) + F'(v₀)(v₁ − v₀)` on the shaft — and the
    // generalised force the chassis reaction reads is the corner's own discrete acceleration,
    // `I (q̇₁ − q̇₀)/dt`, exactly. Checked on every corner of every free tick of the settle drop every
    // fixture in this project starts with (`VehicleTests.cpp`, "the car settles on its springs"):
    // half a metre onto the flat plate. The wheels hang at the droop limit through the fall, the
    // landing runs the fronts onto their bump stops and all four onto their droop stops, and the
    // settle is the ordinary corner — so both stop branches and the plain step are in the sample.
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    REQUIRE(raceengine::bringUpJolt().has_value());

    auto descriptor = raceengine::ProvingGroundDescriptor{};
    descriptor.length = 120.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 1.0;
    descriptor.features = {};
    const auto world = raceengine::PhysicsWorld::create(raceengine::generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = raceengine::VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 1.02, 20.0);

    constexpr auto dt = 1.0 / 360.0;
    auto worstForce = 0.0;
    auto worstAcceleration = 0.0;
    auto largestShare = 0.0;
    auto stopTicks = 0;
    auto clampedTicks = 0;
    auto checkedTicks = 0;

    for (auto tick = 0; tick < 2160; tick++)
    {
        auto before = std::array<double, raceengine::cornerCount>{};
        for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
        {
            before[index] = state.corners[index].wishboneRate;
        }

        const auto stepped = raceengine::stepVehicle(golf.value(), state, raceengine::VehicleInput{},
                                                     raceengine::noDriveTorque, world.value(), dt);
        REQUIRE(stepped.has_value());

        for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
        {
            const auto& corner = golf->corners[index];
            const auto& solution = stepped->corners[index];
            const auto damper = solveDamperForce(corner, solution.suspension, before[index]);
            const auto jacobian = damper.lengthPerAngle;
            const auto slope = damperDampingCoefficient(corner, damper) / (jacobian * jacobian);
            const auto after = state.corners[index].wishboneRate;
            const auto v0 = damper.velocity;
            const auto v1 = -jacobian * after;

            REQUIRE(solution.damperVelocity == v0);
            stopTicks += (solution.forces.bumpStop != 0.0 || solution.forces.droopStop != 0.0) ? 1 : 0;

            // Pass three holds the wishbone inside the linkage's own range and zeroes the rate into
            // a limit it has reached; on those ticks the rate the state carries is not the rate the
            // step solved, and a wheel hanging in the air sits on that limit. Counted and skipped.
            const auto angle = state.corners[index].wishboneAngle;
            if (angle <= corner.hardpoints.droopAngle || angle >= corner.hardpoints.bumpAngle)
            {
                clampedTicks++;
                continue;
            }
            checkedTicks++;

            const auto expected = damper.force + slope * (v1 - v0);
            worstForce = std::max(worstForce, std::abs(solution.forces.damper - expected));
            largestShare = std::max(largestShare, std::abs(expected - damper.force));

            const auto acceleration = solution.generalisedForce / solution.generalisedInertia;
            worstAcceleration = std::max(worstAcceleration, std::abs(acceleration * dt - (after - before[index])));
        }
    }

    raceengine::tearDownJolt();

    std::printf("\n=== the settle drop, 6 s: |forces.damper − linearised| worst %.3e N, |a dt − dq̇| worst %.3e rad/s, "
                "largest implicit share %.1f N, corner-ticks on a stop %d, checked %d, at a linkage limit %d; "
                "chassis y at the end %.4f m ===\n",
                worstForce, worstAcceleration, largestShare, stopTicks, checkedTicks, clampedTicks,
                state.chassis.position.y);

    REQUIRE(worstForce < 1e-6);
    REQUIRE(worstAcceleration < 1e-9);
    REQUIRE(largestShare > 10.0);
    REQUIRE(stopTicks > 100);
    REQUIRE(checkedTicks > 6000);
    REQUIRE(state.chassis.position.y == Catch::Approx(0.5527).margin(0.01));
}

TEST_CASE("the damper-free car, bit for bit", "[.damper-integrator-control]")
{
    // The byte-inert control on the whole car, for reading across two builds: the Golf with every
    // damper curve emptied and every friction zeroed, dropped from 0.52 m onto a plate, its state
    // printed as hexadecimal floats at three ticks — and the shipped car beside it, which the
    // change moves. Run on the build before the change and on the build after; the first block must
    // not differ by a character, and the second must.
    const auto golf = golfGtiMk7();
    REQUIRE(golf.has_value());
    REQUIRE(raceengine::bringUpJolt().has_value());

    auto descriptor = raceengine::ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};
    const auto world = raceengine::PhysicsWorld::create(raceengine::generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    for (const auto damped : {false, true})
    {
        auto setup = golf.value();
        if (!damped)
        {
            for (auto& corner : setup.corners)
            {
                corner.damper = Curve{};
                corner.damperFriction = 0.0;
            }
        }

        auto state = raceengine::VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 0.0);

        std::printf("\n=== %s, dropped from 0.52 m: state bits ===\n", damped ? "the shipped dampers" : "no damper at all");
        for (auto tick = 1; tick <= 1080; tick++)
        {
            const auto stepped = raceengine::stepVehicle(setup, state, raceengine::VehicleInput{},
                                                         raceengine::noDriveTorque, world.value(), 1.0 / 360.0);
            REQUIRE(stepped.has_value());

            if (tick == 90 || tick == 360 || tick == 1080)
            {
                std::printf("  tick %4d chassis y %a vy %a pitch %a\n", tick, state.chassis.position.y,
                            state.chassis.linearVelocity.y, stepped->telemetry.pitch);
                for (auto index = std::size_t{0}; index < raceengine::cornerCount; index++)
                {
                    std::printf("    %s angle %a rate %a damper %a\n",
                                raceengine::cornerAbbreviation(static_cast<raceengine::Corner>(index)),
                                state.corners[index].wishboneAngle, state.corners[index].wishboneRate,
                                stepped->corners[index].forces.damper);
                }
            }
        }
    }

    raceengine::tearDownJolt();
}
