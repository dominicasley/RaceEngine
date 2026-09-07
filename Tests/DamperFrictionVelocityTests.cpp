// Damper friction against shaft speed: the invariants of the capability, and the proof it is inert.
//
// The shipped law is a magnitude and a regularised sign — `friction · tanh(v / width)` — which is
// flat above a few widths. The measurement it came from is not flat: Deubel et al.'s steady-state
// curves for the same strut rise to a maximum near 5 mm/s and then fall by about a third out to
// 300 mm/s. `CornerSetup::damperFrictionShape` is the seam for stating that, and
// `macPhersonStrutFrictionShape()` is the one sourced curve to put in it.
//
// **No car states a shape**, which is asserted below rather than asserted about. These cases hold
// four things: the shipped path is bit-identical with the seam present; the shape does not break the
// physics a friction term has to obey; the sourced curve says what the source says; and the setup
// sheet's key reaches one axle and no other.
//
// The dynamic behaviour — energy, zero crossings, what it is worth against the viscous damper — is
// measured in `DamperFrictionVelocityProbe.cpp`, hidden behind `[.damper-friction-velocity]`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::applyVehicleTune;
using raceengine::CornerSetup;
using raceengine::Curve;
using raceengine::damperDampingCoefficient;
using raceengine::damperFrictionReferenceSpeed;
using raceengine::damperFrictionShapeAt;
using raceengine::golfGtiMk7;
using raceengine::macPhersonStrutFrictionShape;
using raceengine::parseVehicleTune;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperForce;
using raceengine::SuspensionState;

namespace
{

// The velocities every case sweeps, metres per second: the sourced curve's whole measured range and
// past both ends of it, with the signs mirrored and zero included.
constexpr auto sweep = std::array{-0.500, -0.300, -0.200, -0.100, -0.050, -0.020, -0.010, -0.005, -0.001, 0.0,
                                  0.001,  0.005,  0.010,  0.020,  0.050,  0.100,  0.200,  0.300,  0.500};

// A solved front corner at its design position, which is the one place both travel stops are exactly
// zero by construction — the whole fixture rule of this task's stop-contamination clause.
[[nodiscard]] SuspensionState designPosition(const CornerSetup& corner)
{
    const auto solved = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
    REQUIRE(solved.has_value());

    return *solved;
}

// The wishbone rate that puts a wanted velocity on the shaft: the force path takes
// `velocity = −lengthPerAngle · rate`, so this inverts exactly that.
[[nodiscard]] double rateForVelocity(const CornerSetup& corner, const SuspensionState& state, const double velocity)
{
    const auto probe = solveDamperForce(corner, state, 1.0);
    REQUIRE(std::abs(probe.lengthPerAngle) > 1e-9);

    return -velocity / probe.lengthPerAngle;
}

// Friction alone: the whole damper force less the viscous curve at the same velocity. This is the
// subtraction every measurement in this file and its probe is built on, and it is exact rather than
// modelled — the two terms are added in one expression and nothing else is in it.
[[nodiscard]] double frictionOnly(const CornerSetup& corner, const SuspensionState& state, const double velocity)
{
    const auto damper = solveDamperForce(corner, state, rateForVelocity(corner, state, velocity));

    return damper.force - corner.damper.at(damper.velocity);
}

} // namespace

TEST_CASE("no car states a damper friction velocity shape", "[physics][suspension][damper-friction]")
{
    // The inertness proof, and it is a property of the data rather than of a flag: with every
    // corner's shape empty, `solveDamperForce` takes the branch it has always taken. A car that
    // states one would be a car whose damper friction moved, and this is what says none does.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (const auto& corner : built->corners)
    {
        CHECK(corner.damperFrictionShape.points.empty());
        CHECK(corner.damperFriction > 0.0);
    }

    // And the placeholder car, because a default that quietly gained a shape would state one on
    // every car built from it.
    const auto placeholder = CornerSetup{};
    CHECK(placeholder.damperFrictionShape.points.empty());
}

TEST_CASE("an absent shape is exactly the shipped friction law", "[physics][suspension][damper-friction]")
{
    // Bit equality, on every corner of the shipped car, at every velocity in the sweep and at four
    // positions across the travel. Not `Approx`: the claim is that the expression is unchanged, and
    // an epsilon would let a reordering through that thirty seconds of launch amplifies into a
    // parity failure.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (const auto& corner : built->corners)
    {
        const auto smoothing = std::max(corner.damperFrictionSpeed, 1e-9);

        for (const auto through : {0.75 * corner.hardpoints.droopAngle, 0.0, 0.4 * corner.hardpoints.bumpAngle,
                                   0.9 * corner.hardpoints.bumpAngle})
        {
            const auto state = solveCornerWithJacobian(corner.hardpoints, through, 0.0);
            REQUIRE(state.has_value());

            for (const auto velocity : sweep)
            {
                const auto rate = rateForVelocity(corner, *state, velocity);
                const auto damper = solveDamperForce(corner, *state, rate);

                const auto expected =
                    corner.damper.at(damper.velocity) + corner.damperFriction * std::tanh(damper.velocity / smoothing);

                REQUIRE(damper.force == expected);
            }
        }
    }
}

TEST_CASE("the shape multiplies the magnitude and does not restate it", "[physics][suspension][damper-friction]")
{
    // What the architecture is for: one number carries how big the friction is and one curve carries
    // how it varies, and the curve is 1.0 at the velocity the number is quoted at. So installing the
    // shape must not move the friction at that velocity — if it did, the shape would be a second
    // magnitude and the two sources would be tangled.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto shaped = built->corners[0];
    shaped.damperFrictionShape = macPhersonStrutFrictionShape();

    const auto state = designPosition(built->corners[0]);

    const auto flat = frictionOnly(built->corners[0], state, damperFrictionReferenceSpeed);
    const auto curved = frictionOnly(shaped, state, damperFrictionReferenceSpeed);

    CHECK(curved == Catch::Approx(flat).epsilon(1e-12));

    // And it scales with the magnitude rather than replacing it: double the newtons, double the
    // force, at any velocity.
    auto doubled = shaped;
    doubled.damperFriction *= 2.0;

    for (const auto velocity : sweep)
    {
        if (velocity == 0.0)
        {
            continue;
        }

        CHECK(frictionOnly(doubled, state, velocity) ==
              Catch::Approx(2.0 * frictionOnly(shaped, state, velocity)).epsilon(1e-12));
    }
}

TEST_CASE("shaped damper friction obeys the invariants of a friction force", "[physics][suspension][damper-friction]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto shaped = built->corners[0];
    shaped.damperFrictionShape = macPhersonStrutFrictionShape();

    const auto state = designPosition(shaped);

    for (const auto velocity : sweep)
    {
        const auto rate = rateForVelocity(shaped, state, velocity);
        const auto damper = solveDamperForce(shaped, state, rate);
        const auto friction = damper.force - shaped.damper.at(damper.velocity);

        // Finite everywhere, including at both ends of the sweep, which are past the sourced curve's
        // measured range at both ends.
        REQUIRE(std::isfinite(friction));
        REQUIRE(std::isfinite(damper.force));

        // Opposing the motion, so the power it takes out is never negative-of-negative: friction
        // power is `−F·v` and must never be positive. Stated at exactly zero velocity too, where the
        // force must be exactly zero rather than merely small.
        if (damper.velocity == 0.0)
        {
            REQUIRE(friction == 0.0);
        }
        else
        {
            REQUIRE(friction * damper.velocity > 0.0);
        }

        // Bounded by the magnitude times the sourced curve's own maximum. A shape that could exceed
        // its own peak would be an interpolation fault, not a measurement.
        REQUIRE(std::abs(friction) <= shaped.damperFriction * 1.31);
    }

    // Mirror symmetry, exactly — the source's rebound-to-compression eccentricity does not exceed
    // 12 % and the model states none at all, so the two directions must be the same number with
    // opposite signs and not merely close.
    //
    // Fed the **wishbone rate** and its exact negation rather than a velocity round-tripped through
    // the Jacobian twice: `rate` and `-rate` are the same bits with one flipped, so any asymmetry
    // that shows up here is the force law's and not the fixture's.
    //
    // The friction itself is symmetric **by construction** — an odd `tanh` times a shape read at
    // `|velocity|` — and that half is asserted exactly, on the shape. What cannot be asserted to the
    // bit is the *recovered* friction, because recovering it means subtracting a viscous force whose
    // two directions are genuinely different numbers (this damper is kneed and asymmetric), and
    // `(a + b) - a` is not `b`. So the recovered pair is held to a relative 1e-12, which is a
    // statement about the subtraction and not about the law.
    for (const auto velocity : sweep)
    {
        const auto rate = rateForVelocity(shaped, state, velocity);

        const auto forward = solveDamperForce(shaped, state, rate);
        const auto backward = solveDamperForce(shaped, state, -rate);

        REQUIRE(backward.velocity == -forward.velocity);
        REQUIRE(damperFrictionShapeAt(shaped.damperFrictionShape, std::abs(backward.velocity)) ==
                damperFrictionShapeAt(shaped.damperFrictionShape, std::abs(forward.velocity)));

        const auto reboundFriction = backward.force - shaped.damper.at(backward.velocity);
        const auto bumpFriction = forward.force - shaped.damper.at(forward.velocity);

        REQUIRE(reboundFriction == Catch::Approx(-bumpFriction).epsilon(1e-12).margin(1e-9));
    }

    // Zero magnitude is exactly zero force whatever the shape says, because a shape multiplying
    // nothing is nothing. This is the guard against a shape becoming a friction source of its own.
    auto silent = shaped;
    silent.damperFriction = 0.0;

    for (const auto velocity : sweep)
    {
        const auto rate = rateForVelocity(silent, state, velocity);
        const auto damper = solveDamperForce(silent, state, rate);

        REQUIRE(damper.force == silent.damper.at(damper.velocity));
    }
}

TEST_CASE("the shaped friction is continuous through zero", "[physics][suspension][damper-friction]")
{
    // The property the `tanh` exists for, held with the shape on top of it: no step at the crossing,
    // and no epsilon anywhere that a small change in velocity can fall either side of and move the
    // force materially. Walked at a hundredth of the regularisation width, which is a fortieth of
    // what one 360 Hz tick of a settling corner covers.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    auto shaped = built->corners[0];
    shaped.damperFrictionShape = macPhersonStrutFrictionShape();

    const auto state = designPosition(shaped);
    const auto step = shaped.damperFrictionSpeed / 100.0;

    auto previous = frictionOnly(shaped, state, -20.0 * step);
    auto largest = 0.0;

    for (auto index = -19; index <= 20; index++)
    {
        const auto here = frictionOnly(shaped, state, static_cast<double>(index) * step);
        largest = std::max(largest, std::abs(here - previous));
        previous = here;
    }

    // One step of a hundredth of the width may move the force by at most a hundredth of the
    // magnitude times a small factor. The bound is the analytic one — `f/w` is the steepest slope
    // the term has — rather than a number read off this run.
    CHECK(largest < 0.05 * shaped.damperFriction);

    // And the implicit damping coefficient stays finite and non-negative across the same walk, which
    // is what the integration divides by.
    for (auto index = -20; index <= 20; index++)
    {
        const auto velocity = static_cast<double>(index) * step;
        const auto damper = solveDamperForce(shaped, state, rateForVelocity(shaped, state, velocity));
        const auto coefficient = damperDampingCoefficient(shaped, damper);

        REQUIRE(std::isfinite(coefficient));
        REQUIRE(coefficient >= 0.0);
    }
}

TEST_CASE("the sourced strut shape states what the source states", "[physics][suspension][damper-friction]")
{
    const auto shape = macPhersonStrutFrictionShape();

    // Ascending in x, which `Curve::at` requires and nothing else checks.
    for (auto index = std::size_t{1}; index < shape.points.size(); index++)
    {
        REQUIRE(shape.points[index].x > shape.points[index - 1].x);
    }

    // Normalised at the velocity the magnitude is quoted at.
    REQUIRE(damperFrictionShapeAt(shape, damperFrictionReferenceSpeed) == Catch::Approx(1.0).epsilon(1e-12));

    // The three things the source's own prose says, as three assertions rather than as a comment.
    //
    // It rises from the quasi-static value to a maximum — *"friction initially increases with
    // increasing velocity, reaches a maximum, and then decreases"* — and the maximum is inside the
    // first 10 mm/s.
    const auto peak = damperFrictionShapeAt(shape, 0.005);
    CHECK(peak > 1.25);
    CHECK(peak < 1.35);
    CHECK(damperFrictionShapeAt(shape, 0.005) > damperFrictionShapeAt(shape, damperFrictionReferenceSpeed));

    // It then falls monotonically, all the way to the source's 300 mm/s cap.
    auto previous = peak;
    for (const auto velocity : {0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300})
    {
        const auto here = damperFrictionShapeAt(shape, velocity);
        REQUIRE(here < previous);
        previous = here;
    }

    // And it ends about a third below the quasi-static value, which is the size of the effect the
    // shipped flat law has none of.
    CHECK(damperFrictionShapeAt(shape, 0.300) == Catch::Approx(0.692).epsilon(1e-9));

    // Both ends are held rather than extrapolated: below the source's slowest sliding velocity and
    // above its cap the curve says the last measured thing and does not invent a trend.
    CHECK(damperFrictionShapeAt(shape, 0.0) == shape.points.front().y);
    CHECK(damperFrictionShapeAt(shape, 1.0) == shape.points.back().y);
    CHECK(damperFrictionShapeAt(shape, 5.0) == shape.points.back().y);

    // An empty shape is exactly one and not `Curve::at`'s empty-curve zero, which would delete the
    // friction instead of leaving it alone. This is the whole of the inertness.
    const auto none = Curve{};
    for (const auto velocity : {0.0, 0.0005, 0.05, 5.0})
    {
        REQUIRE(damperFrictionShapeAt(none, velocity) == 1.0);
    }
}

TEST_CASE("the setup sheet installs the shape on one axle only", "[physics][suspension][damper-friction]")
{
    auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto tune = parseVehicleTune("front.frictionshape 1\n");
    REQUIRE(tune.has_value());

    auto tuned = built.value();
    applyVehicleTune(*tune, tuned);

    CHECK_FALSE(tuned.corners[0].damperFrictionShape.points.empty());
    CHECK_FALSE(tuned.corners[1].damperFrictionShape.points.empty());
    CHECK(tuned.corners[2].damperFrictionShape.points.empty());
    CHECK(tuned.corners[3].damperFrictionShape.points.empty());

    // The magnitude is untouched by the shape key, which is the separation the architecture is for.
    for (auto index = std::size_t{0}; index < tuned.corners.size(); index++)
    {
        CHECK(tuned.corners[index].damperFriction == built->corners[index].damperFriction);
    }

    // Zero is the way back and is not "install it": a sheet that states the key off must leave the
    // corner on the flat law, or there is no A/B.
    const auto off = parseVehicleTune("front.frictionshape 0\n");
    REQUIRE(off.has_value());

    auto reverted = built.value();
    applyVehicleTune(*off, reverted);

    for (const auto& corner : reverted.corners)
    {
        CHECK(corner.damperFrictionShape.points.empty());
    }
}
