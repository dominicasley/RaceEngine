#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::computeRollCentre;
using raceengine::CornerHardpoints;
using raceengine::CornerSide;
using raceengine::intersectSphereWithCircle;
using raceengine::outboardSign;
using raceengine::solveCorner;
using raceengine::solveCornerWithJacobian;
using raceengine::SuspensionKind;
using raceengine::SuspensionState;
using raceengine::sweepCorner;
using raceengine::validateCorner;
using raceengine::Wishbone;

namespace
{

// A short-long-arm front corner: upper wishbone shorter than the lower, which is what gives camber
// gain in bump. **Placeholder geometry** for a mid-size car, not measured from any vehicle — the
// numbers are plausible and internally consistent, and they are here to exercise the solver rather
// than to describe a car. Real hardpoints replace them wholesale.
CornerHardpoints frontCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * x, y, z);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;

    corner.lower = Wishbone{
        .frontPivot = at(0.30, 0.13, 0.15), .rearPivot = at(0.30, 0.13, -0.15), .ballJoint = at(0.62, 0.12, 0.0)};
    corner.upper = Wishbone{
        .frontPivot = at(0.35, 0.36, 0.12), .rearPivot = at(0.35, 0.36, -0.12), .ballJoint = at(0.58, 0.42, 0.0)};

    corner.damperChassis = at(0.32, 0.55, 0.0);
    corner.damperWishbone = at(0.50, 0.14, 0.0);
    corner.antiRollBarChassis = at(0.45, 0.30, 0.25);
    corner.antiRollBarWishbone = at(0.45, 0.13, 0.25);
    corner.steeringRackOuter = at(0.30, 0.16, 0.16);
    corner.steeringArm = at(0.60, 0.16, 0.14);
    corner.wheelCentre = at(0.72, 0.30, 0.0);
    corner.wheelRadius = 0.31;

    corner.droopAngle = -0.16;
    corner.bumpAngle = 0.16;

    return corner;
}

// A MacPherson strut front corner. **Placeholder geometry** again — proportioned like a front-drive
// hatchback's, with round numbers of its own — and here to exercise the second linkage rather than
// to describe a car.
CornerHardpoints strutCorner(const CornerSide side)
{
    const auto mirror = outboardSign(side);
    const auto at = [mirror](const double x, const double y, const double z)
    {
        return glm::dvec3(mirror * x, y, z);
    };

    auto corner = CornerHardpoints{};
    corner.side = side;
    corner.kind = SuspensionKind::MacPhersonStrut;

    corner.lower = Wishbone{
        .frontPivot = at(0.36, 0.22, 0.15), .rearPivot = at(0.36, 0.21, -0.14), .ballJoint = at(0.70, 0.19, 0.01)};

    // High in the tower, well inboard of the ball joint: that offset is the kingpin inclination.
    corner.strutTop = at(0.52, 0.88, -0.04);

    corner.antiRollBarChassis = at(0.45, 0.30, 0.25);
    corner.antiRollBarWishbone = at(0.45, 0.22, 0.25);
    corner.steeringRackOuter = at(0.46, 0.26, -0.22);
    corner.steeringArm = at(0.72, 0.26, -0.13);
    corner.wheelCentre = at(0.77, 0.33, 0.0);
    corner.wheelRadius = 0.33;

    corner.droopAngle = -0.16;
    corner.bumpAngle = 0.16;

    return corner;
}

} // namespace

TEST_CASE("a sphere meets a circle in at most two places", "[physics][suspension]")
{
    const auto normal = glm::dvec3(0.0, 0.0, 1.0);

    SECTION("two roots, mirrored about the line of centres")
    {
        const auto found = intersectSphereWithCircle(glm::dvec3(1.0, 0.0, 0.0), 1.0, glm::dvec3(0.0), 1.0, normal);

        REQUIRE(found.count == 2);
        REQUIRE(found.first.x == Catch::Approx(0.5));
        REQUIRE(found.second.x == Catch::Approx(0.5));
        REQUIRE(found.first.y == Catch::Approx(-found.second.y));
        REQUIRE(std::abs(found.first.y) == Catch::Approx(std::sqrt(3.0) / 2.0));
    }

    SECTION("a sphere too far away meets nothing")
    {
        REQUIRE(intersectSphereWithCircle(glm::dvec3(5.0, 0.0, 0.0), 1.0, glm::dvec3(0.0), 1.0, normal).count == 0);
    }

    SECTION("a sphere that cannot reach the plane meets nothing")
    {
        REQUIRE(intersectSphereWithCircle(glm::dvec3(0.0, 0.0, 5.0), 1.0, glm::dvec3(0.0), 1.0, normal).count == 0);
    }

    SECTION("concentric is reported as no solution rather than as infinitely many")
    {
        REQUIRE(intersectSphereWithCircle(glm::dvec3(0.0), 1.0, glm::dvec3(0.0), 1.0, normal).count == 0);
    }
}

TEST_CASE("the linkage solves to its own design position at zero", "[physics][suspension]")
{
    // The first thing to be sure of: with no travel and no steering, the solver must reproduce the
    // hardpoints it was given. Anything else means a convention is inverted somewhere, and every
    // curve below would be measured from the wrong place.
    for (const auto side : {CornerSide::Left, CornerSide::Right})
    {
        const auto corner = frontCorner(side);
        const auto solved = solveCorner(corner, 0.0, 0.0);

        REQUIRE(solved.has_value());
        REQUIRE(glm::distance(solved->lowerBallJoint, corner.lower.ballJoint) < 1e-12);
        REQUIRE(glm::distance(solved->upperBallJoint, corner.upper.ballJoint) < 1e-9);
        REQUIRE(glm::distance(solved->steeringArm, corner.steeringArm) < 1e-9);
        REQUIRE(glm::distance(solved->wheelCentre, corner.wheelCentre) < 1e-9);
        REQUIRE(solved->wheelTravel == Catch::Approx(0.0).margin(1e-9));
        REQUIRE(solved->camber == Catch::Approx(0.0).margin(1e-9));
        REQUIRE(solved->toe == Catch::Approx(0.0).margin(1e-9));
    }
}

TEST_CASE("positive travel compresses the corner on both sides", "[physics][suspension]")
{
    // The sign of the wishbone angle is normalised inside the solver, because taken as authored it
    // depends both on which pivot was called "front" and on which side of the car this is. A left
    // corner that droops where the right one bumps is the kind of mistake that reads as a roll
    // instability much later.
    for (const auto side : {CornerSide::Left, CornerSide::Right})
    {
        const auto corner = frontCorner(side);

        const auto bump = solveCorner(corner, 0.1, 0.0);
        const auto droop = solveCorner(corner, -0.1, 0.0);

        REQUIRE(bump.has_value());
        REQUIRE(droop.has_value());
        REQUIRE(bump->wheelTravel > 0.0);
        REQUIRE(droop->wheelTravel < 0.0);
    }
}

TEST_CASE("the geometry produces its own camber curve", "[physics][suspension]")
{
    const auto corner = frontCorner(CornerSide::Right);

    const auto sweep = sweepCorner(corner, 41);
    REQUIRE(sweep.has_value());

    const auto& samples = sweep->samples;
    const auto& atDroop = samples.front();
    const auto& atBump = samples.back();

    // Short upper arm, long lower arm: the top of the wheel is pulled inboard as the corner
    // compresses. Nothing authored this — move the upper ball joint and it changes.
    REQUIRE(atBump.camber < -0.005);
    REQUIRE(atDroop.camber > atBump.camber);

    // And it is a *curve*, not a slope: the camber gain per millimetre is not the same at both ends,
    // which is the whole reason for solving the linkage rather than storing a gradient.
    const auto lowerHalf =
        (samples[20].camber - samples[0].camber) / (samples[20].wheelTravel - samples[0].wheelTravel);
    const auto upperHalf =
        (samples[40].camber - samples[20].camber) / (samples[40].wheelTravel - samples[20].wheelTravel);

    REQUIRE(std::abs(upperHalf - lowerHalf) > 0.05 * std::abs(lowerHalf));

    // Travel is monotonic across the range and covers something like a real suspension's.
    REQUIRE(atBump.wheelTravel - atDroop.wheelTravel > 0.05);
    for (auto index = std::size_t{1}; index < samples.size(); index++)
    {
        REQUIRE(samples[index].wheelTravel > samples[index - 1].wheelTravel);
    }
}

TEST_CASE("the motion ratio varies across the travel", "[physics][suspension]")
{
    // If it did not, a constant would have done and the whole live solve would be waste. It does
    // not, and the variation is what makes a rising-rate suspension rise.
    const auto corner = frontCorner(CornerSide::Right);

    const auto sweep = sweepCorner(corner, 41);
    REQUIRE(sweep.has_value());

    // The ratio is the damper element's, read at the sweep's own angles — the state stopped
    // carrying a copy when the legacy damper path was retired (step 14).
    const auto element = raceengine::damperElementOf(corner);
    auto lowest = 1e30;
    auto highest = -1e30;
    for (const auto& sample : sweep->samples)
    {
        const auto kinematics = raceengine::solveDamperKinematics(corner, element, sample.wishboneAngle);
        REQUIRE(kinematics.has_value());

        // The damper moves less than the wheel, because it picks up inboard of the ball joint.
        REQUIRE(std::abs(kinematics->motionRatio) > 0.1);
        REQUIRE(std::abs(kinematics->motionRatio) < 1.0);

        lowest = std::min(lowest, std::abs(kinematics->motionRatio));
        highest = std::max(highest, std::abs(kinematics->motionRatio));
    }

    REQUIRE(highest - lowest > 0.01);

    // The Jacobian must agree with the damper length it is the derivative of, everywhere and not
    // just on average — which is what catches a single bad sample rather than letting it hide in a
    // curve that is smooth either side of it. The secant between two samples is compared against
    // the mean of the tangents at its ends, which is the trapezoid rule and correct to second
    // order; comparing it against one endpoint's tangent would be off by the curvature and would
    // need a tolerance loose enough to admit a real defect.
    for (auto index = std::size_t{1}; index < sweep->samples.size(); index++)
    {
        const auto& before = sweep->samples[index - 1];
        const auto& after = sweep->samples[index];

        const auto lengthBefore = raceengine::solveElement(corner, element, before.wishboneAngle).length;
        const auto lengthAfter = raceengine::solveElement(corner, element, after.wishboneAngle).length;
        const auto ratioBefore = raceengine::solveDamperKinematics(corner, element, before.wishboneAngle);
        const auto ratioAfter = raceengine::solveDamperKinematics(corner, element, after.wishboneAngle);
        REQUIRE(ratioBefore.has_value());
        REQUIRE(ratioAfter.has_value());

        const auto secant = (lengthAfter - lengthBefore) / (after.wheelTravel - before.wheelTravel);
        const auto mean = 0.5 * (ratioBefore->motionRatio + ratioAfter->motionRatio);

        REQUIRE(secant == Catch::Approx(mean).epsilon(1e-4));
    }
}

TEST_CASE("steering turns the wheel and the geometry decides how much", "[physics][suspension]")
{
    const auto corner = frontCorner(CornerSide::Right);

    const auto centred = solveCorner(corner, 0.0, 0.0);
    const auto steered = solveCorner(corner, 0.0, 0.03);
    const auto other = solveCorner(corner, 0.0, -0.03);

    REQUIRE(centred.has_value());
    REQUIRE(steered.has_value());
    REQUIRE(other.has_value());

    // The rack moves the wheel, and moving it the other way steers the other way by a similar
    // amount — the asymmetry between them is Ackermann, not a bug.
    REQUIRE(std::abs(steered->toe) > 0.02);
    REQUIRE(steered->toe * other->toe < 0.0);
    REQUIRE(std::abs(std::abs(steered->toe) - std::abs(other->toe)) < 0.5 * std::abs(steered->toe));

    // Steering rotates the upright about the kingpin, so it must not move the ball joints at all:
    // they are set by the wishbones alone.
    REQUIRE(glm::distance(steered->lowerBallJoint, centred->lowerBallJoint) < 1e-12);
    REQUIRE(glm::distance(steered->upperBallJoint, centred->upperBallJoint) < 1e-9);
}

TEST_CASE("bump steer is whatever the geometry says and is reported either way", "[physics][suspension]")
{
    // Bump steer emerging from the linkage is correct and desirable. Bump steer emerging from a
    // typo is a car that steers itself over every bump, and the two are indistinguishable in the
    // final frame — which is why the sweep exists and why this case measures rather than demands.
    const auto corner = frontCorner(CornerSide::Right);

    const auto sweep = sweepCorner(corner, 41);
    REQUIRE(sweep.has_value());

    auto worst = 0.0;
    for (const auto& sample : sweep->samples)
    {
        worst = std::max(worst, std::abs(sample.toe));
    }

    // A degree and a half across the full travel is a lot but not absurd for a road car; well past
    // this and the geometry is wrong rather than merely lively.
    REQUIRE(worst < 0.026);
}

TEST_CASE("the left corner is the mirror of the right", "[physics][suspension]")
{
    // Mirrored hardpoints must give mirrored answers, and the two easiest ways to break that are a
    // camber sign read off the wrong axis and a swing direction that does not flip with the side.
    const auto right = frontCorner(CornerSide::Right);
    const auto left = frontCorner(CornerSide::Left);

    for (const auto angle : {-0.1, 0.0, 0.08})
    {
        const auto onRight = solveCorner(right, angle, 0.0);
        const auto onLeft = solveCorner(left, angle, 0.0);

        REQUIRE(onRight.has_value());
        REQUIRE(onLeft.has_value());

        REQUIRE(onLeft->wheelCentre.x == Catch::Approx(-onRight->wheelCentre.x).margin(1e-9));
        REQUIRE(onLeft->wheelCentre.y == Catch::Approx(onRight->wheelCentre.y).margin(1e-9));
        REQUIRE(onLeft->wheelTravel == Catch::Approx(onRight->wheelTravel).margin(1e-9));
        // Camber and toe are signed relative to the car, so a mirrored corner reads the same, not
        // the opposite.
        REQUIRE(onLeft->camber == Catch::Approx(onRight->camber).margin(1e-9));
        REQUIRE(onLeft->toe == Catch::Approx(onRight->toe).margin(1e-9));

        // The damper length mirrors too, read off each side's own element (step 14).
        const auto leftLength = raceengine::solveElement(left, raceengine::damperElementOf(left), angle).length;
        const auto rightLength = raceengine::solveElement(right, raceengine::damperElementOf(right), angle).length;
        REQUIRE(leftLength == Catch::Approx(rightLength).margin(1e-12));
    }
}

TEST_CASE("the solve stays on one branch across the whole travel", "[physics][suspension]")
{
    // Each intersection has a mirrored root that is a perfectly valid linkage and an entirely wrong
    // car. Continuity is what keeps the solve on the right one, and a flip shows as the upright
    // teleporting while the angle barely moved.
    const auto corner = frontCorner(CornerSide::Right);

    const auto sweep = sweepCorner(corner, 201);
    REQUIRE(sweep.has_value());

    for (auto index = std::size_t{1}; index < sweep->samples.size(); index++)
    {
        REQUIRE(glm::distance(sweep->samples[index].upperBallJoint, sweep->samples[index - 1].upperBallJoint) < 0.01);
        REQUIRE(glm::distance(sweep->samples[index].wheelCentre, sweep->samples[index - 1].wheelCentre) < 0.01);
    }

    REQUIRE(validateCorner(corner).has_value());
}

TEST_CASE("geometry that cannot do its stated travel is refused at the point it is read", "[physics][suspension]")
{
    SECTION("a travel range past the linkage's reach")
    {
        auto corner = frontCorner(CornerSide::Right);
        corner.droopAngle = -2.0;
        corner.bumpAngle = 2.0;

        const auto validated = validateCorner(corner);
        REQUIRE_FALSE(validated.has_value());
    }

    SECTION("a stop pair the wrong way round")
    {
        auto corner = frontCorner(CornerSide::Right);
        corner.droopAngle = 0.16;
        corner.bumpAngle = -0.16;

        REQUIRE_FALSE(validateCorner(corner).has_value());
    }

    SECTION("a wishbone whose ball joint sits on its own pivot axis has no swing")
    {
        auto corner = frontCorner(CornerSide::Right);
        // Mirrored with the corner it is being placed on, or it is no longer on the axis and the
        // case stops being degenerate at all.
        corner.lower.ballJoint = glm::dvec3(outboardSign(CornerSide::Right) * 0.30, 0.13, 0.0);

        REQUIRE_FALSE(solveCorner(corner, 0.0, 0.0).has_value());
    }
}

TEST_CASE("the roll centre is an output and it migrates", "[physics][suspension]")
{
    const auto corner = frontCorner(CornerSide::Right);

    const auto sweep = sweepCorner(corner, 41);
    REQUIRE(sweep.has_value());

    const auto& atDroop = sweep->samples.front();
    const auto& atBump = sweep->samples.back();

    REQUIRE(atDroop.instantCentreDefined);
    REQUIRE(atBump.instantCentreDefined);

    // It moves — which is the thing worth knowing about a roll centre and the reason it is computed
    // rather than authored. Nothing reads it back to compute a force.
    REQUIRE(std::abs(atBump.rollCentreHeight - atDroop.rollCentreHeight) > 0.005);

    SECTION("parallel equal wishbones put it on the ground, which is the textbook case")
    {
        auto parallel = frontCorner(CornerSide::Right);
        // Mirrored with the corner they are being placed on, like every other absolute coordinate
        // in this file: unmirrored they describe a linkage on the other side of the car and the
        // wishbones stop being parallel at all.
        const auto side = outboardSign(CornerSide::Right);
        parallel.upper.frontPivot = glm::dvec3(side * 0.30, 0.36, 0.15);
        parallel.upper.rearPivot = glm::dvec3(side * 0.30, 0.36, -0.15);
        parallel.upper.ballJoint = glm::dvec3(side * 0.62, 0.35, 0.0);

        auto solved = solveCorner(parallel, 0.0, 0.0);
        REQUIRE(solved.has_value());
        computeRollCentre(parallel, solved.value());

        REQUIRE_FALSE(solved->instantCentreDefined);
        REQUIRE(solved->rollCentreHeight == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("the solve is pure", "[physics][suspension][determinism]")
{
    // No clock, no global, no cached state between calls: the same corner at the same angle is the
    // same answer whatever was asked before it. Criterion 12 rests on this being true all the way
    // down.
    const auto corner = frontCorner(CornerSide::Right);

    const auto first = solveCornerWithJacobian(corner, 0.07, 0.012);
    static_cast<void>(solveCorner(corner, -0.15, -0.03));
    const auto second = solveCornerWithJacobian(corner, 0.07, 0.012);

    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(first->wheelCentre == second->wheelCentre);
    REQUIRE(first->camber == second->camber);
    REQUIRE(first->toe == second->toe);

    const auto element = raceengine::damperElementOf(corner);
    const auto once = raceengine::solveDamperKinematics(corner, element, 0.07, 0.012);
    const auto again = raceengine::solveDamperKinematics(corner, element, 0.07, 0.012);
    REQUIRE(once.has_value());
    REQUIRE(again.has_value());
    REQUIRE(once->motionRatio == again->motionRatio);
}

TEST_CASE("a strut is the same problem with one step replaced", "[physics][suspension][strut]")
{
    // The second linkage, behind the same function and the same output struct. What differs is one
    // step: a wishbone has to *find* its upper ball joint, and a strut's upper point is bolted to
    // the body and does not move at all.
    const auto corner = strutCorner(CornerSide::Right);

    SECTION("it solves to its own design position")
    {
        const auto solved = solveCorner(corner, 0.0, 0.0);
        REQUIRE(solved.has_value());

        REQUIRE(glm::distance(solved->lowerBallJoint, corner.lower.ballJoint) < 1e-12);
        REQUIRE(glm::distance(solved->wheelCentre, corner.wheelCentre) < 1e-9);
        REQUIRE(solved->camber == Catch::Approx(0.0).margin(1e-9));
        REQUIRE(solved->toe == Catch::Approx(0.0).margin(1e-9));
    }

    SECTION("the top of the kingpin does not move, whatever the wheel does")
    {
        for (const auto angle : {-0.14, -0.05, 0.0, 0.09, 0.15})
        {
            const auto solved = solveCorner(corner, angle, 0.0);
            REQUIRE(solved.has_value());
            REQUIRE(glm::distance(solved->upperBallJoint, corner.strutTop) < 1e-12);
        }
    }

    SECTION("the whole travel solves and is continuous")
    {
        REQUIRE(validateCorner(corner).has_value());

        const auto sweep = sweepCorner(corner, 121);
        REQUIRE(sweep.has_value());

        for (auto index = std::size_t{1}; index < sweep->samples.size(); index++)
        {
            REQUIRE(glm::distance(sweep->samples[index].wheelCentre, sweep->samples[index - 1].wheelCentre) < 0.01);
            REQUIRE(sweep->samples[index].wheelTravel > sweep->samples[index - 1].wheelTravel);
        }
    }
}

TEST_CASE("a strut's motion ratio is near one, and a wishbone's is not", "[physics][suspension][strut]")
{
    // The clearest structural difference between the two, and a good check that the strut is not
    // quietly being solved as a wishbone: the strut *is* the damper, running from its top bearing to
    // the lower ball joint, so the wheel and the damper move very nearly together. A wishbone car's
    // damper picks up inboard on the arm and moves about half as far.
    const auto strut = sweepCorner(strutCorner(CornerSide::Right), 41);
    const auto wishbone = sweepCorner(frontCorner(CornerSide::Right), 41);

    REQUIRE(strut.has_value());
    REQUIRE(wishbone.has_value());

    const auto ratioAt = [](const CornerHardpoints& hardpoints, const double angle)
    {
        const auto kinematics =
            raceengine::solveDamperKinematics(hardpoints, raceengine::damperElementOf(hardpoints), angle);
        REQUIRE(kinematics.has_value());
        return kinematics->motionRatio;
    };

    const auto strutHardpoints = strutCorner(CornerSide::Right);
    for (const auto& sample : strut->samples)
    {
        const auto ratio = ratioAt(strutHardpoints, sample.wishboneAngle);
        REQUIRE(std::abs(ratio) > 0.75);
        REQUIRE(std::abs(ratio) < 1.05);
    }

    const auto wishboneHardpoints = frontCorner(CornerSide::Right);
    for (const auto& sample : wishbone->samples)
    {
        REQUIRE(std::abs(ratioAt(wishboneHardpoints, sample.wishboneAngle)) < 0.70);
    }

    // And neither reverses, which is what says the damper is attached to something real.
    const auto sameSign = [&ratioAt](const CornerHardpoints& hardpoints, const auto& sweep)
    {
        auto sign = 0.0;
        for (const auto& sample : sweep.samples)
        {
            const auto here = ratioAt(hardpoints, sample.wishboneAngle) < 0.0 ? -1.0 : 1.0;
            if (sign != 0.0 && here != sign)
            {
                return false;
            }
            sign = here;
        }
        return true;
    };

    REQUIRE(sameSign(strutHardpoints, strut.value()));
    REQUIRE(sameSign(wishboneHardpoints, wishbone.value()));
}

TEST_CASE("steering a strut changes its camber", "[physics][suspension][strut]")
{
    // A strut's kingpin is steeply inclined — twelve to sixteen degrees on a road car — so turning
    // the upright about it tips the wheel as well as pointing it. That is why a strut car gains
    // camber on the outside wheel when it is steered, and it is the thing that would be missing if
    // the kingpin axis were being taken from the wrong pair of points.
    const auto corner = strutCorner(CornerSide::Right);

    const auto kingpin = glm::normalize(corner.strutTop - corner.lower.ballJoint);
    const auto inclination = std::acos(std::abs(kingpin.y));
    REQUIRE(inclination > 0.14); // eight degrees at the very least
    REQUIRE(inclination < 0.35);

    const auto centred = solveCorner(corner, 0.0, 0.0);
    const auto oneWay = solveCorner(corner, 0.0, 0.035);
    const auto theOther = solveCorner(corner, 0.0, -0.035);

    REQUIRE(centred.has_value());
    REQUIRE(oneWay.has_value());
    REQUIRE(theOther.has_value());

    REQUIRE(std::abs(oneWay->toe) > 0.08);
    REQUIRE(oneWay->toe * theOther->toe < 0.0);

    // Camber moves with steer, and moves *opposite* ways for opposite lock — which is the
    // signature. A wishbone corner with a near-vertical kingpin barely moves at all.
    REQUIRE(std::abs(oneWay->camber) > 0.004);
    REQUIRE(std::abs(theOther->camber) > 0.004);
    REQUIRE(oneWay->camber * theOther->camber < 0.0);

    const auto wishbone = frontCorner(CornerSide::Right);
    const auto steeredWishbone = solveCorner(wishbone, 0.0, 0.035);
    REQUIRE(steeredWishbone.has_value());
    REQUIRE(std::abs(steeredWishbone->camber) < std::abs(oneWay->camber));
}

TEST_CASE("a strut's roll centre uses the perpendicular, not the strut axis", "[physics][suspension][strut]")
{
    // The construction differs from a wishbone's and the difference is not small. A strut's top
    // bearing carries no side load along the strut, so its effective upper link is the line
    // *perpendicular* to the strut axis through that bearing. Treating the strut axis itself as the
    // link — which is what happens if a strut is solved as though it had an upper wishbone — puts
    // the instant centre somewhere else entirely.
    const auto corner = strutCorner(CornerSide::Right);

    auto solved = solveCorner(corner, 0.0, 0.0);
    REQUIRE(solved.has_value());
    computeRollCentre(corner, solved.value());

    REQUIRE(solved->instantCentreDefined);
    // Above ground and below the wheel centre, which is where a front strut's belongs.
    REQUIRE(solved->rollCentreHeight > 0.0);
    REQUIRE(solved->rollCentreHeight < corner.wheelCentre.y);

    SECTION("and it migrates across the travel")
    {
        const auto sweep = sweepCorner(corner, 41);
        REQUIRE(sweep.has_value());

        auto atDroop = sweep->samples.front();
        auto atBump = sweep->samples.back();
        computeRollCentre(corner, atDroop);
        computeRollCentre(corner, atBump);

        REQUIRE(std::abs(atBump.rollCentreHeight - atDroop.rollCentreHeight) > 0.02);
        // A strut's roll centre falls as the car compresses, which is most of why they drop it on
        // corner entry.
        REQUIRE(atBump.rollCentreHeight < atDroop.rollCentreHeight);
    }
}
