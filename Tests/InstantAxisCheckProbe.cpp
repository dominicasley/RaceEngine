// A probe, not a gate. Hidden behind a dotted tag, run by hand:
// `./EngineTests "[.instant-axis-check]"`.
//
// Step 16 (docs/suspension-geometry-audit.md): reconcile the solved camber rate with the
// instant-centre construction. Step 15 reported the roll-centre front-view swing arm lengthening
// as the textbook construction predicts while the solved camber slope fell four times more
// slowly — for a rigid PLANAR linkage that cannot happen, so either the construction's FVSA is
// not the true instant axis (a), the solver's camber is not the upright's rotation (b), or the
// study compared different states (c).
//
// This probe derives the upright's TRUE instantaneous rotation from the production solver's own
// motion — a finite difference of the solved upright pose — and decomposes it:
//
//     dCamber/dTravel  =  (omega x spinAxis) read on the camber channel
//                      =  [non-steer part]  +  [steer part: (omega . kingpin) projected]
//
// The front-view instant-centre construction can only ever see the non-steer part; the tie rod
// imposes a steer rate (bump steer) whose projection through the kingpin's caster and
// inclination also moves camber. The residual column — solver rate minus construction minus
// steer part — is the verdict: near zero means cause (a), the construction neglecting the
// steer-camber coupling, with the solver's camber correct.
//
// Deterministic: pure solves, fixed states, fixed steps. Assertions only on determinism and on
// the probe's own finite-difference consistency; no acceptance thresholds against production.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::CornerHardpoints;
using raceengine::CornerSide;
using raceengine::computeRollCentre;
using raceengine::golfMk7FrontCorner;
using raceengine::outboardSign;
using raceengine::solveCorner;

namespace
{

constexpr auto degrees = 57.29577951308232;
constexpr auto casterTarget = 7.5 / degrees;

// The step-15 geometry builder, pivots dropped with the caster retarget included — the same
// states the study's terrain reported, so the reconciliation answers the study's own numbers.
[[nodiscard]] CornerHardpoints droppedHardpoints(const double pivotDrop, const bool retargetCaster)
{
    auto corner = golfMk7FrontCorner(CornerSide::Right);
    corner.lower.frontPivot.y -= pivotDrop;
    corner.lower.rearPivot.y -= pivotDrop;

    if (retargetCaster)
    {
        corner.strutTop.z =
            corner.lower.ballJoint.z - (corner.strutTop.y - corner.lower.ballJoint.y) * std::tan(casterTarget);
    }

    return corner;
}

[[nodiscard]] std::optional<double> angleForTravel(const CornerHardpoints& hardpoints, const double travel)
{
    const auto atDroop = solveCorner(hardpoints, hardpoints.droopAngle, 0.0);
    const auto atBump = solveCorner(hardpoints, hardpoints.bumpAngle, 0.0);
    if (!atDroop || !atBump || travel < atDroop->wheelTravel || travel > atBump->wheelTravel)
    {
        return std::nullopt;
    }

    auto low = hardpoints.droopAngle;
    auto high = hardpoints.bumpAngle;
    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto solved = solveCorner(hardpoints, middle, 0.0);
        if (!solved)
        {
            return std::nullopt;
        }

        (solved->wheelTravel < travel ? low : high) = middle;
    }

    return 0.5 * (low + high);
}

struct InstantMotion
{
    glm::dvec3 omegaPerTravel{0.0};    // the upright's rotation vector per metre of wheel travel
    glm::dvec3 axisPoint{0.0};         // a point on the instantaneous screw axis (closest to the BJ)
    glm::dvec3 ballJointVelocity{0.0}; // per metre of wheel travel
    double camberRateSolver = 0.0;     // d(camber)/d(travel), rad/m, off the solver's camber channel
    double camberRateFromOmega = 0.0;  // the same rate reconstructed from omega x spinAxis
    double camberRateSteer = 0.0;      // the kingpin-steer part of it
    double toeRate = 0.0;              // d(toe)/d(travel), rad/m
    bool valid = false;
};

[[nodiscard]] InstantMotion instantMotionAt(const CornerHardpoints& hardpoints, const double travel, const double step)
{
    auto motion = InstantMotion{};

    const auto angle = angleForTravel(hardpoints, travel);
    if (!angle)
    {
        return motion;
    }

    const auto centre = solveCorner(hardpoints, *angle, 0.0);
    if (!centre)
    {
        return motion;
    }

    const auto behind = solveCorner(hardpoints, *angle - step, 0.0, &centre.value());
    const auto ahead = solveCorner(hardpoints, *angle + step, 0.0, &centre.value());
    if (!behind || !ahead)
    {
        return motion;
    }

    const auto travelChange = ahead->wheelCentre.y - behind->wheelCentre.y;
    if (std::abs(travelChange) < 1e-15)
    {
        return motion;
    }

    // The relative rotation across the step, as a rotation vector — the upright's angular
    // velocity integrated over the step, per metre of wheel travel.
    const auto relative = glm::normalize(ahead->uprightOrientation * glm::conjugate(behind->uprightOrientation));
    const auto turn = glm::angle(relative);
    const auto axis = turn > 1e-14 ? glm::axis(relative) : glm::dvec3(0.0, 0.0, 1.0);
    motion.omegaPerTravel = (turn / travelChange) * axis;

    motion.ballJointVelocity = (ahead->lowerBallJoint - behind->lowerBallJoint) / travelChange;

    // A point on the instantaneous screw axis: the point closest to the ball joint.
    const auto omegaSquared = glm::dot(motion.omegaPerTravel, motion.omegaPerTravel);
    motion.axisPoint = omegaSquared > 1e-20
                           ? centre->lowerBallJoint + glm::cross(motion.omegaPerTravel, motion.ballJointVelocity) /
                                                          omegaSquared
                           : centre->lowerBallJoint;

    motion.camberRateSolver = (ahead->camber - behind->camber) / travelChange;
    motion.toeRate = (ahead->toe - behind->toe) / travelChange;

    // The same camber rate reconstructed from the rotation: camber = -asin(spinAxis.y), so its
    // rate is -(omega x spinAxis).y over cos(camber). The steer part projects omega onto the
    // kingpin — the axis from the solved ball joint to the strut top, which is what the tie rod
    // turns the upright about.
    const auto spin = centre->uprightOrientation * glm::dvec3(outboardSign(hardpoints.side), 0.0, 0.0);
    const auto cosCamber = std::sqrt(std::max(1.0 - spin.y * spin.y, 1e-12));
    motion.camberRateFromOmega = -glm::cross(motion.omegaPerTravel, spin).y / cosCamber;

    const auto kingpin = glm::normalize(hardpoints.strutTop - centre->lowerBallJoint);
    const auto steerOmega = glm::dot(motion.omegaPerTravel, kingpin) * kingpin;
    motion.camberRateSteer = -glm::cross(steerOmega, spin).y / cosCamber;

    motion.valid = true;
    return motion;
}

struct ConstructionNumbers
{
    double swingArmM = 0.0;
    double impliedRate = 0.0; // rad/m of camber per travel, magnitude, as 1/FVSA claims
    double rollCentreMm = 0.0;
    glm::dvec2 instantCentre{0.0};
    bool defined = false;
};

[[nodiscard]] ConstructionNumbers constructionAt(const CornerHardpoints& hardpoints, const double travel)
{
    auto numbers = ConstructionNumbers{};

    const auto angle = angleForTravel(hardpoints, travel);
    if (!angle)
    {
        return numbers;
    }

    auto solved = solveCorner(hardpoints, *angle, 0.0);
    if (!solved)
    {
        return numbers;
    }

    computeRollCentre(hardpoints, solved.value());
    if (!solved->instantCentreDefined)
    {
        return numbers;
    }

    numbers.swingArmM = std::abs(solved->contactPatch.x - solved->instantCentre.x);
    numbers.impliedRate = 1.0 / numbers.swingArmM;
    numbers.rollCentreMm = solved->rollCentreHeight * 1000.0;
    numbers.instantCentre = glm::dvec2(solved->instantCentre.x, solved->instantCentre.y);
    numbers.defined = true;
    return numbers;
}

// Everything in the sheet-frame slope unit the study reported: degrees per inch of travel,
// droop positive — which is minus the engine-frame camber rate.
[[nodiscard]] double asSheetSlope(const double ratePerMetre)
{
    return -ratePerMetre * degrees * 0.0254;
}

void reportGeometry(const char* name, const CornerHardpoints& hardpoints)
{
    std::printf("\n==== %s ====\n", name);
    std::printf("  all slopes in the study's unit: deg per inch of sheet travel, droop positive.\n");
    std::printf("  %7s | %8s %8s | %8s %8s %8s | %8s %8s | %9s\n", "trav mm", "solver", "from-w", "constr",
                "steer", "resid", "FVSAcon", "FVSAtrue", "toe deg/in");

    constexpr auto step = 1e-5;
    for (const auto travel : {-0.040, -0.020, 0.0, 0.020, 0.040})
    {
        const auto motion = instantMotionAt(hardpoints, travel, step);
        const auto construction = constructionAt(hardpoints, travel);
        if (!motion.valid || !construction.defined)
        {
            std::printf("  %7.0f |   state does not solve\n", travel * 1000.0);
            continue;
        }

        const auto solverSlope = asSheetSlope(motion.camberRateSolver);
        const auto steerSlope = asSheetSlope(motion.camberRateSteer);
        // The construction's implied slope carries the sign the solver's own curve has: camber
        // falls in bump, so the sheet-frame slope is positive.
        const auto constructionSlope = construction.impliedRate * degrees * 0.0254;
        const auto residual = solverSlope - constructionSlope - steerSlope;
        const auto trueSwingArm = std::abs(1.0 / motion.camberRateSolver);

        std::printf("  %7.0f | %+8.3f %+8.3f | %+8.3f %+8.3f %+8.3f | %8.2f %8.2f | %+9.3f\n", travel * 1000.0,
                    solverSlope, asSheetSlope(motion.camberRateFromOmega), constructionSlope, steerSlope, residual,
                    construction.swingArmM, trueSwingArm, asSheetSlope(motion.toeRate));
    }

    // The true instantaneous axis at design, direction and a point on it, beside the
    // construction's front-view instant centre.
    const auto design = instantMotionAt(hardpoints, 0.0, step);
    const auto construction = constructionAt(hardpoints, 0.0);
    if (design.valid && construction.defined)
    {
        const auto direction = glm::normalize(design.omegaPerTravel);
        std::printf("  true axis at design: direction (%+.4f, %+.4f, %+.4f), point (%+.3f, %+.3f, %+.3f) m\n",
                    direction.x, direction.y, direction.z, design.axisPoint.x, design.axisPoint.y,
                    design.axisPoint.z);
        std::printf("  construction IC at design: (%+.3f, %+.3f) m front view, roll centre %+.1f mm\n",
                    construction.instantCentre.x, construction.instantCentre.y, construction.rollCentreMm);
    }
}

} // namespace

TEST_CASE("the solved camber rate against the instant-centre construction", "[.instant-axis-check]")
{
    const auto row0 = golfMk7FrontCorner(CornerSide::Right);

    // Internal consistency of the probe's own differencing: the camber rate read off the solver's
    // camber channel and the rate reconstructed from the rotation vector are the same motion, and
    // the rotation converges as the step refines.
    const auto coarse = instantMotionAt(row0, 0.0, 1e-4);
    const auto fine = instantMotionAt(row0, 0.0, 1e-5);
    const auto finer = instantMotionAt(row0, 0.0, 1e-6);
    REQUIRE(coarse.valid);
    REQUIRE(fine.valid);
    REQUIRE(finer.valid);
    REQUIRE(fine.camberRateSolver == Catch::Approx(fine.camberRateFromOmega).epsilon(1e-5));
    REQUIRE(fine.camberRateSolver == Catch::Approx(finer.camberRateSolver).epsilon(1e-5));
    // The rotation vector converges between the two larger steps; at 1e-6 the quaternion's turn
    // is ~3e-7 rad and the axis extraction is noise-limited, which is a property of the probe's
    // differencing and not of the solve — the scalar rates above stay tight through it.
    REQUIRE(glm::length(coarse.omegaPerTravel - fine.omegaPerTravel) <
            5e-3 * glm::length(fine.omegaPerTravel));

    // And determinism: the same state twice is the same bits.
    const auto again = instantMotionAt(row0, 0.0, 1e-5);
    REQUIRE(again.camberRateSolver == fine.camberRateSolver);
    REQUIRE(again.omegaPerTravel == fine.omegaPerTravel);

    reportGeometry("row 0: current imported geometry (caster as imported)", row0);
    reportGeometry("pivots -70 mm (caster retargeted to 7.5 deg, as in the step-15 study)",
                   droppedHardpoints(0.070, true));
    reportGeometry("pivots -100 mm (caster retargeted to 7.5 deg, as in the step-15 study)",
                   droppedHardpoints(0.100, true));
}
