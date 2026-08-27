// A probe, not a gate. Hidden behind a dotted tag, run by hand:
// `./EngineTests "[.front-hardpoint-study]"`.
//
// Step 15's candidate study (docs/suspension-geometry-audit.md): the front camber gain is ~2.5x
// the project's chosen reference (Data Driven MQB, transcribed in
// FrontCamberValidationProbe.cpp) and the caster solves to 4.6 deg against ~7.5 published. Both
// point at the imported front hardpoints. This probe searches the three points the study is
// allowed to move — the lower arm's two inboard pivots and the strut top — for geometries that
// meet the camber-curve and caster targets, using the production solver and elements throughout,
// and characterises each candidate so the correction can be chosen with its side effects known.
//
// NOTHING here is authored. The Golf's production hardpoints are untouched; every candidate is a
// value in this probe's stack.
//
// The search is deterministic by construction: the caster is set closed-form (the strut top's z
// from the held ball joint and the 7.5 deg target), and the one remaining scalar per family — how
// far the inboard pivots drop — is found by bisection on the design camber slope, sixty
// iterations on a fixed bracket. No randomness, no iteration counts that depend on data.
//
// Conventions as established in step 10: engine travel = -25.4 mm per sheet inch, negative sheet
// travel is bump, camber compared directly.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::CornerSide;
using raceengine::computeRollCentre;
using raceengine::damperElementOf;
using raceengine::damperShaftCompression;
using raceengine::DamperForceSolution;
using raceengine::golfGtiMk7;
using raceengine::golfMk7FrontCorner;
using raceengine::solveCorner;
using raceengine::solveDamperKinematics;
using raceengine::solveElement;
using raceengine::validateCornerSetup;

namespace
{

constexpr auto degrees = 57.29577951308232;
constexpr auto metresPerInch = 0.0254;
constexpr auto casterTarget = 7.5 / degrees;
constexpr auto designSlopeTargetDegPerInch = 0.260;

struct ReferencePoint
{
    double travelInches = 0.0;
    double camberDegrees = 0.0;
};

// The chosen reference: stock ride height, stock ball joints (step 10).
constexpr std::array<ReferencePoint, 19> reference{{{-1.50, -0.28}, {-1.25, -0.25}, {-1.00, -0.21}, {-0.75, -0.16},
                                                    {-0.50, -0.12}, {-0.25, -0.06}, {0.00, 0.00},   {0.25, 0.07},
                                                    {0.50, 0.14},   {0.75, 0.23},   {1.00, 0.32},   {1.25, 0.42},
                                                    {1.50, 0.53},   {1.75, 0.65},   {2.00, 0.78},   {2.25, 0.92},
                                                    {2.50, 1.07},   {2.75, 1.23},   {3.00, 1.41}}};

// A candidate: the current front corner with the two inboard pivots dropped, the strut top moved
// outboard and/or raised, and the strut top's z set closed-form for the caster target. The ball
// joint, wheel centre and everything else are held.
[[nodiscard]] CornerHardpoints candidateHardpoints(const double pivotDrop, const double topOutboard,
                                                   const double topRaise, const bool retargetCaster = true,
                                                   const double tieRodDrop = 0.0)
{
    auto corner = golfMk7FrontCorner(CornerSide::Right);

    corner.lower.frontPivot.y -= pivotDrop;
    corner.lower.rearPivot.y -= pivotDrop;

    // Diagnostic only, never a candidate: the tie rod's two points are OUTSIDE the study's free
    // set, and moving them here exists solely to demonstrate the fourth-point finding.
    corner.steeringRackOuter.y -= tieRodDrop;
    corner.steeringArm.y -= tieRodDrop;
    corner.strutTop.x += raceengine::outboardSign(CornerSide::Right) * topOutboard;
    corner.strutTop.y += topRaise;

    if (retargetCaster)
    {
        // Caster is the kingpin's rearward lean, and the kingpin is ball joint to strut top with
        // the ball joint held — so the top's z is one closed-form move.
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

[[nodiscard]] std::optional<double> camberAtSheetInches(const CornerHardpoints& hardpoints, const double inches)
{
    const auto angle = angleForTravel(hardpoints, -metresPerInch * inches);
    if (!angle)
    {
        return std::nullopt;
    }

    const auto solved = solveCorner(hardpoints, *angle, 0.0);
    return solved ? std::optional{solved->camber * degrees} : std::nullopt;
}

[[nodiscard]] std::optional<double> toeAtTravel(const CornerHardpoints& hardpoints, const double travel)
{
    const auto angle = angleForTravel(hardpoints, travel);
    if (!angle)
    {
        return std::nullopt;
    }

    const auto solved = solveCorner(hardpoints, *angle, 0.0);
    return solved ? std::optional{solved->toe * degrees} : std::nullopt;
}

// The design camber slope in the sheet's frame, degrees per inch, droop positive.
[[nodiscard]] std::optional<double> designSlopeDegPerInch(const CornerHardpoints& hardpoints)
{
    const auto above = camberAtSheetInches(hardpoints, 0.25);
    const auto below = camberAtSheetInches(hardpoints, -0.25);
    if (!above || !below)
    {
        return std::nullopt;
    }

    return (*above - *below) / 0.5;
}

// The one scalar per family: how far the pivots drop, bisected on the design slope. The slope
// falls monotonically as the pivots drop over this bracket (the arm's front-view line rotates
// away from the strut-perpendicular), which is what makes bisection valid.
[[nodiscard]] std::optional<double> dropForTargetSlope(const double topOutboard, const double topRaise)
{
    auto low = 0.0;
    auto high = 0.10;

    const auto slopeAt = [&](const double drop) -> std::optional<double>
    { return designSlopeDegPerInch(candidateHardpoints(drop, topOutboard, topRaise)); };

    const auto atHigh = slopeAt(high);
    if (!atHigh)
    {
        return std::nullopt;
    }
    if (*atHigh > designSlopeTargetDegPerInch)
    {
        // The slope falls monotonically, so the ceiling is the best achievable inside the study's
        // 100 mm displacement bracket; the caller labels it as target-not-met.
        return high;
    }

    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto slope = slopeAt(middle);
        if (!slope)
        {
            return std::nullopt;
        }

        (*slope > designSlopeTargetDegPerInch ? low : high) = middle;
    }

    return 0.5 * (low + high);
}

struct KingpinNumbers
{
    double casterDeg = 0.0;
    double kpiDeg = 0.0;
    double trailMm = 0.0;
    double scrubMm = 0.0;
};

[[nodiscard]] KingpinNumbers kingpinOf(const CornerHardpoints& hardpoints)
{
    const auto kingpin = hardpoints.strutTop - hardpoints.lower.ballJoint;

    // The axis extended to the ground plane; trail and scrub are where it lands relative to the
    // design contact patch (directly below the wheel centre at design).
    const auto toGround = -hardpoints.lower.ballJoint.y / kingpin.y;
    const auto groundX = hardpoints.lower.ballJoint.x + toGround * kingpin.x;
    const auto groundZ = hardpoints.lower.ballJoint.z + toGround * kingpin.z;

    return KingpinNumbers{
        .casterDeg = std::atan2(-kingpin.z, kingpin.y) * degrees,
        .kpiDeg = std::atan2(std::abs(kingpin.x), kingpin.y) * degrees,
        .trailMm = (groundZ - hardpoints.wheelCentre.z) * 1000.0,
        .scrubMm = std::abs(hardpoints.wheelCentre.x - groundX) * 1000.0 *
                   (std::abs(groundX) < std::abs(hardpoints.wheelCentre.x) ? 1.0 : -1.0)};
}

struct RollCentreNumbers
{
    double heightMm = 0.0;
    double swingArmM = 0.0;
    bool defined = false;
};

[[nodiscard]] RollCentreNumbers rollCentreAtTravel(const CornerHardpoints& hardpoints, const double travel)
{
    const auto angle = angleForTravel(hardpoints, travel);
    if (!angle)
    {
        return RollCentreNumbers{};
    }

    auto solved = solveCorner(hardpoints, *angle, 0.0);
    if (!solved)
    {
        return RollCentreNumbers{};
    }

    computeRollCentre(hardpoints, solved.value());
    if (!solved->instantCentreDefined)
    {
        return RollCentreNumbers{};
    }

    return RollCentreNumbers{.heightMm = solved->rollCentreHeight * 1000.0,
                             .swingArmM = std::abs(solved->contactPatch.x - solved->instantCentre.x),
                             .defined = true};
}

// Wheel travel at which a shaft gap closes, through the production compression seam.
[[nodiscard]] double engagementTravelMm(const CornerSetup& corner, const double target)
{
    auto low = 0.0;
    auto high = target > 0.0 ? corner.hardpoints.bumpAngle : corner.hardpoints.droopAngle;
    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto compression = damperShaftCompression(
            corner, DamperForceSolution{
                        .length = solveElement(corner.hardpoints, damperElementOf(corner.hardpoints), middle).length});
        (std::abs(compression) < std::abs(target) ? low : high) = middle;
    }

    const auto solved = solveCorner(corner.hardpoints, 0.5 * (low + high), 0.0);
    return solved ? solved->wheelTravel * 1000.0 : 0.0;
}

struct Candidate
{
    const char* name = "";
    double pivotDrop = 0.0;
    double topOutboard = 0.0;
    double topRaise = 0.0;
    bool retargetCaster = true;
};

void report(const Candidate& candidate, const CornerSetup& productionFront, const double frontSprungCornerMass)
{
    const auto hardpoints =
        candidateHardpoints(candidate.pivotDrop, candidate.topOutboard, candidate.topRaise, candidate.retargetCaster);
    const auto baseline = golfMk7FrontCorner(CornerSide::Right);

    std::printf("\n==== %s ====\n", candidate.name);
    std::printf("  moves [mm]: pivots down %.1f; strut top outboard %.1f, up %.1f, rearward %.1f"
                " (total top move %.1f)\n",
                candidate.pivotDrop * 1000.0, candidate.topOutboard * 1000.0, candidate.topRaise * 1000.0,
                (baseline.strutTop.z - hardpoints.strutTop.z) * 1000.0,
                glm::distance(hardpoints.strutTop, baseline.strutTop) * 1000.0);

    auto setup = productionFront;
    setup.hardpoints = hardpoints;
    const auto validated = validateCornerSetup(setup);
    if (!validated)
    {
        std::printf("  REJECTED by validateCornerSetup: %s\n", validated.error().c_str());
        return;
    }

    // The camber curve against the reference, over the candidate's own solvable range.
    auto sumSquares = 0.0;
    auto worst = 0.0;
    auto counted = 0;
    std::printf("  camber vs reference:  sheet in | meas | model | err\n");
    for (const auto& point : reference)
    {
        const auto model = camberAtSheetInches(hardpoints, point.travelInches);
        if (!model)
        {
            std::printf("    %+5.2f | %+5.2f |   out of range\n", point.travelInches, point.camberDegrees);
            continue;
        }

        const auto error = *model - point.camberDegrees;
        sumSquares += error * error;
        worst = std::max(worst, std::abs(error));
        counted++;
        std::printf("    %+5.2f | %+5.2f | %+6.3f | %+6.3f\n", point.travelInches, point.camberDegrees, *model, error);
    }

    const auto slope = designSlopeDegPerInch(hardpoints);
    std::printf("  camber: design slope %+.3f deg/in (target %+.3f), RMS %.3f deg, max %.3f deg over %d points\n",
                slope.value_or(0.0), designSlopeTargetDegPerInch, counted > 0 ? std::sqrt(sumSquares / counted) : 0.0,
                worst, counted);

    const auto kingpin = kingpinOf(hardpoints);
    std::printf("  kingpin at design: caster %+.2f deg, KPI %.2f deg, trail %.1f mm, scrub %.1f mm\n",
                kingpin.casterDeg, kingpin.kpiDeg, kingpin.trailMm, kingpin.scrubMm);

    for (const auto travel : {-0.040, 0.0, 0.040})
    {
        const auto roll = rollCentreAtTravel(hardpoints, travel);
        if (roll.defined)
        {
            std::printf("  roll centre at %+3.0f mm travel: height %+7.1f mm, front-view swing arm %6.2f m\n",
                        travel * 1000.0, roll.heightMm, roll.swingArmM);
        }
    }

    // Bump steer with the rack untouched: toe across the range and the slope about design. Design
    // toe is zero by construction — the tie-rod length derives from the design pose — so the
    // static toe hold costs no rack adjustment on any candidate.
    const auto toeMinus = toeAtTravel(hardpoints, -0.040);
    const auto toePlus = toeAtTravel(hardpoints, 0.040);
    const auto toeNearBelow = toeAtTravel(hardpoints, -0.005);
    const auto toeNearAbove = toeAtTravel(hardpoints, 0.005);
    if (toeMinus && toePlus && toeNearBelow && toeNearAbove)
    {
        std::printf("  bump steer (rack unchanged): toe %+0.3f deg at -40 mm, %+0.3f deg at +40 mm;"
                    " %+0.4f deg per 10 mm about design\n",
                    *toeMinus, *toePlus, (*toeNearAbove - *toeNearBelow));
    }

    // Motion ratio and what it does to the wheel rate, frequency and stop engagements — rates and
    // gaps unchanged, so these move with the geometry and are reported, not corrected.
    const auto element = damperElementOf(hardpoints);
    const auto ratioAt = [&](const double angle)
    {
        const auto kinematics = solveDamperKinematics(hardpoints, element, angle);
        return kinematics ? std::abs(kinematics->motionRatio) : 0.0;
    };

    const auto designRatio = ratioAt(0.0);
    const auto wheelRate = productionFront.springRate * designRatio * designRatio;
    const auto frequency = std::sqrt(wheelRate / frontSprungCornerMass) / (2.0 * 3.14159265358979323846);
    std::printf("  |MR| droop end %.4f, design %.4f, bump end %.4f -> wheel rate %.0f N/m,"
                " front ride frequency %.3f Hz (rates unchanged)\n",
                ratioAt(hardpoints.droopAngle), designRatio, ratioAt(hardpoints.bumpAngle), wheelRate, frequency);

    std::printf("  stop engagement (gaps unchanged): bump %.1f mm, droop %.1f mm of wheel travel\n",
                engagementTravelMm(setup, setup.bumpStop.gap), engagementTravelMm(setup, -setup.droopStop.gap));

    std::printf("  anti-dive: not reported — the geometry characterisation has no side-view"
                " instant-centre machinery to read it from.\n");
}

} // namespace

TEST_CASE("front hardpoint candidate study", "[.front-hardpoint-study]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto& productionFront = setup->corners[0];

    // Front sprung corner mass by the same statics the car is built with, for the ride frequency.
    auto sprungMass = 0.0;
    auto sprungMoment = 0.0;
    for (const auto& component : setup->sprung)
    {
        sprungMass += component.mass;
        sprungMoment += component.mass * component.centre.z;
    }
    const auto frontAxleZ = setup->corners[0].hardpoints.wheelCentre.z;
    const auto rearAxleZ = setup->corners[2].hardpoints.wheelCentre.z;
    const auto frontSprungCornerMass =
        0.5 * sprungMass * (sprungMoment / sprungMass - rearAxleZ) / (frontAxleZ - rearAxleZ);

    std::printf("\nfront hardpoint study: targets design slope %+.3f deg/in and caster %.1f deg;"
                " held: ball joint, wheel centre, rack, rates, gaps.\n",
                designSlopeTargetDegPerInch, casterTarget * degrees);
    std::printf("front sprung corner mass %.1f kg (from the setup's own ledger)\n", frontSprungCornerMass);

    // Row zero: the current geometry, nothing moved, caster as imported.
    report(Candidate{.name = "row 0: current imported geometry", .retargetCaster = false}, productionFront,
           frontSprungCornerMass);

    // The search terrain, printed before the searches so an unreachable family is diagnosable.
    // The bump-steer and swing-arm columns are the diagnosis: the front-view swing arm lengthens
    // exactly as the swing-arm model predicts, but the solved camber slope falls four times more
    // slowly, and the difference tracks the bump steer the drop induces against the HELD tie rod
    // — steer about the inclined kingpin feeds camber back and opposes the geometry change.
    std::printf("\nterrain, family A (caster retargeted, strut top xy as authored):\n");
    std::printf("  %8s %12s %16s %12s\n", "drop mm", "slope deg/in", "bumpsteer/10mm", "FVSA m");
    for (auto drop = 0; drop <= 100; drop += 10)
    {
        const auto hardpoints = candidateHardpoints(drop / 1000.0, 0.0, 0.0);
        const auto slope = designSlopeDegPerInch(hardpoints);
        const auto toeBelow = toeAtTravel(hardpoints, -0.005);
        const auto toeAbove = toeAtTravel(hardpoints, 0.005);
        const auto roll = rollCentreAtTravel(hardpoints, 0.0);
        if (slope && toeBelow && toeAbove)
        {
            std::printf("  %8d %+12.3f %+16.4f %12.2f\n", drop, *slope, *toeAbove - *toeBelow,
                        roll.defined ? roll.swingArmM : 0.0);
        }
        else
        {
            std::printf("  %8d   does not solve across the range\n", drop);
        }
    }

    // The fourth-point demonstration — NOT a candidate, the tie rod is outside the free set: the
    // same drops with the tie rod's two points lowered by the same amount, so the steer coupling
    // is removed and the swing-arm prediction is recovered.
    std::printf("\nterrain, diagnostic only — tie rod dropped WITH the pivots (outside the free set):\n");
    std::printf("  %8s %12s %16s\n", "drop mm", "slope deg/in", "bumpsteer/10mm");
    for (auto drop = 0; drop <= 100; drop += 10)
    {
        const auto hardpoints = candidateHardpoints(drop / 1000.0, 0.0, 0.0, true, drop / 1000.0);
        const auto slope = designSlopeDegPerInch(hardpoints);
        const auto toeBelow = toeAtTravel(hardpoints, -0.005);
        const auto toeAbove = toeAtTravel(hardpoints, 0.005);
        if (slope && toeBelow && toeAbove)
        {
            std::printf("  %8d %+12.3f %+16.4f\n", drop, *slope, *toeAbove - *toeBelow);
        }
        else
        {
            std::printf("  %8d   does not solve across the range\n", drop);
        }
    }

    // The families. Each spends the caster on the strut top's z closed-form, then bisects the
    // pivot drop onto the design slope. Deterministic: same bracket, same sixty iterations, every
    // run.
    struct Family
    {
        const char* name = "";
        double topOutboard = 0.0;
        double topRaise = 0.0;
    };

    for (const auto& family : {Family{.name = "candidate A: pivots down only"},
                               Family{.name = "candidate B: pivots down + strut top outboard 30 mm",
                                      .topOutboard = 0.030},
                               Family{.name = "candidate C: pivots down + strut top raised 40 mm",
                                      .topRaise = 0.040}})
    {
        const auto drop = dropForTargetSlope(family.topOutboard, family.topRaise);
        if (!drop)
        {
            std::printf("\n==== %s ====\n  the range stops solving before any useful drop.\n", family.name);
            continue;
        }

        report(Candidate{.name = family.name,
                         .pivotDrop = *drop,
                         .topOutboard = family.topOutboard,
                         .topRaise = family.topRaise},
               productionFront, frontSprungCornerMass);
    }

    // Determinism: the probe's own gate — two evaluations of the same family agree exactly.
    const auto once = dropForTargetSlope(0.0, 0.0);
    const auto again = dropForTargetSlope(0.0, 0.0);
    REQUIRE(once.has_value());
    REQUIRE(again.has_value());
    REQUIRE(*once == *again);
}
