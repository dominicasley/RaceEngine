// A probe, not a gate. Hidden behind a dotted tag like the steering-geometry probe, and run by
// hand: `./EngineTests "[.suspension-sweep]"`.
//
// What it exists for is the two held chassis numbers nothing printed before: the rear motion ratio
// (the source's 0.64 against an expected ~0.78, `PublishedCarsImpl.cpp`) and the front bump travel
// (55 mm allowed against Dominic's measured 67-73 mm, `docs/braking-chain-brief.md`). Both are
// claims *about* the linkage, and neither has ever been read *off* the linkage in one table with
// its own conventions beside it. This prints, for every Golf corner across its complete stated
// travel: the generalised coordinate, the wheel's height and travel, the damper's length and
// compression, both Jacobians, the motion ratio as the solver defines it (signed
// dDamperLength/dWheelTravel), camber, toe, caster off the kingpin, the instantaneous centre and
// the roll centre — and then searches *past* the stated stops for the angle at which the linkage
// itself gives out, which is what separates "the geometry only does 55 mm" from "somebody typed
// 0.16 rad".
//
// Deterministic: the solve is pure, the sample counts are fixed, and nothing here reads a clock or
// a random source. Everything is chassis-frame SI; readable columns are converted at the printf.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::CornerHardpoints;
using raceengine::CornerSide;
using raceengine::computeRollCentre;
using raceengine::golfGtiMk7;
using raceengine::golfMk7FrontCorner;
using raceengine::golfMk7RearCorner;
using raceengine::solveCorner;
using raceengine::solveCornerWithJacobian;
using raceengine::SuspensionState;

namespace
{

constexpr auto degrees = 57.29577951308232;

// Caster off the kingpin the state already carries: the axis from the lower ball joint to the
// upper (strut top or upper ball joint), tilted rearward at the top being positive. A derived
// diagnostic of the probe, not a production quantity — the solver does not compute caster.
[[nodiscard]] double casterOf(const SuspensionState& state)
{
    const auto kingpin = glm::normalize(state.upperBallJoint - state.lowerBallJoint);
    return std::atan2(-kingpin.z, kingpin.y);
}

void printPoint(const char* name, const glm::dvec3& point)
{
    std::printf("    %-22s %+9.5f %+9.5f %+9.5f\n", name, point.x, point.y, point.z);
}

void printHardpoints(const CornerHardpoints& hardpoints)
{
    std::printf("  hardpoints (chassis frame, metres; +x car's left, +y up, +z forward):\n");
    printPoint("lower front pivot", hardpoints.lower.frontPivot);
    printPoint("lower rear pivot", hardpoints.lower.rearPivot);
    printPoint("lower ball joint", hardpoints.lower.ballJoint);
    if (hardpoints.kind == raceengine::SuspensionKind::MacPhersonStrut)
    {
        printPoint("strut top", hardpoints.strutTop);
    }
    else
    {
        printPoint("upper front pivot", hardpoints.upper.frontPivot);
        printPoint("upper rear pivot", hardpoints.upper.rearPivot);
        printPoint("upper ball joint", hardpoints.upper.ballJoint);
        printPoint("damper chassis", hardpoints.damperChassis);
        printPoint("damper wishbone", hardpoints.damperWishbone);
    }
    printPoint("steering rack outer", hardpoints.steeringRackOuter);
    printPoint("steering arm", hardpoints.steeringArm);
    printPoint("wheel centre", hardpoints.wheelCentre);
    std::printf("    stated travel range     %.3f .. %+.3f rad of lower-wishbone angle\n", hardpoints.droopAngle,
                hardpoints.bumpAngle);
}

// One corner's stated travel, row by row, with per-row validity rather than sweepCorner's
// all-or-nothing — a row that fails prints its own failure and the sweep continues.
void sweepStatedTravel(const CornerHardpoints& hardpoints, const double designDamperLength)
{
    std::printf("  sweep of the stated travel (conventions: travel +up = bump; damper compression"
                " +in bump; MR = dDamper/dTravel, signed as the solver reports it):\n");
    std::printf("    %9s %9s %8s %9s %8s %9s %9s %8s %7s %7s %7s %8s %8s %8s\n", "q [rad]", "wheelY", "trav mm",
                "dmpr mm", "comp mm", "dTrav/dq", "dDmpr/dq", "MR", "cmb deg", "toe deg", "cas deg", "IC x", "IC y",
                "RC h");

    constexpr auto samples = std::size_t{21};
    auto previous = std::vector<SuspensionState>{};

    for (auto index = std::size_t{0}; index < samples; index++)
    {
        const auto through = static_cast<double>(index) / static_cast<double>(samples - 1);
        const auto angle = hardpoints.droopAngle + through * (hardpoints.bumpAngle - hardpoints.droopAngle);

        auto solved = solveCornerWithJacobian(hardpoints, angle, 0.0, previous.empty() ? nullptr : &previous.back());
        if (!solved)
        {
            std::printf("    %+9.5f INVALID: %s\n", angle, solved.error().c_str());
            continue;
        }

        computeRollCentre(hardpoints, solved.value());
        const auto& s = solved.value();

        // Damper channels off the element — the sole source of damper geometry since step 14; the
        // printed values are the bits the state used to carry, so step-1 output stays comparable.
        const auto damper = raceengine::solveElement(hardpoints, raceengine::damperElementOf(hardpoints), angle);
        const auto kinematics =
            raceengine::solveDamperKinematics(hardpoints, raceengine::damperElementOf(hardpoints), angle);
        const auto ratio = kinematics ? kinematics->motionRatio : 0.0;

        std::printf("    %+9.5f %9.5f %+8.2f %9.2f %+8.2f %+9.5f %+9.5f %+8.4f %+7.3f %+7.3f %+7.3f ", angle,
                    s.wheelCentre.y, s.wheelTravel * 1000.0, damper.length * 1000.0,
                    (designDamperLength - damper.length) * 1000.0, s.travelPerAngle, damper.lengthPerAngle,
                    ratio, s.camber * degrees, s.toe * degrees, casterOf(s) * degrees);

        if (s.instantCentreDefined)
        {
            std::printf("%+8.3f %+8.3f %+8.4f\n", s.instantCentre.x, s.instantCentre.y, s.rollCentreHeight);
        }
        else
        {
            std::printf("%8s %8s %+8.4f\n", "undef", "undef", s.rollCentreHeight);
        }

        previous.push_back(s);
    }
}

// Where the linkage itself gives out, which the stated stops sit inside. Steps outward from a
// stated end, chained by continuity, until the solve fails or the wheel stops rising — and names
// which constraint failed, because "the tie rod cannot reach" and "the wishbone locks" are
// different answers to whether a stated stop is the geometry's own limit.
void searchGeometricLimit(const CornerHardpoints& hardpoints, const double direction)
{
    const auto start = direction > 0.0 ? hardpoints.bumpAngle : hardpoints.droopAngle;
    constexpr auto step = 0.005;

    auto last = solveCornerWithJacobian(hardpoints, start, 0.0);
    if (!last)
    {
        std::printf("  the stated %s end itself does not solve: %s\n", direction > 0.0 ? "bump" : "droop",
                    last.error().c_str());
        return;
    }

    auto atLimit = last.value();
    auto failure = std::string{};

    for (auto angle = start + direction * step; std::abs(angle) < 1.5; angle += direction * step)
    {
        const auto solved = solveCornerWithJacobian(hardpoints, angle, 0.0, &atLimit);
        if (!solved)
        {
            failure = solved.error();
            break;
        }

        if (direction * (solved->wheelTravel - atLimit.wheelTravel) <= 0.0)
        {
            failure = "the wheel stops moving vertically (travel turning point)";
            break;
        }

        atLimit = solved.value();
    }

    const auto limitLength =
        raceengine::solveElement(hardpoints, raceengine::damperElementOf(hardpoints), atLimit.wishboneAngle).length;
    std::printf("  geometric %s limit: q = %+.3f rad, wheel travel %+.1f mm, damper %.1f mm"
                " (stated stop at %+.3f rad, %+.1f mm)\n",
                direction > 0.0 ? "bump" : "droop", atLimit.wishboneAngle, atLimit.wheelTravel * 1000.0,
                limitLength * 1000.0, start, last->wheelTravel * 1000.0);
    std::printf("    what ends it: %s\n", failure.empty() ? "search ceiling reached" : failure.c_str());
}

// The wheel travel at which a damper-axis gap closes, by bisection on the solve. The stops are
// stated on the shaft (`TravelStop::gap` in metres of damper compression or extension), so where
// they engage *in wheel travel* is a property of the linkage, and this reads it off rather than
// assuming a ratio.
[[nodiscard]] double wheelTravelWhereDamperMoves(const CornerHardpoints& hardpoints, const double designDamperLength,
                                                const double damperChange)
{
    auto low = 0.0;
    auto high = damperChange < 0.0 ? hardpoints.bumpAngle : hardpoints.droopAngle;

    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto change =
            raceengine::solveElement(hardpoints, raceengine::damperElementOf(hardpoints), middle).length -
            designDamperLength;
        if (std::abs(change) < std::abs(damperChange))
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }

    const auto at = solveCorner(hardpoints, 0.5 * (low + high), 0.0);
    return at ? at->wheelTravel : 0.0;
}

} // namespace

TEST_CASE("the Golf's suspension geometry, swept and characterised", "[.suspension-sweep]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto& hardpoints = corner.hardpoints;

        const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());

        std::printf("\n==== %s (%s, %s) ====\n", cornerAbbreviation(static_cast<Corner>(index)),
                    hardpoints.kind == raceengine::SuspensionKind::MacPhersonStrut ? "MacPherson strut"
                                                                                  : "double wishbone",
                    hardpoints.side == CornerSide::Left ? "left, +x outboard" : "right, -x outboard");

        printHardpoints(hardpoints);

        const auto element = raceengine::damperElementOf(hardpoints);
        const auto designDamper = raceengine::solveElement(hardpoints, element, 0.0);
        const auto designKinematics = raceengine::solveDamperKinematics(hardpoints, element, 0.0);
        REQUIRE(designKinematics.has_value());

        std::printf("  design point (q = 0, the static ride by construction of the rest length):\n");
        std::printf("    damper length %.4f m, dTravel/dq %+.5f m/rad, dDamper/dq %+.5f m/rad,"
                    " motion ratio %+.4f (|MR| %.4f)\n",
                    designDamper.length, design->travelPerAngle, designDamper.lengthPerAngle,
                    designKinematics->motionRatio, std::abs(designKinematics->motionRatio));

        const auto ratio = std::abs(designKinematics->motionRatio);
        std::printf("    shaft spring rate %.0f N/m -> wheel rate spring x MR^2 = %.0f N/m\n", corner.springRate,
                    corner.springRate * ratio * ratio);

        sweepStatedTravel(hardpoints, designDamper.length);

        const auto bumpEngages = wheelTravelWhereDamperMoves(hardpoints, designDamper.length, -corner.bumpStop.gap);
        const auto droopEngages = wheelTravelWhereDamperMoves(hardpoints, designDamper.length, corner.droopStop.gap);
        std::printf("  stops on the shaft: bump gap %.1f mm engages at %+.1f mm of wheel travel;"
                    " droop gap %.1f mm engages at %+.1f mm\n",
                    corner.bumpStop.gap * 1000.0, bumpEngages * 1000.0, corner.droopStop.gap * 1000.0,
                    droopEngages * 1000.0);

        searchGeometricLimit(hardpoints, 1.0);
        searchGeometricLimit(hardpoints, -1.0);
    }
}
