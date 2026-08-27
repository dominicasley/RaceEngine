// A probe, not a gate. Hidden behind a dotted tag, run by hand:
// `./EngineTests "[.front-camber-validation]"`.
//
// Validation of the front camber-vs-travel curve against measured reference data supplied by the
// project (2026-08-25): a spreadsheet of Mk7 front camber against wheel travel, stock against
// 034 RCO lower ball joints, at stock ride height and at ~0.50 in of lowering. The data is an
// external measured reference and NOT a model parameter; nothing here feeds production.
//
// Conventions, verified from the data rather than assumed (docs/suspension-geometry-audit.md,
// step 10): the sheet zeroes travel and camber at its ride height; **negative sheet travel is
// bump**, proven by the lowered dataset being the stock curve translated 0.5 in toward negative
// travel and re-zeroed — lowering compresses the suspension, so the lowered zero must sit on the
// bump side of the stock curve. Camber signs then align with the engine directly (bump makes
// camber more negative on both sides of the comparison). The mapping is therefore
//
//     engine wheel travel [m] = −0.0254 · sheet travel [in],   camber compared directly.
//
// The probe prints the residual of that translation identity, and of the rejected opposite
// mapping, so the convention stands on numbers rather than on a reading.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::CornerHardpoints;
using raceengine::CornerSide;
using raceengine::golfMk7FrontCorner;
using raceengine::solveCorner;

namespace
{

struct ReferencePoint
{
    double travelInches = 0.0;
    double camberDegrees = 0.0;
};

// Stock ride height, stock ball joints — the primary dataset.
constexpr std::array<ReferencePoint, 19> stockStock{{{-1.50, -0.28}, {-1.25, -0.25}, {-1.00, -0.21}, {-0.75, -0.16},
                                                     {-0.50, -0.12}, {-0.25, -0.06}, {0.00, 0.00},   {0.25, 0.07},
                                                     {0.50, 0.14},   {0.75, 0.23},   {1.00, 0.32},   {1.25, 0.42},
                                                     {1.50, 0.53},   {1.75, 0.65},   {2.00, 0.78},   {2.25, 0.92},
                                                     {2.50, 1.07},   {2.75, 1.23},   {3.00, 1.41}}};

// Stock ride height, 034 RCO ball joints — differential evidence only.
constexpr std::array<ReferencePoint, 19> rcoStock{{{-1.50, -0.38}, {-1.25, -0.33}, {-1.00, -0.28}, {-0.75, -0.22},
                                                   {-0.50, -0.15}, {-0.25, -0.08}, {0.00, 0.00},   {0.25, 0.08},
                                                   {0.50, 0.17},   {0.75, 0.27},   {1.00, 0.38},   {1.25, 0.49},
                                                   {1.50, 0.61},   {1.75, 0.74},   {2.00, 0.87},   {2.25, 1.02},
                                                   {2.50, 1.17},   {2.75, 1.34},   {3.00, 1.51}}};

// Lowered ~0.50 in, stock ball joints — secondary evidence and the convention proof.
constexpr std::array<ReferencePoint, 23> stockLowered{
    {{-2.00, -0.27}, {-1.75, -0.25}, {-1.50, -0.23}, {-1.25, -0.20}, {-1.00, -0.17}, {-0.75, -0.13},
     {-0.50, -0.09}, {-0.25, -0.05}, {0.00, 0.00},   {0.25, 0.05},   {0.50, 0.12},   {0.75, 0.18},
     {1.00, 0.26},   {1.25, 0.34},   {1.50, 0.43},   {1.75, 0.53},   {2.00, 0.64},   {2.25, 0.76},
     {2.50, 0.89},   {2.75, 1.03},   {3.00, 1.18},   {3.25, 1.35},   {3.50, 1.53}}};

constexpr auto metresPerInch = 0.0254;
constexpr auto degrees = 57.29577951308232;

template <std::size_t count>
[[nodiscard]] std::optional<double> referenceAt(const std::array<ReferencePoint, count>& table, const double inches)
{
    for (const auto& point : table)
    {
        if (std::abs(point.travelInches - inches) < 1e-9)
        {
            return point.camberDegrees;
        }
    }

    return std::nullopt;
}

// The wishbone angle whose wheel travel is the target, by bisection — travel is monotonic in the
// angle across the stated range (`validateCorner` refuses anything else).
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

[[nodiscard]] std::optional<double> modelCamberAtSheetTravel(const CornerHardpoints& hardpoints, const double inches)
{
    const auto travel = -metresPerInch * inches;
    const auto angle = angleForTravel(hardpoints, travel);
    if (!angle)
    {
        return std::nullopt;
    }

    const auto solved = solveCorner(hardpoints, *angle, 0.0);
    if (!solved)
    {
        return std::nullopt;
    }

    return solved->camber * degrees;
}

} // namespace

TEST_CASE("front camber against the measured Mk7 curve", "[.front-camber-validation]")
{
    const auto hardpoints = golfMk7FrontCorner(CornerSide::Right);

    // The reference loads as expected: monotonic in travel, zeroed at its own ride height.
    for (auto index = std::size_t{1}; index < stockStock.size(); index++)
    {
        REQUIRE(stockStock[index].travelInches > stockStock[index - 1].travelInches);
        REQUIRE(stockStock[index].camberDegrees > stockStock[index - 1].camberDegrees);
    }
    REQUIRE(referenceAt(stockStock, 0.0).value() == 0.0);

    // The convention conversion is the right way round: mapped bump (negative sheet travel) makes
    // the model's camber more negative, exactly as it makes the sheet's more negative.
    const auto modelBump = modelCamberAtSheetTravel(hardpoints, -1.0);
    const auto modelDroop = modelCamberAtSheetTravel(hardpoints, 1.0);
    REQUIRE(modelBump.has_value());
    REQUIRE(modelDroop.has_value());
    REQUIRE(modelBump.value() < 0.0);
    REQUIRE(modelDroop.value() > 0.0);

    // And the model evaluation is deterministic.
    REQUIRE(modelCamberAtSheetTravel(hardpoints, 0.75).value() == modelCamberAtSheetTravel(hardpoints, 0.75).value());

    std::printf("\nconvention: engine travel = -25.4 mm per sheet inch (negative sheet travel = bump);"
                " camber compared directly\n");

    // The proof the mapping is the data's own: the lowered curve is the stock curve translated
    // 0.5 in toward negative travel and re-zeroed. The rejected opposite mapping is printed too.
    auto chosenWorst = 0.0;
    auto rejectedWorst = 0.0;
    for (const auto& point : stockLowered)
    {
        if (const auto shifted = referenceAt(stockStock, point.travelInches - 0.5))
        {
            chosenWorst = std::max(chosenWorst, std::abs(point.camberDegrees -
                                                         (*shifted - referenceAt(stockStock, -0.5).value())));
        }
        if (const auto shifted = referenceAt(stockStock, point.travelInches + 0.5))
        {
            rejectedWorst = std::max(rejectedWorst, std::abs(point.camberDegrees -
                                                             (*shifted - referenceAt(stockStock, 0.5).value())));
        }
    }
    std::printf("lowered-translation residual: chosen mapping (zero at stock -0.5 in) worst %.3f deg;"
                " rejected mapping worst %.3f deg\n",
                chosenWorst, rejectedWorst);

    std::printf("\nstock ride height, stock ball joints vs the model:\n");
    std::printf("  %8s %9s %9s %9s %8s %8s\n", "sheet in", "engine mm", "meas deg", "model deg", "err deg", "|ratio|");

    for (const auto& point : stockStock)
    {
        const auto model = modelCamberAtSheetTravel(hardpoints, point.travelInches);
        if (!model)
        {
            std::printf("  %8.2f %9.1f %9.2f   outside current production range (stated travel +/-55 mm)\n",
                        point.travelInches, -25.4 * point.travelInches, point.camberDegrees);
            continue;
        }

        const auto ratio = std::abs(point.camberDegrees) > 1e-9 ? *model / point.camberDegrees : 0.0;
        std::printf("  %8.2f %9.1f %9.2f %9.3f %8.3f %8.2f\n", point.travelInches, -25.4 * point.travelInches,
                    point.camberDegrees, *model, *model - point.camberDegrees, ratio);
    }

    // The shape numbers: local slope about the design position and towards each end, in the
    // sheet's own frame (degrees per inch of travel, droop positive).
    const auto slope = [&](const double centreInches, const double halfSpanInches,
                           auto&& evaluate) -> std::optional<double>
    {
        const auto above = evaluate(centreInches + halfSpanInches);
        const auto below = evaluate(centreInches - halfSpanInches);
        if (!above || !below)
        {
            return std::nullopt;
        }
        return (*above - *below) / (2.0 * halfSpanInches);
    };

    const auto measured = [&](const double inches) { return referenceAt(stockStock, inches); };
    const auto model = [&](const double inches) { return modelCamberAtSheetTravel(hardpoints, inches); };

    std::printf("\nlocal slope dCamber/dTravel [deg/in], sheet frame:\n");
    for (const auto centre : {-1.25, 0.0, 1.25, 2.0})
    {
        const auto measuredSlope = slope(centre, 0.25, measured);
        const auto modelSlope = slope(centre, 0.25, model);
        if (measuredSlope && modelSlope)
        {
            std::printf("  at %+5.2f in: measured %+6.3f, model %+6.3f, ratio %5.2f\n", centre, *measuredSlope,
                        *modelSlope, *modelSlope / *measuredSlope);
        }
    }

    // The 034 RCO differential — reference evidence only, no model counterpart exists and none is
    // authored: what a lower-ball-joint change does to the measured curve.
    std::printf("\n034 RCO minus stock (measured differential, stock ride height):\n  ");
    for (const auto& point : rcoStock)
    {
        const auto stock = referenceAt(stockStock, point.travelInches);
        std::printf("%+.2f:%+.2f  ", point.travelInches, point.camberDegrees - stock.value());
    }
    std::printf("\n");
}
