// The longitudinal curve's shape, measured rather than derived: `./EngineTests "[.tyre-shape]"`.
//
// The launch work ended pointing here. The tyre makes about half of `mu_peak * Fz` at the slip ratios
// a standing start reaches, and metering torque is consistently *slower* than spinning — which should
// not be true if the fall-off past peak were right. The diagnosis handed over was that the Magic
// Formula's plateau is a hard function of the shape factor alone:
//
//     F -> D * sin(C * pi/2)   as B*kappa grows
//
// so C = 1.65 caps the sliding force at sin(148.5 deg) = 52.25% of peak whatever B and E are. The
// lateral curve's C = 1.35 gives 85% and is healthy, which is why nothing caught it: **every
// validation this project has run is lateral** — skidpad, step steer, understeer gradient, load
// sensitivity, the whole assist derivation. The longitudinal curve has never been checked against
// anything.
//
// This probe exists because two things in that reasoning need measuring before anything is changed:
//
//   1. **B is not the number in the data.** `longitudinalStiffness` is a slip stiffness; the Magic
//      Formula's B is derived from it as `K / (C * mu)`, so it is 9.32 and not 20, and it *moves*
//      whenever C or the friction peak moves. Any hand calculation using B = 20 is off.
//   2. **`peakSlipOf` ignores E entirely** — it returns `tan(pi/2C)/B`, which is the peak of an E = 0
//      curve. It is already 25% adrift at E = -1, and it is what the combined-slip ellipse normalises
//      against, so a large positive E would leave that normalisation badly wrong.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::evaluateTyre;
using raceengine::golfGtiMk7;
using raceengine::TyreAxis;
using raceengine::tyreFriction;
using raceengine::TyreModel;
using raceengine::TyreSlip;

namespace
{

// What the curve actually does, by scanning it rather than by solving it. The peak of a Magic
// Formula with a non-zero E has no closed form worth writing, and the whole point here is to stop
// trusting closed forms that leave E out.
struct Shape
{
    double peakSlip = 0.0;
    double peakFraction = 0.0;
    double atThree = 0.0;
    double reportedPeakSlip = 0.0;
};

[[nodiscard]] Shape shapeOf(const TyreModel& model, const double load)
{
    const auto available = tyreFriction(model, TyreAxis::Longitudinal, load, 1.0) * load;

    auto shape = Shape{};

    // Pure longitudinal: no slip angle, so the combined-slip ellipse degenerates to the longitudinal
    // curve alone and this is the curve itself rather than a section through a surface.
    for (auto step = 1; step <= 40000; step++)
    {
        const auto ratio = 0.0001 * static_cast<double>(step);
        const auto forces = evaluateTyre(model, load, TyreSlip{.slipRatio = ratio, .slipAngle = 0.0}, 1.0);
        const auto fraction = forces.longitudinal / available;

        if (fraction > shape.peakFraction)
        {
            shape.peakFraction = fraction;
            shape.peakSlip = ratio;
        }

        if (std::abs(ratio - 3.0) < 5e-5)
        {
            shape.atThree = fraction;
            shape.reportedPeakSlip = forces.longitudinalPeakSlip;
        }
    }

    return shape;
}

} // namespace

TEST_CASE("what the longitudinal curve this car runs actually looks like", "[.tyre-shape]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto model = setup->corners.front().tyre;
    const auto load = model.nominalLoad;

    std::printf("\n=== as shipped: C = %.2f, K = %.1f, E = %.2f, mu_x = %.2f at %.0f N ===\n", model.longitudinalShape,
                model.longitudinalStiffness, model.longitudinalCurvature, model.longitudinalPeak, load);

    // The derived B, which is the number the formula actually uses.
    const auto friction = tyreFriction(model, TyreAxis::Longitudinal, load, 1.0);
    const auto derivedB = model.longitudinalStiffness / (model.longitudinalShape * friction);
    std::printf("  the Magic Formula's B is derived, not stated: K / (C * mu) = %.1f / (%.2f * %.3f) = %.3f\n",
                model.longitudinalStiffness, model.longitudinalShape, friction, derivedB);
    std::printf("  the asymptotic plateau is sin(C * 90 deg) = %.1f%% of peak, whatever B and E are\n",
                100.0 * std::sin(model.longitudinalShape * 1.5707963267948966));

    const auto shipped = shapeOf(model, load);
    std::printf("\n  measured: peak %.1f%% of available at slip %.3f; at slip 3.0 it makes %.1f%%\n",
                100.0 * shipped.peakFraction, shipped.peakSlip, 100.0 * shipped.atThree);
    std::printf("  `peakSlipOf` reports the peak at %.3f, which is %.0f%% adrift — it has no E term\n",
                shipped.reportedPeakSlip, 100.0 * (shipped.reportedPeakSlip / shipped.peakSlip - 1.0));

    std::printf("\n%7s %8s\n", "slip", "of peak");
    for (const auto ratio : {0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00, 3.00})
    {
        const auto forces = evaluateTyre(model, load, TyreSlip{.slipRatio = ratio, .slipAngle = 0.0}, 1.0);
        std::printf("%7.2f %7.1f%%\n", ratio,
                    100.0 * forces.longitudinal / (tyreFriction(model, TyreAxis::Longitudinal, load, 1.0) * load));
    }
}

TEST_CASE("the four candidates, judged where tyre data actually exists", "[.tyre-shape]")
{
    // **Judged at kappa = 1, not at 3.** A rig runs a tyre to about 0.2-0.3 of slip ratio, and to 1.0
    // for locked-wheel work; nobody measures one at 500% slip. Past that the Magic Formula is pure
    // extrapolation and its asymptote is an artefact of the functional form rather than a property of
    // any tyre — which is how a published C_x of 1.65 coexists with real tyres that plainly do not
    // lose half their grip when they spin. The 70-80% figure this is aimed at comes from locked-wheel
    // data, so kappa = 1 is where it should be read.
    //
    // The other column that decides this is **progressiveness**: how much force is already there at
    // half the peak slip. A broader curve is more progressive and more forgiving; a peakier one is
    // more like a semislick. That is the basis to choose on, rather than whatever the slip-stiffness
    // constraint happens to force.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto base = setup->corners.front().tyre;
    const auto load = base.nominalLoad;

    struct Candidate
    {
        const char* name;
        double shape;
        double curvature;
        double stiffness;
    };

    std::printf("\n=== the four, at kappa = 1 where the data lives ===\n");
    std::printf("\n%-26s %6s %7s %6s %9s %9s %9s %10s %11s\n", "", "C", "E", "K", "k=0.05", "k=0.20", "k=1.0",
                "asymptote", "half-peak");

    for (const auto& candidate : std::array<Candidate, 4>{{{"shipped", 1.65, -1.00, 20.0},
                                                           {"C 1.55 / E +0.50", 1.55, 0.50, 35.0},
                                                           {"C 1.50 / E 0.0", 1.50, 0.00, 28.0},
                                                           {"C 1.46 / E -1.62, K held", 1.46, -1.62, 20.0}}})
    {
        auto model = base;
        model.longitudinalShape = candidate.shape;
        model.longitudinalCurvature = candidate.curvature;
        model.longitudinalStiffness = candidate.stiffness;

        const auto available = tyreFriction(model, TyreAxis::Longitudinal, load, 1.0) * load;
        const auto at = [&](const double ratio)
        {
            return evaluateTyre(model, load, TyreSlip{.slipRatio = ratio, .slipAngle = 0.0}, 1.0).longitudinal /
                   available;
        };

        const auto measured = shapeOf(model, load);

        std::printf("%-26s %6.2f %7.2f %6.1f %8.3f %9.3f %9.3f %9.3f %10.3f\n", candidate.name, candidate.shape,
                    candidate.curvature, candidate.stiffness, at(0.05), at(0.20), at(1.0),
                    std::sin(candidate.shape * 1.5707963267948966), at(0.5 * measured.peakSlip));
    }

    std::printf("\n  `half-peak` is the fraction of peak already made at half the peak slip: higher is\n"
                "  broader and more progressive, lower is peakier and more semislick-like.\n");
}

TEST_CASE("candidate shapes, and what each costs in slip stiffness", "[.tyre-shape]")
{
    // **The trade this has to be chosen on.** A positive E flattens the curve, which is what raises
    // the sliding plateau — and a flatter curve puts its peak further out in slip. Holding the peak
    // where a semislick's belongs, around 0.10 to 0.15, then costs slip stiffness, and slip stiffness
    // is not free to invent: `K` is `dFx/dkappa` over `Fz`, and 10-30 is the road-tyre range with
    // race constructions reaching the low forties. A candidate that needs 80 is not a semislick.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto base = setup->corners.front().tyre;
    const auto load = base.nominalLoad;

    std::printf("\n=== candidates: what holds the peak near 0.12 and what it costs ===\n");
    std::printf("\n%6s %6s %7s %11s %10s %10s %12s\n", "C", "E", "K", "plateau", "peak slip", "peak", "at slip 3");

    for (const auto shape : {1.65, 1.55, 1.50, 1.45})
    {
        for (const auto curvature : {-1.0, 0.50, 0.80, 0.90, 0.95})
        {
            // Search K for the one that puts the peak nearest 0.12, rather than asserting a value:
            // the relationship runs through the derived B and there is no point pretending otherwise.
            auto bestStiffness = 0.0;
            auto bestDistance = 1e9;
            auto bestShape = Shape{};

            for (auto step = 10; step <= 1200; step += 5)
            {
                auto candidate = base;
                candidate.longitudinalShape = shape;
                candidate.longitudinalCurvature = curvature;
                candidate.longitudinalStiffness = 0.1 * static_cast<double>(step);

                const auto measured = shapeOf(candidate, load);
                const auto distance = std::abs(measured.peakSlip - 0.12);

                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestStiffness = candidate.longitudinalStiffness;
                    bestShape = measured;
                }
            }

            std::printf("%6.2f %6.2f %7.1f %10.1f%% %10.3f %9.1f%% %11.1f%%\n", shape, curvature, bestStiffness,
                        100.0 * std::sin(shape * 1.5707963267948966), bestShape.peakSlip,
                        100.0 * bestShape.peakFraction, 100.0 * bestShape.atThree);
        }
    }
}
