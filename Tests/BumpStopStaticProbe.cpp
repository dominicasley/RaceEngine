#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::cornerCount;
using raceengine::damperElementOf;
using raceengine::golfGtiMk7;
using raceengine::jounceReferenceForce;
using raceengine::solveCornerWithJacobian;
using raceengine::solveElement;
using raceengine::TravelStop;

// The bump stop's **static** force/travel law, audited and printed.
// `./EngineTests "[.stop-static]"`.
//
// Hidden behind a dotted tag like every other probe here: what it produces is a table to read and a
// data question to answer, not a bound to hold. The invariants that came out of it are the CHECKs
// in the last two cases, and they are invariants rather than thresholds — each one is a statement
// that would be a *defect* if it failed, not a number somebody chose.
//
// **Why it exists.** `[.jounce-stop]` measured the sourced *dynamic* branch (Pech et al., VSD 2024,
// 10.1080/00423114.2024.2378858) and found it cannot replace the placed 40000 N.s/m viscous
// constant: 20.4 J against 366.0 J in the representative cycle. That closed the dissipation
// question and left the *static* law open, with one very large discrepancy on the record — this
// model reaches the source's 9 kN reference force in 15.9 mm of stop compression where the measured
// specimen takes about 71. This probe decomposes that number, separates scale from shape, and says
// what data would close it.
//
// **Nothing here changes any car.** Every alternative representation below is evaluated in this
// file and installed nowhere.

namespace
{

// The two turnaround points the source publishes on its own specimen, and the third statement that
// over-determines them. `docs/fetched/jounce.txt`:
//
//   "While the maximum operating displacement xm = 68 mm on the real road causes a maximum jounce
//    force of 3 kN, the safety margin leads to a maximum force of 9 kN, representing a safety
//    margin of about 200%. In terms of potential energy stored in the jounce, the buffer leads to a
//    safety margin of about 70%."
//
// So: displacement +10%, force +200%, stored energy +70%. Three statements, and the third is the
// one that decides the *shape* — see `the shape the source's own three statements imply` below.
constexpr auto specimenOperatingDeflection = 0.068;
constexpr auto specimenOperatingForce = 3000.0;
constexpr auto specimenDisplacementMargin = 0.10;
constexpr auto specimenEnergyMargin = 0.70;

// The deflection at which the specimen reaches `jounceReferenceForce`. **The paper does not print
// it.** Two readings of the text, carried side by side rather than collapsed:
//
//   - 74.8 mm: the maximum operating displacement plus the stated 10% displacement margin, which is
//     the sentence that *defines* the reference point.
//   - 71.0 mm: "the hysteresis height increases monotonically up to a displacement of approx.
//     71 mm", which is what this project's earlier record took for xref. It is a statement about
//     the hysteresis-height curve, and the sentence after it has the hysteresis *falling* before
//     the turnaround — so 71 mm is just below xref rather than equal to it.
//
// Both are used below and neither is asserted. Nothing in the comparison turns on the choice: the
// two differ by 5% against a discrepancy of 350%.
constexpr auto specimenReferenceFromMargin = 0.0748;
constexpr auto specimenReferenceFromHysteresis = 0.071;

// The current law, written once so every table below reads the same arithmetic the car does.
[[nodiscard]] double elasticForce(const TravelStop& stop, const double past)
{
    if (past <= 0.0)
    {
        return 0.0;
    }

    return stop.rate * std::pow(past, stop.progression) / std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);
}

// dF/dx of the same law, analytically. The power rule, so it cannot drift from the force above.
[[nodiscard]] double elasticStiffness(const TravelStop& stop, const double past)
{
    if (past <= 0.0)
    {
        return 0.0;
    }

    return stop.progression * stop.rate * std::pow(past, stop.progression - 1.0) /
           std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);
}

// Where the law reaches a stated force. One root, because the law is a pure power of the deflection.
[[nodiscard]] double deflectionAt(const TravelStop& stop, const double force)
{
    const auto scale = std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);

    return std::pow(force * scale / std::max(stop.rate, 1e-6), 1.0 / stop.progression);
}

// Energy stored by the law from first contact to `past`, in closed form. `F = A x^p` integrates to
// `A x^(p+1)/(p+1)`, which is `F(x) x / (p+1)`.
[[nodiscard]] double storedEnergy(const TravelStop& stop, const double past)
{
    return elasticForce(stop, past) * past / (stop.progression + 1.0);
}

// The damper shaft's compression against its design length at a stated wishbone angle, positive in
// bump. Exactly what `damperShaftCompression` computes in the force pass, evaluated straight off the
// element so this file needs no vehicle state.
[[nodiscard]] double shaftCompressionAt(const CornerHardpoints& hardpoints, const double angle)
{
    const auto element = damperElementOf(hardpoints);

    return solveElement(hardpoints, element, 0.0).length - solveElement(hardpoints, element, angle).length;
}

// The wishbone angle at a stated shaft compression, by bisection between design and the linkage's
// own bump limit. Monotonic over that range on both of this car's axles, which the caller checks.
[[nodiscard]] double angleForCompression(const CornerHardpoints& hardpoints, const double compression)
{
    auto low = 0.0;
    auto high = hardpoints.bumpAngle;

    for (auto step = 0; step < 80; step++)
    {
        const auto middle = 0.5 * (low + high);
        if (shaftCompressionAt(hardpoints, middle) < compression)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }

    return 0.5 * (low + high);
}

// Wheel travel at a stated shaft compression, metres, positive in bump. The map is the linkage's,
// so it carries whatever the motion ratio is doing at that position rather than a constant.
[[nodiscard]] double wheelTravelAtCompression(const CornerHardpoints& hardpoints, const double compression)
{
    const auto solved = solveCornerWithJacobian(hardpoints, angleForCompression(hardpoints, compression), 0.0);
    REQUIRE(solved.has_value());

    const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());

    return solved->wheelTravel - design->wheelTravel;
}

struct Axle
{
    const char* name;
    std::size_t index;
};

constexpr auto axles = std::array{Axle{"front", 0}, Axle{"rear", 2}};

// How far the stop can be compressed before the linkage runs out, metres. This is the honest end of
// every table here: past it the corner meets the range clamp, which has no reaction force, and a
// row printed beyond it would be describing a position the car cannot reach.
[[nodiscard]] double usableStopTravel(const CornerSetup& corner)
{
    return std::max(0.0, shaftCompressionAt(corner.hardpoints, corner.hardpoints.bumpAngle) - corner.bumpStop.gap);
}

} // namespace

TEST_CASE("the shipped Golf bump stop, front and rear, against shaft compression", "[.stop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== the shipped static law ===\n");
    std::printf("  F(x) = rate * x^progression / gap^(progression - 1),  x = shaft compression past the gap\n");
    std::printf("  and the whole stop force is  max(0, F(x) + damping * shaft velocity + hysteresis term).\n");
    std::printf("  Every quantity below is on the DAMPER SHAFT. The stop lives there, not at the wheel.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto& stop = corner.bumpStop;
        const auto usable = usableStopTravel(corner);
        const auto atLimit = shaftCompressionAt(corner.hardpoints, corner.hardpoints.bumpAngle);

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    gap %.1f mm | rate %.0f N/m | progression %.2f | damping %.0f N.s/m | hysteresis %.3f\n",
                    1000.0 * stop.gap, stop.rate, stop.progression, stop.damping, stop.hysteresis);
        std::printf("    dynamic branch stated: %s\n", stop.statesDynamicBranch() ? "YES" : "no");
        std::printf("    shaft compression at the linkage's bump limit  %.2f mm\n", 1000.0 * atLimit);
        std::printf("    so the stop has                                %.2f mm of usable travel\n", 1000.0 * usable);
        std::printf("    wheel travel at that limit                     %.2f mm\n",
                    1000.0 * wheelTravelAtCompression(corner.hardpoints, atLimit));

        std::printf("\n     x past   elastic       dF/dx        wheel |     viscous at shaft velocity (kN)\n");
        std::printf("      (mm)       (kN)       (kN/mm)       (mm)  |   0.05    0.25    1.00    2.00 m/s\n");

        for (const auto millimetres : {0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0})
        {
            const auto past = 0.001 * millimetres;
            if (past > usable + 1e-9)
            {
                std::printf("     %5.1f     -- beyond the linkage's own bump limit, not reachable --\n", millimetres);
                continue;
            }

            const auto travel = wheelTravelAtCompression(corner.hardpoints, stop.gap + past);

            std::printf("     %5.1f   %8.3f   %10.4f   %8.2f  |", millimetres, 0.001 * elasticForce(stop, past),
                        1e-6 * elasticStiffness(stop, past), 1000.0 * travel);

            for (const auto velocity : {0.05, 0.25, 1.0, 2.0})
            {
                std::printf(" %7.2f", 0.001 * stop.damping * velocity);
            }
            std::printf("\n");
        }

        // The corner's own static load on the shaft, which is what the stop force has to be read
        // against. The spring's free length is solved from that load, so the force it carries at the
        // design position **is** the static load referred to the same axis the stop lives on.
        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());
        const auto staticLoad = raceengine::solveSpringForce(corner, design.value()).force;

        std::printf("\n     force anchor    reached at   of usable travel   wheel travel   in static loads\n");
        for (const auto force : {1000.0, 3000.0, 5000.0, 9000.0, 15000.0, 25000.0})
        {
            const auto past = deflectionAt(stop, force);
            if (past > usable)
            {
                std::printf("     %8.0f N    NOT REACHABLE (would need %.1f mm, the stop has %.1f)\n", force,
                            1000.0 * past, 1000.0 * usable);
                continue;
            }

            std::printf("     %8.0f N    %7.2f mm   %14.1f%%   %10.2f mm   %13.2f\n", force, 1000.0 * past,
                        100.0 * past / usable, 1000.0 * wheelTravelAtCompression(corner.hardpoints, stop.gap + past),
                        force / std::max(staticLoad, 1.0));
        }

        std::printf("\n     the corner carries %.0f N on this axis at rest, and at the linkage's own limit\n",
                    staticLoad);
        std::printf("     the law reads %.0f N — %.0f static loads. Nothing sizes that number: it is what\n",
                    elasticForce(stop, usable), elasticForce(stop, usable) / std::max(staticLoad, 1.0));
        std::printf("     the power law extrapolates to, not what anybody measured a bumper doing.\n");
    }

    std::printf("\n  The viscous columns are the SHIPPED 40000 N.s/m, which is a project placement and not a\n");
    std::printf("  measurement (docs/suspension-fidelity-brief.md, 2026-08-29). At the source's own top test\n");
    std::printf("  velocity of 2 m/s it is worth 80 kN by itself, which is why it must be read beside the\n");
    std::printf("  elastic column rather than under it.\n");
}

TEST_CASE("the measured specimen: what the source states, derives and never says", "[.stop-static]")
{
    std::printf("\n=== Pech, Dessort & Prokop, Vehicle System Dynamics 63:8 (2025) 1625, CC-BY ===\n");
    std::printf("  docs/fetched/jounce.txt is the full text. Figures carry no numbers in it.\n");

    std::printf("\n  STATED IN WORDS\n");
    std::printf("    maximum operating displacement on the real road   %6.1f mm\n",
                1000.0 * specimenOperatingDeflection);
    std::printf("    the jounce force there                            %6.0f N\n", specimenOperatingForce);
    std::printf("    displacement safety margin over it                %6.0f %%\n",
                100.0 * specimenDisplacementMargin);
    std::printf("    reference force at that margin                    %6.0f N   (a 200%% force margin)\n",
                jounceReferenceForce);
    std::printf("    stored-energy margin at the same point            %6.0f %%\n", 100.0 * specimenEnergyMargin);
    std::printf("    static Dahl saturation                            %6.0f N   = half w(xref)\n", 900.0);
    std::printf("    dynamic stiffening, peak                          %6.0f N   at high displacement\n", 2800.0);
    std::printf("    maximum hysteresis height (whole programme)       %6.0f N   just below xref\n", 3000.0);
    std::printf("    maximum hysteresis width                          %6.1f mm  at about 0.5 kN\n", 18.0);
    std::printf("    attachment spacing (free travel before contact)   %6.1f mm\n", 5.0);
    std::printf("    static ramp velocity                              %6.2f mm/s\n", 0.83);
    std::printf("    dynamic ramp velocities                        25, 50, 500, 1000, 2000 mm/s\n");
    std::printf("    Maxwell elements needed to fit the envelope            5\n");
    std::printf("    compression only:  \"the jounce can only transfer forces in one direction\"\n");
    std::printf("    the displacement IS jounce travel past first contact, not total specimen height:\n");
    std::printf("      \"the jounce is located on the top-mount and is compressed by the damper housing\",\n");
    std::printf("      with the attachment spacing above stated separately. So it is directly comparable\n");
    std::printf("      with this model's `past`, which is shaft compression past the gap.\n");

    std::printf("\n  DERIVED HERE, NOT PRINTED BY THE SOURCE\n");
    std::printf("    xref from the 10%% displacement margin             %6.1f mm\n",
                1000.0 * specimenReferenceFromMargin);
    std::printf("    xref read as the hysteresis-monotonicity limit    %6.1f mm  (this project's earlier figure)\n",
                1000.0 * specimenReferenceFromHysteresis);
    std::printf("    the source prints NEITHER as xref. Both are carried; nothing below turns on the choice.\n");

    std::printf("\n  UNKNOWN — the source does not state it and it must not be filled in from elsewhere\n");
    std::printf("    the specimen's unstressed (free) length\n");
    std::printf("    its material (the paper says jounce bumpers are \"usually\" elastomer or PUR; not this one)\n");
    std::printf("    its geometry, diameter or mass\n");
    std::printf("    the test temperature, and any temperature conditioning at all\n");
    std::printf("    the vehicle it came off, its class, its mass or its axle\n");
    std::printf("    the three fitted Dahl coefficients (Figure 12's axis carries no numbers)\n");
    std::printf("    the fitted Maxwell contact points and stiffnesses (read off Figure 16 by an\n");
    std::printf("      earlier session and carried in `jounceBumperCandidate`; unverifiable from the text)\n");
    std::printf("    the fade-in half width psmooth, which the source optimises and does not publish\n");

    std::printf("\n  NOT TRANSFERABLE, and this is the whole caveat on everything below:\n");
    std::printf("    this is ONE specimen off ONE unnamed car. It is evidence about what a production\n");
    std::printf("    microcellular jounce bumper does. It is not this car's part and nothing here\n");
    std::printf("    licenses copying its numbers onto the Golf.\n");
}

TEST_CASE("the shape the source's own three statements imply", "[.stop-static]")
{
    // **The decisive arithmetic in this file, and it needs no figure and no fitted parameter.**
    //
    // For ANY pure power law through the origin, `F = A x^n`, the stored energy is `F(x) x /(n+1)`.
    // So the ratio of stored energies at two points is
    //
    //     E2/E1 = (F2 x2) / (F1 x1),
    //
    // and the exponent cancels exactly. The source states F2/F1 = 3 and x2/x1 = 1.1, so every pure
    // power law whatsoever predicts an energy ratio of 3.3 — a 230% margin. The source measured 70%.
    //
    // That is a statement about the FORM and not about its parameters: no choice of `rate` and
    // `progression` can be wrong about this in a way another choice would fix.
    const auto ratio = (jounceReferenceForce / specimenOperatingForce) * (1.0 + specimenDisplacementMargin);

    std::printf("\n=== can a single power law carry the specimen's three published statements? ===\n");
    std::printf("  For F = A x^n the stored energy is F(x) x / (n + 1), so\n");
    std::printf("      E(xref) / E(xm) = (Fref / Fm) * (xref / xm) = %.1f * %.2f = %.2f,\n",
                jounceReferenceForce / specimenOperatingForce, 1.0 + specimenDisplacementMargin, ratio);
    std::printf("  INDEPENDENT of n, because the exponent cancels.\n");
    std::printf("  The source measured                                                  %.2f\n",
                1.0 + specimenEnergyMargin);
    std::printf("  => NO pure power law through the origin can state all three. The form is refuted,\n");
    std::printf("     not a parameterisation of it.\n");

    // The same conclusion the other way round: what the energy statement says about the curve's
    // shape factor, which is the quantity a power law's exponent *is*.
    //
    // E(xref) = 1.7 E(xm), so the energy added between the two points is 0.7 E(xm). That added
    // energy is bounded by the force at each end times the interval, which brackets E(xm) without
    // assuming any shape at all.
    const auto interval = specimenDisplacementMargin * specimenOperatingDeflection;
    const auto addedLow = specimenOperatingForce * interval;
    const auto addedHigh = jounceReferenceForce * interval;
    const auto addedTrapezoid = 0.5 * (specimenOperatingForce + jounceReferenceForce) * interval;

    const auto shapeFactor = [](const double energy) { return energy / (specimenOperatingForce * specimenOperatingDeflection); };
    const auto exponentFor = [](const double factor) { return 1.0 / factor - 1.0; };

    std::printf("\n  what the 70%% says about the bulk of the stroke, with no shape assumed\n");
    std::printf("    the interval xm -> xref is %.2f mm and the force across it runs %.0f -> %.0f N,\n",
                1000.0 * interval, specimenOperatingForce, jounceReferenceForce);
    std::printf("    so the energy added over it is between %.1f and %.1f J (trapezoid %.1f J).\n", addedLow, addedHigh,
                addedTrapezoid);
    std::printf("    That added energy IS 70%% of E(xm), so E(xm) lies between %.1f and %.1f J (%.1f J).\n",
                addedLow / specimenEnergyMargin, addedHigh / specimenEnergyMargin,
                addedTrapezoid / specimenEnergyMargin);
    std::printf("    Mean force over 0..xm as a fraction of the peak: %.3f to %.3f (%.3f).\n",
                shapeFactor(addedLow / specimenEnergyMargin), shapeFactor(addedHigh / specimenEnergyMargin),
                shapeFactor(addedTrapezoid / specimenEnergyMargin));
    std::printf("    A power law's shape factor is 1/(n+1), so the BULK of the specimen behaves like\n");
    std::printf("    n = %.2f to %.2f (%.2f) — and this model's authored progression is 3.00.\n",
                exponentFor(shapeFactor(addedHigh / specimenEnergyMargin)),
                exponentFor(shapeFactor(addedLow / specimenEnergyMargin)),
                exponentFor(shapeFactor(addedTrapezoid / specimenEnergyMargin)));

    // And the local exponent at the top, which is what the bulk figure is not.
    for (const auto reference : {specimenReferenceFromMargin, specimenReferenceFromHysteresis})
    {
        const auto local = std::log(jounceReferenceForce / specimenOperatingForce) /
                           std::log(reference / specimenOperatingDeflection);

        std::printf("\n    but between xm and xref = %.1f mm the LOCAL exponent is %.1f.\n", 1000.0 * reference, local);
    }

    std::printf("\n  So the specimen is a roughly cubic bulk with a hard densification in its last few\n");
    std::printf("  millimetres. That is what a microcellular bumper does, and it is exactly the part a\n");
    std::printf("  single power law cannot carry: one exponent has to be both 2.5 and 11.\n");

    // The shipped car put through the same three statements, so the refutation is demonstrated on
    // this model's own arithmetic rather than only argued about the family in general.
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto& front = base->corners[0].bumpStop;
    const auto reference = deflectionAt(front, jounceReferenceForce);
    const auto operating = deflectionAt(front, specimenOperatingForce);

    std::printf("\n  the same three statements, measured on THIS car's front stop\n");
    std::printf("    it reaches %.0f N at %.2f mm and %.0f N at %.2f mm, so its displacement margin is\n",
                specimenOperatingForce, 1000.0 * operating, jounceReferenceForce, 1000.0 * reference);
    std::printf("    %.1f%% where the specimen's is %.0f%% — its force triples over %.1f%% more travel,\n",
                100.0 * (reference / operating - 1.0), 100.0 * specimenDisplacementMargin,
                100.0 * (reference / operating - 1.0));
    std::printf("    not %.0f%%. And its stored energy goes %.2f J -> %.2f J, a ratio of %.2f against the\n",
                100.0 * specimenDisplacementMargin, storedEnergy(front, operating), storedEnergy(front, reference),
                storedEnergy(front, reference) / storedEnergy(front, operating));
    std::printf("    specimen's %.2f — which is the power law's own (Fref/Fm)(xref/xm), exactly.\n",
                1.0 + specimenEnergyMargin);

    CHECK(std::abs(storedEnergy(front, reference) / storedEnergy(front, operating) -
                   (jounceReferenceForce / specimenOperatingForce) * (reference / operating)) < 1e-9);

    // The one thing an assertion is worth here: that the refutation is not an artefact of which
    // reading of xref is taken. Both readings give a power-law energy ratio far above the measured
    // one, and the two readings differ by 5%.
    for (const auto specimenReference : {specimenReferenceFromMargin, specimenReferenceFromHysteresis})
    {
        const auto predicted =
            (jounceReferenceForce / specimenOperatingForce) * (specimenReference / specimenOperatingDeflection);

        CHECK(predicted > 1.5 * (1.0 + specimenEnergyMargin));
    }
}

TEST_CASE("the shipped stop and the specimen, normalised three ways", "[.stop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto& front = base->corners[0].bumpStop;
    const auto reference = deflectionAt(front, jounceReferenceForce);

    std::printf("\n=== normalised comparison, front bump stop against the specimen ===\n");

    std::printf("\n  A. absolute\n");
    std::printf("     this stop reaches %.0f N at %6.2f mm\n", jounceReferenceForce, 1000.0 * reference);
    std::printf("     the specimen at    %6.1f mm (margin reading) or %6.1f mm (hysteresis reading)\n",
                1000.0 * specimenReferenceFromMargin, 1000.0 * specimenReferenceFromHysteresis);
    std::printf("     so the shipped stop is %.2fx to %.2fx shorter in absolute deflection.\n",
                specimenReferenceFromHysteresis / reference, specimenReferenceFromMargin / reference);

    std::printf("\n  B. force / 9 kN against deflection / deflection at 9 kN\n");
    std::printf("     x/xref    this stop F/Fref    the specimen, where the source states it\n");

    for (const auto fraction : {0.25, 0.50, 0.75, 0.90, 0.909, 0.958, 1.00})
    {
        std::printf("     %6.3f    %13.4f    ", fraction, std::pow(fraction, front.progression));

        if (std::abs(fraction - 1.0) < 1e-9)
        {
            std::printf("1.0000  (definition of xref)\n");
        }
        else if (std::abs(fraction - 0.909) < 1e-3)
        {
            std::printf("0.3333  (xm/xref on the 74.8 mm reading)\n");
        }
        else if (std::abs(fraction - 0.958) < 1e-3)
        {
            std::printf("0.3333  (xm/xref on the 71.0 mm reading)\n");
        }
        else
        {
            std::printf("  --    (the source states no force at this fraction)\n");
        }
    }

    std::printf("\n     At the one interior point the source states, this stop carries %.1f%% or %.1f%% of its\n",
                100.0 * std::pow(specimenOperatingDeflection / specimenReferenceFromMargin, front.progression),
                100.0 * std::pow(specimenOperatingDeflection / specimenReferenceFromHysteresis, front.progression));
    std::printf("     reference force where the specimen carries 33.3%%. That is the shape difference, and\n");
    std::printf("     it survives both readings of xref.\n");

    std::printf("\n  C. tangent stiffness normalised by Fref / xref\n");
    std::printf("     For F = Fref (x/xref)^p the normalised tangent is p (x/xref)^(p-1), so at xref it is\n");
    std::printf("     exactly the progression: %.2f here.\n", front.progression);
    std::printf("     x/xref    this stop k*xref/Fref\n");
    for (const auto fraction : {0.25, 0.50, 0.75, 0.90, 1.00})
    {
        std::printf("     %6.2f    %16.3f\n", fraction, front.progression * std::pow(fraction, front.progression - 1.0));
    }
    std::printf("     The specimen's normalised tangent between xm and xref is about %.1f, and its bulk\n",
                std::log(jounceReferenceForce / specimenOperatingForce) /
                    std::log(specimenReferenceFromMargin / specimenOperatingDeflection));
    std::printf("     average is near %.1f. One number cannot be both, which is C saying B again.\n", 2.5);

    // D. Against the part's OWN travel, which is the normalisation that does not need either part's
    //    absolute size — and the one this car has evidence for.
    const auto usable = usableStopTravel(base->corners[0]);

    std::printf("\n  D. against each part's own travel\n");
    std::printf("     this stop's crushable travel, from the linkage's bump limit less the gap:\n");
    std::printf("       %.2f mm front, %.2f mm rear.\n", 1000.0 * usable, 1000.0 * usableStopTravel(base->corners[2]));
    std::printf("     **and the front figure is DERIVED FROM A MEASUREMENT OF THIS CAR** — `bumpAngle` was\n");
    std::printf("     set from Dominic's own 67-73 mm tape measure, ride height to the stop fully crushed\n");
    std::printf("     (2026-08-29), and the 20 mm gap from a measured 18 mm on a 2019 GTI shaft. So the\n");
    std::printf("     Golf's front jounce bumper crushes about %.0f mm and nothing about that is placed.\n",
                1000.0 * usable);
    std::printf("     For the record, the same two measurements put its FREE length near 55 mm: the\n");
    std::printf("     measured car showed 73 mm of bare shaft at ride height with 18 mm of it free.\n");
    std::printf("\n     this stop reaches %.0f N at %.1f%% of its own travel.\n", jounceReferenceForce,
                100.0 * reference / usable);
    std::printf("     The specimen reaches it at the TOP of its own measurement programme — every\n");
    std::printf("     turnaround point in Table 2 is a fraction of xref and the largest IS xref — and its\n");
    std::printf("     own crushable travel is UNKNOWN, so 100%% is a lower bound on the denominator.\n");
    std::printf("     So even with both absolute sizes taken out, this stop arrives about %.1fx early.\n",
                usable / reference);

    std::printf("\n     x / own travel    this stop's force\n");
    for (const auto fraction : {0.10, 0.25, 0.34, 0.50, 0.75, 1.00})
    {
        std::printf("     %12.2f      %12.0f N   (%.1f x the reference force)\n", fraction,
                    elasticForce(front, fraction * usable), elasticForce(front, fraction * usable) / jounceReferenceForce);
    }

    std::printf("\n  VERDICT: SCALE **and** SHAPE.\n");
    std::printf("    scale — %.1fx in absolute deflection, and %.1fx even after each part is normalised\n",
                specimenReferenceFromMargin / reference, usable / reference);
    std::printf("            by its own travel, which removes the part-size difference entirely.\n");
    std::printf("    shape — the bulk exponents agree (about 3 against about 2.5), and the specimen's\n");
    std::printf("            terminal densification has no representation in a single power law at all.\n");
}

TEST_CASE("what rate, gap and progression actually control", "[.stop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto& front = base->corners[0].bumpStop;

    std::printf("\n=== the current parameterisation, algebraically ===\n");
    std::printf("  F(x) = rate * x^p / gap^(p-1)\n");
    std::printf("    at x = gap:  F = rate * gap          = %.0f N\n", front.rate * front.gap);
    std::printf("    at x = gap:  dF/dx = p * rate        = %.0f N/m\n", front.progression * front.rate);
    std::printf("  So `rate` is the SECANT stiffness at one gap of compression, and the tangent there is\n");
    std::printf("  p times it. Neither is a stiffness anywhere else.\n");
    std::printf("\n  Inverting for the reference deflection:\n");
    std::printf("    x(F) = gap * (F / (rate * gap))^(1/p)\n");
    std::printf("    x(%.0f N) = %.1f mm * (%.0f / %.0f)^(1/%.2f) = %.1f mm * %.4f = %.3f mm\n", jounceReferenceForce,
                1000.0 * front.gap, jounceReferenceForce, front.rate * front.gap, front.progression, 1000.0 * front.gap,
                std::pow(jounceReferenceForce / (front.rate * front.gap), 1.0 / front.progression),
                1000.0 * deflectionAt(front, jounceReferenceForce));

    std::printf("\n  **`gap` carries two jobs and this is the defect in the parameterisation.**\n");
    std::printf("  It is the engagement travel — where the stop first touches, which is geometry and is\n");
    std::printf("  sourced on this car (18 mm measured on a 2019 GTI, 20 mm authored). And it is the\n");
    std::printf("  force law's own length scale, through gap^(p-1). Moving one moves the other:\n");
    std::printf("\n     gap      F at 10 mm past    dF/dx at 10 mm    reaches 9 kN at    change in the law\n");

    for (const auto millimetres : {16.0, 18.0, 20.0, 22.0, 25.0})
    {
        auto variant = front;
        variant.gap = 0.001 * millimetres;

        std::printf("    %4.0f mm   %12.0f N   %13.0f N/m   %11.2f mm    %+7.1f%% of force at 10 mm\n", millimetres,
                    elasticForce(variant, 0.010), elasticStiffness(variant, 0.010),
                    1000.0 * deflectionAt(variant, jounceReferenceForce),
                    100.0 * (elasticForce(variant, 0.010) / elasticForce(front, 0.010) - 1.0));
    }

    std::printf("\n  Two millimetres of ENGAGEMENT — a tape measure's worth, on a quantity this project has\n");
    std::printf("  actually measured — moves the force at a fixed compression by %.0f%%. That is a\n",
                std::abs(100.0 * (elasticForce([&] { auto v = front; v.gap = 0.018; return v; }(), 0.010) /
                                      elasticForce(front, 0.010) -
                                  1.0)));
    std::printf("  measurement of the linkage silently rewriting the material law of a rubber part.\n");
    std::printf("  The two are independent facts about the car and the parameterisation ties them together.\n");

    // The coupling stated as an invariant rather than as a table: with `rate` and `progression`
    // held, the force at a fixed compression must NOT depend on the gap — and here it does.
    auto narrower = front;
    narrower.gap = 0.018;
    CHECK(std::abs(elasticForce(narrower, 0.010) - elasticForce(front, 0.010)) > 1.0);
}

TEST_CASE("alternative static-law representations, evaluated offline", "[.stop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    const auto& front = base->corners[0].bumpStop;
    const auto reference = deflectionAt(front, jounceReferenceForce);

    std::printf("\n=== four representations, none of them installed anywhere ===\n");

    // B. The same family, anchored on a force and a deflection instead of on a rate and the gap.
    //    `F = Fref (x/xref)^n`. Identical expressive power to A — it is the same power law — and
    //    that is the point: the difference is entirely in what the authored numbers MEAN.
    struct Anchored
    {
        double referenceForce = 0.0;
        double referenceDeflection = 0.0;
        double exponent = 0.0;

        [[nodiscard]] double force(const double past) const
        {
            return past <= 0.0 ? 0.0 : referenceForce * std::pow(past / referenceDeflection, exponent);
        }
    };

    const auto anchored = Anchored{jounceReferenceForce, reference, front.progression};

    std::printf("\n  A vs B: the same curve, two parameterisations\n");
    std::printf("     x (mm)     A: rate/gap/p      B: Fref/xref/n      difference\n");
    auto worst = 0.0;
    for (const auto millimetres : {2.0, 5.0, 10.0, 15.0, 20.0, 30.0})
    {
        const auto past = 0.001 * millimetres;
        const auto difference = std::abs(elasticForce(front, past) - anchored.force(past));
        worst = std::max(worst, difference);

        std::printf("     %6.1f   %13.2f N   %15.2f N   %11.2e N\n", millimetres, elasticForce(front, past),
                    anchored.force(past), difference);
    }

    // B reproduces A to the bit-ish, so it is a re-parameterisation and not a different model. That
    // has to be true or the comparison below is measuring two curves rather than two spellings.
    CHECK(worst < 1e-6);
    std::printf("     B reproduces A exactly (worst %.2e N). It is a re-spelling, not a new model.\n", worst);

    std::printf("\n  and what B buys: the engagement point stops being in the force law\n");
    std::printf("     gap      A: F at 10 mm     B: F at 10 mm\n");
    for (const auto millimetres : {16.0, 18.0, 20.0, 22.0, 25.0})
    {
        auto variant = front;
        variant.gap = 0.001 * millimetres;

        std::printf("    %4.0f mm   %12.0f N   %14.0f N\n", millimetres, elasticForce(variant, 0.010),
                    anchored.force(0.010));
    }
    std::printf("     B's column is constant, which is the property being looked for: a sourced\n");
    std::printf("     \"9 kN at X mm\" can be stated without a tape measure of the free travel changing it.\n");

    // C. A monotonic tabulated force/deflection curve, the representation the source itself uses for
    //    its static submodel (`Fnl(x)` is derived from measurement, not fitted to a function).
    struct Tabulated
    {
        std::vector<double> deflection;
        std::vector<double> force;

        [[nodiscard]] double at(const double past) const
        {
            if (past <= deflection.front())
            {
                return force.front();
            }

            for (auto index = std::size_t{1}; index < deflection.size(); index++)
            {
                if (past <= deflection[index])
                {
                    const auto span = deflection[index] - deflection[index - 1];
                    const auto through = (past - deflection[index - 1]) / span;

                    return force[index - 1] + through * (force[index] - force[index - 1]);
                }
            }

            // Past the last point, continue on the last segment's slope rather than saturating: a
            // stop that goes flat is a stop the car can drive through.
            const auto last = deflection.size() - 1;
            const auto slope = (force[last] - force[last - 1]) / (deflection[last] - deflection[last - 1]);

            return force[last] + slope * (past - deflection[last]);
        }
    };

    // **Sampled off the shipped law, deliberately.** The point is to measure what the representation
    // costs, not to propose a curve — proposing one would need Golf data this project does not have.
    auto table = Tabulated{};
    for (const auto millimetres : {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 35.0, 40.0})
    {
        table.deflection.push_back(0.001 * millimetres);
        table.force.push_back(elasticForce(front, 0.001 * millimetres));
    }

    std::printf("\n  C: a monotonic table on the same curve, 14 points to 40 mm\n");
    std::printf("     x (mm)      power law        table       error        as %% of the reference force\n");
    auto worstTable = 0.0;
    auto worstAbsolute = 0.0;
    for (const auto millimetres : {1.0, 3.0, 7.0, 11.0, 13.7, 16.0, 22.0, 27.0, 33.0, 38.0})
    {
        const auto past = 0.001 * millimetres;
        const auto exact = elasticForce(front, past);
        const auto error = table.at(past) - exact;
        worstAbsolute = std::max(worstAbsolute, std::abs(error));

        if (exact > 100.0)
        {
            worstTable = std::max(worstTable, std::abs(error) / exact);
        }

        std::printf("     %6.1f   %12.1f N %12.1f N   %+9.1f N   %+8.3f %%\n", millimetres, exact, table.at(past),
                    error, 100.0 * error / jounceReferenceForce);
    }
    std::printf("     Worst error %.1f N in absolute terms and %.2f %% wherever the force is above 100 N.\n",
                worstAbsolute, 100.0 * worstTable);
    std::printf("     The one large *relative* error is at 1 mm, where the exact force is 2.2 N and the\n");
    std::printf("     table says 9.0 N — a percentage of nothing. Quote the absolute column here.\n");
    std::printf("     Linear interpolation always reads HIGH on a convex curve, which is the safe sign\n");
    std::printf("     for a stop: it never under-reports between samples.\n");

    // D. The source's own representation, stated for completeness: its static submodel IS a
    //    tabulated measured characteristic, so D and C are the same representation.
    std::printf("\n  D: the source's own static representation\n");
    std::printf("     Pech et al. derive Fnl(x) from the measurement itself — \"the measured force for\n");
    std::printf("     static excitations is equal to the sum of the nonlinear force-displacement-\n");
    std::printf("     characteristic and the output force of the modified Dahl model\" — so their static\n");
    std::printf("     law is a measured table, not a function fitted to one. D is C.\n");

    std::printf("\n  and the test that separates the families: the specimen's three statements\n");
    std::printf("     A / B  (one power law) : cannot. The energy ratio is (Fref/Fm)(xref/xm) whatever\n");
    std::printf("                              the exponent, so it predicts %.2f against a measured %.2f.\n",
                (jounceReferenceForce / specimenOperatingForce) * (1.0 + specimenDisplacementMargin),
                1.0 + specimenEnergyMargin);
    std::printf("     C / D  (a table)       : can, trivially — three constraints, and a table has as\n");
    std::printf("                              many degrees of freedom as it has rows.\n");

    std::printf("\n  RANKING, on the criterion the task set — which representation separates engagement\n");
    std::printf("  point, force scale, shape and reference deflection cleanly:\n");
    std::printf("     1. C/D, a table with the engagement gap kept OUTSIDE it. Every published anchor\n");
    std::printf("        goes in as a row; shape is whatever the rows say; nothing couples.\n");
    std::printf("     2. B, Fref/xref/n. Decouples engagement from the law and states a sourced\n");
    std::printf("        \"9 kN at X mm\" directly, but still cannot bend twice.\n");
    std::printf("     3. A, the shipped form. Expressive enough for a smooth progressive stop and\n");
    std::printf("        parameterised so that a geometry measurement rewrites the material law.\n");
}

TEST_CASE("free consistency checks on the shipped stop", "[.stop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== invariants, all four corners, no rig and no seat needed ===\n");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = base->corners[index];
        const auto& stop = corner.bumpStop;

        // 1. Nothing before contact. A stop that carries force at zero compression is a preload
        //    nobody authored.
        CHECK(stop.force(0.0) == 0.0);
        CHECK(stop.force(-0.001) == 0.0);

        // 2. Monotonic and non-negative through the whole usable range, at zero velocity. A stop
        //    that softens as it is compressed is not a stop.
        const auto usable = usableStopTravel(corner);
        auto previous = -1.0;
        for (auto step = 0; step <= 200; step++)
        {
            const auto past = usable * static_cast<double>(step) / 200.0;
            const auto force = stop.force(past);

            CHECK(force >= 0.0);
            CHECK(force >= previous);
            CHECK(std::isfinite(force));
            previous = force;
        }

        // 3. Finite at the linkage's own limit, and at ten times the usable travel. The law is a
        //    power, so it cannot be singular — this is the check that says so rather than assuming.
        CHECK(std::isfinite(stop.force(usable)));
        CHECK(std::isfinite(stop.force(10.0 * usable, 5.0)));

        // 4. The push-only clamp holds: however fast the stop is released, it never pulls.
        for (const auto velocity : {-0.5, -2.0, -10.0, -100.0})
        {
            CHECK(stop.force(0.5 * usable, velocity) >= 0.0);
        }

        // 5. **The stop opposes further travel** — the sign check, and it is a property of the
        //    Jacobian rather than of the force. The generalised contribution is
        //    `force * damperLengthPerAngle`, the shaft SHORTENS as the wishbone angle rises, so
        //    that Jacobian must be negative for a positive stop force to resist bump.
        const auto element = damperElementOf(corner.hardpoints);
        const auto midway = solveElement(corner.hardpoints, element, 0.5 * corner.hardpoints.bumpAngle);
        CHECK(midway.lengthPerAngle < 0.0);

        // 6. Energy over a symmetric in-and-out stroke is never negative — the stop cannot be a
        //    source. Integrated on the shipped law with the shipped viscous term, at a rate inside
        //    the source's own measured band.
        constexpr auto tick = 1.0 / 360.0;
        constexpr auto frequency = 2.0;
        const auto amplitude = 0.5 * usable;
        auto work = 0.0;
        for (auto step = 0; step < 360; step++)
        {
            const auto phase = 2.0 * std::numbers::pi * frequency * (static_cast<double>(step) + 0.5) * tick;
            const auto past = amplitude * (1.0 - std::cos(phase));
            const auto velocity = amplitude * 2.0 * std::numbers::pi * frequency * std::sin(phase);

            work += stop.force(past, velocity) * velocity * tick;
        }

        CHECK(work >= 0.0);

        std::printf("  corner %zu: contact-free below the gap, monotonic to %.1f mm, push-only, opposes bump,\n",
                    index, 1000.0 * usable);
        std::printf("            and takes %.1f J out of a 2 Hz stroke over half its travel.\n", work);
    }

    // 7. Left and right of an axle must carry the same scalar stop. A mirrored corner is the same
    //    part fitted the other way round, and a difference here would be a data mistake nothing
    //    else in the suite would see.
    for (const auto& axle : axles)
    {
        const auto& left = base->corners[axle.index].bumpStop;
        const auto& right = base->corners[axle.index + 1].bumpStop;

        CHECK(left.gap == right.gap);
        CHECK(left.rate == right.rate);
        CHECK(left.progression == right.progression);
        CHECK(left.damping == right.damping);
        CHECK(left.hysteresis == right.hysteresis);

        // And the same again through the geometry, which is the stronger statement: the two
        // corners' linkages must put the stop at the same wheel travel.
        const auto leftTravel =
            wheelTravelAtCompression(base->corners[axle.index].hardpoints, base->corners[axle.index].bumpStop.gap);
        const auto rightTravel = wheelTravelAtCompression(base->corners[axle.index + 1].hardpoints,
                                                          base->corners[axle.index + 1].bumpStop.gap);

        CHECK(std::abs(leftTravel - rightTravel) < 1e-9);

        std::printf("  %s: both sides engage at %.2f mm of wheel travel and carry the same scalar law.\n", axle.name,
                    1000.0 * leftTravel);
    }
}

TEST_CASE("static ride never touches the bump stop", "[.stop-static]")
{
    // The cheapest check there is, and it is the one that would catch a stop gap authored smaller
    // than the car's own static deflection. Solved rather than simulated: at the design position the
    // shaft compression is zero by construction, so the question is whether the settled ride height
    // sits at the design angle.
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== static ride against the stop ===\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];

        // The spring's rest length is solved from the corner's static load, so the design position
        // IS the settled position. The check is that the gap is a real distance from it.
        CHECK(corner.bumpStop.gap > 0.0);
        CHECK(shaftCompressionAt(corner.hardpoints, 0.0) == 0.0);
        CHECK(corner.bumpStop.force(shaftCompressionAt(corner.hardpoints, 0.0) - corner.bumpStop.gap) == 0.0);

        std::printf("  %s: design shaft compression %.6f mm, gap %.1f mm, stop force %.1f N.\n", axle.name,
                    1000.0 * shaftCompressionAt(corner.hardpoints, 0.0), 1000.0 * corner.bumpStop.gap,
                    corner.bumpStop.force(-corner.bumpStop.gap));
    }

    std::printf("  The design position is where the spring's free length was solved for, so this is the\n");
    std::printf("  settled car and not an approximation of it.\n");
}

TEST_CASE("the stop's stiffness against the corner's own integrator", "[.stop-static]")
{
    // **Until 2026-09-08 the corner integrated the stop EXPLICITLY, both halves of it.**
    // `stepVehicle`'s velocity step solved only the damper curve's slope implicitly
    // (`cornerDamping`); the stop's elastic force and its viscous term both rode `generalisedForce`,
    // which is stepped forward. So both were subject to an explicit stability bound and neither was
    // helped by the implicit step. **Since 2026-09-08 the viscous term is in the implicit divisor
    // and scaled by the tangent stiffness** (docs/stop-element-brief.md); the `alpha` rows below are
    // the bound the OLD scheme sat against, kept because they are the measurement that motivated
    // the change. The elastic term is still explicit, and the elastic-only rows are still the
    // margin it has.
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    constexpr auto tick = 1.0 / 360.0;

    std::printf("\n=== numerical margin at the shipped stop's stiffest reachable point ===\n");
    std::printf("  The corner steps  rate += (F/m) dt  then divides by (1 + (c_damper/m) dt).\n");
    std::printf("  The stop is in F, so its stiffness and its viscous term are BOTH explicit.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto& stop = corner.bumpStop;
        const auto usable = usableStopTravel(corner);

        const auto atLimit = angleForCompression(corner.hardpoints, stop.gap + usable);
        const auto solved = solveCornerWithJacobian(corner.hardpoints, atLimit, 0.0);
        REQUIRE(solved.has_value());

        const auto element = damperElementOf(corner.hardpoints);
        const auto jacobian = solveElement(corner.hardpoints, element, atLimit).lengthPerAngle;

        // The generalised inertia the force pass uses with the geometric load path on, which is the
        // Golf's setting.
        const auto inertia =
            std::max(corner.unsprungMass * glm::dot(solved->wheelCentrePerAngle, solved->wheelCentrePerAngle), 1e-6);

        const auto stiffness = elasticStiffness(stop, usable);
        const auto generalisedStiffness = stiffness * jacobian * jacobian;
        const auto generalisedDamping = stop.damping * jacobian * jacobian;

        const auto omega = std::sqrt(generalisedStiffness / inertia);
        const auto alpha = generalisedDamping * tick / inertia;

        // The damper's own implicit coefficient at the same position, for the comparison the
        // recurrence actually depends on. Its curve slope is read at zero velocity, which is where
        // it is largest on a kneed damper — so `beta` below is the *optimistic* end of the band.
        const auto damperSlope = (corner.damper.at(1e-4) - corner.damper.at(-1e-4)) / 2.0e-4;
        const auto beta = damperSlope * jacobian * jacobian * tick / inertia;

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    deepest reachable stop compression        %8.2f mm\n", 1000.0 * usable);
        std::printf("    elastic tangent stiffness there           %8.0f N/m  (%.2f kN/mm)\n", stiffness,
                    1e-6 * stiffness);
        std::printf("    the corner's own spring, on the shaft     %8.0f N/m\n", corner.springRate);
        std::printf("    so the stop is                            %8.1fx the spring\n", stiffness / corner.springRate);
        std::printf("    the tyre's vertical rate                  %8.0f N/m\n", corner.tireVerticalRate);
        std::printf("    so the stop is                            %8.1fx the tyre\n",
                    stiffness / corner.tireVerticalRate);
        std::printf("    damper Jacobian there                     %8.4f m/rad\n", jacobian);
        std::printf("    generalised inertia                       %8.4f kg.m^2\n", inertia);
        std::printf("    elastic omega * dt                        %8.3f\n", omega * tick);
        std::printf("    stop viscous  alpha = c_q dt / m_q        %8.3f\n", alpha);
        std::printf("    damper implicit beta = c_d dt / m_q       %8.3f\n", beta);

        // **The proper statement, and it is a 2x2 and not a scalar.** With `x` the compression past
        // the gap and `v` the corner's rate, the update the third pass runs is
        //
        //     v' = ((1 - alpha) v - (omega^2 dt) x) / (1 + beta),      x' = x + dt v',
        //
        // whose matrix has determinant exactly `(1 - alpha)/(1 + beta)` and trace
        // `1 + that - omega^2 dt^2/(1 + beta)`. The spectral radius of THAT is the growth per tick.
        // A one-dimensional reading of either term alone is not the answer and disagrees with this.
        const auto spectralRadius = [&](const double depth, const bool withViscous)
        {
            const auto k = elasticStiffness(stop, depth) * jacobian * jacobian;
            const auto a = (1.0 - (withViscous ? alpha : 0.0)) / (1.0 + beta);
            const auto square = (k / inertia) * tick * tick / (1.0 + beta);
            const auto trace = 1.0 + a - square;
            const auto discriminant = trace * trace - 4.0 * a;

            if (discriminant < 0.0)
            {
                return std::sqrt(std::abs(a));
            }

            const auto root = std::sqrt(discriminant);

            return std::max(std::abs(0.5 * (trace + root)), std::abs(0.5 * (trace - root)));
        };

        // Where each crosses one, by scanning the reachable travel. `usable` is the end of the
        // linkage, so "never" means the corner cannot reach an unstable depth at all.
        //
        // **The scan starts above zero deliberately.** At exactly zero compression the stop carries
        // nothing at all — `force()` returns before the arithmetic — so the matrix degenerates to
        // the corner's own free motion, whose unit eigenvalue is a rigid body and not a stop.
        const auto crossing = [&](const bool withViscous)
        {
            for (auto step = 1; step <= 2000; step++)
            {
                const auto depth = usable * static_cast<double>(step) / 2000.0;
                if (spectralRadius(depth, withViscous) > 1.0 + 1e-9)
                {
                    return depth;
                }
            }

            return -1.0;
        };

        std::printf("\n     depth (mm)   spectral radius   elastic only\n");
        for (const auto fraction : {0.01, 0.1, 0.25, 0.5, 0.75, 1.0})
        {
            const auto depth = fraction * usable;
            std::printf("     %10.2f   %15.3f   %12.3f\n", 1000.0 * depth, spectralRadius(depth, true),
                        spectralRadius(depth, false));
        }

        const auto withViscous = crossing(true);
        const auto elasticOnly = crossing(false);

        std::printf("\n     with the shipped viscous term it exceeds one at %s\n",
                    withViscous < 0.0 ? "no reachable depth"
                                      : (std::to_string(1000.0 * withViscous) + " mm of compression").c_str());
        std::printf("     with the static law alone it exceeds one at     %s\n",
                    elasticOnly < 0.0 ? "no reachable depth"
                                      : (std::to_string(1000.0 * elasticOnly) + " mm of compression").c_str());

        // **The invariant this case exists to pin, and it is about the STATIC LAW**, which is this
        // task's subject: the authored stiffness must not on its own put the corner's explicit step
        // outside the unit circle anywhere the linkage can reach. It does not, at either end of the
        // car, with margin.
        CHECK(elasticOnly < 0.0);
        CHECK(spectralRadius(usable, false) < 1.0);
    }

    std::printf("\n  (The viscous rows describe the scheme before 2026-09-08; the constant is now solved\n");
    std::printf("  implicitly and scaled by the tangent stiffness — docs/stop-element-brief.md.)\n");
    std::printf("\n  **The static law is not the numerical problem and the placed viscous constant was.**\n");
    std::printf("  Zero the 40000 N.s/m and the corner's explicit step is inside the unit circle across\n");
    std::printf("  the whole of the reachable travel at both ends of the car — the crossing rows above\n");
    std::printf("  say so. Put it back and both axles leave it well inside their own travel, because an\n");
    std::printf("  explicit viscous term needs `alpha < 2 + beta` and `alpha` is already above two on\n");
    std::printf("  its own at both ends. That is the answer to \"is the placed constant doing numerical\n");
    std::printf("  stabilisation work\": it is not — it is spending the margin the static law leaves.\n");
    std::printf("\n  What stops that showing as a car that shakes itself apart is the stop's own push-only\n");
    std::printf("  clamp. On the release stroke `damping * velocity` goes negative and the whole force\n");
    std::printf("  clamps to zero as soon as it passes the elastic term — at 9 kN of elastic force that\n");
    std::printf("  is 0.225 m/s — so the amplifying half of the cycle is deleted before it can run. The\n");
    std::printf("  linear analysis above is therefore an upper bound on a mode the nonlinearity truncates.\n");
    std::printf("\n  Nothing here is a licence to change the constant, reduce the stiffness or touch the\n");
    std::printf("  integrator. It is the margin, measured, so the next person to move any of the three\n");
    std::printf("  knows which one is close to its bound.\n");
}
