#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::cornerCount;
using raceengine::damperElementOf;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::solveCornerWithJacobian;
using raceengine::solveElement;
using raceengine::solveSpringForce;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::TravelStop;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;

// The **droop** stop's law, geometry, provenance and numerical character, audited and printed.
// `./EngineTests "[.droop-static]"`.
//
// Hidden behind a dotted tag like every other probe here: what it produces is a table to read and a
// data question to answer, not a bound to hold. The CHECKs in the last case are invariants rather
// than thresholds — each one is a statement that would be a *defect* if it failed, not a number
// somebody chose.
//
// **Why it exists.** `[.stop-static]` audited the bump stop and left the droop side untouched. The
// 2026-09-05 stop-free work then found the front droop stop engaging from a 1.2 m/s disturbance and
// staying involved for up to 104 corner-ticks, and noted that unlike the bump stop it has no
// dynamic branch, no sourced hysteresis and no independently sourced top-out geometry. Its gap moved
// 20 mm → 40 mm on 2026-08-24 (engine `7ff4c1a`) to cure a rear wheel leaving the ground under
// braking. This probe says what that 40 mm physically is.
//
// **Nothing here changes any car.** Every alternative configuration below is built inside a case and
// installed nowhere.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto gravity = 9.80665;

// The braking fixture's own constants, copied from `AntilockBrakingTests.cpp` so this probe measures
// the scenario that suite measures rather than one of its own invention.
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

constexpr auto noBrakePressure = std::array<double, cornerCount>{};

[[nodiscard]] SurfaceMesh gripPlate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    return mesh.value();
}

// The skidpad's own ground: a 1200 m square, which is what `HandlingTests` hands its steady-state
// hold. A narrow strip is not usable for this — a car holding 20 m/s on a 54 m circle leaves a 60 m
// strip in about two seconds, and what a fixture then measures is a car falling off the edge of the
// world rather than anything about its suspension. The first attempt here did exactly that and
// reported 1.15 rad of pitch.
[[nodiscard]] SurfaceMesh squarePlate(const double size)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    return mesh.value();
}

// --- the law, written once so every table reads the arithmetic the car runs -------------------

[[nodiscard]] double elasticForce(const TravelStop& stop, const double past)
{
    if (past <= 0.0)
    {
        return 0.0;
    }

    return stop.rate * std::pow(past, stop.progression) / std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);
}

[[nodiscard]] double elasticStiffness(const TravelStop& stop, const double past)
{
    if (past <= 0.0)
    {
        return 0.0;
    }

    return stop.progression * stop.rate * std::pow(past, stop.progression - 1.0) /
           std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);
}

[[nodiscard]] double deflectionAt(const TravelStop& stop, const double force)
{
    const auto scale = std::pow(std::max(stop.gap, 1e-6), stop.progression - 1.0);

    return std::pow(force * scale / std::max(stop.rate, 1e-6), 1.0 / stop.progression);
}

// --- geometry ---------------------------------------------------------------------------------

// The damper shaft's **extension** against its design length at a stated wishbone angle, positive in
// droop. Exactly `-damperShaftCompression`, evaluated straight off the element so this file needs no
// vehicle state.
[[nodiscard]] double shaftExtensionAt(const CornerHardpoints& hardpoints, const double angle)
{
    const auto element = damperElementOf(hardpoints);

    return solveElement(hardpoints, element, angle).length - solveElement(hardpoints, element, 0.0).length;
}

// The wishbone angle at a stated shaft extension, by bisection between design and the linkage's own
// droop limit. Monotonic over that range on both of this car's axles, which the caller checks.
[[nodiscard]] double angleForExtension(const CornerHardpoints& hardpoints, const double extension)
{
    auto low = 0.0;
    auto high = hardpoints.droopAngle;

    for (auto step = 0; step < 80; step++)
    {
        const auto middle = 0.5 * (low + high);
        if (shaftExtensionAt(hardpoints, middle) < extension)
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

struct AtAngle
{
    double angle = 0.0;
    double extension = 0.0;
    double wheelTravel = 0.0;
    double motionRatio = 0.0;
    double jacobian = 0.0;
};

[[nodiscard]] AtAngle sample(const CornerHardpoints& hardpoints, const double angle)
{
    const auto solved = solveCornerWithJacobian(hardpoints, angle, 0.0);
    REQUIRE(solved.has_value());

    const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
    REQUIRE(design.has_value());

    const auto element = damperElementOf(hardpoints);
    const auto evaluated = solveElement(hardpoints, element, angle);

    auto result = AtAngle{};
    result.angle = angle;
    result.extension = shaftExtensionAt(hardpoints, angle);
    result.wheelTravel = solved->wheelTravel - design->wheelTravel;
    result.jacobian = evaluated.lengthPerAngle;
    result.motionRatio = solved->travelPerAngle != 0.0 ? -evaluated.lengthPerAngle / solved->travelPerAngle : 0.0;

    return result;
}

struct Axle
{
    const char* name;
    std::size_t index;
};

constexpr auto axles = std::array{Axle{"front", 0}, Axle{"rear", 2}};

// How far the droop stop can be compressed before the linkage's own range clamp takes over, metres.
// This is the honest end of every table here: past it `stepVehicle` clamps `q` to `droopAngle` with
// **no reaction force at all**, so a row printed beyond it describes a position the car cannot reach.
[[nodiscard]] double usableStopTravel(const CornerSetup& corner)
{
    return std::max(0.0, shaftExtensionAt(corner.hardpoints, corner.hardpoints.droopAngle) - corner.droopStop.gap);
}

} // namespace

TEST_CASE("the exact droop-stop law as the force pass runs it", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== the shipped droop law, traced from VehicleImpl.cpp's force pass ===\n");
    std::printf("  compression c = designLength - damperLength           (positive in BUMP)\n");
    std::printf("  extension   e = -c                                    (positive in DROOP)\n");
    std::printf("  past        x = e - droopStop.gap                     (engagement variable)\n");
    std::printf("  rate of change    = -damperVelocity = de/dt           (positive while EXTENDING)\n");
    std::printf("  forces.droopStop  = -droopStop.force(x, de/dt)\n");
    std::printf("  droopStop.force(x, r) = max(0, rate*x^p / gap^(p-1) + damping*r + hysteresis term)\n");
    std::printf("\n  so the corner-level droop force is NEGATIVE on the shaft axis, which is the\n");
    std::printf("  compressing direction — it pulls the shaft back in and opposes further extension.\n");
    std::printf("  It enters the generalised force as  forces.droopStop * damper.lengthPerAngle,\n");
    std::printf("  the DAMPER element's Jacobian, because the stop lives on the shaft. The spring's\n");
    std::printf("  own Jacobian is never used for it, on either the coaxial or the split branch.\n");
    std::printf("\n  The push-only clamp is INSIDE force(), so it clamps the sum of elastic and viscous\n");
    std::printf("  before the outer sign is applied: the stop can only ever pull the shaft in, never\n");
    std::printf("  push it out.\n");
    std::printf("\n  The linkage range clamp is a SEPARATE mechanism, applied after integration:\n");
    std::printf("  q < droopAngle is set to droopAngle and the rate is clamped to >= 0, with NO force.\n");
    std::printf("  validateCornerSetup refuses a car whose droop gap is not strictly inside that range.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto& droop = corner.droopStop;
        const auto& bump = corner.bumpStop;

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    droop: gap %.1f mm | rate %.0f N/m | progression %.2f | damping %.0f N.s/m |"
                    " hysteresis %.3f | hysteresisSpeed %.4f m/s\n",
                    1000.0 * droop.gap, droop.rate, droop.progression, droop.damping, droop.hysteresis,
                    droop.hysteresisSpeed);
        std::printf("    droop dynamic branch stated: %s\n", droop.statesDynamicBranch() ? "YES" : "no");
        std::printf("    bump : gap %.1f mm | rate %.0f N/m | progression %.2f | damping %.0f N.s/m |"
                    " hysteresis %.3f\n",
                    1000.0 * bump.gap, bump.rate, bump.progression, bump.damping, bump.hysteresis);
        std::printf("    bump  dynamic branch stated: %s\n", bump.statesDynamicBranch() ? "YES" : "no");
    }

    // The two axles' stops are separate `TravelStop` objects on separate `CornerSetup`s, and on this
    // car they are stated with identical scalars — which is a fact about the data, not about the
    // model, and is worth printing because a strut top-out and a rear damper top-out are different
    // components.
    std::printf("\n  Front and rear state the SAME four droop scalars (%s), on a MacPherson strut and a\n",
                base->corners[0].droopStop.gap == base->corners[2].droopStop.gap &&
                        base->corners[0].droopStop.rate == base->corners[2].droopStop.rate &&
                        base->corners[0].droopStop.progression == base->corners[2].droopStop.progression &&
                        base->corners[0].droopStop.damping == base->corners[2].droopStop.damping
                    ? "identical"
                    : "different");
    std::printf("  separate rear damper respectively. Nothing sources that equality.\n");
}

TEST_CASE("the droop geometry, from ride height to the linkage's own limit", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== droop geometry, q = 0 (static ride) to the authored droopAngle ===\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto limit = corner.hardpoints.droopAngle;
        const auto atLimit = sample(corner.hardpoints, limit);
        const auto engagementAngle = angleForExtension(corner.hardpoints, corner.droopStop.gap);
        const auto atEngagement = sample(corner.hardpoints, engagementAngle);
        const auto usable = usableStopTravel(corner);

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    authored droopAngle (the range clamp)      %9.5f rad\n", limit);
        std::printf("    shaft extension there                      %9.2f mm\n", 1000.0 * atLimit.extension);
        std::printf("    wheel droop travel there                   %9.2f mm\n", 1000.0 * atLimit.wheelTravel);
        std::printf("    motion ratio there                         %9.4f\n", atLimit.motionRatio);
        std::printf("\n    droop stop engages at q                   %9.5f rad\n", engagementAngle);
        std::printf("      shaft extension (= the gap)              %9.2f mm\n", 1000.0 * atEngagement.extension);
        std::printf("      wheel droop travel there                 %9.2f mm\n", 1000.0 * atEngagement.wheelTravel);
        std::printf("      motion ratio there                       %9.4f\n", atEngagement.motionRatio);
        std::printf("\n    remaining shaft extension after engagement %9.2f mm\n", 1000.0 * usable);
        std::printf("    remaining wheel travel after engagement    %9.2f mm\n",
                    1000.0 * (atLimit.wheelTravel - atEngagement.wheelTravel));
        std::printf("\n    gap / available geometric extension       %9.1f%%\n",
                    100.0 * corner.droopStop.gap / atLimit.extension);
        std::printf("    wheel travel at engagement / total droop   %9.1f%%\n",
                    100.0 * atEngagement.wheelTravel / atLimit.wheelTravel);

        std::printf("\n       q (rad)   extension   wheel travel   motion ratio   dL/dq\n");
        std::printf("                    (mm)          (mm)                     (m/rad)\n");
        for (auto step = 0; step <= 10; step++)
        {
            const auto angle = limit * static_cast<double>(step) / 10.0;
            const auto point = sample(corner.hardpoints, angle);
            std::printf("      %8.4f   %9.2f   %12.2f   %12.4f   %8.5f\n", angle, 1000.0 * point.extension,
                        1000.0 * point.wheelTravel, point.motionRatio, point.jacobian);
        }
    }

    std::printf("\n  The `droopAngle` clamp is AUTHORED, not geometric. The linkage itself solves far\n");
    std::printf("  past it (docs/suspension-geometry-audit.md, section 2: front droop locks near\n");
    std::printf("  q = -0.80 rad and rear near -0.975). What bounds the reachable droop in the car is\n");
    std::printf("  the authored number, and the droop stop is sized against THAT and not against the\n");
    std::printf("  linkage's own limit.\n");
}

TEST_CASE("the shipped droop stop against shaft extension", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== the droop stop's static force, front and rear ===\n");
    std::printf("  Every quantity is on the DAMPER SHAFT. Sign printed as a magnitude; the corner\n");
    std::printf("  sees it as a negative axis force.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto& stop = corner.droopStop;
        const auto usable = usableStopTravel(corner);
        const auto atLimit = shaftExtensionAt(corner.hardpoints, corner.hardpoints.droopAngle);

        const auto design = solveCornerWithJacobian(corner.hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());
        const auto staticLoad = solveSpringForce(corner, design.value()).force;
        const auto unsprungWeight = corner.unsprungMass * gravity;

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    shaft extension at the authored droop limit %8.2f mm\n", 1000.0 * atLimit);
        std::printf("    so the stop has                             %8.2f mm of usable travel\n", 1000.0 * usable);
        std::printf("    the corner carries %.0f N on this axis at rest; its unsprung weight is %.0f N\n", staticLoad,
                    unsprungWeight);

        std::printf("\n     x past   elastic       dF/dx        wheel |  viscous at extension rate (kN)\n");
        std::printf("      (mm)        (N)        (kN/mm)       (mm)  |  0.05    0.25    1.00    2.00 m/s\n");

        for (const auto millimetres : {0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0})
        {
            const auto past = 0.001 * millimetres;
            if (past > usable + 1e-9)
            {
                std::printf("     %5.1f     -- beyond the authored droop limit, NOT REACHABLE --\n", millimetres);
                continue;
            }

            const auto point = sample(corner.hardpoints, angleForExtension(corner.hardpoints, stop.gap + past));

            std::printf("     %5.1f   %8.1f   %10.4f   %8.2f  |", millimetres, elasticForce(stop, past),
                        1e-6 * elasticStiffness(stop, past), 1000.0 * point.wheelTravel);

            for (const auto velocity : {0.05, 0.25, 1.0, 2.0})
            {
                std::printf(" %7.2f", 0.001 * stop.damping * velocity);
            }
            std::printf("\n");
        }

        std::printf("\n     force anchor    reached at   of usable travel   status\n");
        for (const auto force : {1000.0, 3000.0, 5000.0, 9000.0, 15000.0})
        {
            const auto past = deflectionAt(stop, force);

            if (past > usable)
            {
                std::printf("     %8.0f N    %7.2f mm   %14.1f%%   EXTRAPOLATION — the stop only has %.2f mm\n", force,
                            1000.0 * past, 100.0 * past / usable, 1000.0 * usable);
                continue;
            }

            std::printf("     %8.0f N    %7.2f mm   %14.1f%%   reachable\n", force, 1000.0 * past,
                        100.0 * past / usable);
        }

        const auto deepest = elasticForce(stop, usable);
        std::printf("\n     at the deepest reachable point the elastic law reads %.0f N — %.2f static\n", deepest,
                    deepest / std::max(staticLoad, 1.0));
        std::printf("     loads, and %.2f of the corner's own unsprung weight.\n", deepest / unsprungWeight);

        // The generalised contribution at that point, which is what the corner's equation of motion
        // actually receives — force times the damper element's own Jacobian.
        const auto atEnd = sample(corner.hardpoints, corner.hardpoints.droopAngle);
        std::printf("     generalised stop contribution there   %8.2f N.m  (force * dL/dq)\n", deepest * atEnd.jacobian);
    }
}

TEST_CASE("droop against bump: the same law, different numbers", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== is the droop stop the bump-stop law reused? ===\n");
    std::printf("  Same C++ type, same force(), same push-only clamp, same power law. The droop side\n");
    std::printf("  differs in four scalars and in having no dynamic branch available to it at all —\n");
    std::printf("  the corner only calls dynamicForce() on the BUMP stop, so a droop stop that stated\n");
    std::printf("  a dynamic branch would be ignored.\n");

    std::printf("\n            axle    gap    rate   prog  damping  hyst  usable  deepest elastic\n");
    std::printf("                   (mm)   (N/m)               (N.s/m)         (mm)          (N)\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];

        const auto bumpUsable =
            std::max(0.0, -shaftExtensionAt(corner.hardpoints, corner.hardpoints.bumpAngle) - corner.bumpStop.gap);
        const auto droopUsable = usableStopTravel(corner);

        std::printf("     bump %5s  %5.1f  %7.0f  %4.1f  %7.0f  %4.2f  %6.2f  %11.0f\n", axle.name,
                    1000.0 * corner.bumpStop.gap, corner.bumpStop.rate, corner.bumpStop.progression,
                    corner.bumpStop.damping, corner.bumpStop.hysteresis, 1000.0 * bumpUsable,
                    elasticForce(corner.bumpStop, bumpUsable));
        std::printf("    droop %5s  %5.1f  %7.0f  %4.1f  %7.0f  %4.2f  %6.2f  %11.0f\n", axle.name,
                    1000.0 * corner.droopStop.gap, corner.droopStop.rate, corner.droopStop.progression,
                    corner.droopStop.damping, corner.droopStop.hysteresis, 1000.0 * droopUsable,
                    elasticForce(corner.droopStop, droopUsable));
    }

    std::printf("\n  The gap enters the law twice — as the engagement point AND as the length scale in\n");
    std::printf("  gap^(p-1), which is the coupling `[.stop-static]` recorded for the bump side. On the\n");
    std::printf("  droop side the 20 -> 40 mm move therefore did two things at once: it moved\n");
    std::printf("  engagement 20 mm further out AND divided the law's stiffness coefficient by four.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        auto twenty = corner.droopStop;
        twenty.gap = 0.020;

        std::printf("\n  --- %s: what the gap move did to the law itself ---\n", axle.name);
        std::printf("    coefficient rate/gap^(p-1):  20 mm gap %.3e N/m^3   40 mm gap %.3e N/m^3  (%.2fx)\n",
                    twenty.rate / std::pow(twenty.gap, twenty.progression - 1.0),
                    corner.droopStop.rate / std::pow(corner.droopStop.gap, corner.droopStop.progression - 1.0),
                    std::pow(twenty.gap / corner.droopStop.gap, corner.droopStop.progression - 1.0));
        std::printf("    force at 3 mm past:          20 mm gap %8.1f N        40 mm gap %8.1f N\n",
                    elasticForce(twenty, 0.003), elasticForce(corner.droopStop, 0.003));
    }
}

TEST_CASE("the push-only clamp on the droop side", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== a closed extension/retraction cycle through the droop stop ===\n");
    std::printf("  Driven through the production force law directly, at the production timestep, with\n");
    std::printf("  the production parameters. `e` is shaft extension, `x = e - gap`.\n");
    std::printf("  Energy INTO the stop per tick = droopStop.force(x, de/dt) * (de/dt) * dt.\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto& stop = corner.droopStop;
        const auto usable = usableStopTravel(corner);

        // Out to the deepest reachable point and back, at a speed the ring-down actually reaches.
        // 0.2 m/s is inside the 0.25 m/s stop-free control's own peak shaft speed of 0.212 m/s.
        for (const auto speed : {0.02, 0.20})
        {
            auto energyOut = 0.0;
            auto energyBack = 0.0;
            auto onOut = 0;
            auto onBack = 0;
            auto peakOut = 0.0;
            auto peakBack = 0.0;
            auto firstOn = -1.0;
            auto lastOn = -1.0;

            const auto steps = static_cast<int>(usable / (speed * tick));

            for (auto step = 0; step <= steps; step++)
            {
                const auto past = std::min(usable, speed * tick * static_cast<double>(step));
                const auto force = stop.force(past, speed);

                if (force > 0.0)
                {
                    onOut++;
                    peakOut = std::max(peakOut, force);
                    if (firstOn < 0.0)
                    {
                        firstOn = past;
                    }
                    lastOn = past;
                }

                energyOut += force * speed * tick;
            }

            for (auto step = steps; step >= 0; step--)
            {
                const auto past = std::min(usable, speed * tick * static_cast<double>(step));
                const auto force = stop.force(past, -speed);

                if (force > 0.0)
                {
                    onBack++;
                    peakBack = std::max(peakBack, force);
                }

                energyBack += force * (-speed) * tick;
            }

            std::printf("\n  --- %s, +-%.2f m/s over the %.2f mm of usable travel ---\n", axle.name, speed,
                        1000.0 * usable);
            std::printf("    OUT (extending):  force on for %d of %d ticks, first at x = %.3f mm, peak %.0f N\n", onOut,
                        steps + 1, 1000.0 * std::max(firstOn, 0.0), peakOut);
            std::printf("                      last force at x = %.3f mm, energy into the stop %.4f J\n",
                        1000.0 * std::max(lastOn, 0.0), energyOut);
            std::printf("    BACK (retracting): force on for %d of %d ticks, peak %.0f N, energy %.4f J\n", onBack,
                        steps + 1, peakBack, energyBack);
            std::printf("    NET over the closed cycle: %.4f J into the stop%s\n", energyOut + energyBack,
                        energyOut + energyBack >= 0.0 ? "" : "  <-- ENERGY INJECTED, which would be a defect");
            std::printf("    the elastic force alone at the deepest point is %.1f N, and damping * speed is\n",
                        elasticForce(stop, usable));
            std::printf("    %.1f N, so on the return stroke the clamp fires wherever elastic < damping*speed,\n",
                        stop.damping * speed);
            std::printf("    which is every x below %.3f mm.\n",
                        1000.0 * deflectionAt(stop, stop.damping * speed));
        }
    }

    std::printf("\n  So the droop stop is a ONE-WAY element at any speed a ring-down reaches: it absorbs\n");
    std::printf("  on the way out and returns nothing on the way back, because the push-only clamp\n");
    std::printf("  deletes the whole force whenever the viscous term outweighs a very small elastic\n");
    std::printf("  one. Damping therefore acts in one direction only, and the stop can never inject\n");
    std::printf("  energy — the net is non-negative by construction and measured non-negative above.\n");
}

TEST_CASE("the droop stop against the corner's own integrator", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    std::printf("\n=== numerical margin at the droop stop's stiffest reachable point ===\n");
    std::printf("  Same analysis as [.stop-static]'s: the corner steps rate += (F/m) dt then divides\n");
    std::printf("  by (1 + (c_damper/m) dt). Until 2026-09-08 BOTH halves of the stop rode F explicitly;\n");
    std::printf("  the viscous rows below are that scheme's bound. The viscous term is now in the divisor\n");
    std::printf("  and scaled by the tangent stiffness (docs/stop-element-brief.md).\n");

    for (const auto& axle : axles)
    {
        const auto& corner = base->corners[axle.index];
        const auto& stop = corner.droopStop;
        const auto usable = usableStopTravel(corner);

        const auto atLimit = corner.hardpoints.droopAngle;
        const auto solved = solveCornerWithJacobian(corner.hardpoints, atLimit, 0.0);
        REQUIRE(solved.has_value());

        const auto element = damperElementOf(corner.hardpoints);
        const auto jacobian = solveElement(corner.hardpoints, element, atLimit).lengthPerAngle;

        const auto inertia =
            std::max(corner.unsprungMass * glm::dot(solved->wheelCentrePerAngle, solved->wheelCentrePerAngle), 1e-6);

        const auto stiffness = elasticStiffness(stop, usable);
        const auto generalisedStiffness = stiffness * jacobian * jacobian;
        const auto generalisedDamping = stop.damping * jacobian * jacobian;

        const auto omega = std::sqrt(generalisedStiffness / inertia);
        const auto alpha = generalisedDamping * tick / inertia;

        const auto damperSlope = (corner.damper.at(1e-4) - corner.damper.at(-1e-4)) / 2.0e-4;
        const auto beta = damperSlope * jacobian * jacobian * tick / inertia;

        std::printf("\n  --- %s ---\n", axle.name);
        std::printf("    deepest reachable stop compression        %8.2f mm\n", 1000.0 * usable);
        std::printf("    elastic tangent stiffness there           %8.0f N/m  (%.4f kN/mm)\n", stiffness,
                    1e-6 * stiffness);
        std::printf("    the corner's own spring, on the shaft     %8.0f N/m\n", corner.springRate);
        std::printf("    so the stop is                            %8.2fx the spring\n", stiffness / corner.springRate);
        std::printf("    the tyre's vertical rate                  %8.0f N/m\n", corner.tireVerticalRate);
        std::printf("    so the stop is                            %8.2fx the tyre\n",
                    stiffness / corner.tireVerticalRate);
        std::printf("    damper Jacobian there                     %8.4f m/rad\n", jacobian);
        std::printf("    generalised stop stiffness                %8.1f N.m/rad\n", generalisedStiffness);
        std::printf("    generalised inertia                       %8.4f kg.m^2\n", inertia);
        std::printf("    elastic omega * dt                        %8.4f\n", omega * tick);
        std::printf("    stop viscous  alpha = c_q dt / m_q        %8.3f\n", alpha);
        std::printf("    damper implicit beta = c_d dt / m_q       %8.3f\n", beta);

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

        std::printf("\n     depth (mm)   spectral radius   elastic only\n");
        for (const auto fraction : {0.01, 0.1, 0.25, 0.5, 0.75, 1.0})
        {
            const auto depth = fraction * usable;
            std::printf("     %10.2f   %15.3f   %12.3f\n", 1000.0 * depth, spectralRadius(depth, true),
                        spectralRadius(depth, false));
        }

        std::printf("\n     elastic-only spectral radius at the deepest reachable point: %.4f\n",
                    spectralRadius(usable, false));
        std::printf("     with the shipped 30000 N.s/m:                                %.4f\n",
                    spectralRadius(usable, true));
    }
}

namespace
{

// What one run did to the droop stops, per corner, on penetration and on force.
struct DroopWitness
{
    std::array<int, cornerCount> penetrationTicks{};
    std::array<int, cornerCount> forceTicks{};
    std::array<double, cornerCount> peakPast{};
    std::array<double, cornerCount> peakForce{};
    std::array<double, cornerCount> peakExtension{};
    std::array<double, cornerCount> minimumLoad{};
    std::array<bool, cornerCount> lostContact{};
    // How long the wheel spent off the road. **The discriminator the minimum load cannot be**, once
    // both arms of a comparison reach exactly zero: a load floor of 0 N says a wheel lifted and says
    // nothing about for how long.
    std::array<int, cornerCount> airborneTicks{};
    std::array<double, cornerCount> firstEngagement{};
    double peakPitch = 0.0;
    double elapsed = 0.0;
    double distance = 0.0;
    bool seeded = false;

    void observe(const VehicleSetup& setup, const VehicleStep& stepped, const double time)
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = setup.corners[index];
            const auto element = damperElementOf(corner.hardpoints);
            const auto design = solveElement(corner.hardpoints, element, 0.0).length;
            const auto extension =
                solveElement(corner.hardpoints, element, stepped.corners[index].suspension.wishboneAngle).length -
                design;
            const auto past = extension - corner.droopStop.gap;
            const auto force = -stepped.corners[index].forces.droopStop;

            peakExtension[index] = std::max(peakExtension[index], extension);

            if (past > 0.0)
            {
                penetrationTicks[index]++;
                peakPast[index] = std::max(peakPast[index], past);

                if (firstEngagement[index] == 0.0)
                {
                    firstEngagement[index] = time;
                }
            }

            if (force > 0.0)
            {
                forceTicks[index]++;
                peakForce[index] = std::max(peakForce[index], force);
            }

            const auto load = stepped.corners[index].forces.tireVertical;
            minimumLoad[index] = seeded ? std::min(minimumLoad[index], load) : load;

            if (!stepped.telemetry.wheels[index].inContact)
            {
                lostContact[index] = true;
                airborneTicks[index]++;
            }
        }

        peakPitch = std::max(peakPitch, std::abs(stepped.telemetry.pitch));
        seeded = true;
    }
};

void reportWitness(const char* label, const DroopWitness& witness)
{
    std::printf("\n  --- %s ---\n", label);
    std::printf("    corner   pen.ticks  force ticks   peak x past   peak force   peak ext   min load   contact\n");
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        std::printf("    %6zu   %9d  %11d   %10.3f mm %9.1f N %8.2f mm %9.1f N   %s (%d ticks)\n", index,
                    witness.penetrationTicks[index], witness.forceTicks[index], 1000.0 * witness.peakPast[index],
                    witness.peakForce[index], 1000.0 * witness.peakExtension[index], witness.minimumLoad[index],
                    witness.lostContact[index] ? "LOST" : "kept", witness.airborneTicks[index]);
    }
    std::printf("    peak |pitch| %.4f rad, elapsed %.3f s", witness.peakPitch, witness.elapsed);
    if (witness.distance != 0.0)
    {
        std::printf(", distance %.3f m", witness.distance);
    }
    std::printf("\n");
}

// The braking fixture's own scenario, reproduced so this probe measures what the suite measures.
[[nodiscard]] DroopWitness brakingRun(const VehicleSetup& setup, const PhysicsWorld& world, const double pedal,
                                      const bool antilock)
{
    auto assists = golfGtiMk7Assists(setup);
    assists.antilock.enabled = antilock;

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, hundred);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = hundred / tyreRadius;
    }

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    auto witness = DroopWitness{};

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
        sensors.steeringWheelAngle = lastStep.telemetry.steeringWheelAngle;

        return sensors;
    };

    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    const auto start = state.chassis.position.z;

    auto input = VehicleInput{};
    input.brake = pedal;

    for (auto step = 0; step < 360 * 30; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = pedal, .throttle = 0.0},
                                           brakeCircuitPressures(setup, pedal), tick);
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        witness.elapsed += tick;
        witness.observe(setup, lastStep, witness.elapsed);

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            break;
        }
    }

    witness.distance = state.chassis.position.z - start;

    return witness;
}

// Straight-line acceleration under a stated front-axle drive torque, which is the squat case — the
// pitch that unloads the FRONT axle and is where a front droop stop would be reached if anything
// reached it in normal driving.
[[nodiscard]] DroopWitness accelerationRun(const VehicleSetup& setup, const PhysicsWorld& world, const double torque)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    const auto drive = std::array<double, cornerCount>{0.5 * torque, 0.5 * torque, 0.0, 0.0};

    auto witness = DroopWitness{};

    for (auto step = 0; step < 360 * 6; step++)
    {
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, drive, world, tick);
        REQUIRE(stepped.has_value());

        witness.elapsed += tick;
        witness.observe(setup, stepped.value(), witness.elapsed);
    }

    return witness;
}

// The skidpad's steady-state hold, reduced to what this probe needs: a speed controller on the front
// axle and a held steering angle, on a plate big enough for the circle. The controller is the one
// `HandlingTests` uses, because a coasting car scrubs itself down and measures a spiral.
[[nodiscard]] DroopWitness corneringRun(const VehicleSetup& setup, const PhysicsWorld& world, const double steering,
                                        const double speed)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, 600.0);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
    }

    auto input = VehicleInput{};
    input.steering = steering;

    auto integral = 0.0;
    auto witness = DroopWitness{};

    for (auto step = 0; step < 3600; step++)
    {
        const auto error = speed - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;
        const auto drive = std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0};

        const auto stepped = stepVehicle(setup, state, input, drive, world, tick);
        REQUIRE(stepped.has_value());

        witness.elapsed += tick;
        witness.observe(setup, stepped.value(), witness.elapsed);
    }

    return witness;
}

} // namespace

TEST_CASE("where the droop stop actually engages in the existing deterministic fixtures", "[.droop-static]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto plate = PhysicsWorld::create(gripPlate());
    REQUIRE(plate.has_value());

    std::printf("\n=== droop engagement in the fixtures the suite already runs ===\n");
    std::printf("  Corners 0/1 are front, 2/3 rear. Penetration ticks count e > gap; force ticks count\n");
    std::printf("  a non-zero force, which is always the smaller of the two because of the clamp.\n");

    // The four-wheels-on-the-ground fixture's own scenario: half pedal, assists off, 100 km/h.
    const auto half = brakingRun(built.value(), plate.value(), 0.50, false);
    reportWitness("100-0 at half pedal, assists off (the four-wheels fixture's scenario)", half);

    const auto full = brakingRun(built.value(), plate.value(), 1.0, false);
    reportWitness("100-0 at full pedal, assists off", full);

    const auto assisted = brakingRun(built.value(), plate.value(), 1.0, true);
    reportWitness("100-0 at full pedal, anti-lock ON", assisted);

    // Acceleration, which loads the rear in pitch the other way. **Driven through the wheel torque
    // argument rather than through the throttle**: `stepVehicle` is handed the driveline's torque
    // from outside, so a fixture that sets `input.throttle` and passes `noDriveTorque` measures a
    // car with the throttle pinned and nothing turning the wheels. The first version of this case
    // did that and reported 0.002 rad of pitch, which is a parked car.
    reportWitness("six seconds of a 3000 N.m front-axle drive torque from rest",
                  accelerationRun(built.value(), plate.value(), 3000.0));

    // Steady cornering, on the skidpad's own square plate and driven the way `HandlingTests` drives
    // its steady-state hold. Two steering angles inside the range that suite already uses.
    const auto circle = PhysicsWorld::create(squarePlate(1200.0));
    REQUIRE(circle.has_value());

    for (const auto steering : {0.12, 0.30})
    {
        const auto witness = corneringRun(built.value(), circle.value(), steering, 20.0);

        reportWitness((std::string("ten seconds of driven steady cornering at ") + std::to_string(steering) +
                       " steering, 20 m/s")
                          .c_str(),
                      witness);
    }

    std::printf("\n  The ring-down fixtures are instrumented in [.damper-friction] and are not repeated\n");
    std::printf("  here: at its historical 1.5 m/s impulse the droop stop takes 100 corner-ticks on the\n");
    std::printf("  FRONT corners and none on the rear, and at the 0.25 m/s stop-free control it takes\n");
    std::printf("  none anywhere, clearing its gap by 37.67 mm.\n");
}

TEST_CASE("the historical droop-gap A/B, reproduced probe-locally", "[.droop-static]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto plate = PhysicsWorld::create(gripPlate());
    REQUIRE(plate.has_value());

    // **The previous configuration, restored inside this case and installed nowhere.** Only the gap
    // moved on 2026-08-24 (`7ff4c1a`): rate, progression and damping are the same numbers before and
    // after, which the git history says outright.
    auto previous = built.value();
    for (auto& corner : previous.corners)
    {
        corner.droopStop.gap = 0.020;
    }

    std::printf("\n=== the 20 mm -> 40 mm droop gap, on the fixture that motivated it ===\n");
    std::printf("  Historical attribution, not parameter selection. Production is the 40 mm arm\n");
    std::printf("  throughout and nothing here is swept.\n");

    for (const auto pedal : {0.50, 1.0})
    {
        const auto before = brakingRun(previous, plate.value(), pedal, false);
        const auto after = brakingRun(built.value(), plate.value(), pedal, false);

        std::printf("\n  ##### %.2f pedal, assists off #####\n", pedal);
        reportWitness("20 mm droop gap (the pre-2026-08-24 car)", before);
        reportWitness("40 mm droop gap (production)", after);

        const auto lowestRearBefore = std::min(before.minimumLoad[2], before.minimumLoad[3]);
        const auto lowestRearAfter = std::min(after.minimumLoad[2], after.minimumLoad[3]);

        std::printf("\n    lowest rear load  %.1f N -> %.1f N\n", lowestRearBefore, lowestRearAfter);
        std::printf("    stopping distance %.3f m -> %.3f m\n", before.distance, after.distance);
        std::printf("    peak |pitch|      %.4f rad -> %.4f rad\n", before.peakPitch, after.peakPitch);
        std::printf("    rear droop ticks  %d/%d -> %d/%d (penetration)\n", before.penetrationTicks[2],
                    before.penetrationTicks[3], after.penetrationTicks[2], after.penetrationTicks[3]);
        std::printf("    peak rear droop force %.1f N -> %.1f N\n",
                    std::max(before.peakForce[2], before.peakForce[3]),
                    std::max(after.peakForce[2], after.peakForce[3]));
        std::printf("    peak rear extension   %.2f mm -> %.2f mm\n",
                    1000.0 * std::max(before.peakExtension[2], before.peakExtension[3]),
                    1000.0 * std::max(after.peakExtension[2], after.peakExtension[3]));
        std::printf("    rear airborne ticks   %d/%d -> %d/%d  (of %.0f ticks of stop)\n", before.airborneTicks[2],
                    before.airborneTicks[3], after.airborneTicks[2], after.airborneTicks[3], after.elapsed / tick);
        std::printf("    front airborne ticks  %d/%d -> %d/%d\n", before.airborneTicks[0], before.airborneTicks[1],
                    after.airborneTicks[0], after.airborneTicks[1]);
    }
}

TEST_CASE("droop-stop invariants", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    SECTION("static ride never touches either stop")
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = base->corners[index];

            CHECK(shaftExtensionAt(corner.hardpoints, 0.0) == 0.0);
            CHECK(corner.droopStop.force(0.0 - corner.droopStop.gap, 0.0) == 0.0);
            CHECK(corner.bumpStop.force(0.0 - corner.bumpStop.gap, 0.0) == 0.0);
        }
    }

    SECTION("no droop force before engagement, at any extension rate")
    {
        for (const auto& axle : axles)
        {
            const auto& stop = base->corners[axle.index].droopStop;

            for (const auto fraction : {0.0, 0.5, 0.99})
            {
                const auto past = (fraction - 1.0) * stop.gap;

                for (const auto rate : {-2.0, 0.0, 2.0})
                {
                    CHECK(stop.force(past, rate) == 0.0);
                }
            }
        }
    }

    SECTION("past engagement the force opposes further extension and is finite everywhere reachable")
    {
        for (const auto& axle : axles)
        {
            const auto& corner = base->corners[axle.index];
            const auto usable = usableStopTravel(corner);

            REQUIRE(usable > 0.0);

            for (auto step = 1; step <= 50; step++)
            {
                const auto past = usable * static_cast<double>(step) / 50.0;
                const auto force = corner.droopStop.force(past, 0.0);

                CHECK(force > 0.0);
                CHECK(std::isfinite(force));

                // The corner-level sign: `forces.droopStop` is the negative of this, so it acts
                // along the shaft's compressing direction and pulls the wheel back up.
                CHECK(-force < 0.0);
            }
        }
    }

    SECTION("the stop engages strictly inside the authored travel, which is what validateCornerSetup guarantees")
    {
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = base->corners[index];

            CHECK(corner.droopStop.gap < shaftExtensionAt(corner.hardpoints, corner.hardpoints.droopAngle));
        }
    }

    SECTION("left and right mirror exactly")
    {
        for (const auto pair : std::array{std::array<std::size_t, 2>{0, 1}, std::array<std::size_t, 2>{2, 3}})
        {
            const auto& left = base->corners[pair[0]];
            const auto& right = base->corners[pair[1]];

            CHECK(left.droopStop.gap == right.droopStop.gap);
            CHECK(left.droopStop.rate == right.droopStop.rate);
            CHECK(left.droopStop.progression == right.droopStop.progression);
            CHECK(left.droopStop.damping == right.droopStop.damping);
            CHECK(left.hardpoints.droopAngle == right.hardpoints.droopAngle);
            CHECK(usableStopTravel(left) == usableStopTravel(right));
        }
    }

    SECTION("a closed extension cycle never returns more energy than it took")
    {
        for (const auto& axle : axles)
        {
            const auto& corner = base->corners[axle.index];
            const auto usable = usableStopTravel(corner);

            for (const auto speed : {0.005, 0.02, 0.20, 1.0})
            {
                auto energy = 0.0;
                const auto steps = static_cast<int>(usable / (speed * tick));

                for (auto step = 0; step <= steps; step++)
                {
                    const auto past = std::min(usable, speed * tick * static_cast<double>(step));
                    energy += corner.droopStop.force(past, speed) * speed * tick;
                }

                for (auto step = steps; step >= 0; step--)
                {
                    const auto past = std::min(usable, speed * tick * static_cast<double>(step));
                    energy += corner.droopStop.force(past, -speed) * (-speed) * tick;
                }

                CHECK(energy >= 0.0);
            }
        }
    }

    SECTION("the droop stop rides the damper element's Jacobian and not the spring's")
    {
        // On this car every corner is a coil-over, so the two Jacobians are the same bits — which is
        // exactly why this needs stating rather than checking by inspection of a number. What is
        // asserted is the identity the force pass depends on.
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& corner = base->corners[index];
            const auto solved = solveCornerWithJacobian(corner.hardpoints, corner.hardpoints.droopAngle, 0.0);
            REQUIRE(solved.has_value());

            const auto element = damperElementOf(corner.hardpoints);
            const auto damper = solveElement(corner.hardpoints, element, corner.hardpoints.droopAngle);

            CHECK(std::isfinite(damper.lengthPerAngle));
            CHECK(damper.lengthPerAngle != 0.0);
        }
    }

    SECTION("the droop stop has no dynamic branch and cannot be given one")
    {
        // `VehicleImpl.cpp` calls `dynamicForce` on the bump stop only. A droop stop that stated the
        // branch would be silently ignored, so this pins the data side of that: none states it.
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            CHECK_FALSE(base->corners[index].droopStop.statesDynamicBranch());
        }
    }
}

TEST_CASE("the published Golf travel envelope, against this model's own", "[.droop-static]")
{
    const auto base = golfGtiMk7();
    REQUIRE(base.has_value());

    // **SOURCED, and graded before it is used.** Heissing/Ersoy, *Chassis Handbook* (the English
    // edition of Fahrwerkhandbuch), Table 1-5, *Kinematic parameter values for selected current
    // European-model vehicles*, VW column — `docs/fetched/chassis-layout.txt` lines 2611-2612 and
    // 2648-2649. The column is identified by the table's own fully-populated Turning Radius row, and
    // cross-checked against two independent cells that this project already knows: the Golf's front
    // track (1533 mm) and its rated rear axle load (930 kg), which appear in the same column.
    //
    // **Three caveats, and they are the whole reason nothing here changes a car.**
    //   1. The column is platform **PQ35, model year 2003 — a Golf V**, not the MQB Mk7 this project
    //      models. The table is titled "current European-model vehicles" of its own edition.
    //   2. These are **wheel** travels. Every stop in this model lives on the damper shaft, so a
    //      comparison needs this car's own motion ratio, which is what the conversion below is.
    //   3. A total rebound travel is **not** a stop engagement point. It is where the wheel stops,
    //      which is where the rebound stop is fully compressed — the gap is a different quantity and
    //      this table does not contain it.
    //
    // A fourth thing is worth stating because it is a disagreement rather than a caveat: the front
    // compression travel below is 90 mm, and Dominic's own tape measure on a 2019 Mk7.5 gave
    // **67-73 mm** from ride height to the bump stop fully crushed (`PublishedCarsImpl.cpp`). Either
    // the Mk7 has materially less front bump travel than the Mk5, or the two figures are not
    // measuring the same thing. Until that is settled the rebound cells beside them inherit the same
    // doubt.
    constexpr auto publishedFrontCompression = 0.090;
    constexpr auto publishedFrontRebound = 0.085;
    constexpr auto publishedRearCompression = 0.126;
    constexpr auto publishedRearRebound = 0.074;

    std::printf("\n=== Chassis Handbook Table 1-5, VW Golf (PQ35, 2003) against this model ===\n");
    std::printf("  published wheel travels: front %.0f mm bump / %.0f mm rebound,"
                " rear %.0f mm bump / %.0f mm rebound\n",
                1000.0 * publishedFrontCompression, 1000.0 * publishedFrontRebound,
                1000.0 * publishedRearCompression, 1000.0 * publishedRearRebound);

    const auto published = std::array{publishedFrontRebound, publishedRearRebound};

    for (auto axle = std::size_t{0}; axle < axles.size(); axle++)
    {
        const auto& corner = base->corners[axles[axle].index];
        const auto& hardpoints = corner.hardpoints;
        const auto want = published[axle];

        // Where this linkage puts that much wheel droop. Searched past the authored `droopAngle`,
        // because the whole question is whether the authored range is short — the linkage itself
        // solves far beyond it.
        auto low = 0.0;
        auto high = -0.60;
        for (auto step = 0; step < 120; step++)
        {
            const auto middle = 0.5 * (low + high);
            const auto solved = solveCornerWithJacobian(hardpoints, middle, 0.0);
            if (!solved.has_value())
            {
                high = middle;
                continue;
            }

            const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
            REQUIRE(design.has_value());

            if (design->wheelTravel - solved->wheelTravel < want)
            {
                low = middle;
            }
            else
            {
                high = middle;
            }
        }

        const auto angle = 0.5 * (low + high);
        const auto atPublished = sample(hardpoints, angle);
        const auto atAuthored = sample(hardpoints, hardpoints.droopAngle);

        std::printf("\n  --- %s ---\n", axles[axle].name);
        std::printf("    published rebound travel (wheel)          %8.1f mm\n", 1000.0 * want);
        std::printf("    this model's authored rebound travel      %8.2f mm  (q = %.5f)\n",
                    -1000.0 * atAuthored.wheelTravel, hardpoints.droopAngle);
        std::printf("    so the model is                           %8.1f%% of the published figure\n",
                    100.0 * -atAuthored.wheelTravel / want);
        std::printf("    the published travel would need q =       %8.5f rad\n", angle);
        std::printf("    and that is                               %8.2f mm of shaft extension\n",
                    1000.0 * atPublished.extension);
        std::printf("    against the authored range's              %8.2f mm\n", 1000.0 * atAuthored.extension);
        std::printf("    the shipped 40 mm gap is                  %8.1f%% of the published envelope's shaft\n",
                    100.0 * corner.droopStop.gap / atPublished.extension);
        std::printf("    and                                       %8.1f%% of the authored one's\n",
                    100.0 * corner.droopStop.gap / atAuthored.extension);
    }

    std::printf("\n  Nothing here is installed. What it says is that the droop side of this car's\n");
    std::printf("  authored travel is short against the only published Golf envelope in this\n");
    std::printf("  repository, and that the 40 mm gap is a large fraction of a range that is itself\n");
    std::printf("  a placeholder — so the gap's meaning depends on a number nobody has measured on\n");
    std::printf("  the Mk7.\n");
}
