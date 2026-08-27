// A probe, not a gate. Hidden behind a dotted tag, run by hand:
// `./EngineTests "[.spring-damper-kinematics]"`.
//
// Step 2 of the suspension audit. The model carries **one** elastic element per corner — the
// spring rides the damper axis, a coil-over by construction — so one motion ratio serves both the
// spring-rate conversion and the damper-rate conversion. A real Mk7 rear carries its spring and
// damper on different members, which can have different ratios (the 0.64-vs-0.78 question). This
// probe separates the two *as types*: `SpringKinematics` and `DamperKinematics` are distinct
// structs, the wheel-rate conversion accepts only the spring's, the shaft-speed conversion accepts
// only the damper's, and the cross-uses are `= delete`d — so using the damper ratio to convert a
// spring rate is a compile error here, not a latent bug.
//
// **No hardpoint is invented.** The spring element is taken from what is authored today — the
// damper's own axis — and the table says so in its provenance line. The probe *demonstrates* that
// a separately-seated spring needs no solver change: its element evaluation rotates an arbitrary
// lower-arm point about the linkage's own swing axis, independently of the solver, and is checked
// against the solver's damper channel to 1e-9 across the whole travel. Feed it a sourced spring
// seat and it reports that seat's ratio; today it is fed the damper pair and must agree.
//
// Deterministic: pure solve, fixed samples, no clock, no randomness.

#include <cmath>
#include <cstddef>
#include <cstdio>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::Corner;
using raceengine::cornerAbbreviation;
using raceengine::cornerCount;
using raceengine::CornerHardpoints;
using raceengine::golfGtiMk7;
using raceengine::solveCornerWithJacobian;
using raceengine::SuspensionKind;

namespace
{

// A chassis-to-wishbone element: an inboard point bolted to the body and an outboard point rigid
// on the lower wishbone. Both of this car's authored elements fit it — the rear damper
// (damperChassis to damperWishbone) and the front strut (strutTop to the lower ball joint).
struct LinkageElement
{
    glm::dvec3 inboard{0.0};
    glm::dvec3 outboardDesign{0.0};
};

// Where the element's geometry comes from. The point of carrying this beside the numbers: a ratio
// with no provenance is how 0.64 and 0.987 got compared as if they measured the same thing.
enum class Provenance
{
    CoaxialWithAuthoredDamper, // no separate spring data exists; the spring rides the damper axis
    AuthoredStrut,             // the front strut IS the damper and the coaxial spring is the real layout
    PlacedUnsourced            // authored to a stated rationale, not measured from the car
};

[[nodiscard]] const char* describe(const Provenance provenance)
{
    switch (provenance)
    {
    case Provenance::CoaxialWithAuthoredDamper: return "COAXIAL with the damper (no separate data exists)";
    case Provenance::AuthoredStrut: return "the authored strut axis (a real coil-over)";
    case Provenance::PlacedUnsourced: return "PLACED, unsourced (rationale in PublishedCarsImpl.cpp)";
    }
    return "?";
}

// The lower wishbone's swing axis with the solver's own sign rule (`swingOf`): oriented so that a
// positive angle raises the ball joint. Reimplemented here rather than exported, because the probe
// exists to measure the element kinematics *independently* of the solver and then agree with it.
[[nodiscard]] glm::dvec3 lowerSwingAxis(const CornerHardpoints& hardpoints)
{
    auto axis = glm::normalize(hardpoints.lower.rearPivot - hardpoints.lower.frontPivot);
    const auto toBallJoint = hardpoints.lower.ballJoint - hardpoints.lower.frontPivot;
    const auto radius = toBallJoint - glm::dot(toBallJoint, axis) * axis;

    if (glm::cross(axis, radius).y < 0.0)
    {
        axis = -axis;
    }

    return axis;
}

[[nodiscard]] double elementLength(const CornerHardpoints& hardpoints, const LinkageElement& element, const double angle)
{
    const auto axis = lowerSwingAxis(hardpoints);
    const auto outboard = hardpoints.lower.frontPivot +
                          glm::angleAxis(angle, axis) * (element.outboardDesign - hardpoints.lower.frontPivot);

    return glm::distance(element.inboard, outboard);
}

// The two kinematic records. Deliberately separate types with no common base and no conversion:
// the compiler is the guard the arithmetic cannot be.
struct SpringKinematics
{
    double length = 0.0;
    double travel = 0.0;     // design length minus length, positive in bump
    double perAngle = 0.0;   // dLength/dq
    double ratio = 0.0;      // dLength/dWheelTravel, signed as the solver signs its motion ratio
    Provenance provenance = Provenance::CoaxialWithAuthoredDamper;
};

struct DamperKinematics
{
    double length = 0.0;
    double travel = 0.0;
    double perAngle = 0.0;
    double ratio = 0.0;
    Provenance provenance = Provenance::PlacedUnsourced;
};

// A wheel rate comes from the SPRING's ratio and nothing else.
[[nodiscard]] double wheelRateFromSpringRate(const SpringKinematics& spring, const double springRate)
{
    return springRate * spring.ratio * spring.ratio;
}
double wheelRateFromSpringRate(const DamperKinematics&, double) = delete;

// A shaft speed comes from the DAMPER's ratio and nothing else. Per metre-per-second of wheel
// closing speed, the shaft closes |ratio| metres per second.
[[nodiscard]] double shaftSpeedPerWheelSpeed(const DamperKinematics& damper)
{
    return std::abs(damper.ratio);
}
double shaftSpeedPerWheelSpeed(const SpringKinematics&) = delete;

// The element each role is attached to TODAY. A sourced rear spring seat changes springElementOf
// and nothing else — which is the whole demonstration that the separation is cheap.
[[nodiscard]] LinkageElement probeDamperElementOf(const CornerHardpoints& hardpoints)
{
    if (hardpoints.kind == SuspensionKind::MacPhersonStrut)
    {
        return LinkageElement{.inboard = hardpoints.strutTop, .outboardDesign = hardpoints.lower.ballJoint};
    }

    return LinkageElement{.inboard = hardpoints.damperChassis, .outboardDesign = hardpoints.damperWishbone};
}

[[nodiscard]] LinkageElement probeSpringElementOf(const CornerHardpoints& hardpoints)
{
    // No car in this workspace states a separate spring seat, so the spring rides the damper axis.
    // This is the authored assumption, printed as such — not a measurement.
    return probeDamperElementOf(hardpoints);
}

template <typename Kinematics>
[[nodiscard]] Kinematics evaluate(const CornerHardpoints& hardpoints, const LinkageElement& element, const double angle,
                                  const double designLength, const double travelPerAngle, const Provenance provenance)
{
    constexpr auto step = 1e-6;
    const auto length = elementLength(hardpoints, element, angle);
    const auto perAngle =
        (elementLength(hardpoints, element, angle + step) - elementLength(hardpoints, element, angle - step)) /
        (2.0 * step);

    return Kinematics{.length = length,
                      .travel = designLength - length,
                      .perAngle = perAngle,
                      .ratio = perAngle / travelPerAngle,
                      .provenance = provenance};
}

} // namespace

TEST_CASE("spring and damper kinematics, reported separately", "[.spring-damper-kinematics]")
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& corner = setup->corners[index];
        const auto& hardpoints = corner.hardpoints;
        const auto strut = hardpoints.kind == SuspensionKind::MacPhersonStrut;

        const auto damperElement = probeDamperElementOf(hardpoints);
        const auto springElement = probeSpringElementOf(hardpoints);
        const auto damperProvenance = strut ? Provenance::AuthoredStrut : Provenance::PlacedUnsourced;

        const auto design = solveCornerWithJacobian(hardpoints, 0.0, 0.0);
        REQUIRE(design.has_value());
        const auto designSpringLength = elementLength(hardpoints, springElement, 0.0);
        const auto designDamperLength = elementLength(hardpoints, damperElement, 0.0);

        std::printf("\n==== %s (%s) ====\n", cornerAbbreviation(static_cast<Corner>(index)),
                    strut ? "MacPherson strut" : "double wishbone");
        std::printf("  spring geometry: %s\n", describe(Provenance::CoaxialWithAuthoredDamper));
        std::printf("  damper geometry: %s\n", describe(damperProvenance));
        std::printf("  %9s %8s | %9s %8s %8s | %9s %8s %8s | %9s %10s\n", "q [rad]", "trav mm", "spr mm", "sprTr mm",
                    "sprMR", "dmp mm", "dmpTr mm", "dmpMR", "whlRate", "shaft m/s");

        constexpr auto samples = std::size_t{11};
        for (auto sample = std::size_t{0}; sample < samples; sample++)
        {
            const auto through = static_cast<double>(sample) / static_cast<double>(samples - 1);
            const auto angle = hardpoints.droopAngle + through * (hardpoints.bumpAngle - hardpoints.droopAngle);

            const auto solved = solveCornerWithJacobian(hardpoints, angle, 0.0);
            REQUIRE(solved.has_value());

            const auto spring = evaluate<SpringKinematics>(hardpoints, springElement, angle, designSpringLength,
                                                           solved->travelPerAngle,
                                                           Provenance::CoaxialWithAuthoredDamper);
            const auto damper = evaluate<DamperKinematics>(hardpoints, damperElement, angle, designDamperLength,
                                                           solved->travelPerAngle, damperProvenance);

            // The probe's independent element evaluation must reproduce the production element
            // API: two implementations, no shared code path. (The solver's own damper channel is
            // gone since step 14, so the API is the production side of the comparison now.)
            const auto production = raceengine::solveElement(
                hardpoints, raceengine::damperElementOf(hardpoints), angle);
            const auto productionRatio = raceengine::solveDamperKinematics(
                hardpoints, raceengine::damperElementOf(hardpoints), angle);
            REQUIRE(productionRatio.has_value());
            REQUIRE(std::abs(damper.length - production.length) < 1e-9);
            REQUIRE(damper.ratio == Catch::Approx(productionRatio->motionRatio).margin(1e-6));

            // Coaxial by authoring, so the two ratios agree exactly — a statement about the data,
            // not about the car. A sourced spring seat is expected to break this on the rear.
            REQUIRE(spring.ratio == Catch::Approx(damper.ratio).margin(1e-12));

            std::printf("  %+9.5f %+8.2f | %9.2f %+8.2f %+8.4f | %9.2f %+8.2f %+8.4f | %9.1f %10.4f\n", angle,
                        solved->wheelTravel * 1000.0, spring.length * 1000.0, spring.travel * 1000.0, spring.ratio,
                        damper.length * 1000.0, damper.travel * 1000.0, damper.ratio,
                        wheelRateFromSpringRate(spring, corner.springRate), shaftSpeedPerWheelSpeed(damper));
        }

        // The round trip that pins the conversion convention: the shipped shaft rate came from
        // AC's at-the-wheel rate through the design ratio, so the spring ratio must reproduce it.
        const auto designSpring =
            evaluate<SpringKinematics>(hardpoints, springElement, 0.0, designSpringLength, design->travelPerAngle,
                                       Provenance::CoaxialWithAuthoredDamper);
        const auto statedWheelRate = index < 2 ? 35000.0 : 57000.0;
        REQUIRE(wheelRateFromSpringRate(designSpring, corner.springRate) ==
                Catch::Approx(statedWheelRate).epsilon(1e-6));

        std::printf("  design: wheel rate %.1f N/m from shaft %.1f N/m x sprMR^2 (AC states %.0f at the wheel)\n",
                    wheelRateFromSpringRate(designSpring, corner.springRate), corner.springRate, statedWheelRate);
    }
}
