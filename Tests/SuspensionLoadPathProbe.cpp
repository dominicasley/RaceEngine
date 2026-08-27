// What the linkage would carry, if the tyre's in-plane forces were allowed to reach the corner's
// degree of freedom.
//
// A probe, not a criterion: it prints numbers and asserts only the things that would be *wrong*
// rather than merely surprising. It exists because `docs/suspension-load-path-brief.md` section 9
// says the geometric path is only as good as the hardpoints, and this car's rear hardpoints are
// partly authored rather than sourced — so before any of stage 3's numbers are believed, the
// anti-dive and anti-squat percentages the geometry implies have to be printed and read against
// published ranges. A rear axle with 80% anti-squat because a damper was placed for convenience is
// the failure mode to expect.
//
// It also settles two of that brief's open questions with measurements rather than argument:
// whether the Jacobian should be taken at the tyre's free radius or its loaded one, and whether
// `patchPerAngle` is differenced at the tick's own rack position.
//
// Hidden behind a dot tag — `./EngineTests "[.load-path]"` — like every other probe here.

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::computeMassProperties;
using raceengine::computeRollCentre;
using raceengine::cornerCount;
using raceengine::golfGtiMk7;
using raceengine::MassComponent;
using raceengine::solveCorner;
using raceengine::solveCornerWithJacobian;
using raceengine::SuspensionState;

namespace
{

constexpr auto names = std::array<const char*, cornerCount>{"FL", "FR", "RL", "RR"};

// The wheel centre's own Jacobian, which the state does not carry. Differenced exactly as the
// solver differences its own, at the same step, so the two cannot disagree by choice of method.
// The probe needs it for one thing only: separating the free-radius patch Jacobian from the loaded
// one, which differ by the part of the motion that is the *tyre's* radius turning with camber.
[[nodiscard]] glm::dvec3 wheelCentrePerAngle(const raceengine::CornerHardpoints& hardpoints, const double angle,
                                             const double rackTravel)
{
    constexpr auto step = 1e-6;

    const auto behind = solveCorner(hardpoints, angle - step, rackTravel);
    const auto ahead = solveCorner(hardpoints, angle + step, rackTravel);
    REQUIRE(behind.has_value());
    REQUIRE(ahead.has_value());

    return (ahead->wheelCentre - behind->wheelCentre) / (2.0 * step);
}

[[nodiscard]] SuspensionState solved(const raceengine::CornerHardpoints& hardpoints, const double angle,
                                     const double rackTravel)
{
    auto state = solveCornerWithJacobian(hardpoints, angle, rackTravel);
    REQUIRE(state.has_value());

    computeRollCentre(hardpoints, state.value());

    return state.value();
}

} // namespace

TEST_CASE("load path: what the linkage carries, and what the hardpoints imply", "[.load-path]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto& setup = built.value();

    // --- the whole car, for the two lengths every percentage below is relative to ----------------
    //
    // Anti-dive and anti-squat are not properties of a linkage on its own: they are the linkage's
    // side-view ratio measured against the pitch couple the car makes, which needs a wheelbase and
    // a centre-of-gravity height. Both come from the built car rather than from a doc.
    auto components = setup.sprung;
    for (const auto& corner : setup.corners)
    {
        components.push_back(MassComponent{.mass = corner.unsprungMass, .centre = corner.hardpoints.wheelCentre});
    }

    const auto whole = computeMassProperties(components);
    REQUIRE(whole.has_value());

    const auto wheelbase = setup.corners[0].hardpoints.wheelCentre.z - setup.corners[2].hardpoints.wheelCentre.z;
    // The chassis frame puts the design contact plane at y = 0, so the centre of mass's height above
    // it is its y directly.
    const auto centreHeight = whole->centreOfMass.y;

    WARN("=== the car these percentages are relative to ===");
    WARN("mass " << whole->mass << " kg, centre of gravity " << centreHeight * 1000.0 << " mm up, wheelbase "
                 << wheelbase * 1000.0 << " mm");

    // The brake force split at a full pedal, which is what an anti-dive percentage is quoted at.
    // Every corner runs the same radius here, so torque shares are force shares.
    auto brakeTotal = 0.0;
    auto brakeFront = 0.0;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        brakeTotal += setup.corners[index].brakeTorque;
        brakeFront += index < 2 ? setup.corners[index].brakeTorque : 0.0;
    }
    const auto frontBrakeShare = brakeFront / brakeTotal;
    WARN("front brake share at a full pedal " << frontBrakeShare);

    // --- 1. the Jacobian at design --------------------------------------------------------------
    //
    // `travelPerAngle` is the wheel *centre*'s vertical rate and is what the corner's generalised
    // force runs on today. `patchPerAngle` is the whole velocity of the point the road pushes on.
    // The two y components differ, and that difference is camber gain turning the tyre's radius.
    WARN("=== 1. the patch Jacobian at design, chassis frame (metres per radian) ===");

    auto design = std::array<SuspensionState, cornerCount>{};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        design[index] = solved(setup.corners[index].hardpoints, 0.0, 0.0);
        const auto& state = design[index];

        WARN(names[index] << ": travelPerAngle " << state.travelPerAngle << ", patchPerAngle (" << state.patchPerAngle.x
                          << ", " << state.patchPerAngle.y << ", " << state.patchPerAngle.z << ")");
        WARN("    patch y against wheel-centre y: " << state.patchPerAngle.y << " vs " << state.travelPerAngle
                                                    << ", i.e. "
                                                    << (state.patchPerAngle.y / state.travelPerAngle - 1.0) * 100.0
                                                    << "% — camber gain turning the tyre radius");
    }

    // --- 2. the lateral ratio IS the roll centre -------------------------------------------------
    //
    // The cross-check that ties the new term to a diagnostic this project has printed since the
    // geometry audit. Both statements are about the direction the patch scrubs as the wheel rises:
    // the roll centre is where the patch's velocity normal crosses the centreline, so
    // `patchPerAngle.x / patchPerAngle.y` must equal `rollCentreHeight / contactPatch.x`. If it does
    // not, one of the two constructions is wrong and the force path must not be switched on.
    WARN("=== 2. lateral scrub ratio against the roll-centre construction ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];

        const auto scrubRatio = state.patchPerAngle.x / state.patchPerAngle.y;
        const auto fromRollCentre = state.rollCentreHeight / state.contactPatch.x;

        WARN(names[index] << ": scrub ratio " << scrubRatio << ", from the roll centre " << fromRollCentre
                          << " (roll centre " << state.rollCentreHeight * 1000.0 << " mm, half track "
                          << state.contactPatch.x * 1000.0 << " mm), disagreement "
                          << std::abs(scrubRatio - fromRollCentre));

        // Two independent constructions of the same geometry. A percent is generous — the roll
        // centre is built from straight lines through ball joints and the Jacobian is a central
        // difference of the whole solve — and anything larger means they are not the same claim.
        CHECK(std::abs(scrubRatio - fromRollCentre) < 0.01);
    }

    // --- 3. anti-dive, anti-lift and anti-squat ---------------------------------------------------
    //
    // The section 9 check, and the one that can stop this work: the rear damper is *placed*, so the
    // rear's side-view geometry is an authored choice that has never had to produce a number before.
    //
    // The definitions, derived from the same virtual work the force path uses rather than quoted.
    // An equivalent vertical force at the wheel is the one that makes the same generalised force:
    // `Feq = Fz · (patchPerAngle.z / patchPerAngle.y)`. Under braking the front gains `m·a·h/L` and
    // carries `phi_f·m·a` of longitudinal force, so
    //
    //     anti-dive     =  phi_f · tau_front · L / h        (front, resisting compression)
    //     anti-lift     = -phi_r · tau_rear  · L / h        (rear, resisting extension)
    //
    // with `tau = patchPerAngle.z / patchPerAngle.y`, and +z forward.
    //
    // **Under power the front line is the wheel centre's, not the patch's, and that is not a
    // refinement.** This car is front wheel drive with a chassis-mounted transaxle, so the drive
    // torque is applied to the hub *from the chassis* and does virtual work in the corner's own
    // coordinate — the upright turns about the wheel's spin axis as the arm swings. That term
    // converts the patch line into the wheel-centre line exactly. Braking is the other case: the
    // calipers are outboard, the brake couple is internal to the wheel assembly, and the patch line
    // is complete. So the two are printed separately and neither is the other's approximation.
    WARN("=== 3. anti-dive and anti-squat the hardpoints imply ===");

    const auto ratio = wheelbase / centreHeight;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];
        const auto patchTau = state.patchPerAngle.z / state.patchPerAngle.y;

        const auto centreRate = wheelCentrePerAngle(setup.corners[index].hardpoints, 0.0, 0.0);
        const auto centreTau = centreRate.z / centreRate.y;

        const auto share = index < 2 ? frontBrakeShare : 1.0 - frontBrakeShare;
        const auto braking = (index < 2 ? 1.0 : -1.0) * share * patchTau * ratio;

        WARN(names[index] << ": side-view ratio at the patch " << patchTau << ", at the wheel centre " << centreTau);
        WARN("    braking (outboard calipers, patch line): " << braking * 100.0
                                                             << "% anti-dive front / anti-lift rear");

        if (index < 2)
        {
            // Front wheel drive: the driven axle's traction runs the whole longitudinal force, and
            // the chassis-mounted differential puts the line at the wheel centre.
            WARN("    power (inboard differential, wheel-centre line): " << centreTau * ratio * 100.0 << "% anti-lift");
        }
    }

    // --- 4. open question 2: the free radius against the loaded one ------------------------------
    //
    // The brief's section 13 asks whether the Jacobian belongs at the geometric patch — the tyre's
    // *free* radius below the wheel centre, which is what `SuspensionState::contactPatch` is — or at
    // the loaded radius, where the road actually is. The force is transmitted through the tyre's own
    // compliance, so the honest point of application is the loaded one, and the two differ by the
    // part of the patch's motion that is the radius turning with camber.
    //
    // Exactly, and for free, because the patch is the wheel centre plus a radius along a unit
    // direction: scaling that radius interpolates the two Jacobians.
    WARN("=== 4. the Jacobian at the free radius against the loaded radius ===");

    for (const auto deflection : std::vector<double>{0.010, 0.015, 0.030})
    {
        WARN("--- tyre deflection " << deflection * 1000.0 << " mm ---");

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            const auto& state = design[index];
            const auto radius = setup.corners[index].hardpoints.wheelRadius;
            const auto centreRate = wheelCentrePerAngle(setup.corners[index].hardpoints, 0.0, 0.0);

            const auto loaded = centreRate + ((radius - deflection) / radius) * (state.patchPerAngle - centreRate);

            const auto freeScrub = state.patchPerAngle.x / state.patchPerAngle.y;
            const auto loadedScrub = loaded.x / loaded.y;
            const auto freeTau = state.patchPerAngle.z / state.patchPerAngle.y;
            const auto loadedTau = loaded.z / loaded.y;

            WARN(names[index] << ": scrub ratio " << freeScrub << " -> " << loadedScrub << " ("
                              << (loadedScrub / freeScrub - 1.0) * 100.0 << "%), side-view ratio " << freeTau << " -> "
                              << loadedTau << " (" << (loadedTau / freeTau - 1.0) * 100.0 << "%)");
        }
    }

    // --- 5. open question 3: does the rack position reach it? ------------------------------------
    //
    // `solveCornerWithJacobian` differences at whatever rack travel it is called with, and the force
    // pass calls it with the tick's own. Confirmed rather than assumed, because the front geometry's
    // scrub varies with steer and that is exactly where the term matters most.
    WARN("=== 5. the front Jacobian against rack travel ===");

    for (const auto rack : std::vector<double>{-0.05, -0.025, 0.0, 0.025, 0.05})
    {
        const auto steered = solved(setup.corners[0].hardpoints, 0.0, rack);

        WARN("rack " << rack * 1000.0 << " mm: patchPerAngle (" << steered.patchPerAngle.x << ", "
                     << steered.patchPerAngle.y << ", " << steered.patchPerAngle.z << "), scrub ratio "
                     << steered.patchPerAngle.x / steered.patchPerAngle.y << ", side-view ratio "
                     << steered.patchPerAngle.z / steered.patchPerAngle.y);
    }

    // The whole point of the question: if steering did not reach the Jacobian, these would be equal.
    const auto left = solved(setup.corners[0].hardpoints, 0.0, -0.05);
    const auto right = solved(setup.corners[0].hardpoints, 0.0, 0.05);
    CHECK(std::abs(left.patchPerAngle.z - right.patchPerAngle.z) > 1e-6);

    // --- 5b. where the front's side-view ratio comes from ----------------------------------------
    //
    // Section 3 reports a front axle that is **pro-dive**, which is outside the published band for a
    // road car and is exactly the class of finding section 9 of the brief says to stop and look at.
    // So: which authored number produces it. Two candidates, each restored on its own and neither
    // authored anywhere — this is a diagnostic, not a change.
    //
    // (a) The strut top, which step 17 of the geometry audit moved 35.3 mm rearward to make the
    //     published 7.5 deg of caster. A strut's upper point is the top of its kingpin, so it sets
    //     how the upright rotates in side view as the arm swings.
    // (b) The lower arm's pivot axis, whose rear bush is authored 10.6 mm **below** its front one.
    //     Rotating about an axis that slopes down toward the rear swings the wheel rearward as it
    //     rises, and a wheel that moves rearward in bump is a wheel with negative anti-dive.
    WARN("=== 5b. which authored number makes the front pro-dive ===");

    {
        const auto ratioOf = [&](const raceengine::CornerHardpoints& hardpoints)
        {
            const auto state = solved(hardpoints, 0.0, 0.0);
            return state.patchPerAngle.z / state.patchPerAngle.y;
        };

        const auto shipped = setup.corners[0].hardpoints;

        // (a) the strut top as the mod imported it, before the caster correction.
        auto preCaster = shipped;
        preCaster.strutTop.z = shipped.wheelCentre.z - 0.04296;

        // (b) the lower arm's rear pivot raised to its front pivot's height, so the axis is level in
        // side view and the arm swings in a plane square to the car.
        auto levelled = shipped;
        levelled.lower.rearPivot.y = shipped.lower.frontPivot.y;

        // (c) and the mirror of (b), the rear bush raised as far above the front one as it is
        // currently below it — the sign check, so that "lower the rear bush and dive gets worse" is
        // shown both ways rather than argued.
        auto raised = shipped;
        raised.lower.rearPivot.y =
            shipped.lower.frontPivot.y + (shipped.lower.frontPivot.y - shipped.lower.rearPivot.y);

        WARN("shipped                              : side-view ratio "
             << ratioOf(shipped) << ", anti-dive " << frontBrakeShare * ratioOf(shipped) * ratio * 100.0 << "%");
        WARN("strut top as imported (caster 4.59)  : side-view ratio "
             << ratioOf(preCaster) << ", anti-dive " << frontBrakeShare * ratioOf(preCaster) * ratio * 100.0 << "%");
        WARN("lower arm pivots levelled            : side-view ratio "
             << ratioOf(levelled) << ", anti-dive " << frontBrakeShare * ratioOf(levelled) * ratio * 100.0 << "%");
        WARN("lower arm rear pivot raised 10.6 mm  : side-view ratio "
             << ratioOf(raised) << ", anti-dive " << frontBrakeShare * ratioOf(raised) * ratio * 100.0 << "%");
    }

    // --- 6. across the travel: the migration the linear model cannot have ------------------------
    //
    // The brief predicts roll must become progressive, because the front roll centre falls 206 -> 37
    // mm across the travel. That prediction is a property of this table.
    WARN("=== 6. the scrub ratio across the stated travel ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = setup.corners[index].hardpoints;

        WARN("--- " << names[index] << " ---");
        for (const auto through : std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0})
        {
            const auto angle = hardpoints.droopAngle + through * (hardpoints.bumpAngle - hardpoints.droopAngle);
            const auto state = solved(hardpoints, angle, 0.0);

            WARN("  travel " << state.wheelTravel * 1000.0 << " mm: scrub ratio "
                             << state.patchPerAngle.x / state.patchPerAngle.y << ", side-view ratio "
                             << state.patchPerAngle.z / state.patchPerAngle.y << ", roll centre "
                             << state.rollCentreHeight * 1000.0 << " mm");
        }
    }
}
