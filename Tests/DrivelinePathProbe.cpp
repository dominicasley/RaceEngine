// What the upright does about the wheel's own spin axis as the arm swings, and what that is worth.
//
// `docs/suspension-fidelity-brief.md` item 3 stops and asks a question before letting anybody write
// a force term: it reports the side-view ratio as **equal at the patch and at the wheel centre to
// six digits on all four corners**, which reads as "the spin term is negligible on this car", and
// sets that against the geometry audit's step 16, which put the front upright's true instantaneous
// axis at (−0.20, −0.35, −0.92) — a fifth of it along the lateral axis, which is not negligible at
// all. One of the two, it says, is wrong, or they are not the same quantity. Measure first.
//
// **They are not the same quantity, and this probe is the demonstration.**
// `SuspensionState::contactPatch` is built as the wheel centre plus the tyre's radius along the
// *world's* down direction projected into the wheel plane — so it depends on the spin axis and on
// nothing else about how the upright is standing. Rotate the upright about that spin axis and the
// constructed patch does not move at all, while the real contact patch rolls forward or back by the
// radius times the angle. The equality the brief found is therefore a property of the construction
// and carries no information about `dTheta_spin/dq` whatsoever.
//
// What follows from that is bigger than the question. The road force's point of application *is* a
// material point of the wheel — the same code already treats it as one in the other coordinate, where
// `roadTorque = −Fx · r` is exactly the moment of that force about the wheel centre — so the corner's
// Jacobian has to be the material one too, and the constructed patch's is short by the spin term
// alone. The consequences separate cleanly, which is why they are printed separately below:
//
//   * the **lateral** channel is untouched, because a rotation about the spin axis moves the patch
//     fore and aft and never sideways. Roll centre, jacking and scrub are exactly what they were.
//   * the **longitudinal** channel is the whole of the difference, and it is anti-dive, anti-lift
//     and anti-squat.
//
// Hidden behind a dot tag — `./EngineTests "[.driveline-path]"` — like every other probe here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::computeMassProperties;
using raceengine::cornerCount;
using raceengine::DrivelineState;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::MassComponent;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::solveCornerWithJacobian;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::SuspensionState;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelInertias;

namespace
{

constexpr auto names = std::array<const char*, cornerCount>{"FL", "FR", "RL", "RR"};

[[nodiscard]] SuspensionState solved(const raceengine::CornerHardpoints& hardpoints, const double angle,
                                     const double rackTravel)
{
    auto state = solveCornerWithJacobian(hardpoints, angle, rackTravel);
    REQUIRE(state.has_value());

    return state.value();
}

// The axis a positive wheel speed turns about, in the chassis frame.
//
// **Not `outboardSign`'s axis**, and the difference is the whole of why this is written down. The
// pose handed to the contact patch uses `upright · (outboard, 0, 0)`, which points *away from the
// car* and therefore the opposite way on the two sides. Both wheels of a car going forward turn the
// same way, so the rotation axis cannot be that one: a wheel rolling forward has its angular
// velocity along the car's **left**, which is `upright · (+1, 0, 0)` on every corner. Reading the
// outboard axis here would invert the drive term on the right-hand corners and leave the left ones
// correct, which is precisely the class of sign error a symmetric fixture cannot see.
[[nodiscard]] glm::dvec3 spinAxisOf(const SuspensionState& state)
{
    return state.uprightOrientation * glm::dvec3(1.0, 0.0, 0.0);
}

// The point the road pushes on, carried by the upright as the material point it is: the wheel
// centre's own velocity plus the rotation of the radius beneath it.
//
// `radius` is which radius to carry — the tyre's free one reproduces the geometric patch, a smaller
// one is the loaded patch the road is actually at, and the vector below the centre is the direction
// either way. Written as a rescale of the state's own offset rather than as a fresh direction so the
// two cannot disagree about which way is down in the wheel's plane.
[[nodiscard]] glm::dvec3 materialPatchPerAngle(const SuspensionState& state, const double radius)
{
    const auto belowCentre = state.contactPatch - state.wheelCentre;
    const auto scaled = belowCentre * (radius / std::max(glm::length(belowCentre), 1e-12));

    return state.wheelCentrePerAngle + glm::cross(state.uprightRatePerAngle, scaled);
}

} // namespace

TEST_CASE("driveline path: what the upright does about the spin axis, and what it is worth", "[.driveline-path]")
{
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto& setup = built.value();

    auto components = setup.sprung;
    for (const auto& corner : setup.corners)
    {
        components.push_back(MassComponent{.mass = corner.unsprungMass, .centre = corner.hardpoints.wheelCentre});
    }

    const auto whole = computeMassProperties(components);
    REQUIRE(whole.has_value());

    const auto wheelbase = setup.corners[0].hardpoints.wheelCentre.z - setup.corners[2].hardpoints.wheelCentre.z;
    const auto centreHeight = whole->centreOfMass.y;
    const auto ratio = wheelbase / centreHeight;

    auto brakeTotal = 0.0;
    auto brakeFront = 0.0;
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        brakeTotal += setup.corners[index].brakeTorque;
        brakeFront += index < 2 ? setup.corners[index].brakeTorque : 0.0;
    }
    const auto frontBrakeShare = brakeFront / brakeTotal;

    WARN("=== the car these percentages are relative to ===");
    WARN("mass " << whole->mass << " kg, centre of gravity " << centreHeight * 1000.0 << " mm up, wheelbase "
                 << wheelbase * 1000.0 << " mm, front brake share " << frontBrakeShare);

    auto design = std::array<SuspensionState, cornerCount>{};
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        design[index] = solved(setup.corners[index].hardpoints, 0.0, 0.0);
    }

    // --- 1. the upright's whole rotation, and the audit's own axis ---------------------------------
    //
    // Step 16 of the geometry audit derived this same vector by finite-differencing the solved pose
    // in a probe of its own and reported its direction at the front's design row as
    // (−0.20, −0.35, −0.92). If the state's new field is that vector, its normalised direction has to
    // land on that triple — which is the cross-check that says the two instruments agree, and the one
    // the brief asked for.
    WARN("=== 1. dTheta_upright/dWishboneAngle, chassis frame (radians per radian) ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];
        const auto rate = state.uprightRatePerAngle;
        const auto magnitude = glm::length(rate);
        const auto direction = magnitude > 1e-12 ? rate / magnitude : glm::dvec3(0.0);

        WARN(names[index] << ": (" << rate.x << ", " << rate.y << ", " << rate.z << "), magnitude " << magnitude
                          << ", direction (" << direction.x << ", " << direction.y << ", " << direction.z << ")");
    }

    // The audit's step-16 direction, and **it is quoted against geometry this car no longer has.**
    // Two things have to be got right before the two instruments can be compared at all, and both of
    // them are why the naive comparison reads as a 30-degree disagreement:
    //
    //   * step 16's probe builds `golfMk7FrontCorner(CornerSide::Right)`, so its axis is the **right**
    //     corner's. Comparing it against corner 0 mirrors the lateral component and nothing else,
    //     which looks exactly like a real disagreement in x alone.
    //   * step 16 is dated 2026-08-26 and **step 17 moved the strut top 35.3 mm rearward that same
    //     day** to make the published 7.5 degrees of caster. A strut's upper point is the top of its
    //     kingpin, so it is precisely what sets how the upright turns in side view — which is the
    //     lateral component of this axis. The quoted triple is the pre-correction geometry.
    //
    // So the check is run twice: against the shipped corner, which must *not* match, and against the
    // geometry step 16 was measured on, which must.
    {
        const auto audit = glm::normalize(glm::dvec3(-0.20, -0.35, -0.92));

        const auto alignmentOf = [&audit](const SuspensionState& state)
        {
            const auto rate = state.uprightRatePerAngle;

            // Sign-insensitive: an instantaneous axis and its negative are one rotation, and the two
            // probes pick their sign by different constructions.
            return std::abs(glm::dot(rate / glm::length(rate), audit));
        };

        auto preCaster = setup.corners[1].hardpoints;
        preCaster.strutTop.z = preCaster.wheelCentre.z - 0.04296;

        const auto shipped = alignmentOf(design[1]);
        const auto imported = alignmentOf(solved(preCaster, 0.0, 0.0));

        WARN("    FR shipped against the audit's step 16 axis (-0.20, -0.35, -0.92): alignment " << shipped);
        WARN("    FR as imported (the geometry step 16 was measured on):             alignment " << imported);

        // The audit quoted two decimals, so this cannot be tighter than the quotation.
        CHECK(imported > 0.999);

        // And the shipped corner is genuinely somewhere else, which is the caster correction showing
        // up in the one channel it was always going to show up in. Asserted so that a future session
        // cannot read the mismatch above as an instrument fault.
        CHECK(shipped < 0.995);
    }

    // --- 2. dTheta_spin/dq, which is the term the brief could not decide about --------------------
    WARN("=== 2. the component along the wheel's own spin axis ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];
        const auto spin = glm::dot(state.uprightRatePerAngle, spinAxisOf(state));
        const auto radius = setup.corners[index].hardpoints.wheelRadius;

        WARN(names[index] << ": dTheta_spin/dq " << spin << " rad/rad, so the patch rolls " << spin * radius * 1000.0
                          << " mm per radian of arm, against a wheel-centre rate of "
                          << state.wheelCentrePerAngle.z * 1000.0 << " mm/rad fore-and-aft");
    }

    // --- 3. why the brief's two readings are not the same quantity ---------------------------------
    //
    // The constructed patch minus the wheel centre, differenced, has **no** longitudinal or vertical
    // content at all: it is a pure lateral term, the tyre's radius turning with camber. That is the
    // whole reason the two side-view ratios matched to every digit printed, and it is not a small
    // number hiding — it is exactly zero by construction.
    WARN("=== 3. the constructed patch against the wheel centre, componentwise ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];
        const auto difference = state.patchPerAngle - state.wheelCentrePerAngle;

        WARN(names[index] << ": patchPerAngle - wheelCentrePerAngle = (" << difference.x << ", " << difference.y << ", "
                          << difference.z << ")");

        // The claim, asserted rather than admired: the construction can only ever move the patch
        // sideways. If this ever fails, the patch has become a material point and section 4 below is
        // no longer a correction but a duplication.
        CHECK(std::abs(difference.y) < 1e-9);
        CHECK(std::abs(difference.z) < 1e-9);
    }

    // --- 4. the material patch, and what it does to the side-view ratio ----------------------------
    //
    // The road force acts on the tyre, the tyre is attached to the rim, and the rim is carried by the
    // upright — so the point of application moves with the upright. This model already says so in the
    // spin coordinate, where the same force's moment about the wheel centre is `−Fx · r` and is
    // applied to the wheel every tick. Saying it in the corner's coordinate as well is not a new
    // model; it is the same one written down twice consistently.
    WARN("=== 4. the material patch Jacobian against the constructed one ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];
        const auto radius = setup.corners[index].hardpoints.wheelRadius;
        const auto material = materialPatchPerAngle(state, radius);

        const auto constructedTau = state.patchPerAngle.z / state.patchPerAngle.y;
        const auto materialTau = material.z / material.y;

        const auto share = index < 2 ? frontBrakeShare : 1.0 - frontBrakeShare;
        const auto sign = index < 2 ? 1.0 : -1.0;

        WARN(names[index] << ": side-view ratio " << constructedTau << " -> " << materialTau);
        WARN("    braking, outboard calipers: " << sign * share * constructedTau * ratio * 100.0 << "% -> "
                                                << sign * share * materialTau * ratio * 100.0
                                                << "% anti-dive front / anti-lift rear");
        WARN("    scrub ratio " << state.patchPerAngle.x / state.patchPerAngle.y << " -> " << material.x / material.y
                                << " (the lateral channel, which must not move)");
    }

    // --- 5. the identity that says the drive term is right -----------------------------------------
    //
    // With the shaft torque coming from the chassis, the corner's generalised force from a steady
    // tractive force `Fx` is
    //
    //     Q = Fx · (material patch Jacobian).z  +  T · (dTheta_spin/dq),   T = Fx · r
    //
    // and the textbook answer for an inboard differential is that the whole thing collapses onto the
    // **wheel centre** line. It does, exactly, and that is the check: the two added terms are the same
    // number with opposite signs plus the wheel centre's own rate. A sign error anywhere in the chain
    // — the spin axis, the cross product, the torque convention — breaks this identity, and nothing
    // else in this file would notice.
    WARN("=== 5. material patch + drive torque == the wheel-centre line ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& state = design[index];
        const auto radius = setup.corners[index].hardpoints.wheelRadius;
        const auto material = materialPatchPerAngle(state, radius);
        const auto spin = glm::dot(state.uprightRatePerAngle, spinAxisOf(state));

        // One newton of forward force at the patch, and the shaft torque that sustains it.
        const auto throughPatch = material.z;
        const auto throughShaft = radius * spin;
        const auto together = throughPatch + throughShaft;

        WARN(names[index] << ": patch " << throughPatch << " + shaft " << throughShaft << " = " << together
                          << ", wheel centre " << state.wheelCentrePerAngle.z << ", disagreement "
                          << std::abs(together - state.wheelCentrePerAngle.z));

        CHECK(std::abs(together - state.wheelCentrePerAngle.z) < 1e-9);
    }

    // --- 6. the front axle's anti-lift under power, which is what the driven axle gets -------------
    WARN("=== 6. the driven axle under power ===");

    for (auto index = std::size_t{0}; index < 2; index++)
    {
        const auto& state = design[index];
        const auto centreTau = state.wheelCentrePerAngle.z / state.wheelCentrePerAngle.y;

        WARN(names[index] << ": inboard differential, wheel-centre line: " << centreTau * ratio * 100.0
                          << "% anti-lift under power (the whole tractive force is on this axle)");
    }

    // --- 7. across the travel, because none of this is a constant ---------------------------------
    WARN("=== 7. the spin rate across the stated travel ===");

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = setup.corners[index].hardpoints;

        WARN("--- " << names[index] << " ---");
        for (const auto through : std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0})
        {
            const auto angle = hardpoints.droopAngle + through * (hardpoints.bumpAngle - hardpoints.droopAngle);
            const auto state = solved(hardpoints, angle, 0.0);
            const auto material = materialPatchPerAngle(state, hardpoints.wheelRadius);

            WARN("  travel " << state.wheelTravel * 1000.0 << " mm: dTheta_spin/dq "
                             << glm::dot(state.uprightRatePerAngle, spinAxisOf(state)) << ", side-view ratio "
                             << state.patchPerAngle.z / state.patchPerAngle.y << " -> " << material.z / material.y);
        }
    }

    // --- 8. and the rack, because the front's is steered ------------------------------------------
    WARN("=== 8. the front spin rate against rack travel ===");

    for (const auto rack : std::vector<double>{-0.05, -0.025, 0.0, 0.025, 0.05})
    {
        const auto state = solved(setup.corners[0].hardpoints, 0.0, rack);
        const auto material = materialPatchPerAngle(state, setup.corners[0].hardpoints.wheelRadius);

        WARN("rack " << rack * 1000.0 << " mm: dTheta_spin/dq "
                     << glm::dot(state.uprightRatePerAngle, spinAxisOf(state)) << ", side-view ratio "
                     << state.patchPerAngle.z / state.patchPerAngle.y << " -> " << material.z / material.y);
    }
}

// --- and the whole car, because a kinematic identity is not a lap ---------------------------------
//
// `VehicleSetup::drivelineReaction` in its two positions, on the two manoeuvres that can see it: a
// launch, where the wheels are spinning up hardest and the corner's shaft term is live on the driven
// axle, and a full-pedal stop, where the wheels are being slowed hardest and only the chassis half
// applies. Everything else about the two cars is identical.
//
// `./EngineTests "[.driveline-reaction]"`.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto degrees = 180.0 / 3.14159265358979323846;
constexpr auto hundred = 100.0 / 3.6;
constexpr auto plateLength = 900.0;

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

[[nodiscard]] ProvingGroundDescriptor plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = 200.0;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<raceengine::Feature>{};

    return descriptor;
}

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / setup.corners.front().hardpoints.wheelRadius;
    }
}

struct Run
{
    double toHundred = -1.0;
    double peakPitch = 0.0;
    double settledPitch = 0.0;
    double stoppingDistance = -1.0;
};

// Full throttle from an idle in gear on the brakes, shifted on road speed through the gear — the
// same launch `TractionControlTests` measures, reduced to what this question needs.
[[nodiscard]] Run launch(const VehicleSetup& setup, const PhysicsWorld& world)
{
    const auto driveline = golfGtiMk7Driveline();
    const auto inertias = wheelInertias(setup);

    auto state = VehicleState{};
    settle(setup, state, world, 0.0, 20.0);

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    auto road = std::array<double, cornerCount>{};
    auto result = Run{};

    const auto speeds = [&state]
    {
        return std::array<double, cornerCount>{state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                               state.corners[2].wheelSpeed, state.corners[3].wheelSpeed};
    };

    {
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto step = 0; step < 360; step++)
        {
            const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, idling, torques->wheel, world, tick);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());
        }
    }

    auto gear = 1;
    const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

    for (auto step = 1; step <= 360 * 12; step++)
    {
        const auto roadSideSpeed = std::abs(state.chassis.linearVelocity.z) /
                                   setup.corners.front().hardpoints.wheelRadius * driveline.gearbox.reduction(gear);
        if (roadSideSpeed > upshiftSpeed && gear < driveline.gearbox.topGear())
        {
            gear++;
        }

        auto input = VehicleInput{};
        input.throttle = 1.0;
        input.gear = gear;

        const auto torques = stepDriveline(driveline, drivelineState, speeds(), inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick);
        REQUIRE(stepped.has_value());
        road = roadTorques(stepped.value());

        result.peakPitch = std::max(result.peakPitch, std::abs(stepped->telemetry.pitch * degrees));
        result.settledPitch = stepped->telemetry.pitch * degrees;

        if (state.chassis.linearVelocity.z >= hundred)
        {
            result.toHundred = static_cast<double>(step) * tick;
            break;
        }
    }

    return result;
}

// Full pedal from 30 m/s with no electronics, which is the fixture every braking figure here is
// taken on. The wheels are being slowed hard, so the chassis half of the reaction is live and the
// corner half is not — nothing is driving.
[[nodiscard]] Run stop(const VehicleSetup& setup, const PhysicsWorld& world)
{
    auto state = VehicleState{};
    settle(setup, state, world, 30.0, 200.0);

    auto input = VehicleInput{};
    input.brake = 1.0;

    const auto from = state.chassis.position.z;
    auto result = Run{};

    for (auto step = 1; step <= 360 * 12; step++)
    {
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick);
        REQUIRE(stepped.has_value());

        result.peakPitch = std::max(result.peakPitch, std::abs(stepped->telemetry.pitch * degrees));
        result.settledPitch = stepped->telemetry.pitch * degrees;

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            result.stoppingDistance = state.chassis.position.z - from;
            break;
        }
    }

    return result;
}

} // namespace

TEST_CASE("driveline reaction: what the wheels' own spin takes out of the body", "[.driveline-reaction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    auto without = built.value();
    without.drivelineReaction = false;

    auto with = built.value();
    with.drivelineReaction = true;

    const auto launchOff = launch(without, world.value());
    const auto launchOn = launch(with, world.value());

    WARN("=== launch, full throttle from an idle in gear ===");
    WARN("0-100:      " << launchOff.toHundred << " s -> " << launchOn.toHundred << " s ("
                        << (launchOn.toHundred / launchOff.toHundred - 1.0) * 100.0 << "%)");
    WARN("peak pitch: " << launchOff.peakPitch << " deg -> " << launchOn.peakPitch << " deg");
    WARN("at 100:     " << launchOff.settledPitch << " deg -> " << launchOn.settledPitch << " deg");

    const auto stopOff = stop(without, world.value());
    const auto stopOn = stop(with, world.value());

    WARN("=== full-pedal stop from 30 m/s, no electronics ===");
    WARN("distance:   " << stopOff.stoppingDistance << " m -> " << stopOn.stoppingDistance << " m ("
                        << (stopOn.stoppingDistance / stopOff.stoppingDistance - 1.0) * 100.0 << "%)");
    WARN("peak pitch: " << stopOff.peakPitch << " deg -> " << stopOn.peakPitch << " deg");

    // Both runs have to have finished, or the numbers above are comparing a launch against a
    // timeout.
    REQUIRE(launchOff.toHundred > 0.0);
    REQUIRE(launchOn.toHundred > 0.0);
    REQUIRE(stopOff.stoppingDistance > 0.0);
    REQUIRE(stopOn.stoppingDistance > 0.0);
}
