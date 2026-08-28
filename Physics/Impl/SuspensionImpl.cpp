// Suspension bodies. Declarations are in Api/Suspension.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <cmath>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

module raceengine.physics;

namespace raceengine
{

[[nodiscard]] SphereCircleIntersection
intersectSphereWithCircle(const glm::dvec3& sphereCentre, const double sphereRadius, const glm::dvec3& circleCentre,
                          const double circleRadius, const glm::dvec3& circleNormal)
{
    auto result = SphereCircleIntersection{};

    const auto normal = glm::normalize(circleNormal);

    // The sphere cuts the circle's plane in a second circle; the answer is where those two meet.
    const auto toPlane = glm::dot(sphereCentre - circleCentre, normal);
    const auto cutRadiusSquared = sphereRadius * sphereRadius - toPlane * toPlane;
    if (cutRadiusSquared < 0.0)
    {
        return result;
    }

    const auto cutRadius = std::sqrt(cutRadiusSquared);
    const auto cutCentre = sphereCentre - toPlane * normal;

    const auto between = cutCentre - circleCentre;
    const auto distance = glm::length(between);

    // Concentric circles either coincide — infinitely many answers, which is a degenerate linkage
    // rather than a solution — or miss entirely.
    if (distance < 1e-12)
    {
        return result;
    }

    if (distance > circleRadius + cutRadius || distance < std::abs(circleRadius - cutRadius))
    {
        return result;
    }

    const auto along = (circleRadius * circleRadius - cutRadius * cutRadius + distance * distance) / (2.0 * distance);
    const auto acrossSquared = circleRadius * circleRadius - along * along;
    const auto across = acrossSquared > 0.0 ? std::sqrt(acrossSquared) : 0.0;

    const auto direction = between / distance;
    const auto foot = circleCentre + along * direction;
    const auto sideways = glm::cross(normal, direction);

    result.first = foot + across * sideways;
    result.second = foot - across * sideways;
    result.count = across > 1e-12 ? 2u : 1u;

    return result;
}

namespace
{

// Rotation of a point about a line, which is what a wishbone does to its ball joint.
[[nodiscard]] glm::dvec3 rotateAboutLine(const glm::dvec3& point, const glm::dvec3& linePoint,
                                         const glm::dvec3& lineDirection, const double angle)
{
    return linePoint + glm::angleAxis(angle, lineDirection) * (point - linePoint);
}

// The minimal rotation carrying one unit vector onto another. Written out rather than taken from
// glm's gtx so the antiparallel case is visibly handled: there the shortest arc is not unique, and
// a formula that divides by (1 + dot) produces a quiet NaN instead of saying so.
//
// There is deliberately **no near-parallel shortcut**, and that is not an oversight. The obvious
// one — return identity when the alignment is within an epsilon of one — is numerically pointless,
// because w = 1 + alignment is near two there and nothing is ill-conditioned. It is also actively
// harmful: it puts a flat step in the middle of a smooth function, and the motion ratio is
// differenced over an interval small enough to fall entirely inside it. That showed up as the
// motion ratio being wrong by 9% at exactly one point in the travel — the one where the kingpin
// happens to sit at its design angle — and nowhere else. A discontinuity an epsilon wide is still
// a discontinuity to a derivative.
[[nodiscard]] glm::dquat shortestArc(const glm::dvec3& from, const glm::dvec3& to)
{
    const auto alignment = glm::dot(from, to);

    if (alignment < -1.0 + 1e-12)
    {
        auto perpendicular = glm::cross(from, glm::dvec3(1.0, 0.0, 0.0));
        if (glm::length(perpendicular) < 1e-9)
        {
            perpendicular = glm::cross(from, glm::dvec3(0.0, 1.0, 0.0));
        }

        return glm::angleAxis(3.14159265358979323846, glm::normalize(perpendicular));
    }

    return glm::normalize(glm::dquat(1.0 + alignment, glm::cross(from, to)));
}

struct Circle
{
    glm::dvec3 centre{0.0};
    glm::dvec3 normal{0.0};
    double radius = 0.0;
};

// The circle a wishbone's ball joint travels on: centre is its projection onto the pivot axis.
//
// The axis is oriented so that a **positive angle always compresses the corner**, whichever way
// round the pivots were authored and whichever side of the car this is. That is not a convenience:
// taken as written, front-to-rear, the sign of the angle depends on the naming of two hardpoints
// *and* flips between the left and right corners, because mirrored geometry turns the opposite way
// about a shared axis. Every caller would then need to know which corner it was holding to know
// which way was up, and the one that forgot would author a droop stop that bumps.
//
// The test is the vertical response of the ball joint to rotation, (axis x radius)·up, which is the
// derivative of its height at zero angle.
[[nodiscard]] Circle swingOf(const Wishbone& wishbone)
{
    auto axis = glm::normalize(wishbone.rearPivot - wishbone.frontPivot);
    const auto toBallJoint = wishbone.ballJoint - wishbone.frontPivot;
    const auto centre = wishbone.frontPivot + glm::dot(toBallJoint, axis) * axis;
    const auto radius = wishbone.ballJoint - centre;

    if (glm::cross(axis, radius).y < 0.0)
    {
        axis = -axis;
    }

    return Circle{.centre = centre, .normal = axis, .radius = glm::length(radius)};
}

// Everything the upright's orientation decides, read off it rather than tracked beside it. One
// statement of it, because two would be two places for camber and the patch to disagree — and since
// compliance steer re-runs exactly this after twisting the upright, the two paths cannot drift.
//
// The arithmetic is untouched from where it was inline in `solveCorner`: same operations in the same
// order, so the solve is byte-identical across the extraction.
void readOffWheel(const CornerHardpoints& hardpoints, SuspensionState& state)
{
    // The wheel's spin axis, pointing away from the car. Everything about how the wheel is standing
    // is read from this rather than tracked beside it.
    const auto outboard = outboardSign(hardpoints.side);
    const auto spinAxis = state.uprightOrientation * glm::dvec3(outboard, 0.0, 0.0);

    state.camber = -std::asin(glm::clamp(spinAxis.y, -1.0, 1.0));
    state.toe = std::asin(glm::clamp(spinAxis.z, -1.0, 1.0));

    // The contact patch is directly below the wheel centre *in the wheel's own plane*, which for a
    // cambered wheel is not directly below it in the world.
    const auto down = glm::dvec3(0.0, -1.0, 0.0);
    const auto inPlaneDown = down - glm::dot(down, spinAxis) * spinAxis;
    state.contactPatch = state.wheelCentre + hardpoints.wheelRadius * glm::normalize(inPlaneDown);
    state.halfTrack = std::abs(state.contactPatch.x);
}

[[nodiscard]] glm::dvec3 nearestTo(const SphereCircleIntersection& candidates, const glm::dvec3& reference)
{
    if (candidates.count < 2)
    {
        return candidates.first;
    }

    return glm::distance(candidates.first, reference) <= glm::distance(candidates.second, reference)
               ? candidates.first
               : candidates.second;
}

} // namespace

[[nodiscard]] std::expected<SuspensionState, std::string> solveCorner(const CornerHardpoints& hardpoints,
                                                                      const double wishboneAngle,
                                                                      const double rackTravel,
                                                                      const SuspensionState* previous)
{
    const auto strut = hardpoints.kind == SuspensionKind::MacPhersonStrut;

    const auto lowerSwing = swingOf(hardpoints.lower);
    if (lowerSwing.radius < 1e-9)
    {
        return std::unexpected("the lower wishbone's ball joint lies on its own pivot axis, so it has no swing");
    }

    if (!strut && swingOf(hardpoints.upper).radius < 1e-9)
    {
        return std::unexpected("the upper wishbone's ball joint lies on its own pivot axis, so it has no swing");
    }

    // Step 1: the lower ball joint follows its circle directly. This is the degree of freedom.
    //
    // Rotating about the swing's *normalised* axis rather than the raw front-to-rear direction is
    // what makes a positive angle mean bump — and everything the wishbone carries has to turn about
    // the same one, or the damper and the drop link would travel opposite to the wheel.
    const auto lowerAxis = lowerSwing.normal;
    const auto lower =
        rotateAboutLine(hardpoints.lower.ballJoint, hardpoints.lower.frontPivot, lowerAxis, wishboneAngle);

    // Step 2: fix the upper end of the kingpin axis. This is the *only* step that differs between
    // the two linkages, and the strut's version is the simpler of the two: its upper point does not
    // move at all, because the strut's top bearing is bolted to the body. A double wishbone has to
    // find its upper ball joint, which is a fixed distance from the lower one — the upright is
    // rigid — and lies on the upper wishbone's own circle.
    auto upper = hardpoints.strutTop;

    if (!strut)
    {
        const auto upperSwing = swingOf(hardpoints.upper);
        const auto uprightLength = glm::distance(hardpoints.upper.ballJoint, hardpoints.lower.ballJoint);
        const auto upperCandidates =
            intersectSphereWithCircle(lower, uprightLength, upperSwing.centre, upperSwing.radius, upperSwing.normal);

        if (upperCandidates.count == 0)
        {
            return std::unexpected("the upper wishbone cannot reach the upright at " + std::to_string(wishboneAngle) +
                                   " rad; the linkage locks before here");
        }

        upper = nearestTo(upperCandidates, previous != nullptr ? previous->upperBallJoint : hardpoints.upper.ballJoint);
    }

    if (glm::distance(upper, lower) < 1e-9)
    {
        return std::unexpected("the kingpin axis has no length at " + std::to_string(wishboneAngle) + " rad");
    }

    // The kingpin axis is now fixed, and with it everything about the upright except how far it has
    // turned about that axis.
    const auto designUpper = strut ? hardpoints.strutTop : hardpoints.upper.ballJoint;
    const auto designKingpin = glm::normalize(designUpper - hardpoints.lower.ballJoint);
    const auto kingpin = glm::normalize(upper - lower);
    const auto carried = shortestArc(designKingpin, kingpin);

    // Step 3: the tie rod sets that last rotation. The steering arm swings on a circle about the
    // kingpin axis, and it must end up one tie rod from the rack's outer joint.
    const auto armFromLower = carried * (hardpoints.steeringArm - hardpoints.lower.ballJoint);
    const auto alongKingpin = glm::dot(armFromLower, kingpin) * kingpin;
    const auto armCircle =
        Circle{.centre = lower + alongKingpin, .normal = kingpin, .radius = glm::length(armFromLower - alongKingpin)};

    const auto rack = hardpoints.steeringRackOuter + glm::dvec3(rackTravel, 0.0, 0.0);
    const auto tieRodLength = glm::distance(hardpoints.steeringArm, hardpoints.steeringRackOuter);

    auto state = SuspensionState{};
    auto steeringRotation = glm::dquat(1.0, 0.0, 0.0, 0.0);

    if (armCircle.radius < 1e-9)
    {
        // The steering arm sits on the kingpin axis, so steering cannot turn the upright at all.
        // Legitimate for a corner with no steering authored; not something to divide by.
        state.steeringArm = lower + alongKingpin;
    }
    else
    {
        const auto armCandidates =
            intersectSphereWithCircle(rack, tieRodLength, armCircle.centre, armCircle.radius, armCircle.normal);

        if (armCandidates.count == 0)
        {
            return std::unexpected("the tie rod cannot reach the steering arm at " + std::to_string(wishboneAngle) +
                                   " rad and " + std::to_string(rackTravel) + " m of rack travel");
        }

        state.steeringArm =
            nearestTo(armCandidates, previous != nullptr ? previous->steeringArm : hardpoints.steeringArm);

        // Recover the angle turned about the kingpin, so the whole upright can be carried with it.
        const auto reference = glm::normalize(armFromLower - alongKingpin);
        const auto sideways = glm::cross(kingpin, reference);
        const auto offset = state.steeringArm - armCircle.centre;

        steeringRotation = glm::angleAxis(std::atan2(glm::dot(offset, sideways), glm::dot(offset, reference)), kingpin);
    }

    const auto upright = steeringRotation * carried;
    const auto place = [&](const glm::dvec3& designPoint)
    {
        return lower + upright * (designPoint - hardpoints.lower.ballJoint);
    };

    state.wishboneAngle = wishboneAngle;
    state.lowerBallJoint = lower;
    state.upperBallJoint = upper;
    state.uprightOrientation = upright;
    state.wheelCentre = place(hardpoints.wheelCentre);
    state.wheelTravel = state.wheelCentre.y - hardpoints.wheelCentre.y;

    readOffWheel(hardpoints, state);

    // Deliberately absent: the damper and the drop link. They are chassis-to-wishbone elements,
    // and every element quantity — length, Jacobian, motion ratio — is evaluated from the solved
    // state through `solveElement` and the role-typed kinematics since the legacy damper path was
    // retired (docs/suspension-geometry-audit.md, step 14). The state carries the linkage's own
    // degrees of freedom and nothing an element can derive.

    return state;
}

[[nodiscard]] std::expected<SuspensionState, std::string> solveCornerWithJacobian(const CornerHardpoints& hardpoints,
                                                                                  const double wishboneAngle,
                                                                                  const double rackTravel,
                                                                                  const SuspensionState* previous)
{
    auto solved = solveCorner(hardpoints, wishboneAngle, rackTravel, previous);
    if (!solved)
    {
        return solved;
    }

    // Central difference, and stepped in from the ends of the travel so the neighbour is always
    // inside the range the linkage has solutions over.
    constexpr auto step = 1e-6;
    const auto behind = solveCorner(hardpoints, wishboneAngle - step, rackTravel, &solved.value());
    const auto ahead = solveCorner(hardpoints, wishboneAngle + step, rackTravel, &solved.value());

    if (!behind || !ahead)
    {
        return std::unexpected("the linkage has no solution beside " + std::to_string(wishboneAngle) +
                               " rad, so its motion ratio cannot be differenced");
    }

    const auto travelChange = ahead->wheelCentre.y - behind->wheelCentre.y;

    solved->travelPerAngle = travelChange / (2.0 * step);

    // The whole patch velocity, from the two solves already paid for. Its y component is *not*
    // `travelPerAngle`: that is the wheel *centre*'s, and the patch sits a radius below it in the
    // wheel's own plane, so camber gain separates the two. Both are wanted — the corner's
    // generalised force is the tyre force through this one, and the unsprung terms act at the
    // centre.
    solved->patchPerAngle = (ahead->contactPatch - behind->contactPatch) / (2.0 * step);

    // The wheel centre's whole velocity, of which `travelPerAngle` above is the y component. Free
    // from the same two solves, and the corner's generalised inertia and the unsprung reaction on
    // the chassis both want the vector rather than the component.
    solved->wheelCentrePerAngle = (ahead->wheelCentre - behind->wheelCentre) / (2.0 * step);

    // And the upright's own angular velocity, from the same two orientations. The relative rotation
    // across the interval is tiny, so its vector part is half the rotation vector and the scalar part
    // is within an ulp of one — but the sign is still normalised, because a quaternion and its
    // negative are the same rotation and reading the vector part of the wrong one silently inverts
    // every axis.
    const auto turned = ahead->uprightOrientation * glm::conjugate(behind->uprightOrientation);
    const auto aligned = turned.w < 0.0 ? -turned : turned;
    solved->uprightRatePerAngle = glm::dvec3(aligned.x, aligned.y, aligned.z) / step;

    // At a turning point of the wheel's travel every element's motion ratio is genuinely infinite
    // — the linkage is at the end of its useful range — and reporting zero there is a lie the
    // force calculation would silently believe.
    if (std::abs(travelChange) < 1e-12)
    {
        return std::unexpected("the wheel does not move vertically at " + std::to_string(wishboneAngle) +
                               " rad, so the motion ratio is undefined; the travel range is too wide");
    }

    return solved;
}

void applyComplianceSteer(const CornerHardpoints& hardpoints, SuspensionState& state, const double steerAngle)
{
    // About the chassis's own up axis, which is what a toe angle is measured about and what the
    // published figure this is driven by was measured as (a K&C rig reads toe change against an
    // applied lateral force at the contact patch). Not about the kingpin: that would be a steering
    // input, and the bushes that deflect are the arm's and the tie rod's rather than the rack.
    state.uprightOrientation = glm::angleAxis(steerAngle, glm::dvec3(0.0, 1.0, 0.0)) * state.uprightOrientation;

    readOffWheel(hardpoints, state);
}

void applyComplianceCamber(const CornerHardpoints& hardpoints, SuspensionState& state, const double camberAngle)
{
    // About the chassis's own forward axis, which is what a camber angle is measured about and how
    // the K&C figure this is driven by is stated (camber change against a lateral force applied at
    // the contact patch).
    //
    // The sign, derived once so nobody re-derives it wrong. The frame is +x the car's left, +y up,
    // +z forward, and `glm::angleAxis` turns by the right-hand rule on components: a positive angle
    // about +z carries the point a radius *below* the hub — the patch — towards **+x** and the top
    // of the wheel towards −x. The caller passes `lateralForceCamber x (tyre lateral force resolved
    // into the body's +x)`, so with the published positive coefficient the patch complies *with* the
    // force, which is the source's own convention: positive means the contact point displaces
    // vehicle-inward under the rig's inward force and the tyre leans over it. On the loaded outside
    // wheel of a corner the road force points at the turn centre, so the patch tucks under and the
    // top of the wheel leans **out of the turn** — camber lost in the adverse direction, on both
    // wheels of the axle, which is the physical statement the measurement makes.
    //
    // `readOffWheel` then reports it with the sign convention it already owns: the spin axis picks
    // up a y component of `outboard x sin(angle)`, so the same lean reads as positive (top-out)
    // camber on the right wheel and negative on the left, which is SAE camber doing its job.
    state.uprightOrientation = glm::angleAxis(camberAngle, glm::dvec3(0.0, 0.0, 1.0)) * state.uprightOrientation;

    readOffWheel(hardpoints, state);
}

void computeRollCentre(const CornerHardpoints& hardpoints, SuspensionState& state)
{
    // The front view collapses z, so each wishbone becomes the line from where its pivot axis
    // crosses the ball joint's z-plane out to the ball joint itself.
    const auto pivotInView = [](const Wishbone& wishbone, const glm::dvec3& ballJoint) -> std::expected<glm::dvec2, int>
    {
        const auto along = wishbone.rearPivot - wishbone.frontPivot;
        if (std::abs(along.z) < 1e-9)
        {
            // The axis is square across the car, so it does not cross that plane at a point.
            return glm::dvec2(wishbone.frontPivot.x, wishbone.frontPivot.y);
        }

        const auto travel = (ballJoint.z - wishbone.frontPivot.z) / along.z;
        const auto point = wishbone.frontPivot + travel * along;

        return glm::dvec2(point.x, point.y);
    };

    const auto lowerInboard = pivotInView(hardpoints.lower, state.lowerBallJoint).value();
    const auto lowerOutboard = glm::dvec2(state.lowerBallJoint.x, state.lowerBallJoint.y);
    const auto lowerDirection = lowerOutboard - lowerInboard;

    // The upper link, which is where the two linkages part company. A wishbone has a real one and
    // it runs from its inboard pivot to its ball joint. A strut has none: the top bearing carries no
    // side load along the strut, so its effective link is the line *perpendicular* to the strut
    // axis through that bearing. Using the strut axis itself here — which is what falls out if a
    // strut is quietly treated as a wishbone — puts the instant centre in completely the wrong
    // place and the roll centre with it.
    auto upperInboard = glm::dvec2(0.0);
    auto upperDirection = glm::dvec2(0.0);

    if (hardpoints.kind == SuspensionKind::MacPhersonStrut)
    {
        upperInboard = glm::dvec2(state.upperBallJoint.x, state.upperBallJoint.y);

        const auto axis = glm::dvec2(state.upperBallJoint.x - state.lowerBallJoint.x,
                                     state.upperBallJoint.y - state.lowerBallJoint.y);
        upperDirection = glm::dvec2(-axis.y, axis.x);
    }
    else
    {
        upperInboard = pivotInView(hardpoints.upper, state.upperBallJoint).value();
        upperDirection = glm::dvec2(state.upperBallJoint.x, state.upperBallJoint.y) - upperInboard;
    }

    const auto cross = lowerDirection.x * upperDirection.y - lowerDirection.y * upperDirection.x;
    state.instantCentreDefined = std::abs(cross) > 1e-9;

    if (!state.instantCentreDefined)
    {
        // Parallel wishbones put the instant centre at infinity, which is a real and popular
        // geometry rather than an error — the roll centre is then simply at ground level.
        state.instantCentre = glm::dvec3(0.0);
        state.rollCentreHeight = 0.0;
        return;
    }

    const auto between = upperInboard - lowerInboard;
    const auto travel = (between.x * upperDirection.y - between.y * upperDirection.x) / cross;
    const auto centre = lowerInboard + travel * lowerDirection;

    state.instantCentre = glm::dvec3(centre.x, centre.y, state.lowerBallJoint.z);

    // The roll centre is where the line from the contact patch to the instant centre crosses the
    // car's centreline.
    const auto patch = glm::dvec2(state.contactPatch.x, state.contactPatch.y);
    const auto toCentre = centre - patch;

    if (std::abs(toCentre.x) < 1e-9)
    {
        state.rollCentreHeight = patch.y;
        return;
    }

    state.rollCentreHeight = patch.y + toCentre.y * (0.0 - patch.x) / toCentre.x;
}

[[nodiscard]] std::expected<SuspensionSweep, std::string>
sweepCorner(const CornerHardpoints& hardpoints, const std::uint32_t samples, const double rackTravel)
{
    if (samples < 2)
    {
        return std::unexpected("a sweep needs at least two samples");
    }

    if (!(hardpoints.bumpAngle > hardpoints.droopAngle))
    {
        return std::unexpected("the bump stop must sit above the droop stop in wishbone angle");
    }

    auto sweep = SuspensionSweep{};
    sweep.samples.reserve(samples);

    const SuspensionState* previous = nullptr;

    for (auto index = std::uint32_t{0}; index < samples; index++)
    {
        const auto through = static_cast<double>(index) / static_cast<double>(samples - 1);
        const auto angle = hardpoints.droopAngle + through * (hardpoints.bumpAngle - hardpoints.droopAngle);

        auto solved = solveCornerWithJacobian(hardpoints, angle, rackTravel, previous);
        if (!solved)
        {
            return std::unexpected("the linkage has no solution across its own stated travel: " + solved.error());
        }

        computeRollCentre(hardpoints, solved.value());
        sweep.samples.push_back(solved.value());
        previous = &sweep.samples.back();
    }

    return sweep;
}

[[nodiscard]] SuspensionElementState solveElement(const CornerHardpoints& hardpoints, const SuspensionElement& element,
                                                  const double wishboneAngle)
{
    // The same swing transform the solver applies to the damper and the drop link: the wishbone
    // end rides the lower arm's circle, oriented so that a positive angle is bump.
    const auto axis = swingOf(hardpoints.lower).normal;
    const auto lengthAt = [&](const double angle)
    {
        return glm::distance(element.chassis,
                             rotateAboutLine(element.wishbone, hardpoints.lower.frontPivot, axis, angle));
    };

    // The corner Jacobian's own step, so the two differencings cannot disagree by choice of step.
    constexpr auto step = 1e-6;

    return SuspensionElementState{.length = lengthAt(wishboneAngle),
                                  .lengthPerAngle =
                                      (lengthAt(wishboneAngle + step) - lengthAt(wishboneAngle - step)) / (2.0 * step)};
}

[[nodiscard]] SuspensionElement damperElementOf(const CornerHardpoints& hardpoints)
{
    if (hardpoints.kind == SuspensionKind::MacPhersonStrut)
    {
        return SuspensionElement{.chassis = hardpoints.strutTop, .wishbone = hardpoints.lower.ballJoint};
    }

    return SuspensionElement{.chassis = hardpoints.damperChassis, .wishbone = hardpoints.damperWishbone};
}

[[nodiscard]] SuspensionElement springElementOf(const CornerHardpoints& hardpoints)
{
    // The coil-over assumption, stated here and nowhere else: unless the corner mounts its spring
    // separately, the spring rides the damper's axis. No production car sets `OnLowerWishbone` —
    // the real Mk7 rear does mount its spring this way, and the seat coordinates are the unsourced
    // data step 3 of the audit ends on.
    if (hardpoints.springMount == SpringMount::OnLowerWishbone)
    {
        return SuspensionElement{.chassis = hardpoints.springChassis, .wishbone = hardpoints.springWishbone};
    }

    return damperElementOf(hardpoints);
}

[[nodiscard]] SuspensionElement dropLinkElementOf(const CornerHardpoints& hardpoints)
{
    return SuspensionElement{.chassis = hardpoints.antiRollBarChassis, .wishbone = hardpoints.antiRollBarWishbone};
}

[[nodiscard]] bool dropLinkStated(const CornerHardpoints& hardpoints)
{
    // Both default to the origin, so a car that says nothing about its bar's geometry says it here by
    // leaving the two points equal. Exact inequality on purpose: the question is whether coordinates
    // were typed, not whether they are nearly something.
    return hardpoints.antiRollBarChassis != hardpoints.antiRollBarWishbone;
}

[[nodiscard]] bool coaxialSpring(const CornerHardpoints& hardpoints)
{
    const auto spring = springElementOf(hardpoints);
    const auto damper = damperElementOf(hardpoints);

    return spring.chassis == damper.chassis && spring.wishbone == damper.wishbone;
}

namespace
{

// The ratio the role types carry, differenced exactly as `solveCornerWithJacobian` differences the
// damper's: raw element change over raw travel change, from solves at the same step with the same
// continuity seeding. On an element that IS the damper this reproduces the solver's `motionRatio`
// bit for bit — load-bearing, because the spring free length divides by this ratio and the parity
// gates hold the coaxial migration to byte-identity, which survives no regrouping.
[[nodiscard]] std::expected<SuspensionElementState, std::string>
elementWithRatio(const CornerHardpoints& hardpoints, const SuspensionElement& element, const double wishboneAngle,
                 const double rackTravel, double& motionRatio)
{
    const auto centre = solveCorner(hardpoints, wishboneAngle, rackTravel);
    if (!centre)
    {
        return std::unexpected(centre.error());
    }

    constexpr auto step = 1e-6;
    const auto behind = solveCorner(hardpoints, wishboneAngle - step, rackTravel, &centre.value());
    const auto ahead = solveCorner(hardpoints, wishboneAngle + step, rackTravel, &centre.value());
    if (!behind || !ahead)
    {
        return std::unexpected("the linkage has no solution beside " + std::to_string(wishboneAngle) +
                               " rad, so the element's motion ratio cannot be differenced");
    }

    const auto travelChange = ahead->wheelCentre.y - behind->wheelCentre.y;
    if (std::abs(travelChange) < 1e-12)
    {
        return std::unexpected("the wheel does not move vertically at " + std::to_string(wishboneAngle) +
                               " rad, so the element's motion ratio is undefined");
    }

    const auto lengthChange = solveElement(hardpoints, element, wishboneAngle + step).length -
                              solveElement(hardpoints, element, wishboneAngle - step).length;
    motionRatio = lengthChange / travelChange;

    return solveElement(hardpoints, element, wishboneAngle);
}

} // namespace

[[nodiscard]] std::expected<SpringKinematics, std::string> solveSpringKinematics(const CornerHardpoints& hardpoints,
                                                                                 const SuspensionElement& element,
                                                                                 const double wishboneAngle,
                                                                                 const double rackTravel)
{
    auto motionRatio = 0.0;
    const auto state = elementWithRatio(hardpoints, element, wishboneAngle, rackTravel, motionRatio);
    if (!state)
    {
        return std::unexpected(state.error());
    }

    return SpringKinematics{
        .length = state->length, .lengthPerAngle = state->lengthPerAngle, .motionRatio = motionRatio};
}

[[nodiscard]] std::expected<DamperKinematics, std::string> solveDamperKinematics(const CornerHardpoints& hardpoints,
                                                                                 const SuspensionElement& element,
                                                                                 const double wishboneAngle,
                                                                                 const double rackTravel)
{
    auto motionRatio = 0.0;
    const auto state = elementWithRatio(hardpoints, element, wishboneAngle, rackTravel, motionRatio);
    if (!state)
    {
        return std::unexpected(state.error());
    }

    return DamperKinematics{
        .length = state->length, .lengthPerAngle = state->lengthPerAngle, .motionRatio = motionRatio};
}

[[nodiscard]] std::expected<void, std::string> validateCorner(const CornerHardpoints& hardpoints)
{
    const auto sweep = sweepCorner(hardpoints, 81);
    if (!sweep)
    {
        return std::unexpected(sweep.error());
    }

    const auto& samples = sweep->samples;

    if (samples.front().wheelTravel >= samples.back().wheelTravel)
    {
        return std::unexpected("the bump stop does not raise the wheel relative to the chassis; the lower "
                               "wishbone's pivots are probably ordered the other way round");
    }

    for (auto index = std::size_t{1}; index < samples.size(); index++)
    {
        // Travel must be monotonic in the wishbone angle. Where it is not, the linkage has passed a
        // turning point: the motion ratio goes through infinity, and a solver stepping across it
        // sees the wheel reverse direction for no reason the dynamics can account for.
        if (samples[index].wheelTravel <= samples[index - 1].wheelTravel)
        {
            return std::unexpected("the wheel stops rising partway through the travel, at " +
                                   std::to_string(samples[index].wishboneAngle) +
                                   " rad; the stated range reaches past the linkage's turning point");
        }

        // A branch flip shows as the upright teleporting between neighbouring samples while the
        // angle barely moved.
        if (glm::distance(samples[index].upperBallJoint, samples[index - 1].upperBallJoint) > 0.05)
        {
            return std::unexpected("the upper ball joint jumps between adjacent samples near " +
                                   std::to_string(samples[index].wishboneAngle) +
                                   " rad; the solve is flipping to the mirrored configuration");
        }
    }

    return {};
}

} // namespace raceengine
