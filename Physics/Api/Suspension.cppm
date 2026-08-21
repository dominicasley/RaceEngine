module;

#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:Suspension;

namespace raceengine
{

// The suspension solved from real hardpoint geometry, every tick, with no baked curves anywhere in
// it. Camber gain, bump steer, motion-ratio variation and roll-centre migration are not authored
// here and cannot be: they are what the linkage *does*, and the only way to change them is to move
// a hardpoint. That is the whole point — a car whose camber curve is a lookup table has a camber
// curve, and a car whose camber curve comes out of its wishbones has a suspension.
//
// Pure geometry: hardpoints and one travel parameter in, wheel transform and Jacobian out. No
// dynamics, no forces, no time. That separation is what lets this be validated on its own, so that
// when the tire misbehaves three milestones from now the suspension is not also a suspect.
//
// Chassis frame, SI: **+x is the car's left** (see `outboardSign`), +y up, +z forward, metres, radians.

export enum class CornerSide : std::uint32_t { Left, Right };

// **+x is the car's LEFT, and this function is the only place that is allowed to know it.**
//
// The frame is "+x, +y up, +z forward" and it is **left-handed**: right times up is *backward*, not
// forward. Everything that draws it — `glm::lookAt`, and so the whole renderer — is right-handed, so
// screen-right works out as `cross(forward, up)` = `cross(+z, +y)` = **−x**. The picture is not
// wrong; Bathurst reads correctly on screen, which is the check that settled it. What is on the
// left of the screen is on the car's left, and that is +x.
//
// This was stated in five places and got the wrong answer in all of them, and the cost was a day.
// Written as `Right ? +1 : -1`, `CornerSide::Right` was built at +x — the car's *left* — so every
// corner was labelled as its own mirror image. Mostly that is invisible, because a car is
// symmetric and a mirrored car obeys the same physics; it surfaces exactly where a left or a right
// has to be *named*:
//
//   - `rackTravelForSteer` probes what it believes is the right front and reads its toe. Toe is a
//     mirrored quantity, so reading it off the wrong side inverted the answer, and the Golf's rack
//     came out at −0.0700 when the car needs +0.0700. The setup sheet had been quietly correcting
//     that with `steering.invert 0` since 2026-08-20 — a rendering-frame fault being compensated in
//     the vehicle, two layers away from where it lived.
//   - Telemetry's `Susp Pos FL` and `Damper Vel FL` carried the *right* corner's data, in a file
//     whose entire purpose is dropping into i2 beside a real car's.
//   - The cockpit camera's seat offset, which had to be measured off a picture twice.
//
// Two things about it are worth keeping. The frame being left-handed does **not** make the
// simulation wrong — it simulates a mirror-image car, and a mirror-image car is a car. And it
// cannot be settled by reasoning, because every convention it would be reasoned from is one of the
// five statements: the picture is the only oracle, which is why the question that ended it was
// "does the circuit look like Bathurst".
export [[nodiscard]] constexpr double outboardSign(const CornerSide side)
{
    return side == CornerSide::Right ? -1.0 : 1.0;
}

// Which linkage this corner is. The two solve differently in exactly one step and are otherwise the
// same problem, which is why they sit behind one function and one output struct rather than behind
// an abstract base: what a caller wants from a suspension is a wheel transform and a Jacobian, and
// that does not vary by type.
export enum class SuspensionKind : std::uint32_t { DoubleWishbone, MacPhersonStrut };

// A wishbone is two inboard pivots and one outboard ball joint. The line through the pivots is the
// axis it swings about, so the ball joint traces a circle in chassis space and the whole linkage
// has one degree of freedom plus steering.
export struct Wishbone
{
    glm::dvec3 frontPivot{0.0};
    glm::dvec3 rearPivot{0.0};
    glm::dvec3 ballJoint{0.0};
};

export struct CornerHardpoints
{
    CornerSide side = CornerSide::Right;
    SuspensionKind kind = SuspensionKind::DoubleWishbone;

    Wishbone lower;
    // Double wishbone only. A strut has no upper arm: its top mount below plays that part.
    Wishbone upper;

    // MacPherson only: where the strut's top bearing sits in the chassis. The strut is telescopic
    // and its lower end *is* the lower ball joint, so the line from here to there is the kingpin
    // axis — which is why a strut has so much more kingpin inclination than a wishbone car, and why
    // its camber curve is the shape it is.
    glm::dvec3 strutTop{0.0};

    // Inboard on the chassis, outboard on the lower wishbone — which is where a damper usually
    // picks up, and why the motion ratio is not one.
    glm::dvec3 damperChassis{0.0};
    glm::dvec3 damperWishbone{0.0};

    // The drop link, chassis end and wishbone end. The bar's torsion comes from the difference in
    // these across an axle, so a corner on its own reports only its end of it.
    glm::dvec3 antiRollBarChassis{0.0};
    glm::dvec3 antiRollBarWishbone{0.0};

    // The rack's outer joint at centred steering, and the arm on the upright it pulls. The distance
    // between them at design *is* the tie rod's length; it is not stated separately, so it cannot
    // disagree with the geometry.
    glm::dvec3 steeringRackOuter{0.0};
    glm::dvec3 steeringArm{0.0};

    glm::dvec3 wheelCentre{0.0};
    double wheelRadius = 0.33;

    // The travel range, as lower-wishbone angle. Radians, and signed so that positive is whatever
    // direction compresses this corner — which `validateCorner` checks rather than assumes, because
    // it depends on which way round the pivots were authored.
    //
    // Droop is not optional. Without it the linkage runs to its geometric limit whenever a wheel
    // leaves the ground, which in this game is over every kerb and off every ramp, and past that
    // limit the sphere-circle intersection below simply has no solution.
    double droopAngle = -0.20;
    double bumpAngle = 0.20;
};

// Two circles' worth of ambiguity, resolved by continuity. Exported because it is the one piece of
// arithmetic the whole solve rests on and it is worth testing on its own.
export struct SphereCircleIntersection
{
    glm::dvec3 first{0.0};
    glm::dvec3 second{0.0};
    std::uint32_t count = 0;
};

// A sphere meets a circle in at most two points. Both of the solve's steps are this: the upper ball
// joint is a fixed distance from the lower one and lies on the upper wishbone's circle; the
// steering arm is a tie rod's length from the rack and lies on a circle about the kingpin axis.
export [[nodiscard]] SphereCircleIntersection
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

export struct SuspensionState
{
    glm::dvec3 lowerBallJoint{0.0};
    glm::dvec3 upperBallJoint{0.0};
    glm::dvec3 steeringArm{0.0};
    glm::dvec3 wheelCentre{0.0};
    glm::dquat uprightOrientation{1.0, 0.0, 0.0, 0.0};

    // Where the tire's forces are applied. Never the wheel centre — applying them there is what
    // loses the jacking force and the lateral load path, which come out for free when the force
    // acts where it actually acts.
    glm::dvec3 contactPatch{0.0};

    // Read off the wheel's orientation rather than stored beside it, so they cannot disagree with
    // it. Camber is negative when the top of the wheel leans toward the car; toe is positive when
    // the front of the wheel points toward it, on both sides.
    double camber = 0.0;
    double toe = 0.0;
    double halfTrack = 0.0;
    double wheelTravel = 0.0;

    double damperLength = 0.0;
    // dDamperLength/dWheelTravel at this point in the travel, and it varies across it — which is
    // the reason for solving the linkage rather than storing a number. The spring and damper force
    // at the wheel is the force along the damper times this.
    double motionRatio = 0.0;
    // dWheelTravel/dWishboneAngle, and dDamperLength/dWishboneAngle. The corner's degree of freedom
    // is the wishbone angle, so these are the two Jacobians its equation of motion is written in:
    // the first turns a force at the contact patch into a generalised force, the second does the
    // same for a force along the damper. Their ratio is the motion ratio above, which is why that
    // one is the readable diagnostic and these two are what the dynamics actually use.
    double travelPerAngle = 0.0;
    double damperLengthPerAngle = 0.0;

    double dropLinkLength = 0.0;
    double wishboneAngle = 0.0;

    // The outboard ends of the damper and the drop link, which swing with the lower wishbone. The
    // vehicle needs them to apply forces along the real axes rather than along a guess at them.
    glm::dvec3 damperWishbone{0.0};
    glm::dvec3 dropLinkWishbone{0.0};

    // Diagnostics, and outputs of the geometry rather than inputs to anything. Nothing below may
    // read these to compute a force; they exist to be plotted and argued about.
    glm::dvec3 instantCentre{0.0};
    double rollCentreHeight = 0.0;
    bool instantCentreDefined = false;
};

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

// One corner, solved. `previous` is what resolves the two-root ambiguity at both steps: each
// intersection has a mirrored solution that is a perfectly valid linkage and an entirely wrong car,
// so the root is chosen by continuity with where the corner was last tick. Passing nothing picks
// the root nearest the design position, which is the right answer for the first solve and for any
// query that is not part of a sequence.
//
// This is closed form throughout — two sphere-circle intersections and some algebra, a few hundred
// flops. There is no iteration and no solver to fail to converge.
export [[nodiscard]] std::expected<SuspensionState, std::string> solveCorner(const CornerHardpoints& hardpoints,
                                                                             const double wishboneAngle,
                                                                             const double rackTravel,
                                                                             const SuspensionState* previous = nullptr)
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

    // The wheel's spin axis, pointing away from the car. Everything about how the wheel is standing
    // is read from this rather than tracked beside it.
    const auto outboard = outboardSign(hardpoints.side);
    const auto spinAxis = upright * glm::dvec3(outboard, 0.0, 0.0);

    state.camber = -std::asin(glm::clamp(spinAxis.y, -1.0, 1.0));
    state.toe = std::asin(glm::clamp(spinAxis.z, -1.0, 1.0));

    // The contact patch is directly below the wheel centre *in the wheel's own plane*, which for a
    // cambered wheel is not directly below it in the world.
    const auto down = glm::dvec3(0.0, -1.0, 0.0);
    const auto inPlaneDown = down - glm::dot(down, spinAxis) * spinAxis;
    state.contactPatch = state.wheelCentre + hardpoints.wheelRadius * glm::normalize(inPlaneDown);
    state.halfTrack = std::abs(state.contactPatch.x);

    // Where the spring and damper act. On a double wishbone that is a separate unit picking up on
    // the lower arm, so its outboard end swings with the arm. On a strut there is no separate unit:
    // the strut *is* the damper, running from its top bearing down to the lower ball joint, so its
    // "outboard end" is that joint and its length is the kingpin's. That is also why a strut's
    // motion ratio is near one where a wishbone car's is near a half.
    const auto damperOutboard =
        strut ? lower
              : rotateAboutLine(hardpoints.damperWishbone, hardpoints.lower.frontPivot, lowerAxis, wishboneAngle);
    const auto damperInboard = strut ? hardpoints.strutTop : hardpoints.damperChassis;
    state.damperWishbone = damperOutboard;
    state.damperLength = glm::distance(damperInboard, damperOutboard);

    const auto dropLinkOutboard =
        rotateAboutLine(hardpoints.antiRollBarWishbone, hardpoints.lower.frontPivot, lowerAxis, wishboneAngle);
    state.dropLinkWishbone = dropLinkOutboard;
    state.dropLinkLength = glm::distance(hardpoints.antiRollBarChassis, dropLinkOutboard);

    return state;
}

// The motion ratio, by differencing the solve. Analytic would be faster and is not obviously
// clearer; this is one extra evaluation of a few hundred flops, and it cannot drift out of
// agreement with the geometry it is the derivative of.
export [[nodiscard]] std::expected<SuspensionState, std::string>
solveCornerWithJacobian(const CornerHardpoints& hardpoints, const double wishboneAngle, const double rackTravel,
                        const SuspensionState* previous = nullptr)
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
    const auto damperChange = ahead->damperLength - behind->damperLength;

    solved->travelPerAngle = travelChange / (2.0 * step);
    solved->damperLengthPerAngle = damperChange / (2.0 * step);

    // At a turning point of the wheel's travel the ratio is genuinely infinite — the linkage is at
    // the end of its useful range — and reporting zero there is a lie the force calculation would
    // silently believe.
    if (std::abs(travelChange) < 1e-12)
    {
        return std::unexpected("the wheel does not move vertically at " + std::to_string(wishboneAngle) +
                               " rad, so the motion ratio is undefined; the travel range is too wide");
    }

    solved->motionRatio = damperChange / travelChange;

    return solved;
}

// Roll centre and instantaneous centre, in the front view. Diagnostics only, and the reason that is
// stated twice: a roll centre is a *consequence* of where the wishbones point, and a model that
// feeds it back into a force is modelling its own output.
export void computeRollCentre(const CornerHardpoints& hardpoints, SuspensionState& state)
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

// The load-time diagnostic the brief asks for, and the reason it is not optional: bump steer that
// emerges from the geometry is correct and desirable, and bump steer that emerges from a typo in a
// hardpoint is a car that steers itself over every bump and looks for all the world like a physics
// bug. Sweeping the travel and printing what the linkage actually does is how the second is caught
// at the point the data is read.
export struct SuspensionSweep
{
    std::vector<SuspensionState> samples;
};

export [[nodiscard]] std::expected<SuspensionSweep, std::string>
sweepCorner(const CornerHardpoints& hardpoints, const std::uint32_t samples = 41, const double rackTravel = 0.0)
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

// Refuses a corner whose geometry cannot do its own stated travel, rather than letting it fail at
// whichever tick first reaches the angle that breaks it. Also catches the two authoring mistakes
// that produce a plausible car rather than an obvious failure: pivots ordered so that "bump"
// droops, and a linkage that passes through a configuration where the wheel stops moving vertically.
export [[nodiscard]] std::expected<void, std::string> validateCorner(const CornerHardpoints& hardpoints)
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
