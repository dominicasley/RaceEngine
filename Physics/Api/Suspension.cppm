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
                          const double circleRadius, const glm::dvec3& circleNormal);

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
                                                                             const SuspensionState* previous = nullptr);

// The motion ratio, by differencing the solve. Analytic would be faster and is not obviously
// clearer; this is one extra evaluation of a few hundred flops, and it cannot drift out of
// agreement with the geometry it is the derivative of.
export [[nodiscard]] std::expected<SuspensionState, std::string>
solveCornerWithJacobian(const CornerHardpoints& hardpoints, const double wishboneAngle, const double rackTravel,
                        const SuspensionState* previous = nullptr);

// Roll centre and instantaneous centre, in the front view. Diagnostics only, and the reason that is
// stated twice: a roll centre is a *consequence* of where the wishbones point, and a model that
// feeds it back into a force is modelling its own output.
export void computeRollCentre(const CornerHardpoints& hardpoints, SuspensionState& state);

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
sweepCorner(const CornerHardpoints& hardpoints, const std::uint32_t samples = 41, const double rackTravel = 0.0);

// Refuses a corner whose geometry cannot do its own stated travel, rather than letting it fail at
// whichever tick first reaches the angle that breaks it. Also catches the two authoring mistakes
// that produce a plausible car rather than an obvious failure: pivots ordered so that "bump"
// droops, and a linkage that passes through a configuration where the wheel stops moving vertically.
export [[nodiscard]] std::expected<void, std::string> validateCorner(const CornerHardpoints& hardpoints);

} // namespace raceengine
