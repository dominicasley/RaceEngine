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

// Where a corner's spring is mounted; see the fields on `CornerHardpoints`.
export enum class SpringMount : std::uint32_t { CoaxialWithDamper, OnLowerWishbone };

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

    // Where the spring acts. `CoaxialWithDamper` is the coil-over: the spring rides the damper's
    // own element and the two points below are ignored — which is every car here today, and the
    // truth for a strut front. `OnLowerWishbone` is the Mk7's sourced rear topology — spring and
    // damper on the same link at different stations (docs/suspension-geometry-audit.md, step 3) —
    // and reads the two points. No production car sets it: the real seat coordinates are the
    // unsourced data the audit's step 3 ends on.
    SpringMount springMount = SpringMount::CoaxialWithDamper;
    glm::dvec3 springChassis{0.0};
    glm::dvec3 springWishbone{0.0};

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

    // Where the tire's forces are applied. Never the wheel centre — that is the point the resultant
    // a corner hands the chassis genuinely passes through, so the roll couple and the total load
    // transfer come out right for free.
    //
    // **It does not on its own buy the jacking force or the geometric load path, and the comment
    // that said so was half right about the half that does not matter.** Those live on the
    // *corner's* side of the linkage — in `patchPerAngle` below — because what decides whether a
    // spring has to deflect is virtual work in the corner's own coordinate, not where a force is
    // applied to the body. See docs/suspension-load-path-brief.md, section 4.
    glm::dvec3 contactPatch{0.0};

    // Read off the wheel's orientation rather than stored beside it, so they cannot disagree with
    // it. Camber is negative when the top of the wheel leans toward the car; toe is positive when
    // the front of the wheel points toward it, on both sides.
    double camber = 0.0;
    double toe = 0.0;
    double halfTrack = 0.0;
    double wheelTravel = 0.0;

    // dWheelTravel/dWishboneAngle: the corner's degree of freedom is the wishbone angle, so this
    // is the Jacobian that turns a force at the contact patch into a generalised force. Every
    // damper-side quantity — length, Jacobian, motion ratio — lives on the elements now
    // (`solveElement`, `solveDamperKinematics`); the state stopped carrying its own copies when
    // the legacy damper path was retired (docs/suspension-geometry-audit.md, step 14).
    double travelPerAngle = 0.0;

    // d(contactPatch)/dWishboneAngle, chassis frame, metres per radian — the **whole** velocity of
    // the point the road pushes on, not just its vertical component.
    //
    // This is the geometric load path, and it is not a feature so much as the rest of one dot
    // product. A corner is one degree of freedom, so what the spring has to carry is the tyre force
    // dotted with the derivative of its point of application. `travelPerAngle` is that derivative's
    // y component alone, so a model built on it takes only the vertical part of the vertical force
    // and drops the in-plane parts entirely — which is exactly the same statement as "no roll
    // centre, no anti-dive, no anti-squat, no jacking", because those *are* the in-plane terms.
    //
    // Two cross-checks tie it to things this file already computes. The lateral ratio is the roll
    // centre: `patchPerAngle.x / patchPerAngle.y == rollCentreHeight / contactPatch.x`, because both
    // are statements about the direction the patch scrubs as the wheel rises. And the longitudinal
    // ratio is the side-view swing arm, which is anti-dive and anti-lift.
    //
    // Differenced from the same two neighbouring solves as `travelPerAngle`, at the same rack
    // travel, so it costs nothing and cannot disagree with it by choice of method.
    glm::dvec3 patchPerAngle{0.0};

    // d(wheelCentre)/dWishboneAngle, chassis frame, metres per radian — the **whole** velocity of the
    // wheel centre, of which `travelPerAngle` is the y component alone.
    //
    // Differenced from the same two neighbouring solves as everything else here, so it is free. Two
    // things want it and neither is the tyre force. The corner's generalised inertia is
    // `m·|dC/dq|²`, and taking only the vertical component models a corner as lighter than it is
    // (docs/suspension-fidelity-brief.md, item 4). And the unsprung mass's reaction on the chassis
    // acts along *this* direction rather than along the body's up, which is what a wheel snatched
    // sideways by a kerb kicks the body with.
    glm::dvec3 wheelCentrePerAngle{0.0};

    // dTheta_upright/dWishboneAngle, chassis frame, radians per radian — the upright's whole angular
    // velocity as the arm swings, differenced from the same two solves' orientations.
    //
    // **This is the term that separates an inboard drive from an outboard brake, and nothing else in
    // this file can express it.** `contactPatch` above is reconstructed from the world's down
    // direction each solve, so it is *not* a material point of the upright: a rotation of the upright
    // about the wheel's own spin axis moves the real contact patch and leaves this one exactly where
    // it was. Measured, and it is not a rounding effect — `patchPerAngle.z / patchPerAngle.y` and the
    // wheel centre's own ratio agree to every digit printed, on all four corners, because the
    // difference between them is precisely the spin-axis rotation the construction drops.
    //
    // So the honest patch Jacobian is the wheel centre's plus `uprightRatePerAngle × (R · down)`,
    // which is what `SuspensionState` cannot state on its own because the loaded radius is a
    // dynamic quantity. The consumer assembles it; see `VehicleImpl.cpp`.
    glm::dvec3 uprightRatePerAngle{0.0};

    double wishboneAngle = 0.0;

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

// Steer the solved wheel by a small angle about the chassis's up axis, and re-read everything the
// upright's orientation decides — camber, toe, the contact patch and the half track.
//
// **This is the seam bushing compliance arrives through, and it is deliberately outside
// `solveCorner`.** That function is pure geometry: hardpoints and one travel parameter in, wheel
// transform out, no forces and no time. Compliance is a *force* effect — a rubber bush deflecting
// under the tyre's lateral load — so it cannot live inside a solve that is not told about forces,
// and putting it there would also make the kinematic solve unrepeatable.
//
// What it is not: a translation. A real bush deflects the upright sideways and rearward as well as
// twisting it, and only the toe term has a published figure behind it
// (docs/suspension-fidelity-brief.md, item 1). So the hub stays exactly where the linkage put it and
// the wheel turns about it, which is the part that is sourced and nothing else.
//
// The Jacobians are left alone for the same reason they are left alone by steering: they are the
// linkage's, differenced from the linkage's own solves, and a hundredth of a degree of bush twist is
// not a change to what the wishbones do.
export void applyComplianceSteer(const CornerHardpoints& hardpoints, SuspensionState& state, const double steerAngle);

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

// A force-bearing element between the chassis and the lower wishbone: a damper, a spring seated on
// the arm, a drop link. Two points are its whole geometry — the chassis end is fixed, the wishbone
// end swings with the arm — and length, Jacobian and motion ratio are evaluated from the solved
// linkage rather than stored beside it.
//
// This exists because the real Mk7 rear carries its spring and its damper on the *same* link at
// *different* stations (docs/suspension-geometry-audit.md, step 3), which one coaxial element
// cannot express. **Nothing in the production force path reads it yet** — the coil-over model in
// `CornerSetup` is unchanged; this is the geometry API a migration will consume, proven equivalent
// to the existing damper geometry first.
export struct SuspensionElement
{
    glm::dvec3 chassis{0.0};
    glm::dvec3 wishbone{0.0};
};

// Length and its derivative against the corner's generalised coordinate. Closed form — the element
// rides a known circle, so unlike the linkage solve this cannot fail.
export struct SuspensionElementState
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
};

// Evaluated with the same swing transform the solver applies to the damper and the drop link, and
// differenced at the same step as the corner Jacobian, so the two derivatives cannot disagree by
// choice of method.
export [[nodiscard]] SuspensionElementState solveElement(const CornerHardpoints& hardpoints,
                                                         const SuspensionElement& element, const double wishboneAngle);

// The element each role is attached to today. The damper's is the authored pair — and a strut *is*
// the damper, so a strut corner's element runs from the top bearing to the lower ball joint. The
// spring's is the damper's own element, because no car here states a separate spring seat: that is
// the coil-over assumption, stated in one place instead of implied everywhere. A car with a sourced
// seat changes what `springElementOf` answers and nothing else.
export [[nodiscard]] SuspensionElement damperElementOf(const CornerHardpoints& hardpoints);
export [[nodiscard]] SuspensionElement springElementOf(const CornerHardpoints& hardpoints);

// The anti-roll bar's drop link, as an element like the other two — chassis end fixed, wishbone end
// swinging with the lower arm.
//
// **A corner states a drop link or it does not, and most do not.** The two hardpoints default to the
// origin, and a corner that leaves them there has no link geometry at all: the bar then stays on the
// wheel-referred model it has always been on, which is what `dropLinkStated` decides. That is the
// same shape as `SpringMount::CoaxialWithDamper` — the data says which model applies, and a car with
// nothing authored gets the arithmetic it always got, to the bit.
//
// The Golf is one of the cars that does not state one, and that is a data fact rather than an
// oversight: `suspensions.ini` gives `ARB FRONT 34000 / REAR 15000` and no coordinates whatever.
// docs/suspension-fidelity-brief.md said the points were "authored on every corner"; they are
// authored on the *placeholder* corner and on nothing else.
export [[nodiscard]] SuspensionElement dropLinkElementOf(const CornerHardpoints& hardpoints);

// Whether this corner authors a drop link. Exact inequality of the two points, deliberately and for
// the same reason `coaxialSpring` uses exact equality: the condition is "somebody typed coordinates
// here", not "the coordinates are nearly something".
export [[nodiscard]] bool dropLinkStated(const CornerHardpoints& hardpoints);

// Whether this corner's spring rides the damper's own element — the authored coil-over condition,
// decided from the hardpoints alone: the two elements are the same attachment points, exactly.
// This keys the generalised assembly's fused branch, which exists to keep `(s+d+stops)·x` in the
// pre-split arithmetic where all the Jacobians are one number; it is exact point equality and
// deliberately not a tolerance, because the branch's correctness condition is "the same bits",
// not "close". Derived at the point of use rather than stored, so a rebuilt setup cannot carry a
// stale answer.
export [[nodiscard]] bool coaxialSpring(const CornerHardpoints& hardpoints);

// Role-distinct kinematics, deliberately two types with no conversion between them. A wheel rate
// comes from the spring's ratio and a shaft speed from the damper's; after the rear split those are
// different numbers, so handing one to the other's arithmetic should not compile. The convention is
// the solver's own: motionRatio = dElementLength/dWheelTravel, **signed**, negative on a healthy
// corner, and consumers take the magnitude where a magnitude is meant.
export struct SpringKinematics
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
    double motionRatio = 0.0;
};

export struct DamperKinematics
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
    double motionRatio = 0.0;
};

// The ratio is differenced exactly as `solveCornerWithJacobian` differences its own — raw element
// change over raw travel change, from the same solves at the same step — so on an element that IS
// the damper the two ratios are the same bits, not merely close. That is load-bearing: the spring
// free length divides by this ratio, and the parity gates hold the coaxial migration to
// byte-identity, which survives no regrouping. Refused where the wheel does not move vertically,
// exactly as the solver refuses its own motion ratio there.
export [[nodiscard]] std::expected<SpringKinematics, std::string>
solveSpringKinematics(const CornerHardpoints& hardpoints, const SuspensionElement& element, const double wishboneAngle,
                      const double rackTravel = 0.0);

export [[nodiscard]] std::expected<DamperKinematics, std::string>
solveDamperKinematics(const CornerHardpoints& hardpoints, const SuspensionElement& element, const double wishboneAngle,
                      const double rackTravel = 0.0);

} // namespace raceengine
