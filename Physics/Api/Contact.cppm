module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:Contact;

import :PhysicsWorld;
import :RigidBody;

extern "C++" std::uint32_t raceengineJoltCollideBox(std::uint64_t handle, const double* halfExtents,
                                                    const double* centre, const double* orientation,
                                                    std::uint32_t maxContacts, double* outPoints, double* outNormals,
                                                    double* outDepths, std::uint32_t* outProps);
extern "C++" void raceengineJoltReadPropDynamics(std::uint64_t handle, const std::uint32_t* propIndices,
                                                 std::uint32_t count, double* outInverseMasses,
                                                 double* outInverseInertias, double* outCentres,
                                                 double* outLinearVelocities, double* outAngularVelocities,
                                                 double* outBreakForces, double* outBreakTorques,
                                                 unsigned char* outAnchored);

namespace raceengine
{

// Contact between the car's body and the world, resolved by us.
//
// Jolt generates the manifolds and does not touch them afterwards. That division is the whole point:
// ramming, PIT manoeuvres and roadblocks are this game's core verb, so what happens when two things
// touch has to be repeatable and tunable rather than merely plausible, and a general-purpose solver
// optimised for stacking crates is neither.
//
// Tire-to-road contact does **not** come through here. That is the tire model's, and mixing the two
// would have the tire's carefully shaped force fighting a rigid constraint that knows nothing about
// slip.

// The other side of a contact.
//
// **Zero inverse mass is an immovable world**, and that is the road, a building, and a prop that has
// not broken yet — a standing prop is anchored to the ground and its effective mass really is
// infinite, so reporting zero for one is the true answer rather than a special case.
//
// The zeros are also the safety argument for the whole two-body path: adding a zero inverse mass and
// subtracting a zero velocity are *exact* in floating point, so a contact against an immovable second
// body produces bit-identical arithmetic to the one-body solver this replaced. Both frame gates are
// blessed on a track that carries no props, and that is what lets them stay byte-identical.
export struct ContactBody
{
    // Which breakable prop this is, or `noProp` for anything that does not move. It is how the
    // velocity the solver gave this body gets back to the thing that owns it.
    std::uint32_t prop = noProp;

    double inverseMass = 0.0;

    // World frame, and **held constant across the tick**. That is standard for a sequential-impulse
    // solver rather than an approximation worth correcting: a body turns by a fraction of a degree
    // over one tick, and re-deriving the tensor per iteration would make the effective mass a moving
    // target for an accumulation whose whole job is to converge on it.
    glm::dmat3 inverseInertia{0.0};

    // Centre of mass, world. What the impulse arms are taken about, because it is the only frame the
    // inertia above means anything in.
    glm::dvec3 centre{0.0};

    // Velocity, tracked **locally across the solver's iterations and never re-read from the backend
    // part way through**. The solve is changing this body; an iteration that went back to the world
    // for its velocity would be solving against something it had already pushed.
    glm::dvec3 linearVelocity{0.0};
    glm::dvec3 angularVelocity{0.0};

    // What the solve added to the two velocities above, for whoever owns the far body to apply. Left
    // at exactly zero for an immovable one, because a zero inverse mass and a zero inverse inertia
    // make every increment exactly zero rather than merely small.
    glm::dvec3 deltaLinear{0.0};
    glm::dvec3 deltaAngular{0.0};

    // --- The anchor -----------------------------------------------------------------------------
    //
    // A prop that is still standing is bolted to the ground, and while that anchor holds the two
    // fields above are zero and the body is a building. `releaseBrokenProps` is what decides, from
    // the approach load and before any impulse is applied, whether it holds — and if it does not, it
    // moves the two `free` values below into the live fields and sets `released`.
    //
    // **Why the decision has to happen there and not after the solve.** The load an anchor is worth
    // is not the impulse a rigid solver computes against it: against an immovable body that impulse
    // is whatever it takes to stop the whole car, sized by the *car's* mass. Handing that to a
    // twenty-kilogram bin launched it at seventy times the car's approach speed. Scaling it down is
    // not enough either, because by then the solver has already put the car's momentum into an
    // anchor that was about to break, and the car is stopped dead by a bin. The only place both are
    // right is before the solve.
    bool anchored = false;
    bool released = false;

    // Newtons and newton-metres the anchor holds to. Zero for the immovable world and for a prop
    // already loose, neither of which has an anchor left to test.
    double breakForce = 0.0;
    double breakTorque = 0.0;

    // What this body would weigh once its anchor gives. Held apart from the live fields so that the
    // safe default is the old behaviour: a manifold nobody ran `releaseBrokenProps` over resolves
    // every prop as immovable, which is what shipped before props could break at all.
    double freeInverseMass = 0.0;
    glm::dmat3 freeInverseInertia{0.0};
};

// Two orthonormal axes spanning the plane of a normal.
export struct TangentBasis
{
    glm::dvec3 first{1.0, 0.0, 0.0};
    glm::dvec3 second{0.0, 0.0, 1.0};
};

// The tangent plane, as a **pure function of the normal** and of nothing else.
//
// Not of the sliding direction, which is what the scalar friction accumulator this replaced was
// implicitly quoted against: a stored magnitude is only meaningful if the axis it is stored along
// holds still. Deterministic for the same reason a capture has to reproduce across machines and
// sessions — the least-aligned world axis is picked, so the cross product is never near degenerate
// and there is no input where a rounding difference takes a different branch.
export [[nodiscard]] TangentBasis tangentBasis(const glm::dvec3& normal);

export struct ContactPoint
{
    glm::dvec3 position{0.0};
    // Points out of the world and into the body — the direction the body has to move to separate.
    glm::dvec3 normal{0.0, 1.0, 0.0};
    double penetration = 0.0;

    // Which breakable prop this landed on, or `noProp` for the road, a building and anything else
    // that does not move. It is what lets the solver's own impulse be handed back to the thing it
    // was applied to, which is the whole of how a lamp column comes out of the ground here — there
    // is no constraint anywhere and nothing reads a lambda.
    std::uint32_t prop = noProp;

    // The far side of this contact: an index into `ContactManifold::bodies`, whose **element 0 is
    // the immovable world**. A manifold that carries no table at all resolves every point against an
    // immovable body, which is exactly the arithmetic that shipped before a second body existed.
    std::size_t body = 0;

    // The two axes friction is solved along, built once from the normal and fixed for the whole
    // solve.
    //
    // **This is what one `tangentImpulse` scalar could not do.** Its direction was recomputed from
    // the current relative velocity on every one of the eight iterations, so the accumulation that
    // gets clamped against the friction cone was a sum of impulses taken along *different*
    // directions — a magnitude with no direction attached to it, which is also why a friction impulse
    // could not be handed to the prop it was applied to.
    glm::dvec3 tangent1{1.0, 0.0, 0.0};
    glm::dvec3 tangent2{0.0, 0.0, 1.0};

    // Accumulated impulses, carried across the solver's iterations. Keeping them per point and
    // clamping the *accumulation* rather than each increment is what makes sequential impulse
    // converge instead of oscillating: an iteration is allowed to take back what a previous one
    // overshot, so long as the total never pulls.
    double normalImpulse = 0.0;
    double tangentImpulse1 = 0.0;
    double tangentImpulse2 = 0.0;
};

export struct ContactManifold
{
    std::vector<ContactPoint> points;

    // The far side of each contact, indexed by `ContactPoint::body`. **Element 0 is the immovable
    // world**, and the table is left empty entirely when nothing but the immovable world was touched
    // — a point naming index 0 against an empty table resolves against an immovable body all the
    // same, so a circuit with no street furniture allocates nothing here.
    //
    // A table rather than a body per point because a flat hit on one bin lands on four points of the
    // same body, and four copies of a velocity the solver is changing would be four bodies that
    // disagree with each other by the second iteration.
    std::vector<ContactBody> bodies;
};

// What a pair of surfaces does when they meet. Per pair, because a bumper against armco and a
// bumper against a kerb are different conversations.
export struct ContactMaterial
{
    double friction = 0.6;
    double restitution = 0.1;

    // Below this closing speed, restitution is ignored entirely.
    //
    // Without it a resting contact bounces for ever: every tick the solver pushes the body out, the
    // body falls back, and restitution hands back a fraction of a millimetre per second of it. The
    // car buzzes. A real collision only bounces when it arrives with some speed, and this is where
    // that is stated. Half a metre per second is about a centimetre of drop.
    double restitutionThreshold = 0.5;

    // How much of the remaining penetration to push out per second, and how much to leave alone.
    //
    // Baumgarte stabilisation, with a slop. The slop matters as much as the gain: correcting *all*
    // the penetration means the solver is always fighting for the last micron, which is energy it
    // adds to the system and reads as jitter. Leaving a few millimetres uncorrected costs nothing
    // visible and is what lets a parked car sit still.
    double correction = 0.15;
    double allowedPenetration = 0.004;
};

// The body's collision shape, as a box in the chassis frame. A box because that is what a car's
// collider is in every driving game that has ever shipped, and because the manifold it generates
// against a triangle mesh is the shape the solver below wants — a handful of points on a face,
// rather than one deepest point.
export struct CollisionBox
{
    glm::dvec3 centre{0.0};
    glm::dvec3 halfExtents{2.1, 0.6, 0.75};
};

// Ask the world what the body is touching. Up to `limit` points; a box on a triangle mesh rarely
// produces more than a dozen and the cap is there so a pathological mesh cannot allocate without
// bound inside a tick.
//
// Fills `ContactManifold::bodies` as well as its points: one entry per distinct prop the box touched,
// read from the world in a single batched call, behind the immovable entry every other contact names.
// A box that touched no prop leaves the table empty and makes no bridge call.
export [[nodiscard]] ContactManifold collideBody(const PhysicsWorld& world, const RigidBodyState& state,
                                                 const CollisionBox& box, const std::uint32_t limit = 32);

// Decide which of the manifold's anchored props are coming loose, **before anything is resolved
// against them**.
//
// Called between `collideBody` and `resolveContacts`, and the ordering is the whole of it. A prop is
// bolted down, so a car that hits one is resolved against an immovable body and loses its momentum
// into the ground — which is right for a bollard and wrong for a wheelie bin, and there is no way to
// tell the two apart after the fact. This asks the question first, from the impulse a *two-body*
// collision would deliver, and marks the ones whose anchor cannot hold it. The solver then does the
// whole job: the bin leaves at about the speed the car arrived at, and the car barely slows.
//
// The load is the two-body approach impulse over the tick, summed across the prop's contact points,
// against `ContactBody::breakForce`; and its moment about the prop's own centre of mass against
// `breakTorque`. Either alone releases it, which is what makes a glancing blow near the top of a
// lighting column snap it at its plate.
//
// **This is the only place a prop is decided to come loose.** What the caller does with `released` is
// carry it to the world; nothing on the far side tests a threshold again, so the two can never
// disagree about a prop sitting exactly on one.
export void releaseBrokenProps(const RigidBodyState& state, ContactManifold& manifold, const double deltaTime);

// Sequential impulse, iterated. The impulses are applied straight to the body's velocity, which is
// why this runs *before* the integrator rather than contributing to the force accumulator: a
// contact is a velocity constraint, and expressing it as a force over a tick is what makes a stiff
// contact either spongy or explosive depending on the timestep.
//
// **Two bodies.** `state` takes `+j` and the manifold's second body takes `−j`, accumulated into
// `ContactBody::deltaLinear` and `deltaAngular` for the caller to hand back to whatever owns it.
// Every second body at infinite mass reduces this to the one-body solver exactly, term for term.
export void resolveContacts(RigidBodyState& state, ContactManifold& manifold, const ContactMaterial& material,
                            const double deltaTime, const std::uint32_t iterations = 8);

} // namespace raceengine
