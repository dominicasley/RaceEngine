module;

#include <algorithm>
#include <cmath>
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
                                                    double* outDepths);

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

export struct ContactPoint
{
    glm::dvec3 position{0.0};
    // Points out of the world and into the body — the direction the body has to move to separate.
    glm::dvec3 normal{0.0, 1.0, 0.0};
    double penetration = 0.0;

    // Accumulated impulses, carried across the solver's iterations. Keeping them per point and
    // clamping the *accumulation* rather than each increment is what makes sequential impulse
    // converge instead of oscillating: an iteration is allowed to take back what a previous one
    // overshot, so long as the total never pulls.
    double normalImpulse = 0.0;
    double tangentImpulse = 0.0;
};

export struct ContactManifold
{
    std::vector<ContactPoint> points;
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
export [[nodiscard]] ContactManifold collideBody(const PhysicsWorld& world, const RigidBodyState& state,
                                                 const CollisionBox& box, const std::uint32_t limit = 32);

// Sequential impulse, iterated. The impulses are applied straight to the body's velocity, which is
// why this runs *before* the integrator rather than contributing to the force accumulator: a
// contact is a velocity constraint, and expressing it as a force over a tick is what makes a stiff
// contact either spongy or explosive depending on the timestep.
export void resolveContacts(RigidBodyState& state, ContactManifold& manifold, const ContactMaterial& material,
                            const double deltaTime, const std::uint32_t iterations = 8);

} // namespace raceengine
