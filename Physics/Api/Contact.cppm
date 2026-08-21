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
                                                 const CollisionBox& box, const std::uint32_t limit = 32)
{
    auto manifold = ContactManifold{};

    const auto centre = bodyToWorld(state, box.centre);
    const auto orientation =
        glm::dvec4(state.orientation.w, state.orientation.x, state.orientation.y, state.orientation.z);

    auto points = std::vector<double>(limit * 3);
    auto normals = std::vector<double>(limit * 3);
    auto depths = std::vector<double>(limit);

    const auto found = raceengineJoltCollideBox(world.handle(), &box.halfExtents.x, &centre.x, &orientation.x, limit,
                                                points.data(), normals.data(), depths.data());

    manifold.points.reserve(found);
    for (auto index = std::uint32_t{0}; index < found; index++)
    {
        manifold.points.push_back(
            ContactPoint{.position = glm::dvec3(points[index * 3], points[index * 3 + 1], points[index * 3 + 2]),
                         .normal = glm::dvec3(normals[index * 3], normals[index * 3 + 1], normals[index * 3 + 2]),
                         .penetration = depths[index]});
    }

    return manifold;
}

// Sequential impulse, iterated. The impulses are applied straight to the body's velocity, which is
// why this runs *before* the integrator rather than contributing to the force accumulator: a
// contact is a velocity constraint, and expressing it as a force over a tick is what makes a stiff
// contact either spongy or explosive depending on the timestep.
export void resolveContacts(RigidBodyState& state, ContactManifold& manifold, const ContactMaterial& material,
                            const double deltaTime, const std::uint32_t iterations = 8)
{
    if (manifold.points.empty() || deltaTime <= 0.0)
    {
        return;
    }

    const auto inverseMass = state.mass > 0.0 ? 1.0 / state.mass : 0.0;
    const auto inverseInertia = worldInverseInertia(state);

    const auto velocityAt = [&state](const glm::dvec3& point)
    {
        return state.linearVelocity + glm::cross(angularVelocity(state), point - state.position);
    };

    const auto applyImpulse = [&state, inverseMass](const glm::dvec3& impulse, const glm::dvec3& point)
    {
        state.linearVelocity += impulse * inverseMass;
        state.angularMomentum += glm::cross(point - state.position, impulse);
    };

    // The effective mass along a direction at a point: how much impulse it takes to change the
    // relative velocity there by one. Precomputed per point per direction because it does not change
    // across the iterations and it is the expensive part.
    const auto effectiveMass =
        [&state, inverseMass, &inverseInertia](const glm::dvec3& point, const glm::dvec3& direction)
    {
        const auto arm = point - state.position;
        const auto angular = glm::cross(inverseInertia * glm::cross(arm, direction), arm);

        return 1.0 / std::max(inverseMass + glm::dot(direction, angular), 1e-12);
    };

    // Restitution is captured from the *approach* velocity, once, before any impulse is applied.
    // Reading it inside the loop would have each iteration bouncing off the velocity the previous
    // one produced.
    auto restitutionTargets = std::vector<double>(manifold.points.size(), 0.0);
    for (auto index = std::size_t{0}; index < manifold.points.size(); index++)
    {
        const auto& point = manifold.points[index];
        const auto closing = glm::dot(velocityAt(point.position), point.normal);

        restitutionTargets[index] = closing < -material.restitutionThreshold ? -material.restitution * closing : 0.0;
    }

    for (auto iteration = std::uint32_t{0}; iteration < iterations; iteration++)
    {
        for (auto index = std::size_t{0}; index < manifold.points.size(); index++)
        {
            auto& point = manifold.points[index];

            // Normal. The bias pushes out the penetration beyond the slop, spread over a second.
            const auto bias =
                material.correction * std::max(0.0, point.penetration - material.allowedPenetration) / deltaTime;

            const auto separating = glm::dot(velocityAt(point.position), point.normal);
            const auto wanted = restitutionTargets[index] + bias - separating;

            auto increment = wanted * effectiveMass(point.position, point.normal);

            // Clamp the *accumulation*, not the increment. A contact may only push; but an
            // iteration may reduce what an earlier one applied, which is exactly what lets the
            // points on a face agree with each other instead of taking turns.
            const auto before = point.normalImpulse;
            point.normalImpulse = std::max(0.0, before + increment);
            increment = point.normalImpulse - before;

            applyImpulse(increment * point.normal, point.position);
        }

        for (auto& point : manifold.points)
        {
            // Coulomb friction, in the tangent plane, clamped to what the normal impulse allows.
            const auto relative = velocityAt(point.position);
            auto tangent = relative - glm::dot(relative, point.normal) * point.normal;

            const auto speed = glm::length(tangent);
            if (speed < 1e-9)
            {
                continue;
            }

            tangent /= speed;

            auto increment = -speed * effectiveMass(point.position, tangent);
            const auto limit = material.friction * point.normalImpulse;

            const auto before = point.tangentImpulse;
            point.tangentImpulse = std::clamp(before + increment, -limit, limit);
            increment = point.tangentImpulse - before;

            applyImpulse(increment * tangent, point.position);
        }
    }
}

} // namespace raceengine
