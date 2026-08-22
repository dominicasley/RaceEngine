// Contact bodies. Declarations are in Api/Contact.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

module raceengine.physics;

// The bridge declarations this file calls through. Repeated here rather than imported: they sit in
// the partition's purview and are not exported, so an implementation unit of the same module cannot
// see them through the primary interface. Declaring them here puts them in the same module, which is
// what makes the symbol the one Physics/Backend/JoltBackend.cpp defines.
extern "C++" std::uint32_t raceengineJoltCollideBox(std::uint64_t handle, const double* halfExtents,
                                                    const double* centre, const double* orientation,
                                                    std::uint32_t maxContacts, double* outPoints, double* outNormals,
                                                    double* outDepths);

namespace raceengine
{

[[nodiscard]] ContactManifold collideBody(const PhysicsWorld& world, const RigidBodyState& state,
                                          const CollisionBox& box, const std::uint32_t limit)
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

void resolveContacts(RigidBodyState& state, ContactManifold& manifold, const ContactMaterial& material,
                     const double deltaTime, const std::uint32_t iterations)
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
