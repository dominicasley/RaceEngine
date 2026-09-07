// Contact bodies. Declarations are in Api/Contact.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>
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
                                                    double* outDepths, std::uint32_t* outProps);
extern "C++" void raceengineJoltReadPropDynamics(std::uint64_t handle, const std::uint32_t* propIndices,
                                                 std::uint32_t count, double* outInverseMasses,
                                                 double* outInverseInertias, double* outCentres,
                                                 double* outLinearVelocities, double* outAngularVelocities,
                                                 double* outBreakForces, double* outBreakTorques,
                                                 unsigned char* outAnchored);

namespace raceengine
{

namespace
{

// The velocity of a point on the first body relative to the same point on the second.
//
// Shared by the solver and by the anchor test above it, so that the load an anchor is judged on and
// the load the solver then applies are the same quantity read the same way. Against an immovable
// second body the subtracted term is exactly zero, which is what makes a contact with no prop in it
// bit-identical to the one-body solver this replaced.
[[nodiscard]] glm::dvec3 relativeVelocityAt(const RigidBodyState& state, const glm::dvec3& point,
                                            const ContactBody& other)
{
    const auto own = state.linearVelocity + glm::cross(angularVelocity(state), point - state.position);
    const auto theirs = other.linearVelocity + glm::cross(other.angularVelocity, point - other.centre);

    return own - theirs;
}

// The effective mass along a direction at a point: how much impulse it takes to change the relative
// velocity there by one.
//
// The second body's two terms are **added after** the first body's, not folded in beside them, and
// that ordering is load-bearing: `(a + b) + 0.0 + 0.0` is exactly `a + b`, and a regrouping that was
// mathematically identical would not be.
[[nodiscard]] double pairEffectiveMass(const RigidBodyState& state, const double inverseMass,
                                       const glm::dmat3& inverseInertia, const glm::dvec3& point,
                                       const glm::dvec3& direction, const double otherInverseMass,
                                       const glm::dmat3& otherInverseInertia, const glm::dvec3& otherCentre)
{
    const auto arm = point - state.position;
    const auto angular = glm::cross(inverseInertia * glm::cross(arm, direction), arm);

    const auto otherArm = point - otherCentre;
    const auto otherAngular = glm::cross(otherInverseInertia * glm::cross(otherArm, direction), otherArm);

    return 1.0 /
           std::max(inverseMass + glm::dot(direction, angular) + otherInverseMass + glm::dot(direction, otherAngular),
                    1e-12);
}

// Where in the manifold's body table this prop lives, appending an entry if it is the first point to
// name it.
//
// A linear scan, and it is the whole lookup: a manifold caps at 32 points and the distinct props
// among them are a handful, so a map would be an allocation and a tree walk per contact point to
// save nothing. It is also what the house rule about `<map>` in a second global module fragment
// leaves — that include breaks the sandbox *link*, out of a translation unit that never mentions one.
[[nodiscard]] std::size_t bodyIndexFor(std::vector<ContactBody>& bodies, const std::uint32_t prop)
{
    for (auto index = std::size_t{0}; index < bodies.size(); index++)
    {
        if (bodies[index].prop == prop)
        {
            return index;
        }
    }

    bodies.push_back(ContactBody{.prop = prop});

    return bodies.size() - 1;
}

// The dynamics of every prop the manifold named, in one call.
//
// Batched for the reason `castRays` is: the alternative is a body lock and a broadphase-free lookup
// per contact point, and the props a car's bodywork touches in one tick arrive together.
//
// **A prop that is still anchored is presented to the solver as immovable and carries its real mass
// beside that**, in the `free` fields. That split is deliberate and the direction of it matters: the
// safe default is the old behaviour, so a manifold nobody runs `releaseBrokenProps` over resolves
// every prop as a building — visibly wrong in the same way stage 1 was, rather than silently letting
// a car through a bollard.
void readPropDynamics(const PhysicsWorld& world, ContactManifold& manifold)
{
    // Element 0 is the immovable world and is never a prop, so a manifold that touched nothing but
    // the track makes no call at all.
    if (manifold.bodies.size() <= 1)
    {
        return;
    }

    const auto count = manifold.bodies.size() - 1;

    auto indices = std::vector<std::uint32_t>(count);
    auto inverseMasses = std::vector<double>(count);
    auto inverseInertias = std::vector<double>(count * 9);
    auto centres = std::vector<double>(count * 3);
    auto linearVelocities = std::vector<double>(count * 3);
    auto angularVelocities = std::vector<double>(count * 3);
    auto breakForces = std::vector<double>(count);
    auto breakTorques = std::vector<double>(count);
    auto anchored = std::vector<unsigned char>(count);

    for (auto index = std::size_t{0}; index < count; index++)
    {
        indices[index] = manifold.bodies[index + 1].prop;
    }

    raceengineJoltReadPropDynamics(world.handle(), indices.data(), static_cast<std::uint32_t>(count),
                                   inverseMasses.data(), inverseInertias.data(), centres.data(),
                                   linearVelocities.data(), angularVelocities.data(), breakForces.data(),
                                   breakTorques.data(), anchored.data());

    for (auto index = std::size_t{0}; index < count; index++)
    {
        auto& body = manifold.bodies[index + 1];

        body.centre = glm::dvec3(centres[index * 3], centres[index * 3 + 1], centres[index * 3 + 2]);
        body.linearVelocity =
            glm::dvec3(linearVelocities[index * 3], linearVelocities[index * 3 + 1], linearVelocities[index * 3 + 2]);
        body.angularVelocity = glm::dvec3(angularVelocities[index * 3], angularVelocities[index * 3 + 1],
                                          angularVelocities[index * 3 + 2]);

        // Row major across the bridge, and glm indexes a matrix by column first, so this transposes
        // on the way in — the same convention `PhysicsWorld::create` writes a prop's inertia out in.
        const auto* stated = inverseInertias.data() + index * 9;
        for (auto row = 0; row < 3; row++)
        {
            for (auto column = 0; column < 3; column++)
            {
                body.freeInverseInertia[column][row] = stated[row * 3 + column];
            }
        }

        body.freeInverseMass = inverseMasses[index];
        body.anchored = anchored[index] != 0;
        body.breakForce = breakForces[index];
        body.breakTorque = breakTorques[index];

        // An anchored prop stays immovable until something says otherwise; a loose one is already a
        // body and takes its dynamics straight away.
        if (!body.anchored)
        {
            body.inverseMass = body.freeInverseMass;
            body.inverseInertia = body.freeInverseInertia;
        }
    }
}

} // namespace

TangentBasis tangentBasis(const glm::dvec3& normal)
{
    const auto magnitude = glm::length(normal);
    if (magnitude < 1e-12)
    {
        return TangentBasis{};
    }

    const auto unit = normal / magnitude;
    const auto absolute = glm::abs(unit);

    // The least-aligned world axis, which is what keeps the cross product below away from
    // degenerate: against a unit normal its length is never less than sqrt(2/3), so `normalize` has
    // no case to guard. `<=` throughout, so a tie resolves the same way on every machine rather than
    // on whichever comparison happens to be written first.
    const auto axis = absolute.x <= absolute.y
                          ? (absolute.x <= absolute.z ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0))
                          : (absolute.y <= absolute.z ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0));

    const auto first = glm::normalize(glm::cross(unit, axis));

    return TangentBasis{.first = first, .second = glm::cross(unit, first)};
}

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
    auto props = std::vector<std::uint32_t>(limit);

    const auto found = raceengineJoltCollideBox(world.handle(), &box.halfExtents.x, &centre.x, &orientation.x, limit,
                                                points.data(), normals.data(), depths.data(), props.data());

    manifold.points.reserve(found);

    for (auto index = std::uint32_t{0}; index < found; index++)
    {
        auto point =
            ContactPoint{.position = glm::dvec3(points[index * 3], points[index * 3 + 1], points[index * 3 + 2]),
                         .normal = glm::dvec3(normals[index * 3], normals[index * 3 + 1], normals[index * 3 + 2]),
                         .penetration = depths[index],
                         .prop = props[index]};

        if (point.prop != noProp)
        {
            // The immovable world, created on first need. A track with no street furniture on it
            // therefore allocates nothing here at all — which is every tick of a circuit, and is what
            // keeps this call exactly the call it was before a second body existed.
            if (manifold.bodies.empty())
            {
                manifold.bodies.push_back(ContactBody{});
            }

            point.body = bodyIndexFor(manifold.bodies, point.prop);
        }

        manifold.points.push_back(point);
    }

    readPropDynamics(world, manifold);

    return manifold;
}

void releaseBrokenProps(const RigidBodyState& state, ContactManifold& manifold, const double deltaTime)
{
    if (manifold.bodies.size() <= 1 || deltaTime <= 0.0)
    {
        return;
    }

    const auto inverseMass = state.mass > 0.0 ? 1.0 / state.mass : 0.0;
    const auto inverseInertia = worldInverseInertia(state);

    // The load each anchor is carrying, accumulated across every point that landed on it. A flat hit
    // lands on several at once and each alone is under the threshold, which is the same reason
    // `applyPropImpulses` accumulates before it tests.
    auto loads = std::vector<glm::dvec3>(manifold.bodies.size(), glm::dvec3(0.0));
    auto moments = std::vector<glm::dvec3>(manifold.bodies.size(), glm::dvec3(0.0));

    for (const auto& point : manifold.points)
    {
        if (point.body >= manifold.bodies.size())
        {
            continue;
        }

        const auto& body = manifold.bodies[point.body];
        if (!body.anchored)
        {
            continue;
        }

        const auto closing = glm::dot(relativeVelocityAt(state, point.position, body), point.normal);
        if (closing >= 0.0)
        {
            // Separating, so the anchor is carrying nothing at this point. A contact that is coming
            // apart must not add to a load, or a car pulling away from a post would break it.
            continue;
        }

        // **The impulse a two-body collision would deliver, and not the one the solver is about to
        // compute.** The solver is going to resolve this against an immovable body, so its impulse
        // is sized by the car's mass alone — enough to stop 1400 kg, whatever the prop weighs. That
        // is the number that used to be handed to a twenty-kilogram bin, and it is why the bins flew.
        // What an anchor actually carries is the collision the pair would have had, which is this.
        const auto magnitude =
            -closing * pairEffectiveMass(state, inverseMass, inverseInertia, point.position, point.normal,
                                         body.freeInverseMass, body.freeInverseInertia, body.centre);

        loads[point.body] += magnitude * point.normal;
        moments[point.body] += glm::cross(point.position - body.centre, magnitude * point.normal);
    }

    for (auto index = std::size_t{1}; index < manifold.bodies.size(); index++)
    {
        auto& body = manifold.bodies[index];
        if (!body.anchored)
        {
            continue;
        }

        // An impulse over the tick is a force, and a moment over the tick is a torque, which is what
        // the exporter's two thresholds are quoted in. **That the tick appears here at all is a
        // known weakness**: halve the timestep and the same collision reads as twice the force. What
        // an anchor wants is a stated contact time rather than a simulation step, and that is a
        // separate question with a separate answer.
        const auto force = glm::length(loads[index]) / deltaTime;
        const auto torque = glm::length(moments[index]) / deltaTime;

        if (force < body.breakForce && torque < body.breakTorque)
        {
            continue;
        }

        body.released = true;
        body.anchored = false;
        body.inverseMass = body.freeInverseMass;
        body.inverseInertia = body.freeInverseInertia;
    }
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

    // What a point resolves against when its manifold carries no body table at all — a hand-built
    // manifold, and every caller that existed before a second body did. Held here rather than
    // refused, so that "no table" and "a table whose entry is immovable" are the same arithmetic
    // rather than two paths that have to be kept in step.
    auto immovable = ContactBody{};

    const auto bodyOf = [&manifold, &immovable](const std::size_t index) -> ContactBody&
    {
        return index < manifold.bodies.size() ? manifold.bodies[index] : immovable;
    };

    // The **relative** velocity at the point: the car's minus the second body's. Against an
    // immovable one the second term is exactly zero and the subtraction is exact, which is the whole
    // of why a track with no props resolves bit-identically to what shipped before this.
    const auto velocityAt = [&state](const glm::dvec3& point, const ContactBody& other)
    {
        return relativeVelocityAt(state, point, other);
    };

    const auto applyImpulse =
        [&state, inverseMass](const glm::dvec3& impulse, const glm::dvec3& point, ContactBody& other)
    {
        state.linearVelocity += impulse * inverseMass;
        state.angularMomentum += glm::cross(point - state.position, impulse);

        // And the equal and opposite half, onto the body's **local** velocity rather than through to
        // whatever owns it. Seven more iterations are coming and each of them reads this velocity
        // back; one that went to the backend instead would be solving against a body it had already
        // pushed. The accumulated delta beside it is what the owner is handed at the end.
        const auto linear = impulse * other.inverseMass;
        const auto angular = other.inverseInertia * glm::cross(point - other.centre, impulse);

        other.linearVelocity -= linear;
        other.angularVelocity -= angular;
        other.deltaLinear -= linear;
        other.deltaAngular -= angular;
    };

    // The effective mass along a direction at a point: how much impulse it takes to change the
    // relative velocity there by one. Recomputed per point per direction, and shared with the anchor
    // test above so that what an anchor is judged on and what the solver then applies are the same
    // quantity computed the same way.
    const auto effectiveMass = [&state, inverseMass, &inverseInertia](
                                   const glm::dvec3& point, const glm::dvec3& direction, const ContactBody& other)
    {
        return pairEffectiveMass(state, inverseMass, inverseInertia, point, direction, other.inverseMass,
                                 other.inverseInertia, other.centre);
    };

    // The tangent axes, once per contact and from the normal alone. Fixed for the whole solve, which
    // is the point of them: the accumulators below are quoted along these and a basis that moved
    // between iterations would be adding numbers that mean different things.
    for (auto& point : manifold.points)
    {
        const auto basis = tangentBasis(point.normal);

        point.tangent1 = basis.first;
        point.tangent2 = basis.second;
    }

    // Restitution is captured from the *approach* velocity, once, before any impulse is applied.
    // Reading it inside the loop would have each iteration bouncing off the velocity the previous
    // one produced.
    auto restitutionTargets = std::vector<double>(manifold.points.size(), 0.0);
    for (auto index = std::size_t{0}; index < manifold.points.size(); index++)
    {
        const auto& point = manifold.points[index];
        const auto closing = glm::dot(velocityAt(point.position, bodyOf(point.body)), point.normal);

        restitutionTargets[index] = closing < -material.restitutionThreshold ? -material.restitution * closing : 0.0;
    }

    for (auto iteration = std::uint32_t{0}; iteration < iterations; iteration++)
    {
        for (auto index = std::size_t{0}; index < manifold.points.size(); index++)
        {
            auto& point = manifold.points[index];
            auto& other = bodyOf(point.body);

            // Normal. The bias pushes out the penetration beyond the slop, spread over a second.
            const auto bias =
                material.correction * std::max(0.0, point.penetration - material.allowedPenetration) / deltaTime;

            const auto separating = glm::dot(velocityAt(point.position, other), point.normal);
            const auto wanted = restitutionTargets[index] + bias - separating;

            auto increment = wanted * effectiveMass(point.position, point.normal, other);

            // Clamp the *accumulation*, not the increment. A contact may only push; but an
            // iteration may reduce what an earlier one applied, which is exactly what lets the
            // points on a face agree with each other instead of taking turns.
            const auto before = point.normalImpulse;
            point.normalImpulse = std::max(0.0, before + increment);
            increment = point.normalImpulse - before;

            applyImpulse(increment * point.normal, point.position, other);
        }

        for (auto& point : manifold.points)
        {
            auto& other = bodyOf(point.body);

            // Coulomb friction, solved along the two fixed axes and clamped **as a pair** against
            // the cone the normal impulse allows.
            //
            // Clamping each axis on its own is the box approximation most engines ship, and it lets
            // a contact sliding at 45 degrees to the basis take `sqrt(2) * mu * jn` — which is a
            // friction coefficient 40% higher on some contacts than on others, decided by nothing
            // more physical than which way a face happened to point. The pair clamp is barely more
            // arithmetic and is the reason this is the correct fix rather than a different
            // approximation.
            const auto relative = velocityAt(point.position, other);

            const auto along =
                -glm::dot(relative, point.tangent1) * effectiveMass(point.position, point.tangent1, other);
            const auto across =
                -glm::dot(relative, point.tangent2) * effectiveMass(point.position, point.tangent2, other);

            auto first = point.tangentImpulse1 + along;
            auto second = point.tangentImpulse2 + across;

            const auto limit = material.friction * point.normalImpulse;
            const auto magnitude = std::sqrt(first * first + second * second);
            if (magnitude > limit)
            {
                const auto scale = limit / magnitude;

                first *= scale;
                second *= scale;
            }

            const auto increment =
                (first - point.tangentImpulse1) * point.tangent1 + (second - point.tangentImpulse2) * point.tangent2;

            point.tangentImpulse1 = first;
            point.tangentImpulse2 = second;

            applyImpulse(increment, point.position, other);
        }
    }
}

} // namespace raceengine
