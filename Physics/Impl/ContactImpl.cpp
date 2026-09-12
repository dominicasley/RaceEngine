// Contact bodies. Declarations are in Api/Contact.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <span>
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

namespace
{

// One box standing in the world, as the separating-axis test wants it. The axes are the **columns**
// of the matrix, which is what `glm::mat3_cast` hands back for an orientation, so nothing here has
// to transpose anything.
struct OrientedBox
{
    glm::dvec3 centre{0.0};
    glm::dmat3 axes{1.0};
    glm::dvec3 half{1.0};
};

// How far the box reaches from its own centre along a unit direction.
[[nodiscard]] double projectedRadius(const OrientedBox& box, const glm::dvec3& axis)
{
    return box.half.x * std::abs(glm::dot(axis, box.axes[0])) + box.half.y * std::abs(glm::dot(axis, box.axes[1])) +
           box.half.z * std::abs(glm::dot(axis, box.axes[2]));
}

// The corner of the box furthest along a direction.
[[nodiscard]] glm::dvec3 supportPoint(const OrientedBox& box, const glm::dvec3& direction)
{
    auto point = box.centre;

    for (auto index = 0; index < 3; index++)
    {
        const auto reach = glm::dot(direction, box.axes[index]) >= 0.0 ? box.half[index] : -box.half[index];

        point += reach * box.axes[index];
    }

    return point;
}

// Which axis two boxes are least apart on, and by how much.
//
// **The face axes are preferred over the edge ones by a bias, and that is not a tidy-up.** Two cars
// meeting nose to tail are very nearly face to face, and the fifteen candidate axes then include
// several within rounding of each other. Picking a cross-product axis on one tick and a face axis on
// the next moves the contact from a four-point patch to a single point and back, which reads as a
// car buzzing against a bumper. A millimetre of bias settles it on the face, which is also the
// answer that produces a manifold rather than a point.
struct BoxSeparation
{
    bool touching = false;
    // Unit, pointing out of the first box and into the second.
    glm::dvec3 axis{0.0, 1.0, 0.0};
    double penetration = 0.0;
    // 0..2 name a face of the first box, 3..5 a face of the second, and 6 is an edge against an edge.
    int kind = 6;
};

[[nodiscard]] BoxSeparation separateBoxes(const OrientedBox& first, const OrientedBox& second)
{
    // A millimetre, in the units everything below Physics works in. Small against any real overlap
    // between two cars and large against the rounding that makes two near-equal axes trade places.
    constexpr auto edgeBias = 0.001;

    const auto between = second.centre - first.centre;

    auto best = BoxSeparation{};
    auto bestScore = -std::numeric_limits<double>::max();
    auto separated = false;

    const auto test = [&](const glm::dvec3& candidate, const int kind)
    {
        if (separated)
        {
            return;
        }

        const auto lengthSquared = glm::dot(candidate, candidate);

        // A cross product of two parallel edges. It is not an axis at all, and the pair of face
        // axes it came from already covers whatever it would have said.
        if (lengthSquared < 1e-12)
        {
            return;
        }

        const auto axis = candidate / std::sqrt(lengthSquared);
        const auto gap =
            std::abs(glm::dot(between, axis)) - projectedRadius(first, axis) - projectedRadius(second, axis);

        if (gap > 0.0)
        {
            separated = true;

            return;
        }

        const auto score = kind == 6 ? gap - edgeBias : gap;
        if (score <= bestScore)
        {
            return;
        }

        bestScore = score;
        best.axis = glm::dot(between, axis) < 0.0 ? -axis : axis;
        best.penetration = -gap;
        best.kind = kind;
    };

    for (auto index = 0; index < 3; index++)
    {
        test(first.axes[index], index);
    }

    for (auto index = 0; index < 3; index++)
    {
        test(second.axes[index], index + 3);
    }

    for (auto own = 0; own < 3; own++)
    {
        for (auto other = 0; other < 3; other++)
        {
            test(glm::cross(first.axes[own], second.axes[other]), 6);
        }
    }

    best.touching = !separated;

    return best;
}

// The four corners of the face of `box` most square-on to `normal` and facing back along it.
void incidentFace(const OrientedBox& box, const glm::dvec3& normal, std::array<glm::dvec3, 4>& into)
{
    auto axisIndex = 0;
    auto strongest = -1.0;

    for (auto index = 0; index < 3; index++)
    {
        if (const auto alignment = std::abs(glm::dot(box.axes[index], normal)); alignment > strongest)
        {
            strongest = alignment;
            axisIndex = index;
        }
    }

    const auto facing = glm::dot(box.axes[axisIndex], normal) > 0.0 ? -1.0 : 1.0;
    const auto centre = box.centre + facing * box.half[axisIndex] * box.axes[axisIndex];

    const auto first = (axisIndex + 1) % 3;
    const auto second = (axisIndex + 2) % 3;
    const auto alongFirst = box.half[first] * box.axes[first];
    const auto alongSecond = box.half[second] * box.axes[second];

    into[0] = centre - alongFirst - alongSecond;
    into[1] = centre + alongFirst - alongSecond;
    into[2] = centre + alongFirst + alongSecond;
    into[3] = centre - alongFirst + alongSecond;
}

// Sutherland-Hodgman against one half space, keeping what is on the inside of the plane. Fixed
// buffers rather than vectors: this runs inside the tick, a quad clipped by four planes never
// exceeds eight vertices, and an allocation per contact per tick is a cost with nothing to show
// for it.
[[nodiscard]] std::size_t clipToPlane(const std::array<glm::dvec3, 8>& input, const std::size_t count,
                                      const glm::dvec3& planePoint, const glm::dvec3& planeNormal,
                                      std::array<glm::dvec3, 8>& output)
{
    auto kept = std::size_t{0};

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto& current = input[index];
        const auto& previous = input[(index + count - 1) % count];

        const auto currentSide = glm::dot(current - planePoint, planeNormal);
        const auto previousSide = glm::dot(previous - planePoint, planeNormal);

        if (currentSide * previousSide < 0.0 && kept < output.size())
        {
            // The edge crosses the plane, so the crossing point joins the polygon before whichever
            // of its two ends is inside.
            const auto span = previousSide - currentSide;
            const auto along = std::abs(span) > 1e-12 ? previousSide / span : 0.0;

            output[kept] = previous + (current - previous) * along;
            kept++;
        }

        if (currentSide <= 0.0 && kept < output.size())
        {
            output[kept] = current;
            kept++;
        }
    }

    return kept;
}

// The contact patch between two oriented boxes, with the normal pointing out of `first` and into
// `second` — the direction `ContactPoint::normal` means.
//
// A face clip rather than a single deepest point, and that is what makes a car rest against another
// car instead of pivoting on one corner: a sequential-impulse solver needs several points on the
// same plane before it can hold an orientation at all.
[[nodiscard]] std::size_t collideOrientedBoxes(const OrientedBox& first, const OrientedBox& second,
                                               std::array<glm::dvec3, 4>& positions,
                                               std::array<double, 4>& penetrations, glm::dvec3& normal)
{
    const auto separation = separateBoxes(first, second);
    if (!separation.touching)
    {
        return 0;
    }

    normal = separation.axis;

    if (separation.kind == 6)
    {
        // Edge against edge: a corner or a wing mirror's worth of contact, and there is one point in
        // it. Midway between the two support points, which is where the two edges cross.
        positions[0] = 0.5 * (supportPoint(first, normal) + supportPoint(second, -normal));
        penetrations[0] = separation.penetration;

        return 1;
    }

    const auto referenceIsFirst = separation.kind < 3;
    const auto& reference = referenceIsFirst ? first : second;
    const auto& incident = referenceIsFirst ? second : first;

    // Outward from the reference box, pointing at the incident one. The separating axis already *is*
    // that face's normal, so the face centre is one half extent along it and no sign has to be
    // recovered.
    const auto referenceNormal = referenceIsFirst ? normal : -normal;
    const auto axisIndex = referenceIsFirst ? separation.kind : separation.kind - 3;
    const auto faceCentre = reference.centre + reference.half[axisIndex] * referenceNormal;

    auto corners = std::array<glm::dvec3, 4>{};
    incidentFace(incident, referenceNormal, corners);

    auto polygon = std::array<glm::dvec3, 8>{};
    auto scratch = std::array<glm::dvec3, 8>{};
    for (auto index = std::size_t{0}; index < corners.size(); index++)
    {
        polygon[index] = corners[index];
    }

    auto count = corners.size();

    const auto firstSide = (axisIndex + 1) % 3;
    const auto secondSide = (axisIndex + 2) % 3;

    const auto clip = [&](const int side)
    {
        const auto edge = reference.half[side] * reference.axes[side];

        count = clipToPlane(polygon, count, faceCentre + edge, reference.axes[side], scratch);
        polygon = scratch;

        count = clipToPlane(polygon, count, faceCentre - edge, -reference.axes[side], scratch);
        polygon = scratch;
    };

    clip(firstSide);
    clip(secondSide);

    auto found = std::size_t{0};

    for (auto index = std::size_t{0}; index < count && found < positions.size(); index++)
    {
        const auto depth = -glm::dot(polygon[index] - faceCentre, referenceNormal);
        if (depth < 0.0)
        {
            // Clipped into the face's footprint but not behind it. It is not touching.
            continue;
        }

        positions[found] = polygon[index];
        penetrations[found] = depth;
        found++;
    }

    if (found == 0)
    {
        // The clip left nothing — a grazing pass the axis test called an overlap. One point at the
        // deepest corner rather than nothing, so a contact the test found is a contact the solver
        // sees.
        positions[0] = supportPoint(incident, -referenceNormal);
        penetrations[0] = separation.penetration;

        return 1;
    }

    return found;
}

} // namespace

void collideObstacles(const RigidBodyState& state, const CollisionBox& box,
                      std::span<const DynamicObstacle> obstacles, ContactManifold& manifold,
                      const std::uint32_t limitPerObstacle)
{
    // **The whole of the byte-inertness argument.** A world with no traffic in it hands an empty
    // span, this returns before touching the manifold, and what the solver then resolves is exactly
    // what `collideBody` produced — the same points, the same table, the same arithmetic. Both frame
    // gates are blessed on a circuit that has no traffic lanes at all.
    if (obstacles.empty() || limitPerObstacle == 0)
    {
        return;
    }

    auto car = OrientedBox{};
    car.centre = bodyToWorld(state, box.centre);
    car.axes = glm::mat3_cast(state.orientation);
    car.half = box.halfExtents;

    // A bounding-sphere rejection ahead of the fifteen axes. Most obstacles handed in on any tick
    // are the ones a car is driving past rather than into, and this is the test that says so in
    // three multiplies.
    const auto carReach = glm::length(car.half);

    for (const auto& obstacle : obstacles)
    {
        auto solid = OrientedBox{};
        solid.centre = obstacle.centre;
        solid.axes = glm::mat3_cast(obstacle.orientation);
        solid.half = obstacle.halfExtents;

        const auto between = car.centre - solid.centre;
        const auto reach = carReach + glm::length(solid.half);

        if (glm::dot(between, between) > reach * reach)
        {
            continue;
        }

        auto positions = std::array<glm::dvec3, 4>{};
        auto penetrations = std::array<double, 4>{};
        auto normal = glm::dvec3(0.0, 1.0, 0.0);

        const auto found = collideOrientedBoxes(solid, car, positions, penetrations, normal);
        if (found == 0)
        {
            continue;
        }

        // The immovable world, created on first need — the same rule `collideBody` keeps, and it has
        // to be kept here too because a car that touches an obstacle and nothing else arrives with
        // an empty table.
        if (manifold.bodies.empty())
        {
            manifold.bodies.push_back(ContactBody{});
        }

        // One body per obstacle, appended rather than looked up. `bodyIndexFor` matches on `prop`,
        // and every obstacle carries `noProp` — so a lookup would hand every one of them the
        // immovable world's own entry.
        const auto body = manifold.bodies.size();

        manifold.bodies.push_back(ContactBody{.prop = noProp,
                                              .obstacle = obstacle.id,
                                              .inverseMass = obstacle.inverseMass,
                                              .inverseInertia = obstacle.inverseInertia,
                                              .centre = obstacle.centreOfMass,
                                              .linearVelocity = obstacle.linearVelocity,
                                              .angularVelocity = obstacle.angularVelocity});

        const auto kept = std::min(static_cast<std::size_t>(limitPerObstacle), found);

        // Deepest first when the cap bites, so what is dropped is the shallowest corner rather than
        // whichever one the clip happened to emit last.
        for (auto slot = std::size_t{0}; slot < kept; slot++)
        {
            auto deepest = slot;
            for (auto index = slot + 1; index < found; index++)
            {
                if (penetrations[index] > penetrations[deepest])
                {
                    deepest = index;
                }
            }

            std::swap(positions[slot], positions[deepest]);
            std::swap(penetrations[slot], penetrations[deepest]);

            manifold.points.push_back(ContactPoint{.position = positions[slot],
                                                   .normal = normal,
                                                   .penetration = penetrations[slot],
                                                   .prop = noProp,
                                                   .body = body});
        }
    }
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
        auto& point = manifold.points[index];
        const auto closing = glm::dot(velocityAt(point.position, bodyOf(point.body)), point.normal);

        restitutionTargets[index] = closing < -material.restitutionThreshold ? -material.restitution * closing : 0.0;

        // The hit as it arrived, for whoever keeps a damage ledger: the same number restitution
        // reads, kept on the point because nothing after this line can recover it.
        point.approachSpeed = std::max(0.0, -closing);
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
