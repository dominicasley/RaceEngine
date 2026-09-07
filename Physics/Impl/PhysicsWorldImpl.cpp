// PhysicsWorld bodies. Declarations are in Api/PhysicsWorld.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces
// an object and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition
// left in the interface partition is part of the module's BMI instead, and editing one rebuilt
// every importer of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.physics;

// The bridge declarations these bodies call through. Repeated here rather than imported: they sit
// in the partition's purview and are not exported, so an implementation unit of the same module
// cannot see them through the primary interface. Declaring them here puts them in the same module,
// which is what makes the symbol the one Physics/Backend/JoltBackend.cpp defines.
extern "C++" std::uint64_t raceengineJoltCreateWorld(const double* vertexData, std::uint32_t vertexCount,
                                                     const std::uint32_t* indexData, std::uint32_t triangleCount,
                                                     const std::uint32_t* surfaceData, std::uint32_t surfaceCount,
                                                     const double* hullPointData, const std::uint32_t* hullPointCounts,
                                                     const double* hullOrigins, const std::uint32_t* hullSurfaces,
                                                     std::uint32_t hullCount, std::uint32_t propBudget,
                                                     std::string& reason);
extern "C++" std::uint32_t raceengineJoltAddProps(std::uint64_t handle, const std::uint32_t* propHullCounts,
                                                  const std::uint32_t* hullPointCounts, const double* hullPointData,
                                                  const double* origins, const double* masses, const double* inertias,
                                                  const double* breakForces, const double* breakTorques,
                                                  std::uint32_t propCount, std::uint32_t surface, std::string& reason);
extern "C++" void raceengineJoltFinish(std::uint64_t handle);
extern "C++" void raceengineJoltBreakProps(std::uint64_t handle, const std::uint32_t* propIndices,
                                           const double* impulses, const double* points, std::uint32_t count,
                                           double deltaTime);
extern "C++" void raceengineJoltReleaseProps(std::uint64_t handle, const std::uint32_t* propIndices,
                                             std::uint32_t count);
extern "C++" void raceengineJoltAddPropVelocities(std::uint64_t handle, const std::uint32_t* propIndices,
                                                  const double* linear, const double* angular, std::uint32_t count);
extern "C++" void raceengineJoltStepWorld(std::uint64_t handle, double deltaTime);
extern "C++" std::uint32_t raceengineJoltReadFreeProps(std::uint64_t handle, std::uint32_t limit,
                                                       std::uint32_t* outProps, double* outPositions,
                                                       double* outOrientations);
extern "C++" void raceengineJoltDestroyWorld(std::uint64_t handle);
extern "C++" void raceengineJoltCastRays(std::uint64_t handle, const double* origins, const double* directions,
                                         double maxDistance, std::uint32_t count, double* outPoints, double* outNormals,
                                         double* outDistances, std::uint32_t* outSurfaces, unsigned char* outHits);

namespace raceengine
{

PhysicsWorld::PhysicsWorld(PhysicsWorld&& other) noexcept :
    world(std::exchange(other.world, 0)),
    surfaceMaterials(std::move(other.surfaceMaterials))
{
}

PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&& other) noexcept
{
    if (this != &other)
    {
        raceengineJoltDestroyWorld(world);
        world = std::exchange(other.world, 0);
        surfaceMaterials = std::move(other.surfaceMaterials);
    }

    return *this;
}

PhysicsWorld::~PhysicsWorld()
{
    raceengineJoltDestroyWorld(world);
}

std::expected<PhysicsWorld, std::string> PhysicsWorld::create(const SurfaceMesh& mesh,
                                                              const std::span<const ConvexCollider> colliders,
                                                              const std::span<const BreakableProp> props)
{
    // Reported here rather than left to the bridge, because the bridge's own answer to an empty
    // table is "triangle 0 names surface 0, of which there are 0", which reads as a bad index
    // rather than as a missing table. The table is also what `materials()` hands the tire, and a
    // patch aggregated against an empty one has no first element to fall back on.
    if (mesh.materials.empty())
    {
        return std::unexpected("a collision mesh needs at least one surface material");
    }

    auto reason = std::string();

    // The colliders, flattened into the shape the bridge takes. Nothing crosses that boundary but
    // fundamental types and arrays of them — a struct declared in a module unit is attached to that
    // module and is a different type from an identically written one in the plain TU — so the point
    // clouds are concatenated into one array with a count per hull beside them.
    //
    // Built even when the span is empty, because an empty vector's `data()` is a well-defined null
    // and the bridge's loop runs zero times against it. A track with no colliders therefore takes
    // exactly the path it took before this existed.
    auto hullPoints = std::vector<double>();
    auto hullPointCounts = std::vector<std::uint32_t>();
    auto hullOrigins = std::vector<double>();
    auto hullSurfaces = std::vector<std::uint32_t>();

    hullPointCounts.reserve(colliders.size());
    hullOrigins.reserve(colliders.size() * 3);
    hullSurfaces.reserve(colliders.size());

    for (const auto& collider : colliders)
    {
        if (collider.surface >= mesh.materials.size())
        {
            return std::unexpected("a collider names surface " + std::to_string(collider.surface) +
                                   ", of which the mesh states " + std::to_string(mesh.materials.size()));
        }

        hullPointCounts.push_back(static_cast<std::uint32_t>(collider.points.size()));
        hullSurfaces.push_back(collider.surface);
        hullOrigins.insert(hullOrigins.end(), {collider.origin.x, collider.origin.y, collider.origin.z});

        for (const auto& point : collider.points)
        {
            hullPoints.insert(hullPoints.end(), {point.x, point.y, point.z});
        }
    }

    // glm::dvec3 is three doubles with no padding, which the static_assert in the interface is what
    // makes safe to rely on rather than assume — the whole vertex array crosses as one pointer.
    const auto created = raceengineJoltCreateWorld(
        &mesh.vertices.front().x, static_cast<std::uint32_t>(mesh.vertices.size()), mesh.indices.data(),
        static_cast<std::uint32_t>(mesh.triangleCount()), mesh.surfaces.data(),
        static_cast<std::uint32_t>(mesh.materials.size()), hullPoints.data(), hullPointCounts.data(),
        hullOrigins.data(), hullSurfaces.data(), static_cast<std::uint32_t>(colliders.size()),
        static_cast<std::uint32_t>(props.size()), reason);

    if (created == 0)
    {
        return std::unexpected("the physics world was not created: " + reason);
    }

    // Held from here on, so that a failure below destroys the world rather than leaking it.
    auto built = PhysicsWorld(created, mesh.materials);

    if (!props.empty())
    {
        // The same flattening the colliders take, one level deeper: a prop is a list of hulls, so
        // there is a count of hulls per prop as well as a count of points per hull.
        auto propHullCounts = std::vector<std::uint32_t>();
        auto propPointCounts = std::vector<std::uint32_t>();
        auto propPoints = std::vector<double>();
        auto propOrigins = std::vector<double>();
        auto masses = std::vector<double>();
        auto inertias = std::vector<double>();
        auto breakForces = std::vector<double>();
        auto breakTorques = std::vector<double>();

        propHullCounts.reserve(props.size());
        propOrigins.reserve(props.size() * 3);
        masses.reserve(props.size());
        inertias.reserve(props.size() * 9);
        breakForces.reserve(props.size());
        breakTorques.reserve(props.size());

        // Every prop hull takes the same surface, which is the surface every collider takes: what a
        // wheel that lands on a bench should read. It is the mesh's own last material by convention
        // and is stated by the caller through the colliders, so the first collider's is used and a
        // world with props and no colliders is refused rather than guessing.
        if (colliders.empty())
        {
            return std::unexpected("a world with props needs at least one collider, because the props take the "
                                   "surface the colliders were given");
        }

        const auto surface = colliders.front().surface;

        for (const auto& prop : props)
        {
            if (prop.hulls.empty())
            {
                return std::unexpected("a breakable prop carries no hulls");
            }

            if (!(prop.mass > 0.0))
            {
                return std::unexpected("a breakable prop states no mass, and a body with none cannot be integrated");
            }

            propHullCounts.push_back(static_cast<std::uint32_t>(prop.hulls.size()));

            const auto& origin = prop.hulls.front().origin;
            propOrigins.insert(propOrigins.end(), {origin.x, origin.y, origin.z});
            masses.push_back(prop.mass);
            breakForces.push_back(prop.breakForce);
            breakTorques.push_back(prop.breakTorque);

            // Row major, which is what the far side reads it back as.
            for (auto row = 0; row < 3; row++)
            {
                for (auto column = 0; column < 3; column++)
                {
                    // glm indexes a matrix by column first, so this transposes on the way out.
                    inertias.push_back(prop.inertia[column][row]);
                }
            }

            for (const auto& hull : prop.hulls)
            {
                propPointCounts.push_back(static_cast<std::uint32_t>(hull.points.size()));
                for (const auto& point : hull.points)
                {
                    propPoints.insert(propPoints.end(), {point.x, point.y, point.z});
                }
            }
        }

        if (raceengineJoltAddProps(created, propHullCounts.data(), propPointCounts.data(), propPoints.data(),
                                   propOrigins.data(), masses.data(), inertias.data(), breakForces.data(),
                                   breakTorques.data(), static_cast<std::uint32_t>(props.size()), surface, reason) == 0)
        {
            return std::unexpected("the world's props were refused: " + reason);
        }
    }

    // The broadphase tree and the material lookup, once every body this world will ever have is in
    // it. A prop that breaks later changes its motion type and never its existence, which is what
    // keeps this a one-pass build.
    raceengineJoltFinish(created);

    return built;
}

void PhysicsWorld::applyPropImpulses(const std::span<const PropImpulse> impulses, const double deltaTime)
{
    if (impulses.empty() || deltaTime <= 0.0)
    {
        return;
    }

    auto indices = std::vector<std::uint32_t>();
    auto vectors = std::vector<double>();
    auto points = std::vector<double>();

    indices.reserve(impulses.size());
    vectors.reserve(impulses.size() * 3);
    points.reserve(impulses.size() * 3);

    for (const auto& entry : impulses)
    {
        if (entry.prop == noProp)
        {
            continue;
        }

        indices.push_back(entry.prop);
        vectors.insert(vectors.end(), {entry.impulse.x, entry.impulse.y, entry.impulse.z});
        points.insert(points.end(), {entry.point.x, entry.point.y, entry.point.z});
    }

    if (indices.empty())
    {
        return;
    }

    raceengineJoltBreakProps(world, indices.data(), vectors.data(), points.data(),
                             static_cast<std::uint32_t>(indices.size()), deltaTime);
}

void PhysicsWorld::releaseProps(const std::span<const std::uint32_t> props)
{
    if (props.empty())
    {
        return;
    }

    raceengineJoltReleaseProps(world, props.data(), static_cast<std::uint32_t>(props.size()));
}

void PhysicsWorld::applyPropVelocities(const std::span<const PropVelocity> velocities)
{
    if (velocities.empty())
    {
        return;
    }

    auto indices = std::vector<std::uint32_t>();
    auto linear = std::vector<double>();
    auto angular = std::vector<double>();

    indices.reserve(velocities.size());
    linear.reserve(velocities.size() * 3);
    angular.reserve(velocities.size() * 3);

    for (const auto& entry : velocities)
    {
        if (entry.prop == noProp)
        {
            continue;
        }

        indices.push_back(entry.prop);
        linear.insert(linear.end(), {entry.linear.x, entry.linear.y, entry.linear.z});
        angular.insert(angular.end(), {entry.angular.x, entry.angular.y, entry.angular.z});
    }

    if (indices.empty())
    {
        return;
    }

    raceengineJoltAddPropVelocities(world, indices.data(), linear.data(), angular.data(),
                                    static_cast<std::uint32_t>(indices.size()));
}

void PhysicsWorld::step(const double deltaTime)
{
    RACEENGINE_ZONE_N("PhysicsWorld::step");

    raceengineJoltStepWorld(world, deltaTime);
}

void PhysicsWorld::freeProps(std::vector<PropTransform>& into) const
{
    // Sized to whatever the caller last saw plus room to grow, then trimmed. A world where nothing
    // has broken writes nothing and allocates nothing, which is every world before the first crash.
    constexpr auto limit = std::uint32_t{4096};

    auto indices = std::vector<std::uint32_t>(limit);
    auto positions = std::vector<double>(limit * 3);
    auto orientations = std::vector<double>(limit * 4);

    const auto found = raceengineJoltReadFreeProps(world, limit, indices.data(), positions.data(), orientations.data());

    into.clear();
    into.reserve(found);
    for (auto index = std::uint32_t{0}; index < found; index++)
    {
        into.push_back(PropTransform{
            .prop = indices[index],
            .position = glm::dvec3(positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]),
            .orientation = glm::dquat(orientations[index * 4], orientations[index * 4 + 1], orientations[index * 4 + 2],
                                      orientations[index * 4 + 3])});
    }
}

const std::vector<SurfaceMaterial>& PhysicsWorld::materials() const
{
    return surfaceMaterials;
}

void PhysicsWorld::castRays(const std::vector<glm::dvec3>& origins, const std::vector<glm::dvec3>& directions,
                            const double maxDistance, std::vector<SurfaceHit>& results) const
{
    RACEENGINE_ZONE_N("PhysicsWorld::castRays");

    const auto count = origins.size() < directions.size() ? origins.size() : directions.size();
    results.assign(count, SurfaceHit{});

    if (count == 0)
    {
        return;
    }

    // Scratch in the shape the bridge takes. Sized once per call rather than per sample, which
    // is the only allocation on this path and the reason the call is batched.
    auto points = std::vector<double>(count * 3);
    auto normals = std::vector<double>(count * 3);
    auto distances = std::vector<double>(count);
    auto surfaces = std::vector<std::uint32_t>(count);
    auto hits = std::vector<unsigned char>(count);

    raceengineJoltCastRays(world, &origins.front().x, &directions.front().x, maxDistance,
                           static_cast<std::uint32_t>(count), points.data(), normals.data(), distances.data(),
                           surfaces.data(), hits.data());

    for (auto index = std::size_t{0}; index < count; index++)
    {
        results[index] =
            SurfaceHit{.point = glm::dvec3(points[index * 3], points[index * 3 + 1], points[index * 3 + 2]),
                       .normal = glm::dvec3(normals[index * 3], normals[index * 3 + 1], normals[index * 3 + 2]),
                       .distance = distances[index],
                       .surface = surfaces[index],
                       .hit = hits[index] != 0};
    }
}

std::uint64_t PhysicsWorld::handle() const
{
    return world;
}

PhysicsWorld::PhysicsWorld(const std::uint64_t created, std::vector<SurfaceMaterial> surfaces) :
    world(created),
    surfaceMaterials(std::move(surfaces))
{
}

} // namespace raceengine
