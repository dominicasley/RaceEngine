// PhysicsWorld bodies. Declarations are in Api/PhysicsWorld.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces
// an object and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition
// left in the interface partition is part of the module's BMI instead, and editing one rebuilt
// every importer of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.physics;

// The bridge declarations these bodies call through. Repeated here rather than imported: they sit
// in the partition's purview and are not exported, so an implementation unit of the same module
// cannot see them through the primary interface. Declaring them here puts them in the same module,
// which is what makes the symbol the one Physics/Backend/JoltBackend.cpp defines.
extern "C++" std::uint64_t raceengineJoltCreateWorld(const double* vertexData, std::uint32_t vertexCount,
                                                     const std::uint32_t* indexData, std::uint32_t triangleCount,
                                                     const std::uint32_t* surfaceData, std::uint32_t surfaceCount,
                                                     std::string& reason);
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

std::expected<PhysicsWorld, std::string> PhysicsWorld::create(const SurfaceMesh& mesh)
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

    // glm::dvec3 is three doubles with no padding, which the static_assert in the interface is what
    // makes safe to rely on rather than assume — the whole vertex array crosses as one pointer.
    const auto created =
        raceengineJoltCreateWorld(&mesh.vertices.front().x, static_cast<std::uint32_t>(mesh.vertices.size()),
                                  mesh.indices.data(), static_cast<std::uint32_t>(mesh.triangleCount()),
                                  mesh.surfaces.data(), static_cast<std::uint32_t>(mesh.materials.size()), reason);

    if (created == 0)
    {
        return std::unexpected("the physics world was not created: " + reason);
    }

    return PhysicsWorld(created, mesh.materials);
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
