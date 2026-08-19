module;

#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:PhysicsWorld;

import :ProvingGround;

// Jolt's half of the world, declared rather than included — the definitions are in
// Physics/Backend/JoltBackend.cpp, which is a separate target so that Jolt's instruction-set usage
// requirements never reach a module unit. Only fundamental types and std::string cross: a struct
// declared in a module unit is attached to that module and is a *different type* from an
// identically written one in the plain TU, so there is no safe way to pass one.
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

// What one query against the world came back with. Not a contact — a *sample* of the surface, which
// is the raw material the contact patch is aggregated from. A wheel takes a grid of these.
export struct SurfaceHit
{
    glm::dvec3 point{0.0};
    glm::dvec3 normal{0.0, 1.0, 0.0};
    double distance = 0.0;
    // Index into the mesh's material table, which is what carries grip and bumpiness.
    std::uint32_t surface = 0;
    bool hit = false;
};

// Jolt, used for exactly what it is better at than anything worth writing: a broadphase and a BVH
// to ask the track questions against. It does not integrate the vehicle, resolve its contacts, or
// own a solver anywhere near the contact patch — the manifolds it generates are read back and
// resolved by us, and vehicle bodies are kinematic transforms we write.
//
// Move-only and RAII over the backend's handle, because a world is a large thing to copy by
// accident and the alternative to owning it here is an explicit destroy call that a failing test
// skips.
export class PhysicsWorld
{
public:
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    PhysicsWorld(PhysicsWorld&& other) noexcept :
        world(std::exchange(other.world, 0))
    {
    }

    PhysicsWorld& operator=(PhysicsWorld&& other) noexcept
    {
        if (this != &other)
        {
            raceengineJoltDestroyWorld(world);
            world = std::exchange(other.world, 0);
        }

        return *this;
    }

    ~PhysicsWorld()
    {
        raceengineJoltDestroyWorld(world);
    }

    // The surface mesh becomes one static body with per-triangle materials. Requires `bringUpJolt`
    // to have run: Jolt's shape types register into a factory that does not exist before it.
    [[nodiscard]] static std::expected<PhysicsWorld, std::string> create(const SurfaceMesh& mesh)
    {
        auto reason = std::string();

        // glm::dvec3 is three doubles with no padding, which the static_assert below is what makes
        // safe to rely on rather than assume — the whole vertex array crosses as one pointer.
        const auto created =
            raceengineJoltCreateWorld(&mesh.vertices.front().x, static_cast<std::uint32_t>(mesh.vertices.size()),
                                      mesh.indices.data(), static_cast<std::uint32_t>(mesh.triangleCount()),
                                      mesh.surfaces.data(), static_cast<std::uint32_t>(mesh.materials.size()), reason);

        if (created == 0)
        {
            return std::unexpected("the physics world was not created: " + reason);
        }

        return PhysicsWorld(created);
    }

    // Batched, and that is the interface rather than a convenience over a single cast: a wheel's
    // contact patch is a grid of samples taken at one instant, four wheels of them per tick, and
    // asking one at a time would walk the same broadphase nodes for every one of them.
    //
    // `directions` are expected normalised; `maxDistance` is how far along each to look. A miss
    // reports `hit == false` with the distance left at maxDistance, so a caller aggregating a patch
    // can weight by hit without branching into a separate path.
    void castRays(const std::vector<glm::dvec3>& origins, const std::vector<glm::dvec3>& directions,
                  const double maxDistance, std::vector<SurfaceHit>& results) const
    {
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

    // The backend's own identifier for this world. Exposed because the contact bridge is a second
    // entry point into the same world and has to name it; nothing outside this module's partitions
    // has any use for it.
    [[nodiscard]] std::uint64_t handle() const
    {
        return world;
    }

private:
    explicit PhysicsWorld(const std::uint64_t created) :
        world(created)
    {
    }

    std::uint64_t world = 0;
};

// The vertex and direction arrays cross the bridge as bare `const double*` over the whole vector,
// so their element type has to be exactly three packed doubles. glm gives no such guarantee in its
// own words; this is where the assumption is checked rather than discovered.
static_assert(sizeof(glm::dvec3) == 3 * sizeof(double), "the ray bridge passes dvec3 arrays as flat doubles");

} // namespace raceengine
