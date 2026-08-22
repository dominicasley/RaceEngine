// Plain TU, and its own CMake target, for a reason that is not the usual one.
//
// Jolt's imported target carries `-mavx2 -mfma -mbmi -mlzcnt -mf16c -mpopcnt` and eight JPH_USE_*
// defines as usage requirements, and a clang BMI records the target features it was built with. So
// linking Jolt into `Engine` does not merely change Engine's codegen: every translation unit that
// imports an Engine module — in the tests and in the sandbox, neither of which links Jolt — then
// fails outright with "AST file was compiled with the target feature '+avx2' but the current
// translation unit is not". The whole workspace would have to adopt Jolt's instruction set to
// import a module that has nothing to do with physics.
//
// Keeping Jolt behind a non-module target is what stops that. Its usage requirements end here, and
// the engine above sees the free functions below, which name no JPH type. This is the same
// reasoning that keeps the Vulkan backend in an implementation partition, one step further out:
// there the leak was BMI size, here it is the target machine.
//
// The declarations are repeated in Physics/Api/*.cppm rather than shared through a header, as the
// tinygltf bridge in Io/ThirdPartyImpl.cpp is: this workspace has no first-party headers, and a
// module may not include one anyway. That constraint is also why nothing below takes a struct —
// a type declared in a module unit is *attached* to that module and is a different type from an
// identically written one out here, so the boundary is fundamental types, arrays of them, and
// std::string, which both sides get from the same global-module header.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/PhysicsMaterialSimple.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

namespace
{

// Jolt's allocator hooks, factory and type registry are process-wide singletons — that is Jolt's
// design, not a choice made here — so the flag guarding them is process-wide too. This is the case
// the "no function-local statics" rule is about the opposite of: the state being guarded really is
// per-process, so a per-instance flag would be the wrong shape.
bool joltRunning = false;

// Two object layers and two broadphase layers, which is the smallest set that still says the true
// thing: the ground never moves and never needs testing against itself, and everything else does.
// The vehicle bodies this world will carry are kinematic — we drive their transforms — so they live
// in MOVING and are tested against the ground without Jolt ever integrating them.
namespace Layers
{
constexpr JPH::ObjectLayer nonMoving = 0;
constexpr JPH::ObjectLayer moving = 1;
constexpr JPH::ObjectLayer count = 2;
} // namespace Layers

namespace BroadPhaseLayers
{
constexpr JPH::BroadPhaseLayer nonMoving(0);
constexpr JPH::BroadPhaseLayer moving(1);
constexpr JPH::uint count(2);
} // namespace BroadPhaseLayers

class BroadPhaseLayerMapping final : public JPH::BroadPhaseLayerInterface
{
public:
    BroadPhaseLayerMapping()
    {
        layers[Layers::nonMoving] = BroadPhaseLayers::nonMoving;
        layers[Layers::moving] = BroadPhaseLayers::moving;
    }

    [[nodiscard]] JPH::uint GetNumBroadPhaseLayers() const override
    {
        return BroadPhaseLayers::count;
    }

    [[nodiscard]] JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override
    {
        return layers[layer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    [[nodiscard]] const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override
    {
        return layer == BroadPhaseLayers::nonMoving ? "non-moving" : "moving";
    }
#endif

private:
    JPH::BroadPhaseLayer layers[Layers::count] = {BroadPhaseLayers::nonMoving, BroadPhaseLayers::nonMoving};
};

class ObjectVsBroadPhaseFilter final : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer broadPhaseLayer) const override
    {
        return layer == Layers::nonMoving ? broadPhaseLayer == BroadPhaseLayers::moving : true;
    }
};

class ObjectLayerPairFilter final : public JPH::ObjectLayerPairFilter
{
public:
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer first, JPH::ObjectLayer second) const override
    {
        return first == Layers::nonMoving ? second == Layers::moving : true;
    }
};

struct World
{
    // Declared in the order they must be destroyed in reverse of, as the engine's own composition
    // root is: the physics system holds bodies whose shapes hold the materials below it.
    BroadPhaseLayerMapping broadPhaseLayers;
    ObjectVsBroadPhaseFilter objectVsBroadPhase;
    ObjectLayerPairFilter objectLayerPairs;

    JPH::TempAllocatorImpl temporaries{8 * 1024 * 1024};
    std::unique_ptr<JPH::JobSystemThreadPool> jobs;
    JPH::PhysicsSystem system;

    // Our surface index for each Jolt material, in the order the materials were handed over. A hit
    // reports a material *pointer*, and this is what turns it back into the index the surface table
    // is keyed by.
    //
    // **The comment here used to say "three entries, so a linear scan is the whole lookup", and it
    // was written against the proving ground.** A loaded circuit brings its own table: Bathurst's is
    // nine, and the scan ran per ray for every one of the thirty-six a car casts per tick. Sorted
    // once at creation and binary searched since — the world is immutable after `create`, so the
    // ordering can never go stale.
    //
    // Honestly: this is **not** worth a measurable amount. The per-ray cost did not move outside the
    // noise when it landed, because the scan broke on its first comparison for tarmac — which is
    // most rays — and nine entries is a short walk even when it does not. It is kept because it
    // removes a cost that grows with a track's surface count for no complexity, not because it was
    // found to be slow.
    std::vector<const JPH::PhysicsMaterial*> materialOrder;
    std::vector<std::pair<const JPH::PhysicsMaterial*, std::uint32_t>> materialLookup;

    [[nodiscard]] std::uint32_t surfaceOf(const JPH::PhysicsMaterial* material) const
    {
        const auto found = std::lower_bound(materialLookup.begin(), materialLookup.end(), material,
                                            [](const auto& entry, const auto* key) { return entry.first < key; });

        return found != materialLookup.end() && found->first == material ? found->second : 0;
    }
};

// Worlds are handed out as an index rather than as a pointer, so that a handle from a destroyed
// world is caught here instead of dereferenced. Slots are not reused; there is one world.
std::vector<std::unique_ptr<World>> worlds;

World* worldFor(const std::uint64_t handle)
{
    if (handle == 0 || handle > worlds.size())
    {
        return nullptr;
    }

    return worlds[handle - 1].get();
}

} // namespace

bool raceengineJoltBringUp(std::string& reason)
{
    if (joltRunning)
    {
        reason = "Jolt is already running";
        return false;
    }

    // Order is fixed by Jolt: an allocator before anything allocates, the factory before the types
    // that register into it.
    JPH::RegisterDefaultAllocator();

    if (JPH::Factory::sInstance != nullptr)
    {
        reason = "a Jolt factory already exists; something outside this bridge created one";
        return false;
    }

    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    joltRunning = true;

    return true;
}

void raceengineJoltTearDown()
{
    if (!joltRunning)
    {
        return;
    }

    // Worlds first: their shapes hold references into the type registry that UnregisterTypes is
    // about to take down.
    worlds.clear();

    JPH::UnregisterTypes();

    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    joltRunning = false;
}

std::uint64_t raceengineJoltCreateWorld(const double* vertexData, const std::uint32_t vertexCount,
                                        const std::uint32_t* indexData, const std::uint32_t triangleCount,
                                        const std::uint32_t* surfaceData, const std::uint32_t surfaceCount,
                                        std::string& reason)
{
    if (!joltRunning)
    {
        reason = "the physics backend has not been started";
        return 0;
    }

    if (vertexCount == 0 || triangleCount == 0)
    {
        reason = "a collision mesh needs vertices and triangles";
        return 0;
    }

    auto world = std::make_unique<World>();

    // One physics material per surface kind. Jolt keeps them by reference from the shape, and the
    // shape outlives this call, so the list is handed over rather than borrowed.
    JPH::PhysicsMaterialList materials;
    materials.reserve(surfaceCount);
    for (auto surface = std::uint32_t{0}; surface < surfaceCount; surface++)
    {
        auto* material = new JPH::PhysicsMaterialSimple("surface " + std::to_string(surface), JPH::Color::sGrey);
        materials.push_back(material);
        world->materialOrder.push_back(material);
    }

    JPH::VertexList vertices;
    vertices.reserve(vertexCount);
    for (auto vertex = std::uint32_t{0}; vertex < vertexCount; vertex++)
    {
        // Jolt is single precision. At the scale a contact patch is sampled over that is a
        // resolution of about a tenth of a micron, which is nothing; a long way from the origin it
        // is not, and that is the floating-origin question rather than this one.
        vertices.push_back(JPH::Float3(static_cast<float>(vertexData[vertex * 3 + 0]),
                                       static_cast<float>(vertexData[vertex * 3 + 1]),
                                       static_cast<float>(vertexData[vertex * 3 + 2])));
    }

    JPH::IndexedTriangleList triangles;
    triangles.reserve(triangleCount);
    for (auto triangle = std::uint32_t{0}; triangle < triangleCount; triangle++)
    {
        const auto surface = surfaceData == nullptr ? 0u : surfaceData[triangle];
        if (surface >= surfaceCount)
        {
            reason = "triangle " + std::to_string(triangle) + " names surface " + std::to_string(surface) +
                     ", of which there are " + std::to_string(surfaceCount);
            return 0;
        }

        triangles.push_back(JPH::IndexedTriangle(indexData[triangle * 3 + 0], indexData[triangle * 3 + 1],
                                                 indexData[triangle * 3 + 2], surface));
    }

    JPH::MeshShapeSettings meshSettings(vertices, triangles, materials);
    meshSettings.SetEmbedded();

    const auto shape = meshSettings.Create();
    if (shape.HasError())
    {
        reason = "the collision mesh was refused: " + std::string(shape.GetError().c_str());
        return 0;
    }

    const auto hardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    world->jobs = std::make_unique<JPH::JobSystemThreadPool>(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers,
                                                             std::clamp(hardwareThreads - 1, 1, 8));

    world->system.Init(1024, 0, 1024, 1024, world->broadPhaseLayers, world->objectVsBroadPhase,
                       world->objectLayerPairs);

    JPH::BodyCreationSettings groundSettings(shape.Get(), JPH::RVec3::sZero(), JPH::Quat::sIdentity(),
                                             JPH::EMotionType::Static, Layers::nonMoving);

    auto& bodies = world->system.GetBodyInterface();
    const auto ground = bodies.CreateAndAddBody(groundSettings, JPH::EActivation::DontActivate);
    if (ground.IsInvalid())
    {
        reason = "the ground body could not be added";
        return 0;
    }

    // Builds the broadphase tree in one pass rather than incrementally, which is what it is for
    // when every static body is known up front.
    world->system.OptimizeBroadPhase();

    // The material lookup, built once beside the tree and for the same reason: everything about this
    // world is known now and nothing changes it afterwards.
    world->materialLookup.reserve(world->materialOrder.size());
    for (auto index = std::size_t{0}; index < world->materialOrder.size(); index++)
    {
        world->materialLookup.emplace_back(world->materialOrder[index], static_cast<std::uint32_t>(index));
    }
    std::sort(world->materialLookup.begin(), world->materialLookup.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });

    worlds.push_back(std::move(world));

    return worlds.size();
}

void raceengineJoltDestroyWorld(const std::uint64_t handle)
{
    if (auto* world = worldFor(handle); world != nullptr)
    {
        worlds[handle - 1].reset();
    }
}

void raceengineJoltCastRays(const std::uint64_t handle, const double* origins, const double* directions,
                            const double maxDistance, const std::uint32_t count, double* outPoints, double* outNormals,
                            double* outDistances, std::uint32_t* outSurfaces, unsigned char* outHits)
{
    auto* world = worldFor(handle);

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        outHits[index] = 0;
        outDistances[index] = maxDistance;
        outSurfaces[index] = 0;
        outPoints[index * 3 + 0] = origins[index * 3 + 0];
        outPoints[index * 3 + 1] = origins[index * 3 + 1];
        outPoints[index * 3 + 2] = origins[index * 3 + 2];
        outNormals[index * 3 + 0] = 0.0;
        outNormals[index * 3 + 1] = 1.0;
        outNormals[index * 3 + 2] = 0.0;

        if (world == nullptr)
        {
            continue;
        }

        const auto origin =
            JPH::RVec3(static_cast<float>(origins[index * 3 + 0]), static_cast<float>(origins[index * 3 + 1]),
                       static_cast<float>(origins[index * 3 + 2]));
        const auto direction =
            JPH::Vec3(static_cast<float>(directions[index * 3 + 0]), static_cast<float>(directions[index * 3 + 1]),
                      static_cast<float>(directions[index * 3 + 2]));

        // Jolt carries the ray's length in its direction vector rather than as a separate limit.
        const JPH::RRayCast ray{origin, direction * static_cast<float>(maxDistance)};

        // **The locking query, kept deliberately, and the measurement that says so is worth the
        // comment.** This world is immutable after `create` — one static body, no `system.Update`
        // anywhere — so `GetNarrowPhaseQueryNoLock` and `GetBodyLockInterfaceNoLock` are *safe*
        // here, and they were tried on the strength of a decomposition that put about a third of the
        // 324 ns per-ray cost outside tree traversal. **They bought about one percent, which is
        // inside the run-to-run noise.** The lock was not where that third lives.
        //
        // So the precondition is not worth carrying: a no-lock query would have to be reverted the
        // day anything adds a body or steps the system, and it buys nothing measurable today.
        //
        // The one case that would reopen this is the multi-car milestone *if it casts from several
        // threads* — lock contention is a different question from lock acquisition, and this
        // measurement says nothing about it.
        // **A ray can miss a solid mesh, and it costs a wheel.** Measured 2026-08-22 on the kerb
        // crossing: one ray of 31,851 came back with no hit while the two samples either side of it
        // in the same row reported 19.5 and 18.8 millimetres of road, and that single ray is the
        // whole of why `the imported car crosses a kerb continuously` fails — 25.9 mm of patch-centre
        // movement in one tick and 665 N of load, out and straight back the next tick.
        //
        // Characterised rather than guessed at (`./EngineTests "[.kerb-discontinuity]"`): the misses
        // lie in a band **35 micrometres wide in z and indifferent to x**, sitting beside a shared
        // triangle edge of the ground mesh and not on it — a ray cast exactly on the edge hits, and 0
        // of 200 row lines swept lose one. It is independent of the ray's length, so it is not the
        // search distance. That is the signature of the float32 edge test inside the mesh shape
        // losing its sign over a few units in the last place: the ground's own coordinates are tens
        // of metres, where a float's spacing is about 4 um, and the band is a few of those.
        //
        // So the ray is re-cast from a millimetre away, twice, along two directions perpendicular to
        // it and to each other. A dead band is a *line*, so an offset that is not parallel to it
        // clears it, and two independent offsets cannot both be parallel to the same line. A
        // millimetre is
        // 30 times the band and a thirtieth of the contact grid's own pitch, so a retry that hits is
        // reporting the same piece of road. A wheel genuinely over a void misses all three times and
        // is still correctly reported as missing.
        JPH::RayCastResult result;
        auto hitRay = ray;

        if (!world->system.GetNarrowPhaseQuery().CastRay(hitRay, result))
        {
            const auto axis = std::abs(direction.GetX()) < std::abs(direction.GetY())
                                  ? (std::abs(direction.GetX()) < std::abs(direction.GetZ()) ? JPH::Vec3::sAxisX()
                                                                                             : JPH::Vec3::sAxisZ())
                                  : (std::abs(direction.GetY()) < std::abs(direction.GetZ()) ? JPH::Vec3::sAxisY()
                                                                                             : JPH::Vec3::sAxisZ());

            const auto first = direction.Cross(axis).Normalized() * 1.0e-3F;
            const auto second = direction.Cross(first).Normalized() * 1.0e-3F;

            auto recovered = false;
            for (const auto& nudge : {first, second})
            {
                hitRay = JPH::RRayCast{origin + nudge, ray.mDirection};
                if (world->system.GetNarrowPhaseQuery().CastRay(hitRay, result))
                {
                    recovered = true;
                    break;
                }
            }

            if (!recovered)
            {
                continue;
            }
        }

        // The point on the ray that actually hit, which is the nudged one when a retry recovered it.
        // Taking it off the original ray would report the road at the offset ray's depth under the
        // original ray's position, which is a millimetre of road slope reported as a millimetre of
        // tyre compression.
        const auto point = hitRay.GetPointOnRay(result.mFraction);

        JPH::BodyLockRead lock(world->system.GetBodyLockInterface(), result.mBodyID);
        if (!lock.Succeeded())
        {
            continue;
        }

        const auto& body = lock.GetBody();
        const auto normal = body.GetWorldSpaceSurfaceNormal(result.mSubShapeID2, point);
        const auto* material = body.GetShape()->GetMaterial(result.mSubShapeID2);

        outHits[index] = 1;
        outDistances[index] = static_cast<double>(result.mFraction) * maxDistance;
        outPoints[index * 3 + 0] = static_cast<double>(point.GetX());
        outPoints[index * 3 + 1] = static_cast<double>(point.GetY());
        outPoints[index * 3 + 2] = static_cast<double>(point.GetZ());
        outNormals[index * 3 + 0] = static_cast<double>(normal.GetX());
        outNormals[index * 3 + 1] = static_cast<double>(normal.GetY());
        outNormals[index * 3 + 2] = static_cast<double>(normal.GetZ());

        outSurfaces[index] = world->surfaceOf(material);
    }
}

// Contact manifolds between one box and the static world.
//
// This is the second thing Jolt is genuinely better at than anything worth writing, after the
// broadphase: given a convex shape and a triangle mesh it produces the set of contact points, their
// normals and their penetration depths. What it is *not* asked to do is resolve them — the impulses
// are ours, because ramming and PIT manoeuvres are the game's core verb and the outcomes have to be
// repeatable and controllable rather than merely plausible.
//
// The box is built per call rather than cached. It is a handful of floats and the cost is nothing
// against the collision query it is an argument to; caching it would mean keying a cache on the
// extents and holding a reference across ticks for no measurable gain.
std::uint32_t raceengineJoltCollideBox(const std::uint64_t handle, const double* halfExtents, const double* centre,
                                       const double* orientation, const std::uint32_t maxContacts, double* outPoints,
                                       double* outNormals, double* outDepths)
{
    auto* world = worldFor(handle);
    if (world == nullptr || maxContacts == 0)
    {
        return 0;
    }

    JPH::BoxShape box(JPH::Vec3(static_cast<float>(halfExtents[0]), static_cast<float>(halfExtents[1]),
                                static_cast<float>(halfExtents[2])));
    box.SetEmbedded();

    // The orientation arrives as a quaternion in w, x, y, z order, which is how glm writes one.
    const JPH::Quat rotation(static_cast<float>(orientation[1]), static_cast<float>(orientation[2]),
                             static_cast<float>(orientation[3]), static_cast<float>(orientation[0]));
    const JPH::RVec3 position(static_cast<float>(centre[0]), static_cast<float>(centre[1]),
                              static_cast<float>(centre[2]));

    JPH::CollideShapeSettings settings;
    settings.mBackFaceMode = JPH::EBackFaceMode::IgnoreBackFaces;
    settings.mMaxSeparationDistance = 0.0f;

    JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> collector;
    world->system.GetNarrowPhaseQuery().CollideShape(&box, JPH::Vec3::sReplicate(1.0f),
                                                     JPH::RMat44::sRotationTranslation(rotation, position), settings,
                                                     JPH::RVec3::sZero(), collector);

    const auto found = std::min(static_cast<std::uint32_t>(collector.mHits.size()), maxContacts);
    for (auto index = std::uint32_t{0}; index < found; index++)
    {
        const auto& hit = collector.mHits[index];

        // Jolt's penetration axis points from the first shape towards the second and is not
        // normalised. The contact normal wanted here is the direction that pushes the *box* out,
        // which is the other way.
        const auto axis = hit.mPenetrationAxis.Normalized();

        outNormals[index * 3 + 0] = -static_cast<double>(axis.GetX());
        outNormals[index * 3 + 1] = -static_cast<double>(axis.GetY());
        outNormals[index * 3 + 2] = -static_cast<double>(axis.GetZ());

        outPoints[index * 3 + 0] = static_cast<double>(hit.mContactPointOn2.GetX());
        outPoints[index * 3 + 1] = static_cast<double>(hit.mContactPointOn2.GetY());
        outPoints[index * 3 + 2] = static_cast<double>(hit.mContactPointOn2.GetZ());

        outDepths[index] = static_cast<double>(hit.mPenetrationDepth);
    }

    return found;
}
