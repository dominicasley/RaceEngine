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
#include <cstddef>
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
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/PhysicsMaterialSimple.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
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

// What a contact says when it did not land on a prop, which is every contact with the road, a
// building and a static prop. Matches the sentinel the physics module states.
constexpr std::uint32_t noProp = 0xffffffffu;

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

    // One entry per breakable prop, in the order they were handed over, which is the order the index
    // in a contact means. A prop is *static* until it breaks and dynamic afterwards, and `free` is
    // the whole of that state: Jolt's own motion type is the truth, and this is the copy the query
    // path reads so that a filter does not have to lock a body to ask.
    struct Prop
    {
        JPH::BodyID body;
        // Kilograms, kept here because **`BodyInterface` has no inverse-mass accessor** and the car's
        // contact solver needs one to resolve against a loose prop as a body rather than as a wall.
        // It is in hand when the body is created and nowhere afterwards.
        double mass = 0.0;

        // And the inverse of the stated tensor, for the same reason twice over:
        // `Body::GetInverseInertia` asserts the body is dynamic, so a **standing** prop cannot be
        // asked for one at all — and the car has to know what a prop would weigh before it decides
        // whether the prop's anchor gives way. A standing body carries the identity rotation, so its
        // local tensor and its world one are the same matrix.
        JPH::Mat44 inverseInertia = JPH::Mat44::sIdentity();

        double breakForce = 0.0;
        double breakTorque = 0.0;
        bool free = false;
    };

    std::vector<Prop> props;
    // Jolt's body index to our prop index, so a contact can say which prop it landed on. Sized to
    // the body budget and filled where a prop stands, because a body index is dense and small and a
    // map would be a lookup per contact point.
    std::vector<std::uint32_t> propOfBody;
    // How many props are loose. While it is zero nothing in this world moves, so the step is skipped
    // outright rather than walking an empty active list.
    std::uint32_t freeProps = 0;

    [[nodiscard]] std::uint32_t propFor(const JPH::BodyID body) const
    {
        const auto index = body.GetIndex();

        return index < propOfBody.size() ? propOfBody[index] : noProp;
    }

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
                                        const double* hullPointData, const std::uint32_t* hullPointCounts,
                                        const double* hullOrigins, const std::uint32_t* hullSurfaces,
                                        const std::uint32_t hullCount, const std::uint32_t propBudget,
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

    // **The body budget, and it used to be a flat 1024.** Grand City Parkway's derived building hulls
    // are 8147 of them on their own; a world asked for more bodies than it was initialised for does
    // not grow, it starts refusing them, and a refused static body is a facade a car drives through
    // with no error anywhere. So the ceiling is derived from what this world was actually handed,
    // with headroom for the vehicle bodies a later milestone adds.
    //
    // The body-pair and contact-constraint caps used to be dead budgets, because nothing called
    // `PhysicsSystem::Update`. **Props changed that**: a prop that breaks free is integrated by Jolt
    // from then on, so the two caps are now real. They are sized for the loose bodies rather than for
    // the world — a city of eight thousand static hulls generates no pair at all while nothing in it
    // moves, and what is being budgeted for is a heap of knocked-over street furniture.
    const auto bodyBudget = std::max(JPH::uint(1024), JPH::uint(hullCount) + JPH::uint(propBudget) + JPH::uint(64));

    world->system.Init(bodyBudget, 0, 4096, 4096, world->broadPhaseLayers, world->objectVsBroadPhase,
                       world->objectLayerPairs);

    world->propOfBody.assign(bodyBudget, noProp);

    JPH::BodyCreationSettings groundSettings(shape.Get(), JPH::RVec3::sZero(), JPH::Quat::sIdentity(),
                                             JPH::EMotionType::Static, Layers::nonMoving);

    auto& bodies = world->system.GetBodyInterface();
    const auto ground = bodies.CreateAndAddBody(groundSettings, JPH::EActivation::DontActivate);
    if (ground.IsInvalid())
    {
        reason = "the ground body could not be added";
        return 0;
    }

    // The convex colliders that stand *beside* the drivable surface: buildings and street furniture,
    // every one of them derived from drawn geometry rather than authored as collision, and every one
    // of them static.
    //
    // **One body per hull, rather than one compound body per building.** The grouping a compound
    // needs is not in the data that reaches here — a hull arrives as a placement and a point cloud,
    // and which building it belongs to lives in a glTF node name the model loader does not carry —
    // so the choice is between grouping on a guess and not grouping. Eight thousand static bodies in
    // a quadtree broadphase is not a quantity Jolt notices, and the guess would be silent when it
    // was wrong.
    //
    // The points are local to the hull's own origin and the origin is the body's position, which is
    // the arrangement that keeps single precision honest: a city is kilometres across, and a hull
    // whose vertices were baked to world coordinates would be resolving centimetres against
    // thousands of metres.
    auto hullPoints = JPH::Array<JPH::Vec3>();
    auto readPoint = std::size_t{0};

    for (auto hull = std::uint32_t{0}; hull < hullCount; hull++)
    {
        const auto points = hullPointCounts[hull];
        if (points < 4)
        {
            reason = "hull " + std::to_string(hull) + " carries " + std::to_string(points) +
                     " points, and a convex hull needs four";
            return 0;
        }

        const auto surface = hullSurfaces == nullptr ? 0u : hullSurfaces[hull];
        if (surface >= surfaceCount)
        {
            reason = "hull " + std::to_string(hull) + " names surface " + std::to_string(surface) +
                     ", of which there are " + std::to_string(surfaceCount);
            return 0;
        }

        hullPoints.clear();
        hullPoints.reserve(points);
        for (auto point = std::uint32_t{0}; point < points; point++)
        {
            const auto at = (readPoint + point) * 3;
            hullPoints.push_back(JPH::Vec3(static_cast<float>(hullPointData[at + 0]),
                                           static_cast<float>(hullPointData[at + 1]),
                                           static_cast<float>(hullPointData[at + 2])));
        }

        readPoint += points;

        // The default convex radius is kept rather than zeroed: Jolt lowers it by itself where a hull
        // is too small to carry it, which is the case that matters here — the exporter turns a wall
        // panel modelled with no thickness into a 2 cm box, and 2 cm is less than the 5 cm default.
        JPH::ConvexHullShapeSettings hullSettings(hullPoints, JPH::cDefaultConvexRadius, world->materialOrder[surface]);
        hullSettings.SetEmbedded();

        const auto hullShape = hullSettings.Create();
        if (hullShape.HasError())
        {
            reason = "hull " + std::to_string(hull) + " was refused: " + std::string(hullShape.GetError().c_str());
            return 0;
        }

        JPH::BodyCreationSettings hullBody(hullShape.Get(),
                                           JPH::RVec3(static_cast<float>(hullOrigins[hull * 3 + 0]),
                                                      static_cast<float>(hullOrigins[hull * 3 + 1]),
                                                      static_cast<float>(hullOrigins[hull * 3 + 2])),
                                           JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::nonMoving);

        if (bodies.CreateAndAddBody(hullBody, JPH::EActivation::DontActivate).IsInvalid())
        {
            reason = "hull " + std::to_string(hull) + " could not be added; the world's body budget is " +
                     std::to_string(bodyBudget);
            return 0;
        }
    }

    // The broadphase tree is **not** built here any more. The props go in next, through a second
    // call, and a tree built before them would have to be rebuilt anyway — so `raceengineJoltFinish`
    // is what closes the world, and `PhysicsWorld::create` is the only thing that calls either.
    worlds.push_back(std::move(world));

    return worlds.size();
}

// The props: standing street furniture that a hard enough hit sets loose.
//
// **Every one of them is created static**, with `mAllowDynamicOrKinematic` so that Jolt allocates
// its motion properties without simulating it. That flag is the whole design: three and a half
// thousand anchored bodies cost nothing while they stand, there is no constraint to solve and no
// lambda to read back every tick, and breaking one is a motion-type change rather than a body being
// created in the middle of a frame. Jolt has no breakable constraint, and this engine does not need
// one — it computes the car's contact impulses itself, which is the quantity a breakable constraint
// exists to recover.
//
// The mass and the inertia are handed over rather than derived. A lamp column's hull is a solid
// prism and the column is a hollow tube, so a uniform-density tensor is several times too large and
// the post falls like a felled tree in syrup.
std::uint32_t raceengineJoltAddProps(const std::uint64_t handle, const std::uint32_t* propHullCounts,
                                     const std::uint32_t* hullPointCounts, const double* hullPointData,
                                     const double* origins, const double* masses, const double* inertias,
                                     const double* breakForces, const double* breakTorques,
                                     const std::uint32_t propCount, const std::uint32_t surface, std::string& reason)
{
    auto* world = worldFor(handle);
    if (world == nullptr)
    {
        reason = "the prop set names no live world";
        return 0;
    }

    if (surface >= world->materialOrder.size())
    {
        reason = "the props name surface " + std::to_string(surface) + ", of which there are " +
                 std::to_string(world->materialOrder.size());
        return 0;
    }

    auto& bodies = world->system.GetBodyInterface();
    auto hullPoints = JPH::Array<JPH::Vec3>();
    auto readHull = std::size_t{0};
    auto readPoint = std::size_t{0};

    world->props.reserve(propCount);

    for (auto prop = std::uint32_t{0}; prop < propCount; prop++)
    {
        const auto parts = propHullCounts[prop];
        if (parts == 0)
        {
            reason = "prop " + std::to_string(prop) + " carries no hulls";
            return 0;
        }

        // One shape per hull, and a compound only where there is more than one of them. A compound
        // of one is a wrapper Jolt would have to walk on every query for nothing, and most props are
        // a single hull.
        auto shapes = JPH::Array<JPH::Ref<JPH::Shape>>();
        shapes.reserve(parts);

        for (auto part = std::uint32_t{0}; part < parts; part++)
        {
            const auto points = hullPointCounts[readHull + part];
            if (points < 4)
            {
                reason = "prop " + std::to_string(prop) + " has a hull of " + std::to_string(points) +
                         " points, and a convex hull needs four";
                return 0;
            }

            hullPoints.clear();
            hullPoints.reserve(points);
            for (auto point = std::uint32_t{0}; point < points; point++)
            {
                const auto at = (readPoint + point) * 3;
                hullPoints.push_back(JPH::Vec3(static_cast<float>(hullPointData[at + 0]),
                                               static_cast<float>(hullPointData[at + 1]),
                                               static_cast<float>(hullPointData[at + 2])));
            }

            readPoint += points;

            JPH::ConvexHullShapeSettings hullSettings(hullPoints, JPH::cDefaultConvexRadius,
                                                      world->materialOrder[surface]);
            hullSettings.SetEmbedded();

            const auto built = hullSettings.Create();
            if (built.HasError())
            {
                reason =
                    "a hull of prop " + std::to_string(prop) + " was refused: " + std::string(built.GetError().c_str());
                return 0;
            }

            shapes.push_back(built.Get());
        }

        readHull += parts;

        auto shape = shapes.front();
        if (parts > 1)
        {
            // The hull points are already stated in the body's own frame, so every child sits at
            // identity. A compound that re-placed them would be applying the body's origin twice.
            JPH::StaticCompoundShapeSettings compound;
            compound.SetEmbedded();
            for (auto& part : shapes)
            {
                compound.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), part);
            }

            const auto built = compound.Create();
            if (built.HasError())
            {
                reason =
                    "prop " + std::to_string(prop) + " would not compound: " + std::string(built.GetError().c_str());
                return 0;
            }

            shape = built.Get();
        }

        JPH::BodyCreationSettings settings(shape,
                                           JPH::RVec3(static_cast<float>(origins[prop * 3 + 0]),
                                                      static_cast<float>(origins[prop * 3 + 1]),
                                                      static_cast<float>(origins[prop * 3 + 2])),
                                           JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::nonMoving);

        // What makes a static body convertible: without it Jolt allocates no motion properties and
        // the switch to dynamic is not merely refused, it is undefined.
        settings.mAllowDynamicOrKinematic = true;
        settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
        settings.mMassPropertiesOverride.mMass = static_cast<float>(masses[prop]);

        // Row major from the caller, and the body's origin **is** its centre of mass, so the tensor
        // goes in at the origin of the local frame with no parallel-axis shift.
        const auto* stated = inertias + static_cast<std::size_t>(prop) * 9;

        auto tensor = JPH::Mat44::sIdentity();
        for (auto row = std::uint32_t{0}; row < 3; row++)
        {
            tensor.SetColumn4(row, JPH::Vec4(static_cast<float>(stated[0 * 3 + row]),
                                             static_cast<float>(stated[1 * 3 + row]),
                                             static_cast<float>(stated[2 * 3 + row]), 0.0F));
        }
        tensor.SetColumn4(3, JPH::Vec4(0.0F, 0.0F, 0.0F, 1.0F));
        settings.mMassPropertiesOverride.mInertia = tensor;

        const auto body = bodies.CreateAndAddBody(settings, JPH::EActivation::DontActivate);
        if (body.IsInvalid())
        {
            reason = "prop " + std::to_string(prop) + " could not be added; the world's body budget is too small";
            return 0;
        }

        if (body.GetIndex() < world->propOfBody.size())
        {
            world->propOfBody[body.GetIndex()] = static_cast<std::uint32_t>(world->props.size());
        }

        world->props.push_back(World::Prop{.body = body,
                                           .mass = masses[prop],
                                           .inverseInertia = tensor.Inversed(),
                                           .breakForce = breakForces[prop],
                                           .breakTorque = breakTorques[prop]});
    }

    return 1;
}

// Closes the world: the broadphase tree in one pass, and the material lookup beside it.
void raceengineJoltFinish(const std::uint64_t handle)
{
    auto* world = worldFor(handle);
    if (world == nullptr)
    {
        return;
    }

    // Builds the broadphase tree in one pass rather than incrementally, which is what it is for
    // when every static body is known up front.
    world->system.OptimizeBroadPhase();

    // The material lookup, built once beside the tree and for the same reason: every material in
    // this world is known now and no later call adds one.
    world->materialLookup.reserve(world->materialOrder.size());
    for (auto index = std::size_t{0}; index < world->materialOrder.size(); index++)
    {
        world->materialLookup.emplace_back(world->materialOrder[index], static_cast<std::uint32_t>(index));
    }
    std::sort(world->materialLookup.begin(), world->materialLookup.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });
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
// **Loose props are reported like anything else.** This used to refuse them outright, because a
// solver that treated every contact as being against an immovable world would have made a
// twenty-kilogram bin a wall, and a car stopped dead by a bin is worse than a car that drives
// through one. The solver takes a second body now, so the refusal has nothing left to protect and
// the debris is real.

std::uint32_t raceengineJoltCollideBox(const std::uint64_t handle, const double* halfExtents, const double* centre,
                                       const double* orientation, const std::uint32_t maxContacts, double* outPoints,
                                       double* outNormals, double* outDepths, std::uint32_t* outProps)
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
        outProps[index] = world->propFor(hit.mBodyID2);
    }

    return found;
}

// Every triangle a wheel-sized cylinder is touching, for the kerb-contact path.
//
// The third thing Jolt is asked for, after the broadphase and the bodywork's manifolds: the tyre's
// **side view** of the road. The contact patch samples the road with vertical rays and cannot see a
// face steeper than its own grid — GCP's 150 mm kerbs arrive as a one-tick road step under the
// leading samples (docs/kerb-contact-brief.md). A cylinder of the tyre's own radius and width,
// collided against the same mesh, reports every triangle it overlaps with the shortest push-out for
// each: on a kerb face that is the contact the rays miss, and on flat road it is the contact the
// rays already carry. Telling those two apart is the caller's exclusion rule, not this function's.
//
// `axes` are the spin axes. Jolt's cylinder stands on Y, so each is rotated onto its own axis.
//
// **Active-edge mode is the default, `CollideOnlyWithActive`, and it is kept deliberately.** The
// road's internal triangle edges are inactive — coplanar neighbours, under the mesh's own five
// degree threshold — so a contact whose closest feature is one of them comes back with its
// triangle's face normal rather than a phantom edge normal, and a wheel rolling over the seam
// between two flat triangles reads flat road. A kerb's top edge is active (ninety degrees) and
// keeps the diagonal push-out a round tyre meeting an edge really has. Back faces are ignored, as
// they are for the bodywork. **Results are per triangle**, so a face split in two reports twice
// with the same axis and depth; merging those is the caller's as well.
//
// The convex radius is Jolt's default 5 cm, lowered where a stated extent is smaller than it, and
// it rounds the cylinder's two rims — which is what a tyre's shoulders are. The radius and the
// half-width are the stated extents regardless: Jolt keeps the outer surface where it was told to
// and rounds inward.
//
// Slots are `index * maxContacts + n`, and `outCounts[index]` says how many of a cylinder's slots
// were written. A cylinder that touches nothing writes none.
void raceengineJoltCollideCylinders(const std::uint64_t handle, const double* centres, const double* axes,
                                    const double* radii, const double* halfWidths, const std::uint32_t count,
                                    const std::uint32_t maxContacts, std::uint32_t* outCounts, double* outAxes,
                                    double* outNormals, double* outDepths, double* outPoints,
                                    std::uint32_t* outSurfaces)
{
    auto* world = worldFor(handle);

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        outCounts[index] = 0;
    }

    if (world == nullptr || maxContacts == 0)
    {
        return;
    }

    JPH::CollideShapeSettings settings;
    settings.mBackFaceMode = JPH::EBackFaceMode::IgnoreBackFaces;
    settings.mMaxSeparationDistance = 0.0f;

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        const auto radius = static_cast<float>(radii[index]);
        const auto halfWidth = static_cast<float>(halfWidths[index]);
        if (!(radius > 0.0f) || !(halfWidth > 0.0f))
        {
            continue;
        }

        auto axis = JPH::Vec3(static_cast<float>(axes[index * 3 + 0]), static_cast<float>(axes[index * 3 + 1]),
                              static_cast<float>(axes[index * 3 + 2]));
        if (!(axis.LengthSq() > 0.0f))
        {
            continue;
        }
        axis = axis.Normalized();

        // Jolt's default convex radius, 5 cm, restated as a literal because the physics module's
        // `obstacleConvexRadius` has to be the same number and cannot see this file: the rule on
        // that side rebuilds this cylinder's support point from it. Jolt refuses a convex radius
        // larger than either extent, so a very small stated cylinder gets a correspondingly small
        // rounding rather than a refusal.
        const auto convexRadius = std::min({0.05f, radius, halfWidth});

        JPH::CylinderShape cylinder(halfWidth, radius, convexRadius);
        cylinder.SetEmbedded();

        const auto rotation = JPH::Quat::sFromTo(JPH::Vec3::sAxisY(), axis);
        const JPH::RVec3 position(static_cast<float>(centres[index * 3 + 0]),
                                  static_cast<float>(centres[index * 3 + 1]),
                                  static_cast<float>(centres[index * 3 + 2]));

        JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> collector;
        world->system.GetNarrowPhaseQuery().CollideShape(&cylinder, JPH::Vec3::sReplicate(1.0f),
                                                         JPH::RMat44::sRotationTranslation(rotation, position),
                                                         settings, JPH::RVec3::sZero(), collector);

        auto written = std::uint32_t{0};
        for (const auto& hit : collector.mHits)
        {
            if (written >= maxContacts)
            {
                break;
            }

            // Jolt's penetration axis points from the cylinder towards the road and is not
            // normalised. What is wanted is the direction the road pushes the tyre, which is the
            // other way — the same turn the bodywork's contacts take.
            const auto length = hit.mPenetrationAxis.Length();
            if (!(length > 0.0f))
            {
                continue;
            }

            const auto normal = -hit.mPenetrationAxis / length;

            // The triangle's own face normal at the contact point, beside the push-out — which are
            // the same direction for a face contact and not for an edge or vertex one. The rule on
            // the far side reads the difference: a contact against a triangle the bottom grid already
            // carries is the grid's whatever its edges do to the push-out.
            auto surface = std::uint32_t{0};
            auto faceNormal = normal;
            {
                JPH::BodyLockRead lock(world->system.GetBodyLockInterface(), hit.mBodyID2);
                if (lock.Succeeded())
                {
                    const auto& body = lock.GetBody();
                    const auto* material = body.GetShape()->GetMaterial(hit.mSubShapeID2);
                    surface = world->surfaceOf(material);
                    faceNormal = body.GetWorldSpaceSurfaceNormal(hit.mSubShapeID2, hit.mContactPointOn2);
                }
            }

            const auto slot = static_cast<std::size_t>(index) * maxContacts + written;

            outAxes[slot * 3 + 0] = static_cast<double>(normal.GetX());
            outAxes[slot * 3 + 1] = static_cast<double>(normal.GetY());
            outAxes[slot * 3 + 2] = static_cast<double>(normal.GetZ());

            outNormals[slot * 3 + 0] = static_cast<double>(faceNormal.GetX());
            outNormals[slot * 3 + 1] = static_cast<double>(faceNormal.GetY());
            outNormals[slot * 3 + 2] = static_cast<double>(faceNormal.GetZ());

            outPoints[slot * 3 + 0] = static_cast<double>(hit.mContactPointOn2.GetX());
            outPoints[slot * 3 + 1] = static_cast<double>(hit.mContactPointOn2.GetY());
            outPoints[slot * 3 + 2] = static_cast<double>(hit.mContactPointOn2.GetZ());

            outDepths[slot] = static_cast<double>(hit.mPenetrationDepth);
            outSurfaces[slot] = surface;

            written++;
        }

        outCounts[index] = written;
    }
}

// The dynamics of the props a manifold named, plus the anchor holding each one down.
//
// **A standing prop reports its real mass and its real inertia here, not zeros**, and that is the
// change that stopped the bins flying. What a prop weighs is what decides whether its anchor gives
// way, and it is what the collision is worth once it has — both of which have to be known *before*
// the car's solver resolves anything against it. Presenting a standing prop as immovable is a
// decision the module above makes with these numbers in hand; it is not something to bake in here.
//
// `Body::GetInverseInertia` asserts the body is dynamic, so a standing prop's tensor comes from the
// copy taken when it was created rather than from Jolt. Its velocity is zero because it is static.
//
// Batched, for the reason the ray cast is: the alternative is a body lock per contact point, and the
// props a car's bodywork touches in one tick arrive together.
void raceengineJoltReadPropDynamics(const std::uint64_t handle, const std::uint32_t* propIndices,
                                    const std::uint32_t count, double* outInverseMasses, double* outInverseInertias,
                                    double* outCentres, double* outLinearVelocities, double* outAngularVelocities,
                                    double* outBreakForces, double* outBreakTorques, unsigned char* outAnchored)
{
    // Zeroed first, so that a dead handle and an unknown index are both the immovable world without
    // two paths saying so.
    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        outInverseMasses[index] = 0.0;
        outBreakForces[index] = 0.0;
        outBreakTorques[index] = 0.0;
        outAnchored[index] = 0;

        for (auto element = std::uint32_t{0}; element < 9; element++)
        {
            outInverseInertias[index * 9 + element] = 0.0;
        }

        for (auto element = std::uint32_t{0}; element < 3; element++)
        {
            outCentres[index * 3 + element] = 0.0;
            outLinearVelocities[index * 3 + element] = 0.0;
            outAngularVelocities[index * 3 + element] = 0.0;
        }
    }

    auto* world = worldFor(handle);
    if (world == nullptr)
    {
        return;
    }

    const auto& bodies = world->system.GetBodyInterface();

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        const auto which = propIndices[index];
        if (which >= world->props.size())
        {
            continue;
        }

        const auto& prop = world->props[which];

        // The **centre of mass** and not the body origin. It is the frame the inertia tensor is
        // stated in and the point every impulse arm is taken about. `raceengineJoltReadFreeProps`
        // deliberately reads the other one, because what draws a prop wants the body frame its
        // vertices are stated in — the two differ by the hull centroid's offset.
        const auto centre = JPH::Vec3(bodies.GetCenterOfMassPosition(prop.body));

        outCentres[index * 3 + 0] = static_cast<double>(centre.GetX());
        outCentres[index * 3 + 1] = static_cast<double>(centre.GetY());
        outCentres[index * 3 + 2] = static_cast<double>(centre.GetZ());

        outInverseMasses[index] = prop.mass > 0.0 ? 1.0 / prop.mass : 0.0;
        outBreakForces[index] = prop.breakForce;
        outBreakTorques[index] = prop.breakTorque;
        outAnchored[index] = prop.free ? 0 : 1;

        // World frame, and Jolt hands back a 4x4 whose upper 3x3 is the tensor. Written row major,
        // which is the convention the whole of this bridge's matrices already use.
        const auto inverse = prop.free ? bodies.GetInverseInertia(prop.body) : prop.inverseInertia;
        for (auto row = std::uint32_t{0}; row < 3; row++)
        {
            for (auto column = std::uint32_t{0}; column < 3; column++)
            {
                outInverseInertias[index * 9 + row * 3 + column] = static_cast<double>(inverse(row, column));
            }
        }

        if (!prop.free)
        {
            continue;
        }

        const auto linear = bodies.GetLinearVelocity(prop.body);
        const auto angular = bodies.GetAngularVelocity(prop.body);

        outLinearVelocities[index * 3 + 0] = static_cast<double>(linear.GetX());
        outLinearVelocities[index * 3 + 1] = static_cast<double>(linear.GetY());
        outLinearVelocities[index * 3 + 2] = static_cast<double>(linear.GetZ());
        outAngularVelocities[index * 3 + 0] = static_cast<double>(angular.GetX());
        outAngularVelocities[index * 3 + 1] = static_cast<double>(angular.GetY());
        outAngularVelocities[index * 3 + 2] = static_cast<double>(angular.GetZ());
    }
}

// Let go of the props the car's contact solver decided were coming loose.
//
// **The decision is not made here and there is no threshold in this function**, which is the whole
// point of it. A prop has to be free *before* the solver resolves the collision that freed it, or
// the car's momentum goes into an anchor that is about to break and the car is stopped dead by a
// bin. So `releaseBrokenProps` decides from the approach load, on the tick, before `resolveContacts`
// runs — and this only carries the decision out. One decision, in one place, is also what stops this
// and the module above ever disagreeing about a prop at exactly its threshold.
void raceengineJoltReleaseProps(const std::uint64_t handle, const std::uint32_t* propIndices, const std::uint32_t count)
{
    auto* world = worldFor(handle);
    if (world == nullptr)
    {
        return;
    }

    auto& bodies = world->system.GetBodyInterface();

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        const auto which = propIndices[index];
        if (which >= world->props.size() || world->props[which].free)
        {
            continue;
        }

        auto& prop = world->props[which];

        bodies.SetMotionType(prop.body, JPH::EMotionType::Dynamic, JPH::EActivation::Activate);
        bodies.SetObjectLayer(prop.body, Layers::moving);

        prop.free = true;
        world->freeProps++;
    }
}

// One prop, hit hard enough or not.
//
// The impulse and its moment are the *car's* — computed by this engine's own sequential-impulse
// solver, which is why there is no constraint here to read a lambda off. Divided by the tick they
// are a force and a torque, and those are what the exporter's thresholds are quoted in.
//
// A prop already free takes the impulse and stays free; a prop that has just broken is switched to
// dynamic, moved into the layer that collides with the ground, woken, and handed the same impulse at
// the point it was struck. The switch needs `mAllowDynamicOrKinematic` to have been set when the
// body was created, which it was.
void raceengineJoltBreakProps(const std::uint64_t handle, const std::uint32_t* propIndices, const double* impulses,
                              const double* points, const std::uint32_t count, const double deltaTime)
{
    auto* world = worldFor(handle);
    if (world == nullptr || deltaTime <= 0.0)
    {
        return;
    }

    auto& bodies = world->system.GetBodyInterface();

    // Accumulated per prop first and tested once, because a flat hit lands on several contact points
    // at once and each of them alone is under the threshold. Summing the *moments* rather than the
    // impulses is what makes a glancing blow on the top of a post break it: two impulses that nearly
    // cancel as forces still add as a couple.
    //
    // A linear scan over the props touched this tick rather than a map: a car's manifold caps at 32
    // points and the props among them are a handful.
    struct Load
    {
        std::uint32_t prop = noProp;
        JPH::Vec3 impulse = JPH::Vec3::sZero();
        JPH::Vec3 moment = JPH::Vec3::sZero();
        JPH::Vec3 point = JPH::Vec3::sZero();
    };

    auto loads = std::vector<Load>();

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        const auto which = propIndices[index];
        if (which >= world->props.size())
        {
            continue;
        }

        const auto impulse =
            JPH::Vec3(static_cast<float>(impulses[index * 3 + 0]), static_cast<float>(impulses[index * 3 + 1]),
                      static_cast<float>(impulses[index * 3 + 2]));
        const auto point =
            JPH::Vec3(static_cast<float>(points[index * 3 + 0]), static_cast<float>(points[index * 3 + 1]),
                      static_cast<float>(points[index * 3 + 2]));

        // The moment is taken about the body's own centre of mass, which is the only frame the
        // exporter's break torque means anything in.
        const auto centre = JPH::Vec3(bodies.GetCenterOfMassPosition(world->props[which].body));

        auto* found = static_cast<Load*>(nullptr);
        for (auto& load : loads)
        {
            if (load.prop == which)
            {
                found = &load;
                break;
            }
        }

        if (found == nullptr)
        {
            loads.push_back(Load{.prop = which, .point = point});
            found = &loads.back();
        }

        found->impulse += impulse;
        found->moment += (point - centre).Cross(impulse);
    }

    for (const auto& load : loads)
    {
        auto& prop = world->props[load.prop];

        // **A prop that is already loose is not pushed here, and this used to push it every tick.**
        // Its momentum now arrives through the contact solver's two-body exchange, applied as a
        // velocity change just before the world steps; adding the same collision again as an impulse
        // would count it twice. What is left is the break test and the *first* shove — the one a body
        // still anchored cannot get any other way, because a standing prop reports an infinite
        // effective mass to the solver and is therefore given nothing by it.
        if (prop.free)
        {
            continue;
        }

        const auto force = static_cast<double>(load.impulse.Length()) / deltaTime;
        const auto torque = static_cast<double>(load.moment.Length()) / deltaTime;

        if (force < prop.breakForce && torque < prop.breakTorque)
        {
            continue;
        }

        bodies.SetMotionType(prop.body, JPH::EMotionType::Dynamic, JPH::EActivation::Activate);
        bodies.SetObjectLayer(prop.body, Layers::moving);

        prop.free = true;
        world->freeProps++;

        bodies.AddImpulse(prop.body, load.impulse, JPH::RVec3(load.point));
    }
}

// The velocity changes the car's own contact solver worked out for the loose props it resolved
// against, applied just before the world is stepped — which is exactly an impulse at the start of
// the step.
//
// **Never called on a body that is not free**: a static body's motion properties are not there to be
// written and Jolt asserts. The guard is here rather than trusted to the caller because the caller
// cannot see a break that happened after its manifold was built.
void raceengineJoltAddPropVelocities(const std::uint64_t handle, const std::uint32_t* propIndices, const double* linear,
                                     const double* angular, const std::uint32_t count)
{
    auto* world = worldFor(handle);
    if (world == nullptr)
    {
        return;
    }

    auto& bodies = world->system.GetBodyInterface();

    for (auto index = std::uint32_t{0}; index < count; index++)
    {
        const auto which = propIndices[index];
        if (which >= world->props.size() || !world->props[which].free)
        {
            continue;
        }

        bodies.AddLinearAndAngularVelocity(
            world->props[which].body,
            JPH::Vec3(static_cast<float>(linear[index * 3 + 0]), static_cast<float>(linear[index * 3 + 1]),
                      static_cast<float>(linear[index * 3 + 2])),
            JPH::Vec3(static_cast<float>(angular[index * 3 + 0]), static_cast<float>(angular[index * 3 + 1]),
                      static_cast<float>(angular[index * 3 + 2])));
    }
}

// Integrate whatever is loose.
//
// **Skipped outright while nothing is**, which is the ordinary case and is why three and a half
// thousand props cost nothing: a world whose every body is static has no active set to walk, and the
// cheapest way to walk none of it is not to call. One collision step per tick, at 360 Hz, which is
// four times the rate Jolt's own defaults assume and well inside its stability range.
void raceengineJoltStepWorld(const std::uint64_t handle, const double deltaTime)
{
    auto* world = worldFor(handle);
    if (world == nullptr || world->freeProps == 0 || deltaTime <= 0.0)
    {
        return;
    }

    world->system.Update(static_cast<float>(deltaTime), 1, &world->temporaries, world->jobs.get());
}

// Where the loose props are now, for whoever is drawing them. Writes at most `limit` of them and
// returns how many it wrote; the orientation is w, x, y, z, which is how glm writes one.
std::uint32_t raceengineJoltReadFreeProps(const std::uint64_t handle, const std::uint32_t limit,
                                          std::uint32_t* outProps, double* outPositions, double* outOrientations)
{
    auto* world = worldFor(handle);
    if (world == nullptr || world->freeProps == 0 || limit == 0)
    {
        return 0;
    }

    const auto& bodies = world->system.GetBodyInterface();

    auto written = std::uint32_t{0};
    for (auto index = std::size_t{0}; index < world->props.size() && written < limit; index++)
    {
        if (!world->props[index].free)
        {
            continue;
        }

        // The **body's** transform and not its centre-of-mass frame. The two differ by whatever
        // offset the hull's own centroid has from the origin the exporter placed the body at, and
        // the vertices anything draws with are stated in the body frame — so a picture built from
        // the centre-of-mass transform is a prop that jumps by that offset the moment it comes free.
        const auto position = bodies.GetPosition(world->props[index].body);
        const auto rotation = bodies.GetRotation(world->props[index].body);

        outProps[written] = static_cast<std::uint32_t>(index);
        outPositions[written * 3 + 0] = static_cast<double>(position.GetX());
        outPositions[written * 3 + 1] = static_cast<double>(position.GetY());
        outPositions[written * 3 + 2] = static_cast<double>(position.GetZ());
        outOrientations[written * 4 + 0] = static_cast<double>(rotation.GetW());
        outOrientations[written * 4 + 1] = static_cast<double>(rotation.GetX());
        outOrientations[written * 4 + 2] = static_cast<double>(rotation.GetY());
        outOrientations[written * 4 + 3] = static_cast<double>(rotation.GetZ());

        written++;
    }

    return written;
}
