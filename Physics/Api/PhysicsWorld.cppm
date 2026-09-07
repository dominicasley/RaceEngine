module;

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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
                                                     const double* hullPointData, const std::uint32_t* hullPointCounts,
                                                     const double* hullOrigins, const std::uint32_t* hullSurfaces,
                                                     std::uint32_t hullCount, std::string& reason);
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

// One convex collider standing beside the drivable surface: a building, a bench, a lamp column.
//
// **This is not a surface a car drives on and it is not carried by `SurfaceMesh`**, which is the
// distinction the two types exist to keep. A surface mesh is triangles a wheel ray samples; a
// collider is a solid a body is stopped by, and the reason it is convex rather than a triangle soup
// is that a soup has no inside — a car that reached the middle of a facade would find nothing to
// push it out.
//
// `points` are stated in the collider's own frame and `origin` is where that frame stands in the
// world. Kept apart rather than baked together on purpose: Jolt is single precision, a city is
// kilometres across, and a hull whose vertices were already in world coordinates would be resolving
// centimetres against thousands of metres. Interior points are allowed — the hull is built from
// whatever is handed over, so an exporter that unrolls a hull's triangles per face costs load time
// and nothing else.
//
// `surface` indexes the same material table `SurfaceMesh::materials` states, because a ray that hits
// a building has to report something true and the fallback is index 0, which on an authored track is
// tarmac.
export struct ConvexCollider
{
    std::vector<glm::dvec3> points;
    glm::dvec3 origin{0.0};
    std::uint32_t surface = 0;
};

// A prop that stands until something hits it hard enough, and is loose debris afterwards.
//
// Everything here is derived rather than authored — Assetto Corsa states no mass for a lamp column
// and no collision for one either. `mass` is an *effective* density times the hull's volume, and the
// two thresholds are the exporter's own invention. They are a starting point for the seat, not a
// source, and the place to retune them is the exporter rather than this struct.
//
// **The inertia is handed over and must not be derived.** A lamp column's hull is a solid prism and
// the column is a hollow tube, so a uniform-density tensor is several times too large in the two
// large terms and the post falls like a felled tree in syrup. It is stated about the centre of mass
// with axes parallel to the world, and the body's origin *is* its centre of mass, so it goes in at
// the origin of the local frame with no parallel-axis shift.
export struct BreakableProp
{
    // The hulls this prop is made of. They share one origin — the body's — and each hull's points
    // are stated in that frame; `ConvexCollider::origin` is read from the first of them and the rest
    // are expected to agree, because a prop is one body however many convex pieces it takes.
    std::vector<ConvexCollider> hulls;
    double mass = 0.0;
    glm::dmat3 inertia{1.0};
    // Newtons and newton-metres at which the base gives way. Either one alone releases it.
    double breakForce = 0.0;
    double breakTorque = 0.0;
};

// What a contact reports when it did not land on a prop, which is every contact with the road, a
// building, and a prop too big to shift.
export inline constexpr std::uint32_t noProp = 0xffffffffu;

// One impulse the car's own contact solver applied, and where it applied it.
//
// **Stated as plain vectors rather than as a `ContactManifold`, and the reason is the module graph
// rather than taste**: `:Contact` imports this partition, so this partition may not name a contact.
// It is also the truer boundary — what a prop needs to know is that something pushed it this hard
// here, and not that a car's bodywork was what did it.
//
// Several of these may name the same prop in one call. They are accumulated on the far side and the
// threshold is tested once against the total, which is what makes a flat hit on four contact points
// break a post that one of those points alone would not.
export struct PropImpulse
{
    std::uint32_t prop = noProp;
    // Newton-seconds, world frame, and **what the prop received** — not what the car did. The two
    // are the halves of one pair and differ by a sign, and the caller owns that sign because the
    // caller is the only side that knows which way its own contact normals point.
    glm::dvec3 impulse{0.0};
    // Where it acted, world frame. The moment is taken about the body's own centre of mass on the
    // far side, which is the only place that centre is known.
    glm::dvec3 point{0.0};
};

// A velocity change the car's own contact solver worked out for one loose prop.
//
// **Not an impulse, and the distinction is the whole reason this type exists beside `PropImpulse`.**
// The solver ran eight iterations against this body, tracking its velocity locally the whole way and
// never re-reading it from the world; what comes out is the change it converged on, and dividing it
// back into a force to hand over as an impulse would be undoing arithmetic that has already been
// done exactly. Applied before the world is stepped, which is exactly an impulse at the start of the
// step.
//
// A prop that is still anchored never appears here: the solver is told its inverse mass is zero, so
// the change it computes for one is exactly zero. The impulse that first breaks it is `PropImpulse`'s
// job and stays there.
export struct PropVelocity
{
    std::uint32_t prop = noProp;
    // Metres per second and radians per second, world frame.
    glm::dvec3 linear{0.0};
    glm::dvec3 angular{0.0};
};

// Where one loose prop is now.
export struct PropTransform
{
    std::uint32_t prop = noProp;
    glm::dvec3 position{0.0};
    glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
};

// Jolt, used for exactly what it is better at than anything worth writing: a broadphase and a BVH
// to ask the track questions against. It does not integrate the vehicle, resolve its contacts, or
// own a solver anywhere near the contact patch — the manifolds it generates are read back and
// resolved by us, and vehicle bodies are kinematic transforms we write.
//
// Move-only and RAII over the backend's handle, because a world is a large thing to copy by
// accident and the alternative to owning it here is an explicit destroy call that a failing test
// skips.
//
// **Every member is defined in Physics/Impl/PhysicsWorldImpl.cpp, not here.** A definition written
// inside the class is part of this partition's BMI, so editing one rebuilds every importer of
// `raceengine` — 117 ninja edges and ~87 s. Declared here and defined in an implementation unit,
// the same edit is one object file. The accessors below are the only ones that lose an inline by
// it, and both return a word from a member that is read once per wheel per tick. Full account:
// docs/build-times.md.
export class PhysicsWorld
{
public:
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    PhysicsWorld(PhysicsWorld&& other) noexcept;
    PhysicsWorld& operator=(PhysicsWorld&& other) noexcept;
    ~PhysicsWorld();

    // The surface mesh becomes one static body with per-triangle materials, and each collider becomes
    // one static body of its own. Requires `bringUpJolt` to have run: Jolt's shape types register
    // into a factory that does not exist before it.
    //
    // **The colliders are taken here rather than added afterwards, and that is the invariant rather
    // than the convenience.** This world is immutable once it exists — nothing steps it, nothing adds
    // to it, and that is what makes it safe to read from the simulation thread while the main thread
    // draws. An `addCollider` would move that guarantee from the type to whoever remembered to stop
    // calling it. A track with no collider export passes nothing and gets exactly the world it got
    // before this parameter existed.
    [[nodiscard]] static std::expected<PhysicsWorld, std::string> create(const SurfaceMesh& mesh,
                                                                         std::span<const ConvexCollider> colliders = {},
                                                                         std::span<const BreakableProp> props = {});

    // Hand the solver's own contact impulses back to the world, so that whatever they landed on can
    // give way.
    //
    // **This is the whole of the breakable-anchor mechanism and there is no constraint in it.** Jolt
    // has no breakable joint, and the usual answer — a fixed constraint per prop whose accumulated
    // lambda is read every tick — would be three and a half thousand constraints solving a problem
    // this engine has already solved: it computes the car's contact impulses itself, on purpose, and
    // an impulse over a tick *is* the force a break threshold is quoted in.
    //
    // Called after `resolveContacts` and before `step`, from the thread that owns this world.
    // Entries naming `noProp` are skipped, which is nearly all of them.
    void applyPropImpulses(std::span<const PropImpulse> impulses, double deltaTime);

    // Let go of the props the car's contact solver decided were coming loose.
    //
    // **There is no threshold in this call.** `releaseBrokenProps` makes that decision on the tick,
    // from the approach load, *before* `resolveContacts` runs — because a prop that is still anchored
    // when the collision is resolved takes the car's whole momentum into the ground, and the car is
    // then stopped dead by a twenty-kilogram bin. This only carries the decision out, and keeping
    // the test in exactly one place is what stops the two sides disagreeing about a prop sitting on
    // its threshold.
    //
    // Called before `applyPropVelocities`, so that the velocity has a dynamic body to land on.
    void releaseProps(std::span<const std::uint32_t> props);

    // Hand the solver's own two-body velocity changes back to the loose props they were computed
    // for.
    //
    // Called after `applyPropImpulses` and before `step`, from the thread that owns this world, and
    // the ordering is the whole of it: a prop that broke on this tick was immovable while the car
    // resolved against it, so its motion comes from the break impulse alone and its entry here is
    // zero. Every tick after that it is the other way round.
    //
    // Entries naming `noProp`, and props that are not loose, are skipped — a static body has no
    // motion properties for a velocity to be added to.
    void applyPropVelocities(std::span<const PropVelocity> velocities);

    // Integrate whatever is loose. Does nothing at all while nothing is, which is the ordinary case:
    // a world of static bodies has no active set, and the cheapest way to walk none of it is not to
    // call into it. **This is the one method on this class that mutates the world**, and it is why
    // the world lives inside `Simulation` rather than beside it.
    void step(double deltaTime);

    // Where the loose props are, for the thread that draws them. Fills `into` and leaves it empty
    // while nothing has broken.
    void freeProps(std::vector<PropTransform>& into) const;

    // What each surface index is worth to a tire, in the order the mesh declared them.
    //
    // It lives with the world and not with the caller because the `surface` a `SurfaceHit` reports
    // is an index into *this* mesh's table and is meaningless against any other. The vehicle used to
    // aggregate its contact patches against `defaultSurfaceMaterials()` regardless of what world it
    // was querying, which is exact for a generated proving ground — the generator emits that table —
    // and silently wrong for anything else: an authored circuit states nine surfaces, and every
    // index past the third fell back on the first, so a wheel in the gravel gripped like tarmac.
    [[nodiscard]] const std::vector<SurfaceMaterial>& materials() const;

    // Batched, and that is the interface rather than a convenience over a single cast: a wheel's
    // contact patch is a grid of samples taken at one instant, four wheels of them per tick, and
    // asking one at a time would walk the same broadphase nodes for every one of them.
    //
    // `directions` are expected normalised; `maxDistance` is how far along each to look. A miss
    // reports `hit == false` with the distance left at maxDistance, so a caller aggregating a patch
    // can weight by hit without branching into a separate path.
    void castRays(const std::vector<glm::dvec3>& origins, const std::vector<glm::dvec3>& directions,
                  const double maxDistance, std::vector<SurfaceHit>& results) const;

    // The backend's own identifier for this world. Exposed because the contact bridge is a second
    // entry point into the same world and has to name it; nothing outside this module's partitions
    // has any use for it.
    [[nodiscard]] std::uint64_t handle() const;

private:
    PhysicsWorld(const std::uint64_t created, std::vector<SurfaceMaterial> surfaces);

    std::uint64_t world = 0;
    std::vector<SurfaceMaterial> surfaceMaterials;
};

// The vertex and direction arrays cross the bridge as bare `const double*` over the whole vector,
// so their element type has to be exactly three packed doubles. glm gives no such guarantee in its
// own words; this is where the assumption is checked rather than discovered.
static_assert(sizeof(glm::dvec3) == 3 * sizeof(double), "the ray bridge passes dvec3 arrays as flat doubles");

} // namespace raceengine
