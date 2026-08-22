#include <algorithm>
#include <cmath>
#include <expected>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::sampleProvingGround;
using raceengine::SurfaceHit;
using raceengine::SurfaceKind;
using raceengine::tearDownJolt;

namespace
{

// Jolt's factory and type registry are process-wide, so a case that stands them up has to take them
// down again whatever it does in between — including failing an assertion, which is why this is a
// destructor rather than a call at the end of the case.
struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

// Purpose-built and small. The default proving ground is 200 x 40 m, which at a pitch fine enough
// to resolve a 0.3 m kerb chamfer would be well over a million triangles to build for one assertion.
ProvingGroundDescriptor patch(const FeatureKind kind, const double length, const double width, const double cellSize)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = cellSize;
    descriptor.features = {Feature{.kind = kind, .from = 0.0, .to = length}};

    return descriptor;
}

std::vector<SurfaceHit> probe(const PhysicsWorld& world, const std::vector<glm::dvec3>& from)
{
    auto directions = std::vector<glm::dvec3>(from.size(), glm::dvec3(0.0, -1.0, 0.0));
    auto results = std::vector<SurfaceHit>{};
    world.castRays(from, directions, 50.0, results);

    return results;
}

} // namespace

TEST_CASE("the world answers where the ground is and what it is made of", "[physics][world]")
{
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 40.0;
    descriptor.width = 20.0;
    descriptor.cellSize = 0.25;
    descriptor.features = {Feature{.kind = FeatureKind::SurfaceBoundary, .from = 10.0, .to = 30.0}};

    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    const auto hits = probe(world.value(), {
                                               glm::dvec3(-5.0, 10.0, 5.0),  // flat tarmac, before the boundary
                                               glm::dvec3(-5.0, 10.0, 20.0), // tarmac side of the boundary
                                               glm::dvec3(5.0, 10.0, 20.0),  // grass side
                                               glm::dvec3(0.0, 10.0, 300.0), // past the end of the ground
                                           });

    REQUIRE(hits.size() == 4);

    SECTION("a ray onto flat ground lands on it, pointing up")
    {
        REQUIRE(hits[0].hit);
        REQUIRE(hits[0].point.y == Catch::Approx(0.0).margin(1e-5));
        REQUIRE(hits[0].normal.y == Catch::Approx(1.0).margin(1e-5));
        REQUIRE(hits[0].distance == Catch::Approx(10.0).margin(1e-4));
        REQUIRE(hits[0].surface == static_cast<std::uint32_t>(SurfaceKind::Tarmac));
    }

    SECTION("the two sides of a boundary come back as different surfaces")
    {
        REQUIRE(hits[1].hit);
        REQUIRE(hits[2].hit);
        REQUIRE(hits[1].surface == static_cast<std::uint32_t>(SurfaceKind::Tarmac));
        REQUIRE(hits[2].surface == static_cast<std::uint32_t>(SurfaceKind::Grass));

        // And the grass sits below the tarmac, so straddling is a load change as well as a grip one.
        REQUIRE(hits[2].point.y < hits[1].point.y);
        REQUIRE(hits[2].point.y == Catch::Approx(-descriptor.boundaryLip).margin(1e-5));
    }

    SECTION("a ray with nothing under it reports a miss rather than a hit at infinity")
    {
        REQUIRE_FALSE(hits[3].hit);
        // A miss still carries usable values, so a caller aggregating a patch weights by `hit`
        // rather than branching into a separate path for it.
        REQUIRE(hits[3].distance == Catch::Approx(50.0));
        REQUIRE(hits[3].normal.y == Catch::Approx(1.0));
    }
}

TEST_CASE("a contact patch straddling a kerb sees the kerb", "[physics][world]")
{
    // The case the brief singles out, and the one single-ray contact gets wrong: the wheel is half
    // on the chamfer. A ray at the wheel centre reports flat ground at one height and one surface;
    // the grid reports a spread of heights and two surfaces, which is what an aggregate can turn
    // into a tilted plane and an offset patch centre.
    const JoltGuard jolt;

    const auto descriptor = patch(FeatureKind::Kerb, 20.0, 10.0, 0.05);
    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    // A 0.2 m patch centred on the kerb's inner *edge*, three across by three along — so half of it
    // is on flat tarmac and half is climbing the chamfer. Centred on the middle of the chamfer
    // instead, every sample is on the kerb and the case proves nothing.
    const auto centreX = descriptor.kerbInnerEdge;
    auto origins = std::vector<glm::dvec3>{};
    for (auto along = -1; along <= 1; along++)
    {
        for (auto across = -1; across <= 1; across++)
        {
            origins.emplace_back(centreX + static_cast<double>(across) * 0.1, 5.0,
                                 10.0 + static_cast<double>(along) * 0.1);
        }
    }

    const auto hits = probe(world.value(), origins);
    REQUIRE(hits.size() == 9);

    auto lowest = 1e30;
    auto highest = -1e30;
    auto surfaces = std::set<std::uint32_t>{};

    for (const auto& hit : hits)
    {
        REQUIRE(hit.hit);
        lowest = std::min(lowest, hit.point.y);
        highest = std::max(highest, hit.point.y);
        surfaces.insert(hit.surface);
    }

    // The patch spans real height across its width — a single ray would have reported one number.
    // The outboard column is 0.1 m up the chamfer; the inboard two are on flat ground.
    REQUIRE(lowest == Catch::Approx(0.0).margin(1e-5));
    REQUIRE(highest == Catch::Approx(0.1 / descriptor.kerbChamfer * descriptor.kerbHeight).margin(2e-3));

    // And it spans two surfaces, which is what makes blended grip possible rather than a snap.
    REQUIRE(surfaces.size() == 2);
    REQUIRE(surfaces.contains(static_cast<std::uint32_t>(SurfaceKind::Tarmac)));
    REQUIRE(surfaces.contains(static_cast<std::uint32_t>(SurfaceKind::Kerb)));

    // The chamfer is a plane, so its normal is tilted across the car and not along it — the moment
    // this produces about the roll axis is the point of sampling the patch at all. Asked well
    // inside the chamfer rather than at the patch's centre, which sits exactly on the edge where
    // either answer is defensible.
    const auto chamfer =
        probe(world.value(), {glm::dvec3(descriptor.kerbInnerEdge + 0.5 * descriptor.kerbChamfer, 5.0, 10.0)});
    REQUIRE(chamfer[0].hit);
    REQUIRE(chamfer[0].normal.y < 1.0);
    REQUIRE(chamfer[0].normal.x ==
            Catch::Approx(-std::sin(std::atan(descriptor.kerbHeight / descriptor.kerbChamfer))).margin(1e-3));
    REQUIRE(chamfer[0].normal.z == Catch::Approx(0.0).margin(1e-5));
}

TEST_CASE("a banked surface reports a banked normal", "[physics][world]")
{
    const JoltGuard jolt;

    const auto descriptor = patch(FeatureKind::Camber, 30.0, 20.0, 0.25);
    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    // Halfway along the band the easing is at its peak, which is where the bank is the full angle.
    const auto hits = probe(world.value(), {glm::dvec3(4.0, 10.0, 15.0), glm::dvec3(-4.0, 10.0, 15.0)});

    REQUIRE(hits[0].hit);
    REQUIRE(hits[1].hit);

    // The two sides of the centreline sit opposite ways about it, and each stands where the sampler
    // said it would.
    REQUIRE(hits[0].point.y == Catch::Approx(sampleProvingGround(descriptor, 4.0, 15.0).height).margin(2e-3));
    REQUIRE(hits[1].point.y == Catch::Approx(-hits[0].point.y).margin(2e-3));

    // The *heights* mirror but the normals do not, and that distinction is the feature: a bank is
    // one tilted plane through the centreline, not a ridge, so both sides lean the same way and a
    // car crossing the centreline feels no discontinuity. Asserting mirrored normals here would be
    // asserting a crease down the middle of the road.
    const auto bank = std::sin(descriptor.camberAngle);
    REQUIRE(hits[0].normal.x == Catch::Approx(-bank).margin(2e-3));
    REQUIRE(hits[1].normal.x == Catch::Approx(-bank).margin(2e-3));
    REQUIRE(hits[0].normal.y > 0.9);

    // Sampled at the peak of the easing, where the bank is not changing along z, so the tilt is
    // purely across the ground.
    REQUIRE(hits[0].normal.z == Catch::Approx(0.0).margin(5e-3));
}

TEST_CASE("scene queries are deterministic", "[physics][world][determinism]")
{
    // Criterion 12 reaches the queries too: a contact patch is sampled every tick, and a broadphase
    // that answered differently between runs would put the determinism of everything above it out
    // of reach however careful the integrator was.
    const JoltGuard jolt;

    const auto descriptor = patch(FeatureKind::Kerb, 20.0, 10.0, 0.1);
    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    auto origins = std::vector<glm::dvec3>{};
    for (auto index = 0; index < 64; index++)
    {
        const auto offset = static_cast<double>(index) * 0.05;
        origins.emplace_back(-2.0 + offset, 5.0, 3.0 + offset);
    }

    const auto first = PhysicsWorld::create(mesh.value());
    REQUIRE(first.has_value());
    const auto firstHits = probe(first.value(), origins);

    // A second world over the same mesh, so the broadphase is rebuilt rather than reused.
    const auto second = PhysicsWorld::create(mesh.value());
    REQUIRE(second.has_value());
    const auto secondHits = probe(second.value(), origins);

    REQUIRE(firstHits.size() == secondHits.size());
    for (auto index = std::size_t{0}; index < firstHits.size(); index++)
    {
        REQUIRE(firstHits[index].hit == secondHits[index].hit);
        REQUIRE(firstHits[index].point == secondHits[index].point);
        REQUIRE(firstHits[index].normal == secondHits[index].normal);
        REQUIRE(firstHits[index].surface == secondHits[index].surface);
    }
}

TEST_CASE("a ray fired straight down at solid ground always finds it", "[physics][world][determinism]")
{
    // **A ray can miss a mesh it is pointed straight at, and it cost a wheel.** This is the
    // regression for that: it fires at the coordinates where one did.
    //
    // Found 2026-08-22 while dissecting why `the imported car crosses a kerb continuously` failed. One
    // ray of 31,851 over a kerb crossing came back with no hit while its two neighbours in the same
    // row of the contact grid reported 19.5 and 18.8 mm of road under them. That single lost ray was
    // the whole of the failure: 25.9 mm of patch-centre movement in one tick and 665 N of load, out
    // and back the next tick, against a distribution whose next-worst tick was 2.4 mm.
    //
    // Characterised before it was repaired. The misses lie in a band **35 micrometres wide in z and
    // indifferent to x**, sitting *beside* a shared triangle edge rather than on it — a ray cast
    // exactly on the row line hits, and none of 200 row lines swept loses one — and it does not depend
    // on the ray's length. That is the float32 edge test inside the mesh shape losing its sign over a
    // few units in the last place at coordinates of tens of metres. The repair is a retry from a
    // millimetre away along two perpendicular directions, in `raceengineJoltCastRays`.
    //
    // The whole band is swept rather than the one origin, so this does not become a test that passes
    // because the degeneracy moved a micrometre.
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 30.0;
    descriptor.cellSize = 0.10;
    descriptor.kerbInnerEdge = 0.60;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 40.0, .to = 160.0}};

    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());
    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    // The lost ray's own origin, and the band it sits in, at a micrometre's resolution.
    auto origins = std::vector<glm::dvec3>{};
    for (auto step = 0; step <= 200; step++)
    {
        origins.emplace_back(2.3742377382, 1.0298479792, 63.4999 + 1e-6 * static_cast<double>(step));
    }

    auto directions = std::vector<glm::dvec3>(origins.size(), glm::dvec3(0.0, -1.0, 0.0));
    auto hits = std::vector<SurfaceHit>{};
    world->castRays(origins, directions, 2.0, hits);

    REQUIRE(hits.size() == origins.size());
    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        CAPTURE(index, origins[index].z);
        REQUIRE(hits[index].hit);

        // And it found the road rather than something a metre away that a wild retry might have
        // reached. The kerb's flat top is 50 mm here and the ground either side is zero.
        REQUIRE(hits[index].point.y == Catch::Approx(descriptor.kerbHeight).margin(1e-3));
    }
}

TEST_CASE("a world that cannot be built is reported", "[physics][world]")
{
    const JoltGuard jolt;

    auto empty = raceengine::SurfaceMesh{};
    empty.vertices.emplace_back(0.0, 0.0, 0.0);
    empty.materials = raceengine::defaultSurfaceMaterials();

    const auto world = PhysicsWorld::create(empty);
    REQUIRE_FALSE(world.has_value());
    REQUIRE(world.error().find("vertices and triangles") != std::string::npos);
}

TEST_CASE("a world with no surface table is refused", "[physics][world]")
{
    const JoltGuard jolt;

    auto mesh = generateProvingGround(patch(FeatureKind::Kerb, 10.0, 10.0, 1.0));
    REQUIRE(mesh.has_value());

    mesh->materials.clear();

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE_FALSE(world.has_value());
    REQUIRE(world.error().find("surface material") != std::string::npos);
}

// The table a world hands back has to be the one its mesh declared, whatever length that is. An
// authored circuit states nine surfaces against the generator's three, and everything downstream
// indexes it by the surface a hit reports.
TEST_CASE("a world carries the surface table its mesh declared", "[physics][world]")
{
    const JoltGuard jolt;

    auto mesh = generateProvingGround(patch(FeatureKind::Kerb, 10.0, 10.0, 1.0));
    REQUIRE(mesh.has_value());

    mesh->materials.push_back(raceengine::SurfaceMaterial{
        .gripMultiplier = 0.37, .bumpiness = 0.011, .damping = 0.05, .kind = SurfaceKind::Gravel});

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    REQUIRE(world->materials().size() == mesh->materials.size());
    REQUIRE(world->materials().back().gripMultiplier == Catch::Approx(0.37));
    REQUIRE(world->materials().back().bumpiness == Catch::Approx(0.011));
    REQUIRE(world->materials().back().damping == Catch::Approx(0.05));
    REQUIRE(world->materials().back().kind == SurfaceKind::Gravel);
}

// The failure this pins is silent and does not look like a table problem at all.
//
// `stepVehicle` aggregated every contact patch against `defaultSurfaceMaterials()`, which has three
// entries, and `aggregateContactPatch` answers an out-of-range surface with `materials.front()`. So
// a car on a circuit whose gravel trap is surface 6 gripped like tarmac, with nothing anywhere
// reporting anything — the car simply did not slow down where it should have.
TEST_CASE("a vehicle reads grip from the world it is standing on", "[physics][world][vehicle]")
{
    const JoltGuard jolt;

    auto descriptor = raceengine::ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 200.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    // A ninth surface, past anything the generator's own table carries, and every triangle tagged
    // with it — so a fallback to the first entry reads 1.00 where the answer is 0.37.
    constexpr auto gravel = std::uint32_t{8};
    mesh->materials.resize(gravel + 1, raceengine::SurfaceMaterial{});
    mesh->materials[gravel] =
        raceengine::SurfaceMaterial{.gripMultiplier = 0.37, .bumpiness = 0.011, .kind = SurfaceKind::Gravel};
    std::fill(mesh->surfaces.begin(), mesh->surfaces.end(), gravel);

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    const auto setup = raceengine::placeholderSedan();
    REQUIRE(setup.has_value());

    auto state = raceengine::VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

    auto stepped = std::expected<raceengine::VehicleStep, std::string>{};
    for (auto step = 0; step < 720; step++)
    {
        stepped = stepVehicle(setup.value(), state, raceengine::VehicleInput{}, raceengine::noDriveTorque,
                              world.value(), 1.0 / 360.0);
        REQUIRE(stepped.has_value());
    }

    for (const auto& corner : stepped->corners)
    {
        REQUIRE(corner.patch.inContact);
        REQUIRE(corner.patch.gripMultiplier == Catch::Approx(0.37));
        REQUIRE(corner.patch.bumpiness == Catch::Approx(0.011));
    }
}
