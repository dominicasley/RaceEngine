#include <cmath>
#include <set>
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
