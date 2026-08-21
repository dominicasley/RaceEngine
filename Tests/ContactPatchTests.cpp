#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::aggregateContactPatch;
using raceengine::bringUpJolt;
using raceengine::contactPatchSamples;
using raceengine::ContactPatchSampling;
using raceengine::defaultSurfaceMaterials;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::SurfaceHit;
using raceengine::SurfaceKind;
using raceengine::SurfaceMaterial;
using raceengine::tearDownJolt;
using raceengine::WheelPose;

namespace
{

// Flat ground at a chosen height, answered without a world at all: the aggregation is pure, so most
// of what it has to get right can be stated as arithmetic.
//
// Each hit lands directly under its own sample rather than all of them at the origin. That is not
// cosmetic — the patch centre is the load-weighted mean of the hit points, so a fixture that puts
// every hit at the same place reports a patch that can never migrate, and the case that exists to
// prove migration passes trivially and proves nothing.
std::vector<SurfaceHit> flatGround(const raceengine::ContactSampleGeometry& geometry, const double height,
                                   const std::uint32_t surface = 0)
{
    auto hits = std::vector<SurfaceHit>{};
    for (const auto& sample : geometry.tireSurface)
    {
        hits.push_back(SurfaceHit{.point = glm::dvec3(sample.x, height, sample.z),
                                  .normal = glm::dvec3(0.0, 1.0, 0.0),
                                  .distance = 0.0,
                                  .surface = surface,
                                  .hit = true});
    }

    return hits;
}

WheelPose uprightWheel(const double height)
{
    return WheelPose{.centre = glm::dvec3(0.0, height, 0.0),
                     .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                     .forward = glm::dvec3(0.0, 0.0, 1.0),
                     .radius = 0.31};
}

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

} // namespace

TEST_CASE("the sample grid covers the patch and follows the tread", "[physics][contact]")
{
    const auto sampling = ContactPatchSampling{};
    const auto geometry = contactPatchSamples(uprightWheel(0.31), sampling);

    REQUIRE(geometry.origins.size() == 9);
    REQUIRE(geometry.tireSurface.size() == 9);
    REQUIRE(geometry.directions.size() == 9);

    SECTION("the middle sample sits one radius below the wheel centre")
    {
        REQUIRE(geometry.tireSurface[4].y == Catch::Approx(0.0));
        REQUIRE(geometry.tireSurface[4].x == Catch::Approx(0.0).margin(1e-15));
        REQUIRE(geometry.tireSurface[4].z == Catch::Approx(0.0).margin(1e-15));
    }

    SECTION("the leading and trailing rows sit higher, because the tread curves away")
    {
        // A cylinder, not a flat plate. Ignoring this reports the ends of the patch as penetrating
        // less than they do and quietly shrinks the contact patch under load.
        const auto half = 0.5 * sampling.length;
        const auto expected = 0.31 - std::sqrt(0.31 * 0.31 - half * half);

        REQUIRE(geometry.tireSurface[1].y == Catch::Approx(expected));
        REQUIRE(geometry.tireSurface[7].y == Catch::Approx(expected));
        REQUIRE(geometry.tireSurface[1].y > geometry.tireSurface[4].y);
    }

    SECTION("the grid spans the stated patch dimensions")
    {
        REQUIRE(geometry.tireSurface[3].x == Catch::Approx(-0.5 * sampling.width));
        REQUIRE(geometry.tireSurface[5].x == Catch::Approx(0.5 * sampling.width));
        REQUIRE(geometry.tireSurface[1].z == Catch::Approx(-0.5 * sampling.length));
        REQUIRE(geometry.tireSurface[7].z == Catch::Approx(0.5 * sampling.length));
    }

    SECTION("every ray starts above the tire and looks down")
    {
        for (auto index = std::size_t{0}; index < geometry.origins.size(); index++)
        {
            REQUIRE(geometry.origins[index].y ==
                    Catch::Approx(geometry.tireSurface[index].y + sampling.searchDistance));
            REQUIRE(geometry.directions[index] == glm::dvec3(0.0, -1.0, 0.0));
        }
    }
}

TEST_CASE("a wheel clear of the ground is not in contact", "[physics][contact]")
{
    const auto sampling = ContactPatchSampling{};
    const auto geometry = contactPatchSamples(uprightWheel(0.50), sampling);
    const auto patch = aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

    REQUIRE_FALSE(patch.inContact);
    REQUIRE(patch.contactingSamples == 0);
    REQUIRE(patch.penetration == 0.0);
}

TEST_CASE("a wheel square on flat ground is centred and level", "[physics][contact]")
{
    const auto sampling = ContactPatchSampling{};

    // 20 mm into the ground. At 10 mm the contact patch is only 156 mm long against the grid's
    // 160 mm, so the leading and trailing rows genuinely miss — the tread has curved above the
    // road by half a millimetre. That is correct, and it is why this case is not run there.
    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);
    const auto patch = aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

    REQUIRE(patch.inContact);
    REQUIRE(patch.contactingSamples == 9);
    REQUIRE(patch.normal.y == Catch::Approx(1.0));
    REQUIRE(patch.centre.x == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(patch.gripMultiplier == Catch::Approx(defaultSurfaceMaterials().front().gripMultiplier));

    // The middle row is 10 mm in; the leading and trailing rows less, because the tread curves away.
    // The reported penetration is the mean over the whole grid, so it sits below the deepest sample.
    REQUIRE(patch.penetration > 0.0);
    REQUIRE(patch.penetration < 0.020);
}

TEST_CASE("penetration is the mean over the grid, not over what is touching", "[physics][contact]")
{
    // The distinction that decides whether a wheel hanging half off a step carries half its load or
    // all of it. Vertical force goes as total compression across the patch, so a sample touching
    // nothing must pull the average down rather than be excluded from it.
    const auto sampling = ContactPatchSampling{};
    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);

    auto materials = defaultSurfaceMaterials();

    auto full = flatGround(geometry, 0.0);
    const auto whole = aggregateContactPatch(geometry, full, materials, sampling);

    // Drop the outboard column into a hole deep enough to miss.
    auto stepped = full;
    for (const auto index : {2u, 5u, 8u})
    {
        stepped[index].hit = false;
    }

    const auto partial = aggregateContactPatch(geometry, stepped, materials, sampling);

    REQUIRE(partial.inContact);
    REQUIRE(partial.contactingSamples == 6);
    REQUIRE(partial.penetration < whole.penetration);
    // Two thirds of the patch carrying, so about two thirds of the compression.
    REQUIRE(partial.penetration == Catch::Approx(whole.penetration * 2.0 / 3.0).epsilon(0.02));

    // And the load has moved inboard, which is the migration the whole exercise exists for.
    REQUIRE(partial.centre.x < whole.centre.x);
}

TEST_CASE("a patch straddling two surfaces blends their grip", "[physics][contact]")
{
    const auto sampling = ContactPatchSampling{};
    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);
    const auto materials = defaultSurfaceMaterials();

    auto hits = flatGround(geometry, 0.0, static_cast<std::uint32_t>(SurfaceKind::Tarmac));
    for (const auto index : {2u, 5u, 8u})
    {
        hits[index].surface = static_cast<std::uint32_t>(SurfaceKind::Grass);
    }

    const auto patch = aggregateContactPatch(geometry, hits, materials, sampling);

    const auto tarmac = materials[static_cast<std::size_t>(SurfaceKind::Tarmac)].gripMultiplier;
    const auto grass = materials[static_cast<std::size_t>(SurfaceKind::Grass)].gripMultiplier;

    // Between the two, and not at either — a snap between them is exactly what this replaces.
    REQUIRE(patch.gripMultiplier < tarmac);
    REQUIRE(patch.gripMultiplier > grass);
}

TEST_CASE("one bad triangle does not carry the wheel", "[physics][contact]")
{
    // Load share goes as penetration, so the deepest sample dominates — which is fine when it is
    // road and ruinous when it is a seam in the collision mesh. A spike must not tilt the plane.
    const auto sampling = ContactPatchSampling{};
    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);
    const auto materials = defaultSurfaceMaterials();

    auto hits = flatGround(geometry, 0.0);
    const auto clean = aggregateContactPatch(geometry, hits, materials, sampling);

    // One sample reporting the road half a metre higher than its neighbours, tilted on its side.
    hits[0].point.y = 0.5;
    hits[0].normal = glm::normalize(glm::dvec3(0.7, 0.7, 0.0));

    const auto spiked = aggregateContactPatch(geometry, hits, materials, sampling);

    REQUIRE(spiked.inContact);
    // The plane stays essentially level rather than being dragged 45 degrees over by one sample.
    REQUIRE(spiked.normal.y > 0.99);
    REQUIRE(std::abs(spiked.normal.x) < 0.12);
    // And the wheel is not reported as half a metre into the ground.
    REQUIRE(spiked.penetration < clean.penetration + sampling.spikeRejection);
}

TEST_CASE("a wheel half on a real kerb tilts and its patch migrates", "[physics][contact][world]")
{
    // The whole case, end to end and against real geometry: generate a kerb, put a wheel across its
    // edge, cast the grid through Jolt and aggregate. A single ray at the wheel centre is compared
    // against it, because the point is not that the grid is accurate — it is that the single ray is
    // qualitatively wrong.
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 20.0;
    descriptor.width = 10.0;
    descriptor.cellSize = 0.05;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 0.0, .to = 20.0}};

    const auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    const auto world = PhysicsWorld::create(mesh.value());
    REQUIRE(world.has_value());

    // Straddling the kerb's inner edge, just touching.
    const auto sampling = ContactPatchSampling{};
    auto wheel = uprightWheel(0.305);
    wheel.centre.x = descriptor.kerbInnerEdge;
    wheel.centre.z = 10.0;

    const auto geometry = contactPatchSamples(wheel, sampling);
    auto hits = std::vector<SurfaceHit>{};
    world->castRays(geometry.origins, geometry.directions, 2.0, hits);

    const auto patch = aggregateContactPatch(geometry, hits, mesh->materials, sampling);

    REQUIRE(patch.inContact);

    // The effective plane is tilted across the car, which is the moment that rolls it. A single ray
    // reports a flat plane here and no moment at all.
    REQUIRE(patch.normal.x < -0.01);
    REQUIRE(patch.normal.y > 0.95);

    // The load has moved outboard, onto the kerb — the patch centre is no longer under the wheel's
    // middle, and tire forces applied there produce the right moment without it being modelled.
    REQUIRE(patch.centre.x > wheel.centre.x);

    // And it is standing on two surfaces at once, so its grip is between them.
    const auto kerbGrip = mesh->materials[static_cast<std::size_t>(SurfaceKind::Kerb)].gripMultiplier;
    const auto tarmacGrip = mesh->materials[static_cast<std::size_t>(SurfaceKind::Tarmac)].gripMultiplier;
    REQUIRE(patch.gripMultiplier < tarmacGrip);
    REQUIRE(patch.gripMultiplier > kerbGrip);
}

TEST_CASE("the sample count is data driven and the aggregate converges", "[physics][contact]")
{
    // Three by three is where the brief says to start, not where it has to stay. What must be true
    // of the count is not that it changes nothing — the penetration is a quadrature over a curved
    // tread and three points across a curve is three points across a curve — but that refining it
    // converges, and that the quantities with no curvature in them do not move at all.
    const auto wheel = uprightWheel(0.29);
    const auto materials = defaultSurfaceMaterials();

    const auto measure = [&](const std::uint32_t count)
    {
        const auto sampling = ContactPatchSampling{.across = count, .along = count};
        const auto geometry = contactPatchSamples(wheel, sampling);
        return aggregateContactPatch(geometry, flatGround(geometry, 0.0), materials, sampling);
    };

    const auto coarse = measure(3);
    const auto fine = measure(7);
    const auto finest = measure(31);

    REQUIRE(coarse.totalSamples == 9);
    REQUIRE(fine.totalSamples == 49);

    // Converging: seven is nearer the truth than three, and by a clear margin rather than by noise.
    const auto coarseError = std::abs(coarse.penetration - finest.penetration);
    const auto fineError = std::abs(fine.penetration - finest.penetration);
    REQUIRE(fineError < 0.5 * coarseError);

    // And the aggregates that are not integrals of a curve do not move with the count at all.
    REQUIRE(fine.normal.y == Catch::Approx(coarse.normal.y));
    REQUIRE(fine.gripMultiplier == Catch::Approx(coarse.gripMultiplier));
    REQUIRE(fine.centre.x == Catch::Approx(coarse.centre.x).margin(1e-12));
}

// A probe rather than a gate, hidden like the shimmy one: run it with
// `./EngineTests "[.patch-chamfer]"`. It exists to answer whether the known kerb-chamfer
// roughness is the patch aggregation's fault and, if so, which of two candidate repairs
// removes it — a question that wants a swept table rather than a pass or a fail.
TEST_CASE("probe: a patch crossing a kerb chamfer", "[.patch-chamfer]")
{
    const auto sampling = ContactPatchSampling{};
    const auto materials = defaultSurfaceMaterials();

    // The proving ground's own kerb, stated here so the probe cannot drift from it.
    constexpr auto kerbHeight = 0.05;
    constexpr auto kerbChamfer = 0.30;
    constexpr auto kerbInnerEdge = 3.0;

    const auto surfaceAt = [](const double x)
    {
        const auto across = x - kerbInnerEdge;
        if (across <= 0.0)
        {
            return 0.0;
        }

        return kerbHeight * (across < kerbChamfer ? across / kerbChamfer : 1.0);
    };

    WARN("  wheelX   contacting  rejected   shipped(mm)  wholeGrid(mm)  loadWtd(mm)  touchingOnly(mm)");

    for (auto step = 0; step <= 24; step++)
    {
        // Walk the wheel centre out across the chamfer, a centimetre at a time.
        const auto wheelX = kerbInnerEdge - 0.12 + 0.02 * static_cast<double>(step);

        // Yawed thirty degrees, because the recorded roughness is a wheel crossing *at an angle*
        // and a perpendicular crossing spans less of the chamfer than an angled one does.
        constexpr auto yaw = 0.5235987755982988;
        const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));
        const auto spin = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));

        // Ridden over rather than driven through: the centre is placed so the deepest sample sits
        // at a constant 15 mm of compression, which is what a loaded tyre does over a kerb. A wheel
        // held at a fixed height instead reports penetrations no suspension would ever allow.
        auto centreY = 0.31;
        for (auto pass = 0; pass < 40; pass++)
        {
            const auto probe = WheelPose{.centre = glm::dvec3(wheelX, centreY, 0.0),
                                         .spinAxis = spin,
                                         .forward = forward,
                                         .radius = 0.31};
            const auto probed = contactPatchSamples(probe, sampling);

            auto deepest = -1.0;
            for (const auto& sample : probed.tireSurface)
            {
                deepest = std::max(deepest, surfaceAt(sample.x) - sample.y);
            }

            centreY += deepest - 0.015;
        }

        const auto wheel = WheelPose{.centre = glm::dvec3(wheelX, centreY, 0.0),
                                     .spinAxis = spin,
                                     .forward = forward,
                                     .radius = 0.31};

        const auto geometry = contactPatchSamples(wheel, sampling);

        auto hits = std::vector<SurfaceHit>{};
        for (const auto& sample : geometry.tireSurface)
        {
            const auto height = surfaceAt(sample.x);
            hits.push_back(SurfaceHit{.point = glm::dvec3(sample.x, height, sample.z),
                                      .normal = glm::dvec3(0.0, 1.0, 0.0),
                                      .distance = 0.0,
                                      .surface = 0,
                                      .hit = height >= sample.y - 0.30});
        }

        const auto patch = aggregateContactPatch(geometry, hits, materials, sampling);

        // The same arithmetic the shipping aggregation does, restated so the two candidate
        // repairs can be evaluated against identical samples.
        auto depths = std::vector<double>{};
        for (auto index = std::size_t{0}; index < hits.size(); index++)
        {
            depths.push_back(hits[index].point.y - geometry.tireSurface[index].y);
        }

        auto touching = std::vector<double>{};
        for (const auto depth : depths)
        {
            if (depth > 0.0)
            {
                touching.push_back(depth);
            }
        }

        auto shipped = 0.0;
        auto wholeGrid = 0.0;
        auto weighted = 0.0;
        auto weight = 0.0;
        auto kept = 0;
        auto rejected = 0;

        if (!touching.empty())
        {
            std::sort(touching.begin(), touching.end());
            const auto ceiling = touching[touching.size() / 2] + sampling.spikeRejection;

            for (const auto depth : depths)
            {
                if (depth <= 0.0)
                {
                    continue;
                }

                wholeGrid += depth;

                if (depth > ceiling)
                {
                    rejected++;
                    continue;
                }

                kept++;
                shipped += depth;
                weighted += depth * depth;
                weight += depth;
            }
        }

        const auto count = static_cast<double>(hits.size());
        WARN("  " << wheelX << "    " << kept << "          " << rejected << "        "
                  << patch.penetration * 1000.0 << "        " << wholeGrid / count * 1000.0 << "       "
                  << (weight > 0.0 ? weighted / weight : 0.0) * 1000.0 << "        "
                  << (kept > 0 ? shipped / static_cast<double>(kept) : 0.0) * 1000.0);
    }
}
