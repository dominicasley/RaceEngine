// The three questions the resolution tension turns on: `./EngineTests "[.envelope-resolution]"`.
//
// E2's crux is that a *local* enveloping method wants more sample points than 3x3 gives, and the
// sampler already costs 72% of the vehicle tick and scales linearly in rays. Before accepting that as
// binding, three things are worth measuring, and none of them needs a seat:
//
//   1. **Where the 0.35 microseconds a ray actually goes.** It was attributed to the sampler as a
//      whole and then spoken of as a ray cost, which are not the same number. If most of it is not
//      traversal, the cost table is soft and the argument built on it moves.
//   2. **What an anisotropic grid costs.** A kerb is a longitudinal feature crossed transversely, so
//      the break the model cannot resolve is *lateral*. Resolution bought only across the patch is a
//      fraction of the price of resolution bought in both directions.
//   3. **Whether the normals already carry the break.** Every ray comes back with the hit triangle's
//      normal and `aggregateContactPatch` load-weights them into one vector. Two samples either side
//      of a kerb edge return distinctly different normals, and depth does not do that — a slope break
//      is continuous in height and discontinuous in gradient. If so, the geometry is already in hand
//      at 3x3 and the resolution needed is to *locate* a break, not to resolve one.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::aggregateContactPatch;
using raceengine::bringUpJolt;
using raceengine::contactPatchSamples;
using raceengine::ContactPatchSampling;
using raceengine::ContactSampleGeometry;
using raceengine::defaultSurfaceMaterials;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::sampleProvingGround;
using raceengine::SurfaceHit;
using raceengine::tearDownJolt;
using raceengine::WheelPose;

namespace
{

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

[[nodiscard]] ProvingGroundDescriptor plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

[[nodiscard]] ProvingGroundDescriptor kerb()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 20.0;
    descriptor.width = 12.0;
    descriptor.cellSize = 0.05;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 4.0, .to = 16.0}};

    return descriptor;
}

// Median rather than mean, for the same reason the tick budget uses one: what a typical call costs is
// the number wanted, and a mean is hostage to whatever else the machine was doing during one of them.
[[nodiscard]] double medianOf(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());

    return samples[samples.size() / 2];
}

} // namespace

TEST_CASE("where the microseconds in a ray actually go", "[.envelope-resolution]")
{
    // **The number the whole cost argument rests on, taken apart.** 12.55 microseconds for 36 rays was
    // measured across the *sampler* — build the grid, cast, aggregate — and then quoted as 0.35 a ray.
    // Those are different quantities and only one of them is a raycast.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    std::printf("\n=== the cast alone, one batched call, against ray count ===\n");
    std::printf("\n%8s %14s %14s\n", "rays", "per call us", "per ray ns");

    auto perRayAt = std::vector<std::pair<std::size_t, double>>{};

    for (const auto count : {1u, 4u, 9u, 36u, 100u, 324u, 900u})
    {
        auto origins = std::vector<glm::dvec3>{};
        auto directions = std::vector<glm::dvec3>{};

        for (auto index = std::uint32_t{0}; index < count; index++)
        {
            // Spread over a metre of ground so the traversal is not answering the same query every
            // time out of a warm cache line.
            const auto offset = 0.03 * static_cast<double>(index);
            origins.push_back(glm::dvec3(offset, 1.0, 20.0 + offset));
            directions.push_back(glm::dvec3(0.0, -1.0, 0.0));
        }

        auto hits = std::vector<SurfaceHit>{};
        auto samples = std::vector<double>{};

        for (auto pass = 0; pass < 2000; pass++)
        {
            const auto started = std::chrono::steady_clock::now();
            world->castRays(origins, directions, 2.0, hits);
            samples.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
        }

        const auto median = medianOf(samples);
        perRayAt.emplace_back(count, median);

        std::printf("%8u %14.3f %14.1f\n", count, median, 1000.0 * median / static_cast<double>(count));
    }

    // A straight line through the two ends separates the part that is paid once per call from the part
    // that is paid once per ray. A large intercept means the cost is in setting the call up.
    const auto [smallCount, smallCost] = perRayAt.front();
    const auto [largeCount, largeCost] = perRayAt.back();
    const auto perRay = (largeCost - smallCost) / static_cast<double>(largeCount - smallCount);
    const auto fixed = smallCost - perRay * static_cast<double>(smallCount);

    std::printf("\n  fitted: %.3f us fixed per call + %.1f ns per ray.\n", fixed, 1000.0 * perRay);

    SECTION("and how much of that fixed cost is the call boundary rather than the traversal")
    {
        // Thirty-six rays one way and then the other. The bridge is already batched — one crossing per
        // call, not per ray — so this is not asking whether batching helps. It is asking what a call
        // costs to set up, which is the allocation of five scratch vectors plus the result vector.
        auto origins = std::vector<glm::dvec3>{};
        auto directions = std::vector<glm::dvec3>{};
        for (auto index = 0; index < 36; index++)
        {
            const auto offset = 0.03 * static_cast<double>(index);
            origins.push_back(glm::dvec3(offset, 1.0, 20.0 + offset));
            directions.push_back(glm::dvec3(0.0, -1.0, 0.0));
        }

        auto hits = std::vector<SurfaceHit>{};

        auto batched = std::vector<double>{};
        for (auto pass = 0; pass < 2000; pass++)
        {
            const auto started = std::chrono::steady_clock::now();
            world->castRays(origins, directions, 2.0, hits);
            batched.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
        }

        auto split = std::vector<double>{};
        for (auto pass = 0; pass < 2000; pass++)
        {
            const auto started = std::chrono::steady_clock::now();
            for (auto index = std::size_t{0}; index < origins.size(); index++)
            {
                const auto one = std::vector<glm::dvec3>{origins[index]};
                const auto direction = std::vector<glm::dvec3>{directions[index]};
                world->castRays(one, direction, 2.0, hits);
            }
            split.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
        }

        std::printf("\n  36 rays in one call: %.3f us.  36 calls of one ray: %.3f us.  ratio %.1fx\n",
                    medianOf(batched), medianOf(split), medianOf(split) / medianOf(batched));
    }

    SECTION("and whether the per-ray cost is tree traversal or the work wrapped around it")
    {
        // **The measurement that says whether 332 ns is reducible.** Two things happen per ray that are
        // not traversal: a `BodyLockRead` is taken, and the hit material is found by a linear scan of
        // the world's material table. Both are per ray and neither depends on how big the mesh is.
        //
        // Traversal does. So casting the same rays against meshes of very different triangle counts
        // separates them: a cost that grows with the log of the triangle count is the tree, and a cost
        // that is flat is the wrapper.
        std::printf("\n%12s %12s %14s %14s\n", "cell size m", "triangles", "per ray ns", "vs coarsest");

        auto coarsest = 0.0;

        for (const auto cell : {8.0, 2.0, 0.5, 0.25})
        {
            auto descriptor = plate();
            descriptor.length = 200.0;
            descriptor.width = 200.0;
            descriptor.cellSize = cell;

            const auto built = generateProvingGround(descriptor);
            REQUIRE(built.has_value());
            const auto sized = PhysicsWorld::create(built.value());
            REQUIRE(sized.has_value());

            auto origins = std::vector<glm::dvec3>{};
            auto directions = std::vector<glm::dvec3>{};
            for (auto index = 0; index < 36; index++)
            {
                const auto offset = 0.03 * static_cast<double>(index);
                origins.push_back(glm::dvec3(offset, 1.0, 20.0 + offset));
                directions.push_back(glm::dvec3(0.0, -1.0, 0.0));
            }

            auto hits = std::vector<SurfaceHit>{};
            auto samples = std::vector<double>{};
            for (auto pass = 0; pass < 2000; pass++)
            {
                const auto started = std::chrono::steady_clock::now();
                sized->castRays(origins, directions, 2.0, hits);
                samples.push_back(
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
            }

            const auto perRayNanos = 1000.0 * medianOf(samples) / 36.0;
            coarsest = coarsest > 0.0 ? coarsest : perRayNanos;

            const auto quads = (200.0 / cell) * (200.0 / cell);
            std::printf("%12.2f %12.0f %14.1f %13.2fx\n", cell, 2.0 * quads, perRayNanos, perRayNanos / coarsest);
        }

        std::printf("\n  a flat column is a per-ray wrapper cost that no amount of tree tuning removes;\n"
                    "  a rising one is traversal, and traversal is what a bigger grid genuinely buys more of.\n");
    }
}

TEST_CASE("the sampler taken apart into its three phases", "[.envelope-resolution]")
{
    // Which of the three the 12.55 microseconds is actually in. If the cast is a minority of it, then
    // the ray count is not what the cost table is really measuring and refining the grid is cheaper
    // than the linear-in-rays reading suggests.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto radius = setup->corners.front().hardpoints.wheelRadius;

    std::printf("\n=== one car's contact sampling, split by phase ===\n");
    std::printf("\n%8s %8s %12s %12s %12s %12s\n", "grid", "rays", "build us", "cast us", "aggregate", "total us");

    for (const auto [across, along] :
         std::array<std::pair<std::uint32_t, std::uint32_t>, 6>{{{3, 3}, {5, 3}, {7, 3}, {9, 3}, {5, 5}, {7, 7}}})
    {
        auto sampling = ContactPatchSampling{};
        sampling.across = across;
        sampling.along = along;

        auto poses = std::array<WheelPose, 4>{};
        for (auto index = std::size_t{0}; index < 4; index++)
        {
            const auto x = index % 2 == 0 ? -0.78 : 0.78;
            const auto z = 20.0 + (index < 2 ? 1.3 : -1.3);
            poses[index] = WheelPose{.centre = glm::dvec3(x, radius - 0.015, z),
                                     .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                     .forward = glm::dvec3(0.0, 0.0, 1.0),
                                     .radius = radius};
        }

        auto build = std::vector<double>{};
        auto cast = std::vector<double>{};
        auto aggregate = std::vector<double>{};

        for (auto pass = 0; pass < 1000; pass++)
        {
            auto geometries = std::array<ContactSampleGeometry, 4>{};
            auto origins = std::vector<glm::dvec3>{};
            auto directions = std::vector<glm::dvec3>{};

            auto mark = std::chrono::steady_clock::now();
            for (auto index = std::size_t{0}; index < 4; index++)
            {
                geometries[index] = contactPatchSamples(poses[index], sampling);
                origins.insert(origins.end(), geometries[index].origins.begin(), geometries[index].origins.end());
                directions.insert(directions.end(), geometries[index].directions.begin(),
                                  geometries[index].directions.end());
            }
            build.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mark).count());

            auto hits = std::vector<SurfaceHit>{};
            mark = std::chrono::steady_clock::now();
            world->castRays(origins, directions, sampling.searchDistance * 2.0, hits);
            cast.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mark).count());

            mark = std::chrono::steady_clock::now();
            auto consumed = std::size_t{0};
            for (auto index = std::size_t{0}; index < 4; index++)
            {
                const auto span = geometries[index].origins.size();
                const auto slice = std::vector<SurfaceHit>(hits.begin() + static_cast<std::ptrdiff_t>(consumed),
                                                           hits.begin() + static_cast<std::ptrdiff_t>(consumed + span));
                consumed += span;
                const auto patch = aggregateContactPatch(geometries[index], slice, world->materials(), sampling);
                REQUIRE(patch.totalSamples == span);
            }
            aggregate.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mark).count());
        }

        const auto b = medianOf(build);
        const auto c = medianOf(cast);
        const auto a = medianOf(aggregate);

        std::printf("%6ux%-2u %8u %12.2f %12.2f %12.2f %12.2f\n", across, along, across * along * 4, b, c, a,
                    b + c + a);
    }

    std::printf("\n  a 5x3 grid buys resolution ACROSS the patch, which is the axis a kerb edge breaks,\n"
                "  and leaves the along-patch direction at three where nothing is happening.\n");
}

TEST_CASE("whether the normals already locate the break at three by three", "[.envelope-resolution]")
{
    // **The cheapest possible answer to the resolution problem: it may already be paid for.** Every
    // ray comes back with the hit triangle's normal, and `aggregateContactPatch` load-weights all of
    // them into one vector and keeps nothing else. That aggregate is exactly what a break destroys.
    //
    // Depth cannot locate a slope break — a chamfer meets flat ground continuously in height, which is
    // why every global summary of the depths has failed. The *gradient* is discontinuous there, and
    // the gradient is what a normal is.
    const JoltGuard jolt;

    const auto descriptor = kerb();
    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto radius = setup->corners.front().hardpoints.wheelRadius;

    const auto sampling = ContactPatchSampling{};

    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 10.0;
    constexpr auto compression = 0.015;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    std::printf("\n=== per-sample normals across the kerb, 3x3, 30 degrees of yaw ===\n");
    std::printf("  the chamfer's inner edge is x = %.2f and its top edge x = %.2f.\n", descriptor.kerbInnerEdge,
                descriptor.kerbInnerEdge + descriptor.kerbChamfer);
    std::printf("\n  `worst pair` is the largest angle between the normals of two ADJACENT samples across\n"
                "  the patch. On unbroken road it is zero however tilted that road is; it is non-zero\n"
                "  only where a break in slope lies between two samples.\n");
    std::printf("\n%8s %6s %12s %14s %16s\n", "x", "cont", "aggregate", "worst pair deg", "depth spread mm");

    for (auto step = 0; step <= 16; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.04 * static_cast<double>(step);

        const auto beneath = sampleProvingGround(descriptor, x, middleZ);
        const auto wheel = WheelPose{.centre = glm::dvec3(x, beneath.height + radius - compression, middleZ),
                                     .spinAxis = spinAxis,
                                     .forward = forward,
                                     .radius = radius};

        const auto geometry = contactPatchSamples(wheel, sampling);

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

        const auto patch = aggregateContactPatch(geometry, hits, world->materials(), sampling);

        // Adjacent across the patch, which is the axis the grid is laid out on and the axis a
        // longitudinal kerb breaks. Row by row, comparing neighbours within each row.
        auto worst = 0.0;
        for (auto along = std::uint32_t{0}; along < sampling.along; along++)
        {
            for (auto across = std::uint32_t{0}; across + 1 < sampling.across; across++)
            {
                const auto left = static_cast<std::size_t>(along * sampling.across + across);
                const auto right = left + 1;

                if (!hits[left].hit || !hits[right].hit)
                {
                    continue;
                }

                const auto cosine = std::clamp(glm::dot(hits[left].normal, hits[right].normal), -1.0, 1.0);
                worst = std::max(worst, std::acos(cosine) * 57.29577951308232);
            }
        }

        // The aggregate the model actually keeps, for comparison: one number, and it moves smoothly
        // whether or not there is a break under the patch.
        const auto aggregateTilt = std::acos(std::clamp(patch.normal.y, -1.0, 1.0)) * 57.29577951308232;

        std::printf("%8.3f %6u %11.2f%s %14.2f %16.2f\n", x, patch.contactingSamples, aggregateTilt, " deg", worst,
                    patch.depthSpread * 1000.0);
    }
}
