// A probe, hidden behind a dotted tag: `./EngineTests "[.patch-chamfer]"`.
//
// It exists because the recorded diagnosis of the kerb-edge load dip named the wrong mechanism, and
// the only way to name the right one is to look at what the nine rays individually came back with as
// a loaded wheel rides up a chamfer. Aggregates cannot answer this: every candidate explanation
// produces the same falling number.
//
// It rides the wheel across the kerb's chamfer at a **held compression**, which is the whole trick.
// A wheel driven across it by the suspension changes its own load, so a falling load proves nothing;
// a wheel whose centre is placed at a fixed height above the road under its own middle is asked to
// report a compression that by construction is not changing, and any change in the answer is the
// sampler's.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::aggregateContactPatch;
using raceengine::bringUpJolt;
using raceengine::ContactSampleGeometry;
using raceengine::contactPatchSamples;
using raceengine::ContactPatchSampling;
using raceengine::defaultSurfaceMaterials;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::SurfaceHit;
using raceengine::sampleProvingGround;
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

// The kerb band, with the wheel crossing it at 30 degrees of yaw — which is what makes the case
// interesting, because a square-on wheel meets the chamfer with a whole row at once and a yawed one
// meets it a sample at a time.
[[nodiscard]] ProvingGroundDescriptor kerbGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 0.05;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 60.0, .to = 90.0}};

    return descriptor;
}

// The car's own numbers rather than the probe's. **Two of the three figures the original measurement
// was taken with have since been corrected** — the wheel radius by 11 mm and the unsprung mass by
// nearly half — so a probe carrying its own 0.31 would keep re-deriving a baseline against a car that
// no longer exists. The tyre rate is unchanged at 298,926 N/m and is read from the same place for the
// same reason.
struct CarNumbers
{
    double radius = 0.0;
    double tyreRate = 0.0;
};

[[nodiscard]] CarNumbers golfNumbers()
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    return CarNumbers{.radius = setup->corners.front().hardpoints.wheelRadius,
                      .tyreRate = setup->corners.front().tireVerticalRate};
}

} // namespace

TEST_CASE("what the nine rays see as a held wheel rides a kerb chamfer", "[.patch-chamfer]")
{
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto sampling = ContactPatchSampling{};

    constexpr auto compression = 0.015;
    constexpr auto radius = 0.31;
    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 75.0;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    std::printf("\n  the kerb's inner edge is at x = %.2f and its chamfer runs to %.2f\n", descriptor.kerbInnerEdge,
                descriptor.kerbInnerEdge + descriptor.kerbChamfer);
    std::printf("\n  the wheel is placed so its DEEPEST sample penetrates exactly %.0f mm at every position,\n"
                "  which is the only control under which a change in the reported number is the sampler's.\n",
                compression * 1000.0);
    std::printf("\n%7s %5s %5s %9s %9s %9s   %s\n", "x", "hit", "cont", "pen mm", "meanC mm", "load N",
                "per-sample depth into the tread, mm (blank = not touching)");

    // Just inboard of the edge to well up on the flat top.
    for (auto step = 0; step <= 24; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.025 * static_cast<double>(step);

        const auto poseAt = [&](const double height)
        {
            return WheelPose{.centre = glm::dvec3(x, height, middleZ),
                             .spinAxis = spinAxis,
                             .forward = forward,
                             .radius = radius};
        };

        const auto deepestFor = [&](const double height)
        {
            const auto probeGeometry = contactPatchSamples(poseAt(height), sampling);

            auto probeHits = std::vector<SurfaceHit>{};
            world->castRays(probeGeometry.origins, probeGeometry.directions, sampling.searchDistance * 2.0, probeHits);

            auto deepest = -sampling.searchDistance;
            for (auto index = std::size_t{0}; index < probeHits.size(); index++)
            {
                if (probeHits[index].hit)
                {
                    deepest = std::max(deepest, probeHits[index].point.y - probeGeometry.tireSurface[index].y);
                }
            }

            return deepest;
        };

        // Bisect the wheel's height until the deepest sample is exactly the compression asked for.
        // Monotonic in the height by construction — lowering the wheel deepens every sample — so
        // thirty halvings take it to well under a micron.
        const auto beneath = sampleProvingGround(descriptor, x, middleZ);
        auto low = beneath.height + radius - 0.20;
        auto high = beneath.height + radius + 0.20;

        for (auto iteration = 0; iteration < 40; iteration++)
        {
            const auto middle = 0.5 * (low + high);
            (deepestFor(middle) > compression ? low : high) = middle;
        }

        const auto wheel = poseAt(0.5 * (low + high));

        const auto geometry = contactPatchSamples(wheel, sampling);

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

        const auto patch = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);

        // Every sample's own depth into the tread. Negative is the road standing off the undeformed
        // tyre, which is the quantity the whole design question is about: how far below is it, and
        // is that because the road fell away or because the tread curved up out of reach of a road
        // that is still perfectly well there?
        auto gaps = std::string{};
        auto carrying = 0.0;
        auto contacting = std::uint32_t{0};

        for (auto index = std::size_t{0}; index < hits.size(); index++)
        {
            const auto& surface = geometry.tireSurface[index];
            const auto depth = hits[index].hit ? hits[index].point.y - surface.y : -sampling.searchDistance;

            if (depth > 0.0)
            {
                carrying += depth;
                contacting++;
            }

            auto cell = std::array<char, 32>{};
            std::snprintf(cell.data(), cell.size(), depth > 0.0 ? "%7.1f*" : "%7.1f ", depth * 1000.0);
            gaps += cell.data();
        }

        const auto meanCarrying = contacting > 0 ? carrying / static_cast<double>(contacting) : 0.0;

        std::printf("%7.3f %5u %5u %9.2f %9.2f %9.0f   %s\n", x, patch.totalSamples, patch.contactingSamples,
                    patch.penetration * 1000.0, meanCarrying * 1000.0, 298926.0 * patch.penetration, gaps.c_str());
    }
}

TEST_CASE("does the kerb-edge load dip converge away with sample count", "[.patch-chamfer]")
{
    // **The question that decides whether any of this is a code change at all.**
    //
    // The 3x3 grid spans about 0.25 m of a 0.30 m chamfer, so it is being asked about a feature it
    // can barely resolve, and the recorded suspicion was that the dip might be a quadrature artefact
    // — in which case the answer is a bigger number in the vehicle's data and no new mechanism.
    // Refining the grid is what settles it: a quadrature error shrinks, and a model error does not.
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    constexpr auto compression = 0.015;
    constexpr auto radius = 0.31;
    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 75.0;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    std::printf("\n%9s", "x");
    for (const auto count : {3u, 5u, 7u, 9u, 15u, 31u})
    {
        std::printf("   %2ux%-2u N", count, count);
    }
    std::printf("     rejected\n");

    for (auto step = 0; step <= 12; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.05 * static_cast<double>(step);
        auto rejected = std::uint32_t{0};

        std::printf("%9.3f", x);

        for (const auto count : {3u, 5u, 7u, 9u, 15u, 31u})
        {
            auto sampling = ContactPatchSampling{};
            sampling.across = count;
            sampling.along = count;

            const auto poseAt = [&](const double height)
            {
                return WheelPose{.centre = glm::dvec3(x, height, middleZ),
                                 .spinAxis = spinAxis,
                                 .forward = forward,
                                 .radius = radius};
            };

            const auto cast = [&](const double height, ContactSampleGeometry& outGeometry,
                                  std::vector<SurfaceHit>& outHits)
            {
                outGeometry = contactPatchSamples(poseAt(height), sampling);
                world->castRays(outGeometry.origins, outGeometry.directions, sampling.searchDistance * 2.0, outHits);
            };

            // **The deepest sample is held, not the wheel height**, and it is held *per grid* — a
            // finer grid finds a deeper point, so holding the height instead would compare six
            // different compressions and call the difference convergence.
            auto low = 0.0;
            auto high = 0.0;
            {
                const auto beneath = sampleProvingGround(descriptor, x, middleZ);
                low = beneath.height + radius - 0.20;
                high = beneath.height + radius + 0.20;
            }

            for (auto iteration = 0; iteration < 40; iteration++)
            {
                const auto middle = 0.5 * (low + high);

                auto probeGeometry = ContactSampleGeometry{};
                auto probeHits = std::vector<SurfaceHit>{};
                cast(middle, probeGeometry, probeHits);

                auto deepest = -sampling.searchDistance;
                for (auto index = std::size_t{0}; index < probeHits.size(); index++)
                {
                    if (probeHits[index].hit)
                    {
                        deepest = std::max(deepest, probeHits[index].point.y - probeGeometry.tireSurface[index].y);
                    }
                }

                (deepest > compression ? low : high) = middle;
            }

            auto geometry = ContactSampleGeometry{};
            auto hits = std::vector<SurfaceHit>{};
            cast(0.5 * (low + high), geometry, hits);

            const auto patch = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);

            // Spike rejection, counted rather than assumed. The recorded diagnosis blamed it; the
            // audit said it never fires here. This is the arithmetic that settles that.
            if (count == 3)
            {
                auto touching = std::vector<double>{};
                for (auto index = std::size_t{0}; index < hits.size(); index++)
                {
                    if (hits[index].hit)
                    {
                        const auto depth = hits[index].point.y - geometry.tireSurface[index].y;
                        if (depth > 0.0)
                        {
                            touching.push_back(depth);
                        }
                    }
                }

                if (!touching.empty())
                {
                    std::sort(touching.begin(), touching.end());
                    const auto ceiling = touching[touching.size() / 2] + sampling.spikeRejection;
                    for (const auto depth : touching)
                    {
                        rejected += depth > ceiling ? 1u : 0u;
                    }
                }
            }

            std::printf("  %9.0f", 298926.0 * patch.penetration);
        }

        std::printf("   %10u\n", rejected);
    }
}

TEST_CASE("does the count and the spread together separate a kerb from a light tyre", "[.patch-chamfer]")
{
    // **The measurement E1 exists to take**, and the one that decides which enveloping model E2
    // should be. The proposed discriminator is the pair (`Contact Samples`, `Patch Depth Spread`):
    //
    //   - many samples, wide spread  -> road geometry under a loaded tyre; load should be preserved
    //   - few samples, narrow spread -> a genuinely light tyre; load should be low
    //
    // Both halves are printed here across a chamfer crossing at a held compression, beside the same
    // wheel on flat road at the compression that makes the *reported load match*. If the pair
    // separates them, the discriminator is sound and a filter can be built on it. If the chamfer's
    // spread collapses along with its count, the pair does not separate them and the proposal needs
    // re-thinking before anything is built on it.
    //
    // Nothing is asserted. This is a number-producing probe and the numbers are the deliverable.
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto car = golfNumbers();
    const auto sampling = ContactPatchSampling{};

    constexpr auto compression = 0.015;
    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 75.0;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    // The tread's own rise at the leading and trailing rows. It is the spread a level patch reports
    // on flat ground, so it is the floor every number in the table below stands on.
    const auto rise = car.radius - std::sqrt(car.radius * car.radius - 0.25 * sampling.length * sampling.length);

    std::printf("\n  the Golf's corrected numbers: radius %.4f m, tyre rate %.0f N/m.\n", car.radius, car.tyreRate);
    std::printf("  the tread stands %.2f mm proud at the leading and trailing rows, which is the\n"
                "  spread a LEVEL patch reports and the floor under every figure below.\n",
                rise * 1000.0);
    // **E3, answered here rather than in a milestone of its own.** The brief asks whether spurious
    // slip-ratio spikes from a missing effective rolling radius are already visible, and the premise
    // needs correcting first: the model *has* an effective rolling radius, `(2R + (R - pen)) / 3`,
    // and it is already a function of the penetration the defect corrupts. So there is nothing to
    // build — the radius error is the same fault, divided by three, and it will go when the
    // aggregation is fixed. The column below is what it is worth.
    std::printf("\n%8s %6s %10s %10s %10s %10s   %s\n", "x", "cont", "spread mm", "pen mm", "load N", "Re err %",
                "what that reads as");

    for (auto step = 0; step <= 20; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.03 * static_cast<double>(step);

        const auto poseAt = [&](const double height)
        {
            return WheelPose{.centre = glm::dvec3(x, height, middleZ),
                             .spinAxis = spinAxis,
                             .forward = forward,
                             .radius = car.radius};
        };

        const auto deepestFor = [&](const double height)
        {
            const auto probeGeometry = contactPatchSamples(poseAt(height), sampling);

            auto probeHits = std::vector<SurfaceHit>{};
            world->castRays(probeGeometry.origins, probeGeometry.directions, sampling.searchDistance * 2.0, probeHits);

            auto deepest = -sampling.searchDistance;
            for (auto index = std::size_t{0}; index < probeHits.size(); index++)
            {
                if (probeHits[index].hit)
                {
                    deepest = std::max(deepest, probeHits[index].point.y - probeGeometry.tireSurface[index].y);
                }
            }

            return deepest;
        };

        const auto beneath = sampleProvingGround(descriptor, x, middleZ);
        auto low = beneath.height + car.radius - 0.20;
        auto high = beneath.height + car.radius + 0.20;

        for (auto iteration = 0; iteration < 40; iteration++)
        {
            const auto middle = 0.5 * (low + high);
            (deepestFor(middle) > compression ? low : high) = middle;
        }

        const auto geometry = contactPatchSamples(poseAt(0.5 * (low + high)), sampling);

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

        const auto patch = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);

        // What a *flat* road would have to be doing to report this same load: three samples at one
        // depth over a divisor of nine, so the matching compression is three times the penetration —
        // and on flat ground below the tread's rise that patch reads a spread of exactly zero. That
        // is the comparison the whole channel is for, and it is why the classification column can
        // say what a reader of `Fz` alone would have concluded.
        const auto matchingFlat = 3.0 * patch.penetration;
        const auto* reads = patch.contactingSamples == 9     ? "full patch"
                            : patch.depthSpread > 0.5 * rise ? "PARTIAL, and tilted -- load should be preserved"
                                                             : "partial and level -- genuinely light";

        // Against the effective radius the same wheel would have at the flat-ground reading either
        // side of the kerb, as a percentage of the free radius. A slip ratio is `(w.Re - v) / |v|`,
        // so at steady rolling an error in `Re` is a slip-ratio error of the same fraction.
        const auto effective = car.radius - patch.penetration / 3.0;
        const auto onFlat = car.radius - 0.00820 / 3.0;

        std::printf("%8.3f %6u %10.2f %10.2f %10.0f %10.3f   %s\n", x, patch.contactingSamples,
                    patch.depthSpread * 1000.0, patch.penetration * 1000.0, car.tyreRate * patch.penetration,
                    100.0 * (effective - onFlat) / car.radius, reads);

        if (patch.contactingSamples > 0 && patch.contactingSamples < 9)
        {
            std::printf("%8s %6s %10s %10.2f %10s   a flat road reporting this load is at %.2f mm with spread 0.00\n",
                        "", "", "", patch.penetration * 1000.0, "", matchingFlat * 1000.0);
        }
    }
}

TEST_CASE("how much of the dip is the mesh's facets rather than the model", "[.patch-chamfer]")
{
    // **The brief asks for a re-characterisation on a mesh kerb, on the grounds that every previous
    // measurement was against "an analytic chamfer". That premise is worth checking first, and it is
    // not right**: this probe has always cast its rays through Jolt against a generated *triangle
    // mesh*, and the analytic `sampleProvingGround` is used only to bracket the bisection. What was
    // never varied is the facet size, which has always been 0.05 m — six facets across a 0.30 m
    // chamfer, uniform and axis aligned, which is not what a loaded track's kerb looks like.
    //
    // So the question that is actually open is facet sensitivity, and this sweeps it. A model fault
    // is indifferent to how finely the road is tessellated; a sampling artefact is not.
    //
    // **Read the result knowing what it can and cannot show.** The chamfer is a *plane*, and every
    // tessellation of a plane is the same plane, so this sweep can only ever return one number — and
    // it does, identically, from twenty-four facets across the chamfer down to one. That is worth
    // having, because it rules the tessellation out as a cause of the W3 dip on the geometry the
    // whole finding was measured against. It is **not** an answer to the general facet question the
    // brief raises about a two-cam tangency search, and it cannot be: nothing in the proving ground
    // has curvature at the contact patch's 0.16 m scale, so there is no facet-scale feature here to
    // be sensitive to. That question needs a loaded track mesh under a driven wheel, and the channel
    // this probe was extended for is what will answer it off a lap.
    const JoltGuard jolt;

    const auto car = golfNumbers();
    const auto sampling = ContactPatchSampling{};

    constexpr auto compression = 0.015;
    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 10.0;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    std::printf("\n  the chamfer is 0.30 m wide, so the facet count across it is 0.30 / cell.\n");
    std::printf("\n%10s %8s %12s %12s %10s %10s\n", "cell m", "facets", "flat N", "chamfer N", "dip N", "spread mm");

    for (const auto cell : {0.0125, 0.025, 0.05, 0.10, 0.15, 0.30})
    {
        auto descriptor = ProvingGroundDescriptor{};
        descriptor.length = 20.0;
        descriptor.width = 12.0;
        descriptor.cellSize = cell;
        descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 4.0, .to = 16.0}};

        const auto ground = generateProvingGround(descriptor);
        REQUIRE(ground.has_value());
        const auto world = PhysicsWorld::create(ground.value());
        REQUIRE(world.has_value());

        const auto readAt = [&](const double x)
        {
            const auto poseAt = [&](const double height)
            {
                return WheelPose{.centre = glm::dvec3(x, height, middleZ),
                                 .spinAxis = spinAxis,
                                 .forward = forward,
                                 .radius = car.radius};
            };

            const auto deepestFor = [&](const double height)
            {
                const auto probeGeometry = contactPatchSamples(poseAt(height), sampling);

                auto probeHits = std::vector<SurfaceHit>{};
                world->castRays(probeGeometry.origins, probeGeometry.directions, sampling.searchDistance * 2.0,
                                probeHits);

                auto deepest = -sampling.searchDistance;
                for (auto index = std::size_t{0}; index < probeHits.size(); index++)
                {
                    if (probeHits[index].hit)
                    {
                        deepest = std::max(deepest, probeHits[index].point.y - probeGeometry.tireSurface[index].y);
                    }
                }

                return deepest;
            };

            auto low = car.radius - 0.20;
            auto high = car.radius + 0.25;
            for (auto iteration = 0; iteration < 40; iteration++)
            {
                const auto middle = 0.5 * (low + high);
                (deepestFor(middle) > compression ? low : high) = middle;
            }

            const auto geometry = contactPatchSamples(poseAt(0.5 * (low + high)), sampling);

            auto hits = std::vector<SurfaceHit>{};
            world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

            return aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);
        };

        // Well clear of the kerb, and in the middle of its chamfer — the same two positions the
        // characterisation test pins.
        const auto flat = readAt(descriptor.kerbInnerEdge - 0.15);
        const auto chamfer = readAt(descriptor.kerbInnerEdge + 0.10);

        std::printf("%10.4f %8.1f %12.0f %12.0f %10.0f %10.2f\n", cell, 0.30 / cell, car.tyreRate * flat.penetration,
                    car.tyreRate * chamfer.penetration, car.tyreRate * (flat.penetration - chamfer.penetration),
                    chamfer.depthSpread * 1000.0);
    }
}

TEST_CASE("what the brief's bridging rule would cost on flat ground", "[.patch-chamfer]")
{
    // **The check that decides whether the proposed rule can be shipped**, and it is a check on the
    // case the rule is not aimed at.
    //
    // The rule is: a sample whose road has *not* fallen away below the effective contact plane is
    // spanned by the carcass, carries no load, and leaves the divisor — so the reported compression
    // becomes the mean over the samples that are touching rather than over the whole grid. At a
    // chamfer that removes the dip. The question is what it does everywhere else, and the answer is
    // that a chamfer is a plane and so is flat ground: the road has not fallen away in *either*
    // case, so the rule fires identically on both.
    //
    // No world needed. Flat ground under an upright wheel is arithmetic.
    constexpr auto radius = 0.31;
    const auto sampling = ContactPatchSampling{};

    std::printf("\n  the tread stands %0.1f mm proud at the leading and trailing rows, so below that\n"
                "  compression only the middle row is touching at all.\n",
                1000.0 * (radius - std::sqrt(radius * radius - 0.25 * sampling.length * sampling.length)));
    std::printf("\n%10s %8s %12s %12s %8s\n", "compress", "touching", "as shipped N", "bridged N", "ratio");

    for (const auto compression : {2.0, 5.0, 8.0, 10.0, 12.0, 13.4, 16.0, 20.0, 25.0})
    {
        const auto wheel = WheelPose{.centre = glm::dvec3(0.0, radius - 0.001 * compression, 0.0),
                                     .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                     .forward = glm::dvec3(0.0, 0.0, 1.0),
                                     .radius = radius};

        const auto geometry = contactPatchSamples(wheel, sampling);

        auto hits = std::vector<SurfaceHit>{};
        for (const auto& surface : geometry.tireSurface)
        {
            hits.push_back(SurfaceHit{.point = glm::dvec3(surface.x, 0.0, surface.z),
                                      .normal = glm::dvec3(0.0, 1.0, 0.0),
                                      .distance = 0.0,
                                      .surface = 0,
                                      .hit = true});
        }

        const auto patch = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);

        auto carrying = 0.0;
        auto touching = std::uint32_t{0};
        for (auto index = std::size_t{0}; index < hits.size(); index++)
        {
            const auto depth = hits[index].point.y - geometry.tireSurface[index].y;
            if (depth > 0.0)
            {
                carrying += depth;
                touching++;
            }
        }

        const auto shipped = 298926.0 * patch.penetration;
        const auto bridged = touching > 0 ? 298926.0 * carrying / static_cast<double>(touching) : 0.0;

        std::printf("%9.1f  %8u %12.0f %12.0f %8.2f\n", compression, touching, shipped, bridged,
                    shipped > 0.0 ? bridged / shipped : 0.0);
    }
}
