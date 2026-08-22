// The E2 bake-off: `./EngineTests "[.envelope-candidates]"`.
//
// **This exists to try to kill a recommendation before anything is built on it.** The E-series brief
// offers three enveloping models and says to scope them against data rather than default to the most
// standard. Two of them can be scored here, cheaply and with no seat time, because the aggregation is
// a pure function of the sampled hits — so a candidate is a few lines and the chamfer is arithmetic.
//
// The gate has three parts and a candidate has to pass all three:
//
//   1. **The chamfer.** Load must be roughly preserved where a real tyre's belt would bridge a
//      cross-slope. This is the W3 defect and the only thing a candidate is *for*.
//   2. **Flat ground.** Load must stay monotonic in compression. This is what killed the earlier
//      bridging rule — it trebled the vertical rate below 10.5 mm and put a step at 10.5 where load
//      *fell* as the tyre was compressed further. Negative stiffness diverges.
//   3. **A genuine overhang.** Where the road really has fallen away, load must fall. A candidate
//      that preserves load here has not distinguished the two cases, it has just stopped looking.
//
// The control is **the wheel centre's height above the road directly beneath it**, not the deepest
// sample's compression. That is a change from `[.patch-chamfer]` and it is deliberate: the deepest
// sample is a quantity each candidate would redefine, so holding it would compare six different
// wheel placements and call the difference a result. Height above the road under the centre is what
// the suspension actually controls and is the same number for every candidate.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
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

// Where a sample sits in the patch's own frame. `contactPatchSamples` walks `along` in the outer loop
// and `across` in the inner one, and does not report these, so they are rebuilt against that order —
// a candidate that filters or fits needs to know which samples are neighbours.
[[nodiscard]] double offsetAt(const std::size_t index, const std::uint32_t count, const double extent)
{
    if (count < 2)
    {
        return 0.0;
    }

    return (static_cast<double>(index) / static_cast<double>(count - 1) - 0.5) * extent;
}

struct PatchFrame
{
    std::vector<double> u;
    std::vector<double> v;
};

[[nodiscard]] PatchFrame patchFrame(const ContactPatchSampling& sampling)
{
    auto frame = PatchFrame{};

    for (auto along = std::uint32_t{0}; along < sampling.along; along++)
    {
        for (auto across = std::uint32_t{0}; across < sampling.across; across++)
        {
            frame.u.push_back(offsetAt(across, sampling.across, sampling.width));
            frame.v.push_back(offsetAt(along, sampling.along, sampling.length));
        }
    }

    return frame;
}

// The shipped rule, restated so the candidates are scored against the same divisor rather than
// against `aggregateContactPatch`'s extra machinery. Positive depths summed over the *whole* grid,
// which is the aggregate the defect lives in.
[[nodiscard]] double penetrationOf(const std::vector<double>& depths)
{
    auto summed = 0.0;
    for (const auto depth : depths)
    {
        summed += depth > 0.0 ? depth : 0.0;
    }

    return summed / static_cast<double>(depths.size());
}

// **Candidate A — the brief's option 2 as literally written**: filter the sampled height field with a
// length scale set by carcass bending, then aggregate pointwise as before.
[[nodiscard]] std::vector<double> filteredDepths(const std::vector<double>& roadHeight,
                                                 const std::vector<double>& tyreHeight, const PatchFrame& frame,
                                                 const double bendingLength)
{
    auto depths = std::vector<double>(roadHeight.size(), 0.0);

    for (auto index = std::size_t{0}; index < roadHeight.size(); index++)
    {
        auto weighted = 0.0;
        auto weight = 0.0;

        for (auto other = std::size_t{0}; other < roadHeight.size(); other++)
        {
            const auto du = frame.u[index] - frame.u[other];
            const auto dv = frame.v[index] - frame.v[other];
            const auto w = std::exp(-(du * du + dv * dv) / (2.0 * bendingLength * bendingLength));

            weighted += w * roadHeight[other];
            weight += w;
        }

        depths[index] = weighted / weight - tyreHeight[index];
    }

    return depths;
}

// **Candidate B — a conforming belt.** Fit the road across the patch, then let the tyre's footprint
// *rotate* to lie against it, up to a limit set by how far the carcass can actually bend. Beyond that
// limit the belt stops following and the samples past it genuinely lose contact.
//
// This is the candidate the recommendation should have been, and the reason is in the arithmetic
// below: it acts on the road's *slope*, which is what a chamfer has and flat ground does not, where
// candidate A acts on the road's *curvature*, which a chamfer has none of.
[[nodiscard]] std::vector<double> conformingDepths(const std::vector<double>& roadHeight,
                                                   const std::vector<double>& tyreHeight, const PatchFrame& frame,
                                                   const double conformityRadians)
{
    // The road's slope **across** the patch, by least squares. Only this one component is wanted:
    // the cross-slope is the axis the Winkler bed fails on, and the tread's curvature already owns
    // the along-patch direction. The grid is symmetric about its own centre, so the sums in `u`
    // vanish and the normal equations separate into this one line — a general 3x3 solve here would be
    // arithmetic hiding behind a matrix type.
    auto sumU = 0.0;
    auto sumH = 0.0;
    auto sumUU = 0.0;
    auto sumUH = 0.0;
    const auto count = static_cast<double>(roadHeight.size());

    for (auto index = std::size_t{0}; index < roadHeight.size(); index++)
    {
        const auto u = frame.u[index];
        const auto h = roadHeight[index];

        sumU += u;
        sumH += h;
        sumUU += u * u;
        sumUH += u * h;
    }

    const auto spread = sumUU - sumU * sumU / count;
    const auto acrossSlope = spread > 1e-12 ? (sumUH - sumU * sumH / count) / spread : 0.0;

    // The whole rule, and it is one clamp. A cross-slope the carcass can bend to is followed; one it
    // cannot is followed as far as the carcass goes and no further.
    const auto limit = std::tan(conformityRadians);
    const auto followed = std::clamp(acrossSlope, -limit, limit);

    auto depths = std::vector<double>(roadHeight.size(), 0.0);
    for (auto index = std::size_t{0}; index < roadHeight.size(); index++)
    {
        // The footprint rotates about the patch's own centre, so the wheel is not moved and the
        // sample under the middle of the patch reads exactly what it read before.
        depths[index] = roadHeight[index] - (tyreHeight[index] + followed * frame.u[index]);
    }

    return depths;
}

// One case: the road under each sample, and the undeformed tyre under each sample.
struct Case
{
    std::vector<double> road;
    std::vector<double> tyre;
};

// Analytic ground, stated as a height function across the patch. No world, so the geometry is exactly
// what it says it is and the result cannot be an artefact of a mesh.
template <typename Height>
[[nodiscard]] Case analytic(const ContactPatchSampling& sampling, const double radius, const double compression,
                            Height height)
{
    const auto wheel = WheelPose{.centre = glm::dvec3(0.0, radius - compression, 0.0),
                                 .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                 .forward = glm::dvec3(0.0, 0.0, 1.0),
                                 .radius = radius};

    const auto geometry = contactPatchSamples(wheel, sampling);

    auto built = Case{};
    for (const auto& surface : geometry.tireSurface)
    {
        built.road.push_back(height(surface.x));
        built.tyre.push_back(surface.y);
    }

    return built;
}

} // namespace

TEST_CASE("candidate A cannot fix a chamfer, and the reason is that a chamfer is a plane", "[.envelope-candidates]")
{
    // **The refutation, and it is of my own recommendation.** The brief's option 2 filters the
    // sampled height field with a carcass length scale. A symmetric normalised kernel convolved with
    // a *linear* function returns that same linear function — so on any planar road the filter is the
    // identity everywhere its kernel is not truncated, and a chamfer is planar.
    //
    // What is left is a grid-edge artefact: on a 3x3 grid every sample but the centre has a truncated
    // kernel, so the edges get pulled toward the interior. That is not a fix, and the table below
    // shows it going the wrong way — it **shrinks** as the grid refines, where the defect grows.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto radius = setup->corners.front().hardpoints.wheelRadius;
    const auto rate = setup->corners.front().tireVerticalRate;

    constexpr auto slope = 0.16657; // tan(9.46 deg), the chamfer's own cross-slope
    constexpr auto compression = 0.015;

    std::printf("\n  a %.2f degree cross-slope under the Golf's wheel, held at %.0f mm at the patch centre.\n",
                std::atan(slope) * 57.29577951308232, compression * 1000.0);
    std::printf("\n%8s %14s %14s %12s\n", "grid", "as shipped N", "candidate A N", "change");

    for (const auto count : {3u, 5u, 9u, 15u})
    {
        auto sampling = ContactPatchSampling{};
        sampling.across = count;
        sampling.along = count;

        const auto frame = patchFrame(sampling);
        const auto built = analytic(sampling, radius, compression, [&](const double x) { return -slope * x; });

        auto shipped = std::vector<double>(built.road.size(), 0.0);
        for (auto index = std::size_t{0}; index < built.road.size(); index++)
        {
            shipped[index] = built.road[index] - built.tyre[index];
        }

        // 30 mm of carcass bending length, which is the order a tyre sidewall gives.
        const auto filtered = filteredDepths(built.road, built.tyre, frame, 0.030);

        const auto before = rate * penetrationOf(shipped);
        const auto after = rate * penetrationOf(filtered);

        std::printf("%6ux%-2u %14.0f %14.0f %11.1f%%\n", count, count, before, after,
                    before > 0.0 ? 100.0 * (after - before) / before : 0.0);
    }

    std::printf("\n  the change is an edge effect and it vanishes under refinement. Candidate A is refused.\n");
}

TEST_CASE("candidate B against all three parts of the gate", "[.envelope-candidates]")
{
    // A conforming belt: the footprint rotates to lie against the road's cross-slope, up to a limit.
    // The discriminator is an **angle**, which is what separates this from the refuted bridging rule
    // — that one keyed on depth below the contact plane, and a plane has none.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto radius = setup->corners.front().hardpoints.wheelRadius;
    const auto rate = setup->corners.front().tireVerticalRate;

    const auto sampling = ContactPatchSampling{};
    const auto frame = patchFrame(sampling);

    const auto score = [&](const Case& built, const double conformity)
    {
        auto shipped = std::vector<double>(built.road.size(), 0.0);
        for (auto index = std::size_t{0}; index < built.road.size(); index++)
        {
            shipped[index] = built.road[index] - built.tyre[index];
        }

        const auto conforming = conformingDepths(built.road, built.tyre, frame, conformity);

        return std::pair{rate * penetrationOf(shipped), rate * penetrationOf(conforming)};
    };

    const auto conformity = 12.0 * 3.14159265358979323846 / 180.0;

    SECTION("part 2 first: flat ground must stay monotonic, which is what killed the bridging rule")
    {
        std::printf("\n=== part 2: flat ground, load against compression ===\n");
        std::printf("  the rule that was refused trebled the rate below 10.5 mm and then LOST load as\n"
                    "  the tyre was compressed further. Both columns here must climb, every row.\n");
        std::printf("\n%10s %14s %14s\n", "compress mm", "as shipped N", "candidate B N");

        auto lastShipped = -1.0;
        auto lastConforming = -1.0;

        for (const auto millimetres : {2.0, 5.0, 8.0, 10.0, 10.5, 12.0, 15.0, 20.0, 25.0})
        {
            const auto built = analytic(sampling, radius, 0.001 * millimetres, [](const double) { return 0.0; });
            const auto [shipped, conforming] = score(built, conformity);

            std::printf("%10.1f %14.0f %14.0f\n", millimetres, shipped, conforming);

            // **The gate.** Monotonic, both of them.
            REQUIRE(shipped > lastShipped);
            REQUIRE(conforming > lastConforming);
            lastShipped = shipped;
            lastConforming = conforming;
        }

        // And flat ground must be left completely alone: a level road has no cross-slope to follow,
        // so the candidate must be the identity here rather than merely monotonic.
        const auto built = analytic(sampling, radius, 0.015, [](const double) { return 0.0; });
        const auto [shipped, conforming] = score(built, conformity);
        REQUIRE(conforming == Catch::Approx(shipped).epsilon(1e-12));
    }

    SECTION("part 1: the chamfer, which is the only thing the candidate is for")
    {
        std::printf("\n=== part 1: a cross-slope, held at 15 mm under the patch centre ===\n");
        std::printf("\n%10s %14s %14s %12s   %s\n", "slope deg", "as shipped N", "candidate B N", "recovered", "");

        for (const auto degrees : {0.0, 2.0, 5.0, 9.46, 15.0, 25.0, 40.0})
        {
            const auto slope = std::tan(degrees * 3.14159265358979323846 / 180.0);
            const auto built = analytic(sampling, radius, 0.015, [&](const double x) { return -slope * x; });
            const auto [shipped, conforming] = score(built, conformity);

            // What a level road at the same centre compression reports, which is the load that ought
            // to survive a cross-slope the carcass can bridge.
            const auto level =
                score(analytic(sampling, radius, 0.015, [](const double) { return 0.0; }), conformity).first;

            std::printf("%10.2f %14.0f %14.0f %11.0f%%   %s\n", degrees, shipped, conforming,
                        100.0 * conforming / level, degrees <= 12.0 ? "within carcass conformity" : "beyond it");
        }
    }

    SECTION("part 3: a genuine overhang, where load must still fall")
    {
        std::printf("\n=== part 3: half the patch over a cliff, held at 15 mm under the patch centre ===\n");
        std::printf("  a candidate that holds load up here has stopped distinguishing the two cases.\n");
        std::printf("\n%10s %14s %14s %12s\n", "drop mm", "as shipped N", "candidate B N", "of level");

        const auto level = score(analytic(sampling, radius, 0.015, [](const double) { return 0.0; }), conformity).first;

        for (const auto drop : {0.0, 5.0, 20.0, 50.0, 200.0})
        {
            const auto built =
                analytic(sampling, radius, 0.015, [&](const double x) { return x > 0.001 ? -0.001 * drop : 0.0; });
            const auto [shipped, conforming] = score(built, conformity);

            std::printf("%10.1f %14.0f %14.0f %11.0f%%\n", drop, shipped, conforming, 100.0 * conforming / level);
        }
    }

    SECTION("and the conformity limit is a real discriminator rather than a tuned one")
    {
        // The chamfer is 9.46 degrees, so a limit of 10 would pass it by a whisker and prove nothing.
        // Swept instead: what matters is that there is a *range* of limits over which the chamfer is
        // recovered and the cliff is not, and how wide that range is.
        std::printf("\n=== the limit, swept: chamfer recovered vs cliff wrongly held up ===\n");
        std::printf("\n%12s %18s %18s\n", "limit deg", "chamfer % of level", "200mm cliff % of level");

        for (const auto degrees : {4.0, 8.0, 12.0, 16.0, 20.0, 30.0})
        {
            const auto limit = degrees * 3.14159265358979323846 / 180.0;
            const auto level = score(analytic(sampling, radius, 0.015, [](const double) { return 0.0; }), limit).first;

            const auto chamferSlope = 0.16657;
            const auto chamfer =
                score(analytic(sampling, radius, 0.015, [&](const double x) { return -chamferSlope * x; }), limit)
                    .second;
            const auto cliff =
                score(analytic(sampling, radius, 0.015, [](const double x) { return x > 0.001 ? -0.200 : 0.0; }), limit)
                    .second;

            std::printf("%12.0f %17.0f%% %17.0f%%\n", degrees, 100.0 * chamfer / level, 100.0 * cliff / level);
        }
    }
}

TEST_CASE("candidate B on the meshed kerb, against the recorded numbers", "[.envelope-candidates]")
{
    // The analytic cases above are exact but synthetic. This is the same candidate against the real
    // generated kerb through Jolt, at the positions `[.patch-chamfer]` and the characterisation test
    // already pin, so the numbers can be read straight against the record.
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 20.0;
    descriptor.width = 12.0;
    descriptor.cellSize = 0.05;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 4.0, .to = 16.0}};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());
    const auto radius = setup->corners.front().hardpoints.wheelRadius;
    const auto rate = setup->corners.front().tireVerticalRate;

    const auto sampling = ContactPatchSampling{};
    const auto frame = patchFrame(sampling);

    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 10.0;
    constexpr auto compression = 0.015;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));
    const auto conformity = 12.0 * 3.14159265358979323846 / 180.0;

    std::printf("\n=== the meshed kerb, 30 degrees of yaw, wheel centre held %.0f mm above the road beneath it ===\n",
                compression * 1000.0);
    std::printf("\n%8s %6s %14s %14s %12s %10s %11s\n", "x", "cont", "as shipped N", "candidate B N", "of flat",
                "fit deg", "fit resid mm");

    auto flatReference = 0.0;

    for (auto step = 0; step <= 14; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.04 * static_cast<double>(step);

        // The control: height above the road directly beneath the wheel centre.
        const auto beneath = sampleProvingGround(descriptor, x, middleZ);
        const auto wheel = WheelPose{.centre = glm::dvec3(x, beneath.height + radius - compression, middleZ),
                                     .spinAxis = spinAxis,
                                     .forward = forward,
                                     .radius = radius};

        const auto geometry = contactPatchSamples(wheel, sampling);

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

        auto road = std::vector<double>{};
        auto tyre = std::vector<double>{};
        auto shipped = std::vector<double>{};
        auto contacting = std::uint32_t{0};

        for (auto index = std::size_t{0}; index < hits.size(); index++)
        {
            // A ray that found nothing is road that is at least a search distance down, which is an
            // overhang by any measure and must not be fitted through.
            const auto height =
                hits[index].hit ? hits[index].point.y : geometry.tireSurface[index].y - sampling.searchDistance;

            road.push_back(height);
            tyre.push_back(geometry.tireSurface[index].y);
            shipped.push_back(height - geometry.tireSurface[index].y);
            contacting += height > geometry.tireSurface[index].y ? 1u : 0u;
        }

        const auto before = rate * penetrationOf(shipped);
        const auto after = rate * penetrationOf(conformingDepths(road, tyre, frame, conformity));

        flatReference = flatReference > 0.0 ? flatReference : before;

        // How well a *single plane* actually describes the road under this patch. The candidate's
        // whole premise is that it does; the residual says where that premise fails, and it is the
        // number that decides whether the result below is a tuning problem or a modelling one.
        auto sumU = 0.0;
        auto sumH = 0.0;
        auto sumUU = 0.0;
        auto sumUH = 0.0;
        const auto n = static_cast<double>(road.size());
        for (auto index = std::size_t{0}; index < road.size(); index++)
        {
            sumU += frame.u[index];
            sumH += road[index];
            sumUU += frame.u[index] * frame.u[index];
            sumUH += frame.u[index] * road[index];
        }
        const auto slope = (sumUH - sumU * sumH / n) / (sumUU - sumU * sumU / n);
        const auto mean = sumH / n;

        auto residual = 0.0;
        for (auto index = std::size_t{0}; index < road.size(); index++)
        {
            const auto predicted = mean + slope * (frame.u[index] - sumU / n);
            residual = std::max(residual, std::abs(road[index] - predicted));
        }

        std::printf("%8.3f %6u %14.0f %14.0f %11.0f%% %10.2f %11.1f\n", x, contacting, before, after,
                    100.0 * after / flatReference, std::atan(-slope) * 57.29577951308232, residual * 1000.0);
    }

    std::printf("\n  `fit resid` is the worst departure of the road from the single plane the candidate\n"
                "  assumes. Where it is small the candidate is on its own premise; where it is large the\n"
                "  patch is straddling a break in slope and the fit describes neither side.\n");
}
