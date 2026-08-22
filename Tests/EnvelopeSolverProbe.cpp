// Scoping the belt solver before it is committed to: `./EngineTests "[.envelope-solver]"`.
//
// E2's brief names one open risk and it is not the model, it is the arithmetic under it: a coupled bed
// is a **unilateral** contact problem, so it is a linear complementarity problem per patch per tick
// rather than a matrix inversion, and **convergence quality at a fixed iteration budget is the open
// question**. A fixed budget is not negotiable — both parity gates are byte-identical and an early-out
// on a floating-point convergence test makes the frame a function of how the arithmetic happened to
// round — so what has to be established is what budget buys, on the hardest case, before anything is
// built on top of it.
//
// Four questions, in the order they decide things:
//
//   1. **What a sweep buys.** Load error against a heavily converged reference, per budget, on the
//      hardest active set the geometry produces.
//   2. **What the belt's one number does.** The bridging length swept against flat ground, the
//      chamfer, a genuine overhang and a kerb edge — the four cases acceptance is stated in.
//   3. **Whether the answer depends on the grid**, which it must not, because the coupling is
//      supposed to be a property of the tyre and not of how finely it is being asked.
//   4. **What it costs**, against the 50 microsecond tick budget the sampler already spends 34% of.
//
// Nothing is asserted beyond the fixtures' own preconditions. This is a number-producing probe and the
// numbers are the deliverable.

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
using raceengine::beltShearRate;
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
using raceengine::solveBeltBed;
using raceengine::SurfaceHit;
using raceengine::tearDownJolt;
using raceengine::WheelPose;

namespace
{

constexpr auto pi = 3.14159265358979323846;
constexpr auto yaw = 30.0 * pi / 180.0;
constexpr auto compression = 0.015;
constexpr auto middleZ = 10.0;

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

[[nodiscard]] ProvingGroundDescriptor kerbGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 20.0;
    descriptor.width = 12.0;
    descriptor.cellSize = 0.05;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 4.0, .to = 16.0}};

    return descriptor;
}

[[nodiscard]] double golfRadius()
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    return setup->corners.front().hardpoints.wheelRadius;
}

[[nodiscard]] double golfTyreRate()
{
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    return setup->corners.front().tireVerticalRate;
}

// One reading, taken with the wheel placed so its deepest sample penetrates exactly the asked-for
// amount. The same control the whole W3 investigation used, and for the same reason: a wheel at a
// fixed *height* over a tilted road is at a different compression, so a held height would compare
// different things and read the difference as the model's.
struct Placed
{
    ContactSampleGeometry geometry;
    std::vector<SurfaceHit> hits;
};

[[nodiscard]] Placed placeAt(const PhysicsWorld& world, const ProvingGroundDescriptor& descriptor,
                             const ContactPatchSampling& sampling, const double x, const double radius)
{
    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    const auto poseAt = [&](const double height)
    {
        return WheelPose{
            .centre = glm::dvec3(x, height, middleZ), .spinAxis = spinAxis, .forward = forward, .radius = radius};
    };

    auto placed = Placed{};

    const auto cast = [&](const double height)
    {
        placed.geometry = contactPatchSamples(poseAt(height), sampling);
        world.castRays(placed.geometry.origins, placed.geometry.directions, sampling.searchDistance * 2.0, placed.hits);

        auto deepest = -sampling.searchDistance;
        for (auto index = std::size_t{0}; index < placed.hits.size(); index++)
        {
            if (placed.hits[index].hit)
            {
                deepest = std::max(deepest, placed.hits[index].point.y - placed.geometry.tireSurface[index].y);
            }
        }

        return deepest;
    };

    const auto beneath = sampleProvingGround(descriptor, x, middleZ);
    auto low = beneath.height + radius - 0.20;
    auto high = beneath.height + radius + 0.20;

    for (auto iteration = 0; iteration < 40; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        (cast(middle) > compression ? low : high) = middle;
    }

    cast(0.5 * (low + high));

    // **The fixture's own precondition.** Every conclusion below is "at a held 15 mm", and a bisection
    // that failed to converge — because the bracket missed, or because nothing is under the wheel —
    // would quietly hand back a reading at some other compression and every number after it would be
    // measuring the placement rather than the belt.
    auto deepest = -sampling.searchDistance;
    for (auto index = std::size_t{0}; index < placed.hits.size(); index++)
    {
        if (placed.hits[index].hit)
        {
            deepest = std::max(deepest, placed.hits[index].point.y - placed.geometry.tireSurface[index].y);
        }
    }
    REQUIRE(std::abs(deepest - compression) < 1e-6);

    return placed;
}

// Flat ground under an upright wheel, and an overhang: the same wheel with everything outboard of the
// middle column standing over nothing at all. No world needed — both are arithmetic — and the overhang
// is the case that says whether the belt bridges a void it should not.
[[nodiscard]] std::vector<SurfaceHit> flatHits(const ContactSampleGeometry& geometry, const double height)
{
    auto hits = std::vector<SurfaceHit>{};
    for (const auto& sample : geometry.tireSurface)
    {
        hits.push_back(SurfaceHit{.point = glm::dvec3(sample.x, height, sample.z),
                                  .normal = glm::dvec3(0.0, 1.0, 0.0),
                                  .distance = 0.0,
                                  .surface = 0,
                                  .hit = true});
    }

    return hits;
}

[[nodiscard]] std::vector<SurfaceHit> overhangHits(const ContactSampleGeometry& geometry, const double height)
{
    auto hits = flatHits(geometry, height);
    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        hits[index].hit = geometry.tireSurface[index].x <= 1e-9;
    }

    return hits;
}

// The bed's inputs, extracted the same way the aggregate extracts them, so the solver is scoped on
// exactly what it is fed in anger rather than on something shaped like it.
struct Problem
{
    std::vector<double> gaps;
    std::vector<char> constrained;
    std::vector<char> cutAcross;
    std::vector<char> cutAlong;
    double shearAcross = 0.0;
    double shearAlong = 0.0;
};

[[nodiscard]] Problem problemFrom(const ContactSampleGeometry& geometry, const std::vector<SurfaceHit>& hits,
                                  const ContactPatchSampling& sampling)
{
    auto problem = Problem{};
    problem.gaps.assign(hits.size(), 0.0);
    problem.constrained.assign(hits.size(), char{0});
    problem.cutAcross.assign(hits.size(), char{0});
    problem.cutAlong.assign(hits.size(), char{0});

    auto touching = std::vector<double>{};
    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        if (!hits[index].hit)
        {
            continue;
        }

        problem.gaps[index] = hits[index].point.y - geometry.tireSurface[index].y;
        if (problem.gaps[index] > 0.0)
        {
            touching.push_back(problem.gaps[index]);
        }
    }

    REQUIRE_FALSE(touching.empty());
    std::sort(touching.begin(), touching.end());
    const auto ceiling = touching[touching.size() / 2] + sampling.spikeRejection;

    for (auto index = std::size_t{0}; index < hits.size(); index++)
    {
        problem.constrained[index] =
            hits[index].hit && problem.gaps[index] > 0.0 && problem.gaps[index] <= ceiling ? char{1} : char{0};
    }

    const auto breakBetween = [&](const std::size_t a, const std::size_t b)
    {
        if (!hits[a].hit || !hits[b].hit)
        {
            return char{0};
        }

        const auto cosine = std::clamp(glm::dot(hits[a].normal, hits[b].normal), -1.0, 1.0);

        return std::acos(cosine) > sampling.beltBreakAngle ? char{1} : char{0};
    };

    for (auto row = std::uint32_t{0}; row < sampling.along; row++)
    {
        for (auto column = std::uint32_t{0}; column + 1 < sampling.across; column++)
        {
            const auto left = static_cast<std::size_t>(row) * sampling.across + column;
            problem.cutAcross[left] = breakBetween(left, left + 1);
        }
    }

    for (auto row = std::uint32_t{0}; row + 1 < sampling.along; row++)
    {
        for (auto column = std::uint32_t{0}; column < sampling.across; column++)
        {
            const auto near = static_cast<std::size_t>(row) * sampling.across + column;
            problem.cutAlong[near] = breakBetween(near, near + sampling.across);
        }
    }

    problem.shearAcross = beltShearRate(sampling.width / static_cast<double>(std::max(sampling.across, 2u) - 1),
                                        sampling.beltBridgingLength);
    problem.shearAlong = beltShearRate(sampling.length / static_cast<double>(std::max(sampling.along, 2u) - 1),
                                       sampling.beltBridgingLength);

    return problem;
}

[[nodiscard]] double meanDeflection(const raceengine::BeltBed& bed)
{
    auto total = 0.0;
    for (const auto deflection : bed.deflection)
    {
        total += deflection;
    }

    return total / static_cast<double>(bed.deflection.size());
}

// **The equilibrium residual, and it costs nothing to take.** The sum of the contact pressures equals
// the sum of the deflections exactly at the solution — every shear term appears twice with opposite
// signs and cancels — so the gap between those two sums is a direct measure of how far from solved the
// bed is, in the units the answer is reported in. It needs no reference solution to compute.
[[nodiscard]] double equilibriumResidual(const raceengine::BeltBed& bed)
{
    auto deflection = 0.0;
    auto pressure = 0.0;

    for (auto index = std::size_t{0}; index < bed.deflection.size(); index++)
    {
        deflection += bed.deflection[index];
        pressure += bed.pressure[index];
    }

    return deflection > 0.0 ? std::abs(pressure - deflection) / deflection : 0.0;
}

} // namespace

TEST_CASE("what a sweep of projected Gauss-Seidel buys", "[.envelope-solver]")
{
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    const auto radius = golfRadius();
    const auto rate = golfTyreRate();

    std::printf("\n=== load against sweep count, at 7x3, against a 20000-sweep reference ===\n");
    std::printf("  the hardest case the geometry offers is the one with the most elements changing\n"
                "  their mind about whether they are touching, so the chamfer and the kerb's two\n"
                "  edges are all here, and so is the strongest coupling any grid here produces.\n");

    for (const auto bridging : {0.015, 0.030, 0.060, 0.120})
    {
        auto sampling = ContactPatchSampling{};
        sampling.across = 7;
        sampling.along = 3;
        sampling.beltBridgingLength = bridging;

        std::printf("\n  bridging length %.0f mm: shear across %.4f of an element's radial rate, along %.4f\n",
                    1000.0 * bridging,
                    beltShearRate(sampling.width / static_cast<double>(sampling.across - 1), bridging),
                    beltShearRate(sampling.length / static_cast<double>(sampling.along - 1), bridging));
        std::printf("\n%12s %10s", "x", "reference");
        for (const auto sweeps : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
        {
            std::printf(" %7u", sweeps);
        }
        std::printf("   %12s\n", "resid at 8");

        for (const auto x : {2.85, 3.00, 3.05, 3.10, 3.20, 3.28, 3.32, 3.45})
        {
            const auto placed = placeAt(world.value(), descriptor, sampling, x, radius);
            const auto problem = problemFrom(placed.geometry, placed.hits, sampling);

            const auto solveWith = [&](const std::uint32_t sweeps)
            {
                return solveBeltBed(problem.gaps, problem.constrained, problem.cutAcross, problem.cutAlong,
                                    sampling.across, sampling.along, problem.shearAcross, problem.shearAlong, sweeps);
            };

            const auto reference = rate * meanDeflection(solveWith(20000));
            std::printf("%12.3f %10.1f", x, reference);

            for (const auto sweeps : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
            {
                std::printf(" %7.2f", rate * meanDeflection(solveWith(sweeps)) - reference);
            }

            std::printf("   %12.2e\n", equilibriumResidual(solveWith(8)));
        }
    }

    std::printf("\n  the columns after `reference` are the error in newtons, not the load.\n");
}

TEST_CASE("what the belt's one number does to the four cases acceptance is stated in", "[.envelope-solver]")
{
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    const auto radius = golfRadius();
    const auto rate = golfTyreRate();

    auto base = ContactPatchSampling{};
    base.across = 7;
    base.along = 3;

    std::printf("\n=== the bridging length swept, at 7x3, eight sweeps ===\n");
    std::printf("  flat      : well inboard of the kerb, every element on the road.\n"
                "  chamfer   : the middle of the 9.46-degree ramp, no break under the patch at all.\n"
                "  inner edge: the break between flat road and the ramp lies inside the patch.\n"
                "  top edge  : the break between the ramp and the kerb's flat top lies inside it.\n"
                "  overhang  : half the patch over nothing, on level ground -- the case that must\n"
                "              still shed load, and the one every over-eager bridging rule fails.\n");
    std::printf("\n%10s %10s %10s %12s %10s %10s %12s\n", "L mm", "flat N", "chamfer N", "inner edge", "top edge",
                "overhang", "overhang %");

    for (const auto bridging : {0.0, 0.010, 0.020, 0.025, 0.030, 0.040, 0.060, 0.100, 0.200})
    {
        auto sampling = base;
        sampling.beltBridgingLength = bridging;

        const auto loadAt = [&](const double x)
        {
            const auto placed = placeAt(world.value(), descriptor, sampling, x, radius);

            return rate *
                   aggregateContactPatch(placed.geometry, placed.hits, defaultSurfaceMaterials(), sampling).penetration;
        };

        // The overhang is taken at the compression that makes the *uncoupled* full patch read the
        // same as flat ground at 15 mm, so the two columns are comparable rather than two different
        // wheels. Upright, level, and everything outboard of the middle column over a void.
        const auto upright = WheelPose{.centre = glm::dvec3(0.0, radius - 0.025, 0.0),
                                       .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                       .forward = glm::dvec3(0.0, 0.0, 1.0),
                                       .radius = radius};
        const auto uprightGeometry = contactPatchSamples(upright, sampling);

        const auto whole = rate * aggregateContactPatch(uprightGeometry, flatHits(uprightGeometry, 0.0),
                                                        defaultSurfaceMaterials(), sampling)
                                      .penetration;
        const auto hanging = rate * aggregateContactPatch(uprightGeometry, overhangHits(uprightGeometry, 0.0),
                                                          defaultSurfaceMaterials(), sampling)
                                        .penetration;

        std::printf("%10.0f %10.0f %10.0f %12.0f %10.0f %10.0f %11.0f%%\n", 1000.0 * bridging, loadAt(2.85),
                    loadAt(3.10), loadAt(3.00), loadAt(3.31), hanging, 100.0 * hanging / whole);
    }

    std::printf("\n  the overhang column is a wheel at 25 mm rather than the held 15 mm of the others,\n"
                "  and its percentage is against the same wheel with the void filled in. Four of nine\n"
                "  columns of the grid carry, so an uncoupled bed reads 4/7 of the whole.\n");
}

TEST_CASE("the load a wheel at a fixed ride height meets as it crosses the kerb", "[.envelope-solver]")
{
    // **The control variable, changed on purpose, and it is the measurement the whole E-series turns
    // on.**
    //
    // Every figure recorded for the kerb-edge load dip — 2450 N on the flat against 898 on the
    // chamfer, a 1755 N phantom unload — is taken with the wheel placed so its *deepest sample*
    // penetrates exactly 15 mm. That control was chosen for a good reason: it is the only one under
    // which a change in the reported number is the sampler's rather than the placement's, which is
    // what a convergence study needs.
    //
    // It is not, however, a control any driving car has. A wheel crossing a kerb is held by a spring
    // at roughly a *height*, and the road comes up to meet it. Under a held deepest sample the wheel
    // on a cross-slope is barely touching — only its up-slope shoulder is on the road at all, and the
    // road under the middle of its patch is fourteen millimetres *below* the tread — so a low load
    // there is not obviously a defect. Under a held height it is not, and the two controls can
    // disagree about the sign.
    //
    // So: the wheel is put at one height and walked across the kerb, which is what the car does. What
    // a driver feels is not the level of this curve but its *roughness* — a kink in the load is a tug
    // through the steering — so the second difference is the number to read, and it is reported
    // against the first difference so that a smooth ramp and a staircase are told apart.
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    const auto radius = golfRadius();
    const auto rate = golfTyreRate();

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    // The height the front of this car actually rides at: 1348 kg at 61.4% front over two wheels is
    // 4061 N, which on a 298,926 N/m tyre is 13.6 mm of mean compression. Solved rather than stated,
    // so the number follows the car.
    const auto rideHeight = [&](const ContactPatchSampling& sampling)
    {
        auto low = radius - 0.10;
        auto high = radius + 0.01;

        for (auto iteration = 0; iteration < 60; iteration++)
        {
            const auto middle = 0.5 * (low + high);
            const auto geometry = contactPatchSamples(WheelPose{.centre = glm::dvec3(0.0, middle, 0.0),
                                                                .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                                                .forward = glm::dvec3(0.0, 0.0, 1.0),
                                                                .radius = radius},
                                                      sampling);
            const auto patch =
                aggregateContactPatch(geometry, flatHits(geometry, 0.0), defaultSurfaceMaterials(), sampling);

            (rate * patch.penetration > 4061.0 ? low : high) = middle;
        }

        return 0.5 * (low + high);
    };

    std::printf("\n=== a wheel walked across the kerb at one height, 0.5 mm at a time ===\n");
    std::printf("  the height is set per configuration so that each reads 4061 N on flat road, which\n"
                "  is what this car's front corner actually carries. Otherwise the curves would be\n"
                "  compared at different loads and the roughness would scale with the difference.\n");
    std::printf("\n%8s %8s %12s %14s %14s %14s %14s\n", "grid", "L mm", "flat N", "worst dF N", "worst ddF N",
                "worst dC mm", "worst ddC mm");

    for (const auto [across, bridging] : std::array<std::pair<std::uint32_t, double>, 6>{
             {{3, 0.0}, {7, 0.0}, {7, 0.020}, {7, 0.030}, {7, 0.040}, {7, 0.060}}})
    {
        auto sampling = ContactPatchSampling{};
        sampling.across = across;
        sampling.along = 3;
        sampling.beltBridgingLength = bridging;
        sampling.beltIterations = 32;

        const auto height = rideHeight(sampling);

        auto loads = std::vector<double>{};
        auto centres = std::vector<double>{};

        for (auto step = 0; step <= 800; step++)
        {
            const auto x = descriptor.kerbInnerEdge - 0.20 + 0.0005 * static_cast<double>(step);

            const auto beneath = sampleProvingGround(descriptor, x, middleZ);
            const auto geometry =
                contactPatchSamples(WheelPose{.centre = glm::dvec3(x, beneath.height + height, middleZ),
                                              .spinAxis = spinAxis,
                                              .forward = forward,
                                              .radius = radius},
                                    sampling);

            auto hits = std::vector<SurfaceHit>{};
            world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

            const auto patch = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);

            loads.push_back(rate * patch.penetration);
            centres.push_back(patch.inContact ? patch.centre.x - x : 0.0);
        }

        const auto worstDifference = [](const std::vector<double>& values, const std::size_t order)
        {
            auto worst = 0.0;
            for (auto index = order; index < values.size(); index++)
            {
                const auto difference = order == 1 ? values[index] - values[index - 1]
                                                   : values[index] - 2.0 * values[index - 1] + values[index - 2];
                worst = std::max(worst, std::abs(difference));
            }

            return worst;
        };

        std::printf("%6ux3 %8.0f %12.0f %14.1f %14.2f %14.3f %14.4f\n", across, 1000.0 * bridging, loads.front(),
                    worstDifference(loads, 1), worstDifference(loads, 2), 1000.0 * worstDifference(centres, 1),
                    1000.0 * worstDifference(centres, 2));
    }

    std::printf("\n  dF is the load change per half millimetre of travel and is the ramp; ddF is the\n"
                "  change in that ramp and is the kink. The kink is what a driver meets.\n");

    // And the same question asked of the level, so the two controls can be read side by side.
    std::printf("\n=== the same crossing's load profile, held height against held deepest sample ===\n");
    std::printf("\n%10s %16s %16s %16s\n", "x", "held height N", "held deepest N", "road rise mm");

    auto sampling = ContactPatchSampling{};
    sampling.across = 7;
    sampling.along = 3;
    sampling.beltIterations = 32;
    const auto height = rideHeight(sampling);

    for (auto step = 0; step <= 16; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.03 * static_cast<double>(step);
        const auto beneath = sampleProvingGround(descriptor, x, middleZ);

        const auto geometry = contactPatchSamples(WheelPose{.centre = glm::dvec3(x, beneath.height + height, middleZ),
                                                            .spinAxis = spinAxis,
                                                            .forward = forward,
                                                            .radius = radius},
                                                  sampling);

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

        const auto held = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);

        const auto placed = placeAt(world.value(), descriptor, sampling, x, radius);
        const auto deepest = aggregateContactPatch(placed.geometry, placed.hits, defaultSurfaceMaterials(), sampling);

        std::printf("%10.3f %16.0f %16.0f %16.1f\n", x, rate * held.penetration, rate * deepest.penetration,
                    1000.0 * beneath.height);
    }

    std::printf("\n  If the held-height column has no dip, the 1755 N phantom unload is a property of\n"
                "  the other control and not of the spring bed, and the enveloping model was never\n"
                "  the thing that would remove it.\n");
}

TEST_CASE("whether the belt's response depends on how finely it is asked", "[.envelope-solver]")
{
    // The coupling is supposed to be a property of the tyre. If the answer moves when the grid does,
    // it is a property of the grid instead, and the bridging length is a tuning knob wearing a
    // physical name. The exact discrete shear rate is what this is testing: the continuum `(L/dx)^2`
    // it approximates would not hold this.
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    const auto radius = golfRadius();
    const auto rate = golfTyreRate();

    std::printf("\n=== the same wheel in the same place, at six grids, L = 30 mm ===\n");
    std::printf("\n%8s %10s %12s %12s %12s %14s\n", "grid", "spacing mm", "chamfer off", "chamfer on", "on / off",
                "shear/radial");

    for (const auto across : {3u, 5u, 7u, 9u, 15u, 31u})
    {
        auto sampling = ContactPatchSampling{};
        sampling.across = across;
        sampling.along = 3;
        sampling.beltIterations = 256;

        const auto loadAt = [&](const double x)
        {
            const auto placed = placeAt(world.value(), descriptor, sampling, x, radius);

            return rate *
                   aggregateContactPatch(placed.geometry, placed.hits, defaultSurfaceMaterials(), sampling).penetration;
        };

        // **Against the same grid uncoupled, and not against flat ground.** The chamfer's own
        // quadrature is known not to have converged even at 31 samples, so a ratio taken against flat
        // ground moves for two reasons at once and cannot say which. Coupled over uncoupled at the
        // same grid isolates what the belt did, and *that* is what has to hold still if the bridging
        // length is a property of the tyre.
        sampling.beltBridgingLength = 0.0;
        const auto off = loadAt(3.10);

        sampling.beltBridgingLength = 0.030;
        const auto on = loadAt(3.10);

        const auto spacing = sampling.width / static_cast<double>(across - 1);

        std::printf("%6ux3 %10.1f %12.0f %12.0f %12.3f %14.4f\n", across, 1000.0 * spacing, off, on, on / off,
                    beltShearRate(spacing, sampling.beltBridgingLength));
    }

    std::printf("\n  `on / off` is what has to converge. If it drifts, the bridging length is a\n"
                "  property of the grid wearing a physical name.\n");
}

TEST_CASE("what the belt solver costs against the tick budget", "[.envelope-solver]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(kerbGround()).value());
    REQUIRE(world.has_value());

    const auto radius = golfRadius();

    std::printf("\n=== four wheels' worth of sampling, split by phase, with the belt in it ===\n");
    std::printf("\n%10s %8s %12s %12s %12s %12s %12s\n", "grid", "sweeps", "build us", "cast us", "aggregate",
                "total us", "of 50 us");

    for (const auto [across, along, sweeps] : std::array<std::array<std::uint32_t, 3>, 8>{
             {{3, 3, 0}, {3, 3, 8}, {7, 3, 0}, {7, 3, 4}, {7, 3, 8}, {7, 3, 16}, {7, 3, 32}, {5, 5, 8}}})
    {
        auto sampling = ContactPatchSampling{};
        sampling.across = across;
        sampling.along = along;
        sampling.beltIterations = sweeps;
        sampling.beltBridgingLength = sweeps > 0 ? 0.030 : 0.0;

        auto poses = std::array<WheelPose, 4>{};
        for (auto index = std::size_t{0}; index < 4; index++)
        {
            poses[index] = WheelPose{
                .centre = glm::dvec3(index % 2 == 0 ? -0.78 : 0.78, radius - 0.015, 10.0 + (index < 2 ? 1.3 : -1.3)),
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

        const auto median = [](std::vector<double> samples)
        {
            std::sort(samples.begin(), samples.end());

            return samples[samples.size() / 2];
        };

        const auto b = median(build);
        const auto c = median(cast);
        const auto a = median(aggregate);

        std::printf("%8ux%-2u %8u %12.2f %12.2f %12.2f %12.2f %11.0f%%\n", across, along, sweeps, b, c, a, b + c + a,
                    100.0 * (b + c + a) / 50.0);
    }
}
