#include <algorithm>
#include <cmath>
#include <cstdint>
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

// Ground tilted across the wheel — the kerb chamfer's own shape, without the kerb. `slope` is the
// fall per metre travelled along the spin axis, so a sample outboard of the wheel's middle finds the
// road lower than one inboard of it, which is precisely the case a bed of independent springs cannot
// carry.
std::vector<SurfaceHit> crossSlopedGround(const raceengine::ContactSampleGeometry& geometry, const double slope,
                                          const std::uint32_t surface = 0)
{
    auto hits = std::vector<SurfaceHit>{};
    for (const auto& sample : geometry.tireSurface)
    {
        const auto height = -slope * sample.x;

        hits.push_back(SurfaceHit{.point = glm::dvec3(sample.x, height, sample.z),
                                  .normal = glm::normalize(glm::dvec3(slope, 1.0, 0.0)),
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

// How far the tread has curved above the patch's middle at the leading and trailing rows. It is the
// spread a level patch reports on flat ground, so several cases below are stated against it rather
// than against a number somebody measured once.
[[nodiscard]] double treadRise(const double radius, const double length)
{
    return radius - std::sqrt(radius * radius - 0.25 * length * length);
}

// **The bed as it was before the belt, stated rather than inherited.**
//
// Most of what follows pins the *aggregation's arithmetic* — the sagitta of the tread, what a depth
// spread means, what a divisor does to a patch with a hole in it. That arithmetic is a property of
// the algorithm, not of whichever tyre the car happens to be wearing, so these cases state the grid
// and the belt they are asking about instead of picking up the shipped car's. Inheriting the default
// is what put ten of them red the day the Golf's tyre gained a belt, and re-deriving a statement
// about a sagitta against a new vehicle number is how a test stops being a statement about anything.
//
// The cases that pin the *car's* behaviour rather than the algorithm's — the mesh-kerb crossing, and
// the held-height section of the characterisation below — deliberately do the opposite and take the
// default, because what they are for is noticing when the shipped car moves.
[[nodiscard]] ContactPatchSampling uncoupled()
{
    auto sampling = ContactPatchSampling{};
    sampling.across = 3;
    sampling.along = 3;
    sampling.beltBridgingLength = 0.0;

    return sampling;
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

// How far below its free radius an upright wheel's centre has to sit on flat ground to carry a stated
// load, **solved rather than stated**. The reported compression is a mean over the grid and most of
// the grid is shallower than the deepest sample, so the two differ by a factor of about two here and
// writing down either one as though it were the other is a fifty per cent error in the load. That is
// not hypothetical: it was done, and the assertion that caught it is the one in the case below that
// checks this returns the load it was asked for.
[[nodiscard]] double heightForLoad(const ContactPatchSampling& sampling, const double radius, const double rate,
                                   const double load)
{
    auto low = radius - 0.15;
    auto high = radius;

    for (auto iteration = 0; iteration < 60; iteration++)
    {
        const auto middle = 0.5 * (low + high);
        const auto wheel = WheelPose{.centre = glm::dvec3(0.0, middle, 0.0),
                                     .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                     .forward = glm::dvec3(0.0, 0.0, 1.0),
                                     .radius = radius};
        const auto geometry = contactPatchSamples(wheel, sampling);
        const auto patch =
            aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

        (rate * patch.penetration > load ? low : high) = middle;
    }

    return 0.5 * (low + high);
}

} // namespace

TEST_CASE("the sample grid covers the patch and follows the tread", "[physics][contact]")
{
    const auto sampling = uncoupled();
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
    const auto sampling = uncoupled();
    const auto geometry = contactPatchSamples(uprightWheel(0.50), sampling);
    const auto patch = aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

    REQUIRE_FALSE(patch.inContact);
    REQUIRE(patch.contactingSamples == 0);
    REQUIRE(patch.penetration == 0.0);
}

TEST_CASE("a wheel square on flat ground is centred and level", "[physics][contact]")
{
    const auto sampling = uncoupled();

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
    const auto sampling = uncoupled();
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

TEST_CASE("a level patch's depth spread is the tread's own curvature", "[physics][contact][spread]")
{
    // The floor the channel reads from, and it is not zero. The tread curves away fore and aft, so
    // even on perfectly flat ground the leading and trailing rows sit shallower than the middle one
    // by exactly the sagitta of the tread over the patch's length. Anything reading this channel has
    // to know that number is always there, or it will read a level road as a tilted one.
    const auto sampling = uncoupled();
    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);
    const auto patch = aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

    REQUIRE(patch.contactingSamples == 9);
    REQUIRE(patch.depthSpread == Catch::Approx(treadRise(0.31, sampling.length)));
    // About ten and a half millimetres, stated so a reader does not have to evaluate the sagitta to
    // know the order of magnitude of the thing this channel sits on top of.
    REQUIRE(patch.depthSpread == Catch::Approx(0.0105).margin(0.0002));
}

TEST_CASE("a patch too shallow to reach the tread's curve has no spread", "[physics][contact][spread]")
{
    // Below the tread's own rise only the middle row touches at all, and across the width the tyre is
    // a cylinder — no curvature, so every touching sample is at the same depth. Exactly zero, and it
    // is the other end of the channel's range.
    const auto sampling = uncoupled();
    const auto compression = 0.5 * treadRise(0.31, sampling.length);
    const auto geometry = contactPatchSamples(uprightWheel(0.31 - compression), sampling);
    const auto patch = aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

    REQUIRE(patch.contactingSamples == 3);
    REQUIRE(patch.depthSpread == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("the same load on the same sample count, told apart by the spread", "[physics][contact][spread]")
{
    // **Why the channel exists, stated as the case that motivates it.**
    //
    // Two wheels. One is resting lightly on flat road; the other is loaded onto a road tilted at the
    // kerb chamfer's own 9.46 degrees. They are constructed here to report *the same* penetration —
    // which is to say the same `Tyre Fz`, since load is `tireVerticalRate * penetration` — and the
    // same number of contacting samples. Neither `Fz` nor `Contact Samples` can tell them apart, and
    // one of them is a tyre that genuinely has almost no load on it while the other is a tyre the
    // spring bed is under-reporting because it cannot carry load across a cross-slope.
    //
    // The spread separates them, and it is the only channel in the frame that does.
    const auto sampling = uncoupled();
    const auto materials = defaultSurfaceMaterials();
    constexpr auto radius = 0.31;
    // tan(9.46 deg) — the chamfer's cross-slope, so this is the probe's own geometry.
    constexpr auto slope = 0.16657;

    // Placed so exactly three samples carry, which is what the chamfer itself produces at the
    // compressions the probe holds.
    const auto tiltedGeometry = contactPatchSamples(uprightWheel(radius + 0.003), sampling);
    const auto tilted =
        aggregateContactPatch(tiltedGeometry, crossSlopedGround(tiltedGeometry, slope), materials, sampling);

    REQUIRE(tilted.contactingSamples == 3);

    // The flat wheel is then placed to match the tilted one's reported penetration exactly: three
    // samples at one depth over a divisor of nine, so the compression that matches is three times it.
    const auto compression = 3.0 * tilted.penetration;
    REQUIRE(compression < treadRise(radius, sampling.length));

    const auto flatGeometry = contactPatchSamples(uprightWheel(radius - compression), sampling);
    const auto flat = aggregateContactPatch(flatGeometry, flatGround(flatGeometry, 0.0), materials, sampling);

    // Indistinguishable in both of the channels that existed before this one.
    REQUIRE(flat.contactingSamples == tilted.contactingSamples);
    REQUIRE(flat.penetration == Catch::Approx(tilted.penetration).epsilon(1e-9));

    // And separated in the one that does not.
    REQUIRE(flat.depthSpread == Catch::Approx(0.0).margin(1e-15));
    REQUIRE(tilted.depthSpread > 0.008);

    // **What this case does *not* show, and it matters for how the channel is read.** The three
    // samples carrying the tilted wheel all sit in the same column of the grid, so the spread they
    // report is the tread's own rise and not the road's tilt — the tilt did its work by deciding
    // *which* samples touch rather than by appearing in their depths. So the pair is the
    // discriminator and neither half is one alone: a fully contacting patch on flat ground reads
    // this same spread at nine samples, and it is the count that separates that from this.
    REQUIRE(tilted.depthSpread == Catch::Approx(treadRise(radius, sampling.length)).epsilon(1e-6));
}

TEST_CASE("a cambered wheel spreads its patch on perfectly level ground", "[physics][contact][spread]")
{
    // **The correction to the obvious reading of this channel, and it is not a detail.**
    //
    // A level patch's spread looked like it was the tread's own rise and nothing else, which would
    // have made "spread above the rise" a sharp, parameter-free test for a road that is not level.
    // It is not: a cambered wheel leans its patch across its width, so on perfectly level ground the
    // depths vary linearly along the spin axis as well as quadratically along the tread. The ceiling
    // is `width * sin(camber) + rise * cos(camber)`, and both terms are properties of the *wheel*.
    // The cosine is on the rise and not on the lean, because the tread's curve is measured in the
    // wheel's own plane and it is the vertical *projection* of it that shortens as the wheel leans.
    //
    // Measured on the scripted launch, this is most of what the channel reads on flat road: the four
    // corners sit at 14.25, 14.57, 11.63 and 12.92 mm against a 10.21 mm rise, and each is constant
    // to within about 0.4 mm from the tenth percentile to the ninetieth. A rule keyed on the rise
    // alone fires on 97.8% of a flat launch and is worth nothing. A rule keyed on this ceiling has a
    // near-noiseless baseline to work against, which is what makes the channel usable at all.
    const auto sampling = uncoupled();
    constexpr auto radius = 0.31;

    for (const auto camber : {0.0, 0.5, 1.0, 2.0})
    {
        const auto radians = camber * 3.14159265358979323846 / 180.0;

        // Leant about the direction of travel, which is what camber is. Deep enough that the whole
        // grid stays in contact at every angle here, so the spread is the geometry's and not an
        // artefact of samples dropping out.
        const auto wheel = WheelPose{.centre = glm::dvec3(0.0, radius - 0.025, 0.0),
                                     .spinAxis = glm::dvec3(std::cos(radians), std::sin(radians), 0.0),
                                     .forward = glm::dvec3(0.0, 0.0, 1.0),
                                     .radius = radius};

        const auto geometry = contactPatchSamples(wheel, sampling);
        const auto patch =
            aggregateContactPatch(geometry, flatGround(geometry, 0.0), defaultSurfaceMaterials(), sampling);

        CAPTURE(camber, patch.contactingSamples, patch.depthSpread);
        REQUIRE(patch.contactingSamples == 9);

        const auto expected =
            sampling.width * std::sin(radians) + treadRise(radius, sampling.length) * std::cos(radians);
        REQUIRE(patch.depthSpread == Catch::Approx(expected).epsilon(1e-6));
    }
}

TEST_CASE("a rejected spike does not enter the depth spread", "[physics][contact][spread]")
{
    // The spread is taken over the load-carrying set, which is the set `contactingSamples` counts and
    // therefore the set that survives spike rejection. A mesh fault reporting the road half a metre
    // up must not be allowed to report a half-metre spread — that would put the one sample nothing
    // else believes into the channel built to decide what the tyre model should be.
    const auto sampling = uncoupled();
    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);
    const auto materials = defaultSurfaceMaterials();

    auto hits = flatGround(geometry, 0.0);
    const auto clean = aggregateContactPatch(geometry, hits, materials, sampling);

    hits[0].point.y = 0.5;
    const auto spiked = aggregateContactPatch(geometry, hits, materials, sampling);

    REQUIRE(spiked.contactingSamples == clean.contactingSamples - 1);
    REQUIRE(spiked.depthSpread < sampling.spikeRejection);
    REQUIRE(spiked.depthSpread == Catch::Approx(clean.depthSpread));
}

TEST_CASE("a patch straddling two surfaces blends their grip", "[physics][contact]")
{
    const auto sampling = uncoupled();
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
    const auto sampling = uncoupled();
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
    const auto sampling = uncoupled();
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

TEST_CASE("the belt is the bed of independent springs at zero bridging length", "[physics][contact][belt]")
{
    // **The property that makes the coupled bed provable rather than merely plausible**, and the
    // reason the parameter's zero is a real setting rather than a degenerate one. Every quantity the
    // aggregate reports has to come back exactly — not nearly — when the belt is given no length to
    // spread a load over, because with no coupling each row of the problem is `w = max(0, g)` and
    // that is what the bed it replaced computed directly.
    //
    // Stated over four geometries rather than one, because the interesting cases are the ones where
    // the two models *could* differ: a patch with elements missing, a patch on a cross-slope, and a
    // patch with a spike in it.
    const auto materials = defaultSurfaceMaterials();
    auto sampling = ContactPatchSampling{};
    sampling.beltBridgingLength = 0.0;

    const auto geometry = contactPatchSamples(uprightWheel(0.29), sampling);

    auto withHole = flatGround(geometry, 0.0);
    for (const auto index : {2u, 5u, 8u})
    {
        withHole[index].hit = false;
    }

    auto withSpike = flatGround(geometry, 0.0);
    withSpike[0].point.y = 0.5;

    for (const auto& hits : {flatGround(geometry, 0.0), withHole, withSpike, crossSlopedGround(geometry, 0.16657)})
    {
        const auto patch = aggregateContactPatch(geometry, hits, materials, sampling);

        // The bed it replaced, evaluated here rather than remembered: the mean depth over the whole
        // grid, weighting the plane and the centre by that same depth.
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

        REQUIRE_FALSE(touching.empty());
        std::sort(touching.begin(), touching.end());
        const auto ceiling = touching[touching.size() / 2] + sampling.spikeRejection;

        auto summed = 0.0;
        auto weight = 0.0;
        auto centre = glm::dvec3(0.0);
        auto counted = std::uint32_t{0};

        for (auto index = std::size_t{0}; index < hits.size(); index++)
        {
            if (!hits[index].hit)
            {
                continue;
            }

            const auto depth = hits[index].point.y - geometry.tireSurface[index].y;
            if (depth <= 0.0 || depth > ceiling)
            {
                continue;
            }

            summed += depth;
            weight += depth;
            centre += depth * hits[index].point;
            counted++;
        }

        CAPTURE(counted, patch.contactingSamples);
        REQUIRE(patch.contactingSamples == counted);
        REQUIRE(patch.penetration == summed / static_cast<double>(hits.size()));
        REQUIRE(patch.centre.x == Catch::Approx((centre / weight).x).margin(1e-15));
    }
}

TEST_CASE("a patch pressing everywhere carries the same load however stiff the belt", "[physics][contact][belt]")
{
    // **The identity the whole model rests on, and it is exact arithmetic rather than a tolerance.**
    //
    // Summing every row of the bed, each shear term appears twice with opposite signs and cancels, so
    // the total contact pressure is the total deflection whatever the coupling is worth. While every
    // element is pressing on something that means the coupling **redistributes pressure and cannot
    // invent load** — which is precisely the promise none of the four refused candidates could make,
    // and it is why flat ground is untouched by a change aimed at kerbs.
    //
    // Flat ground and a cross-slope the whole patch stays on are both here. The cross-slope matters:
    // a tilted road is the case the enveloping model exists for, and as long as the patch is still
    // pressing everywhere the answer must not move at all.
    const auto materials = defaultSurfaceMaterials();
    constexpr auto radius = 0.31;

    // Deep enough that every one of the nine elements is on the road at every tilt below.
    const auto wheel = uprightWheel(radius - 0.030);

    // **The theorem has a precondition and the sweep runs past it deliberately.** A belt stiff enough
    // relative to the patch stops being held down by the road everywhere: the deep middle row drags
    // the leading and trailing rows inward until they lift clear, the contact patch shortens, and the
    // sum is no longer over a fully constrained set. That is a real and correct behaviour of a stiff
    // carcass rather than a break in the model, and it moves the load *up* — the rim feels every
    // radial spring, including the ones no longer resting on anything. Both sides are asserted, and
    // the last length is required to reach the far one so the boundary is exercised and not assumed.
    auto everLiftedOff = false;

    for (const auto slope : {0.0, 0.05, 0.10})
    {
        auto reference = 0.0;

        for (const auto bridging : {0.0, 0.010, 0.030, 0.100, 0.500})
        {
            auto sampling = ContactPatchSampling{};
            sampling.beltBridgingLength = bridging;
            sampling.beltIterations = 400;

            const auto geometry = contactPatchSamples(wheel, sampling);
            const auto hits = slope > 0.0 ? crossSlopedGround(geometry, slope) : flatGround(geometry, 0.0);
            const auto patch = aggregateContactPatch(geometry, hits, materials, sampling);

            CAPTURE(slope, bridging, patch.contactingSamples);

            reference = bridging == 0.0 ? patch.penetration : reference;
            REQUIRE(reference > 0.0);

            if (patch.contactingSamples == sampling.across * sampling.along)
            {
                REQUIRE(patch.penetration == Catch::Approx(reference).epsilon(1e-9));
            }
            else
            {
                everLiftedOff = true;
                REQUIRE(patch.penetration > reference);
            }
        }
    }

    REQUIRE(everLiftedOff);
}

TEST_CASE("the belt's shear rate gives it the response length it was asked for", "[physics][contact][belt]")
{
    // `beltShearRate` inverts the discrete decay exactly rather than using the continuum `(L/dx)^2`
    // it tends to, and this is what that buys: hold one element down and the rest of the belt follows
    // it as `exp(-dx / L)` per element, at whatever grid pitch, rather than only in the limit.
    //
    // It is the difference between a bridging length that is a property of the tyre and one that is a
    // property of how finely the tyre is being asked, and the second of those would be a tuning knob
    // wearing a physical name.
    constexpr auto elements = std::uint32_t{40};

    for (const auto spacingOverLength : {0.5, 1.0, 2.0})
    {
        constexpr auto bridging = 0.030;
        const auto spacing = spacingOverLength * bridging;

        auto gaps = std::vector<double>(elements, -1.0);
        auto constrained = std::vector<char>(elements, char{0});
        gaps.front() = 0.010;
        constrained.front() = char{1};

        const auto bed = raceengine::solveBeltBed(gaps, constrained, std::vector<char>(elements, char{0}),
                                                  std::vector<char>(elements, char{0}), elements, 1,
                                                  raceengine::beltShearRate(spacing, bridging), 0.0, 4000);

        // Away from both ends, where neither the held element nor the chain running out can be felt.
        for (auto index = std::size_t{2}; index < 8; index++)
        {
            CAPTURE(spacingOverLength, index, bed.deflection[index], bed.deflection[index - 1]);
            REQUIRE(bed.deflection[index] / bed.deflection[index - 1] ==
                    Catch::Approx(std::exp(-spacingOverLength)).epsilon(1e-6));
        }
    }
}

TEST_CASE("the belt solver is converged inside the budget it is given", "[physics][contact][belt]")
{
    // **The open risk the brief named, pinned.** A coupled bed is a unilateral contact problem, so it
    // is solved rather than evaluated, and a fixed iteration count is not negotiable: both parity
    // gates are byte-identical and an early-out on a convergence test makes the frame a function of
    // how the arithmetic rounded. So what the fixed budget buys has to be a stated, tested property.
    //
    // Measured at 7x3 over the kerb crossing (`./EngineTests "[.envelope-solver]"`): eight sweeps is
    // within 0.03 N of a twenty-thousand-sweep reference at a 30 mm bridging length, two sweeps are
    // enough at 15 mm, and thirty-two are needed by 120 mm. The budget is adequate for the lengths
    // this tyre would use and *not* for arbitrary ones, which is the honest statement of it.
    //
    // The residual is free to compute and needs no reference: the sum of the pressures equals the sum
    // of the deflections at the solution, so the gap between them measures how far from solved it is.
    const auto materials = defaultSurfaceMaterials();

    auto sampling = ContactPatchSampling{};
    sampling.across = 7;
    sampling.along = 3;
    sampling.beltBridgingLength = 0.030;

    // A cross-slope steep enough that most of the patch has lost the road, which is the active set
    // that makes the problem hard rather than the one that makes it big.
    const auto geometry = contactPatchSamples(uprightWheel(0.31 + 0.003), sampling);
    const auto hits = crossSlopedGround(geometry, 0.16657);

    sampling.beltIterations = 20000;
    const auto converged = aggregateContactPatch(geometry, hits, materials, sampling);

    sampling.beltIterations = 8;
    const auto budgeted = aggregateContactPatch(geometry, hits, materials, sampling);

    CAPTURE(converged.penetration, budgeted.penetration);
    REQUIRE(converged.penetration > 0.0);
    REQUIRE(budgeted.penetration == Catch::Approx(converged.penetration).epsilon(1e-4));
    REQUIRE(budgeted.contactingSamples == converged.contactingSamples);
}

TEST_CASE("a coupled belt still sheds load over a genuine overhang", "[physics][contact][belt]")
{
    // The case every over-eager bridging rule fails, and the reason the coupling is between adjacent
    // elements rather than a summary: a wheel with half its patch over nothing must carry less than a
    // wheel with the void filled in, however good the belt is at spanning a kerb.
    //
    // It carries *more* than the uncoupled bed said, and that is the point rather than a concession —
    // the belt over the void is dragged inward by the loaded half and its radial springs push back on
    // the rim. What bounds it is that the drag decays over the bridging length, so a void much wider
    // than that is still a void.
    const auto materials = defaultSurfaceMaterials();
    constexpr auto radius = 0.31;

    auto previous = 0.0;

    for (const auto bridging : {0.0, 0.020, 0.040, 0.080})
    {
        auto sampling = ContactPatchSampling{};
        sampling.beltBridgingLength = bridging;
        sampling.beltIterations = 400;

        const auto geometry = contactPatchSamples(uprightWheel(radius - 0.025), sampling);

        auto hanging = flatGround(geometry, 0.0);
        for (auto index = std::size_t{0}; index < hanging.size(); index++)
        {
            hanging[index].hit = geometry.tireSurface[index].x <= 1e-9;
        }

        const auto whole = aggregateContactPatch(geometry, flatGround(geometry, 0.0), materials, sampling);
        const auto over = aggregateContactPatch(geometry, hanging, materials, sampling);

        CAPTURE(bridging, over.penetration, whole.penetration);

        // Sheds load, always, and by a clear margin rather than by a rounding error.
        REQUIRE(over.penetration < 0.85 * whole.penetration);

        // And a longer belt bridges more of it, monotonically, which is the knob doing what its name
        // says rather than something incidental.
        REQUIRE(over.penetration > previous);
        previous = over.penetration;

        // The load has moved onto the supported side, which is the moment that rolls the car.
        REQUIRE(over.centre.x < whole.centre.x);
    }
}

TEST_CASE("a wheel crossing a mesh kerb meets no step in load or in its patch centre",
          "[physics][contact][world][kerb]")
{
    // **The mesh-kerb regression, and it is aimed at a class of fault rather than at a model.**
    //
    // A wheel is walked across a generated kerb at one height, half a millimetre at a time, and every
    // step is required to be small. What that catches is anything that makes the crossing
    // *discontinuous* — a lost raycast, a sample rejected for one step, an aggregate that changes
    // shape when the contacting set does. It is deliberately not a bound on the load itself: how much
    // load a wheel on a kerb should carry is a modelling question and this is a continuity question.
    //
    // It exists because a single lost ray of 31,851 was the entire reason `the imported car crosses a
    // kerb continuously` failed, and nothing in the suite would have caught that one directly. The
    // bounds below are about five times the largest step this crossing genuinely takes; a lost ray at
    // 3x3 costs roughly a ninth of the load and clears them by a factor of seven.
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

    const auto sampling = ContactPatchSampling{};
    const auto radius = golfGtiMk7().value().corners.front().hardpoints.wheelRadius;

    // Yawed, so the wheel meets the chamfer a sample at a time rather than a whole row at once, which
    // is the harder case and the one the crossing actually is.
    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));
    constexpr auto middleZ = 10.0;

    // The height this corner rides at, solved on the flat rather than written down.
    const auto rate = golfGtiMk7().value().corners.front().tireVerticalRate;
    const auto ride = heightForLoad(sampling, radius, rate, 4061.0);

    auto previousPenetration = 0.0;
    auto previousOffset = 0.0;
    auto first = true;
    auto worstLoadStep = 0.0;
    auto worstCentreStep = 0.0;
    auto sawKerb = false;

    for (auto step = 0; step <= 700; step++)
    {
        const auto x = descriptor.kerbInnerEdge - 0.15 + 0.0005 * static_cast<double>(step);

        // Held at one height above the road under the wheel's own middle, which is what a suspension
        // does. **Not the held deepest sample the characterisation probes use** — that control is
        // right for a convergence study and wrong for a continuity one, because it moves the wheel
        // by whatever it takes to keep one number fixed and hides a step in the placement.
        const auto beneath = raceengine::sampleProvingGround(descriptor, x, middleZ);
        const auto geometry = contactPatchSamples(WheelPose{.centre = glm::dvec3(x, beneath.height + ride, middleZ),
                                                            .spinAxis = spinAxis,
                                                            .forward = forward,
                                                            .radius = radius},
                                                  sampling);

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

        const auto patch = aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling);
        REQUIRE(patch.inContact);

        sawKerb = sawKerb || patch.centre.y > 0.01;

        const auto offset = patch.centre.x - x;

        if (!first)
        {
            worstLoadStep = std::max(worstLoadStep, std::abs(patch.penetration - previousPenetration));
            worstCentreStep = std::max(worstCentreStep, std::abs(offset - previousOffset));
        }

        previousPenetration = patch.penetration;
        previousOffset = offset;
        first = false;
    }

    // The fixture's own precondition: a sweep that never reached the kerb would pass this trivially.
    REQUIRE(sawKerb);

    // **Re-derived against the shipped tyre on 2026-08-22, not loosened to fit it.** This case takes
    // the default sampling on purpose — it is here to notice when the *car* moves — so giving the
    // Golf a coupled belt over a 7x3 patch legitimately moved both numbers: measured, 0.253 mm of
    // penetration and 2.25 mm of patch centre per half-millimetre of travel, against 0.037 mm and
    // 0.09 mm on the uncoupled 3x3 bed. The belt makes a kerb crossing step harder, which is the
    // thing it was turned on to do.
    //
    // **The margin is thinner than it looks and that is worth stating.** What this case exists to
    // catch is a lost raycast, and a lost ray costs a share of the load equal to one element: at 3x3
    // that was a ninth, about 450 N, and at 7x3 it is a twenty-first, about 190 N or 0.65 mm of
    // penetration. So the bound below sits between what the crossing genuinely does and what one
    // lost ray would do, with less room either side than before — the finer grid bought lateral
    // resolution and cost this detector sensitivity.
    //
    // **And it does not in fact catch that fault, which is worth saying rather than assuming.**
    // Checked by removing the retry in `raceengineJoltCastRays` and running the suite: two cases go
    // red, `a ray fired straight down at solid ground always finds it` and `the imported car crosses
    // a kerb continuously`, and this one stays green. The reason is that the dead band is a property
    // of particular coordinates and this sweep's ground has none in it. So this case guards the
    // *class* — any aggregation that steps when the contacting set changes — and the precise guard
    // for a lost ray is the PhysicsWorldTests case, which fires at the coordinates where one was.
    CAPTURE(worstLoadStep, worstCentreStep);
    REQUIRE(worstLoadStep < 0.00045);
    REQUIRE(worstCentreStep < 0.0045);
}

TEST_CASE("the kerb-edge load dip, characterised rather than fixed", "[physics][contact][known-issue]")
{
    // **The regression behind an open finding, and it pins three facts rather than a repair.**
    //
    // The symptom: a loaded wheel riding a kerb chamfer reports a large drop in vertical load while
    // its deepest compression is held constant. `docs/post-m2-remediation-brief.md` carries the
    // account.
    //
    // What this pins, so none of it has to be re-derived:
    //
    //   1. **Spike rejection never fires here.** It was the recorded cause and it is not.
    //   2. **The dip is this big under this control.** Any change to the aggregation moves it.
    //   3. **Refining the grid makes it worse, not better.** That rules out a quadrature too coarse
    //      for a 0.30 m chamfer.
    //   4. **And it is mostly the control** (added 2026-08-22, E2). The section at the end walks the
    //      same wheel across the same kerb at a held ride height instead, and there is no dip.
    //
    // **The conclusion that used to be drawn here was wrong and is worth stating as such**, because
    // it sequenced a milestone. Points 1 to 3 were read as naming the mechanism — a bed of
    // independent radial springs, unable to conform to a road tilted under the wheel — and a coupled
    // belt was built to fix it. It does not: at a 30 mm bridging length the chamfer reads 813 N
    // against 727 uncoupled at the same grid. Point 4 is why. A wheel placed so that its *deepest
    // sample* holds 15 mm on a 9.46-degree cross-slope is a wheel touching the road with one shoulder
    // and standing fourteen millimetres clear of it under the middle of its own patch, and it ought
    // to read a low load. The control is right for a convergence study, which is what it was chosen
    // for, and it is not a statement about a car.
    //
    // The full sweep, the per-sample depths and the convergence table are the probe next door:
    // `./EngineTests "[.patch-chamfer]"`; the belt and its refusal are `[.envelope-solver]`.
    const JoltGuard jolt;

    // Small on purpose: a 0.05 m grid is what resolves a 0.30 m chamfer, and at the proving
    // ground's usual 200 x 40 m that is three million cells to generate for a wheel that never
    // leaves one square metre of it.
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 20.0;
    descriptor.width = 12.0;
    descriptor.cellSize = 0.05;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 4.0, .to = 16.0}};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    constexpr auto compression = 0.015;
    // **The car's own radius, and it is no longer the 0.31 this was first measured at.** The Golf's
    // wheel radius was corrected to 0.3186 m on 2026-08-22, which moves the flat-ground reading from
    // 2391 N to 2450 N; the chamfer reading barely moves, so the dip gets slightly *deeper*. Read
    // from the setup rather than restated here, so the next correction cannot leave this behind.
    const auto radius = golfGtiMk7().value().corners.front().hardpoints.wheelRadius;
    constexpr auto yaw = 30.0 * 3.14159265358979323846 / 180.0;
    constexpr auto middleZ = 10.0;

    const auto spinAxis = glm::dvec3(std::cos(yaw), 0.0, -std::sin(yaw));
    const auto forward = glm::dvec3(std::sin(yaw), 0.0, std::cos(yaw));

    // Reported compression with the wheel placed so its **deepest** sample penetrates exactly the
    // asked-for amount. Holding the deepest sample rather than the wheel's height is what makes the
    // comparison mean anything: a finer grid finds a deeper point, so a held height would compare
    // different compressions and read the difference as convergence.
    const auto reportedAt = [&](const double x, const std::uint32_t count)
    {
        // **Explicitly the uncoupled bed, which is what every figure in this case was measured
        // against.** The shipped tyre has had a belt since 2026-08-22 and reads different numbers
        // here; those are the held-height section's business at the end. What this case is for is
        // preserving the historic measurement exactly, so that the correction to its *conclusion*
        // can be read against the numbers that produced it rather than against re-derived ones.
        auto sampling = uncoupled();
        sampling.across = count;
        sampling.along = count;

        auto geometry = raceengine::ContactSampleGeometry{};
        auto hits = std::vector<SurfaceHit>{};

        const auto cast = [&](const double height)
        {
            geometry = contactPatchSamples(WheelPose{.centre = glm::dvec3(x, height, middleZ),
                                                     .spinAxis = spinAxis,
                                                     .forward = forward,
                                                     .radius = radius},
                                           sampling);
            world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

            auto deepest = -sampling.searchDistance;
            for (auto index = std::size_t{0}; index < hits.size(); index++)
            {
                if (hits[index].hit)
                {
                    deepest = std::max(deepest, hits[index].point.y - geometry.tireSurface[index].y);
                }
            }

            return deepest;
        };

        // Twenty halvings of a 0.45 m bracket is under half a micron, which is four orders of
        // magnitude finer than anything asserted below and keeps this off the suite's critical path.
        auto low = radius - 0.20;
        auto high = radius + 0.25;
        for (auto iteration = 0; iteration < 20; iteration++)
        {
            const auto middle = 0.5 * (low + high);
            (cast(middle) > compression ? low : high) = middle;
        }

        cast(0.5 * (low + high));

        // How many rays came back touching at all, before the aggregate had a chance to drop any of
        // them. `aggregateContactPatch` does not report what it rejected, so the two counts side by
        // side are the only way to ask whether it rejected anything.
        auto touching = std::uint32_t{0};
        for (auto index = std::size_t{0}; index < hits.size(); index++)
        {
            if (hits[index].hit && hits[index].point.y > geometry.tireSurface[index].y)
            {
                touching++;
            }
        }

        struct Reading
        {
            raceengine::ContactPatch patch;
            std::uint32_t touching;
        };

        return Reading{aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling), touching};
    };

    // Well clear of the kerb, and in the middle of its chamfer.
    constexpr auto onTheFlat = 2.85;
    constexpr auto onTheChamfer = 3.10;

    const auto flatCoarse = reportedAt(onTheFlat, 3).patch;
    const auto chamferCoarse = reportedAt(onTheChamfer, 3).patch;

    REQUIRE(flatCoarse.inContact);
    REQUIRE(chamferCoarse.inContact);

    SECTION("no sample is ever rejected as a spike, anywhere across the crossing")
    {
        // **The recorded diagnosis, disproved.** Rejection is against the median of the touching
        // samples, and a 3x3 grid spanning 0.25 m of a 9.5-degree ramp cannot open a 50 mm spread.
        //
        // `aggregateContactPatch` does not report how many it dropped, so the arithmetic is redone
        // here against the same rule: what is asserted is that the number of samples the aggregate
        // says are contacting equals the number that are touching at all, which is only true when
        // nothing was rejected between the two.
        for (auto step = 0; step <= 6; step++)
        {
            const auto x = descriptor.kerbInnerEdge - 0.15 + 0.10 * static_cast<double>(step);

            const auto reading = reportedAt(x, 3);

            CAPTURE(x, reading.patch.contactingSamples, reading.touching);
            REQUIRE(reading.patch.contactingSamples == reading.touching);
        }
    }

    SECTION("the dip is this big, so any change to the aggregation shows here")
    {
        // **Re-measured 2026-08-22 on the corrected car**: 2450 N on the flat against 898 N on the
        // chamfer, at a held 15 mm, on the Golf's 298,926 N/m tyre rate. That is a 63% unload with
        // the compression not moving. It was 2391/888 at the old 0.31 m radius, so the correction
        // made the dip 49 N deeper rather than explaining any of it.
        //
        // The other correction of that day — unsprung mass, which was roughly double — does not
        // enter this number at all, and that is worth saying because the brief expected it to. This
        // probe *holds* the deepest sample's compression and asks what the sampler reports, so no
        // mass is ever integrated: it is a kinematic measurement of the aggregation, and the only
        // vehicle data it can be sensitive to is the wheel radius and the tyre rate.
        const auto rate = golfGtiMk7().value().corners.front().tireVerticalRate;
        const auto flat = rate * flatCoarse.penetration;
        const auto chamfer = rate * chamferCoarse.penetration;

        CAPTURE(flat, chamfer);
        REQUIRE(flat == Catch::Approx(2450.0).margin(30.0));
        REQUIRE(chamfer == Catch::Approx(898.0).margin(30.0));
        REQUIRE(chamfer < 0.45 * flat);
    }

    SECTION("and the depth spread tells the dip apart from a genuinely light tyre")
    {
        // **The regression for E1's finding**, and the reason the channel was added rather than the
        // reason it is interesting: on this chamfer the wheel reports 898 N, and a wheel resting
        // lightly on flat road can report 898 N too. `Tyre Fz` cannot separate them, and neither can
        // `Contact Samples` — both patches carry three of nine.
        //
        // The spread can. On level ground under *this* wheel — upright, no camber — the only thing
        // that spreads the contacting depths is the tread's own curvature, so the spread is bounded
        // above by the sagitta of the tread over the patch's length, 10.21 mm here, and is exactly
        // zero below the compression at which the outer rows reach the road at all.
        //
        // **That ceiling is the uncambered one and it is not a general rule.** A cambered wheel
        // leans its patch across its width and reads `width * sin(camber) + rise * cos(camber)` on
        // perfectly level ground — see the case above, and note the scripted launch reads 11.6 to
        // 14.6 mm per corner on flat Bathurst tarmac against this 10.21. A discriminator has to
        // stand on the *cambered* ceiling, which is still a pure function of the wheel and still
        // known per corner per tick. What this section pins is that the mechanism is there and how
        // big it is on the fixture where the arithmetic is exact.
        const auto rise =
            radius - std::sqrt(radius * radius - 0.25 * ContactPatchSampling{}.length * ContactPatchSampling{}.length);

        CAPTURE(rise, flatCoarse.depthSpread, chamferCoarse.depthSpread);

        // Flat road, fully contacting: exactly the tread's own rise and not a millimetre more.
        REQUIRE(flatCoarse.contactingSamples == 9);
        REQUIRE(flatCoarse.depthSpread == Catch::Approx(rise).epsilon(1e-6));

        // The chamfer: three samples carrying, and a spread that is over the level ceiling by a
        // clear margin — 14.43 mm against 10.21. That margin is the road's tilt, and it is the whole
        // signal.
        REQUIRE(chamferCoarse.contactingSamples == 3);
        REQUIRE(chamferCoarse.depthSpread > rise * 1.3);

        // And the flat wheel that would report this same load reads a spread of exactly zero,
        // because at that compression only the middle row is touching and the tyre is a cylinder
        // across its width. Three samples, the same load, and the two cases are still separated.
        const auto matching = 3.0 * chamferCoarse.penetration;
        REQUIRE(matching < rise);

        // The same bed the chamfer reading above was taken on. Comparing a light tyre on the
        // shipped 7x3 coupled patch against a chamfer measured on the uncoupled 3x3 one would be
        // two different tyres, and the whole point of the case is that these two are alike in every
        // channel but the spread.
        const auto sampling3 = uncoupled();
        const auto lightGeometry = contactPatchSamples(WheelPose{.centre = glm::dvec3(0.0, radius - matching, 0.0),
                                                                 .spinAxis = glm::dvec3(1.0, 0.0, 0.0),
                                                                 .forward = glm::dvec3(0.0, 0.0, 1.0),
                                                                 .radius = radius},
                                                       sampling3);
        const auto light =
            aggregateContactPatch(lightGeometry, flatGround(lightGeometry, 0.0), defaultSurfaceMaterials(), sampling3);

        REQUIRE(light.contactingSamples == chamferCoarse.contactingSamples);
        REQUIRE(light.penetration == Catch::Approx(chamferCoarse.penetration).epsilon(1e-9));
        REQUIRE(light.depthSpread == Catch::Approx(0.0).margin(1e-15));
    }

    SECTION("and refining the grid deepens it, which is what says it is not the quadrature")
    {
        // The one measurement that separates "too few samples" from "the wrong model". A quadrature
        // error shrinks as the grid refines. This one grows: on the flat the load climbs toward its
        // converged value while on the chamfer it falls further, so the ratio between them widens
        // rather than closing.
        const auto flatFine = reportedAt(onTheFlat, 9).patch;
        const auto chamferFine = reportedAt(onTheChamfer, 9).patch;

        const auto coarseRatio = chamferCoarse.penetration / flatCoarse.penetration;
        const auto fineRatio = chamferFine.penetration / flatFine.penetration;

        CAPTURE(coarseRatio, fineRatio);
        REQUIRE(flatFine.penetration > flatCoarse.penetration);
        REQUIRE(chamferFine.penetration < chamferCoarse.penetration);
        REQUIRE(fineRatio < coarseRatio);
    }

    SECTION("and at a held ride height, which is the control a car has, there is no dip at all")
    {
        // **The correction to what the three sections above were read as meaning, pinned so that it
        // cannot quietly be forgotten again** (2026-08-22, E2). It is the single most useful thing
        // this session produced and it is the fifth instance in this project of a fixture measuring
        // something other than what its conclusion was about.
        //
        // Every figure in the sections above holds the wheel so that its *deepest sample* penetrates
        // 15 mm. On a 9.46-degree cross-slope across a 0.20 m patch the depth falls 28.9 mm from one
        // side to the other, so holding the deepest sample at 15 mm puts the road *below* the tread
        // by 14 mm under the middle of the patch: the wheel is barely touching, and a low load there
        // is not a defect, it is a wheel that has been lifted.
        //
        // A car does not have that control. A suspension holds a wheel at roughly a height and the
        // road comes up to meet it, so that is what is asked here — and the load does not dip, it
        // *rises*, by a third at the kerb's inner edge and by six per cent through the ramp. Nothing
        // an enveloping model does was ever going to remove a dip that is not there.
        //
        // What this section does *not* say is that there is no defect: the load falls by about a
        // third across the kerb's top edge, where the patch straddles a crest and the road curves
        // away on both sides, and that is a genuine enveloping case. It is asserted here so that a
        // model which fixes it will announce itself.
        const auto rate = golfGtiMk7().value().corners.front().tireVerticalRate;
        const auto sampling = ContactPatchSampling{};

        // The height this car's front corner actually rides at: 1348 kg at 61.4% front over two
        // wheels is 4061 N. Solved on the flat rather than written down — see `heightForLoad`.
        const auto ride = heightForLoad(sampling, radius, rate, 4061.0);

        const auto atHeldHeight = [&](const double x)
        {
            const auto beneath = raceengine::sampleProvingGround(descriptor, x, middleZ);
            const auto geometry = contactPatchSamples(WheelPose{.centre = glm::dvec3(x, beneath.height + ride, middleZ),
                                                                .spinAxis = spinAxis,
                                                                .forward = forward,
                                                                .radius = radius},
                                                      sampling);

            auto hits = std::vector<SurfaceHit>{};
            world->castRays(geometry.origins, geometry.directions, sampling.searchDistance * 2.0, hits);

            return rate * aggregateContactPatch(geometry, hits, defaultSurfaceMaterials(), sampling).penetration;
        };

        const auto flat = atHeldHeight(onTheFlat);
        const auto innerEdge = atHeldHeight(descriptor.kerbInnerEdge);
        const auto chamfer = atHeldHeight(onTheChamfer);
        const auto topEdge = atHeldHeight(descriptor.kerbInnerEdge + descriptor.kerbChamfer);

        CAPTURE(flat, innerEdge, chamfer, topEdge);

        // The reference: the same wheel on flat road, carrying what this corner carries.
        REQUIRE(flat == Catch::Approx(4061.0).margin(40.0));

        // The chamfer does not unload. It carries more than flat road, not 63% less.
        REQUIRE(chamfer > flat);
        REQUIRE(chamfer < 1.25 * flat);

        // And the inner edge is a load *spike*, which is the opposite sign to the recorded finding
        // and is what a wheel meeting a kerb should do.
        REQUIRE(innerEdge > 1.25 * flat);

        // **The top edge, as a band rather than a bound, because both ends of it say something.**
        //
        // This is the one place the patch genuinely loses load it should be carrying: it straddles
        // the kerb's crest and the road curves away on both sides. On the uncoupled bed it read
        // 2941 N, 72% of flat. With the belt the elements over the far side are dragged inward by
        // the loaded ones and it reads 3223 N, 79% — so **the coupling is worth about a tenth of the
        // crest's load, and this is where the enveloping model earns its cost.** That is the only
        // case in this file where the belt does something a driver could plausibly meet, and it is
        // asserted rather than described so that turning the belt off shows up here.
        //
        // The lower bound is the belt's benefit and the upper bound is the remaining defect. A model
        // that carries the crest properly breaks the upper one and should.
        REQUIRE(topEdge > 0.75 * flat);
        REQUIRE(topEdge < 0.85 * flat);
    }
}
