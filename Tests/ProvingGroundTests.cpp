#include <cmath>
#include <set>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::defaultProvingGround;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::ProvingGroundDescriptor;
using raceengine::sampleProvingGround;
using raceengine::SurfaceKind;

TEST_CASE("the proving ground carries the features a flat plane would hide", "[physics][ground]")
{
    const auto ground = defaultProvingGround();

    SECTION("the long opening run is flat tarmac, which is what settling and the skidpad need")
    {
        for (const auto z : {1.0, 15.0, 40.0, 59.0})
        {
            for (const auto x : {-15.0, -3.0, 0.0, 3.0, 15.0})
            {
                const auto sample = sampleProvingGround(ground, x, z);
                REQUIRE(sample.height == Catch::Approx(0.0).margin(1e-15));
                REQUIRE(sample.kind == SurfaceKind::Tarmac);
            }
        }
    }

    SECTION("the kerb is raised on one side only, with a chamfer to half-mount")
    {
        const auto z = 75.0;

        // Inboard of the edge there is no kerb at all: a wheel on the left is on flat tarmac while
        // the one on the right is climbing, which is the whole point of the feature.
        REQUIRE(sampleProvingGround(ground, 0.0, z).height == Catch::Approx(0.0).margin(1e-15));
        REQUIRE(sampleProvingGround(ground, ground.kerbInnerEdge - 0.01, z).kind == SurfaceKind::Tarmac);

        // Across the chamfer it rises linearly, so a contact patch spanning it aggregates a tilted
        // plane rather than a step.
        const auto midChamfer = sampleProvingGround(ground, ground.kerbInnerEdge + 0.5 * ground.kerbChamfer, z);
        REQUIRE(midChamfer.height == Catch::Approx(0.5 * ground.kerbHeight));
        REQUIRE(midChamfer.kind == SurfaceKind::Kerb);

        // And past it, the flat top.
        REQUIRE(sampleProvingGround(ground, ground.kerbInnerEdge + 1.0, z).height == Catch::Approx(ground.kerbHeight));
    }

    SECTION("the surface boundary changes grip without changing it everywhere")
    {
        const auto z = 115.0;

        REQUIRE(sampleProvingGround(ground, -2.0, z).kind == SurfaceKind::Tarmac);
        REQUIRE(sampleProvingGround(ground, 2.0, z).kind == SurfaceKind::Grass);
        // The grass sits a little low, which is what makes straddling it a load transfer as well as
        // a grip change.
        REQUIRE(sampleProvingGround(ground, 2.0, z).height == Catch::Approx(-ground.boundaryLip));
    }

    SECTION("the camber eases in and out rather than arriving as a crease")
    {
        const auto entry = sampleProvingGround(ground, 5.0, 140.0).height;
        const auto peak = sampleProvingGround(ground, 5.0, 155.0).height;
        const auto exit = sampleProvingGround(ground, 5.0, 169.99).height;

        REQUIRE(entry == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(peak == Catch::Approx(5.0 * std::tan(ground.camberAngle)));
        REQUIRE(std::abs(exit) < 1e-3);

        // It banks about the centreline, so the two sides go opposite ways.
        REQUIRE(sampleProvingGround(ground, -5.0, 155.0).height == Catch::Approx(-peak));

        // And the gradient is bounded everywhere across the entry, which is what "eased" has to
        // mean numerically: a crease would show as one large step between adjacent samples.
        auto previous = 0.0;
        auto worstStep = 0.0;
        for (auto z = 139.0; z < 171.0; z += 0.25)
        {
            const auto height = sampleProvingGround(ground, 5.0, z).height;
            worstStep = std::max(worstStep, std::abs(height - previous));
            previous = height;
        }
        REQUIRE(worstStep < 0.02);
    }

    SECTION("the ramp climbs and then the ground stops")
    {
        REQUIRE(sampleProvingGround(ground, 0.0, 185.0).height == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(sampleProvingGround(ground, 0.0, 192.5).height == Catch::Approx(0.5 * ground.rampHeight));

        // Past the end of the ground the sampler still answers, and answers flat: a query that
        // walks off the edge is a normal event mid-flight, not an error to handle at every site.
        REQUIRE(sampleProvingGround(ground, 0.0, 260.0).height == Catch::Approx(0.0).margin(1e-15));
    }
}

TEST_CASE("the generated mesh is well formed", "[physics][ground]")
{
    // Coarse on purpose: the properties below are about structure, and a 0.25 m ground is 128000
    // triangles to assert them over.
    auto descriptor = defaultProvingGround();
    descriptor.cellSize = 1.0;

    const auto generated = generateProvingGround(descriptor);
    REQUIRE(generated.has_value());

    const auto& mesh = generated.value();
    const auto across = static_cast<std::size_t>(descriptor.width / descriptor.cellSize);
    const auto along = static_cast<std::size_t>(descriptor.length / descriptor.cellSize);

    REQUIRE(mesh.vertices.size() == (across + 1) * (along + 1));
    REQUIRE(mesh.triangleCount() == across * along * 2);
    // One surface index per triangle, and every one of them addressable.
    REQUIRE(mesh.surfaces.size() == mesh.triangleCount());
    for (const auto surface : mesh.surfaces)
    {
        REQUIRE(surface < mesh.materials.size());
    }

    SECTION("every index is in range and no triangle is degenerate")
    {
        auto worstArea = 1e30;
        for (auto triangle = std::size_t{0}; triangle < mesh.triangleCount(); triangle++)
        {
            const auto a = mesh.indices[triangle * 3 + 0];
            const auto b = mesh.indices[triangle * 3 + 1];
            const auto c = mesh.indices[triangle * 3 + 2];

            REQUIRE(a < mesh.vertices.size());
            REQUIRE(b < mesh.vertices.size());
            REQUIRE(c < mesh.vertices.size());

            const auto area =
                0.5 * glm::length(glm::cross(mesh.vertices[b] - mesh.vertices[a], mesh.vertices[c] - mesh.vertices[a]));
            worstArea = std::min(worstArea, area);
        }

        // A zero-area triangle has no normal, and a normal is what every contact sample reads.
        REQUIRE(worstArea > 1e-6);
    }

    SECTION("triangles wind counter-clockwise seen from above")
    {
        // The engine's front face is counter-clockwise, and a ground whose normals point into the
        // planet is the sort of thing that reads as "the ray query is broken" much later.
        for (auto triangle = std::size_t{0}; triangle < mesh.triangleCount(); triangle++)
        {
            const auto& a = mesh.vertices[mesh.indices[triangle * 3 + 0]];
            const auto& b = mesh.vertices[mesh.indices[triangle * 3 + 1]];
            const auto& c = mesh.vertices[mesh.indices[triangle * 3 + 2]];

            REQUIRE(glm::cross(b - a, c - a).y > 0.0);
        }
    }

    SECTION("a barrier faces the road it is protecting")
    {
        // The one feature whose winding is not obvious, and the one where getting it wrong is
        // invisible until much later: a backwards wall is still solid to a ray cast, because a ray
        // does not care which side of a triangle it meets, so scene queries report it exactly as
        // they should. A *shape* query ignores back faces, so the car drives through it. This
        // asserts the thing that actually matters.
        auto walled = descriptor;
        walled.barrierX = 5.0;
        walled.features = {Feature{.kind = FeatureKind::Barrier, .from = 20.0, .to = 60.0}};

        const auto withWall = generateProvingGround(walled);
        REQUIRE(withWall.has_value());

        auto vertical = std::size_t{0};
        for (auto triangle = std::size_t{0}; triangle < withWall->triangleCount(); triangle++)
        {
            const auto& a = withWall->vertices[withWall->indices[triangle * 3 + 0]];
            const auto& b = withWall->vertices[withWall->indices[triangle * 3 + 1]];
            const auto& c = withWall->vertices[withWall->indices[triangle * 3 + 2]];

            const auto normal = glm::cross(b - a, c - a);
            if (std::abs(normal.y) > 1e-9)
            {
                continue; // ground, checked above
            }

            vertical++;
            // Facing back towards the centreline, which is where the road is.
            REQUIRE(normal.x < 0.0);
        }

        REQUIRE(vertical > 0);
    }

    SECTION("all three surfaces are actually present")
    {
        auto seen = std::set<std::uint32_t>(mesh.surfaces.begin(), mesh.surfaces.end());
        REQUIRE(seen.size() == 3);
    }

    SECTION("the same descriptor generates the same ground")
    {
        // It is an input to a deterministic simulation, so it has to be one itself.
        const auto again = generateProvingGround(descriptor);
        REQUIRE(again.has_value());
        REQUIRE(again->vertices == mesh.vertices);
        REQUIRE(again->indices == mesh.indices);
        REQUIRE(again->surfaces == mesh.surfaces);
    }
}

TEST_CASE("a ground that cannot be built is reported", "[physics][ground]")
{
    auto descriptor = ProvingGroundDescriptor{};

    descriptor.cellSize = 0.0;
    REQUIRE_FALSE(generateProvingGround(descriptor).has_value());

    descriptor = ProvingGroundDescriptor{};
    descriptor.width = 0.0;
    REQUIRE_FALSE(generateProvingGround(descriptor).has_value());

    descriptor = ProvingGroundDescriptor{};
    descriptor.cellSize = 500.0;
    REQUIRE_FALSE(generateProvingGround(descriptor).has_value());
}
