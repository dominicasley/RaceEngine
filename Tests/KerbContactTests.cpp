// The kerb-contact path's seam and rule, against hand-built meshes.
//
// Stage 1 of docs/kerb-contact-brief.md: the cylinder query returns the road's push on a wheel-sized
// cylinder per triangle; the exclusion rule drops the contacts the bottom grid already carries; a
// vertical step the rays cannot see comes back as a face contact whose depth is zero at first touch
// and grows continuously as the wheel advances. No vehicle is stood up here — that is stage 2's
// proving-ground work — so what is pinned is the query and the rule, on their own.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::defaultSurfaceMaterials;
using raceengine::keepObstacleContacts;
using raceengine::maxObstacleContacts;
using raceengine::ObstacleContact;
using raceengine::PhysicsWorld;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::WheelCylinder;

namespace
{

// Jolt's factory and type registry are process-wide, so a case that stands them up has to take them
// down again whatever it does in between — including failing an assertion.
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

// The Golf's tyre: 225/40 R18, outer radius 0.3186 m, section half-width 0.1125 m.
constexpr auto tyreRadius = 0.3186;
constexpr auto tyreHalfWidth = 0.1125;
constexpr auto degrees = 57.29577951308232;

// A flat square of tarmac, two triangles, wound so `(v1 - v0) x (v2 - v0)` points up — the front
// face the engine's mesh generator emits and the one the query's back-face culling keeps.
[[nodiscard]] SurfaceMesh flatPlane(const double halfExtent)
{
    auto mesh = SurfaceMesh{};
    mesh.materials = defaultSurfaceMaterials();
    mesh.vertices = {glm::dvec3(-halfExtent, 0.0, -halfExtent), glm::dvec3(halfExtent, 0.0, -halfExtent),
                     glm::dvec3(halfExtent, 0.0, halfExtent), glm::dvec3(-halfExtent, 0.0, halfExtent)};
    mesh.indices = {0, 3, 1, 1, 3, 2};
    mesh.surfaces = {0, 0};

    return mesh;
}

// A vertical step: tarmac at y = 0 for z < 0, a kerb face at z = 0 rising `height`, and the kerb's
// flat top beyond it. The generator cannot make one — its kerb has a chamfer and its features are
// eased in — and a 150 mm vertical face is exactly the surface the finding is about. Hand-built,
// six triangles, every one wound with its normal outward: up on the two flats, towards -z on the
// face a wheel arriving from -z meets.
[[nodiscard]] SurfaceMesh verticalStep(const double height, const double halfWidth, const double length)
{
    auto mesh = SurfaceMesh{};
    mesh.materials = defaultSurfaceMaterials();
    mesh.vertices = {
        glm::dvec3(-halfWidth, 0.0, -length),   // 0
        glm::dvec3(halfWidth, 0.0, -length),    // 1
        glm::dvec3(halfWidth, 0.0, 0.0),        // 2
        glm::dvec3(-halfWidth, 0.0, 0.0),       // 3
        glm::dvec3(halfWidth, height, 0.0),     // 4
        glm::dvec3(-halfWidth, height, 0.0),    // 5
        glm::dvec3(halfWidth, height, length),  // 6
        glm::dvec3(-halfWidth, height, length), // 7
    };
    mesh.indices = {
        0, 3, 1, 1, 3, 2, // the road
        3, 5, 2, 2, 5, 4, // the face
        5, 7, 4, 4, 7, 6, // the kerb top
    };
    // Tarmac, then the kerb material for the face and its top.
    mesh.surfaces = {0, 0, 1, 1, 1, 1};

    return mesh;
}

[[nodiscard]] PhysicsWorld worldOf(const SurfaceMesh& mesh)
{
    auto world = PhysicsWorld::create(mesh);
    REQUIRE(world.has_value());

    return std::move(world).value();
}

// One wheel, spin axis along +x, at a centre.
[[nodiscard]] WheelCylinder wheelAt(const glm::dvec3& centre)
{
    return WheelCylinder{
        .centre = centre, .spinAxis = glm::dvec3(1.0, 0.0, 0.0), .radius = tyreRadius, .halfWidth = tyreHalfWidth};
}

// What the query returned for one wheel, unfiltered.
[[nodiscard]] std::vector<ObstacleContact> query(const PhysicsWorld& world, const WheelCylinder& wheel)
{
    const auto cylinders = std::array<WheelCylinder, 1>{wheel};

    auto results = std::vector<ObstacleContact>{};
    auto counts = std::vector<std::uint32_t>{};
    world.collideCylinders(cylinders, 32, results, counts);

    REQUIRE(counts.size() == 1);
    REQUIRE(counts[0] == results.size());

    return results;
}

// ...and what the rule keeps of it.
[[nodiscard]] std::vector<ObstacleContact> survivors(const std::vector<ObstacleContact>& found,
                                                     const WheelCylinder& wheel)
{
    auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
    const auto count = keepObstacleContacts(found, wheel, glm::dvec3(0.0), kept);

    return std::vector<ObstacleContact>(kept.begin(), kept.begin() + count);
}

[[nodiscard]] double elevationDegrees(const ObstacleContact& contact)
{
    return std::asin(std::clamp(contact.axis.y, -1.0, 1.0)) * degrees;
}

} // namespace

TEST_CASE("a wheel on flat road reports only the contacts the bottom grid already carries", "[physics][world][kerb]")
{
    const JoltGuard jolt;
    const auto world = worldOf(flatPlane(10.0));

    SECTION("a centimetre into the road, every triangle pushes straight up by that centimetre")
    {
        const auto wheel = wheelAt(glm::dvec3(0.0, tyreRadius - 0.010, 0.0));
        const auto found = query(world, wheel);

        REQUIRE(!found.empty());
        for (const auto& contact : found)
        {
            CAPTURE(contact.axis.x, contact.axis.y, contact.axis.z, contact.depth);
            REQUIRE(contact.axis.y > std::cos(1.0 / degrees));
            REQUIRE(contact.depth == Catch::Approx(0.010).margin(5e-4));
            REQUIRE(contact.point.y == Catch::Approx(0.0).margin(1e-4));
            // Tarmac is surface 0 of the default table.
            REQUIRE(contact.surface == 0);
        }

        // And the exclusion rule keeps none of them: on flat road the path is inert by construction.
        REQUIRE(survivors(found, wheel).empty());
    }

    SECTION("a wheel held above the road touches nothing")
    {
        REQUIRE(query(world, wheelAt(glm::dvec3(0.0, tyreRadius + 0.005, 0.0))).empty());
    }

    SECTION("a cambered wheel's flat-road contacts are still the grid's, read against its own axis")
    {
        // Three degrees of camber: the spin axis tilts, and the wheel plane's down with it. The road
        // still pushes straight up, which is within the cone of that tilted down, so it is dropped —
        // the rule is stated against the wheel and not against the world.
        const auto camber = 3.0 / degrees;
        const auto cylinders =
            std::array<WheelCylinder, 1>{WheelCylinder{.centre = glm::dvec3(0.0, tyreRadius - 0.010, 0.0),
                                                       .spinAxis = glm::dvec3(std::cos(camber), std::sin(camber), 0.0),
                                                       .radius = tyreRadius,
                                                       .halfWidth = tyreHalfWidth}};

        auto results = std::vector<ObstacleContact>{};
        auto counts = std::vector<std::uint32_t>{};
        world.collideCylinders(cylinders, 32, results, counts);
        REQUIRE(!results.empty());

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(results, cylinders[0], glm::dvec3(0.0), kept) == 0);
    }
}

TEST_CASE("a 150 mm step is a face contact that starts at zero depth and grows continuously", "[physics][world][kerb]")
{
    const JoltGuard jolt;

    constexpr auto height = 0.150;
    const auto world = worldOf(verticalStep(height, 5.0, 10.0));

    // A wheel of this radius meets the kerb's top edge when its hub is this far short of the face:
    // the chord at the kerb's height. 0.270 m on the Golf's tyre.
    const auto reach = std::sqrt(tyreRadius * tyreRadius - (tyreRadius - height) * (tyreRadius - height));
    REQUIRE(reach == Catch::Approx(0.2703).margin(5e-4));

    // Two millimetres into the road, so the flat contact exists at every station and is dropped at
    // every station: the survivors are the face's alone.
    const auto hubHeight = tyreRadius - 0.002;

    // Rolled towards the face in two-millimetre stations, from two centimetres short of first touch
    // to seven centimetres past it.
    constexpr auto station = 0.002;
    const auto first = -reach - 0.020;
    const auto last = -reach + 0.070;

    struct Station
    {
        double z = 0.0;
        std::size_t count = 0;
        double depth = 0.0;
        double elevation = 0.0;
        glm::dvec3 axis{0.0};
    };

    auto stations = std::vector<Station>{};
    for (auto z = first; z <= last + 1e-9; z += station)
    {
        const auto wheel = wheelAt(glm::dvec3(0.0, hubHeight, z));
        const auto found = query(world, wheel);
        const auto kept = survivors(found, wheel);

        auto entry = Station{.z = z, .count = kept.size()};
        if (!kept.empty())
        {
            entry.depth = kept.front().depth;
            entry.elevation = elevationDegrees(kept.front());
            entry.axis = kept.front().axis;
        }

        stations.push_back(entry);
    }

    // Nothing survives until the wheel reaches the edge, and one face survives from then on: the
    // face and the kerb top both report the same push-out off the shared edge and merge to one.
    auto touched = std::size_t{0};
    while (touched < stations.size() && stations[touched].count == 0)
    {
        touched++;
    }

    REQUIRE(touched > 0);
    REQUIRE(touched < stations.size());
    // Within one station of the geometric first touch.
    REQUIRE(stations[touched].z == Catch::Approx(-reach).margin(station + 1e-6));

    // Zero at first touch — under one station's worth of advance — and pointing backward and up off
    // the edge at the angle a round wheel meeting a 150 mm edge makes: 32 degrees above the horizon.
    REQUIRE(stations[touched].depth >= 0.0);
    REQUIRE(stations[touched].depth < 0.003);
    REQUIRE(stations[touched].axis.z < 0.0);
    REQUIRE(stations[touched].elevation == Catch::Approx(32.0).margin(4.0));

    for (auto index = touched; index < stations.size(); index++)
    {
        const auto& at = stations[index];
        CAPTURE(at.z, at.count, at.depth, at.elevation);

        REQUIRE(at.count == 1);
        // Continuous: each two-millimetre station deepens the contact by less than that plus a
        // float's worth, and never shallows it.
        if (index > touched)
        {
            REQUIRE(at.depth >= stations[index - 1].depth - 1e-6);
            REQUIRE(at.depth - stations[index - 1].depth < 0.003);
            // And the axis rotates upward as the edge falls behind the hub — never back down.
            REQUIRE(at.elevation >= stations[index - 1].elevation - 0.05);
        }

        // The kerb material, surface 1 of the default table, on every surviving contact.
        REQUIRE(at.axis.z < 0.0);
    }

    // Seven centimetres in, the depth is most of the advance — radial off the edge, not the whole
    // horizontal overlap — and the axis has come up from the edge's angle.
    const auto& deepest = stations.back();
    REQUIRE(deepest.depth > 0.050);
    REQUIRE(deepest.depth < 0.070);
    REQUIRE(deepest.elevation > stations[touched].elevation + 5.0);
}

TEST_CASE("the exclusion rule and the merge are stated against the wheel plane", "[physics][kerb]")
{
    // A wheel at the origin, and contacts made to satisfy the depth identity a real push-out has:
    // the road point sits `depth` inside the cylinder's surface along the axis.
    const auto spin = wheelAt(glm::dvec3(0.0));

    const auto tilted = [](const double elevationFromUp)
    {
        const auto angle = elevationFromUp / degrees;
        return glm::dvec3(0.0, std::cos(angle), -std::sin(angle));
    };

    // Face contacts: the triangle's normal is the push-out.
    const auto contact = [&](const glm::dvec3& axis, const double depth)
    {
        return ObstacleContact{.axis = axis, .normal = axis, .depth = depth, .point = -axis * (tyreRadius - depth)};
    };

    SECTION("fourteen degrees from the wheel's up is the grid's; sixteen is the cylinder's")
    {
        const auto found = std::vector<ObstacleContact>{
            contact(tilted(14.0), 0.01),
            contact(tilted(16.0), 0.02),
        };

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 1);
        REQUIRE(kept[0].depth == Catch::Approx(0.02));
    }

    SECTION("the hand-over weight fades from thirty degrees to the cone")
    {
        const auto found = std::vector<ObstacleContact>{
            contact(tilted(16.0), 0.01),
            contact(tilted(45.0), 0.01),
            contact(tilted(29.0), 0.01),
        };

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 3);
        // One degree past the cone: a fifteenth of the way, in the cosine.
        const auto oneDegree = (std::cos(15.0 / degrees) - std::cos(16.0 / degrees)) /
                               (std::cos(15.0 / degrees) - std::cos(30.0 / degrees));
        REQUIRE(kept[0].weight == Catch::Approx(oneDegree).epsilon(1e-9));
        REQUIRE(kept[1].weight == Catch::Approx(1.0));
        REQUIRE(kept[2].weight < 1.0);
        REQUIRE(kept[2].weight > 0.9);
    }

    SECTION("on a hill the cone stands on the grid's normal, not the wheel's up")
    {
        // A fifteen and a half degree grade: past the cone measured from the wheel's up, which on a
        // grade is the world's, and dead centre of it measured from the grid's own normal — which is
        // what the two slope fixtures found the hard way when the whole suite ran.
        const auto grade = 15.5 / degrees;
        const auto hill = glm::dvec3(0.0, std::cos(grade), -std::sin(grade));
        const auto found = std::vector<ObstacleContact>{contact(hill, 0.01)};

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 1);
        REQUIRE(keepObstacleContacts(found, spin, hill, kept) == 0);
    }

    SECTION("two triangles of one face merge to the deeper report")
    {
        const auto found = std::vector<ObstacleContact>{
            contact(tilted(60.0), 0.010),
            contact(tilted(62.0), 0.012),
            contact(tilted(61.0), 0.011),
        };

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 1);
        REQUIRE(kept[0].depth == Catch::Approx(0.012));
    }

    SECTION("two faces ten degrees apart are two contacts")
    {
        const auto found = std::vector<ObstacleContact>{
            contact(tilted(50.0), 0.010),
            contact(tilted(60.0), 0.012),
        };

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 2);
    }

    SECTION("a sidewall push, along the spin axis, is ninety degrees from the wheel's up and is kept")
    {
        // A kerb face met by the sidewall: the push is along the spin axis, the face is vertical,
        // and the point is below the hub, as a kerb's is.
        const auto axis = glm::dvec3(1.0, 0.0, 0.0);
        const auto found = std::vector<ObstacleContact>{ObstacleContact{
            .axis = axis, .normal = axis, .depth = 0.01, .point = glm::dvec3(-(tyreHalfWidth - 0.01), -0.1, 0.0)}};

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 1);
    }

    SECTION("a sideways sliver off a road triangle's edge is triangulation, not a face")
    {
        // The E2 chamfer's artefact by hand: a true push-out, sideways, a few millimetres, against a
        // triangle whose face is the road. Dropped for the triangle, not for the push.
        const auto axis = glm::dvec3(1.0, 0.0, 0.0);
        const auto found = std::vector<ObstacleContact>{
            ObstacleContact{.axis = axis,
                            .normal = glm::dvec3(0.0, std::cos(9.46 / degrees), -std::sin(9.46 / degrees)),
                            .depth = 0.003,
                            .point = glm::dvec3(-(tyreHalfWidth - 0.003), -0.3, 0.0)}};

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 0);
    }

    SECTION("a contact whose depth is not the cylinder's protrusion along its axis is not a push-out")
    {
        // Jolt's swapped-normal artefact, by hand: a horizontal axis with the tiny depth of an
        // inactive edge, at a road point the cylinder is eleven centimetres past along that axis.
        const auto found = std::vector<ObstacleContact>{ObstacleContact{.axis = glm::dvec3(0.0, 0.0, -1.0),
                                                                        .normal = glm::dvec3(0.0, 0.0, -1.0),
                                                                        .depth = 0.0003,
                                                                        .point = glm::dvec3(0.0, -0.24, 0.21)}};

        auto kept = std::array<ObstacleContact, maxObstacleContacts>{};
        REQUIRE(keepObstacleContacts(found, spin, glm::dvec3(0.0), kept) == 0);
    }
}

TEST_CASE("a wheel pressed onto a 45 degree face reads the compression it was pressed by", "[physics][world][kerb]")
{
    // The isotropy check, at the level this stage can make it: the cylinder's contact against a
    // face at 45 degrees reports the same depth the wheel was pushed in by along that face's normal,
    // and the same depth the flat road reports for the same push — so the spring on the far side of
    // the rule, which is the same tread rate times that depth, reads the same normal load either way.
    const JoltGuard jolt;

    // The plane y = z, wound so its normal is up and towards -z, where the wheel is.
    auto mesh = SurfaceMesh{};
    mesh.materials = defaultSurfaceMaterials();
    mesh.vertices = {glm::dvec3(-5.0, -5.0, -5.0), glm::dvec3(5.0, -5.0, -5.0), glm::dvec3(5.0, 5.0, 5.0),
                     glm::dvec3(-5.0, 5.0, 5.0)};
    mesh.indices = {0, 2, 1, 0, 3, 2};
    mesh.surfaces = {0, 0};

    const auto tilted = worldOf(mesh);
    const auto flat = worldOf(flatPlane(10.0));

    constexpr auto press = 0.020;
    const auto normal = glm::normalize(glm::dvec3(0.0, 1.0, -1.0));

    const auto onFace = wheelAt(normal * (tyreRadius - press));
    const auto faceContacts = survivors(query(tilted, onFace), onFace);

    REQUIRE(faceContacts.size() == 1);
    REQUIRE(faceContacts.front().depth == Catch::Approx(press).margin(5e-4));
    REQUIRE(glm::dot(faceContacts.front().axis, normal) > std::cos(1.0 / degrees));

    const auto onFlat = wheelAt(glm::dvec3(0.0, tyreRadius - press, 0.0));
    const auto flatContacts = query(flat, onFlat);
    REQUIRE(!flatContacts.empty());

    auto deepestFlat = 0.0;
    for (const auto& contact : flatContacts)
    {
        deepestFlat = std::max(deepestFlat, contact.depth);
    }

    // The same compression, so the same load: the tread rate on both, because both axes lie in the
    // wheel plane. The Golf's rate, stated for the arithmetic to be visible.
    constexpr auto treadRate = 298926.0;
    REQUIRE(treadRate * faceContacts.front().depth == Catch::Approx(treadRate * deepestFlat).epsilon(0.025));
}

TEST_CASE("a wall the hub is not above is the chassis collider's case", "[physics][world][kerb]")
{
    // The query reports the wall — a cylinder pushed into a vertical face meets it at hub height with
    // a horizontal push — and the rule keeps nothing of it: a push at or above the wheel centre is
    // not a kerb a tyre can climb, and the bodywork's contact solver is what stops a car at a wall.
    const JoltGuard jolt;

    auto mesh = flatPlane(10.0);
    // A wall across the road at z = 0, a metre tall, facing -z.
    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.insert(mesh.vertices.end(), {glm::dvec3(-10.0, 0.0, 0.0), glm::dvec3(10.0, 0.0, 0.0),
                                               glm::dvec3(10.0, 1.0, 0.0), glm::dvec3(-10.0, 1.0, 0.0)});
    mesh.indices.insert(mesh.indices.end(), {base, base + 3, base + 1, base + 1, base + 3, base + 2});
    mesh.surfaces.insert(mesh.surfaces.end(), {1, 1});

    const auto world = worldOf(mesh);

    // Two centimetres into the wall, resting two millimetres into the road.
    const auto wheel = wheelAt(glm::dvec3(0.0, tyreRadius - 0.002, -tyreRadius + 0.020));
    const auto found = query(world, wheel);

    auto sawWall = false;
    for (const auto& contact : found)
    {
        if (contact.axis.z < -0.99)
        {
            sawWall = true;
            REQUIRE(contact.depth == Catch::Approx(0.020).margin(1e-3));
            REQUIRE(contact.point.y == Catch::Approx(wheel.centre.y).margin(1e-3));
        }
    }

    REQUIRE(sawWall);
    REQUIRE(survivors(found, wheel).empty());
}
