// The navigation mesh: the scanner, the build and point location, on a mesh written in code in the
// export's own format (docs/pursuit-navigation-brief.md, stage 1b).
//
// Tagged `[police]` because the mesh exists for the police and nothing else reads it: the tag is
// what Dominic runs after a pursuit change, and a mesh fault is a pursuit fault.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::buildNavMesh;
using raceengine::loadNavMesh;
using raceengine::locateOnNavMesh;
using raceengine::NavMesh;
using raceengine::NavMeshArea;
using raceengine::navMeshCentre;
using raceengine::navMeshHeight;
using raceengine::NavMeshLink;
using raceengine::navMeshNeighbour;
using raceengine::NavMeshPolygon;
using raceengine::NavMeshSource;
using raceengine::noPolygon;
using raceengine::parseNavMesh;

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace
{

// Two tiles of road side by side, a strip of grass across the top of both (so the grass polygon
// meets each road tile at a T-junction), an island of road in its own component, and a bridge over
// the first tile at six metres with the same footprint. The second tile rises one metre across its
// width.
[[nodiscard]] NavMeshSource fixtureSource()
{
    auto source = NavMeshSource{};
    source.areas = {NavMeshArea{.key = "ROAD", .costHint = 1.0, .friction = 0.98},
                    NavMeshArea{.key = "GRASS", .costHint = 5.0, .friction = 0.85}};
    source.polygons = {
        NavMeshPolygon{.x0 = 0.0, .z0 = 0.0, .x1 = 10.0, .z1 = 10.0, .area = 0, .component = 0},
        NavMeshPolygon{
            .x0 = 10.0, .z0 = 0.0, .x1 = 20.0, .z1 = 10.0, .y11 = 1.0, .y10 = 1.0, .area = 0, .component = 0},
        NavMeshPolygon{.x0 = 0.0, .z0 = 10.0, .x1 = 20.0, .z1 = 20.0, .area = 1, .component = 0},
        NavMeshPolygon{.x0 = 30.0, .z0 = 0.0, .x1 = 40.0, .z1 = 10.0, .area = 0, .component = 1},
        NavMeshPolygon{.x0 = 0.0,
                       .z0 = 0.0,
                       .x1 = 10.0,
                       .z1 = 10.0,
                       .y00 = 6.0,
                       .y01 = 6.0,
                       .y11 = 6.0,
                       .y10 = 6.0,
                       .area = 0,
                       .component = 0},
    };
    source.links = {
        NavMeshLink{.a = 0, .b = 1, .p = glm::dvec3(10.0, 0.0, 0.0), .q = glm::dvec3(10.0, 0.0, 10.0)},
        NavMeshLink{.a = 0, .b = 2, .p = glm::dvec3(0.0, 0.0, 10.0), .q = glm::dvec3(10.0, 0.0, 10.0)},
        NavMeshLink{.a = 1, .b = 2, .p = glm::dvec3(10.0, 0.0, 10.0), .q = glm::dvec3(20.0, 1.0, 10.0)},
        NavMeshLink{.a = 1, .b = 3, .p = glm::dvec3(20.0, 1.0, 0.0), .q = glm::dvec3(20.0, 1.0, 10.0)},
    };
    source.statedPolygons = 5;
    source.statedLinks = 4;
    source.agentRadiusMetres = 1.0;

    return source;
}

// The fixture as the exporter would write it, with the members the reader has to step over: notes
// whose text names the arrays, a nested audit with strings, escapes, booleans and nulls in it, and
// extra fields on every area and count.
[[nodiscard]] std::string fixtureDocument(const NavMeshSource& source)
{
    auto text = std::string();
    text += "{\"schema\":\"ac-car-data navmesh 1\",\"units\":\"metres\",";
    text += "\"axes\":\"right-handed, +X left, +Y up, +Z forward \\u2014 unconverted\",";
    text += "\"agent\":{\"radius_m\":1.0,\"height_m\":2.0,\"note\":\"a car\"},";
    text += "\"areas\":[";

    for (auto index = std::size_t{0}; index < source.areas.size(); index++)
    {
        const auto& area = source.areas[index];
        text += index == 0 ? "" : ",";
        text += "{\"index\":" + std::to_string(index) + ",\"key\":\"" + area.key +
                "\",\"friction\":" + std::to_string(area.friction) +
                ",\"is_valid_track\":true,\"cost_hint\":" + std::to_string(area.costHint) +
                ",\"source_triangles\":12,\"polygons\":3}";
    }

    text += "],\"cost_hint_note\":\"a suggested edge weight per surface [polygons] {links}\",";
    text += "\"counts\":{\"polygons\":" + std::to_string(source.statedPolygons) +
            ",\"links\":" + std::to_string(source.statedLinks) + ",\"components\":2,\"triangles\":10},";
    text += "\"components\":[{\"index\":0,\"polygons\":4},{\"index\":1,\"polygons\":1}],";
    text += "\"audit\":{\"coverage\":{\"note\":\"quoted \\\"text\\\" with a slash \\/ and a tab \\t\",";
    text +=
        "\"by_surface\":[{\"key\":\"ROAD\",\"examples_off\":[[-1.5,0.0,2.25],[3e-05,-0.0,1]],\"fraction\":0.9468}]},";
    text += "\"spawn\":{\"on_navmesh\":true,\"polygon\":null,\"empty\":{},\"none\":[]}},";
    text += "\"polygons_note\":\"[x0, z0, x1, z1, y00, y01, y11, y10, area, component]\",";
    text += "\"polygons\":[";

    for (auto index = std::size_t{0}; index < source.polygons.size(); index++)
    {
        const auto& polygon = source.polygons[index];
        text += index == 0 ? "" : ",";
        text += "[" + std::to_string(polygon.x0) + "," + std::to_string(polygon.z0) + "," + std::to_string(polygon.x1) +
                "," + std::to_string(polygon.z1) + "," + std::to_string(polygon.y00) + "," +
                std::to_string(polygon.y01) + "," + std::to_string(polygon.y11) + "," + std::to_string(polygon.y10) +
                "," + std::to_string(polygon.area) + "," + std::to_string(polygon.component) + "]";
    }

    text += "],\"links\":[";

    for (auto index = std::size_t{0}; index < source.links.size(); index++)
    {
        const auto& link = source.links[index];
        text += index == 0 ? "" : ",";
        text += "[" + std::to_string(link.a) + "," + std::to_string(link.b) + "," + std::to_string(link.p.x) + "," +
                std::to_string(link.p.y) + "," + std::to_string(link.p.z) + "," + std::to_string(link.q.x) + "," +
                std::to_string(link.q.y) + "," + std::to_string(link.q.z) + "]";
    }

    text += "],\"seconds\":21.2}\n";

    return text;
}

[[nodiscard]] std::size_t degree(const NavMesh& mesh, const std::uint32_t polygon)
{
    return mesh.neighbourStarts[polygon + 1] - mesh.neighbourStarts[polygon];
}

} // namespace

TEST_CASE("the navmesh scanner reads the export's document and steps over everything it does not want",
          "[police][police-navmesh]")
{
    const auto expected = fixtureSource();
    const auto document = fixtureDocument(expected);

    const auto parsed = parseNavMesh(document);
    REQUIRE(parsed.has_value());

    const auto& source = parsed.value();
    REQUIRE(source.areas.size() == 2);
    REQUIRE(source.areas[0].key == "ROAD");
    REQUIRE(source.areas[1].key == "GRASS");
    REQUIRE_THAT(source.areas[1].costHint, WithinAbs(5.0, 1e-9));
    REQUIRE_THAT(source.areas[0].friction, WithinAbs(0.98, 1e-9));
    REQUIRE(source.statedPolygons == 5);
    REQUIRE(source.statedLinks == 4);
    REQUIRE_THAT(source.agentRadiusMetres, WithinAbs(1.0, 1e-9));

    REQUIRE(source.polygons.size() == expected.polygons.size());
    for (auto index = std::size_t{0}; index < expected.polygons.size(); index++)
    {
        const auto& read = source.polygons[index];
        const auto& wanted = expected.polygons[index];
        REQUIRE_THAT(read.x0, WithinAbs(wanted.x0, 1e-9));
        REQUIRE_THAT(read.z1, WithinAbs(wanted.z1, 1e-9));
        REQUIRE_THAT(read.y11, WithinAbs(wanted.y11, 1e-9));
        REQUIRE(read.area == wanted.area);
        REQUIRE(read.component == wanted.component);
    }

    REQUIRE(source.links.size() == expected.links.size());
    REQUIRE(source.links[2].a == 1);
    REQUIRE(source.links[2].b == 2);
    REQUIRE_THAT(source.links[2].q.y, WithinAbs(1.0, 1e-9));

    // The bake's own numbers round-trip through the scanner: a negative exponent and a negative zero.
    const auto tiny =
        parseNavMesh("{\"areas\":[{\"key\":\"ROAD\"}],\"polygons\":[[-1e-05,-0.0,1.5,2,0.001,0,0,0,0,0]]}");
    REQUIRE(tiny.has_value());
    REQUIRE_THAT(tiny->polygons[0].x0, WithinAbs(-1e-05, 1e-12));
    REQUIRE_THAT(tiny->polygons[0].y00, WithinAbs(0.001, 1e-12));
}

TEST_CASE("the navmesh scanner refuses a short row, a link off the document and a count that disagrees",
          "[police][police-navmesh]")
{
    const auto source = fixtureSource();

    auto shortRow = source;
    auto text = fixtureDocument(shortRow);
    // Nine numbers in the first polygon row: its trailing ",0]" becomes "]".
    const auto first = text.find("\"polygons\":[[");
    REQUIRE(first != std::string::npos);
    const auto rowEnd = text.find(']', first + 13);
    text.replace(text.rfind(',', rowEnd), rowEnd - text.rfind(',', rowEnd), "");
    const auto refusedRow = parseNavMesh(text);
    REQUIRE_FALSE(refusedRow.has_value());
    REQUIRE_THAT(refusedRow.error(), ContainsSubstring("polygon row 0"));

    auto claims = source;
    claims.statedPolygons = 6;
    const auto refusedCount = parseNavMesh(fixtureDocument(claims));
    REQUIRE_FALSE(refusedCount.has_value());
    REQUIRE_THAT(refusedCount.error(), ContainsSubstring("claims 6 polygons"));

    auto offDocument = source;
    offDocument.links.push_back(NavMeshLink{.a = 1, .b = 9});
    offDocument.statedLinks = 5;
    const auto refusedLink = buildNavMesh(offDocument);
    REQUIRE_FALSE(refusedLink.has_value());
    REQUIRE_THAT(refusedLink.error(), ContainsSubstring("names a polygon the document does not carry"));

    auto noMain = source;
    for (auto& polygon : noMain.polygons)
    {
        polygon.component = 1;
    }
    const auto refusedComponent = buildNavMesh(noMain);
    REQUIRE_FALSE(refusedComponent.has_value());
    REQUIRE_THAT(refusedComponent.error(), ContainsSubstring("no polygon in component 0"));

    auto trailing = fixtureDocument(source);
    trailing += "{}";
    const auto refusedTrailing = parseNavMesh(trailing);
    REQUIRE_FALSE(refusedTrailing.has_value());
}

TEST_CASE("the navmesh build keeps the main component, remaps the links and lays out the adjacency",
          "[police][police-navmesh]")
{
    const auto built = buildNavMesh(fixtureSource());
    REQUIRE(built.has_value());

    const auto& mesh = built.value();

    // Four of five polygons kept — the island went with its component — and the one link onto it.
    REQUIRE(mesh.polygonCount() == 4);
    REQUIRE(mesh.droppedPolygons == 1);
    REQUIRE(mesh.droppedComponents == 1);
    REQUIRE(mesh.linkCount() == 3);
    REQUIRE(mesh.droppedLinks == 1);
    REQUIRE(mesh.areas.size() == 2);
    REQUIRE_THAT(mesh.agentRadiusMetres, WithinAbs(1.0, 1e-9));

    // The kept polygons keep their order: 0, 1, 2 and then the bridge as 3.
    REQUIRE_THAT(mesh.x0[3], WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(mesh.y00[3], WithinAbs(6.0, 1e-9));
    REQUIRE(mesh.area[2] == 1);

    // Every link is at both its ends, and the neighbour at the other end is the other polygon.
    REQUIRE(degree(mesh, 0) == 2);
    REQUIRE(degree(mesh, 1) == 2);
    REQUIRE(degree(mesh, 2) == 2);
    REQUIRE(degree(mesh, 3) == 0);

    auto neighboursOfOne = std::vector<std::uint32_t>();
    for (auto entry = mesh.neighbourStarts[1]; entry < mesh.neighbourStarts[2]; entry++)
    {
        neighboursOfOne.push_back(navMeshNeighbour(mesh, 1, mesh.neighbourLinks[entry]));
    }
    REQUIRE(neighboursOfOne.size() == 2);
    REQUIRE(
        ((neighboursOfOne[0] == 0 && neighboursOfOne[1] == 2) || (neighboursOfOne[0] == 2 && neighboursOfOne[1] == 0)));

    // The bilinear height and the centre.
    REQUIRE_THAT(navMeshHeight(mesh, 1, 15.0, 5.0), WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(navMeshHeight(mesh, 1, 20.0, 0.0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(navMeshHeight(mesh, 1, 10.0, 10.0), WithinAbs(0.0, 1e-9));
    const auto centre = navMeshCentre(mesh, 1);
    REQUIRE_THAT(centre.x, WithinAbs(15.0, 1e-9));
    REQUIRE_THAT(centre.y, WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(centre.z, WithinAbs(5.0, 1e-9));

    // The bucket covers the kept extent and every polygon is in it.
    REQUIRE(mesh.columns >= 3);
    REQUIRE(mesh.rows >= 3);
    REQUIRE(mesh.bucketEntries.size() >= mesh.polygonCount());
}

TEST_CASE("a point finds the polygon it is on, a bridge finds the bridge, and a point in the margin finds "
          "the nearest polygon within reach",
          "[police][police-navmesh]")
{
    const auto built = buildNavMesh(fixtureSource());
    REQUIRE(built.has_value());

    const auto& mesh = built.value();

    // On the road under the bridge: the road, not the bridge six metres up.
    const auto onRoad = locateOnNavMesh(mesh, glm::dvec3(5.0, 0.1, 5.0), 0.0);
    REQUIRE(onRoad.found());
    REQUIRE(onRoad.polygon == 0);
    REQUIRE_THAT(onRoad.heightMetres, WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(onRoad.planarMetres, WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(onRoad.verticalMetres, WithinAbs(0.1, 1e-9));

    // On the bridge: the bridge.
    const auto onBridge = locateOnNavMesh(mesh, glm::dvec3(5.0, 6.1, 5.0), 0.0);
    REQUIRE(onBridge.found());
    REQUIRE(onBridge.polygon == 3);

    // Halfway between the two in height, past the tolerance of both: nothing.
    REQUIRE_FALSE(locateOnNavMesh(mesh, glm::dvec3(5.0, 3.0, 5.0), 0.0, 2.0).found());

    // On the sloping tile, the mesh's own height under the point.
    const auto onSlope = locateOnNavMesh(mesh, glm::dvec3(15.0, 0.6, 5.0), 0.0);
    REQUIRE(onSlope.polygon == 1);
    REQUIRE_THAT(onSlope.heightMetres, WithinAbs(0.5, 1e-9));

    // A metre off the sloping tile's far edge — the lane point in the bake's margin: found within
    // one and a half metres, at the edge's own height, and not within half a metre.
    const auto inMargin = locateOnNavMesh(mesh, glm::dvec3(21.0, 0.9, 5.0), 1.5);
    REQUIRE(inMargin.found());
    REQUIRE(inMargin.polygon == 1);
    REQUIRE_THAT(inMargin.planarMetres, WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(inMargin.heightMetres, WithinAbs(1.0, 1e-9));
    REQUIRE_FALSE(locateOnNavMesh(mesh, glm::dvec3(21.0, 0.9, 5.0), 0.5).found());

    // Beside the grass strip, the grass and not the road further away.
    const auto nearGrass = locateOnNavMesh(mesh, glm::dvec3(21.0, 0.0, 15.0), 1.5);
    REQUIRE(nearGrass.found());
    REQUIRE(nearGrass.polygon == 2);

    // The island's footprint: dropped with its component, so nothing is there.
    REQUIRE_FALSE(locateOnNavMesh(mesh, glm::dvec3(35.0, 0.0, 5.0), 1.5).found());

    // Well off the bucket altogether.
    REQUIRE_FALSE(locateOnNavMesh(mesh, glm::dvec3(-500.0, 0.0, 900.0), 1.5).found());
}

// The real export, by hand: `OSR_NAVMESH_FILE=RaceEngineSandbox/assets/Tracks/gcp/grand_city_parkway_navmesh.json
// EngineTests "[.navmesh-load]"`. Prints what the game's log line will say, and puts the spawn and the
// five lane points the bake's audit lists as off the mesh through `locateOnNavMesh`.
TEST_CASE("the real navmesh loads, and the audit's off-mesh lane points are found within the margin", "[.navmesh-load]")
{
    const auto* path = std::getenv("OSR_NAVMESH_FILE");
    REQUIRE(path != nullptr);

    const auto loaded = loadNavMesh(path);
    REQUIRE(loaded.has_value());

    const auto& mesh = loaded.value();

    std::printf("navmesh %s: %zu polygons kept (%zu dropped in %zu minor components), %zu links (%zu dropped), "
                "%zu bucket entries over %d x %d cells; read %.0f ms, parse %.0f ms, build %.0f ms\n",
                path, mesh.polygonCount(), mesh.droppedPolygons, mesh.droppedComponents, mesh.linkCount(),
                mesh.droppedLinks, mesh.bucketEntries.size(), mesh.columns, mesh.rows, 1000.0 * mesh.timing.readSeconds,
                1000.0 * mesh.timing.parseSeconds, 1000.0 * mesh.timing.buildSeconds);

    for (const auto& area : mesh.areas)
    {
        std::printf("  area %s cost %.1f friction %.2f\n", area.key.c_str(), area.costHint, area.friction);
    }

    const auto spawn = locateOnNavMesh(mesh, glm::dvec3(-588.69733, 0.15, -18.80842), 0.0);
    REQUIRE(spawn.found());
    std::printf("  spawn on polygon %u (%s), height %.3f\n", spawn.polygon,
                mesh.areas[mesh.area[spawn.polygon]].key.c_str(), spawn.heightMetres);

    const auto offMesh = std::vector<glm::dvec3>{glm::dvec3(-360.21, -0.05, 442.57), glm::dvec3(-253.55, -0.03, 513.02),
                                                 glm::dvec3(-113.77, -0.03, 559.13), glm::dvec3(417.16, -0.03, 394.01),
                                                 glm::dvec3(63.17, -0.03, 246.48)};

    for (const auto& point : offMesh)
    {
        const auto exact = locateOnNavMesh(mesh, point, 0.0);
        const auto near = locateOnNavMesh(mesh, point, 1.5);
        std::printf("  lane point (%.2f, %.2f, %.2f): on the mesh %s; within 1.5 m: polygon %u at %.2f m (%s)\n",
                    point.x, point.y, point.z, exact.found() ? "yes" : "no", near.polygon, near.planarMetres,
                    near.found() ? mesh.areas[mesh.area[near.polygon]].key.c_str() : "-");
        REQUIRE(near.found());
    }
}
