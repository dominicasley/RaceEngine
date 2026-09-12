// The pursuit router: the roads and the ground between them, on a network and a mesh built in code
// (docs/pursuit-navigation-brief.md, stages 1 and 1b).
//
// **No world and no vehicle model in the room.** Every case is a request against a router and a
// property of the route it answers with: where it goes, what it avoids, which layer it took, how
// long it took. The unit that drives the route is the director's business and is proved beside it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::buildLaneNetwork;
using raceengine::buildNavMesh;
using raceengine::LaneNetwork;
using raceengine::LaneSource;
using raceengine::NavMesh;
using raceengine::NavMeshArea;
using raceengine::NavMeshLink;
using raceengine::NavMeshPolygon;
using raceengine::NavMeshSource;
using raceengine::projectOntoRoute;
using raceengine::pullThroughPortals;
using raceengine::PursuitRoute;
using raceengine::PursuitRouter;
using raceengine::PursuitRouterOptions;
using raceengine::RouteLegKind;
using raceengine::RouteRequest;
using raceengine::walkNavMesh;

using Catch::Matchers::WithinAbs;

namespace
{

// A rectangle in plan.
struct Box
{
    double x0 = 0.0;
    double z0 = 0.0;
    double x1 = 0.0;
    double z1 = 0.0;

    [[nodiscard]] bool contains(const double x, const double z, const double margin = 0.0) const
    {
        return x > x0 + margin && x < x1 - margin && z > z0 + margin && z < z1 - margin;
    }

    // The box grown by the bake's erosion: what the mesh leaves out around a solid.
    [[nodiscard]] Box eroded(const double radius) const
    {
        return Box{.x0 = x0 - radius, .z0 = z0 - radius, .x1 = x1 + radius, .z1 = z1 + radius};
    }
};

// A mesh in the export's own format: the plane tiled into cells, every cell that is not solid a
// polygon, every pair of side-by-side cells a link on their shared edge. `areaOf` says which surface
// a cell is; `solid` says which cells the bake left out.
[[nodiscard]] NavMesh tileMesh(const Box& extent, const double cell, const std::function<bool(double, double)>& solid,
                               const std::function<std::uint32_t(double, double)>& areaOf)
{
    auto source = NavMeshSource{};
    source.areas = {NavMeshArea{.key = "ROAD", .costHint = 1.0, .friction = 0.98},
                    NavMeshArea{.key = "GRASS", .costHint = 5.0, .friction = 0.85}};
    source.agentRadiusMetres = 1.0;

    const auto columns = static_cast<int>(std::lround((extent.x1 - extent.x0) / cell));
    const auto rows = static_cast<int>(std::lround((extent.z1 - extent.z0) / cell));

    auto index = std::vector<std::uint32_t>(static_cast<std::size_t>(columns * rows), raceengine::noPolygon);

    for (auto row = 0; row < rows; row++)
    {
        for (auto column = 0; column < columns; column++)
        {
            const auto x0 = extent.x0 + cell * static_cast<double>(column);
            const auto z0 = extent.z0 + cell * static_cast<double>(row);
            const auto centreX = x0 + 0.5 * cell;
            const auto centreZ = z0 + 0.5 * cell;

            if (solid(centreX, centreZ))
            {
                continue;
            }

            index[static_cast<std::size_t>(row * columns + column)] =
                static_cast<std::uint32_t>(source.polygons.size());
            source.polygons.push_back(NavMeshPolygon{.x0 = x0,
                                                     .z0 = z0,
                                                     .x1 = x0 + cell,
                                                     .z1 = z0 + cell,
                                                     .area = areaOf(centreX, centreZ),
                                                     .component = 0});
        }
    }

    for (auto row = 0; row < rows; row++)
    {
        for (auto column = 0; column < columns; column++)
        {
            const auto here = index[static_cast<std::size_t>(row * columns + column)];

            if (here == raceengine::noPolygon)
            {
                continue;
            }

            const auto& polygon = source.polygons[here];

            if (column + 1 < columns)
            {
                const auto east = index[static_cast<std::size_t>(row * columns + column + 1)];

                if (east != raceengine::noPolygon)
                {
                    source.links.push_back(NavMeshLink{.a = std::min(here, east),
                                                       .b = std::max(here, east),
                                                       .p = glm::dvec3(polygon.x1, 0.0, polygon.z0),
                                                       .q = glm::dvec3(polygon.x1, 0.0, polygon.z1)});
                }
            }

            if (row + 1 < rows)
            {
                const auto north = index[static_cast<std::size_t>((row + 1) * columns + column)];

                if (north != raceengine::noPolygon)
                {
                    source.links.push_back(NavMeshLink{.a = std::min(here, north),
                                                       .b = std::max(here, north),
                                                       .p = glm::dvec3(polygon.x0, 0.0, polygon.z1),
                                                       .q = glm::dvec3(polygon.x1, 0.0, polygon.z1)});
                }
            }
        }
    }

    auto built = buildNavMesh(source);
    REQUIRE(built.has_value());

    return std::move(built).value();
}

[[nodiscard]] NavMesh roadMesh(const Box& extent, const double cell, const std::vector<Box>& solids)
{
    return tileMesh(
        extent, cell, [solids](const double x, const double z)
        { return std::ranges::any_of(solids, [&](const Box& box) { return box.eroded(1.0).contains(x, z); }); },
        [](const double, const double) { return std::uint32_t{0}; });
}

[[nodiscard]] LaneNetwork network(std::vector<LaneSource> sources)
{
    auto built = buildLaneNetwork(std::move(sources));
    REQUIRE(built.has_value());

    return std::move(built).value();
}

// A straight lane along z at a stated x, from one z to another, points every twenty metres.
[[nodiscard]] LaneSource straightLane(const int id, const double x, const double zFrom, const double zTo)
{
    auto points = std::vector<glm::dvec3>();
    const auto count = std::max(1, static_cast<int>(std::abs(zTo - zFrom) / 20.0));

    for (auto index = 0; index <= count; index++)
    {
        points.push_back(
            glm::dvec3(x, 0.0, zFrom + (zTo - zFrom) * static_cast<double>(index) / static_cast<double>(count)));
    }

    return LaneSource{
        .id = id, .name = "lane " + std::to_string(id), .speedLimitMetresPerSecond = 15.8, .points = std::move(points)};
}

// Two carriageways as one lane: up at `xUp`, across at the far end, down at `xDown`.
[[nodiscard]] LaneSource hairpinLane(const int id, const double xUp, const double xDown, const double zNear,
                                     const double zFar)
{
    auto points = std::vector<glm::dvec3>();
    const auto count = std::max(1, static_cast<int>((zFar - zNear) / 20.0));

    for (auto index = 0; index <= count; index++)
    {
        points.push_back(
            glm::dvec3(xUp, 0.0, zNear + (zFar - zNear) * static_cast<double>(index) / static_cast<double>(count)));
    }

    const auto across = std::max(1, static_cast<int>(std::abs(xDown - xUp) / 10.0));

    for (auto index = 1; index <= across; index++)
    {
        points.push_back(
            glm::dvec3(xUp + (xDown - xUp) * static_cast<double>(index) / static_cast<double>(across), 0.0, zFar));
    }

    for (auto index = 1; index <= count; index++)
    {
        points.push_back(
            glm::dvec3(xDown, 0.0, zFar - (zFar - zNear) * static_cast<double>(index) / static_cast<double>(count)));
    }

    return LaneSource{.id = id, .name = "hairpin", .speedLimitMetresPerSecond = 15.8, .points = std::move(points)};
}

[[nodiscard]] bool anyPointInside(const PursuitRoute& route, const Box& box, const double margin = 0.05)
{
    return std::ranges::any_of(route.points, [&](const raceengine::RoutePoint& point)
                               { return box.contains(point.positionMetres.x, point.positionMetres.z, margin); });
}

[[nodiscard]] bool hasLeg(const PursuitRoute& route, const RouteLegKind kind)
{
    return std::ranges::any_of(route.legs, [&](const raceengine::RouteLeg& leg) { return leg.kind == kind; });
}

[[nodiscard]] double planar(const glm::dvec3& a, const glm::dvec3& b)
{
    return std::hypot(a.x - b.x, a.z - b.z);
}

// The route's polyline is what a unit drives: it starts where the unit is, ends where the player is,
// and is never further than the spacing between points.
void requireWellFormed(const PursuitRoute& route, const RouteRequest& request, const double spacing = 4.0)
{
    REQUIRE(route.points.size() >= 2);
    REQUIRE_THAT(planar(route.points.front().positionMetres, request.fromMetres), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(planar(route.points.back().positionMetres, request.toMetres), WithinAbs(0.0, 1e-6));

    for (auto index = std::size_t{1}; index < route.points.size(); index++)
    {
        const auto gap = glm::length(route.points[index].positionMetres - route.points[index - 1].positionMetres);
        REQUIRE(gap <= spacing + 1e-6);
        REQUIRE(route.points[index].distanceMetres >= route.points[index - 1].distanceMetres);
    }

    REQUIRE_THAT(route.points.back().distanceMetres, WithinAbs(route.lengthMetres, 1e-9));
}

} // namespace

TEST_CASE("the string pull bends round the inner corner of an L and a straight line over the mesh knows a wall",
          "[police][police-route]")
{
    // Two polygons in an L: the upright and the arm, sharing the arm's bottom edge over the upright.
    auto source = NavMeshSource{};
    source.areas = {NavMeshArea{.key = "ROAD", .costHint = 1.0}};
    source.polygons = {NavMeshPolygon{.x0 = 0.0, .z0 = 0.0, .x1 = 10.0, .z1 = 30.0},
                       NavMeshPolygon{.x0 = 0.0, .z0 = 30.0, .x1 = 40.0, .z1 = 40.0}};
    source.links = {NavMeshLink{.a = 0, .b = 1, .p = glm::dvec3(0.0, 0.0, 30.0), .q = glm::dvec3(10.0, 0.0, 30.0)}};

    const auto built = buildNavMesh(source);
    REQUIRE(built.has_value());
    const auto& mesh = built.value();

    const auto links = std::vector<std::uint32_t>{0};

    // From the foot of the upright to the end of the arm: round the inner corner at (10, 30).
    const auto pulled = pullThroughPortals(mesh, glm::dvec3(5.0, 0.0, 5.0), links, glm::dvec3(35.0, 0.0, 35.0));
    REQUIRE(pulled.size() == 3);
    REQUIRE_THAT(pulled[1].x, WithinAbs(10.0, 1e-9));
    REQUIRE_THAT(pulled[1].z, WithinAbs(30.0, 1e-9));

    // ...and the other way, the same corner.
    const auto back = pullThroughPortals(mesh, glm::dvec3(35.0, 0.0, 35.0), links, glm::dvec3(5.0, 0.0, 5.0));
    REQUIRE(back.size() == 3);
    REQUIRE_THAT(back[1].x, WithinAbs(10.0, 1e-9));
    REQUIRE_THAT(back[1].z, WithinAbs(30.0, 1e-9));

    // Straight up the upright into the arm: the line stays on the mesh; from the foot to the end of
    // the arm it leaves the upright through its wall.
    const auto straight = walkNavMesh(mesh, 0, glm::dvec3(5.0, 0.0, 5.0), glm::dvec3(5.0, 0.0, 35.0));
    REQUIRE(straight.reached);
    REQUIRE_THAT(straight.costMetres, WithinAbs(30.0, 1e-9));
    REQUIRE(straight.lastPolygon == 1);

    const auto diagonal = walkNavMesh(mesh, 0, glm::dvec3(5.0, 0.0, 5.0), glm::dvec3(35.0, 0.0, 35.0));
    REQUIRE_FALSE(diagonal.reached);
    REQUIRE(diagonal.lastPolygon == 0);
}

TEST_CASE("the wall: a unit beside a column of world reaches the player on the other carriageway without touching it",
          "[police][police-route]")
{
    // Two carriageways thirty metres apart joined at z = 300, a wall between them from the near end
    // to z = 200, the mesh all road.
    const auto wall = Box{.x0 = 11.0, .z0 = -19.0, .x1 = 19.0, .z1 = 199.0};
    const auto mesh = roadMesh(Box{.x0 = -20.0, .z0 = -20.0, .x1 = 50.0, .z1 = 320.0}, 5.0, {wall});
    const auto lanes = network({hairpinLane(1, 0.0, 30.0, 0.0, 300.0)});

    auto router = PursuitRouter(lanes, &mesh);
    REQUIRE(router.hasGround());
    REQUIRE(router.lanePlacesOnMesh() == router.lanePlaceCount());

    const auto request =
        RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 10.0), .toMetres = glm::dvec3(30.0, 0.0, 10.0)};
    const auto route = router.route(request);

    REQUIRE(route.found);
    REQUIRE_FALSE(route.blocked);
    requireWellFormed(route, request);
    REQUIRE_FALSE(anyPointInside(route, wall));

    // Round the wall's end over the ground — up one face, across its end, down the other, the
    // string pull taking the diagonals, about 390 m — rather than round the hairpin (about 610).
    REQUIRE(route.lengthMetres > 370.0);
    REQUIRE(route.lengthMetres < 520.0);
    REQUIRE(hasLeg(route, RouteLegKind::Ground));

    // The same request with no mesh named routes on the lanes alone, round the hairpin.
    auto lanesOnly = PursuitRouter(lanes, nullptr);
    REQUIRE_FALSE(lanesOnly.hasGround());

    const auto road = lanesOnly.route(request);
    REQUIRE(road.found);
    REQUIRE_FALSE(road.direct);
    REQUIRE(road.legs.size() == 1);
    REQUIRE(road.legs[0].kind == RouteLegKind::Road);
    REQUIRE(road.lengthMetres > 600.0);
    requireWellFormed(road, request);
}

TEST_CASE("the gap: two lanes end to end are bridged by the ground, and a wall across the gap sends the route round it",
          "[police][police-route]")
{
    const auto lanes = network({straightLane(1, 0.0, 0.0, 100.0), straightLane(2, 0.0, 120.0, 220.0)});
    const auto request =
        RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 10.0), .toMetres = glm::dvec3(0.0, 0.0, 210.0)};

    SECTION("the plate between them")
    {
        const auto mesh = roadMesh(Box{.x0 = -60.0, .z0 = -10.0, .x1 = 60.0, .z1 = 230.0}, 5.0, {});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        REQUIRE_FALSE(route.blocked);
        requireWellFormed(route, request);
        REQUIRE_THAT(route.lengthMetres, WithinAbs(200.0, 1.0));
        REQUIRE((route.direct || hasLeg(route, RouteLegKind::Ground)));
    }

    SECTION("a wall across the gap, short of the plate's edges")
    {
        const auto wall = Box{.x0 = -19.0, .z0 = 106.0, .x1 = 19.0, .z1 = 114.0};
        const auto mesh = roadMesh(Box{.x0 = -60.0, .z0 = -10.0, .x1 = 60.0, .z1 = 230.0}, 5.0, {wall});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        REQUIRE_FALSE(route.blocked);
        REQUIRE_FALSE(route.direct);
        requireWellFormed(route, request);
        REQUIRE_FALSE(anyPointInside(route, wall));
        // Round the wall's end at x = 20: two diagonals of 97 m and the wall's 10 m thickness, against
        // the 200 m straight.
        REQUIRE(route.lengthMetres > 204.0);
        REQUIRE(route.lengthMetres < 240.0);
        REQUIRE(hasLeg(route, RouteLegKind::Ground));
    }

    SECTION("a wall across the whole plate")
    {
        const auto wall = Box{.x0 = -70.0, .z0 = 106.0, .x1 = 70.0, .z1 = 114.0};
        const auto mesh = roadMesh(Box{.x0 = -60.0, .z0 = -10.0, .x1 = 60.0, .z1 = 230.0}, 5.0, {wall});
        auto router = PursuitRouter(lanes, &mesh);

        // Nothing connects: the straight line is the route of last resort, and says so.
        const auto route = router.route(request);
        REQUIRE_FALSE(route.found);
        REQUIRE(route.blocked);
        REQUIRE(route.direct);
        requireWellFormed(route, request);
    }

    // And the lanes alone: the two are not joined, so the answer is the last resort.
    auto lanesOnly = PursuitRouter(lanes, nullptr);
    const auto road = lanesOnly.route(request);
    REQUIRE_FALSE(road.found);
    REQUIRE(road.blocked);
}

TEST_CASE("the median: the unit crosses the grass when the road round is dear and takes the road when it is not, "
          "and a fence sends it round",
          "[police][police-route]")
{
    // Two carriageways twelve metres apart with a four-metre grass median between them; road under
    // each carriageway.
    const auto extent = Box{.x0 = -10.0, .z0 = -10.0, .x1 = 22.0, .z1 = 330.0};
    const auto areaOf = [](const double x, const double)
    {
        return x > 4.0 && x < 8.0 ? std::uint32_t{1} : std::uint32_t{0};
    };

    const auto request =
        RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 50.0), .toMetres = glm::dvec3(12.0, 0.0, 50.0)};

    SECTION("the hairpin is far: the grass")
    {
        const auto mesh = tileMesh(extent, 2.0, [](const double, const double) { return false; }, areaOf);
        const auto lanes = network({hairpinLane(1, 0.0, 12.0, 0.0, 300.0)});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        requireWellFormed(route, request);
        REQUIRE((route.direct || hasLeg(route, RouteLegKind::Ground)));
        REQUIRE(route.lengthMetres < 30.0);
        // Four metres of grass at five times the off-road factor of four, four of road either side at
        // one, and the twelve for leaving the road (docs/police-driving-brief.md §10): about a hundred,
        // and still a fraction of six hundred metres of road round the hairpin.
        REQUIRE_THAT(route.costMetres, WithinAbs(96.0, 10.0));
    }

    SECTION("the hairpin is near: the road")
    {
        const auto mesh = tileMesh(extent, 2.0, [](const double, const double) { return false; }, areaOf);
        const auto lanes = network({hairpinLane(1, 0.0, 12.0, 0.0, 55.0)});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        requireWellFormed(route, request);
        REQUIRE_FALSE(route.direct);
        REQUIRE_FALSE(hasLeg(route, RouteLegKind::Ground));
        REQUIRE(std::ranges::none_of(route.points, [](const raceengine::RoutePoint& point) { return point.ground; }));
        REQUIRE_THAT(route.lengthMetres, WithinAbs(22.0, 3.0));
    }

    SECTION("a fence down the median: the road, however far the hairpin")
    {
        const auto fence = Box{.x0 = 5.0, .z0 = -20.0, .x1 = 7.0, .z1 = 290.0};
        const auto mesh = tileMesh(
            extent, 2.0, [fence](const double x, const double z) { return fence.eroded(1.0).contains(x, z); }, areaOf);
        const auto lanes = network({hairpinLane(1, 0.0, 12.0, 0.0, 300.0)});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        requireWellFormed(route, request);
        REQUIRE_FALSE(route.direct);
        REQUIRE_FALSE(anyPointInside(route, fence));
        // Round the fence's far end at z = 290 over the ground, or round the hairpin at 300: either
        // way about half a kilometre, not the twelve metres across.
        REQUIRE(route.lengthMetres > 450.0);
    }
}

TEST_CASE(
    "off the network: a player on the plate is reached over the ground, and one behind a wall is reached round it",
    "[police][police-route]")
{
    const auto lanes = network({straightLane(1, 0.0, 0.0, 300.0)});
    const auto request =
        RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 20.0), .toMetres = glm::dvec3(30.0, 0.0, 150.0)};

    SECTION("the open plate")
    {
        const auto mesh = roadMesh(Box{.x0 = -20.0, .z0 = -10.0, .x1 = 60.0, .z1 = 310.0}, 5.0, {});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        requireWellFormed(route, request);
        REQUIRE(route.points.back().ground);
        // The straight line is clear and shorter than the road and the turn: it is the route.
        REQUIRE(route.direct);
        REQUIRE_THAT(route.lengthMetres, WithinAbs(std::hypot(30.0, 130.0), 0.5));
    }

    SECTION("behind a wall")
    {
        const auto wall = Box{.x0 = 16.0, .z0 = 61.0, .x1 = 19.0, .z1 = 179.0};
        const auto mesh = roadMesh(Box{.x0 = -20.0, .z0 = -10.0, .x1 = 60.0, .z1 = 310.0}, 5.0, {wall});
        auto router = PursuitRouter(lanes, &mesh);

        const auto route = router.route(request);
        REQUIRE(route.found);
        REQUIRE_FALSE(route.direct);
        requireWellFormed(route, request);
        REQUIRE_FALSE(anyPointInside(route, wall));
        REQUIRE(hasLeg(route, RouteLegKind::Ground));
        REQUIRE(route.points.back().ground);
    }

    SECTION("with no mesh, the last resort")
    {
        auto router = PursuitRouter(lanes, nullptr);
        const auto route = router.route(request);
        REQUIRE_FALSE(route.found);
        REQUIRE(route.blocked);
        REQUIRE(route.direct);
        requireWellFormed(route, request);
    }
}

TEST_CASE("round the building: the route hugs the building's corners, filleted, and never enters it",
          "[police][police-route]")
{
    // A lane up the west side of a fifty-metre building; the player in the street on its east side,
    // off the network. The straight line runs into the building. The mesh's hole is the building
    // eroded by the bake's metre, so the building's faces sit a metre inside the mesh's boundary.
    const auto building = Box{.x0 = 11.0, .z0 = 101.0, .x1 = 59.0, .z1 = 149.0};
    const auto mesh = roadMesh(Box{.x0 = -20.0, .z0 = -10.0, .x1 = 100.0, .z1 = 260.0}, 5.0, {building});
    const auto lanes = network({straightLane(1, 0.0, 0.0, 250.0)});
    auto router = PursuitRouter(lanes, &mesh);

    const auto request =
        RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 20.0), .toMetres = glm::dvec3(70.0, 0.0, 125.0)};
    const auto route = router.route(request);

    REQUIRE(route.found);
    REQUIRE_FALSE(route.direct);
    requireWellFormed(route, request);
    REQUIRE_FALSE(anyPointInside(route, building));
    REQUIRE(hasLeg(route, RouteLegKind::Ground));

    // Round the building's south-east corner: from the unit to the corner (100 m) and on to the
    // player (27 m), give or take the fillets, and never the 200 m round the north.
    REQUIRE(route.lengthMetres > 120.0);
    REQUIRE(route.lengthMetres < 160.0);

    // The string pull bends at the building's eroded corners, and the fillet rounds them: no route
    // point sits on a corner itself, and the arc's intrusion past the eroded line is inside the
    // stated 0.6 m — the car's centre stays 0.4 m off the wall.
    const auto eroded = building.eroded(1.0);
    for (const auto corner : {glm::dvec3(eroded.x0, 0.0, eroded.z0), glm::dvec3(eroded.x1, 0.0, eroded.z0),
                              glm::dvec3(eroded.x0, 0.0, eroded.z1), glm::dvec3(eroded.x1, 0.0, eroded.z1)})
    {
        for (const auto& point : route.points)
        {
            REQUIRE(planar(point.positionMetres, corner) > 0.3);
        }
    }

    REQUIRE_FALSE(anyPointInside(route, building.eroded(1.0 - 0.6 - 1e-6)));

    // The turns are spread over the fillet rather than taken at one vertex: no consecutive pair of
    // segments turns through more than sixty degrees.
    for (auto index = std::size_t{1}; index + 1 < route.points.size(); index++)
    {
        const auto a = route.points[index].positionMetres - route.points[index - 1].positionMetres;
        const auto b = route.points[index + 1].positionMetres - route.points[index].positionMetres;
        const auto la = std::hypot(a.x, a.z);
        const auto lb = std::hypot(b.x, b.z);

        if (la < 1e-6 || lb < 1e-6)
        {
            continue;
        }

        const auto cosine = (a.x * b.x + a.z * b.z) / (la * lb);
        REQUIRE(cosine > std::cos(60.0 * std::numbers::pi / 180.0));
    }

    // A unit part-way along finds its place beside last tick's rather than by walking the route.
    const auto midway = route.points[route.points.size() / 2].positionMetres + glm::dvec3(0.5, 0.0, 0.0);
    const auto whole = projectOntoRoute(route, midway);
    const auto windowed = projectOntoRoute(route, midway, route.points.size() / 2, 3);
    REQUIRE(whole.segment == windowed.segment);
    REQUIRE_THAT(whole.distanceMetres, WithinAbs(windowed.distanceMetres, 1e-9));
    REQUIRE(whole.offsetMetres < 0.6);
}

TEST_CASE(
    "the loop and the wrong way: a player behind the unit on a ring road is reached the wrong way at twice the metres",
    "[police][police-route]")
{
    // A ring road of a kilometre, its last point a few metres short of its first, so the network
    // derives the loop.
    auto points = std::vector<glm::dvec3>();
    const auto radius = 160.0;
    const auto count = 100;

    for (auto index = 0; index < count; index++)
    {
        const auto angle = 2.0 * std::numbers::pi * static_cast<double>(index) / static_cast<double>(count);
        points.push_back(glm::dvec3(radius * std::sin(angle), 0.0, radius * std::cos(angle)));
    }

    const auto lanes =
        network({LaneSource{.id = 1, .name = "ring", .speedLimitMetresPerSecond = 15.8, .points = points}});
    REQUIRE(lanes.lanes[0].loop);

    auto router = PursuitRouter(lanes, nullptr);

    // The unit at the start of the lane, the player fifty metres behind it round the ring.
    const auto behind = 2.0 * std::numbers::pi * (1.0 - 50.0 / (2.0 * std::numbers::pi * radius));
    const auto request = RouteRequest{
        .fromMetres = points[0], .toMetres = glm::dvec3(radius * std::sin(behind), 0.0, radius * std::cos(behind))};
    const auto route = router.route(request);

    REQUIRE(route.found);
    REQUIRE(route.legs.size() == 1);
    REQUIRE(route.legs[0].kind == RouteLegKind::Road);
    REQUIRE(route.legs[0].reversed);
    REQUIRE_THAT(route.lengthMetres, WithinAbs(50.0, 3.0));
    REQUIRE_THAT(route.costMetres, WithinAbs(100.0, 6.0));
    requireWellFormed(route, request);

    // And ahead of it, the right way at the metres alone.
    const auto ahead = 2.0 * std::numbers::pi * (50.0 / (2.0 * std::numbers::pi * radius));
    const auto forward = router.route(RouteRequest{
        .fromMetres = points[0], .toMetres = glm::dvec3(radius * std::sin(ahead), 0.0, radius * std::cos(ahead))});
    REQUIRE(forward.found);
    REQUIRE(forward.legs.size() == 1);
    REQUIRE_FALSE(forward.legs[0].reversed);
    REQUIRE_THAT(forward.costMetres, WithinAbs(50.0, 3.0));
}

TEST_CASE("a hundred thousand rectangles: a route across a city of blocks is found inside the time pinned here",
          "[police][police-route][police-route-timing]")
{
    // A 3.6 km by 2.4 km city of 6 m cells with a grid of 90 m blocks and 30 m streets: 240 000
    // cells of which the streets are 105 000 polygons; and one lane down its western edge.
    const auto extent = Box{.x0 = 0.0, .z0 = 0.0, .x1 = 3600.0, .z1 = 2400.0};
    const auto mesh = tileMesh(
        extent, 6.0,
        [](const double x, const double z)
        {
            const auto column = std::fmod(x, 120.0);
            const auto row = std::fmod(z, 120.0);

            return column >= 30.0 && row >= 30.0;
        },
        [](const double, const double) { return std::uint32_t{0}; });

    REQUIRE(mesh.polygonCount() >= 100000);

    const auto lanes = network({straightLane(1, 15.0, 0.0, 2400.0)});

    const auto built = std::chrono::steady_clock::now();
    auto router = PursuitRouter(lanes, &mesh);
    const auto ready = std::chrono::steady_clock::now();

    // Corner to corner, the long way across the blocks.
    const auto request =
        RouteRequest{.fromMetres = glm::dvec3(15.0, 0.0, 15.0), .toMetres = glm::dvec3(3495.0, 0.0, 2385.0)};

    auto worst = 0.0;
    auto route = PursuitRoute{};

    for (auto run = 0; run < 3; run++)
    {
        route = router.route(request);
        worst = std::max(worst, route.searchSeconds);
    }

    std::printf("route over %zu polygons, %zu links: %zu expanded, %.1f ms (worst of three), built in %.1f ms, "
                "%zu points, %.0f m\n",
                mesh.polygonCount(), mesh.linkCount(), route.expanded, 1000.0 * worst,
                1000.0 * std::chrono::duration<double>(ready - built).count(), route.points.size(), route.lengthMetres);

    REQUIRE(route.found);
    REQUIRE_FALSE(route.blocked);
    requireWellFormed(route, request);
    // Between the straight line (4.2 km) and the Manhattan distance (5.9 km): the string pull
    // takes every intersection on the diagonal.
    REQUIRE(route.lengthMetres > 4600.0);

    for (const auto& point : route.points)
    {
        const auto column = std::fmod(point.positionMetres.x, 120.0);
        const auto row = std::fmod(point.positionMetres.z, 120.0);
        REQUIRE_FALSE((column > 31.0 && row > 31.0 && column < 119.0 && row < 119.0));
    }

    // The pin: this machine answers in a few milliseconds; a change that makes it a hundred is a
    // change to the search and not to the city.
    REQUIRE(worst < 0.100);
}
