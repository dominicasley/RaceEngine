module;

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.traffic:Navmesh;

namespace raceengine
{

// The navigation mesh: a track's drivable area as a graph of axis-aligned rectangles, for anything
// that has to find its way to a point — a pursuing unit routing round a block, or across the grass
// where the lane export stops (docs/pursuit-navigation-brief.md, stage 1b).
//
// **It is a bake, and the bake's agent is a car.** `~/dev/ac-car-data`'s `navmesh.py` erodes the
// drivable surfaces of `_track.glb`, with both collider files solid, by a 1 m radius, a 2 m height,
// a 0.3 m step and a 30° slope. So every polygon edge is already the radius away from anything
// solid, and a route runs the car's *centre* over the mesh with no margin of its own. Dynamic props
// are solid in it (`--props all`): a unit routes round where a knocked-over bin stood. Its height
// is the mesh's, within 0.15 m of the collision hull; a wheel height is still a ray's business.
//
// The document's own account is `grand_city_parkway_engine_brief.md` §8, and the contract in
// short: a polygon is `[x0, z0, x1, z1, y00, y01, y11, y10, area, component]` — a rectangle in
// plan with a height at each corner in the order (x0,z0), (x0,z1), (x1,z1), (x1,z0), the height
// anywhere on it the bilinear of the four; a link is `[a, b, px, py, pz, qx, qy, qz]`, the edge
// P–Q that polygons `a` and `b` share, which is the portal a path crosses between them, listed
// once per pair and two-way. **Neighbouring rectangles meet at T-junctions, so adjacency is the
// link list and never a shared vertex.** Areas — ROAD, MARK, KERB, GRASS — each carry a
// `cost_hint`, which is the pursuit brief's road-1 / ground-5 preference as a number per surface.
// Only the largest connected component (`component == 0`) is kept: the other 142 on Grand City
// Parkway are courtyards and slivers a car cannot reach.
//
// Metres, in the same right-handed axes glTF, the track export and the traffic export share: +x
// left, +y up, +z forward. Nothing is converted on the way in, for the reason the lane points are
// not (`osr.game:TrafficNetwork`): the exporter states this frame and this module works in it.
//
// **Structs of arrays, and the document is scanned rather than parsed into a tree.** 180 816
// polygons and 442 632 links are five million numbers in 33 MB; a document-object reader that made
// a node of each would allocate once per polygon and spend most of a second doing it.
// `parseNavMesh` walks the root object member by member, reads the arrays it wants straight into
// rows with `std::from_chars`, and skips everything else; `buildNavMesh` lays the rows out as
// arrays, with the adjacency and the bucket in compressed rows. That is why this partition — unlike
// `:Lanes` — reads the asset itself: the shape of the read *is* the structure, and a
// `LaneSource`-style hand-off would be the per-polygon copy the format was chosen to avoid. The
// scanner is pure over a string, so a fixture builds its mesh in code through `NavMeshSource` and
// never touches the disk.

export inline constexpr std::uint32_t noPolygon = 0xffffffffu;

// One surface class of the bake, as `areas[]` states it.
export struct NavMeshArea
{
    std::string key{};
    // The edge weight a consumer that prefers the road applies to a step *onto* this surface.
    double costHint = 1.0;
    double friction = 1.0;
};

// One polygon row as the export states it, and the shape a fixture builds one in.
export struct NavMeshPolygon
{
    double x0 = 0.0;
    double z0 = 0.0;
    double x1 = 0.0;
    double z1 = 0.0;
    // Heights at (x0,z0), (x0,z1), (x1,z1), (x1,z0).
    double y00 = 0.0;
    double y01 = 0.0;
    double y11 = 0.0;
    double y10 = 0.0;
    std::uint32_t area = 0;
    std::uint32_t component = 0;
};

// One link row: the portal between polygons `a` and `b`.
export struct NavMeshLink
{
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    glm::dvec3 p{0.0};
    glm::dvec3 q{0.0};
};

// The document as rows, before it is laid out. What `parseNavMesh` returns and a fixture writes.
export struct NavMeshSource
{
    std::vector<NavMeshArea> areas{};
    std::vector<NavMeshPolygon> polygons{};
    std::vector<NavMeshLink> links{};
    // The document's own `counts`, zero where it states none; checked against what was read.
    std::size_t statedPolygons = 0;
    std::size_t statedLinks = 0;
    // The radius the mesh was eroded by, metres, zero where unstated.
    double agentRadiusMetres = 0.0;
};

export struct NavMeshOptions
{
    // The bucket's cell, metres. A polygon is at most 32 of the bake's 0.5 m cells on a side, so at
    // eight metres one touches at most three cells each way and the 2.6 km square is a hundred
    // thousand cells.
    double bucketCellMetres = 8.0;
    // The one connected component kept. Zero is the largest.
    std::uint32_t component = 0;
};

// How long the three phases of `loadNavMesh` took, for the one log line and the decision it
// informs: a load over a second is the case for a binary cache.
export struct NavMeshTiming
{
    double readSeconds = 0.0;
    double parseSeconds = 0.0;
    double buildSeconds = 0.0;
};

export struct NavMesh
{
    std::vector<NavMeshArea> areas{};

    // --- the polygons, one entry each, in the order kept -----------------------------------------
    std::vector<double> x0{};
    std::vector<double> z0{};
    std::vector<double> x1{};
    std::vector<double> z1{};
    std::vector<double> y00{};
    std::vector<double> y01{};
    std::vector<double> y11{};
    std::vector<double> y10{};
    std::vector<std::uint8_t> area{};

    // --- the links, one entry each ---------------------------------------------------------------
    std::vector<std::uint32_t> linkA{};
    std::vector<std::uint32_t> linkB{};
    std::vector<glm::dvec3> portalP{};
    std::vector<glm::dvec3> portalQ{};

    // --- adjacency, compressed rows -------------------------------------------------------------
    //
    // The links of polygon `p` are `neighbourLinks[neighbourStarts[p] .. neighbourStarts[p + 1])`,
    // and `navMeshNeighbour` names the polygon at the other end of each.
    std::vector<std::uint32_t> neighbourStarts{};
    std::vector<std::uint32_t> neighbourLinks{};

    // --- the bucket over the rectangles ---------------------------------------------------------
    //
    // Compressed rows over cells, a cell's index `row * columns + column` with the column along x
    // and the row along z; a polygon is entered in every cell it overlaps.
    double cellMetres = 8.0;
    double originX = 0.0;
    double originZ = 0.0;
    int columns = 0;
    int rows = 0;
    std::vector<std::uint32_t> bucketStarts{};
    std::vector<std::uint32_t> bucketEntries{};

    // --- what the build did, for the one log line -----------------------------------------------
    std::size_t droppedPolygons = 0;
    std::size_t droppedComponents = 0;
    std::size_t droppedLinks = 0;
    double agentRadiusMetres = 0.0;
    NavMeshTiming timing{};

    [[nodiscard]] std::size_t polygonCount() const
    {
        return x0.size();
    }

    [[nodiscard]] std::size_t linkCount() const
    {
        return linkA.size();
    }
};

// Where a point is on the mesh.
export struct NavMeshPlace
{
    std::uint32_t polygon = noPolygon;
    // The mesh's height under (or beside) the point, and how far the point was from the polygon:
    // in plan (zero inside it) and in height.
    double heightMetres = 0.0;
    double planarMetres = 0.0;
    double verticalMetres = 0.0;

    [[nodiscard]] bool found() const
    {
        return polygon != noPolygon;
    }
};

// Read the document. Refused rather than repaired: a row that is not ten numbers, a link naming a
// polygon the document does not carry, or a count that disagrees with what was read is a truncated
// export, and a unit routing on it would read as a driver fault.
export [[nodiscard]] std::expected<NavMeshSource, std::string> parseNavMesh(std::string_view document);

// Lay the rows out: keep the one component, remap the links onto what is kept, build the adjacency
// and the bucket.
export [[nodiscard]] std::expected<NavMesh, std::string> buildNavMesh(const NavMeshSource& source,
                                                                      const NavMeshOptions& options = {});

// Both, off a file, with the three phases timed onto `NavMesh::timing`.
export [[nodiscard]] std::expected<NavMesh, std::string> loadNavMesh(const std::string& filePath,
                                                                     const NavMeshOptions& options = {});

// The polygon's height at a point in plan, the bilinear of its four corners; the point is clamped
// onto the rectangle first.
export [[nodiscard]] double navMeshHeight(const NavMesh& mesh, std::uint32_t polygon, double x, double z);

// The polygon's centre, at the mesh's height.
export [[nodiscard]] glm::dvec3 navMeshCentre(const NavMesh& mesh, std::uint32_t polygon);

// The polygon at the other end of a link from this one.
export [[nodiscard]] std::uint32_t navMeshNeighbour(const NavMesh& mesh, std::uint32_t polygon, std::uint32_t link);

// The distance in plan from a point to a polygon's rectangle; zero inside it.
export [[nodiscard]] double navMeshPlanarDistance(const NavMesh& mesh, std::uint32_t polygon, double x, double z);

// Which polygon a point is on — or, off the mesh, the nearest within `reachMetres` in plan — among
// the polygons within `heightToleranceMetres` of the point's height. Nearest by the distance in
// plan and in height together, so a point on a bridge finds the bridge and not the road under it,
// and a lane point in the bake's 1 m margin finds the polygon beside it.
export [[nodiscard]] NavMeshPlace locateOnNavMesh(const NavMesh& mesh, const glm::dvec3& pointMetres,
                                                  double reachMetres, double heightToleranceMetres = 2.0);

} // namespace raceengine
