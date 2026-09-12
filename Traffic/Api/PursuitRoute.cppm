module;

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.traffic:PursuitRoute;

import :Lanes;
import :Navmesh;

namespace raceengine
{

// The pursuit router: the way from a unit to the player through the roads it has and the ground
// between them, roads preferred by a stated factor (docs/pursuit-navigation-brief.md, stages 1 and 1b).
//
// **Two layers, one search.** The road layer is the lane network's own samples — a node every four
// metres of every lane, the edges *along* a lane (its metres; twice its metres the wrong way, which
// is allowed), *round* a loop, *onto* a successor and *across* to a neighbour. The ground layer is
// the navigation mesh — a node per polygon, an edge per portal, costing the distance between portal
// midpoints times the surface's `cost_hint` (road 1, grass 5). The two meet at every lane sample
// that stands on a polygon, at zero cost either way. A* with the straight-line heuristic runs over
// both at once, so a unit on the far carriageway takes the grass when the road round is more than
// five times longer and the road when it is not, and a ring road whose connecting road the export
// left out is bridged by whatever surface actually lies between its ends.
//
// **The road layer stays the road.** The lane network keeps its direction, its lane changes and the
// population's occupant lists; the mesh never replaces it. What the mesh adds is the ground where
// the lanes run out, and the price of driving on it.
//
// **Pure.** Built once from a network and a mesh (or no mesh: a track that names none routes on its
// lanes alone), and every route is a function of the request and nothing else — no world, no
// vehicle model, no clock — so every case is proved on a network and a mesh built in code.

export struct PursuitRouterOptions
{
    // A lane driven against its direction costs this many times its metres. Two, and allowed: a
    // unit takes the wrong way when the right way is twice as long, which on a parkway of kilometre
    // loops is most of the time it matters.
    double wrongWayCostFactor = 2.0;
    // A lane change costs this on top of the road it takes, and takes this much road.
    double laneChangeCostMetres = 15.0;
    double laneChangeLengthMetres = 30.0;
    // Leaving the road for the mesh costs this, once, at the lane place it leaves from — a kerb's
    // worth. Not zero, and the reason is measured: at zero, with the mesh under the road costed at
    // one, the search left the lane at every corner of its polyline and came back a portal later,
    // because a portal hop across the corner is shorter than the polyline round it. The way back onto
    // the road costs the metres from where the search arrived on the polygon to the lane place, at
    // the polygon's hint — nothing more, and never nothing: a hop that cost nothing moved the search
    // a polygon's width for free, and ate one lane step in every polygon that held two places.
    double groundEntryCostMetres = 12.0;
    // Off the road — a polygon whose hint is more than a painted mark's — costs this many times its
    // hint on top (docs/police-driving-brief.md §10, Dominic: "they should always prefer to nav on
    // the road and only go off it if theres no other choice"): a kerb at four times its 2.0 is eight
    // metres a metre, grass twenty. With the entry cost above at twelve (three before), the straight
    // line across a corner's pavement loses to the road round it unless the road is several times
    // longer, which is what "no other choice" means on a map.
    double offRoadCostFactor = 4.0;
    // A lane sample links to the nearest polygon within this. The bake erodes the mesh by the car's
    // radius, so five of Grand City Parkway's lane points stand in that margin, a hand's width off
    // the polygon beside them.
    double lanePlaceReachMetres = 1.5;
    // A route's two ends join the nearest polygon within this. A unit shoved onto a footpath, or a
    // player parked against a wall, is still a few metres from drivable ground.
    double endpointReachMetres = 6.0;
    // ...and the nearest lane within this, laterally, either direction.
    double projectionLateralMetres = 7.0;
    // Every corner of a ground leg, and every place a ground leg meets anything, is filleted with
    // this radius or half the shorter leg, whichever is less, so the planner reads a corner and not
    // a spike.
    double filletRadiusMetres = 8.0;
    // A string-pulled corner sits on the mesh's boundary — the bake's eroded line, a car's radius off
    // the wall it bends round — and a fillet cuts *inside* the corner by its sagitta `r (1 − cos α/2)`:
    // eight metres round a right angle is 2.3 m into the wall. So at a boundary corner the radius is
    // also capped so the arc goes no further than this inside the corner (2.0 m round a right angle,
    // 8 m still below 30°). A junction with the road is not on a boundary and keeps the plain rule.
    double filletIntrusionMetres = 0.6;
    // The route polyline's spacing, the network's own.
    double sampleSpacingMetres = 4.0;
    // A search that has expanded this many nodes without reaching the player stops: a hopeless goal
    // costs a bounded time, and the straight line is the answer it always was.
    std::size_t maximumExpansions = 300000;
};

export enum class RouteLegKind : std::uint8_t
{
    // Along a lane: the lane's own polyline, in its direction or against it.
    Road,
    // Across the mesh: string-pulled through its portals and filleted.
    Ground,
    // The straight line from the unit to the player, over the mesh where it is clear all the way,
    // or as the route of last resort where nothing else connects.
    Direct
};

export struct RouteLeg
{
    RouteLegKind kind = RouteLegKind::Road;
    // Road: which lane, and the run along it from `fromMetres` to `toMetres`, unwrapped past a
    // loop's length so the run is monotonic; `toMetres < fromMetres` when driven the wrong way.
    std::size_t lane = 0;
    double fromMetres = 0.0;
    double toMetres = 0.0;
    bool reversed = false;
    // This leg's points in `PursuitRoute::points`.
    std::size_t firstPoint = 0;
    std::size_t pointCount = 0;
    // What the search paid for it, metres times the factors.
    double costMetres = 0.0;
};

export struct RoutePoint
{
    glm::dvec3 positionMetres{0.0};
    // Along the route from its first point.
    double distanceMetres = 0.0;
    // On a ground or direct leg, or the fillet where one meets the road: the planner's off-road grip
    // applies.
    bool ground = false;
};

export struct PursuitRoute
{
    // A way exists: through the graph, or the straight line clear over the mesh.
    bool found = false;
    // The straight line is the route: it was the cheapest, or it is the last resort.
    bool direct = false;
    // The last resort: nothing connected the two ends and the straight line is not known drivable.
    bool blocked = false;
    double costMetres = 0.0;
    double lengthMetres = 0.0;
    std::vector<RouteLeg> legs{};
    // The polyline a unit drives, never more than the spacing apart, first point at the unit's place
    // and last at the player's.
    std::vector<RoutePoint> points{};
    // The search, for the log and the timing pin.
    std::size_t expanded = 0;
    double searchSeconds = 0.0;
};

export struct RouteRequest
{
    glm::dvec3 fromMetres{0.0};
    glm::dvec3 toMetres{0.0};
};

// A place on a route's polyline: the nearest point to a query, its segment and its distance along.
export struct RoutePlace
{
    std::size_t segment = 0;
    double distanceMetres = 0.0;
    glm::dvec3 positionMetres{0.0};
    // How far the query was from the polyline, in plan.
    double offsetMetres = 0.0;
};

// Project a point onto the route, searching the segments from `hintSegment` to `window` segments
// either side of it (the whole route when `window` is zero), so a unit's place is found beside last
// tick's rather than by walking the route.
export [[nodiscard]] RoutePlace projectOntoRoute(const PursuitRoute& route, const glm::dvec3& pointMetres,
                                                 std::size_t hintSegment = 0, std::size_t window = 0);

// The route's point at a distance along it, clamped to its two ends.
export [[nodiscard]] glm::dvec3 routePointAt(const PursuitRoute& route, double distanceMetres);

// A straight line walked over the mesh from a point on `fromPolygon`: whether it stays on the mesh
// all the way to `to`, crossing portals only, and what it costs — each polygon's share of the line
// times its surface's hint. Plan coordinates; the heights are the mesh's.
export struct NavMeshWalk
{
    bool reached = false;
    double costMetres = 0.0;
    double lengthMetres = 0.0;
    std::uint32_t lastPolygon = noPolygon;
};

export [[nodiscard]] NavMeshWalk walkNavMesh(const NavMesh& mesh, std::uint32_t fromPolygon, const glm::dvec3& from,
                                             const glm::dvec3& to, double offRoadCostFactor = 1.0);

// The string pull: the shortest path from `from` to `to` through a sequence of portals (links, in
// travel order), as its corners — `from`, each portal endpoint the path bends round, `to`. The
// simple stupid funnel; the portal's two sides are told apart from the direction of travel.
export [[nodiscard]] std::vector<glm::dvec3> pullThroughPortals(const NavMesh& mesh, const glm::dvec3& from,
                                                                std::span<const std::uint32_t> links,
                                                                const glm::dvec3& to);

export class PursuitRouter
{
public:
    // `mesh` may be null: no ground layer, the lanes alone. Both are borrowed for the router's life.
    PursuitRouter(const LaneNetwork& network, const NavMesh* mesh, PursuitRouterOptions options = {});

    [[nodiscard]] PursuitRoute route(const RouteRequest& request);

    [[nodiscard]] bool hasGround() const;
    [[nodiscard]] const PursuitRouterOptions& settings() const;
    // What the build found, for the log line: lane places, how many stand on a polygon, lane edges.
    [[nodiscard]] std::size_t lanePlaceCount() const;
    [[nodiscard]] std::size_t lanePlacesOnMesh() const;
    [[nodiscard]] std::size_t laneEdgeCount() const;

private:
    enum class LaneEdgeKind : std::uint8_t
    {
        Along,
        AlongBack,
        Wrap,
        WrapBack,
        Join,
        JoinBack,
        Change
    };

    struct LaneEdge
    {
        std::uint32_t to = 0;
        double costMetres = 0.0;
        // The signed metres along the lane this edge covers, for a leg's unwrapped run.
        double alongMetres = 0.0;
        LaneEdgeKind kind = LaneEdgeKind::Along;
    };

    struct HeapEntry
    {
        double estimate = 0.0;
        double cost = 0.0;
        std::uint32_t node = 0;
    };

    // How a node was reached, for the walk back.
    struct Via
    {
        std::uint32_t parent = 0;
        // A lane edge index, a mesh link index, or one of the codes below.
        std::uint32_t edge = 0;
    };

    // Endpoint hooks, per route: the samples and the polygon each end joins, and at what cost.
    struct EndHook
    {
        // The lane the end projects onto, where along it, and the samples either side of that place
        // with the metres to each.
        bool onRoad = false;
        std::size_t lane = 0;
        double distanceMetres = 0.0;
        std::uint32_t sampleAhead = 0xffffffffu;
        std::uint32_t sampleBehind = 0xffffffffu;
        double metresAhead = 0.0;
        double metresBehind = 0.0;
        // The polygon the end stands on or beside, the point on it nearest the end, and the cost of
        // the gap between the two.
        std::uint32_t polygon = noPolygon;
        glm::dvec3 polygonPoint{0.0};
        double polygonCost = 0.0;
    };

    void buildLaneLayer();
    void buildMeshLinks();
    [[nodiscard]] EndHook hook(const glm::dvec3& pointMetres) const;
    [[nodiscard]] std::uint32_t nearestSample(std::size_t lane, double distanceMetres) const;
    void relax(std::uint32_t from, std::uint32_t to, double cost, const glm::dvec3& arrival, std::uint32_t edge,
               const glm::dvec3& goal);
    void reconstruct(PursuitRoute& route, const EndHook& start, const EndHook& goal, const RouteRequest& request);
    void assemble(PursuitRoute& route);
    void directRoute(PursuitRoute& route, const RouteRequest& request, double costMetres);

    const LaneNetwork* network;
    const NavMesh* mesh;
    PursuitRouterOptions options;

    // --- the road layer --------------------------------------------------------------------------
    std::vector<std::uint32_t> laneFirstSample{};
    std::vector<std::uint32_t> laneSampleCount{};
    std::vector<std::uint32_t> laneEdgeStarts{};
    std::vector<LaneEdge> laneEdges{};
    // The polygon under each sample, or none.
    std::vector<std::uint32_t> samplePolygon{};
    // ...and the samples on each polygon, compressed rows.
    std::vector<std::uint32_t> polygonSampleStarts{};
    std::vector<std::uint32_t> polygonSamples{};
    std::size_t onMesh = 0;

    // --- the search's scratch, sized once ----------------------------------------------------------
    std::uint32_t stamp = 0;
    std::vector<std::uint32_t> seen{};
    std::vector<std::uint32_t> closed{};
    std::vector<double> cost{};
    std::vector<glm::dvec3> arrival{};
    std::vector<Via> via{};
    std::vector<HeapEntry> heap{};
    std::vector<std::uint32_t> pathNodes{};
    std::vector<std::uint32_t> pathEdges{};
    // The polyline under assembly: leg vertices, then the fillet, then the spacing.
    struct Vertex
    {
        glm::dvec3 position{0.0};
        // Fillet me; and whether I sit on the mesh's boundary (a string-pulled corner) rather than
        // in the open (a junction with the road).
        bool corner = false;
        bool boundary = false;
        bool ground = false;
        std::size_t leg = 0;
    };
    std::vector<Vertex> vertices{};
    std::vector<Vertex> filleted{};
    std::vector<std::uint32_t> legLinks{};
};

// --- the route service (docs/pursuit-radio-brief.md, §1.1) ------------------------------------------
//
// The director never searches itself: it asks, drives whatever route it has, and takes the answer
// when it lands. Two services stand behind that contract. `InlineRouteService` is the router on the
// calling thread at a stated number of searches per `collect` — one, so a tick pays one search at
// most, which is what every fixture and a driven capture (whose tick count must stay a function of
// the frame) see. `PursuitRouteWorker` is the router on a thread of its own, which is what the game
// runs: a search that took 17 ms on the parkway stalled the 360 Hz simulation for six ticks, and a
// swarm asked for thirty to fifty of them a second (`docs/verify/police-profile-2026-09-10/`).
//
// A request names the unit and a serial. A newer request for the same unit supersedes one the
// service has not begun; an answer carries the serial it was asked under, and a director whose unit
// has moved on drops it. The service keeps no other state about units.

export struct PendingRoute
{
    std::uint32_t agent = 0;
    std::uint64_t serial = 0;
    RouteRequest request{};
};

export struct RouteAnswer
{
    std::uint32_t agent = 0;
    std::uint64_t serial = 0;
    PursuitRoute route{};
};

export class RouteService
{
public:
    RouteService() = default;
    virtual ~RouteService() = default;
    RouteService(const RouteService&) = delete;
    RouteService(RouteService&&) = delete;
    RouteService& operator=(const RouteService&) = delete;
    RouteService& operator=(RouteService&&) = delete;

    // Ask for a route for a unit. A newer request for the same unit replaces one not yet begun.
    virtual void request(std::uint32_t agent, std::uint64_t serial, const RouteRequest& request) = 0;
    // Take every answer finished since the last call, in the order they finished.
    virtual void collect(std::vector<RouteAnswer>& into) = 0;
    // Requests asked and not yet answered, begun or not.
    [[nodiscard]] virtual std::size_t pending() const = 0;
    // The searches answered so far and the longest of them, for the log.
    [[nodiscard]] virtual std::size_t searches() const = 0;
    [[nodiscard]] virtual double worstSearchSeconds() const = 0;
};

// The router on the calling thread: `collect` runs up to `searchesPerCollect` of the pending
// requests, oldest first, and answers them there and then. Borrows the router.
export class InlineRouteService final : public RouteService
{
public:
    explicit InlineRouteService(PursuitRouter& router, std::size_t searchesPerCollect = 1);

    void request(std::uint32_t agent, std::uint64_t serial, const RouteRequest& request) override;
    void collect(std::vector<RouteAnswer>& into) override;
    [[nodiscard]] std::size_t pending() const override;
    [[nodiscard]] std::size_t searches() const override;
    [[nodiscard]] double worstSearchSeconds() const override;

private:
    PursuitRouter* router;
    std::size_t budget;
    std::vector<PendingRoute> queue{};
    std::size_t answered = 0;
    double worstSeconds = 0.0;
};

// The router on its own thread. Takes the router (built and counted by the caller) and searches its
// queue one request at a time, oldest first, until it is destroyed; `request` and `collect` take a
// lock for the length of a copy and never wait on a search.
export class PursuitRouteWorker final : public RouteService
{
public:
    explicit PursuitRouteWorker(PursuitRouter router);
    ~PursuitRouteWorker() override;
    PursuitRouteWorker(const PursuitRouteWorker&) = delete;
    PursuitRouteWorker(PursuitRouteWorker&&) = delete;
    PursuitRouteWorker& operator=(const PursuitRouteWorker&) = delete;
    PursuitRouteWorker& operator=(PursuitRouteWorker&&) = delete;

    void request(std::uint32_t agent, std::uint64_t serial, const RouteRequest& request) override;
    void collect(std::vector<RouteAnswer>& into) override;
    [[nodiscard]] std::size_t pending() const override;
    [[nodiscard]] std::size_t searches() const override;
    [[nodiscard]] double worstSearchSeconds() const override;

private:
    // The thread, the lock and the queues, behind a pointer so this interface names no thread header.
    struct State;
    std::unique_ptr<State> state;
};

} // namespace raceengine
