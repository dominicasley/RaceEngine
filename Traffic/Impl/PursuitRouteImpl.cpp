// Pursuit router bodies. Declarations are in Api/PursuitRoute.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export` — so a cost, a
// radius or a search detail the seat moves rebuilds one object file and no importer.
module;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.traffic;

namespace raceengine
{

namespace
{

constexpr auto worldUp = glm::dvec3(0.0, 1.0, 0.0);
constexpr auto noSample = std::uint32_t{0xffffffffu};

// How a node was reached, past the real edge indices.
constexpr auto viaStart = std::uint32_t{0xfffffff0u};
constexpr auto viaStartRoad = std::uint32_t{0xfffffff1u};
constexpr auto viaStartMesh = std::uint32_t{0xfffffff2u};
constexpr auto viaSampleToMesh = std::uint32_t{0xfffffff3u};
constexpr auto viaMeshToSample = std::uint32_t{0xfffffff4u};
constexpr auto viaGoal = std::uint32_t{0xfffffff5u};

[[nodiscard]] double planar(const glm::dvec3& a, const glm::dvec3& b)
{
    return std::hypot(a.x - b.x, a.z - b.z);
}

[[nodiscard]] glm::dvec3 flat(const glm::dvec3& vector)
{
    return glm::dvec3(vector.x, 0.0, vector.z);
}

[[nodiscard]] glm::dvec3 leftOf(const glm::dvec3& forward)
{
    const auto side = glm::cross(worldUp, forward);
    const auto length = glm::length(side);

    return length > 1e-12 ? side / length : glm::dvec3(1.0, 0.0, 0.0);
}

// Twice the signed area of the triangle a, b, c in plan — Recast's `triarea2`, positive when `c` is
// on the side of a→b the funnel below calls the right. Only self-consistency matters: the portal
// sides are told apart with the same function, so the world's handedness never enters.
[[nodiscard]] double triarea2(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c)
{
    const auto abx = b.x - a.x;
    const auto abz = b.z - a.z;
    const auto acx = c.x - a.x;
    const auto acz = c.z - a.z;

    return acx * abz - abx * acz;
}

[[nodiscard]] bool samePlace(const glm::dvec3& a, const glm::dvec3& b)
{
    return planar(a, b) < 1e-6;
}

[[nodiscard]] glm::dvec3 clampIntoPolygon(const NavMesh& mesh, const std::uint32_t polygon, const glm::dvec3& point)
{
    const auto x = std::clamp(point.x, mesh.x0[polygon], mesh.x1[polygon]);
    const auto z = std::clamp(point.z, mesh.z0[polygon], mesh.z1[polygon]);

    return glm::dvec3(x, navMeshHeight(mesh, polygon, x, z), z);
}

// The road and its marks at their hints; anything else — a kerb, the grass — at its hint times the
// off-road factor (`PursuitRouterOptions::offRoadCostFactor`).
[[nodiscard]] double hintOf(const NavMesh& mesh, const std::uint32_t polygon, const double offRoadCostFactor)
{
    const auto hint = mesh.areas[mesh.area[polygon]].costHint;

    return hint > 1.5 ? hint * offRoadCostFactor : hint;
}

[[nodiscard]] double seconds(const std::chrono::steady_clock::time_point from,
                             const std::chrono::steady_clock::time_point to)
{
    return std::chrono::duration<double>(to - from).count();
}

} // namespace

// --- the straight line over the mesh ---------------------------------------------------------------

NavMeshWalk walkNavMesh(const NavMesh& mesh, const std::uint32_t fromPolygon, const glm::dvec3& from,
                        const glm::dvec3& to, const double offRoadCostFactor)
{
    auto walk = NavMeshWalk{};

    if (fromPolygon >= mesh.polygonCount())
    {
        return walk;
    }

    auto polygon = fromPolygon;
    auto at = from;
    const auto step = flat(to - from);
    const auto total = glm::length(step);

    if (total < 1e-9)
    {
        walk.reached = true;
        walk.lastPolygon = polygon;

        return walk;
    }

    const auto direction = step / total;
    constexpr auto tolerance = 1e-6;
    auto stalled = 0;
    auto previous = noPolygon;

    for (auto iteration = 0; iteration < 100000; iteration++)
    {
        const auto x0 = mesh.x0[polygon];
        const auto x1 = mesh.x1[polygon];
        const auto z0 = mesh.z0[polygon];
        const auto z1 = mesh.z1[polygon];

        if (to.x >= x0 - tolerance && to.x <= x1 + tolerance && to.z >= z0 - tolerance && to.z <= z1 + tolerance)
        {
            const auto rest = planar(at, to);
            walk.costMetres += rest * hintOf(mesh, polygon, offRoadCostFactor);
            walk.lengthMetres += rest;
            walk.reached = true;
            walk.lastPolygon = polygon;

            return walk;
        }

        // Where the line leaves this rectangle.
        auto exitAt = std::numeric_limits<double>::max();

        if (direction.x > 1e-12)
        {
            exitAt = std::min(exitAt, (x1 - at.x) / direction.x);
        }
        else if (direction.x < -1e-12)
        {
            exitAt = std::min(exitAt, (x0 - at.x) / direction.x);
        }

        if (direction.z > 1e-12)
        {
            exitAt = std::min(exitAt, (z1 - at.z) / direction.z);
        }
        else if (direction.z < -1e-12)
        {
            exitAt = std::min(exitAt, (z0 - at.z) / direction.z);
        }

        if (exitAt == std::numeric_limits<double>::max())
        {
            walk.lastPolygon = polygon;

            return walk;
        }

        exitAt = std::max(exitAt, 0.0);
        stalled = exitAt < 1e-9 ? stalled + 1 : 0;

        if (stalled > 4)
        {
            // Bouncing on a corner between two polygons without moving: not a way through.
            walk.lastPolygon = polygon;

            return walk;
        }

        const auto exitPoint = at + direction * exitAt;
        walk.costMetres += exitAt * hintOf(mesh, polygon, offRoadCostFactor);
        walk.lengthMetres += exitAt;

        // The portal that carries the exit point. At a corner the point lies on two portals, and the
        // first match can lead straight back into the polygon just left — so the neighbour that holds
        // the point a millimetre further along is taken first, any other match second, and the
        // polygon just left only when nothing else will do.
        const auto probe = at + direction * (exitAt + 1e-3);
        auto next = noPolygon;
        auto fallback = noPolygon;
        auto retreat = noPolygon;

        for (auto entry = mesh.neighbourStarts[polygon]; entry < mesh.neighbourStarts[polygon + 1] && next == noPolygon;
             entry++)
        {
            const auto link = mesh.neighbourLinks[entry];
            const auto& p = mesh.portalP[link];
            const auto& q = mesh.portalQ[link];

            const auto onLine = std::abs(p.x - q.x) < tolerance ? std::abs(exitPoint.x - p.x) < tolerance &&
                                                                      exitPoint.z >= std::min(p.z, q.z) - tolerance &&
                                                                      exitPoint.z <= std::max(p.z, q.z) + tolerance
                                                                : std::abs(exitPoint.z - p.z) < tolerance &&
                                                                      exitPoint.x >= std::min(p.x, q.x) - tolerance &&
                                                                      exitPoint.x <= std::max(p.x, q.x) + tolerance;

            if (!onLine)
            {
                continue;
            }

            const auto candidate = navMeshNeighbour(mesh, polygon, link);

            if (candidate == previous)
            {
                retreat = candidate;

                continue;
            }

            const auto holdsProbe =
                probe.x >= mesh.x0[candidate] - tolerance && probe.x <= mesh.x1[candidate] + tolerance &&
                probe.z >= mesh.z0[candidate] - tolerance && probe.z <= mesh.z1[candidate] + tolerance;

            if (holdsProbe)
            {
                next = candidate;
            }
            else if (fallback == noPolygon)
            {
                fallback = candidate;
            }
        }

        if (next == noPolygon)
        {
            next = fallback != noPolygon ? fallback : retreat;
        }

        if (next == noPolygon)
        {
            walk.lastPolygon = polygon;

            return walk;
        }

        previous = polygon;
        polygon = next;
        at = glm::dvec3(exitPoint.x, navMeshHeight(mesh, polygon, exitPoint.x, exitPoint.z), exitPoint.z);
    }

    walk.lastPolygon = polygon;

    return walk;
}

// --- the string pull -------------------------------------------------------------------------------

std::vector<glm::dvec3> pullThroughPortals(const NavMesh& mesh, const glm::dvec3& from,
                                           const std::span<const std::uint32_t> links, const glm::dvec3& to)
{
    const auto count = links.size();
    auto left = std::vector<glm::dvec3>(count + 2);
    auto right = std::vector<glm::dvec3>(count + 2);

    left[0] = from;
    right[0] = from;
    left[count + 1] = to;
    right[count + 1] = to;

    // Each portal's two sides, told apart from the direction of travel through it: from the last
    // portal's midpoint (the start, for the first) to this one's, or on to the next where the two
    // coincide.
    auto reference = from;

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto link = links[index];
        const auto& p = mesh.portalP[link];
        const auto& q = mesh.portalQ[link];
        const auto middle = 0.5 * (p + q);

        auto behind = reference;

        if (planar(behind, middle) < 1e-6)
        {
            const auto ahead =
                index + 1 < count ? 0.5 * (mesh.portalP[links[index + 1]] + mesh.portalQ[links[index + 1]]) : to;
            behind = middle - (ahead - middle);
        }

        if (triarea2(behind, middle, p) > 0.0)
        {
            right[index + 1] = p;
            left[index + 1] = q;
        }
        else
        {
            right[index + 1] = q;
            left[index + 1] = p;
        }

        reference = middle;
    }

    // The simple stupid funnel.
    auto path = std::vector<glm::dvec3>();
    path.push_back(from);

    auto apex = from;
    auto portalLeft = from;
    auto portalRight = from;
    auto apexIndex = std::size_t{0};
    auto leftIndex = std::size_t{0};
    auto rightIndex = std::size_t{0};

    for (auto index = std::size_t{1}; index < count + 2; index++)
    {
        const auto& nextLeft = left[index];
        const auto& nextRight = right[index];

        if (triarea2(apex, portalRight, nextRight) <= 0.0)
        {
            if (samePlace(apex, portalRight) || triarea2(apex, portalLeft, nextRight) > 0.0)
            {
                portalRight = nextRight;
                rightIndex = index;
            }
            else
            {
                path.push_back(portalLeft);
                apex = portalLeft;
                apexIndex = leftIndex;
                portalLeft = apex;
                portalRight = apex;
                leftIndex = apexIndex;
                rightIndex = apexIndex;
                index = apexIndex;

                continue;
            }
        }

        if (triarea2(apex, portalLeft, nextLeft) >= 0.0)
        {
            if (samePlace(apex, portalLeft) || triarea2(apex, portalRight, nextLeft) < 0.0)
            {
                portalLeft = nextLeft;
                leftIndex = index;
            }
            else
            {
                path.push_back(portalRight);
                apex = portalRight;
                apexIndex = rightIndex;
                portalLeft = apex;
                portalRight = apex;
                leftIndex = apexIndex;
                rightIndex = apexIndex;
                index = apexIndex;

                continue;
            }
        }
    }

    if (!samePlace(path.back(), to))
    {
        path.push_back(to);
    }

    return path;
}

// --- the route polyline ----------------------------------------------------------------------------

RoutePlace projectOntoRoute(const PursuitRoute& route, const glm::dvec3& pointMetres, const std::size_t hintSegment,
                            const std::size_t window)
{
    auto place = RoutePlace{};
    const auto& points = route.points;

    if (points.empty())
    {
        return place;
    }

    if (points.size() == 1)
    {
        place.positionMetres = points[0].positionMetres;
        place.offsetMetres = planar(pointMetres, points[0].positionMetres);

        return place;
    }

    const auto segments = points.size() - 1;
    auto first = std::size_t{0};
    auto last = segments;

    if (window > 0)
    {
        first = hintSegment > window ? hintSegment - window : 0;
        last = std::min(segments, hintSegment + window + 1);
    }

    auto best = std::numeric_limits<double>::max();

    for (auto segment = first; segment < last; segment++)
    {
        const auto& a = points[segment].positionMetres;
        const auto& b = points[segment + 1].positionMetres;
        const auto ab = flat(b - a);
        const auto length2 = glm::dot(ab, ab);
        const auto along = length2 > 1e-12 ? std::clamp(glm::dot(flat(pointMetres - a), ab) / length2, 0.0, 1.0) : 0.0;
        const auto foot = a + (b - a) * along;
        const auto offset = planar(pointMetres, foot);

        if (offset < best)
        {
            best = offset;
            place.segment = segment;
            place.positionMetres = foot;
            place.offsetMetres = offset;
            place.distanceMetres = points[segment].distanceMetres +
                                   (points[segment + 1].distanceMetres - points[segment].distanceMetres) * along;
        }
    }

    return place;
}

glm::dvec3 routePointAt(const PursuitRoute& route, const double distanceMetres)
{
    const auto& points = route.points;

    if (points.empty())
    {
        return glm::dvec3(0.0);
    }

    if (distanceMetres <= points.front().distanceMetres)
    {
        return points.front().positionMetres;
    }

    if (distanceMetres >= points.back().distanceMetres)
    {
        return points.back().positionMetres;
    }

    const auto after = std::ranges::upper_bound(points, distanceMetres, {}, &RoutePoint::distanceMetres);
    const auto index = static_cast<std::size_t>(after - points.begin());
    const auto& a = points[index - 1];
    const auto& b = points[index];
    const auto span = b.distanceMetres - a.distanceMetres;
    const auto along = span > 1e-12 ? (distanceMetres - a.distanceMetres) / span : 0.0;

    return a.positionMetres + (b.positionMetres - a.positionMetres) * along;
}

// --- the router ------------------------------------------------------------------------------------

PursuitRouter::PursuitRouter(const LaneNetwork& laneNetwork, const NavMesh* ground, PursuitRouterOptions settings) :
    network(&laneNetwork),
    mesh(ground != nullptr && ground->polygonCount() > 0 ? ground : nullptr),
    options(settings)
{
    buildLaneLayer();
    buildMeshLinks();
}

bool PursuitRouter::hasGround() const
{
    return mesh != nullptr;
}

const PursuitRouterOptions& PursuitRouter::settings() const
{
    return options;
}

std::size_t PursuitRouter::lanePlaceCount() const
{
    return network->samples.size();
}

std::size_t PursuitRouter::lanePlacesOnMesh() const
{
    return onMesh;
}

std::size_t PursuitRouter::laneEdgeCount() const
{
    return laneEdges.size();
}

// The sample of a lane nearest a distance along it.
std::uint32_t PursuitRouter::nearestSample(const std::size_t lane, const double distanceMetres) const
{
    if (lane >= laneSampleCount.size() || laneSampleCount[lane] == 0)
    {
        return noSample;
    }

    const auto first = laneFirstSample[lane];
    const auto count = laneSampleCount[lane];
    const auto& samples = network->samples;

    // Samples are stepped at the spacing from zero, so the index is the arithmetic; the search is the
    // guard against a last sample that stands short of the step.
    const auto spacing = std::max(1e-6, samples.size() > first + 1 && count > 1
                                            ? samples[first + 1].distanceMetres - samples[first].distanceMetres
                                            : options.sampleSpacingMetres);
    auto index = static_cast<std::uint32_t>(
        std::clamp(std::llround(distanceMetres / spacing), 0LL, static_cast<long long>(count) - 1));

    while (index + 1 < count && samples[first + index + 1].distanceMetres < distanceMetres &&
           std::abs(samples[first + index + 1].distanceMetres - distanceMetres) <
               std::abs(samples[first + index].distanceMetres - distanceMetres))
    {
        index++;
    }

    while (index > 0 && std::abs(samples[first + index - 1].distanceMetres - distanceMetres) <
                            std::abs(samples[first + index].distanceMetres - distanceMetres))
    {
        index--;
    }

    return first + index;
}

void PursuitRouter::buildLaneLayer()
{
    const auto& lanes = network->lanes;
    const auto& samples = network->samples;

    laneFirstSample.assign(lanes.size(), 0);
    laneSampleCount.assign(lanes.size(), 0);

    // The network lays every lane's samples out contiguously and in order (Impl/LanesImpl.cpp).
    for (auto index = std::size_t{0}; index < samples.size(); index++)
    {
        const auto lane = samples[index].lane;

        if (lane >= lanes.size())
        {
            continue;
        }

        if (laneSampleCount[lane] == 0)
        {
            laneFirstSample[lane] = static_cast<std::uint32_t>(index);
        }

        laneSampleCount[lane]++;
    }

    const auto wrongWay = options.wrongWayCostFactor;
    auto pending = std::vector<std::pair<std::uint32_t, LaneEdge>>();

    for (auto laneIndex = std::size_t{0}; laneIndex < lanes.size(); laneIndex++)
    {
        const auto& lane = lanes[laneIndex];
        const auto first = laneFirstSample[laneIndex];
        const auto count = laneSampleCount[laneIndex];

        if (count == 0)
        {
            continue;
        }

        const auto last = first + count - 1;
        const auto length = laneLength(lane);

        // Along the lane, both ways.
        for (auto offset = std::uint32_t{0}; offset + 1 < count; offset++)
        {
            const auto here = first + offset;
            const auto step = samples[here + 1].distanceMetres - samples[here].distanceMetres;

            if (step <= 0.0)
            {
                continue;
            }

            pending.emplace_back(
                here, LaneEdge{.to = here + 1, .costMetres = step, .alongMetres = step, .kind = LaneEdgeKind::Along});
            pending.emplace_back(here + 1, LaneEdge{.to = here,
                                                    .costMetres = step * wrongWay,
                                                    .alongMetres = -step,
                                                    .kind = LaneEdgeKind::AlongBack});
        }

        // Round a loop.
        if (lane.loop && count >= 2)
        {
            const auto remaining = std::max(length - samples[last].distanceMetres, 0.01);

            pending.emplace_back(
                last,
                LaneEdge{.to = first, .costMetres = remaining, .alongMetres = remaining, .kind = LaneEdgeKind::Wrap});
            pending.emplace_back(first, LaneEdge{.to = last,
                                                 .costMetres = remaining * wrongWay,
                                                 .alongMetres = -remaining,
                                                 .kind = LaneEdgeKind::WrapBack});
        }

        // Onto a successor, from the end.
        for (const auto& successor : lane.successors)
        {
            if (successor.lane >= lanes.size() || laneSampleCount[successor.lane] == 0)
            {
                continue;
            }

            const auto landing = wrapDistance(lanes[successor.lane], successor.distanceMetres);
            const auto target = nearestSample(successor.lane, landing);

            if (target == noSample)
            {
                continue;
            }

            const auto metres = std::max(0.01, std::max(length - samples[last].distanceMetres, 0.0) +
                                                   std::abs(landing - samples[target].distanceMetres));

            pending.emplace_back(
                last, LaneEdge{.to = target, .costMetres = metres, .alongMetres = 0.0, .kind = LaneEdgeKind::Join});
            pending.emplace_back(target, LaneEdge{.to = last,
                                                  .costMetres = metres * wrongWay,
                                                  .alongMetres = 0.0,
                                                  .kind = LaneEdgeKind::JoinBack});
        }

        // Across to a neighbour, anywhere inside the run: the same place on the neighbour, a change's
        // length further on.
        for (const auto& run : lane.neighbours)
        {
            if (run.lane >= lanes.size() || laneSampleCount[run.lane] == 0)
            {
                continue;
            }

            for (auto offset = std::uint32_t{0}; offset < count; offset++)
            {
                const auto here = first + offset;
                const auto distance = samples[here].distanceMetres;

                if (distance < run.fromMetres || distance > run.toMetres)
                {
                    continue;
                }

                const auto shifted = wrapDistance(lanes[run.lane], distance + neighbourShiftAt(run, distance) +
                                                                       options.laneChangeLengthMetres);
                const auto target = nearestSample(run.lane, shifted);

                if (target == noSample || target == here)
                {
                    continue;
                }

                pending.emplace_back(
                    here, LaneEdge{.to = target,
                                   .costMetres = options.laneChangeLengthMetres + options.laneChangeCostMetres,
                                   .alongMetres = 0.0,
                                   .kind = LaneEdgeKind::Change});
            }
        }
    }

    std::ranges::stable_sort(pending, [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    laneEdgeStarts.assign(samples.size() + 1, 0);
    laneEdges.clear();
    laneEdges.reserve(pending.size());

    for (const auto& [from, edge] : pending)
    {
        laneEdgeStarts[from + 1]++;
        laneEdges.push_back(edge);
    }

    for (auto index = std::size_t{0}; index < samples.size(); index++)
    {
        laneEdgeStarts[index + 1] += laneEdgeStarts[index];
    }
}

void PursuitRouter::buildMeshLinks()
{
    const auto& samples = network->samples;

    samplePolygon.assign(samples.size(), noPolygon);
    onMesh = 0;

    if (mesh == nullptr)
    {
        polygonSampleStarts.assign(1, 0);
        polygonSamples.clear();

        return;
    }

    const auto polygons = mesh->polygonCount();
    polygonSampleStarts.assign(polygons + 1, 0);

    for (auto index = std::size_t{0}; index < samples.size(); index++)
    {
        const auto place = locateOnNavMesh(*mesh, samples[index].positionMetres, options.lanePlaceReachMetres);

        if (place.found())
        {
            samplePolygon[index] = place.polygon;
            polygonSampleStarts[place.polygon + 1]++;
            onMesh++;
        }
    }

    for (auto index = std::size_t{0}; index < polygons; index++)
    {
        polygonSampleStarts[index + 1] += polygonSampleStarts[index];
    }

    polygonSamples.assign(polygonSampleStarts.back(), 0);

    auto fill = std::vector<std::uint32_t>(polygonSampleStarts.begin(), polygonSampleStarts.end() - 1);

    for (auto index = std::size_t{0}; index < samples.size(); index++)
    {
        if (samplePolygon[index] != noPolygon)
        {
            polygonSamples[fill[samplePolygon[index]]++] = static_cast<std::uint32_t>(index);
        }
    }
}

// Where a route's end joins the graph: its lane, either direction, and its polygon.
PursuitRouter::EndHook PursuitRouter::hook(const glm::dvec3& pointMetres) const
{
    auto end = EndHook{};
    const auto& lanes = network->lanes;
    const auto& samples = network->samples;

    const auto projection = projectOntoNetwork(*network, pointMetres, glm::dvec3(0.0), options.projectionLateralMetres);

    if (projection.found && projection.lane < lanes.size() && laneSampleCount[projection.lane] > 0)
    {
        const auto& lane = lanes[projection.lane];
        const auto length = laneLength(lane);
        const auto distance = lane.loop ? wrapDistance(lane, projection.distanceMetres)
                                        : std::clamp(projection.distanceMetres, 0.0, length);

        const auto first = laneFirstSample[projection.lane];
        const auto count = laneSampleCount[projection.lane];
        const auto last = first + count - 1;

        end.onRoad = true;
        end.lane = projection.lane;
        end.distanceMetres = distance;

        // The first sample at or past the place, and the last one before it.
        auto ahead = first;
        while (ahead <= last && samples[ahead].distanceMetres < distance)
        {
            ahead++;
        }

        if (ahead <= last)
        {
            end.sampleAhead = ahead;
            end.metresAhead = samples[ahead].distanceMetres - distance;
        }
        else if (lane.loop)
        {
            end.sampleAhead = first;
            end.metresAhead = std::max(length - distance, 0.0) + samples[first].distanceMetres;
        }

        if (ahead > first)
        {
            end.sampleBehind = ahead - 1;
            end.metresBehind = distance - samples[ahead - 1].distanceMetres;
        }
        else if (lane.loop && count >= 2)
        {
            end.sampleBehind = last;
            end.metresBehind = distance + std::max(length - samples[last].distanceMetres, 0.0);
        }
    }

    if (mesh != nullptr)
    {
        const auto place = locateOnNavMesh(*mesh, pointMetres, options.endpointReachMetres);

        if (place.found())
        {
            end.polygon = place.polygon;
            end.polygonPoint = clampIntoPolygon(*mesh, place.polygon, pointMetres);
            end.polygonCost = place.planarMetres * hintOf(*mesh, place.polygon, options.offRoadCostFactor);
        }
    }

    return end;
}

void PursuitRouter::relax(const std::uint32_t from, const std::uint32_t to, const double newCost,
                          const glm::dvec3& arrivedAt, const std::uint32_t edge, const glm::dvec3& goal)
{
    if (seen[to] == stamp && (closed[to] == stamp || newCost >= cost[to]))
    {
        return;
    }

    seen[to] = stamp;
    cost[to] = newCost;
    arrival[to] = arrivedAt;
    via[to] = Via{.parent = from, .edge = edge};

    heap.push_back(HeapEntry{.estimate = newCost + planar(arrivedAt, goal), .cost = newCost, .node = to});
    std::ranges::push_heap(heap,
                           [](const HeapEntry& lhs, const HeapEntry& rhs) { return lhs.estimate > rhs.estimate; });
}

PursuitRoute PursuitRouter::route(const RouteRequest& request)
{
    RACEENGINE_ZONE_N("pursuit route search");

    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();

    auto answer = PursuitRoute{};

    const auto& samples = network->samples;
    const auto sampleCount = static_cast<std::uint32_t>(samples.size());
    const auto polygonCount = mesh != nullptr ? static_cast<std::uint32_t>(mesh->polygonCount()) : 0u;
    const auto startNode = sampleCount + polygonCount;
    const auto goalNode = startNode + 1;
    const auto nodes = static_cast<std::size_t>(goalNode) + 1;

    if (seen.size() != nodes)
    {
        seen.assign(nodes, 0);
        closed.assign(nodes, 0);
        cost.assign(nodes, 0.0);
        arrival.assign(nodes, glm::dvec3(0.0));
        via.assign(nodes, Via{});
        stamp = 0;
    }

    stamp++;

    if (stamp == 0)
    {
        std::ranges::fill(seen, 0u);
        std::ranges::fill(closed, 0u);
        stamp = 1;
    }

    heap.clear();

    const auto start = hook(request.fromMetres);
    const auto goal = hook(request.toMetres);
    const auto& to = request.toMetres;
    const auto wrongWay = options.wrongWayCostFactor;

    seen[startNode] = stamp;
    cost[startNode] = 0.0;
    arrival[startNode] = request.fromMetres;
    via[startNode] = Via{.parent = startNode, .edge = viaStart};
    heap.push_back(HeapEntry{.estimate = planar(request.fromMetres, to), .cost = 0.0, .node = startNode});

    const auto byEstimate = [](const HeapEntry& lhs, const HeapEntry& rhs)
    {
        return lhs.estimate > rhs.estimate;
    };

    auto reached = false;
    auto expanded = std::size_t{0};

    while (!heap.empty())
    {
        std::ranges::pop_heap(heap, byEstimate);
        const auto entry = heap.back();
        heap.pop_back();

        const auto node = entry.node;

        if (closed[node] == stamp || entry.cost > cost[node] + 1e-9)
        {
            continue;
        }

        closed[node] = stamp;
        expanded++;

        if (node == goalNode)
        {
            reached = true;

            break;
        }

        if (expanded > options.maximumExpansions)
        {
            break;
        }

        const auto here = arrival[node];
        const auto soFar = cost[node];

        if (node == startNode)
        {
            if (start.sampleAhead != noSample)
            {
                relax(node, start.sampleAhead, soFar + start.metresAhead, samples[start.sampleAhead].positionMetres,
                      viaStartRoad, to);
            }

            if (start.sampleBehind != noSample)
            {
                relax(node, start.sampleBehind, soFar + start.metresBehind * wrongWay,
                      samples[start.sampleBehind].positionMetres, viaStartRoad, to);
            }

            if (start.polygon != noPolygon)
            {
                // A unit on the road leaves it to reach the mesh, like any lane place.
                const auto leaving = start.onRoad ? options.groundEntryCostMetres : 0.0;

                relax(node, sampleCount + start.polygon, soFar + start.polygonCost + leaving,
                      start.polygonCost > 0.0 ? start.polygonPoint : request.fromMetres, viaStartMesh, to);
            }

            continue;
        }

        if (node < sampleCount)
        {
            for (auto index = laneEdgeStarts[node]; index < laneEdgeStarts[node + 1]; index++)
            {
                const auto& edge = laneEdges[index];
                relax(node, edge.to, soFar + edge.costMetres, samples[edge.to].positionMetres, index, to);
            }

            if (mesh != nullptr && samplePolygon[node] != noPolygon)
            {
                relax(node, sampleCount + samplePolygon[node], soFar + options.groundEntryCostMetres, here,
                      viaSampleToMesh, to);
            }

            if (node == goal.sampleBehind)
            {
                relax(node, goalNode, soFar + goal.metresBehind, to, viaGoal, to);
            }

            if (node == goal.sampleAhead)
            {
                relax(node, goalNode, soFar + goal.metresAhead * wrongWay, to, viaGoal, to);
            }

            continue;
        }

        // A polygon.
        const auto polygon = node - sampleCount;

        for (auto entryIndex = mesh->neighbourStarts[polygon]; entryIndex < mesh->neighbourStarts[polygon + 1];
             entryIndex++)
        {
            const auto link = mesh->neighbourLinks[entryIndex];
            const auto next = navMeshNeighbour(*mesh, polygon, link);
            const auto middle = 0.5 * (mesh->portalP[link] + mesh->portalQ[link]);

            relax(node, sampleCount + next, soFar + planar(here, middle) * hintOf(*mesh, next, options.offRoadCostFactor), middle, link, to);
        }

        for (auto entryIndex = polygonSampleStarts[polygon]; entryIndex < polygonSampleStarts[polygon + 1];
             entryIndex++)
        {
            const auto sample = polygonSamples[entryIndex];
            const auto& place = samples[sample].positionMetres;

            relax(node, sample, soFar + planar(here, place) * hintOf(*mesh, polygon, options.offRoadCostFactor), place, viaMeshToSample, to);
        }

        if (polygon == goal.polygon)
        {
            relax(node, goalNode, soFar + planar(here, goal.polygonPoint) * hintOf(*mesh, polygon, options.offRoadCostFactor) + goal.polygonCost,
                  to, viaGoal, to);
        }
    }

    answer.expanded = expanded;

    // The straight line, over the mesh: a candidate whenever both ends are on it, the route whenever
    // it is clear and cheaper than the graph's, and the last resort whenever the graph has nothing.
    auto walk = NavMeshWalk{};

    if (mesh != nullptr && start.polygon != noPolygon && goal.polygon != noPolygon)
    {
        walk = walkNavMesh(*mesh, start.polygon, start.polygonPoint, goal.polygonPoint, options.offRoadCostFactor);
    }

    const auto graphCost = reached ? cost[goalNode] : std::numeric_limits<double>::max();
    const auto directCost = walk.reached ? walk.costMetres + start.polygonCost + goal.polygonCost +
                                               (start.onRoad ? options.groundEntryCostMetres : 0.0)
                                         : std::numeric_limits<double>::max();

    if (walk.reached && directCost < graphCost)
    {
        directRoute(answer, request, directCost);
        answer.found = true;
    }
    else if (reached)
    {
        reconstruct(answer, start, goal, request);
        answer.found = true;
    }
    else
    {
        directRoute(answer, request, planar(request.fromMetres, to));
        answer.blocked = true;
    }

    answer.searchSeconds = seconds(started, Clock::now());

    return answer;
}

void PursuitRouter::directRoute(PursuitRoute& route, const RouteRequest& request, const double costMetres)
{
    route.direct = true;
    route.costMetres = costMetres;
    route.legs.clear();
    route.legs.push_back(RouteLeg{.kind = RouteLegKind::Direct, .costMetres = costMetres});

    vertices.clear();
    vertices.push_back(Vertex{.position = request.fromMetres, .ground = true, .leg = 0});
    vertices.push_back(Vertex{.position = request.toMetres, .ground = true, .leg = 0});

    assemble(route);
}

// The node chain back from the goal, turned into legs and the vertices of each.
void PursuitRouter::reconstruct(PursuitRoute& route, const EndHook& start, const EndHook& goal,
                                const RouteRequest& request)
{
    const auto& samples = network->samples;
    const auto& lanes = network->lanes;
    const auto sampleCount = static_cast<std::uint32_t>(samples.size());
    const auto polygonCount = mesh != nullptr ? static_cast<std::uint32_t>(mesh->polygonCount()) : 0u;
    const auto startNode = sampleCount + polygonCount;
    const auto goalNode = startNode + 1;

    pathNodes.clear();
    pathEdges.clear();

    for (auto node = goalNode; node != startNode; node = via[node].parent)
    {
        pathNodes.push_back(node);
        pathEdges.push_back(via[node].edge);

        if (pathNodes.size() > seen.size())
        {
            break;
        }
    }

    pathNodes.push_back(startNode);
    std::ranges::reverse(pathNodes);
    std::ranges::reverse(pathEdges);

    route.costMetres = cost[goalNode];
    route.legs.clear();
    vertices.clear();
    legLinks.clear();

    const auto isSample = [&](const std::uint32_t node)
    {
        return node < sampleCount;
    };
    const auto isPolygon = [&](const std::uint32_t node)
    {
        return node >= sampleCount && node < startNode;
    };

    // A road leg under construction.
    auto roadOpen = false;
    auto roadLane = std::size_t{0};
    auto roadFrom = 0.0;
    auto roadTo = 0.0;
    auto roadCost = 0.0;
    // A ground leg under construction: its entry point and the portals so far.
    auto groundOpen = false;
    auto groundEntry = glm::dvec3(0.0);
    auto groundCost = 0.0;
    // The last vertex a leg ended at, so a junction is one vertex and not two.
    auto lastEnd = glm::dvec3(std::numeric_limits<double>::quiet_NaN());

    const auto pushVertex = [&](const glm::dvec3& position, const bool corner, const bool boundary, const bool ground,
                                const std::size_t leg)
    {
        if (!vertices.empty() && samePlace(vertices.back().position, position))
        {
            // A junction: the ground side's flags win, and it is a corner.
            auto& tail = vertices.back();
            tail.corner = tail.corner || corner || ground || tail.ground;
            tail.boundary = tail.boundary || boundary;
            tail.ground = tail.ground || ground;

            return;
        }

        vertices.push_back(
            Vertex{.position = position, .corner = corner, .boundary = boundary, .ground = ground, .leg = leg});
    };

    const auto closeRoad = [&]()
    {
        if (!roadOpen)
        {
            return;
        }

        roadOpen = false;

        const auto& lane = lanes[roadLane];
        const auto run = roadTo - roadFrom;

        if (std::abs(run) < 1e-6)
        {
            return;
        }

        const auto legIndex = route.legs.size();
        route.legs.push_back(RouteLeg{.kind = RouteLegKind::Road,
                                      .lane = roadLane,
                                      .fromMetres = roadFrom,
                                      .toMetres = roadTo,
                                      .reversed = run < 0.0,
                                      .costMetres = roadCost});

        const auto steps = std::max(1, static_cast<int>(std::ceil(std::abs(run) / options.sampleSpacingMetres)));

        for (auto step = 0; step <= steps; step++)
        {
            const auto distance = roadFrom + run * static_cast<double>(step) / static_cast<double>(steps);
            const auto place = laneAt(lane, wrapDistance(lane, distance));

            if (place)
            {
                pushVertex(place->positionMetres, false, false, false, legIndex);
            }
        }

        lastEnd = vertices.empty() ? lastEnd : vertices.back().position;
    };

    const auto closeGround = [&](const glm::dvec3& exit)
    {
        if (!groundOpen)
        {
            return;
        }

        groundOpen = false;

        const auto legIndex = route.legs.size();
        route.legs.push_back(RouteLeg{.kind = RouteLegKind::Ground, .costMetres = groundCost});

        const auto pulled = pullThroughPortals(*mesh, groundEntry, legLinks, exit);

        for (auto index = std::size_t{0}; index < pulled.size(); index++)
        {
            const auto interior = index > 0 && index + 1 < pulled.size();
            pushVertex(pulled[index], interior, interior, true, legIndex);
        }

        legLinks.clear();
        lastEnd = vertices.empty() ? lastEnd : vertices.back().position;
    };

    for (auto step = std::size_t{0}; step + 1 < pathNodes.size(); step++)
    {
        const auto from = pathNodes[step];
        const auto node = pathNodes[step + 1];
        const auto edge = pathEdges[step];
        const auto stepCost = cost[node] - cost[from];

        if (from == startNode)
        {
            if (isSample(node))
            {
                roadOpen = true;
                roadLane = start.lane;
                roadFrom = start.distanceMetres;
                roadTo = roadFrom + (node == start.sampleAhead ? start.metresAhead : -start.metresBehind);
                roadCost = stepCost;
            }
            else if (isPolygon(node))
            {
                groundOpen = true;
                groundEntry = request.fromMetres;
                groundCost = stepCost;
                legLinks.clear();
            }

            continue;
        }

        if (node == goalNode)
        {
            if (isSample(from))
            {
                roadTo += from == goal.sampleBehind ? goal.metresBehind : -goal.metresAhead;
                roadCost += stepCost;
                closeRoad();
            }
            else if (isPolygon(from))
            {
                groundCost += stepCost;
                closeGround(request.toMetres);
            }

            continue;
        }

        if (isSample(from) && isSample(node))
        {
            const auto& laneEdge = laneEdges[edge];

            switch (laneEdge.kind)
            {
            case LaneEdgeKind::Along:
            case LaneEdgeKind::AlongBack:
            case LaneEdgeKind::Wrap:
            case LaneEdgeKind::WrapBack:
                roadTo += laneEdge.alongMetres;
                roadCost += stepCost;
                break;
            case LaneEdgeKind::Join:
            case LaneEdgeKind::JoinBack:
            case LaneEdgeKind::Change:
                closeRoad();
                roadOpen = true;
                roadLane = samples[node].lane;
                roadFrom = samples[node].distanceMetres;
                roadTo = roadFrom;
                roadCost = stepCost;
                // A change or a join is a straight from where the last leg ended to where this one
                // begins; the first sample of this leg draws it.
                if (laneEdge.kind == LaneEdgeKind::Change && !std::isnan(lastEnd.x))
                {
                    pushVertex(lastEnd, false, false, false, route.legs.size());
                }
                break;
            }

            continue;
        }

        if (isSample(from) && isPolygon(node))
        {
            closeRoad();
            groundOpen = true;
            groundEntry = samples[from].positionMetres;
            groundCost = stepCost;
            legLinks.clear();

            continue;
        }

        if (isPolygon(from) && isPolygon(node))
        {
            legLinks.push_back(edge);
            groundCost += stepCost;

            continue;
        }

        if (isPolygon(from) && isSample(node))
        {
            groundCost += stepCost;
            closeGround(samples[node].positionMetres);
            roadOpen = true;
            roadLane = samples[node].lane;
            roadFrom = samples[node].distanceMetres;
            roadTo = roadFrom;
            roadCost = 0.0;

            continue;
        }
    }

    closeRoad();
    closeGround(request.toMetres);

    // The route begins where the unit is and ends where the player is, whatever the graph's ends
    // snapped to.
    if (vertices.empty() || !samePlace(vertices.front().position, request.fromMetres))
    {
        vertices.insert(vertices.begin(), Vertex{.position = request.fromMetres,
                                                 .ground = false,
                                                 .leg = vertices.empty() ? 0 : vertices.front().leg});
    }

    if (!samePlace(vertices.back().position, request.toMetres))
    {
        vertices.push_back(
            Vertex{.position = request.toMetres, .ground = vertices.back().ground, .leg = vertices.back().leg});
    }

    assemble(route);
}

// The vertices into the route's points: the fillet at every corner, then the spacing, then the
// distances and the mesh's heights.
void PursuitRouter::assemble(PursuitRoute& route)
{
    route.points.clear();

    const auto count = vertices.size();

    if (count == 0)
    {
        for (auto& leg : route.legs)
        {
            leg.firstPoint = 0;
            leg.pointCount = 0;
        }

        return;
    }

    // Arc length along the raw vertices.
    auto along = std::vector<double>(count, 0.0);

    for (auto index = std::size_t{1}; index < count; index++)
    {
        along[index] = along[index - 1] + planar(vertices[index - 1].position, vertices[index].position);
    }

    const auto pointAt = [&](const double s) -> glm::dvec3
    {
        if (s <= along.front())
        {
            return vertices.front().position;
        }

        if (s >= along.back())
        {
            return vertices.back().position;
        }

        const auto after = std::ranges::upper_bound(along, s);
        const auto index = static_cast<std::size_t>(after - along.begin());
        const auto span = along[index] - along[index - 1];
        const auto fraction = span > 1e-12 ? (s - along[index - 1]) / span : 0.0;

        return vertices[index - 1].position + (vertices[index].position - vertices[index - 1].position) * fraction;
    };

    const auto directionAt = [&](const double s) -> glm::dvec3
    {
        auto index = static_cast<std::size_t>(std::ranges::upper_bound(along, s) - along.begin());
        index = std::clamp(index, std::size_t{1}, count - 1);

        while (index + 1 < count && along[index] - along[index - 1] < 1e-9)
        {
            index++;
        }

        const auto step = flat(vertices[index].position - vertices[index - 1].position);
        const auto length = glm::length(step);

        return length > 1e-12 ? step / length : glm::dvec3(0.0, 0.0, 1.0);
    };

    // The corners, and the arc length to the corner (or end) either side of each.
    auto corners = std::vector<std::size_t>();

    for (auto index = std::size_t{1}; index + 1 < count; index++)
    {
        if (vertices[index].corner)
        {
            corners.push_back(index);
        }
    }

    filleted.clear();
    auto nextOriginal = std::size_t{0};

    for (auto which = std::size_t{0}; which < corners.size(); which++)
    {
        const auto index = corners[which];
        const auto& corner = vertices[index];
        const auto previousAlong = which == 0 ? along.front() : along[corners[which - 1]];
        const auto nextAlong = which + 1 == corners.size() ? along.back() : along[corners[which + 1]];
        const auto legBefore = along[index] - previousAlong;
        const auto legAfter = nextAlong - along[index];
        const auto shorter = std::min(legBefore, legAfter);

        const auto incoming = directionAt(along[index] - 1e-6);
        const auto outgoing = directionAt(along[index] + 1e-6);
        const auto turn = std::acos(std::clamp(glm::dot(incoming, outgoing), -1.0, 1.0));

        auto radius = std::min(options.filletRadiusMetres, 0.5 * shorter);

        if (corner.boundary && turn > 1e-6)
        {
            // The arc goes no further inside the corner than the stated intrusion: r (1 − cos α/2).
            const auto sagitta = 1.0 - std::cos(0.5 * turn);
            radius = sagitta > 1e-9 ? std::min(radius, options.filletIntrusionMetres / sagitta) : radius;
        }

        if (turn < 0.02 || radius < 0.25 || shorter < 0.5)
        {
            continue;
        }

        auto tangent = radius * std::tan(0.5 * turn);

        if (tangent > 0.5 * shorter)
        {
            tangent = 0.5 * shorter;
            radius = tangent / std::tan(0.5 * turn);
        }

        const auto beginAlong = along[index] - tangent;
        const auto endAlong = along[index] + tangent;

        while (nextOriginal < count && along[nextOriginal] < beginAlong - 1e-9)
        {
            filleted.push_back(vertices[nextOriginal++]);
        }

        const auto begin = pointAt(beginAlong);
        const auto end = pointAt(endAlong);
        const auto directionIn = directionAt(beginAlong - 1e-6);
        const auto directionOut = directionAt(endAlong + 1e-6);
        const auto sweep = std::acos(std::clamp(glm::dot(directionIn, directionOut), -1.0, 1.0));

        filleted.push_back(Vertex{.position = begin, .ground = corner.ground, .leg = corner.leg});

        if (sweep > 1e-3)
        {
            const auto arcRadius = tangent / std::tan(0.5 * sweep);
            const auto left = leftOf(directionIn);
            const auto sign = glm::dot(directionOut, left) >= 0.0 ? 1.0 : -1.0;
            const auto centre = begin + left * (sign * arcRadius);
            const auto spoke = begin - centre;
            const auto segments =
                std::max(2, static_cast<int>(std::ceil(arcRadius * sweep / options.sampleSpacingMetres)));

            const auto rotate = [&](const double angle)
            {
                const auto c = std::cos(sign * angle);
                const auto s = std::sin(sign * angle);

                return centre + glm::dvec3(spoke.x * c + spoke.z * s, 0.0, -spoke.x * s + spoke.z * c);
            };

            // Where the arc lands against where the polyline's own trim point is: on straight sides
            // the two coincide, on a curving road side the difference is spread along the arc.
            const auto landing = rotate(sweep);
            const auto correction = flat(end - landing);

            for (auto step = 1; step < segments; step++)
            {
                const auto fraction = static_cast<double>(step) / static_cast<double>(segments);
                auto point = rotate(sweep * fraction) + correction * fraction;
                point.y = begin.y + (end.y - begin.y) * fraction;

                filleted.push_back(Vertex{.position = point, .ground = corner.ground, .leg = corner.leg});
            }
        }

        filleted.push_back(Vertex{.position = end, .ground = corner.ground, .leg = corner.leg});

        while (nextOriginal < count && along[nextOriginal] <= endAlong + 1e-9)
        {
            nextOriginal++;
        }
    }

    while (nextOriginal < count)
    {
        filleted.push_back(vertices[nextOriginal++]);
    }

    // The spacing, the distances, the legs' point ranges, and the ground's own heights.
    auto& points = route.points;
    auto distance = 0.0;
    auto legOfPoint = std::vector<std::size_t>();

    const auto emit = [&](const glm::dvec3& position, const bool ground, const std::size_t leg)
    {
        if (!points.empty())
        {
            const auto gap = glm::length(position - points.back().positionMetres);

            // Two fillets that meet, or a leg that ends where the next begins: one point, not two.
            if (gap < 1e-6)
            {
                points.back().ground = points.back().ground || ground;

                return;
            }

            distance += gap;
        }

        points.push_back(RoutePoint{.positionMetres = position, .distanceMetres = distance, .ground = ground});
        legOfPoint.push_back(leg);
    };

    for (auto index = std::size_t{0}; index < filleted.size(); index++)
    {
        const auto& vertex = filleted[index];

        if (index == 0)
        {
            emit(vertex.position, vertex.ground, vertex.leg);

            continue;
        }

        const auto& previous = filleted[index - 1];
        const auto gap = glm::length(vertex.position - previous.position);
        const auto steps = std::max(1, static_cast<int>(std::ceil(gap / options.sampleSpacingMetres)));

        for (auto step = 1; step < steps; step++)
        {
            const auto fraction = static_cast<double>(step) / static_cast<double>(steps);
            emit(previous.position + (vertex.position - previous.position) * fraction, previous.ground && vertex.ground,
                 vertex.leg);
        }

        emit(vertex.position, vertex.ground, vertex.leg);
    }

    if (mesh != nullptr)
    {
        for (auto& point : points)
        {
            if (!point.ground)
            {
                continue;
            }

            const auto place = locateOnNavMesh(*mesh, point.positionMetres, 2.0, 4.0);

            if (place.found())
            {
                point.positionMetres.y = place.heightMetres;
            }
        }
    }

    for (auto& leg : route.legs)
    {
        leg.firstPoint = 0;
        leg.pointCount = 0;
    }

    for (auto index = std::size_t{0}; index < points.size(); index++)
    {
        const auto leg = legOfPoint[index];

        if (leg >= route.legs.size())
        {
            continue;
        }

        if (route.legs[leg].pointCount == 0)
        {
            route.legs[leg].firstPoint = index;
        }

        route.legs[leg].pointCount = index + 1 - route.legs[leg].firstPoint;
    }

    route.lengthMetres = points.empty() ? 0.0 : points.back().distanceMetres;
}

// --- the route service ---------------------------------------------------------------------------

namespace
{

// Newest per unit: a request for a unit already waiting replaces it in place, so a queue is never
// longer than the roster and a unit is never answered twice for one ask.
void enqueue(std::vector<PendingRoute>& queue, const std::uint32_t agent, const std::uint64_t serial,
             const RouteRequest& request)
{
    for (auto& waiting : queue)
    {
        if (waiting.agent == agent)
        {
            waiting.serial = serial;
            waiting.request = request;

            return;
        }
    }

    queue.push_back(PendingRoute{.agent = agent, .serial = serial, .request = request});
}

} // namespace

InlineRouteService::InlineRouteService(PursuitRouter& way, const std::size_t searchesPerCollect) :
    router(&way),
    budget(std::max<std::size_t>(searchesPerCollect, 1))
{
}

void InlineRouteService::request(const std::uint32_t agent, const std::uint64_t serial, const RouteRequest& request)
{
    enqueue(queue, agent, serial, request);
}

void InlineRouteService::collect(std::vector<RouteAnswer>& into)
{
    const auto count = std::min(budget, queue.size());

    for (auto index = std::size_t{0}; index < count; index++)
    {
        const auto& waiting = queue[index];
        auto answer = RouteAnswer{.agent = waiting.agent, .serial = waiting.serial, .route = router->route(waiting.request)};

        answered++;
        worstSeconds = std::max(worstSeconds, answer.route.searchSeconds);
        into.push_back(std::move(answer));
    }

    queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(count));
}

std::size_t InlineRouteService::pending() const
{
    return queue.size();
}

std::size_t InlineRouteService::searches() const
{
    return answered;
}

double InlineRouteService::worstSearchSeconds() const
{
    return worstSeconds;
}

struct PursuitRouteWorker::State
{
    explicit State(PursuitRouter way) :
        router(std::move(way))
    {
    }

    PursuitRouter router;
    mutable std::mutex lock;
    std::condition_variable_any wake;
    std::vector<PendingRoute> queue;
    std::vector<RouteAnswer> finished;
    // A request the thread has taken off the queue and is searching: still pending to the caller.
    std::size_t searching = 0;
    std::size_t answered = 0;
    double worstSeconds = 0.0;
    // Last: it stops and joins before anything above is destroyed.
    std::jthread thread;

    void pump(const std::stop_token& stop)
    {
        RACEENGINE_THREAD("pursuit route worker");

        while (true)
        {
            auto waiting = PendingRoute{};

            {
                auto held = std::unique_lock<std::mutex>(lock);

                if (!wake.wait(held, stop, [this] { return !queue.empty(); }))
                {
                    return;
                }

                waiting = queue.front();
                queue.erase(queue.begin());
                searching++;
            }

            auto answer = RouteAnswer{.agent = waiting.agent, .serial = waiting.serial, .route = router.route(waiting.request)};

            {
                const auto guard = std::lock_guard<std::mutex>(lock);

                searching--;
                answered++;
                worstSeconds = std::max(worstSeconds, answer.route.searchSeconds);
                finished.push_back(std::move(answer));
            }
        }
    }
};

PursuitRouteWorker::PursuitRouteWorker(PursuitRouter router) :
    state(std::make_unique<State>(std::move(router)))
{
    state->thread = std::jthread([this](const std::stop_token& stop) { state->pump(stop); });
}

PursuitRouteWorker::~PursuitRouteWorker()
{
    if (state)
    {
        state->thread.request_stop();
        state->wake.notify_all();
        state->thread.join();
    }
}

void PursuitRouteWorker::request(const std::uint32_t agent, const std::uint64_t serial, const RouteRequest& request)
{
    RACEENGINE_ZONE_N("pursuit route request");

    {
        const auto guard = std::lock_guard<std::mutex>(state->lock);
        enqueue(state->queue, agent, serial, request);
    }

    state->wake.notify_one();
}

void PursuitRouteWorker::collect(std::vector<RouteAnswer>& into)
{
    RACEENGINE_ZONE_N("pursuit route collect");

    const auto guard = std::lock_guard<std::mutex>(state->lock);

    for (auto& answer : state->finished)
    {
        into.push_back(std::move(answer));
    }

    state->finished.clear();
}

std::size_t PursuitRouteWorker::pending() const
{
    const auto guard = std::lock_guard<std::mutex>(state->lock);

    return state->queue.size() + state->searching;
}

std::size_t PursuitRouteWorker::searches() const
{
    const auto guard = std::lock_guard<std::mutex>(state->lock);

    return state->answered;
}

double PursuitRouteWorker::worstSearchSeconds() const
{
    const auto guard = std::lock_guard<std::mutex>(state->lock);

    return state->worstSeconds;
}

} // namespace raceengine
