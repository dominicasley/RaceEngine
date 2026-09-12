// Pursuit director bodies. Declarations are in Api/Pursuit.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export` — so every rate
// and distance the seat moves rebuilds one object file and no importer.
module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.traffic;

import raceengine.physics;

namespace raceengine
{

namespace
{

constexpr auto worldUp = glm::dvec3(0.0, 1.0, 0.0);
constexpr auto standardGravity = 9.80665;

[[nodiscard]] glm::dvec3 flat(const glm::dvec3& vector)
{
    return glm::dvec3(vector.x, 0.0, vector.z);
}

[[nodiscard]] glm::dvec3 flatUnit(const glm::dvec3& vector, const glm::dvec3& fallback)
{
    const auto level = flat(vector);
    const auto length = glm::length(level);

    return length > 1e-6 ? level / length : fallback;
}

// Left of a heading, in this engine's frame: cross(up, forward) is +x, and +x is the car's left.
// The heading a full-lock turn has swung through when the leading front corner reaches a wall `room`
// to the turn's side (docs/police-driving-brief.md §14): the turning centre moves R (1 − cos θ) toward
// the wall, the corner sits half the body ahead and half the width out, so its reach is
// R + (w/2 − R) cos θ + (L/2) sin θ — rising to a peak past a quarter turn. Half a turn when the corner
// never gets there.
[[nodiscard]] double cornerSweepRadians(const double room, const double radius, const double length, const double width)
{
    constexpr auto pi = 3.14159265358979323846;

    const auto reach = [&](const double theta)
    { return radius + (0.5 * width - radius) * std::cos(theta) + 0.5 * length * std::sin(theta); };

    if (reach(0.0) >= room)
    {
        return 0.0;
    }

    // The reach peaks where its slope is zero: tan θ = −(L/2) / (R − w/2), past a quarter turn.
    const auto peak = pi - std::atan2(0.5 * length, std::max(radius - 0.5 * width, 1e-6));
    if (reach(peak) < room)
    {
        return pi;
    }

    auto low = 0.0;
    auto high = peak;
    for (auto step = 0; step < 40; step++)
    {
        const auto mid = 0.5 * (low + high);
        (reach(mid) < room ? low : high) = mid;
    }

    return low;
}

[[nodiscard]] glm::dvec3 leftOf(const glm::dvec3& forward)
{
    return glm::normalize(glm::cross(worldUp, forward));
}

[[nodiscard]] bool inChase(const AgentMode mode)
{
    return mode == AgentMode::Pursuing || mode == AgentMode::External;
}

[[nodiscard]] bool upright(const TrafficAgent& agent)
{
    return glm::dot(agent.orientation * worldUp, worldUp) > 0.55;
}

} // namespace

const char* offenceName(const Offence offence)
{
    switch (offence)
    {
    case Offence::None:
        return "none";
    case Offence::Speeding:
        return "speeding";
    case Offence::OffRoad:
        return "driving off the road";
    case Offence::WrongWay:
        return "driving the wrong way";
    case Offence::HitTraffic:
        return "hitting traffic";
    case Offence::HitPolice:
        return "hitting a police car";
    }

    return "none";
}

const char* pursuitRoleName(const PursuitRole role)
{
    switch (role)
    {
    case PursuitRole::Chase:
        return "chase";
    case PursuitRole::Tail:
        return "tail";
    case PursuitRole::Lead:
        return "lead";
    case PursuitRole::FlankLeft:
        return "flank left";
    case PursuitRole::FlankRight:
        return "flank right";
    }

    return "chase";
}

PursuitDirector::PursuitDirector(PursuitOptions settings) :
    options(settings)
{
}

std::span<const PursuitUnit> PursuitDirector::units() const
{
    return roster;
}

const PursuitUnit* PursuitDirector::unit(const std::uint32_t agent) const
{
    for (const auto& entry : roster)
    {
        if (entry.agent == agent)
        {
            return &entry;
        }
    }

    return nullptr;
}

void PursuitDirector::setFullModel(const std::uint32_t agent, const bool full)
{
    for (auto& entry : roster)
    {
        if (entry.agent == agent)
        {
            entry.fullModel = full;
        }
    }
}

void PursuitDirector::reportUnitImpact(const std::uint32_t agent, const double metresPerSecond)
{
    if (agent >= reportedImpacts.size())
    {
        reportedImpacts.resize(static_cast<std::size_t>(agent) + 1, 0.0);
    }

    reportedImpacts[agent] += std::max(0.0, metresPerSecond);
}

const PursuitStatus& PursuitDirector::status() const
{
    return summary;
}

const PursuitOptions& PursuitDirector::settings() const
{
    return options;
}

void PursuitDirector::setRouter(PursuitRouter* way)
{
    inlineService.reset();
    service = nullptr;

    if (way != nullptr)
    {
        inlineService.emplace(*way);
        service = &*inlineService;
    }
}

void PursuitDirector::setRouteService(RouteService* way)
{
    inlineService.reset();
    service = way;
}

void PursuitDirector::reset()
{
    roster.clear();
    sight.clear();
    reportedImpacts.clear();
    summary = PursuitStatus{};
    sightRemainder = 1.0e9;
    roleRemainder = 1.0e9;
    arrestRemainder = 0.0;
    report = Report{};
    search = Search{};
    answers.clear();
    // The serials are not reset: an answer still on the worker for a unit of the old chase must not
    // match a unit of the new one.
}

void PursuitDirector::setLevel(const int level)
{
    if (level < 1 || level > 5 || summary.busted || summary.wrecked)
    {
        return;
    }

    // The band a level reads as is [(level − 1) / 5, level / 5): the middle of it for one to four, so
    // the meter has a little to climb and a little to decay before it reads differently; the top of
    // the meter for five, which is the swarm.
    summary.felony = level == 5 ? options.swarmFelony : (static_cast<double>(level) - 1.0) / 5.0 + 0.1;
    summary.level = std::min(5, static_cast<int>(summary.felony * 5.0) + 1);
    summary.swarm = summary.felony >= options.swarmFelony;

    if (!summary.active)
    {
        summary.active = true;
        summary.unobservedSeconds = 0.0;
        summary.pursuitsStarted++;
        roleRemainder = 1.0e9;
        arrestRemainder = 0.0;
    }
}

// Who can see the player. Every patrol car within the sight radius that is looking the right way
// gets one ray, cast in a batch, and the answer is held until the next sight tick.
void PursuitDirector::watch(const PursuitPlayer& player, const TrafficPopulation& population, const PhysicsWorld& world)
{
    RACEENGINE_ZONE_N("pursuit sight rays");

    const auto agents = population.agents();

    if (sight.size() != agents.size())
    {
        sight.assign(agents.size(), Sight{});
    }

    rayOrigins.clear();
    rayDirections.clear();
    rayAgents.clear();
    rayDistances.clear();
    rayRoster.clear();

    const auto target = player.positionMetres + worldUp * options.targetHeightMetres;

    for (const auto id : population.policeIds())
    {
        if (id >= agents.size())
        {
            continue;
        }

        auto& seen = sight[id];
        seen.sighted = false;

        const auto& agent = agents[id];
        const auto eye = agent.positionMetres + worldUp * options.eyeHeightMetres;
        const auto to = target - eye;
        const auto distance = glm::length(to);

        if (distance > options.sightRadiusMetres || distance < 1e-3)
        {
            // Standing inside the patrol car is being seen.
            seen.sighted = distance < 1e-3;

            continue;
        }

        const auto direction = to / distance;

        // A cruising car looks through its cone; a car in the chase is looking for the player.
        if (!inChase(agent.mode) && glm::dot(flatUnit(direction, agent.heading), flatUnit(agent.heading, direction)) <
                                        options.sightConeCosine)
        {
            continue;
        }

        rayOrigins.push_back(eye);
        rayDirections.push_back(direction);
        rayAgents.push_back(id);
        rayDistances.push_back(distance);
    }

    // The line of drive, for every unit in the chase: from its origin to the player's, low — over a
    // kerb, under nothing a car fits under. What it answers is whether the unit can drive straight at
    // the player or has to route (docs/pursuit-navigation-brief.md, stage 2).
    const auto sightRays = rayOrigins.size();
    const auto driveTarget = player.positionMetres + worldUp * options.lineOfDriveHeightMetres;

    for (auto index = std::size_t{0}; index < roster.size(); index++)
    {
        const auto& entry = roster[index];

        if (entry.agent >= agents.size())
        {
            continue;
        }

        const auto origin = agents[entry.agent].positionMetres + worldUp * options.lineOfDriveHeightMetres;
        const auto to = driveTarget - origin;
        const auto distance = glm::length(to);

        if (distance < 1e-3)
        {
            roster[index].lineClear = true;

            continue;
        }

        rayOrigins.push_back(origin);
        rayDirections.push_back(to / distance);
        rayDistances.push_back(distance);
        rayRoster.push_back(index);
    }

    // The corridor (docs/police-driving-brief.md §2.1, §7): rays a unit at the same height, out to its
    // reach. A unit driving at a point casts three straight ones — the centre and the half width
    // either side — at its aim of the last tick (its heading before it has one). A unit on a route
    // casts the centre ray as a **chain along the route** — three legs, each to the route's point a
    // third of the reach further on — and the two side rays along the first leg, so a building on
    // the outside of a bend the route turns before is not an obstacle and a prop on the route is.
    // What they answer is whether the unit is driving at something, and how far along its path it
    // is. The traffic half of the corridor is read off the occupant lists on the same tick, below.
    const auto driveRays = rayOrigins.size();
    rayCorridor.clear();
    rayCorridorOffset.clear();

    for (auto index = std::size_t{0}; index < roster.size(); index++)
    {
        auto& entry = roster[index];

        if (entry.agent >= agents.size())
        {
            continue;
        }

        const auto& agent = agents[entry.agent];
        const auto heading = flatUnit(agent.heading, glm::dvec3(0.0, 0.0, 1.0));
        const auto origin = agent.positionMetres + worldUp * options.corridorHeightMetres;

        entry.corridorReachMetres = std::max(entry.corridorReachMetres, options.corridorMinimumMetres);
        const auto reach = entry.corridorReachMetres;

        const auto cast = [&](const glm::dvec3& from, const glm::dvec3& direction, const double length,
                              const double offset)
        {
            rayOrigins.push_back(from);
            rayDirections.push_back(direction);
            rayDistances.push_back(length);
            rayCorridor.push_back(index);
            rayCorridorOffset.push_back(offset);
        };

        const auto onRoute = entry.routing && entry.route.points.size() >= 2 && entry.turnPhase == 0;

        if (onRoute)
        {
            const auto place = projectOntoRoute(entry.route, agent.positionMetres, entry.routeSegment, 12);
            auto from = origin;
            auto along = place.distanceMetres;
            auto covered = 0.0;
            auto lastDirection = flatUnit(entry.aimMetres - agent.positionMetres, heading);

            for (auto leg = 0; leg < 3 && covered < reach; leg++)
            {
                along += reach / 3.0;
                const auto waypoint = routePointAt(entry.route, along);
                const auto to = glm::dvec3(waypoint.x - from.x, 0.0, waypoint.z - from.z);
                const auto length = std::min(glm::length(to), reach - covered);

                if (length < 0.5)
                {
                    break;
                }

                const auto direction = to / glm::length(to);
                cast(from, direction, length, covered);

                if (leg == 0)
                {
                    const auto left = leftOf(direction);
                    cast(from + left * options.corridorHalfWidthMetres, direction, length, 0.0);
                    cast(from - left * options.corridorHalfWidthMetres, direction, length, 0.0);
                }

                from += direction * length;
                covered += length;
                lastDirection = direction;
            }

            // The route ran out inside the reach — it ends at the player, or at a goal — and the
            // rest of the reach is cast straight on, the way the route was going: what is just past
            // the goal is what the unit drives at next.
            if (reach - covered > 0.5)
            {
                cast(from, lastDirection, reach - covered, covered);
            }
        }
        else
        {
            const auto toAim = flat(entry.aimMetres - agent.positionMetres);
            const auto direction = glm::length(toAim) > 1.0 ? toAim / glm::length(toAim) : heading;
            const auto left = leftOf(direction);

            for (const auto offset : std::array{0.0, options.corridorHalfWidthMetres, -options.corridorHalfWidthMetres})
            {
                cast(origin + left * offset, direction, reach, 0.0);
            }
        }

        readTraffic(entry, agent, population);
    }

    // The room to either side, for a turn round (docs/police-driving-brief.md §10): one ray each way
    // at the same height, out to the reach a full-lock arc needs.
    const auto roomRays = rayOrigins.size();
    rayRoom.clear();

    for (auto index = std::size_t{0}; index < roster.size(); index++)
    {
        const auto& entry = roster[index];

        if (entry.agent >= agents.size())
        {
            continue;
        }

        const auto& agent = agents[entry.agent];
        const auto heading = flatUnit(agent.heading, glm::dvec3(0.0, 0.0, 1.0));
        const auto left = leftOf(heading);
        const auto origin = agent.positionMetres + worldUp * options.corridorHeightMetres;

        for (const auto side : std::array{1.0, -1.0})
        {
            rayOrigins.push_back(origin);
            rayDirections.push_back(left * side);
            rayDistances.push_back(options.turnRoomReachMetres);
            rayRoom.push_back(index);
        }
    }

    if (rayOrigins.empty())
    {
        return;
    }

    auto reach = options.sightRadiusMetres;
    for (auto index = sightRays; index < rayDistances.size(); index++)
    {
        reach = std::max(reach, rayDistances[index] + 1.0);
    }

    world.castRays(rayOrigins, rayDirections, reach, rayHits);

    for (auto index = std::size_t{0}; index < sightRays && index < rayHits.size(); index++)
    {
        // The world in the way, short of the player's own car, is a wall. A hit further than the
        // player is the road behind them, or nothing.
        const auto blocked = rayHits[index].hit && rayHits[index].distance < rayDistances[index] - 3.0;

        sight[rayAgents[index]].sighted = !blocked;
    }

    for (auto index = sightRays; index < driveRays && index < rayHits.size(); index++)
    {
        // The world in the way short of the player — a wall, a median, a kerb's face — and not the
        // road on a rise, whose normal is mostly up.
        const auto blocked = rayHits[index].hit &&
                             rayHits[index].distance < rayDistances[index] - options.lineOfDriveClearanceMetres &&
                             rayHits[index].normal.y <= options.corridorGroundNormalY;

        roster[rayRoster[index - sightRays]].lineClear = !blocked;
    }

    // The corridor: the nearest hit inside the reach that is not the road on a rise, per unit; and
    // blocked when it is inside the route distance at the unit's speed.
    for (auto& entry : roster)
    {
        entry.corridorHit = false;
        entry.corridorHitMetres = entry.corridorReachMetres;
        entry.corridorBlocked = false;
    }

    for (auto index = roomRays; index < rayHits.size(); index++)
    {
        auto& entry = roster[rayRoom[index - roomRays]];
        const auto& hit = rayHits[index];
        const auto room = hit.hit && hit.distance <= options.turnRoomReachMetres && hit.normal.y <= options.corridorGroundNormalY
                              ? hit.distance
                              : options.turnRoomReachMetres;

        ((index - roomRays) % 2 == 0 ? entry.roomLeftMetres : entry.roomRightMetres) = room;
    }

    for (auto index = driveRays; index < roomRays && index < rayHits.size(); index++)
    {
        auto& entry = roster[rayCorridor[index - driveRays]];
        const auto& hit = rayHits[index];
        const auto alongPath = rayCorridorOffset[index - driveRays] + hit.distance;

        if (!hit.hit || hit.distance > rayDistances[index] || alongPath > entry.corridorReachMetres ||
            hit.normal.y > options.corridorGroundNormalY)
        {
            continue;
        }

        entry.corridorHit = true;
        entry.corridorHitMetres = std::min(entry.corridorHitMetres, alongPath);
    }

    for (auto& entry : roster)
    {
        if (entry.agent >= agents.size() || !entry.corridorHit)
        {
            continue;
        }

        const auto speed = glm::length(flat(agents[entry.agent].velocityMetresPerSecond));
        const auto routeReach =
            std::min(entry.corridorReachMetres, std::max(options.corridorMinimumMetres, options.corridorRouteSeconds * speed));

        entry.corridorBlocked = entry.corridorHitMetres < routeReach;
    }
}

double PursuitDirector::brakingFor(const PursuitUnit& unit) const
{
    return unit.fullModel ? options.fullModelBrakingMetresPerSecondSquared : options.cheapBodyBrakingMetresPerSecondSquared;
}

// The traffic half of the corridor, on the sight tick (docs/police-driving-brief.md §2.1, §7): the
// lines the unit could drive along off the occupant lists — its lane's centre, the neighbours'
// centres, the seams between — each with the car that blocks it. The unit drives for its own
// lane's centre unless another line is worth the shift: a chasing or routing unit takes the line
// whose safe speed beats its own lane's by the gain when the car behind on it is far enough back to
// merge in front of, holds it by name until it is on that lane's centre or its own lane is as good
// again, and never shifts while holding a station (the line is beside the player), reversing or
// turning round. A seam is how a unit threads through traffic that has pulled aside for it.
void PursuitDirector::readTraffic(PursuitUnit& unit, const TrafficAgent& agent, const TrafficPopulation& population)
{
    const auto heading = flatUnit(agent.heading, glm::dvec3(0.0, 0.0, 1.0));
    const auto speed = glm::length(flat(agent.velocityMetresPerSecond));
    const auto reach = std::max(unit.corridorReachMetres, options.corridorMinimumMetres);
    const auto corridor = population.trafficAround(agent.positionMetres, heading, options.corridorLateralMetres, reach);

    unit.trafficAhead = false;
    unit.trafficGapMetres = 0.0;
    unit.trafficSpeedMetresPerSecond = 0.0;

    if (!corridor.onRoad || corridor.lineCount == 0)
    {
        unit.laneShifting = false;
        unit.laneShiftMetres = 0.0;

        return;
    }

    const auto braking = options.corridorBrakeShare * brakingFor(unit);

    // The safe approach: the speed from which the unit can slow to the car's over the gap, less the
    // standstill, at its share of its braking. A car coming the other way has a negative speed and
    // the same formula.
    const auto safe = [&](const TrafficAhead& ahead)
    {
        if (!ahead.found)
        {
            return std::numeric_limits<double>::max();
        }

        return std::max(0.0, ahead.speedMetresPerSecond +
                                 std::sqrt(2.0 * braking *
                                           std::max(ahead.gapMetres - options.followStandstillMetres, 0.0)));
    };

    const auto lines = std::span<const TrafficLine>(corridor.lines.data(), corridor.lineCount);
    const auto& own = lines[0];
    const auto aligned = std::abs(glm::dot(heading, corridor.direction)) > options.laneShiftAlignment;
    const auto mayShift =
        (unit.role == PursuitRole::Chase || unit.routing) && !unit.reversing && unit.turnPhase == 0 && aligned;
    const TrafficLine* chosen = &own;

    // The line being held, found again by its name.
    const TrafficLine* held = nullptr;
    if (unit.laneShifting)
    {
        for (const auto& line : lines)
        {
            if (line.lane == unit.laneShiftLane && line.side == unit.laneShiftSide)
            {
                held = &line;
            }
        }
    }

    // The merge: the car behind on the line must be further back than what it closes over the
    // merge, plus the standstill.
    const auto followerClear = [&](const TrafficLine& line)
    {
        const auto closing = line.behind.found ? std::max(line.behind.speedMetresPerSecond - speed, 0.0) : 0.0;

        return !line.behind.found ||
               line.behind.gapMetres > options.mergeFollowerSeconds * closing + options.followStandstillMetres;
    };

    // A started shift is held until it is done — the unit within the done distance of the line —
    // or unsafe: the line worse than the one under the unit, or a car behind on it closing. Judged
    // every sight tick against the entry margin instead, it was dropped half a lane in when a car
    // appeared ahead on the line, and the aim snapped a lane's width back with the unit already
    // yawing for it (docs/police-driving-brief.md §13). Once the unit is nearer the line than any
    // other it is `own`, and the hold finishes the change onto its centre.
    if (held != nullptr && mayShift && std::abs(held->offsetMetres) > options.laneShiftDoneMetres &&
        (held == &own || (safe(held->ahead) >= safe(own.ahead) && followerClear(*held))))
    {
        chosen = held;
        unit.laneShiftMetres = held->offsetMetres;
    }
    else
    {
        unit.laneShifting = false;
        unit.laneShiftMetres = 0.0;

        if (mayShift && own.ahead.found)
        {
            auto best = safe(own.ahead);
            const TrafficLine* bestLine = nullptr;

            for (const auto& line : lines.subspan(1))
            {
                if (!followerClear(line))
                {
                    continue;
                }

                const auto candidate = safe(line.ahead);
                if (candidate > best + options.laneShiftGainMetresPerSecond)
                {
                    best = candidate;
                    bestLine = &line;
                }
            }

            if (bestLine != nullptr)
            {
                unit.laneShifting = true;
                unit.laneShiftLane = bestLine->lane;
                unit.laneShiftSide = bestLine->side;
                unit.laneShiftMetres = bestLine->offsetMetres;
                chosen = bestLine;
            }
        }
    }

    unit.trafficAhead = chosen->ahead.found;
    unit.trafficGapMetres = chosen->ahead.gapMetres;
    unit.trafficSpeedMetresPerSecond = chosen->ahead.speedMetresPerSecond;
}

// The roles, dealt by where each unit is in the player's frame, nearest first: the tail to a car
// behind the player, the lead to one ahead, each flank to a car on that side; a unit holding a
// station it is still placed for keeps it, so a deal does not send a flank across the player's nose
// because it drifted a metre nearer than the other; and only when no unit is placed for a station
// does the nearest spare take it and drive to it. Everybody outside reach chases until a station
// comes free. Dealt by distance alone — the first cut — a unit abreast of the player was handed the
// tail and sent 9 m behind itself at speed, which the driver answered with full lock.
void PursuitDirector::assignRoles(const PursuitPlayer& player, const TrafficPopulation& population)
{
    const auto agents = population.agents();

    const auto velocity = flat(player.velocityMetresPerSecond);
    const auto forward = flatUnit(glm::length(velocity) > 1.5 ? velocity : player.forward, glm::dvec3(0.0, 0.0, 1.0));
    const auto left = leftOf(forward);
    const auto playerPosition = flat(player.positionMetres);

    struct Place
    {
        double along = 0.0;
        double across = 0.0;
        PursuitRole previous = PursuitRole::Chase;
    };

    auto places = std::vector<Place>(roster.size());

    for (auto index = std::size_t{0}; index < roster.size(); index++)
    {
        auto& entry = roster[index];
        places[index].previous = entry.role;
        entry.role = PursuitRole::Chase;

        if (entry.agent >= agents.size())
        {
            entry.distanceMetres = std::numeric_limits<double>::max();

            continue;
        }

        const auto offset = flat(agents[entry.agent].positionMetres) - playerPosition;
        entry.distanceMetres = glm::length(offset);
        places[index].along = glm::dot(offset, forward);
        places[index].across = glm::dot(offset, left);
    }

    auto order = std::vector<std::size_t>();
    order.reserve(roster.size());
    for (auto index = std::size_t{0}; index < roster.size(); index++)
    {
        order.push_back(index);
    }

    std::ranges::sort(order,
                      [&](const std::size_t lhs, const std::size_t rhs)
                      { return roster[lhs].distanceMetres < roster[rhs].distanceMetres; });

    constexpr auto stations =
        std::array{PursuitRole::Tail, PursuitRole::FlankLeft, PursuitRole::FlankRight, PursuitRole::Lead};

    // Whether a unit is on the station's side of the player: a holder keeps its station until it is
    // the margin onto the wrong side, so a flank hovering on the centreline is not dealt the other
    // flank — and its aim the other side — every second and a half.
    const auto placedFor = [&](const std::size_t index, const PursuitRole role, const double margin)
    {
        const auto& place = places[index];

        switch (role)
        {
        case PursuitRole::Tail:
            return place.along < margin;
        case PursuitRole::Lead:
            return place.along > -margin;
        case PursuitRole::FlankLeft:
            return place.across > -margin;
        case PursuitRole::FlankRight:
            return place.across < margin;
        case PursuitRole::Chase:
            break;
        }

        return false;
    };

    const auto spare = [&](const std::size_t index)
    { return roster[index].role == PursuitRole::Chase && roster[index].distanceMetres < options.chaseDistanceMetres; };

    auto taken = std::array<bool, stations.size()>{};

    // Three passes: the holders that are still placed for their station keep it; then the nearest
    // unit placed for each open station; then the nearest spare for whatever is still open.
    for (auto pass = 0; pass < 3; pass++)
    {
        for (auto slot = std::size_t{0}; slot < stations.size(); slot++)
        {
            if (taken[slot])
            {
                continue;
            }

            for (const auto index : order)
            {
                const auto keeps = pass == 0 && places[index].previous == stations[slot];
                const auto placed = pass == 1;
                const auto fills = pass == 2;

                if (!spare(index) || !(keeps || placed || fills))
                {
                    continue;
                }

                if ((keeps || placed) && !placedFor(index, stations[slot], keeps ? options.roleHoldMarginMetres : 0.0))
                {
                    continue;
                }

                roster[index].role = stations[slot];
                taken[slot] = true;

                break;
            }
        }
    }
}

// --- the radio and the search (docs/pursuit-radio-brief.md) ----------------------------------------

// The report is the player as last seen by any patrol car — a cruising car that sees the player calls
// it in, the same sight that starts a chase — and the broadcast is the general location the far
// units are sent to: re-broadcast only when the report has moved `radioUpdateMetres` from it, so a
// far unit's route is rebuilt on the broadcast and never on the clock. The search starts when nobody
// has seen the player for `searchAfterSeconds`, centred on the report, headed the way it was going;
// any sighting ends it.
void PursuitDirector::listen(const PursuitPlayer& player, const bool observed, const double deltaTime,
                             const LaneNetwork& network)
{
    const auto velocity = flat(player.velocityMetresPerSecond);

    if (observed || !report.valid)
    {
        report.valid = true;
        report.positionMetres = player.positionMetres;
        report.velocityMetresPerSecond = velocity;
        report.forward = flatUnit(glm::length(velocity) > 1.5 ? velocity : player.forward, glm::dvec3(0.0, 0.0, 1.0));
        report.ageSeconds = 0.0;

        // The lane it was on, going the way it was going, for the reckoning below.
        const auto projection = projectOntoNetwork(network, player.positionMetres, velocity, options.offRoadLateralMetres);
        report.onRoad = projection.found && projection.lane < network.lanes.size();
        report.lane = projection.lane;
        report.distanceMetres = projection.distanceMetres;
        report.speedAlongMetresPerSecond = 0.0;

        if (report.onRoad)
        {
            if (const auto place = laneAt(network.lanes[report.lane], report.distanceMetres); place)
            {
                report.speedAlongMetresPerSecond = glm::dot(velocity, flat(place->direction));
            }
        }
    }
    else
    {
        report.ageSeconds += deltaTime;
    }

    // Where the player is by now if it kept going (docs/police-driving-brief.md §7): along its lane
    // at the speed it had, for as long as it has been unseen up to the reckoning time — round the
    // loop and onto the successor, the way the traffic advances — or in a straight line off the road.
    const auto travelled = std::min(report.ageSeconds, options.deadReckonSeconds);
    auto predicted = report.positionMetres + report.velocityMetresPerSecond * travelled;

    if (report.onRoad && report.speedAlongMetresPerSecond > 0.5 && report.lane < network.lanes.size())
    {
        const auto step =
            advanceAlongLane(network, report.lane, report.distanceMetres, report.speedAlongMetresPerSecond * travelled);

        if (step.lane < network.lanes.size())
        {
            if (const auto place = laneAt(network.lanes[step.lane], step.distanceMetres); place)
            {
                predicted = place->positionMetres;
            }
        }
    }

    report.predictedMetres = predicted;

    if (!report.broadcastValid || glm::length(flat(predicted - report.broadcastMetres)) > options.radioUpdateMetres)
    {
        report.broadcastValid = true;
        report.broadcastMetres = predicted;
        summary.broadcasts++;
    }

    summary.broadcastMetres = report.broadcastMetres;

    if (observed)
    {
        search.active = false;
        search.seconds = 0.0;
    }
    else if (!search.active && summary.unobservedSeconds > options.searchAfterSeconds)
    {
        search.active = true;
        search.seconds = 0.0;
        search.centreMetres = report.predictedMetres;
        search.heading = report.forward;

        for (auto& entry : roster)
        {
            entry.searchBearingDealt = false;
            entry.searchGoalSet = false;
            entry.searchRadiusMetres = 0.0;
        }
    }

    if (search.active)
    {
        search.seconds += deltaTime;
    }

    summary.reportAgeSeconds = report.ageSeconds;
    summary.searching = search.active;
    summary.searchSeconds = search.seconds;
    summary.searchRadiusMetres = search.active ? ringRadius() : 0.0;
    summary.searchCentreMetres = search.centreMetres;
}

double PursuitDirector::ringRadius() const
{
    return options.searchRadiusMetres + options.searchGrowthMetresPerSecond * search.seconds;
}

// Sixteen bearings round the compass from the search heading; the unit takes the one furthest in
// angle from every bearing already dealt, **the front half of the compass first** — ahead, then
// the two sides, then the front diagonals, and only once those nine are taken the back — and keeps
// it for the search. Dealt ahead-then-back, a swarm sent half of itself the wrong way at every
// break in the sight line (docs/police-driving-brief.md §7).
void PursuitDirector::dealSearchBearing(PursuitUnit& unit) const
{
    constexpr auto candidates = 16;
    constexpr auto pi = 3.14159265358979323846;
    // The bearings within a right angle of ahead: 0, ±22.5°, ±45°, ±67.5°, ±90°.
    constexpr auto frontCandidates = 9;

    const auto inFront = [](const double bearing) { return std::cos(bearing) >= -1e-9; };

    auto frontDealt = 0;
    for (const auto& other : roster)
    {
        if (&other != &unit && other.searchBearingDealt && inFront(other.searchBearingRadians))
        {
            frontDealt++;
        }
    }

    const auto frontOnly = frontDealt < frontCandidates;

    auto bestIndex = 0;
    auto bestSeparation = -1.0;

    for (auto candidate = 0; candidate < candidates; candidate++)
    {
        const auto bearing = 2.0 * pi * static_cast<double>(candidate) / static_cast<double>(candidates);

        if (frontOnly && !inFront(bearing))
        {
            continue;
        }

        auto separation = 2.0 * pi;

        for (const auto& other : roster)
        {
            if (&other == &unit || !other.searchBearingDealt)
            {
                continue;
            }

            auto difference = std::fmod(std::abs(bearing - other.searchBearingRadians), 2.0 * pi);
            difference = std::min(difference, 2.0 * pi - difference);
            separation = std::min(separation, difference);
        }

        if (separation > bestSeparation + 1e-9)
        {
            bestSeparation = separation;
            bestIndex = candidate;
        }
    }

    unit.searchBearingRadians = 2.0 * pi * static_cast<double>(bestIndex) / static_cast<double>(candidates);
    unit.searchBearingDealt = true;
}

// The unit's next search goal: the centre plus its bearing times the ring — the larger of the ring
// now and the ring it stepped out to — snapped to the nearest lane within reach; a bearing with no
// road in reach is rotated by 45° until one is found, and the raw point is the last resort.
glm::dvec3 PursuitDirector::searchGoalFor(PursuitUnit& unit, const LaneNetwork& network) const
{
    constexpr auto pi = 3.14159265358979323846;

    unit.searchRadiusMetres = std::max(unit.searchRadiusMetres, ringRadius());

    const auto heading = search.heading;
    const auto left = leftOf(heading);
    auto fallback = glm::dvec3(0.0);

    for (auto attempt = 0; attempt < 8; attempt++)
    {
        // The bearing itself, then 45° steps alternating either side of it, the front-going one
        // first: a side bearing on a road with no side street falls back to a goal ahead, not
        // behind (it walked round through the rear once the ring had grown past the snap distance —
        // found by the search fixture on 2026-09-12 later, when a faster turn-round got a unit to
        // its first goal sooner).
        const auto steps = static_cast<double>((attempt + 1) / 2);
        // Front-going: toward a bearing of zero, the bearing read in (−π, π].
        const auto toward = std::remainder(unit.searchBearingRadians, 2.0 * pi) > 0.0 ? -1.0 : 1.0;
        const auto sign = attempt % 2 == 1 ? toward : -toward;
        const auto bearing = unit.searchBearingRadians + sign * 0.25 * pi * steps;
        const auto direction = std::cos(bearing) * heading + std::sin(bearing) * left;
        const auto raw = search.centreMetres + direction * unit.searchRadiusMetres;

        if (attempt == 0)
        {
            fallback = raw;
        }

        const auto projection = projectOntoNetwork(network, raw, glm::dvec3(0.0), options.searchSnapMetres);

        if (projection.found && projection.lane < network.lanes.size())
        {
            if (const auto place = laneAt(network.lanes[projection.lane], projection.distanceMetres); place)
            {
                return place->positionMetres;
            }
        }
    }

    return fallback;
}

// The answers the service has finished since last tick, each installed on the unit that asked for
// it — under the serial it asked under, or dropped: the unit left and rejoined, or the chase was
// reset while the search was out. The profile is planned here, on the calling thread (microseconds),
// with the unit's own limits and an exit speed for what the goal is: the player's pace plus the
// closing margin for a unit on the player, the report's pace for one on the broadcast, and the
// search goal speed for one searching.
void PursuitDirector::takeAnswers(const PursuitPlayer& player)
{
    if (service == nullptr)
    {
        return;
    }

    answers.clear();
    service->collect(answers);

    const auto playerSpeed = glm::length(flat(player.velocityMetresPerSecond));
    const auto reportSpeed = glm::length(report.velocityMetresPerSecond);

    for (auto& answer : answers)
    {
        auto* entry = static_cast<PursuitUnit*>(nullptr);
        for (auto& candidate : roster)
        {
            if (candidate.agent == answer.agent)
            {
                entry = &candidate;
            }
        }

        if (entry == nullptr || entry->routeSerial != answer.serial || !entry->routePending)
        {
            continue;
        }

        entry->routePending = false;
        entry->route = std::move(answer.route);

        auto limits = PlannerLimits{};
        limits.maximumSpeedMetresPerSecond = options.maximumUnitSpeedMetresPerSecond;
        limits.corneringLimitG = entry->fullModel ? options.fullModelCorneringLimitG : options.cheapBodyCorneringLimitG;
        limits.cornerMargin = options.cornerMargin;
        limits.offRoadGripFactor = options.offRoadGripFactor;
        limits.brakingMetresPerSecondSquared = entry->fullModel ? options.fullModelBrakingMetresPerSecondSquared
                                                                : options.cheapBodyBrakingMetresPerSecondSquared;
        limits.accelerationMetresPerSecondSquared = options.accelerationMetresPerSecondSquared;
        // A far unit does not slow for its goal: the general location is a point on the runner's
        // way and not a place to stop, and a unit that planned down to the report's speed at it
        // arrived late at every one of them (docs/police-driving-brief.md §7).
        limits.entrySpeedMetresPerSecond = options.maximumUnitSpeedMetresPerSecond;
        limits.exitSpeedMetresPerSecond = entry->searching  ? options.searchGoalSpeedMetresPerSecond
                                          : entry->nearPlayer ? playerSpeed + options.closingMarginMetresPerSecond
                                                              : options.maximumUnitSpeedMetresPerSecond;
        static_cast<void>(reportSpeed);

        // Planned on the route's core: its first point is the unit itself and its last the goal, and
        // the segment from either onto the road is the unit's or the goal's own lateral offset, not a
        // corner of the road — read as one, a unit a metre and a half off its lane was planned down
        // to 10 m/s to join it.
        const auto points = std::span<const RoutePoint>(entry->route.points);
        const auto core = points.size() >= 4 ? points.subspan(1, points.size() - 2) : points;

        entry->plan = planSpeedProfile(core, limits);
        entry->routeAgeSeconds = 0.0;
        entry->routeDrifted = false;
        entry->routeSegment = 0;

        summary.routesBuilt++;
        summary.routeWorstSeconds = std::max(summary.routeWorstSeconds, entry->route.searchSeconds);
    }

    summary.routesPending = service->pending();
}

// Where each unit should be, and how fast it should go to get there.
void PursuitDirector::aimUnits(const PursuitPlayer& player, TrafficPopulation& population, const double deltaTime)
{
    RACEENGINE_ZONE_N("pursuit aim units");

    const auto agents = population.agents();
    const auto& network = population.network();

    const auto velocity = flat(player.velocityMetresPerSecond);
    const auto playerSpeed = glm::length(velocity);
    const auto forward = flatUnit(playerSpeed > 1.5 ? velocity : player.forward, glm::dvec3(0.0, 0.0, 1.0));
    const auto left = leftOf(forward);

    // The box closes when the player is slow: the stations move in to touching.
    const auto boxing = playerSpeed < options.boxSpeedMetresPerSecond;
    const auto tailGap = boxing ? options.boxedTailGapMetres : options.tailGapMetres;
    const auto leadGap = boxing ? options.boxedLeadGapMetres : options.leadGapMetres;
    const auto flankGap = boxing ? options.boxedFlankGapMetres : options.flankGapMetres;

    const auto ramming = summary.level >= options.ramLevel && !boxing;

    summary.nearestUnitMetres = std::numeric_limits<double>::max();
    summary.sightedUnits = 0;
    summary.fullModels = 0;
    summary.routingUnits = 0;
    summary.nearUnits = 0;
    summary.farUnits = 0;
    summary.searchingUnits = 0;

    // What the service has finished, before anybody asks for more.
    takeAnswers(player);

    for (auto& entry : roster)
    {
        if (entry.agent >= agents.size())
        {
            continue;
        }

        const auto& agent = agents[entry.agent];
        const auto unitPosition = flat(agent.positionMetres);
        const auto playerPosition = flat(player.positionMetres);

        entry.joinedSeconds += deltaTime;
        entry.sighted = entry.agent < sight.size() && sight[entry.agent].sighted;
        entry.distanceMetres = glm::length(unitPosition - playerPosition);

        if (entry.sighted)
        {
            summary.sightedUnits++;
        }
        if (entry.fullModel)
        {
            summary.fullModels++;
        }
        if (entry.distanceMetres < summary.nearestUnitMetres)
        {
            summary.nearestUnitMetres = entry.distanceMetres;
            summary.nearestUnitMetresPosition = agent.positionMetres;
        }

        // The unit's own motion, which the look-ahead, the backing-up rules and the stuck test read.
        const auto unitSpeed = glm::length(flat(agent.velocityMetresPerSecond));
        const auto unitHeading = flatUnit(agent.heading, forward);
        const auto unitAlong = glm::dot(unitPosition - playerPosition, forward);
        const auto unitAcross = glm::dot(unitPosition - playerPosition, left);

        // The corridor's reach for the next sight tick: the stopping distance at the unit's share of
        // its braking, plus a reaction, floored (docs/police-driving-brief.md §2.1).
        const auto braking = options.corridorBrakeShare * brakingFor(entry);
        entry.corridorReachMetres = std::max(options.corridorMinimumMetres, options.reactionSeconds * unitSpeed +
                                                                                unitSpeed * unitSpeed / (2.0 * braking));

        // --- near, far, or searching (docs/pursuit-radio-brief.md) ---
        //
        // Near — sees the player, or inside the radio's near distance of it — and the goal is the
        // player. Searching, and the goal is the unit's own search goal. Otherwise the goal is the
        // broadcast, the general location.
        entry.nearPlayer = entry.sighted || entry.distanceMetres < options.radioNearMetres;
        entry.searching = search.active && !entry.nearPlayer;

        if (entry.searching)
        {
            summary.searchingUnits++;

            if (!entry.searchBearingDealt)
            {
                dealSearchBearing(entry);
            }

            // Arrived at this goal: the next is a step further out on the same bearing.
            if (entry.searchGoalSet && glm::length(flat(entry.searchGoalMetres) - unitPosition) < options.searchArriveMetres)
            {
                entry.searchRadiusMetres += options.searchStepMetres;
                entry.searchGoalSet = false;
            }

            if (!entry.searchGoalSet)
            {
                entry.searchGoalMetres = searchGoalFor(entry, network);
                entry.searchGoalSet = true;
            }
        }
        else
        {
            entry.searchGoalSet = false;

            if (entry.nearPlayer)
            {
                summary.nearUnits++;
            }
            else
            {
                summary.farUnits++;
            }
        }

        const auto goal = entry.searching    ? entry.searchGoalMetres
                          : entry.nearPlayer ? player.positionMetres
                                             : report.broadcastMetres;

        // --- routing or not (docs/pursuit-navigation-brief.md, stage 2) ---
        //
        // Blocked — the line of drive, or the corridor in front of the unit — or the player out of
        // reach, and the unit routes. Clear and inside reach for the hold and it comes off its route —
        // a wall's end does not flick it between the two every tick.
        const auto needsRoute = service != nullptr && (!entry.lineClear || entry.corridorBlocked ||
                                                       entry.distanceMetres > options.chaseDistanceMetres);

        if (needsRoute)
        {
            entry.lineClearSeconds = 0.0;
            entry.routing = true;
        }
        else
        {
            entry.lineClearSeconds += deltaTime;

            if (entry.routing && entry.lineClearSeconds >= options.routeHoldSeconds)
            {
                entry.routing = false;
            }
        }

        entry.routeAgeSeconds += deltaTime;

        if (!entry.routing)
        {
            entry.routeAgeSeconds = 1.0e9;
            entry.curvatureAhead = 0.0;
        }
        else
        {
            summary.routingUnits++;

            // A near unit's route is rebuilt on the interval; a far or searching unit's only when its
            // goal moves — the broadcast, or the next ring — and any unit's at once when it has
            // drifted off the one it has. Never while an answer is out: the unit drives what it has.
            const auto goalMoved = !entry.nearPlayer && glm::length(flat(goal - entry.goalMetres)) > 0.5;
            const auto onInterval = entry.nearPlayer && entry.routeAgeSeconds >= options.routeIntervalSeconds;
            const auto stale = entry.route.points.size() < 2 || goalMoved || onInterval || entry.routeDrifted;

            if (stale && !entry.routePending)
            {
                entry.routeSerial = ++routeSerials;
                entry.routePending = true;
                entry.routeDrifted = false;
                entry.routeAgeSeconds = 0.0;
                entry.goalMetres = goal;

                service->request(entry.agent, entry.routeSerial,
                                 RouteRequest{.fromMetres = agent.positionMetres, .toMetres = goal});
                summary.routesRequested++;
            }
        }

        const auto drivesRoute = entry.routing && entry.route.points.size() >= 2 && !entry.plan.empty();

        // The station in the player's frame: along and across.
        auto along = 0.0;
        auto across = 0.0;
        auto passSide = 0.0;

        switch (entry.role)
        {
        case PursuitRole::Chase:
            break;
        case PursuitRole::Tail:
            along = -(player.lengthMetres + tailGap);
            break;
        case PursuitRole::Lead:
            along = player.lengthMetres + leadGap;
            passSide = 1.0;
            break;
        case PursuitRole::FlankLeft:
            across = 0.5 * player.widthMetres + 0.92 + flankGap;
            break;
        case PursuitRole::FlankRight:
            across = -(0.5 * player.widthMetres + 0.92 + flankGap);
            break;
        }

        auto aim = glm::dvec3(0.0);
        auto station = glm::dvec3(0.0);
        auto wanted = 0.0;

        if (entry.role == PursuitRole::Chase)
        {
            // Lead pursuit at the player: where it will be in a second, at full chat. A unit on the
            // broadcast or a search goal drives at the goal itself.
            aim = entry.nearPlayer ? playerPosition + velocity * 1.0 : flat(goal);
            station = aim;
            wanted = options.maximumUnitSpeedMetresPerSecond;
        }
        else
        {
            station = playerPosition + forward * along + left * across;
            const auto toStation = station - unitPosition;
            // Along the player's heading, positive with the station ahead of the unit; and across it.
            const auto behind = glm::dot(toStation, forward);
            auto lateral = glm::dot(toStation, left);

            // A unit that has to get past the player — a lead coming up from behind, a tail dropping
            // back from ahead — passes beside the player rather than through it: the lead on its own
            // side, the tail on whichever side it is already on. A flank's line is clear of the
            // player already.
            const auto passingForward = passSide != 0.0 && behind > 4.0 && unitAlong < 0.5 * along;
            const auto droppingBack = entry.role == PursuitRole::Tail && behind < -4.0 && unitAlong > 0.5 * along;

            if (passingForward || droppingBack)
            {
                // The side, chosen when the pass begins and held for it: chosen every tick from which
                // side the unit was on, a tail dropping back down the centreline swapped sides every
                // tick and swerved.
                if (entry.passSide == 0)
                {
                    entry.passSide = passSide != 0.0 ? static_cast<int>(passSide) : (unitAcross >= 0.0 ? 1 : -1);
                }

                lateral += static_cast<double>(entry.passSide) * 3.6;
            }
            else
            {
                entry.passSide = 0;
            }

            // The wheel steers at a point on the station's own line a look-ahead ahead of the unit,
            // never at the station: the across error is the wheel's to close and the along error the
            // speed's. Aimed at the station itself, a car abreast of its flank station was asked for
            // a right-angle turn, and the cornering cap in both drivers then held it to walking pace.
            const auto lookAhead =
                std::max(options.stationLookAheadMetres, options.stationLookAheadSeconds * unitSpeed);
            aim = unitPosition + forward * lookAhead + left * lateral;

            // Match the player and close the gap along the road at a rate a driver keeps station at,
            // faster when ramming.
            const auto closing = std::clamp(0.8 * behind, -options.closingMarginMetresPerSecond,
                                            options.closingMarginMetresPerSecond);
            wanted = playerSpeed + closing + (ramming && entry.role == PursuitRole::Tail ? 3.0 : 0.0);

            if (glm::length(toStation) > options.chaseDistanceMetres)
            {
                wanted = options.maximumUnitSpeedMetresPerSecond;
            }

            // A station well behind the unit is not closed by the speed: the unit turns round for it
            // — the aim is the station itself, which the turn-round below reads as behind
            // (docs/police-driving-brief.md §10). Before this a lead unit whose player had stopped
            // thirty metres behind it sat and waited.
            if (behind < -options.reverseAimMetres)
            {
                aim = station;
                wanted = options.turnAroundSpeedMetresPerSecond;
            }

        }

        // --- the route (docs/pursuit-navigation-brief.md, stage 3) ---
        //
        // The aim is the point a look-ahead ahead of the unit's place on the route, **along the route's
        // tangent there** rather than the route's own point a look-ahead on: pure pursuit at the
        // route's point already carries the route's curvature (its demand is κ − 2e/L² − 2ψ/L), so
        // aimed there and given the curvature as feedforward too, a unit on a steady arc asked for
        // twice the arc. Aimed along the tangent, pure pursuit is the error feedback alone — the
        // cross-track and the heading — and the feedforward, the curvature one look-ahead ahead, is
        // the turn-in. The wanted speed is the profile's one reaction time ahead of the car.
        if (drivesRoute)
        {
            const auto place = projectOntoRoute(entry.route, agent.positionMetres, entry.routeSegment, 12);

            if (place.offsetMetres > options.routeDriftMetres)
            {
                entry.routeDrifted = true;
            }

            entry.routeSegment = place.segment;
            entry.routeDistanceMetres = place.distanceMetres;

            const auto& points = entry.route.points;
            const auto next = std::min(place.segment + 1, points.size() - 1);
            const auto tangent =
                flatUnit(points[next].positionMetres - points[place.segment].positionMetres, unitHeading);
            const auto lookAhead = std::max(options.routeLookAheadMetres, options.routeLookAheadSeconds * unitSpeed);
            const auto remaining = entry.route.lengthMetres - place.distanceMetres;

            // Pure pursuit at the route's own point a look-ahead on — not at a point on the tangent
            // with the curvature fed forward (docs/police-driving-brief.md §12). The profile's curvature
            // estimate spreads eight metres either side of a corner, so a wheel fed it turned in early
            // and straightened early, took an early apex and ran three metres wide at the exit, and a
            // tangent aim could not hold the arc; pure pursuit at the point carries the arc itself, to
            // within L² / 8R — half a metre at forty metres a second on a 100 m bend, centimetres in
            // a city corner — with no term to lead or lag.
            aim = remaining > lookAhead ? routePointAt(entry.route, place.distanceMetres + lookAhead)
                                        : points.back().positionMetres;

            // The line through a bend (§12). The route is the lane's chord polyline, which sits inside
            // the true arc by the chords' sagitta — the aim is pushed back out by the mean of it —
            // and the apex is then cut back inside by the allowance that leaves the stated clearance
            // to a kerb at the lane's edge, in full at a tight bend and not at all at a gentle one or
            // a hairpin. Inside is the turn's side: curvature is left positive.
            if (remaining > lookAhead)
            {
                const auto bend = profileCurvatureAt(entry.plan, place.distanceMetres + lookAhead);
                const auto sharpness = std::abs(bend);
                const auto chordBias = options.routeChordMetres * options.routeChordMetres * sharpness / 12.0;
                const auto rising = std::clamp((sharpness - options.apexGentleCurvature) /
                                                   std::max(options.apexTightCurvature - options.apexGentleCurvature, 1e-9),
                                               0.0, 1.0);
                const auto falling = std::clamp((options.apexHairpinCurvature - sharpness) /
                                                    std::max(options.apexHairpinCurvature - options.apexTightCurvature, 1e-9),
                                                0.0, 1.0);
                const auto apexShare = std::min(rising, falling);
                const auto inside = bend >= 0.0 ? 1.0 : -1.0;

                aim += leftOf(tangent) * (inside * (apexShare * options.apexInsideMetres - chordBias));
            }

            station = aim;

            // No feedforward: the aim on the route carries the corner. The drivers still add
            // `curvatureAhead` to their wheel for whoever sets it; a routing unit sets nothing.
            entry.curvatureAhead = 0.0;
            static_cast<void>(tangent);
            wanted = profileSpeedAt(entry.plan, place.distanceMetres + options.reactionSeconds * unitSpeed);
        }

        // Searching is looking, not blasting.
        if (entry.searching)
        {
            wanted = std::min(wanted, options.searchSpeedMetresPerSecond);
        }

        // --- the ram (docs/police-driving-brief.md §12) ---
        //
        // From the ram level, a unit inside range with the player in front of it drives into the
        // player at the maximum, whatever its role and whatever the player is doing — the tail too,
        // since 2026-09-12 later (§13): it kept the PIT before, and a routing tail did neither.
        // Wrecking is the goal; a station is not.
        entry.ramming = false;

        {
            const auto toPlayerFlat = playerPosition - unitPosition;
            const auto bearing = entry.distanceMetres > 1e-6 ? glm::dot(toPlayerFlat, unitHeading) / entry.distanceMetres : 1.0;
            if (summary.level >= options.ramLevel && !entry.reversing && entry.turnPhase == 0 && !entry.searching &&
                entry.distanceMetres < options.ramRangeMetres && bearing > options.ramFacingCosine)
            {
                entry.ramming = true;
                aim = playerPosition + velocity * options.ramLeadSeconds;
                station = aim;
                wanted = options.maximumUnitSpeedMetresPerSecond;
                entry.curvatureAhead = 0.0;
            }
        }

        // --- the block (docs/police-driving-brief.md §12) ---
        //
        // Ahead of the player, the player coming at it within the seconds, and not lined up with the
        // player's direction either way — out of a side street, round a corner: the unit pulls across
        // the player's lane and stops broadside, and holds until the player has passed or stopped.
        // Lined up head on it rams instead (§10); lined up the same way it is a lead.
        {
            const auto alignment = glm::dot(unitHeading, forward);
            const auto closing = std::max(-glm::dot(velocity, unitHeading), 0.0) +
                                 std::max(glm::dot(flat(agent.velocityMetresPerSecond), unitHeading), 0.0);
            const auto secondsToMeet = closing > 1e-6 ? entry.distanceMetres / closing : 1.0e9;

            if (!entry.blocking && !entry.ramming && !entry.reversing && entry.turnPhase == 0 && !entry.searching &&
                !boxing && unitAlong > 4.0 && std::abs(alignment) < options.blockAlignment && closing > 5.0 &&
                secondsToMeet < options.blockSeconds)
            {
                entry.blocking = true;
                entry.blockSeconds = 0.0;

                // Along its own line, then toward the player's centreline by up to the across.
                const auto towardCentre = unitAcross >= 0.0 ? -1.0 : 1.0;
                const auto blockAcross = std::min(std::abs(unitAcross), options.blockAcrossMetres);
                entry.blockPointMetres =
                    unitPosition + unitHeading * options.blockRunMetres + left * (towardCentre * blockAcross);
            }

            if (entry.blocking)
            {
                entry.blockSeconds += deltaTime;

                const auto passed = unitAlong < -2.0;
                const auto stopped = playerSpeed < 2.0 && entry.blockSeconds > 2.0;

                if (passed || stopped || entry.ramming || entry.blockSeconds > options.blockHoldSeconds)
                {
                    entry.blocking = false;
                }
                else
                {
                    aim = entry.blockPointMetres;
                    station = aim;
                    const auto toBlock = glm::length(flat(entry.blockPointMetres) - unitPosition);
                    wanted = toBlock > 3.0 ? options.turnAroundSpeedMetresPerSecond : 0.0;
                    entry.curvatureAhead = 0.0;
                }
            }
        }

        // --- turning round (docs/police-driving-brief.md §10) ---
        //
        // The aim behind the unit and the unit slow enough to turn: the room to either side, read by
        // the two room rays on the sight tick, says whether a full-lock arc fits. Where it does, the
        // driver's own turn-round — full lock at the turn-round speed — is left to do it. Where it
        // does not, a three-point turn: forward on full lock toward the turn side until the nose has
        // used the room, back on the opposite lock while the nose keeps swinging, forward again —
        // each leg timed by the heading turned, capped by the clock, and the drivers told the side.
        {
            constexpr auto pi = 3.14159265358979323846;

            const auto toAimFlat = flat(aim) - unitPosition;
            const auto aimAhead = glm::dot(toAimFlat, unitHeading);
            const auto unitLeft = leftOf(unitHeading);
            const auto radius = options.turnRadiusMetres;

            if (entry.turnPhase == 0 && aimAhead < 0.0 && glm::length(toAimFlat) > 6.0 && !entry.reversing &&
                unitSpeed < options.turnAroundSpeedMetresPerSecond + 1.0)
            {
                // The side is where the full-lock arc fits: the aim's when both do, the roomier when
                // neither (a three-point turn wants the space). The aim's side regardless put a unit
                // with 6.6 m on its left and 16 on its right into the left wall (§14).
                const auto roomOn = [&](const int which)
                { return which > 0 ? entry.roomLeftMetres : entry.roomRightMetres; };
                const auto arc = 2.0 * radius + options.turnBodyWidthMetres;
                const auto fits = [&](const int which) { return roomOn(which) >= arc; };
                const auto aimSide = glm::dot(toAimFlat, unitLeft) >= 0.0 ? 1 : -1;

                auto side = aimSide;
                if (fits(aimSide) != fits(-aimSide))
                {
                    side = fits(aimSide) ? aimSide : -aimSide;
                }
                else if (!fits(aimSide) && roomOn(-aimSide) > roomOn(aimSide))
                {
                    side = -aimSide;
                }

                const auto room = roomOn(side);

                entry.turnSide = side;
                entry.turnHeadingRadians = 0.0;
                entry.turnSeconds = 0.0;
                entry.turnLastHeading = unitHeading;

                if (room < arc)
                {
                    // The three-point turn: forward until the leading front corner has used the room,
                    // less the margin; at least a little, so the turn makes progress.
                    entry.turnPhase = 1;
                    entry.turnGoalRadians = std::max(
                        cornerSweepRadians(room, radius, options.turnBodyLengthMetres, options.turnBodyWidthMetres) -
                            options.turnCornerMarginRadians,
                        0.3);
                }
                else
                {
                    // The arc fits: the drivers' own turn-round, told the side, and over once the aim
                    // is ahead — the third leg on its own.
                    entry.turnPhase = 3;
                    entry.turnGoalRadians = pi;
                }
            }

            if (entry.turnPhase != 0)
            {
                const auto cosine = std::clamp(glm::dot(entry.turnLastHeading, unitHeading), -1.0, 1.0);
                entry.turnHeadingRadians += std::acos(cosine);
                entry.turnLastHeading = unitHeading;
                entry.turnSeconds += deltaTime;

                const auto side = static_cast<double>(entry.turnSide);
                // ...or the car has stopped against something: on with the next leg, not the throttle.
                const auto stopped = entry.turnSeconds > 0.5 && unitSpeed < 0.3;
                const auto legDone = entry.turnHeadingRadians >= entry.turnGoalRadians ||
                                     entry.turnSeconds > options.turnPhaseSeconds || stopped;

                if (entry.turnPhase == 1)
                {
                    // Forward on full lock toward the turn side: an aim behind and beside the unit.
                    aim = unitPosition - unitHeading * 10.0 + unitLeft * (side * 6.0);
                    wanted = options.turnAroundSpeedMetresPerSecond;

                    if (legDone)
                    {
                        entry.turnPhase = 2;
                        entry.turnHeadingRadians = 0.0;
                        entry.turnSeconds = 0.0;
                        entry.turnGoalRadians = options.turnReverseRadians;
                    }
                }
                else if (entry.turnPhase == 2)
                {
                    // Back on the opposite lock; the drivers read the side for the wheel, below.
                    wanted = options.reverseSpeedMetresPerSecond;

                    if (legDone)
                    {
                        entry.turnPhase = 3;
                        entry.turnHeadingRadians = 0.0;
                        entry.turnSeconds = 0.0;
                        entry.turnGoalRadians = pi;
                    }
                }
                else
                {
                    // Forward again at the aim; the driver's own turn-round finishes it, and the
                    // turn is over once the aim is ahead.
                    wanted = std::min(wanted, options.turnAroundSpeedMetresPerSecond);

                    if (aimAhead > 0.0 || entry.turnSeconds > options.turnPhaseSeconds)
                    {
                        entry.turnPhase = 0;
                        entry.turnSide = 0;
                    }
                }

                station = aim;
                entry.curvatureAhead = 0.0;
            }
        }

        // --- the corridor (docs/police-driving-brief.md §2.1) ---
        //
        // The speed is capped to stop short of whatever the rays found in front of the unit, and to
        // the safe approach to the car ahead on the lane it drives for; and a unit shifting onto a
        // neighbour aims at a point a look-ahead ahead of itself on that neighbour's line, toward
        // wherever it was going — the same near aim a station unit steers at, which is what turns a
        // far aim's gentle drift into a lane change.
        auto cap = options.maximumUnitSpeedMetresPerSecond;

        // A ramming unit is capped for nothing beyond the player: it meets the player first, and that
        // is the point. The second log had a ram lifted to 24 m/s for a building on the outside of a
        // bend fifty metres on, with the player pulling away at thirty twenty metres ahead (§13).
        const auto beyondPlayer = entry.ramming && entry.corridorHitMetres > entry.distanceMetres;

        if (entry.corridorHit && !beyondPlayer)
        {
            cap = std::min(cap, std::sqrt(2.0 * braking *
                                          std::max(entry.corridorHitMetres - options.corridorStopMarginMetres, 0.0)));
        }

        if (entry.trafficAhead)
        {
            cap = std::min(cap, std::max(0.0, entry.trafficSpeedMetresPerSecond +
                                                  std::sqrt(2.0 * braking *
                                                            std::max(entry.trafficGapMetres - options.followStandstillMetres,
                                                                     0.0))));
        }

        // Threading a seam between two lanes is done at the seam speed.
        if (entry.laneShifting && entry.laneShiftSide != 0)
        {
            cap = std::min(cap, options.seamSpeedMetresPerSecond);
        }

        entry.corridorCapMetresPerSecond = cap;
        wanted = std::min(wanted, cap);

        // The shift, blended (docs/police-driving-brief.md §13): the offset the aim carries moves toward
        // the target the sight tick read — zero once the shift is over — at the shift rate, and is
        // capped at the offset that crabs the car sideways at that rate through the look-ahead (the
        // aim a lane's width aside at a 12 m look-ahead asked for three times the tyre's grip at
        // 22 m/s, and the car turned in at its limit for a lane change). The blend runs on while the
        // unit reverses, turns or blocks, so the offset is gone by the time it drives on.
        {
            const auto lookAhead = std::max(options.routeLookAheadMetres, options.routeLookAheadSeconds * unitSpeed);
            const auto ceiling = options.laneShiftRateMetresPerSecond * lookAhead / std::max(unitSpeed, 1.0);
            const auto target = std::clamp(entry.laneShifting ? entry.laneShiftMetres : 0.0, -ceiling, ceiling);
            const auto step = options.laneShiftRateMetresPerSecond * deltaTime;

            entry.laneShiftAppliedMetres += std::clamp(target - entry.laneShiftAppliedMetres, -step, step);

            if (std::abs(entry.laneShiftAppliedMetres) > 1e-3 && !entry.reversing && entry.turnPhase == 0 &&
                !entry.blocking)
            {
                const auto toward = flatUnit(aim - unitPosition, unitHeading);

                aim = unitPosition + toward * lookAhead + leftOf(unitHeading) * entry.laneShiftAppliedMetres;
                station = aim;
            }
        }

        // --- backing up ---
        //
        // A unit that cannot get where it is told going forward backs up. Two cases: the station is
        // behind it and close — it has overshot, or the player has stopped short — and it backs into
        // the station; or it is stuck against something and backs off for a while, the nose swinging
        // toward the aim, before trying again. Read off the unit's own motion, so a body pushing on
        // the player's car and a body wedged on a kerb are the same case. Judged on the station and
        // not on the steering aim, which for a station role is always ahead.
        const auto toStationFlat = flat(station) - unitPosition;
        const auto stationDistance = glm::length(toStationFlat);
        const auto stationAhead = glm::dot(toStationFlat, unitHeading);

        // A three-point turn's backing leg is the director's, not the stuck rule's: held while the leg
        // lasts, dropped the tick the leg ends.
        if (entry.turnPhase == 2)
        {
            entry.reversing = true;
            entry.reversingToStation = false;
            entry.reverseSeconds = 0.0;
            entry.stuckSeconds = 0.0;
        }
        else if (entry.turnPhase == 3 && entry.reversing && !entry.reversingToStation)
        {
            entry.reversing = false;
            entry.reverseSeconds = 0.0;
        }

        if (entry.reversing)
        {
            entry.reverseSeconds += deltaTime;

            const auto timedOut = entry.reverseSeconds >= options.reverseSeconds;
            const auto arrived = entry.reversingToStation && (stationAhead > -1.0 || stationDistance < 1.5);

            if (timedOut || arrived)
            {
                entry.reversing = false;
                entry.reversingToStation = false;
                entry.stuckSeconds = 0.0;
                entry.reverseSeconds = 0.0;
            }
        }
        else
        {
            const auto stationBehind = entry.role != PursuitRole::Chase && stationAhead < -1.0 &&
                                       stationDistance < options.reverseAimMetres && unitSpeed < 3.0;
            const auto wantsToMove = wanted > 2.0 && stationDistance > 2.0;

            entry.stuckSeconds =
                wantsToMove && unitSpeed < options.stuckSpeedMetresPerSecond ? entry.stuckSeconds + deltaTime : 0.0;

            if (stationBehind || entry.stuckSeconds >= options.stuckSeconds)
            {
                entry.reversing = true;
                entry.reversingToStation = stationBehind;
                entry.reverseSeconds = 0.0;
                entry.stuckSeconds = 0.0;
            }
        }

        if (entry.reversing)
        {
            // Both drivers back up by reverse pure pursuit at the aim, so while backing the aim is
            // the station itself.
            wanted = options.reverseSpeedMetresPerSecond;
            aim = station;
        }

        wanted = std::clamp(wanted, 0.0, options.maximumUnitSpeedMetresPerSecond);

        // The aim rides at the player's height: the population's driver flattens it and a full
        // model's driver does too.
        aim.y = player.positionMetres.y;
        station.y = player.positionMetres.y;

        entry.aimMetres = aim;
        entry.stationMetres = station;
        entry.wantedSpeedMetresPerSecond = wanted;

        population.setPursuitAim(entry.agent, aim, wanted, true, entry.reversing, entry.curvatureAhead, entry.turnSide);
    }
}

void PursuitDirector::endChase(TrafficPopulation& population)
{
    for (const auto& entry : roster)
    {
        population.endPursuit(entry.agent, false);
    }

    roster.clear();
    summary.active = false;
    summary.units = 0;
    summary.sightedUnits = 0;
    summary.fullModels = 0;
    summary.nearestUnitMetres = std::numeric_limits<double>::max();
    arrestRemainder = 0.0;
}

const PursuitStatus& PursuitDirector::update(const double deltaTime, const PursuitPlayer& player,
                                             TrafficPopulation& population, const PhysicsWorld& world)
{
    RACEENGINE_ZONE_N("pursuit director update");

    if (deltaTime <= 0.0)
    {
        return summary;
    }

    const auto agents = population.agents();
    const auto& lanes = population.network();

    // --- what the player is doing --------------------------------------------------------------
    const auto velocity = flat(player.velocityMetresPerSecond);
    const auto speed = glm::length(velocity);
    const auto moving = speed > options.offenceSpeedFloorMetresPerSecond;

    summary.speeding = false;
    summary.offRoad = false;
    summary.wrongWay = false;
    summary.overspeedMetresPerSecond = 0.0;

    if (moving)
    {
        const auto projection =
            projectOntoNetwork(lanes, player.positionMetres, glm::dvec3(0.0), options.offRoadLateralMetres);

        if (projection.found && projection.lane < lanes.lanes.size())
        {
            const auto& lane = lanes.lanes[projection.lane];
            const auto place = laneAt(lane, projection.distanceMetres);

            const auto overspeed = speed - (lane.speedLimitMetresPerSecond + options.speedingToleranceMetresPerSecond);
            if (overspeed > 0.0)
            {
                summary.speeding = true;
                summary.overspeedMetresPerSecond = overspeed;
            }

            if (place && glm::dot(velocity, flat(place->direction)) < -options.wrongWaySpeedMetresPerSecond)
            {
                summary.wrongWay = true;
            }
        }
        else if (!lanes.lanes.empty())
        {
            summary.offRoad = true;
        }
    }

    const auto hitPolice = player.impactPoliceMetresPerSecond > options.collisionThresholdMetresPerSecond;
    const auto hitTraffic = player.impactTrafficMetresPerSecond > options.collisionThresholdMetresPerSecond;

    summary.offence = hitPolice          ? Offence::HitPolice
                      : hitTraffic       ? Offence::HitTraffic
                      : summary.wrongWay ? Offence::WrongWay
                      : summary.offRoad  ? Offence::OffRoad
                      : summary.speeding ? Offence::Speeding
                                         : Offence::None;

    // --- who can see it ------------------------------------------------------------------------
    sightRemainder += deltaTime;
    if (sightRemainder >= options.sightIntervalSeconds)
    {
        sightRemainder = 0.0;
        watch(player, population, world);
    }

    if (sight.size() != agents.size())
    {
        sight.assign(agents.size(), Sight{});
    }

    auto observed = false;
    for (const auto id : population.policeIds())
    {
        observed = observed || (id < sight.size() && sight[id].sighted);
    }

    summary.observed = observed;

    // --- the meter -----------------------------------------------------------------------------
    auto felony = summary.felony;

    if (observed)
    {
        if (summary.speeding)
        {
            felony += options.felonyPerSecondSpeeding * (1.0 + summary.overspeedMetresPerSecond / 5.0) * deltaTime;
        }
        if (summary.offRoad)
        {
            felony += options.felonyPerSecondOffRoad * deltaTime;
        }
        if (summary.wrongWay)
        {
            felony += options.felonyPerSecondWrongWay * deltaTime;
        }
        if (hitTraffic)
        {
            felony += options.felonyPerTrafficHit;
        }
        if (summary.active)
        {
            felony += options.felonyPerSecondEvading * deltaTime;
        }
    }
    else if (!summary.active)
    {
        felony -= options.felonyDecayPerSecond * deltaTime;
    }

    if (hitPolice)
    {
        felony += options.felonyPerPoliceHit;
    }

    summary.felony = std::clamp(felony, 0.0, 1.0);
    summary.level = summary.felony <= 0.0 ? 0 : std::min(5, static_cast<int>(summary.felony * 5.0) + 1);
    summary.swarm = summary.felony >= options.swarmFelony;

    // --- the player's own damage ---------------------------------------------------------------
    const auto impact = player.impactPoliceMetresPerSecond + player.impactTrafficMetresPerSecond +
                        player.impactWorldMetresPerSecond;
    if (impact > options.collisionThresholdMetresPerSecond)
    {
        summary.playerDamage = std::min(
            1.0, summary.playerDamage + (impact - options.collisionThresholdMetresPerSecond) * options.damagePerImpactSpeed);
    }

    const auto wreckedNow = !summary.wrecked && summary.playerDamage >= 1.0;
    summary.wrecked = summary.wrecked || wreckedNow;

    // --- starting a chase ----------------------------------------------------------------------
    const auto offending = summary.offence != Offence::None;

    if (!summary.active && !summary.busted && !summary.wrecked && (hitPolice || (observed && offending)))
    {
        summary.active = true;
        summary.unobservedSeconds = 0.0;
        summary.pursuitsStarted++;
        roleRemainder = 1.0e9;
        arrestRemainder = 0.0;
    }

    if (!summary.active)
    {
        summary.units = 0;
        summary.sightedUnits = 0;
        summary.fullModels = 0;
        summary.nearestUnitMetres = std::numeric_limits<double>::max();
        report = Report{};
        search = Search{};
        summary.searching = false;
        summary.searchSeconds = 0.0;
        summary.searchRadiusMetres = 0.0;

        return summary;
    }

    // --- the radio (docs/pursuit-radio-brief.md) ----------------------------------------------
    summary.unobservedSeconds = observed ? 0.0 : summary.unobservedSeconds + deltaTime;
    listen(player, observed, deltaTime, lanes);

    // --- who is in it --------------------------------------------------------------------------
    //
    // Units the population has written off — rolled, or otherwise no longer chasing — leave first,
    // so their places can be dealt again below.
    std::erase_if(roster,
                  [&](const PursuitUnit& entry)
                  { return entry.agent >= agents.size() || !inChase(agents[entry.agent].mode); });

    // Damage, from what the population's bodies took and what the caller reported for the full
    // models. A wreck leaves the chase and stops where it is.
    if (reportedImpacts.size() < agents.size())
    {
        reportedImpacts.resize(agents.size(), 0.0);
    }

    for (auto& entry : roster)
    {
        auto taken = population.takeImpact(entry.agent);
        if (entry.agent < reportedImpacts.size())
        {
            taken += reportedImpacts[entry.agent];
            reportedImpacts[entry.agent] = 0.0;
        }

        if (taken > options.collisionThresholdMetresPerSecond)
        {
            entry.damage =
                std::min(1.0, entry.damage + (taken - options.collisionThresholdMetresPerSecond) * options.damagePerImpactSpeed);
        }
    }

    for (const auto& entry : roster)
    {
        if (entry.damage >= 1.0)
        {
            population.endPursuit(entry.agent, true);
            summary.policeWrecked++;
        }
    }

    std::erase_if(roster, [](const PursuitUnit& entry) { return entry.damage >= 1.0; });

    // Joining: every patrol car close enough, or every patrol car at all under a swarm.
    const auto limit = summary.swarm ? population.policeIds().size() : options.maximumUnits;

    for (const auto id : population.policeIds())
    {
        if (roster.size() >= limit)
        {
            break;
        }

        if (id >= agents.size() || unit(id) != nullptr)
        {
            continue;
        }

        const auto& agent = agents[id];
        if (!upright(agent))
        {
            continue;
        }

        const auto distance = glm::length(flat(agent.positionMetres - player.positionMetres));
        const auto sees = id < sight.size() && sight[id].sighted;

        if (!summary.swarm && !sees && distance > options.joinRadiusMetres)
        {
            continue;
        }

        if (population.beginPursuit(id, world))
        {
            roster.push_back(PursuitUnit{.agent = id});
            roleRemainder = 1.0e9;
        }
    }

    // --- the tactics ---------------------------------------------------------------------------
    roleRemainder += deltaTime;
    if (roleRemainder >= options.roleIntervalSeconds)
    {
        roleRemainder = 0.0;
        assignRoles(player, population);
    }

    aimUnits(player, population, deltaTime);
    summary.units = roster.size();

    // --- how it ends ---------------------------------------------------------------------------
    const auto cornered = summary.nearestUnitMetres < options.arrestRadiusMetres;
    arrestRemainder = cornered && speed < options.arrestSpeedMetresPerSecond ? arrestRemainder + deltaTime : 0.0;

    if (arrestRemainder >= options.arrestSeconds)
    {
        summary.busted = true;
        endChase(population);

        return summary;
    }

    if (summary.wrecked && cornered)
    {
        // A wrecked runner with a patrol car on it is caught.
        summary.busted = true;
        endChase(population);

        return summary;
    }

    if (summary.unobservedSeconds > options.loseSightSeconds || roster.empty())
    {
        endChase(population);
    }

    return summary;
}

// Pure pursuit for a full model. The same construction the population's own driver uses, through a
// road wheel: the arc to the aim is a curvature, the curvature is a road wheel angle through the
// wheelbase, and the angle is a demand against the car's lock. A positive demand is a right turn and
// +x is the car's left, so an aim to the left is a negative demand.
PursuitDrive PursuitDirector::drive(const PursuitUnit& unit, const PursuitCarPose& pose) const
{
    auto drive = PursuitDrive{};

    const auto forward = flatUnit(pose.orientation * glm::dvec3(0.0, 0.0, 1.0), glm::dvec3(0.0, 0.0, 1.0));
    const auto left = leftOf(forward);
    const auto along = glm::dot(pose.velocityMetresPerSecond, forward);
    const auto speed = glm::length(flat(pose.velocityMetresPerSecond));

    const auto toAim = flat(unit.aimMetres - pose.positionMetres);
    const auto distance = glm::length(toAim);

    if (distance < 0.3)
    {
        drive.brake = along > 0.5 ? 0.5 : 0.0;

        return drive;
    }

    const auto ahead = glm::dot(toAim, forward);
    const auto aside = glm::dot(toAim, left);

    if (unit.reversing)
    {
        // Backing up, on the director's word, the cheap driver's two cases through a road wheel: toward
        // a station behind the car the wheel turns the way it would going forward (the rear follows
        // the front wheels round the same circle, so an aim to the left is still a negative demand);
        // away from something in the way it is turned away from the aim, so the nose swings toward it.
        drive.reverse = true;

        const auto lockNow = std::max(pose.lockRadians, 1e-3);
        auto steering = 0.0;

        if (unit.turnSide != 0)
        {
            // Backing through a three-point turn: the opposite lock to the turn, so the nose keeps
            // swinging round — a left turn backs with the wheel right, a positive demand
            // (docs/police-driving-brief.md §10).
            steering = static_cast<double>(unit.turnSide);
        }
        else if (ahead < 0.0)
        {
            const auto angle = std::atan2(aside, std::max(-ahead, 0.5));
            const auto curvature = 2.0 * std::sin(angle) / std::max(distance, 4.0);

            steering = -std::atan(pose.wheelbaseMetres * curvature) / lockNow;
        }
        else
        {
            steering = std::copysign(0.7, aside);
        }

        drive.steering = std::clamp(steering, -1.0, 1.0);

        // Still rolling forward: the brakes first. Then back up at the speed asked for.
        if (along > 0.5)
        {
            drive.brake = 1.0;

            return drive;
        }

        const auto error = unit.wantedSpeedMetresPerSecond + along;
        drive.throttle = std::clamp(error / 2.0, 0.0, 1.0);
        drive.brake = std::clamp(-error / 2.0, 0.0, 1.0);

        return drive;
    }

    // The aim behind the car and near: the station has been overshot. Stop and let it come back.
    if (ahead < 0.0 && distance < 6.0)
    {
        drive.brake = 1.0;

        return drive;
    }

    // Pure pursuit: the circle through the car's position, tangent to its heading, through the
    // aim. The lookahead floor keeps a close aim from asking for a curvature no car can make.
    const auto lookAhead = std::max({distance, 6.0, 0.55 * speed});
    const auto angle = std::atan2(aside, std::max(ahead, 0.5));
    const auto curvature = 2.0 * std::sin(angle) / lookAhead;
    const auto lock = std::max(pose.lockRadians, 1e-3);
    const auto lateralLimit = pose.corneringLimitG * standardGravity;

    // The demand is the pure pursuit plus the turn-in — the route's curvature one look-ahead ahead,
    // zero off a route (docs/pursuit-navigation-brief.md, stage 3) — **limited to the curvature the
    // tyre holds at this speed**: at speed the wheel corrects a lateral error at the tyre's limit
    // and no harder, the way a real car understeers, and the speed is not touched for it
    // (docs/police-driving-brief.md §9). Left positive, and a left turn is a negative demand.
    const auto tyreCurvature = lateralLimit / std::max(speed * speed, 1.0);
    const auto demanded = curvature + unit.curvatureAhead;
    const auto gripWheel = std::atan(pose.wheelbaseMetres * tyreCurvature);

    // The slide (docs/police-driving-brief.md §13): the body slip — the velocity's angle off the nose,
    // left positive — past the threshold says the car is yawing on its own and the tyre is past its
    // peak. The counter-steer is then **the slip itself**, fed forward under the pursuit's wheel, so
    // the front wheels point along the velocity whatever the aim asks: blended in from the threshold
    // to twice it, and bounded by the lock alone. Merely allowing the wheel past the grip angle was
    // not enough — the pursuit's demand toward the aim is not a counter-steer, and the second log had
    // a unit yawing at 37°/s with 12° of slip and 4° of wheel (2026-09-12, later).
    const auto slip = speed > 1.0 ? std::atan2(glm::dot(flat(pose.velocityMetresPerSecond), left), std::max(along, 0.5))
                                  : 0.0;
    const auto slideShare = std::clamp((std::abs(slip) - options.slideSlipRadians) / std::max(options.slideSlipRadians, 1e-6), 0.0, 1.0);
    const auto pursuitWheel = std::clamp(std::atan(pose.wheelbaseMetres * demanded), -gripWheel, gripWheel);
    const auto roadWheel = std::clamp(pursuitWheel + slideShare * slip, -lock, lock);

    drive.steering = std::clamp(-roadWheel / lock, -1.0, 1.0);

    // Facing away from the aim. At speed a straight-line stop first — full lock at twenty metres a
    // second is a spin, and a spun patrol car is a wreck for the director to write off. Slow, full
    // lock toward it and a gentle throttle turn the car round in the road; the director's backing-up
    // rule takes over if it cannot.
    if (ahead < 0.0)
    {
        if (speed > options.turnAroundSpeedMetresPerSecond)
        {
            drive.steering = 0.0;
            drive.brake = 1.0;

            return drive;
        }

        // Full lock toward the side the director chose — where the arc fits — and toward the aim
        // when it chose none (§14). A left turn is a negative demand.
        drive.steering = unit.turnSide != 0 ? -static_cast<double>(unit.turnSide) : (aside > 0.0 ? -1.0 : 1.0);
        drive.throttle = 0.4;

        return drive;
    }

    // The ram (docs/police-driving-brief.md §13): the pedal is the throttle and nothing else. The
    // corner cap below read a ramming unit's own yaw — the player twenty degrees off its nose because
    // the car was crossed up — as a hairpin and braked to twelve for it (the police log, 2026-09-12).
    // A wall the director's corridor capped the wanted speed for is met on a lifted throttle.
    if (unit.ramming)
    {
        drive.throttle = std::clamp((unit.wantedSpeedMetresPerSecond - along) / 4.0, 0.0, 1.0);

        return drive;
    }

    // The speed the director asked for, capped for a corner in the path: the route's curvature
    // ahead, and — off a route, or with the aim well off the nose — the pure-pursuit curvature. A
    // small error's demand is not a corner and caps nothing; the wheel above deals with it.
    auto pathCurvature = std::abs(unit.curvatureAhead);
    if (std::abs(angle) > options.steeringCapAngleRadians)
    {
        pathCurvature = std::max(pathCurvature, std::abs(curvature));
    }

    const auto cap = pathCurvature > 1e-4 ? std::sqrt(lateralLimit / pathCurvature) : std::numeric_limits<double>::max();
    const auto wanted = std::min(unit.wantedSpeedMetresPerSecond, cap);
    const auto error = wanted - along;

    // The brake is the deceleration the wanted speed asks for over the reaction horizon it is read
    // at, as a share of what the car can do — a feedforward, so the car meets a corner at the
    // profile's speed rather than a few metres a second over it; the old proportional pedal needed
    // four metres a second of lag for full brake, and that lag ran every bend wide
    // (docs/police-driving-brief.md §12). The throttle stays proportional.
    drive.throttle = std::clamp(error / 4.0, 0.0, 1.0);

    if (error < 0.0)
    {
        const auto horizon = std::max(options.reactionSeconds * along, 2.0);
        const auto needed = (along * along - wanted * wanted) / (2.0 * horizon);
        const auto braking = std::max(options.fullModelBrakingMetresPerSecondSquared, 1.0);

        drive.brake = std::clamp(std::max(needed / braking, -error / 4.0), 0.0, 1.0);
    }

    return drive;
}

} // namespace raceengine
