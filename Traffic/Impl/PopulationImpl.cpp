// Traffic population bodies. Declarations are in Api/Population.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export`. Which matters
// here more than anywhere: every number in the driver model is one the seat will move, and moving
// one rebuilds an object file rather than every importer of `raceengine`.
module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <Profiling/RaceEngineProfile.hpp>

module raceengine.traffic;

// The physics the two tiers are built on. A partition's own `import` is not re-exported, so an
// implementation unit of this module has to name it again — the same repetition the engine's
// `extern "C++"` bridge declarations need, and for the same reason.
import raceengine.physics;

namespace raceengine
{

namespace
{

constexpr auto worldUp = glm::dvec3(0.0, 1.0, 0.0);

// Smooth in and smooth out across a lane change, and its rate. A linear crossing starts and stops
// with a step in lateral velocity, which is visible as a flick of the body at both ends.
[[nodiscard]] double easeInOut(const double t)
{
    const auto clamped = std::clamp(t, 0.0, 1.0);

    return clamped * clamped * (3.0 - 2.0 * clamped);
}

[[nodiscard]] double easeRate(const double t)
{
    const auto clamped = std::clamp(t, 0.0, 1.0);

    return 6.0 * clamped * (1.0 - clamped);
}

// How much road a lane change takes at the least and at the most. The floor is a hard pull-out — full
// lock on a road car is about a five-metre radius, so a lane's width is crossed in a dozen metres —
// and the ceiling keeps a fast car's change from stretching over a hundred metres of road.
constexpr auto minimumChangeLengthMetres = 12.0;
constexpr auto maximumChangeLengthMetres = 90.0;
// A run has to have this much left in it before a change is offered, and the change is sized to end
// inside it.
constexpr auto minimumChangeRoomMetres = 22.0;
// Until the change has moved the car this far across (as a fraction of the lane's gap), its own
// lane's car and light still hold it. Just under half a lane: the car is mostly clear of it by then.
constexpr auto ownLaneClearanceWeight = 0.45;
// A change given up part way (docs/police-driving-brief.md §7) runs its blend backwards this many
// times faster than it went out — the jerk — and is not given up once the car is this far across:
// past three quarters of a lane, finishing is the safer of the two.
constexpr auto abortRateFactor = 2.0;
constexpr auto abortWeightLimit = 0.75;
// The standstill gap a car keeps while nosing out round the one in front, where the following model
// keeps a queue's. A queue's is what stopped every car from ever pulling out of one.
constexpr auto pullOutGapMetres = 0.8;
// Over how much road a residual lateral offset eases out: a car length, so it halves in about four.
constexpr auto settleLengthMetres = 6.0;
// How far the same place on the lane being joined may stand from the car's own lane point before
// the change is given up: a neighbour is within a lane's width and the shift names the same place
// along it, so the two are a few metres apart on any bend a road draws. Further is a run whose shift
// has stepped to another pass of the same lane, and a car blending toward it crosses the map
// (2026-09-11, docs/traffic-system-brief.md §20). The runs are built not to carry such a step; this
// is the pose refusing to drive on one if one ever arrives.
constexpr auto laneChangeReachMetres = 12.0;
// The body that follows its plan. It aims this far ahead on the plan's own path (a floor, and a time
// at speed), closes what it has fallen behind by with this gain, and the plan is pulled back to it
// when it leads by more than the leash — so the two never part company. Past the lost distances the
// body is not following anything any more and is handed to recovery.
constexpr auto followLookAheadMetres = 5.0;
constexpr auto followLookAheadSeconds = 0.6;
constexpr auto followGapGainPerSecond = 0.8;
constexpr auto planLeadLimitMetres = 2.5;
constexpr auto bodyLostLateralMetres = 3.5;
constexpr auto bodyLostAlongMetres = 8.0;
// The contact solver pushes penetration out at a fixed fraction per tick, which is fifty-four times
// the depth per second: a body stood up two centimetres into the road is nudged at a metre a second,
// and one stood up half a metre inside a kerb or a wall leaves at twenty-seven. So a body is stood up
// only where nothing is deeper than the first, and a following body is dropped back to a point
// before the solver sees anything deeper than the second — a lane whose chord cuts a building has
// taken it somewhere the lane model cannot see. The third is a speed no traffic car reaches.
constexpr auto spawnPenetrationLimitMetres = 0.02;
constexpr auto deepPenetrationMetres = 0.15;
constexpr auto bodySpeedLimitMetresPerSecond = 60.0;
// How fast the heading may turn per metre covered. A city corner is an eight-metre radius, which is
// 0.125 rad/m, so nothing a lane's own geometry asks for is ever limited; a forty-five degree kink
// at a join takes five metres instead of one tick.
constexpr auto maximumYawRadiansPerMetre = 0.15;

// An orientation whose +z is `forward` and whose +y is as near world up as it can be. Built by hand
// rather than through glm's look-at helpers, which are quoted in a view convention where the camera
// looks down -z: using one here points every car in the city backwards.
// `heading` turned toward `wanted` by at most `limitRadians`, about the axis between them — or about
// up when they are opposite, which is the one case with no axis of its own.
[[nodiscard]] glm::dvec3 turnToward(const glm::dvec3& heading, const glm::dvec3& wanted, const double limitRadians)
{
    const auto cosine = std::clamp(glm::dot(heading, wanted), -1.0, 1.0);
    const auto angle = std::acos(cosine);

    if (angle <= limitRadians)
    {
        return wanted;
    }

    auto axis = glm::cross(heading, wanted);
    const auto axisLength = glm::length(axis);
    axis = axisLength > 1e-9 ? axis / axisLength : worldUp;

    return glm::normalize(glm::angleAxis(limitRadians, axis) * heading);
}

[[nodiscard]] glm::dquat orientationAlong(const glm::dvec3& forward)
{
    const auto length = glm::length(forward);
    if (length < 1e-9)
    {
        return glm::dquat(1.0, 0.0, 0.0, 0.0);
    }

    const auto ahead = forward / length;
    auto left = glm::cross(worldUp, ahead);
    const auto leftLength = glm::length(left);

    if (leftLength < 1e-9)
    {
        return glm::dquat(1.0, 0.0, 0.0, 0.0);
    }

    left /= leftLength;

    // Columns are the x, y and z axes, and +x is this engine's left.
    return glm::quat_cast(glm::dmat3(left, glm::cross(ahead, left), ahead));
}

} // namespace

TrafficPopulation::TrafficPopulation(LaneNetwork network, TrafficPopulationOptions settings,
                                     std::vector<DriverProfile> drivers) :
    lanes(std::move(network)),
    options(settings),
    profiles(std::move(drivers)),
    generator(settings.seed)
{
    if (profiles.empty())
    {
        profiles = defaultDriverProfiles();
    }

    // Cumulative weights, normalised once. The draw is then one random number and a linear scan of a
    // list three entries long.
    auto total = 0.0;
    for (const auto& profile : profiles)
    {
        total += std::max(0.0, profile.weight);
    }

    profileWeights.reserve(profiles.size());
    auto running = 0.0;

    for (const auto& profile : profiles)
    {
        running += total > 0.0 ? std::max(0.0, profile.weight) / total : 1.0 / static_cast<double>(profiles.size());
        profileWeights.push_back(running);
    }

    occupants.resize(lanes.lanes.size());
    disturbed.resize(options.maximumDisturbed);

    // The same car with the pursuit's forces on it and nothing else moved — the box, the wheelbase
    // and the rays are `options.vehicle`'s, so a pursuing body is batched, collided and drawn as any
    // other body is.
    pursuitVehicle = options.vehicle;
    pursuitVehicle.maximumDriveForceNewtons = options.pursuitDriveForceNewtons;
    pursuitVehicle.maximumBrakeForceNewtons = options.pursuitBrakeForceNewtons;
    pursuitVehicle.frictionCoefficient = options.pursuitFrictionCoefficient;
}

const LaneNetwork& TrafficPopulation::network() const
{
    return lanes;
}

const TrafficPopulationOptions& TrafficPopulation::settings() const
{
    return options;
}

const std::vector<DriverProfile>& TrafficPopulation::driverProfiles() const
{
    return profiles;
}

const TrafficReport& TrafficPopulation::report() const
{
    return summary;
}

std::span<const TrafficAgent> TrafficPopulation::agents() const
{
    return pool;
}

std::vector<std::size_t> TrafficPopulation::profileCounts() const
{
    auto counts = std::vector<std::size_t>(profiles.size(), 0);

    for (const auto& agent : pool)
    {
        if (agent.profile < counts.size())
        {
            counts[agent.profile]++;
        }
    }

    return counts;
}

// SplitMix64. A named published generator rather than a hand-rolled one, because the whole value of
// it here is that two runs of a capture agree: a generator whose sequence depends on anything but
// its own state would put a different city in front of the camera.
double TrafficPopulation::nextRandom()
{
    generator += 0x9E3779B97F4A7C15ULL;

    auto value = generator;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;

    // The top 53 bits, which is exactly the mantissa a double can hold without rounding twice.
    return static_cast<double>(value >> 11) / static_cast<double>(1ULL << 53);
}

std::size_t TrafficPopulation::pickProfile()
{
    const auto draw = nextRandom();

    for (auto index = std::size_t{0}; index < profileWeights.size(); index++)
    {
        if (draw <= profileWeights[index])
        {
            return index;
        }
    }

    return profileWeights.empty() ? 0 : profileWeights.size() - 1;
}

bool TrafficPopulation::placeOnNetwork(TrafficAgent& agent, const glm::dvec3& focusMetres,
                                       const glm::dvec3& focusForward, const bool awayFromFocus)
{
    if (lanes.samples.empty())
    {
        return false;
    }

    // Twenty-four attempts and then give up for this tick. A network whose free space has run out is
    // a network that is full, and the answer to that is a lower density rather than a longer search.
    for (auto attempt = 0; attempt < 24; attempt++)
    {
        const auto pick = static_cast<std::size_t>(nextRandom() * static_cast<double>(lanes.samples.size()));
        const auto& sample = lanes.samples[std::min(pick, lanes.samples.size() - 1)];

        if (awayFromFocus && !outOfSight(sample.positionMetres, focusMetres, focusForward))
        {
            continue;
        }

        auto crowded = false;

        for (const auto& other : pool)
        {
            if (other.id == agent.id || other.id == noAgent)
            {
                continue;
            }

            if (glm::distance(other.positionMetres, sample.positionMetres) < 12.0)
            {
                crowded = true;

                break;
            }
        }

        if (crowded)
        {
            continue;
        }

        const auto& lane = lanes.lanes[sample.lane];
        const auto& behaviour = profiles[agent.profile].behaviour;

        agent.mode = AgentMode::Cruising;
        agent.lane = sample.lane;
        agent.distanceMetres = sample.distanceMetres;
        agent.changeTarget = sample.lane;
        agent.changeProgress = 0.0;
        agent.changeAborting = false;
        agent.changeLengthMetres = 0.0;
        agent.changeShiftMetres = 0.0;
        agent.settleLateralMetres = 0.0;
        agent.changeCooldownSeconds = 0.0;
        agent.holdSeconds = 0.0;
        agent.yieldSeconds = 0.0;
        agent.yieldLateralMetres = 0.0;
        agent.yieldTargetMetres = 0.0;
        agent.disturbedSeconds = 0.0;
        agent.stillSeconds = 0.0;
        agent.rolledMetres = 0.0;
        agent.bodyLagMetres = 0.0;
        agent.slot = noSlot;
        agent.siren = false;
        agent.impactMetresPerSecond = 0.0;
        agent.pursuitSpeedMetresPerSecond = 0.0;
        agent.pursuitReverse = false;
        agent.wrecked = false;

        // The lane's limit, this driver's offset, and their own share of the spread. Drawn once and
        // kept: a car whose desired speed changed every tick would never settle behind anybody.
        agent.desiredSpeedMetresPerSecond =
            std::max(2.0, lane.speedLimitMetresPerSecond + behaviour.speedOffsetMetresPerSecond +
                              (nextRandom() * 2.0 - 1.0) * behaviour.speedSpreadMetresPerSecond);
        agent.speedMetresPerSecond = agent.desiredSpeedMetresPerSecond;

        refreshCruisingPose(agent, 0.0, true);

        return true;
    }

    return false;
}

bool TrafficPopulation::outOfSight(const glm::dvec3& pointMetres, const glm::dvec3& focusMetres,
                                   const glm::dvec3& focusForward) const
{
    const auto to = pointMetres - focusMetres;
    const auto distance = glm::length(to);

    if (distance < options.recycleDistanceMetres)
    {
        return false;
    }

    return distance >= options.recycleAheadMetres || glm::dot(to, focusForward) <= 0.0;
}

void TrafficPopulation::seed()
{
    generator = options.seed;
    pool.clear();
    nextId = 0;

    for (auto& slot : disturbed)
    {
        slot = DisturbedSlot{};
    }

    const auto wanted = std::min(options.maximumAgents, static_cast<std::size_t>(options.densityPerKilometre *
                                                                                 lanes.totalLengthMetres / 1000.0));

    pool.reserve(wanted);

    for (auto index = std::size_t{0}; index < wanted; index++)
    {
        auto agent = TrafficAgent{};

        // **The id is the pool index**, and the pool never shrinks — a car that is recycled is
        // rewritten in place. That is what makes the route back from a contact solver's body table a
        // subscript rather than a search, and it is why recycling does not hand out a new id.
        agent.id = static_cast<std::uint32_t>(index);
        agent.profile = pickProfile();

        // The patrol cars are a stated share of the city rather than a fifth of it: one draw decides
        // whether this car is one, and a car that is not draws its shape from the other shapes. With
        // no police body stated the second draw is the old uniform one, bit for bit, and the first
        // draw is not made at all — so every fleet without a patrol car seeds the city it always did.
        const auto policeStated = options.policeBody != noPoliceBody && options.policeBody < options.bodyCount;

        if (policeStated && nextRandom() < options.policeShare)
        {
            agent.police = true;
            agent.body = options.policeBody;
        }
        else if (policeStated && options.bodyCount > 1)
        {
            const auto others = static_cast<double>(options.bodyCount - 1);
            auto drawn = static_cast<std::uint8_t>(nextRandom() * others);
            drawn = static_cast<std::uint8_t>(std::min<unsigned>(drawn, options.bodyCount - 2));
            agent.body = drawn >= options.policeBody ? static_cast<std::uint8_t>(drawn + 1) : drawn;
        }
        else
        {
            agent.body = options.bodyCount == 0
                             ? std::uint8_t{0}
                             : static_cast<std::uint8_t>(nextRandom() * static_cast<double>(options.bodyCount));
        }

        agent.colour = options.colourCount == 0
                           ? std::uint8_t{0}
                           : static_cast<std::uint8_t>(nextRandom() * static_cast<double>(options.colourCount));

        pool.push_back(agent);

        if (!placeOnNetwork(pool.back(), glm::dvec3(0.0), glm::dvec3(0.0), false))
        {
            // No free space left. What is already down is the population this network can hold.
            pool.pop_back();

            break;
        }
    }

    nextId = static_cast<std::uint32_t>(pool.size());

    police.clear();
    for (const auto& agent : pool)
    {
        if (agent.police)
        {
            police.push_back(agent.id);
        }
    }
}

void TrafficPopulation::insertOccupant(const std::size_t lane, const Occupant& occupant)
{
    if (lane >= occupants.size())
    {
        return;
    }

    auto& list = occupants[lane];
    const auto at = std::ranges::lower_bound(list, occupant.distanceMetres, {}, &Occupant::distanceMetres);

    list.insert(at, occupant);
}

void TrafficPopulation::rebuildLaneOccupants(const TrafficUpdate& input)
{
    for (auto& list : occupants)
    {
        list.clear();
    }

    const auto length = 2.0 * options.vehicle.bodyHalfExtentsMetres.z;

    // A car that is not on a lane's centre — an outside vehicle, a body off its lane — is entered on
    // the lane it projects onto and on each neighbour **whose width its body overlaps**, the body's
    // span across the road read off its heading: a car on the seam between two lanes is in both, and
    // a patrol car turning round across two lanes is in both, and the traffic on either brakes for
    // it. Before 2026-09-11 later a player or a patrol car on a seam was in neither
    // (docs/police-driving-brief.md §7, §9).
    const auto halfWidth = options.vehicle.bodyHalfExtentsMetres.x;
    const auto halfLength = options.vehicle.bodyHalfExtentsMetres.z;

    const auto enterWide = [&](const LaneProjection& found, const glm::dvec3& forward, const Occupant& occupant)
    {
        const auto& own = lanes.lanes[found.lane];
        const auto here = laneAt(own, found.distanceMetres);
        if (!here)
        {
            insertOccupant(found.lane, occupant);

            return;
        }

        // How far the body reaches across the road: its width along the road, its length across it.
        const auto side = laneSide(here->tangent);
        const auto along = std::abs(glm::dot(glm::dvec3(forward.x, 0.0, forward.z), here->tangent));
        const auto across = std::abs(glm::dot(glm::dvec3(forward.x, 0.0, forward.z), side));
        const auto reach = along * halfWidth + across * halfLength;

        if (std::abs(found.lateralMetres) < options.laneHalfWidthMetres + reach)
        {
            insertOccupant(found.lane, occupant);
        }

        for (const auto& neighbour : own.neighbours)
        {
            if (found.distanceMetres < neighbour.fromMetres || found.distanceMetres > neighbour.toMetres ||
                neighbour.lane >= lanes.lanes.size())
            {
                continue;
            }

            const auto& other = lanes.lanes[neighbour.lane];
            const auto target = wrapDistance(other, found.distanceMetres + neighbourShiftAt(neighbour, found.distanceMetres));
            const auto there = laneAt(other, target);
            if (!there)
            {
                continue;
            }

            const auto centre = glm::dot(there->positionMetres - here->positionMetres, side);
            const auto aside = found.lateralMetres - centre;

            if (std::abs(aside) < options.laneHalfWidthMetres + reach)
            {
                auto beside = occupant;
                beside.distanceMetres = target;
                beside.asideMetres = aside;
                insertOccupant(neighbour.lane, beside);
            }
        }
    };

    for (const auto& agent : pool)
    {
        if (agent.mode == AgentMode::Cruising || agent.mode == AgentMode::Embodied)
        {
            // A body is entered where its body is, not where its plan is: the plan may lead the body
            // by the leash, and the car behind keeps its gap to the metal.
            const auto distance = wrapDistance(lanes.lanes[agent.lane], agent.distanceMetres - agent.bodyLagMetres);

            insertOccupant(agent.lane, Occupant{.distanceMetres = distance,
                                                .speedMetresPerSecond = agent.speedMetresPerSecond,
                                                .lengthMetres = length,
                                                .agent = agent.id,
                                                .siren = agent.siren,
                                                .asideMetres = asideOf(agent, agent.lane, distance)});

            // A car part way through a lane change is in **both** lanes, which is what stops a car
            // in the target lane driving into the side of it. It is entered at the same place on the
            // target that it occupies on its own, which is what the run's shift is for.
            if (agent.changeProgress > 0.0 && agent.changeTarget != agent.lane)
            {
                const auto onTarget = wrapDistance(lanes.lanes[agent.changeTarget], distance + agent.changeShiftMetres);

                insertOccupant(agent.changeTarget, Occupant{.distanceMetres = onTarget,
                                                            .speedMetresPerSecond = agent.speedMetresPerSecond,
                                                            .lengthMetres = length,
                                                            .agent = agent.id,
                                                            .siren = agent.siren,
                                                            .asideMetres = asideOf(agent, agent.changeTarget, onTarget)});
            }

            continue;
        }

        // An external car is one of the caller's own vehicles and arrives through `input.vehicles`
        // below; entering it here as well would put two cars on the road where there is one.
        if (agent.mode == AgentMode::External)
        {
            continue;
        }

        // A car that is off its lane is still in the road, so it goes back onto the network as an
        // obstacle for whoever is coming up behind it. This is what makes a stopped car cause a queue
        // rather than a pile-up.
        const auto forward = agent.orientation * glm::dvec3(0.0, 0.0, 1.0);
        const auto found = projectOntoNetwork(lanes, agent.positionMetres, glm::dvec3(0.0), options.occupantLateralMetres);

        if (found.found)
        {
            // Entered at its speed along the lane, so a pursuing body driving the wrong way reads as
            // a car coming the other way to whoever is on that lane.
            const auto place = laneAt(lanes.lanes[found.lane], found.distanceMetres);
            const auto along = place ? glm::dot(agent.velocityMetresPerSecond, place->direction)
                                     : glm::dot(agent.velocityMetresPerSecond, forward);

            enterWide(found, forward,
                      Occupant{.distanceMetres = found.distanceMetres,
                               .speedMetresPerSecond = along,
                               .lengthMetres = length,
                               .agent = agent.id,
                               .siren = agent.siren,
                               .asideMetres = found.lateralMetres});
        }
    }

    for (const auto& vehicle : input.vehicles)
    {
        const auto found = projectOntoNetwork(lanes, vehicle.positionMetres, glm::dvec3(0.0), options.occupantLateralMetres);
        if (!found.found)
        {
            continue;
        }

        const auto place = laneAt(lanes.lanes[found.lane], found.distanceMetres);
        const auto along = place ? glm::dot(vehicle.velocityMetresPerSecond, place->direction) : 0.0;

        enterWide(found, vehicle.forward,
                  Occupant{.distanceMetres = found.distanceMetres,
                           .speedMetresPerSecond = along,
                           .lengthMetres = vehicle.lengthMetres,
                           .agent = noAgent,
                           .siren = vehicle.siren,
                           .asideMetres = found.lateralMetres});
    }
}

// --- the corridor and the yield (docs/police-driving-brief.md) --------------------------------------
//
// Walks over the occupant lists. Along the lane only — no hop onto a successor and no wrap round a
// loop — because a corridor of a few dozen metres and a siren eighty metres back do not need one,
// and the walks are run for every unit on the sight tick and every car on the decision clock.

TrafficPopulation::Ahead TrafficPopulation::sirenBehindOn(const std::size_t lane, const double distanceMetres,
                                                          const double reachMetres) const
{
    auto answer = Ahead{};

    if (lane >= occupants.size())
    {
        return answer;
    }

    const auto ownHalf = options.vehicle.bodyHalfExtentsMetres.z;
    const auto& list = occupants[lane];
    const auto start = std::ranges::lower_bound(list, distanceMetres, {}, &Occupant::distanceMetres);

    for (auto entry = start; entry != list.begin();)
    {
        --entry;

        const auto gap = distanceMetres - ownHalf - entry->distanceMetres - 0.5 * entry->lengthMetres;
        if (gap > reachMetres)
        {
            return answer;
        }

        if (!entry->siren)
        {
            continue;
        }

        answer.found = true;
        answer.gapMetres = std::max(gap, 0.0);
        answer.speedMetresPerSecond = entry->speedMetresPerSecond;

        return answer;
    }

    return answer;
}

double TrafficPopulation::asideOf(const TrafficAgent& agent, const std::size_t lane, const double distanceMetres) const
{
    if (lane >= lanes.lanes.size())
    {
        return 0.0;
    }

    const auto place = laneAt(lanes.lanes[lane], distanceMetres);
    if (!place)
    {
        return 0.0;
    }

    return glm::dot(agent.positionMetres - place->positionMetres, laneSide(place->tangent));
}

// The lines a point could drive along, in the lane's own frame: the centre, each neighbour's centre,
// the seam between the lane and each neighbour, and a seam's width out where there is no neighbour.
// A car on any of the three lanes blocks a line when its body — at the offset the lists carry for it
// — overlaps the line's width. Everything is built in the lane's frame and turned into the query's
// at the end, so a query driving the lane the wrong way reads the same lines with the sides, the
// speeds and ahead-behind reversed.
TrafficCorridor TrafficPopulation::trafficAround(const glm::dvec3& pointMetres, const glm::dvec3& heading,
                                                 const double lateralMetres, const double reachMetres) const
{
    auto corridor = TrafficCorridor{};

    const auto projection = projectOntoNetwork(lanes, pointMetres, glm::dvec3(0.0), lateralMetres);
    if (!projection.found || projection.lane >= lanes.lanes.size())
    {
        return corridor;
    }

    const auto& own = lanes.lanes[projection.lane];
    const auto place = laneAt(own, projection.distanceMetres);
    if (!place)
    {
        return corridor;
    }

    const auto level = glm::dvec3(heading.x, 0.0, heading.z);
    const auto wrongWay = glm::dot(level, place->tangent) < 0.0;
    const auto sense = wrongWay ? -1.0 : 1.0;

    corridor.onRoad = true;
    corridor.lane = projection.lane;
    corridor.distanceMetres = projection.distanceMetres;
    corridor.direction = place->tangent;
    corridor.wrongWay = wrongWay;
    corridor.lateralMetres = sense * projection.lateralMetres;

    // The three lists that can put a car on a line: the lane and its neighbour on either side, each
    // with its centre's offset from this lane's centre and the same place along it.
    struct Source
    {
        std::size_t lane = 0;
        double centreMetres = 0.0;
        double distanceMetres = 0.0;
        double side = 0.0;
    };

    auto sources = std::array<Source, 3>{};
    auto sourceCount = std::size_t{1};
    sources[0] = Source{.lane = projection.lane, .centreMetres = 0.0, .distanceMetres = projection.distanceMetres};

    const auto side = laneSide(place->tangent);

    for (const auto& neighbour : own.neighbours)
    {
        if (sourceCount >= sources.size())
        {
            break;
        }

        if (projection.distanceMetres < neighbour.fromMetres || projection.distanceMetres > neighbour.toMetres ||
            neighbour.lane >= lanes.lanes.size())
        {
            continue;
        }

        auto taken = false;
        for (auto index = std::size_t{1}; index < sourceCount; index++)
        {
            taken = taken || sources[index].side == neighbour.side;
        }
        if (taken)
        {
            continue;
        }

        const auto& other = lanes.lanes[neighbour.lane];
        const auto target =
            wrapDistance(other, projection.distanceMetres + neighbourShiftAt(neighbour, projection.distanceMetres));
        const auto there = laneAt(other, target);
        if (!there)
        {
            continue;
        }

        sources[sourceCount++] = Source{.lane = neighbour.lane,
                                        .centreMetres = glm::dot(there->positionMetres - place->positionMetres, side),
                                        .distanceMetres = target,
                                        .side = neighbour.side};
    }

    // The lines, in the lane's frame. A seam between two lanes is named by the lower-numbered lane
    // and the side the other is on, so a unit crossing from one to the other keeps holding the same
    // line.
    struct Line
    {
        std::size_t lane = 0;
        int side = 0;
        double offsetMetres = 0.0;
    };

    auto lines = std::array<Line, 5>{};
    auto lineCount = std::size_t{1};
    lines[0] = Line{.lane = projection.lane, .side = 0, .offsetMetres = 0.0};

    for (const auto laneSideSign : std::array{1.0, -1.0})
    {
        const Source* beside = nullptr;
        for (auto index = std::size_t{1}; index < sourceCount; index++)
        {
            if (sources[index].side == laneSideSign)
            {
                beside = &sources[index];
            }
        }

        if (beside != nullptr)
        {
            lines[lineCount++] = Line{.lane = beside->lane, .side = 0, .offsetMetres = beside->centreMetres};

            const auto lower = std::min(projection.lane, beside->lane);
            const auto seamSide = lower == projection.lane ? laneSideSign : -laneSideSign;
            lines[lineCount++] =
                Line{.lane = lower, .side = static_cast<int>(seamSide), .offsetMetres = 0.5 * beside->centreMetres};
        }
        else
        {
            lines[lineCount++] = Line{.lane = projection.lane,
                                      .side = static_cast<int>(laneSideSign),
                                      .offsetMetres = laneSideSign * options.seamWithoutNeighbourMetres};
        }
    }

    // A car blocks a line when the two bodies would overlap across the road: their centres closer
    // than a car's width plus the clearance.
    const auto blockingSpan = 2.0 * options.vehicle.bodyHalfExtentsMetres.x + options.lineClearanceMetres;
    const auto ownHalf = options.vehicle.bodyHalfExtentsMetres.z;
    const auto backReach = std::max(reachMetres, options.mergeLookBackMetres);

    auto ahead = std::array<Ahead, 5>{};
    auto behind = std::array<Ahead, 5>{};

    for (auto sourceIndex = std::size_t{0}; sourceIndex < sourceCount; sourceIndex++)
    {
        const auto& source = sources[sourceIndex];
        if (source.lane >= occupants.size())
        {
            continue;
        }

        const auto& list = occupants[source.lane];

        const auto blocks = [&](const Occupant& occupant, const std::size_t lineIndex)
        {
            return std::abs(source.centreMetres + occupant.asideMetres - lines[lineIndex].offsetMetres) < blockingSpan;
        };

        // Forward from the place: the nearest ahead on each line.
        const auto start = std::ranges::upper_bound(list, source.distanceMetres, {}, &Occupant::distanceMetres);
        for (auto entry = start; entry != list.end(); ++entry)
        {
            if (entry->siren || entry->agent == noAgent)
            {
                continue;
            }

            const auto gap = entry->distanceMetres - 0.5 * entry->lengthMetres - source.distanceMetres - ownHalf;
            if (gap > reachMetres)
            {
                break;
            }

            for (auto lineIndex = std::size_t{0}; lineIndex < lineCount; lineIndex++)
            {
                if ((ahead[lineIndex].found && gap >= ahead[lineIndex].gapMetres) || !blocks(*entry, lineIndex))
                {
                    continue;
                }

                ahead[lineIndex].found = true;
                ahead[lineIndex].gapMetres = std::max(gap, 0.0);
                ahead[lineIndex].speedMetresPerSecond = entry->speedMetresPerSecond;
            }
        }

        // And back from it: the nearest behind on each line, further, for the merge.
        const auto from = std::ranges::lower_bound(list, source.distanceMetres, {}, &Occupant::distanceMetres);
        for (auto entry = from; entry != list.begin();)
        {
            --entry;

            if (entry->siren || entry->agent == noAgent)
            {
                continue;
            }

            const auto gap = source.distanceMetres - ownHalf - entry->distanceMetres - 0.5 * entry->lengthMetres;
            if (gap > backReach)
            {
                break;
            }

            for (auto lineIndex = std::size_t{0}; lineIndex < lineCount; lineIndex++)
            {
                if ((behind[lineIndex].found && gap >= behind[lineIndex].gapMetres) || !blocks(*entry, lineIndex))
                {
                    continue;
                }

                behind[lineIndex].found = true;
                behind[lineIndex].gapMetres = std::max(gap, 0.0);
                behind[lineIndex].speedMetresPerSecond = entry->speedMetresPerSecond;
            }
        }
    }

    const auto toQuery = [&](const Ahead& found)
    {
        return TrafficAhead{.found = found.found,
                            .gapMetres = found.gapMetres,
                            .speedMetresPerSecond = sense * found.speedMetresPerSecond};
    };

    corridor.lineCount = lineCount;
    for (auto lineIndex = std::size_t{0}; lineIndex < lineCount; lineIndex++)
    {
        auto& out = corridor.lines[lineIndex];
        out.lane = lines[lineIndex].lane;
        out.side = wrongWay ? -lines[lineIndex].side : lines[lineIndex].side;
        out.offsetMetres = sense * (lines[lineIndex].offsetMetres - projection.lateralMetres);
        out.ahead = toQuery(wrongWay ? behind[lineIndex] : ahead[lineIndex]);
        out.behind = toQuery(wrongWay ? ahead[lineIndex] : behind[lineIndex]);
    }

    corridor.ahead = corridor.lines[0].ahead;

    return corridor;
}

// Whether the gap a changing car is moving into is still there: the same two tests the change was
// decided on, read again — the leader on the target lane not about to be hit, and the follower
// there, looked for as far back as a fast car can come from, not forced past the safe deceleration
// (docs/police-driving-brief.md §7).
bool TrafficPopulation::changeStillSafe(const TrafficAgent& agent, const DriverBehaviour& behaviour) const
{
    if (agent.changeTarget >= lanes.lanes.size())
    {
        return true;
    }

    const auto onTarget = wrapDistance(lanes.lanes[agent.changeTarget], agent.distanceMetres + agent.changeShiftMetres);
    const auto leader = leaderOn(agent.changeTarget, onTarget, agent.id);
    const auto follower = followerOn(agent.changeTarget, onTarget, agent.id, options.mergeLookBackMetres);

    if (leader.found)
    {
        const auto approach = agent.speedMetresPerSecond - leader.speedMetresPerSecond;
        const auto own = followAcceleration(behaviour, agent.speedMetresPerSecond, agent.desiredSpeedMetresPerSecond,
                                            leader.gapMetres, approach);

        if (leader.gapMetres < 2.0 || own < -behaviour.safeDecelerationMetresPerSecondSquared)
        {
            return false;
        }
    }

    if (follower.found)
    {
        if (follower.gapMetres < 2.0)
        {
            return false;
        }

        const auto after =
            followAcceleration(behaviour, follower.speedMetresPerSecond, agent.desiredSpeedMetresPerSecond,
                               follower.gapMetres, follower.speedMetresPerSecond - agent.speedMetresPerSecond);

        if (after < -behaviour.safeDecelerationMetresPerSecondSquared)
        {
            return false;
        }
    }

    return true;
}

// The yield. Read on the decision clock off the lists as they stand: the nearest siren car behind
// this car on its lane and on each neighbour running beside it; closing, or near, and the car yields
// — the hold re-armed every decision the siren is still there, so it counts down only once it is
// gone. The kerb is the side with no neighbour.
void TrafficPopulation::considerYield(TrafficAgent& agent)
{
    if (agent.lane >= lanes.lanes.size())
    {
        return;
    }

    const auto& own = lanes.lanes[agent.lane];
    const auto reach = options.yieldBehindMetres;

    auto siren = sirenBehindOn(agent.lane, agent.distanceMetres, reach);
    auto leftNeighbour = false;
    auto rightNeighbour = false;

    for (const auto& neighbour : own.neighbours)
    {
        if (agent.distanceMetres < neighbour.fromMetres || agent.distanceMetres > neighbour.toMetres ||
            neighbour.lane >= lanes.lanes.size())
        {
            continue;
        }

        (neighbour.side > 0.0 ? leftNeighbour : rightNeighbour) = true;

        const auto& other = lanes.lanes[neighbour.lane];
        const auto target = wrapDistance(other, agent.distanceMetres + neighbourShiftAt(neighbour, agent.distanceMetres));
        const auto beside = sirenBehindOn(neighbour.lane, target, reach);

        if (beside.found && (!siren.found || beside.gapMetres < siren.gapMetres))
        {
            siren = beside;
        }
    }

    const auto closing = siren.found && siren.speedMetresPerSecond > agent.speedMetresPerSecond - 1.0;
    const auto near = siren.found && siren.gapMetres < options.yieldNearMetres;

    if (!closing && !near)
    {
        return;
    }

    agent.yieldSeconds = options.yieldHoldSeconds;

    if (leftNeighbour && rightNeighbour)
    {
        // A middle lane pulls to the stated side, so the seam on its other side opens too.
        agent.yieldTargetMetres = options.yieldSideWithNoNeighbour * options.yieldLateralMetres;
    }
    else if (leftNeighbour)
    {
        agent.yieldTargetMetres = -options.yieldLateralMetres;
    }
    else if (rightNeighbour)
    {
        agent.yieldTargetMetres = options.yieldLateralMetres;
    }
    else
    {
        agent.yieldTargetMetres = options.yieldSideWithNoNeighbour * options.yieldLateralMetres;
    }
}

double TrafficPopulation::followAcceleration(const DriverBehaviour& behaviour, const double speed,
                                             const double desiredSpeed, const double gapMetres,
                                             const double approachMetresPerSecond) const
{
    const auto free =
        desiredSpeed > 0.0 ? std::pow(std::max(0.0, speed) / desiredSpeed, behaviour.accelerationExponent) : 1.0;

    auto acceleration = behaviour.comfortableAccelerationMetresPerSecondSquared * (1.0 - free);

    if (gapMetres < std::numeric_limits<double>::max())
    {
        // The Intelligent Driver Model's desired gap: a standstill distance, a time headway, and the
        // term that closes a gap being eaten into. The last one is what makes the model brake early
        // for a queue instead of late.
        const auto interaction = 2.0 * std::sqrt(behaviour.comfortableAccelerationMetresPerSecondSquared *
                                                 behaviour.comfortableDecelerationMetresPerSecondSquared);

        const auto wanted =
            behaviour.minimumGapMetres +
            std::max(0.0, speed * behaviour.desiredHeadwaySeconds +
                              (interaction > 1e-9 ? speed * approachMetresPerSecond / interaction : 0.0));

        // A floor on the gap rather than a division by zero. Two cars that are already overlapping
        // are a collision, and what the model owes then is the hardest braking it has.
        const auto gap = std::max(gapMetres, 0.25);
        const auto ratio = wanted / gap;

        acceleration -= behaviour.comfortableAccelerationMetresPerSecondSquared * ratio * ratio;
    }

    return std::max(acceleration, -behaviour.emergencyDecelerationMetresPerSecondSquared);
}

TrafficPopulation::Ahead TrafficPopulation::leaderOn(const std::size_t lane, const double distanceMetres,
                                                     const std::uint32_t ignore) const
{
    auto answer = Ahead{};

    if (lane >= occupants.size() || lane >= lanes.lanes.size())
    {
        return answer;
    }

    const auto ownHalf = options.vehicle.bodyHalfExtentsMetres.z;
    const auto& list = occupants[lane];

    const auto consider = [&](const Occupant& occupant, const double at)
    {
        if (occupant.agent == ignore)
        {
            return false;
        }

        const auto gap = at - 0.5 * occupant.lengthMetres - distanceMetres - ownHalf;
        if (gap > options.lookAheadMetres)
        {
            return false;
        }

        answer.found = true;
        answer.gapMetres = std::max(gap, 0.0);
        answer.speedMetresPerSecond = occupant.speedMetresPerSecond;
        answer.siren = occupant.siren;

        return true;
    };

    const auto start = std::ranges::upper_bound(list, distanceMetres, {}, &Occupant::distanceMetres);

    for (auto entry = start; entry != list.end(); ++entry)
    {
        if (consider(*entry, entry->distanceMetres))
        {
            return answer;
        }
    }

    // Nothing left on this lane, so the road ahead is whatever this lane leads to. One hop, which is
    // all a look-ahead of a hundred and forty metres can reach on any lane this project carries.
    const auto& own = lanes.lanes[lane];
    const auto total = laneLength(own);

    if (own.loop)
    {
        for (const auto& occupant : list)
        {
            if (occupant.distanceMetres > distanceMetres)
            {
                break;
            }

            if (consider(occupant, occupant.distanceMetres + total))
            {
                return answer;
            }
        }

        return answer;
    }

    if (own.successors.empty())
    {
        return answer;
    }

    const auto& link = own.successors.front();
    if (link.lane >= occupants.size())
    {
        return answer;
    }

    for (const auto& occupant : occupants[link.lane])
    {
        if (occupant.distanceMetres < link.distanceMetres)
        {
            continue;
        }

        // Distances on two lanes are not the same measure, so the leader is placed at how far past
        // this lane's end it is.
        if (consider(occupant, total + (occupant.distanceMetres - link.distanceMetres)))
        {
            return answer;
        }
    }

    return answer;
}

TrafficPopulation::Ahead TrafficPopulation::followerOn(const std::size_t lane, const double distanceMetres,
                                                       const std::uint32_t ignore, const double reachMetres) const
{
    auto answer = Ahead{};

    if (lane >= occupants.size())
    {
        return answer;
    }

    const auto reach = reachMetres < 0.0 ? options.lookAheadMetres : reachMetres;
    const auto ownHalf = options.vehicle.bodyHalfExtentsMetres.z;
    const auto& list = occupants[lane];
    const auto start = std::ranges::lower_bound(list, distanceMetres, {}, &Occupant::distanceMetres);

    for (auto entry = start; entry != list.begin();)
    {
        --entry;

        if (entry->agent == ignore)
        {
            continue;
        }

        const auto gap = distanceMetres - ownHalf - entry->distanceMetres - 0.5 * entry->lengthMetres;
        if (gap > reach)
        {
            return answer;
        }

        answer.found = true;
        answer.gapMetres = std::max(gap, 0.0);
        answer.speedMetresPerSecond = entry->speedMetresPerSecond;

        return answer;
    }

    return answer;
}

TrafficPopulation::Ahead TrafficPopulation::signalAhead(const std::size_t lane, const double distanceMetres,
                                                        const double seconds) const
{
    auto answer = Ahead{};

    // Empty on every track this project carries — Grand City Parkway's export states no lights and
    // no intersections. The loop is here because the behaviour that reads it is stated, and a stated
    // behaviour with no code behind it is a claim rather than a feature.
    for (const auto& signal : lanes.signals)
    {
        if (signal.lane != lane || signal.distanceMetres <= distanceMetres)
        {
            continue;
        }

        const auto gap = signal.distanceMetres - distanceMetres - options.vehicle.bodyHalfExtentsMetres.z;
        if (gap > options.lookAheadMetres || (answer.found && gap >= answer.gapMetres))
        {
            continue;
        }

        const auto aspect = signalAspect(signal, seconds);
        if (aspect == SignalAspect::Green)
        {
            continue;
        }

        answer.found = true;
        answer.gapMetres = std::max(gap, 0.0);
        answer.speedMetresPerSecond = 0.0;
        answer.signal = true;
    }

    return answer;
}

TrafficPopulation::Ahead TrafficPopulation::laneEndAhead(const std::size_t lane, const double distanceMetres) const
{
    auto answer = Ahead{};

    if (lane >= lanes.lanes.size())
    {
        return answer;
    }

    const auto& own = lanes.lanes[lane];
    if (own.loop || !own.successors.empty())
    {
        return answer;
    }

    const auto gap = laneLength(own) - distanceMetres - options.vehicle.bodyHalfExtentsMetres.z;
    if (gap > options.lookAheadMetres)
    {
        return answer;
    }

    answer.found = true;
    answer.gapMetres = std::max(gap, 0.0);
    answer.speedMetresPerSecond = 0.0;

    return answer;
}

void TrafficPopulation::refreshCruisingPose(TrafficAgent& agent, const double travelMetres, const bool snapHeading)
{
    if (agent.lane >= lanes.lanes.size())
    {
        return;
    }

    const auto place = laneAt(lanes.lanes[agent.lane], agent.distanceMetres);
    if (!place)
    {
        return;
    }

    // Where the car is and which way the road runs there, on its own lane. The tangent and not the
    // chord's direction: the chord steps at every point of the polyline, and a car that followed it
    // snapped its heading at each one.
    auto position = place->positionMetres;
    auto direction = place->tangent;
    // Every sideways motion below is lateral metres **per metre of road covered**, and is turned into
    // a velocity by the speed the car actually has — so a car standing still has none of it, and
    // neither slides nor turns.
    auto crossingPerMetre = glm::dvec3(0.0);

    // A lane change is a blend between the point on the lane being left and the same place on the
    // lane being joined, both read fresh every tick. At the end of the change the blend *is* the
    // target lane's point, so handing the car to that lane moves it by nothing. The old pose was the
    // old lane's point displaced by the run's *mean* gap, and the handover landed wherever the two
    // lanes were not exactly the mean apart — which was the snap at the end of every change.
    if (agent.changeProgress > 0.0 && agent.changeTarget != agent.lane && agent.changeTarget < lanes.lanes.size())
    {
        const auto& other = lanes.lanes[agent.changeTarget];
        const auto target = laneAt(other, wrapDistance(other, agent.distanceMetres + agent.changeShiftMetres));
        if (target)
        {
            const auto weight = easeInOut(agent.changeProgress);
            const auto across = target->positionMetres - place->positionMetres;

            position = place->positionMetres + across * weight;
            direction = glm::normalize(glm::mix(place->tangent, target->tangent, weight));
            // Going out at the blend's rate; coming back, when the change is given up, at the jerk.
            const auto rate = agent.changeAborting ? -abortRateFactor : 1.0;
            crossingPerMetre = across * (rate * easeRate(agent.changeProgress) / std::max(agent.changeLengthMetres, 1.0));
        }
    }

    const auto side = laneSide(direction);

    // Whatever the car still carries from wherever it was put onto this lane — a change abandoned at
    // a join, a body that rejoined a little off centre — eased out over the road rather than dropped.
    // The rate matches the decay `updateCruising` applies, so the heading agrees with the motion.
    if (agent.settleLateralMetres != 0.0)
    {
        position += side * agent.settleLateralMetres;
        crossingPerMetre -= side * (agent.settleLateralMetres / settleLengthMetres);
    }

    // The yield's offset toward the kerb, and the crossing it is still making toward its target —
    // the same construction as the settle, eased in and out at the yield's own length of road
    // (docs/police-driving-brief.md §2.3).
    if (agent.yieldLateralMetres != 0.0 || agent.yieldTargetMetres != 0.0)
    {
        position += side * agent.yieldLateralMetres;
        crossingPerMetre += side * ((agent.yieldTargetMetres - agent.yieldLateralMetres) / options.yieldLengthMetres);
    }

    agent.positionMetres = position;
    agent.velocityMetresPerSecond = (direction + crossingPerMetre) * agent.speedMetresPerSecond;

    // Where the car wants to point: along its motion. A metre per second of reference speed along the
    // road, so a car standing still points down its lane; the crossing term carries the speed the
    // car actually has, so a car at rest is not turned by a change it has not yet moved on.
    const auto wanted = glm::normalize(direction * std::max(agent.speedMetresPerSecond, 1.0) +
                                       crossingPerMetre * agent.speedMetresPerSecond);

    // And it gets there at a rate a car can turn at, per metre covered, so a join in the polyline or
    // a rejoin from the body tier turns the car through the corner rather than snapping it round.
    agent.heading =
        snapHeading ? wanted : turnToward(agent.heading, wanted, maximumYawRadiansPerMetre * travelMetres);
    agent.orientation = orientationAlong(agent.heading);
}

void TrafficPopulation::settleOntoLane(TrafficAgent& agent)
{
    if (agent.lane >= lanes.lanes.size())
    {
        return;
    }

    const auto place = laneAt(lanes.lanes[agent.lane], agent.distanceMetres);
    if (!place)
    {
        return;
    }

    // Bounded, because a residual is something to ease out and not somewhere to drive from. Less
    // the yield's own offset, which the pose adds on its own account and would otherwise be counted
    // twice.
    agent.settleLateralMetres =
        std::clamp(glm::dot(agent.positionMetres - place->positionMetres, laneSide(place->tangent)) -
                       agent.yieldLateralMetres,
                   -6.0, 6.0);
}

void TrafficPopulation::considerLaneChange(TrafficAgent& agent, const double ownAcceleration)
{
    if (agent.lane >= lanes.lanes.size())
    {
        return;
    }

    const auto& own = lanes.lanes[agent.lane];
    if (!own.allowLaneChanges || own.neighbours.empty())
    {
        return;
    }

    const auto& behaviour = profiles[agent.profile].behaviour;

    // Whether the road is about to make this driver move. A cautious driver changes lanes for this
    // reason and for no other, which is the whole of what "does not change lanes unless they have
    // to" means — and it has to mean *something*, or a cautious car sits at a dead end for ever.
    const auto ending = laneEndAhead(agent.lane, agent.distanceMetres);
    const auto forced = ending.found && ending.gapMetres < 90.0;

    if (!behaviour.changesLanesToGetOn && !forced)
    {
        return;
    }

    // The urge to get past something slower, which published MOBIL does not have. Without it a
    // driver only moves over when the arithmetic of the next lane is better *right now*, and a car
    // sitting behind a slow one at a comfortable distance is not in any trouble at all.
    const auto held = agent.desiredSpeedMetresPerSecond - agent.speedMetresPerSecond > 2.0;

    auto bestScore = behaviour.changeThresholdMetresPerSecondSquared;
    const auto* chosen = static_cast<const LaneNeighbour*>(nullptr);
    auto chosenRun = std::size_t{0};

    for (auto runIndex = std::size_t{0}; runIndex < own.neighbours.size(); runIndex++)
    {
        const auto& neighbour = own.neighbours[runIndex];

        if (agent.distanceMetres < neighbour.fromMetres || agent.distanceMetres > neighbour.toMetres)
        {
            continue;
        }

        // Room to finish the change inside the run that offered it.
        if (neighbour.toMetres - agent.distanceMetres < minimumChangeRoomMetres)
        {
            continue;
        }

        // The same place on the neighbour, at this distance on this lane and brought onto it: past a
        // loop's origin the sum is past the loop's length.
        const auto target = wrapDistance(lanes.lanes[neighbour.lane],
                                         agent.distanceMetres + neighbourShiftAt(neighbour, agent.distanceMetres));

        const auto leader = leaderOn(neighbour.lane, target, agent.id);
        const auto follower = followerOn(neighbour.lane, target, agent.id);

        // A geometric floor before any of the model runs: two cars cannot be in the same place, and
        // an incentive that is happy about it is an incentive computed for a collision.
        if ((leader.found && leader.gapMetres < 2.0) || (follower.found && follower.gapMetres < 2.0))
        {
            continue;
        }

        // At the siren standoff behind a patrol car there, as the following model keeps it: judged at
        // the driver's own gap, a lane with the same patrol car across it read as a better lane.
        const auto leaderStandoff =
            leader.siren ? std::max(options.sirenStandoffMetres - behaviour.minimumGapMetres, 0.0) : 0.0;
        const auto ownNew =
            followAcceleration(behaviour, agent.speedMetresPerSecond, agent.desiredSpeedMetresPerSecond,
                               leader.found ? leader.gapMetres - leaderStandoff : std::numeric_limits<double>::max(),
                               leader.found ? agent.speedMetresPerSecond - leader.speedMetresPerSecond : 0.0);

        auto courtesy = 0.0;

        if (follower.found)
        {
            // **The follower is judged with this driver's own behaviour, not its own.** The occupant
            // list carries a place and a speed and deliberately not a profile: putting one there
            // would make every lane-change evaluation a lookup into another agent, and the answer
            // moves by a few per cent. It is a stated simplification of MOBIL rather than an
            // oversight.
            const auto behind = followerOn(neighbour.lane, target - follower.gapMetres - 0.1, agent.id);

            const auto before =
                followAcceleration(behaviour, follower.speedMetresPerSecond, agent.desiredSpeedMetresPerSecond,
                                   behind.found ? behind.gapMetres : std::numeric_limits<double>::max(), 0.0);

            const auto after =
                followAcceleration(behaviour, follower.speedMetresPerSecond, agent.desiredSpeedMetresPerSecond,
                                   follower.gapMetres, follower.speedMetresPerSecond - agent.speedMetresPerSecond);

            // The safety criterion, and it is the one number in the model that is not a preference:
            // nobody may be forced to brake harder than this to let somebody in.
            if (after < -behaviour.safeDecelerationMetresPerSecondSquared)
            {
                continue;
            }

            courtesy = behaviour.politeness * (after - before);
        }

        auto score = (ownNew - ownAcceleration) + courtesy;

        if (held && leader.found && ownNew > ownAcceleration)
        {
            score += behaviour.overtakeUrgeMetresPerSecondSquared;
        }

        if (forced)
        {
            // The road is running out. Whatever the arithmetic says about comfort, moving over beats
            // stopping in the carriageway.
            score += 4.0;
        }

        if (score <= bestScore)
        {
            continue;
        }

        bestScore = score;
        chosen = &neighbour;
        chosenRun = runIndex;
    }

    if (chosen == nullptr)
    {
        return;
    }

    agent.changeTarget = chosen->lane;
    agent.changeRun = chosenRun;
    agent.changeShiftMetres = neighbourShiftAt(*chosen, agent.distanceMetres);
    // How much road the change takes: this driver's own time at the speed they have now, floored at a
    // hard pull-out and capped so the change ends inside the run that offered it. A stopped car may
    // commit — it is how a car leaves a queue — and then covers the floor's dozen metres to get over.
    agent.changeLengthMetres =
        std::clamp(std::min(agent.speedMetresPerSecond * behaviour.changeDurationSeconds,
                            chosen->toMetres - agent.distanceMetres - 2.0),
                   minimumChangeLengthMetres, maximumChangeLengthMetres);
    // Started off zero rather than at zero, so the change is under way from the tick it was decided.
    agent.changeProgress = 1e-6;
}

void TrafficPopulation::updateCruising(TrafficAgent& agent, const TrafficUpdate& input, const bool decide)
{
    const auto& behaviour = profiles[agent.profile].behaviour;
    const auto deltaTime = input.deltaTimeSeconds;

    if (agent.changeCooldownSeconds > 0.0)
    {
        agent.changeCooldownSeconds -= deltaTime;
    }

    auto changing = agent.changeProgress > 0.0 && agent.changeTarget != agent.lane;

    // The shift onto the target at where the car is now. It moves along a bend — the outer lane is
    // the longer — and a shift read once at the start of the change put the same place metres ahead
    // or behind by the end of it.
    if (changing && agent.lane < lanes.lanes.size() && agent.changeRun < lanes.lanes[agent.lane].neighbours.size())
    {
        agent.changeShiftMetres =
            neighbourShiftAt(lanes.lanes[agent.lane].neighbours[agent.changeRun], agent.distanceMetres);

        // And where that puts the same place on the target against the car's own lane point. Past
        // the reach a neighbour can have, the run's shift is naming somewhere else on that lane, and
        // the change is given up where the car stands — what it had moved across eased out over the
        // road, as an abandoned change is — rather than blended toward across the map.
        const auto& other = lanes.lanes[agent.changeTarget];
        const auto here = laneAt(lanes.lanes[agent.lane], agent.distanceMetres);
        const auto there = laneAt(other, wrapDistance(other, agent.distanceMetres + agent.changeShiftMetres));

        if (!here || !there || glm::distance(here->positionMetres, there->positionMetres) > laneChangeReachMetres)
        {
            settleOntoLane(agent);
            agent.changeProgress = 0.0;
            agent.changeAborting = false;
            agent.changeLengthMetres = 0.0;
            agent.changeShiftMetres = 0.0;
            agent.changeTarget = agent.lane;
            agent.changeCooldownSeconds = behaviour.changeCooldownSeconds;
            changing = false;
        }
    }

    const auto nearer = [](Ahead& into, const Ahead& candidate)
    {
        if (candidate.found && (!into.found || candidate.gapMetres < into.gapMetres))
        {
            into = candidate;
        }
    };

    // --- the yield (docs/police-driving-brief.md §2.3) ------------------------------------------
    //
    // A siren behind, read on the decision clock; the hold counts down once it is gone, and the
    // sideways target goes back to the lane's centre with it.
    if (decide)
    {
        considerYield(agent);
    }

    if (agent.yieldSeconds > 0.0)
    {
        agent.yieldSeconds -= deltaTime;

        if (agent.yieldSeconds <= 0.0)
        {
            agent.yieldSeconds = 0.0;
            agent.yieldTargetMetres = 0.0;
        }
    }

    const auto yielding = agent.yieldSeconds > 0.0;
    const auto yieldDesired = std::max(options.yieldSpeedFloorMetresPerSecond,
                                       options.yieldSpeedFactor * lanes.lanes[agent.lane].speedLimitMetresPerSecond);

    // The car-following model's answer against one thing in the way, the gap opened by an allowance
    // where the standstill distance the model keeps is not the one that applies — and closed by the
    // siren standoff behind a patrol car, so the car stops short of it and leaves it room
    // (docs/police-driving-brief.md §9).
    const auto followAt = [&](const Ahead& ahead, const double gapAllowanceMetres, const double desired)
    {
        const auto standoff =
            ahead.siren ? std::max(options.sirenStandoffMetres - behaviour.minimumGapMetres, 0.0) : 0.0;
        const auto gap =
            ahead.found ? ahead.gapMetres + gapAllowanceMetres - standoff : std::numeric_limits<double>::max();
        const auto approach = ahead.found ? agent.speedMetresPerSecond - ahead.speedMetresPerSecond : 0.0;

        return followAcceleration(behaviour, agent.speedMetresPerSecond, desired, gap, approach);
    };

    // ...and the same under the yield: slowing for a siren is a comfortable slow-down and never
    // emergency braking — the model's free term at half its desired speed asks for many times the
    // comfortable rate — while whatever the car in front demands is kept whole.
    const auto follow = [&](const Ahead& ahead, const double gapAllowanceMetres)
    {
        const auto plain = followAt(ahead, gapAllowanceMetres, agent.desiredSpeedMetresPerSecond);

        if (!yielding || agent.desiredSpeedMetresPerSecond <= yieldDesired)
        {
            return plain;
        }

        const auto eased = followAt(ahead, gapAllowanceMetres, yieldDesired);

        return std::min(plain, std::max(eased, -behaviour.comfortableDecelerationMetresPerSecondSquared));
    };

    // --- the merge, re-checked (docs/police-driving-brief.md §7) ------------------------------
    //
    // A change was decided against the lists as they stood; the gap it is moving into can close
    // while it is under way — a car coming up fast behind on the target lane, the leader there
    // braking. Re-read on the decision clock while the car is still mostly in its own lane, and if
    // the gap is no longer safe the change is given up: the blend runs backwards at the jerk rate
    // until the car is back where it started.
    if (changing && decide && !agent.changeAborting && easeInOut(agent.changeProgress) < abortWeightLimit &&
        !changeStillSafe(agent, behaviour))
    {
        agent.changeAborting = true;
    }

    const auto committed = changing && !agent.changeAborting;

    // What is in the way, which is the nearest of three things: the car in front, a light, and the
    // end of the road. One lookup each and then the closest wins, so the car-following model has
    // exactly one leader however many kinds of thing could be it.
    auto constraint = Ahead{};
    auto acceleration = 0.0;

    if (!committed)
    {
        constraint = leaderOn(agent.lane, agent.distanceMetres, agent.id);
        nearer(constraint, signalAhead(agent.lane, agent.distanceMetres, input.simulatedSeconds));
        nearer(constraint, laneEndAhead(agent.lane, agent.distanceMetres));

        acceleration = follow(constraint, 0.0);

        // Jerking back out of a lane it was part way into: that lane's car still holds it, at the
        // pull-out gap, until it is mostly out.
        if (changing && easeInOut(agent.changeProgress) > ownLaneClearanceWeight)
        {
            const auto onTarget =
                wrapDistance(lanes.lanes[agent.changeTarget], agent.distanceMetres + agent.changeShiftMetres);
            const auto target = leaderOn(agent.changeTarget, onTarget, agent.id);

            acceleration = std::min(acceleration, follow(target, behaviour.minimumGapMetres - pullOutGapMetres));
        }
    }
    else
    {
        // **A car that has committed to a lane change follows the lane it is going to.** That lane's
        // car, light and end constrain it from the first tick, which is how a car gets out of a
        // stopped queue: the gap it is moving into is what it accelerates against, and the change —
        // paced by the road it covers, below — advances as it goes. Its own lane's car and light
        // still hold it while it is mostly in that lane, at the gap a car nosing out round the one in
        // front keeps rather than the one a queue keeps. Its own lane's *end* never holds it: leaving
        // that lane is the point, and braking for its end is what parked cars at dead ends.
        const auto onTarget =
            wrapDistance(lanes.lanes[agent.changeTarget], agent.distanceMetres + agent.changeShiftMetres);

        constraint = leaderOn(agent.changeTarget, onTarget, agent.id);
        nearer(constraint, signalAhead(agent.changeTarget, onTarget, input.simulatedSeconds));
        nearer(constraint, laneEndAhead(agent.changeTarget, onTarget));

        acceleration = follow(constraint, 0.0);

        if (easeInOut(agent.changeProgress) < ownLaneClearanceWeight)
        {
            auto own = leaderOn(agent.lane, agent.distanceMetres, agent.id);
            nearer(own, signalAhead(agent.lane, agent.distanceMetres, input.simulatedSeconds));

            acceleration = std::min(acceleration, follow(own, behaviour.minimumGapMetres - pullOutGapMetres));
        }
    }

    // **The reaction delay, and it is re-armed rather than triggered.** While the car is stopped and
    // something is in the way, the timer is held at the driver's delay; the moment the way clears it
    // stops being re-armed and counts down. So the delay is exactly the time between the road
    // clearing and this driver moving, which is what "slow off the line" means — and it needs one
    // field and no edge detection. Judged on the lane the car is following, so a car pulling out of
    // a queue is not held by the queue it is leaving.
    const auto blocked = constraint.found && constraint.gapMetres < behaviour.minimumGapMetres + 1.0;

    if (agent.speedMetresPerSecond < 0.08 && blocked)
    {
        agent.holdSeconds = constraint.signal ? behaviour.greenDelaySeconds : behaviour.startDelaySeconds;
    }
    else if (agent.holdSeconds > 0.0)
    {
        agent.holdSeconds -= deltaTime;
    }

    if (agent.holdSeconds > 0.0)
    {
        acceleration = std::min(acceleration, 0.0);
        agent.speedMetresPerSecond = 0.0;
    }

    agent.accelerationMetresPerSecondSquared = acceleration;
    agent.speedMetresPerSecond = std::max(0.0, agent.speedMetresPerSecond + acceleration * deltaTime);

    const auto travel = agent.speedMetresPerSecond * deltaTime;

    // --- the lane change, paced by the road covered and never by the clock ----------------------
    //
    // A change that ran on time slid a stopped car sideways into the next lane and turned it up to
    // forty-five degrees across the road while it did so. Over distance, a car that is not moving
    // does not change lanes, and one that is crosses at the angle its speed and the change's length
    // give it.
    if (changing && agent.changeAborting)
    {
        agent.changeProgress -= abortRateFactor * travel / std::max(agent.changeLengthMetres, 1.0);

        if (agent.changeProgress <= 0.0)
        {
            // Back in its own lane, exactly: the blend at zero weight is the lane's own point.
            agent.changeProgress = 0.0;
            agent.changeAborting = false;
            agent.changeLengthMetres = 0.0;
            agent.changeShiftMetres = 0.0;
            agent.changeTarget = agent.lane;
            agent.changeCooldownSeconds = behaviour.changeCooldownSeconds;
        }
    }
    else if (changing)
    {
        agent.changeProgress += travel / std::max(agent.changeLengthMetres, 1.0);

        if (agent.changeProgress >= 1.0)
        {
            const auto previousLimit = lanes.lanes[agent.lane].speedLimitMetresPerSecond;

            // The pose at full weight is the target lane's own point at this shifted distance, so
            // this moves the car by nothing.
            agent.lane = agent.changeTarget;
            agent.distanceMetres = wrapDistance(lanes.lanes[agent.lane], agent.distanceMetres + agent.changeShiftMetres);

            // The driver's own offset and spread ride across the change; only the lane's limit moves.
            agent.desiredSpeedMetresPerSecond =
                std::max(2.0, agent.desiredSpeedMetresPerSecond + lanes.lanes[agent.lane].speedLimitMetresPerSecond -
                                  previousLimit);

            agent.changeProgress = 0.0;
            agent.changeAborting = false;
            agent.changeLengthMetres = 0.0;
            agent.changeShiftMetres = 0.0;
            agent.changeTarget = agent.lane;
            agent.changeCooldownSeconds = behaviour.changeCooldownSeconds;
        }
    }
    else if (decide && agent.changeCooldownSeconds <= 0.0 && !yielding)
    {
        considerLaneChange(agent, acceleration);
    }

    // --- along the road ----------------------------------------------------------------------
    const auto step = advanceAlongLane(lanes, agent.lane, agent.distanceMetres, travel);

    if (step.ranOut)
    {
        // The road has run out and there is nowhere to go. The car stops at the end and the recycle
        // pass moves it once it is somewhere nobody is looking.
        agent.distanceMetres = step.distanceMetres;
        agent.speedMetresPerSecond = 0.0;
        agent.stillSeconds += deltaTime;
    }
    else
    {
        if (step.changedLane)
        {
            // The lane changed underneath the car, so a change that was in progress is against a
            // neighbour of a lane it is no longer on. Abandoned rather than carried, because its
            // shift means nothing on the new one.
            agent.changeProgress = 0.0;
            agent.changeAborting = false;
            agent.changeLengthMetres = 0.0;
            agent.changeShiftMetres = 0.0;
        }

        agent.lane = step.lane;
        agent.distanceMetres = step.distanceMetres;
        agent.changeTarget = agent.changeProgress > 0.0 ? agent.changeTarget : agent.lane;

        // A body's wheels roll by what the body does, in `updateBodies`; only a point rolls by its plan.
        if (agent.mode == AgentMode::Cruising)
        {
            agent.rolledMetres += travel;
        }
        agent.stillSeconds =
            agent.speedMetresPerSecond < options.stillSpeedMetresPerSecond ? agent.stillSeconds + deltaTime : 0.0;

        if (step.changedLane)
        {
            // Whatever the car carries onto the new lane is eased out over the road rather than
            // dropped on this tick: what was left of a crossing when the lane ran out under it, and
            // — at a join — the sideways gap between a lane's last point and the lane it runs into,
            // up to the junction radius, which snapped every merging car a lane's width in one tick
            // (2026-09-11, docs/traffic-system-brief.md §20). On a loop's own wrap the residual is
            // whatever the car already carried, read again off its pose.
            settleOntoLane(agent);
        }
    }

    // The residual eases out with the road, at the rate the pose's crossing term states it does.
    agent.settleLateralMetres *= std::exp(-travel / settleLengthMetres);
    if (std::abs(agent.settleLateralMetres) < 1e-3)
    {
        agent.settleLateralMetres = 0.0;
    }

    // And the yield's offset eases toward its target with the road, at its own length.
    if (agent.yieldLateralMetres != agent.yieldTargetMetres)
    {
        agent.yieldLateralMetres += (agent.yieldTargetMetres - agent.yieldLateralMetres) *
                                    (1.0 - std::exp(-travel / options.yieldLengthMetres));

        if (std::abs(agent.yieldLateralMetres - agent.yieldTargetMetres) < 1e-3)
        {
            agent.yieldLateralMetres = agent.yieldTargetMetres;
        }
    }

    refreshCruisingPose(agent, travel, false);
}

void TrafficPopulation::driveRecovery(TrafficAgent& agent, DisturbedSlot& slot, SimpleVehicleInput& into,
                                      const TrafficUpdate& input) const
{
    static_cast<void>(input);

    if (agent.mode == AgentMode::Stopped)
    {
        into.brake = 1.0;

        return;
    }

    const auto origin = simpleVehicleOrigin(options.vehicle, slot.state);
    const auto forward = slot.state.body.orientation * glm::dvec3(0.0, 0.0, 1.0);
    const auto left = slot.state.body.orientation * glm::dvec3(1.0, 0.0, 0.0);
    const auto along = glm::dot(slot.state.body.linearVelocity, forward);

    // Twelve metres of tolerance, which is three lanes: a car knocked into the next carriageway
    // should still know where the road is.
    const auto projection = projectOntoNetwork(lanes, origin, forward, 12.0);
    if (!projection.found)
    {
        // Off the network entirely — through a gap, up a kerb, into a car park. Stop, and let the
        // transition below decide it is finished.
        into.brake = 1.0;

        return;
    }

    const auto& lane = lanes.lanes[projection.lane];

    // Aim ahead rather than at the car's own foot, or the steering oscillates: the further the aim
    // point, the gentler the correction. Proportional to speed with a floor, which is the standard
    // pure-pursuit answer and needs no gains.
    const auto lookAhead = std::max(8.0, std::abs(along) * 1.2);
    const auto ahead = laneAt(lane, wrapDistance(lane, projection.distanceMetres + lookAhead));
    if (!ahead)
    {
        into.brake = 1.0;

        return;
    }

    auto aim = ahead->positionMetres;

    if (agent.mode == AgentMode::PullingOver)
    {
        // Over to the right, off the driving line. Not a shoulder — the network states none — but
        // out of the way of whatever is still coming.
        aim -= laneSide(ahead->direction) * 2.8;
    }

    const auto toAim = aim - origin;
    const auto angle = std::atan2(glm::dot(toAim, left), std::max(glm::dot(toAim, forward), 0.5));

    into.steer = std::clamp(angle / options.vehicle.maximumSteerRadians, -1.0, 1.0);

    // A car facing the wrong way does not reverse: it stops. Turning one round is a manoeuvre this
    // model has no room for, and a car that tries it in traffic is worse than one that gives up.
    if (glm::dot(forward, ahead->direction) < 0.2)
    {
        into.steer = 0.0;
        into.brake = 1.0;

        return;
    }

    const auto wanted = agent.mode == AgentMode::PullingOver
                            ? 0.0
                            : std::min(agent.desiredSpeedMetresPerSecond, lane.speedLimitMetresPerSecond);

    const auto error = wanted - along;

    into.throttle = std::clamp(error / 4.0, 0.0, 1.0);
    into.brake = std::clamp(-error / 4.0, 0.0, 1.0);
}

void TrafficPopulation::updateBodies(const TrafficUpdate& input, const PhysicsWorld& world)
{
    RACEENGINE_ZONE_N("traffic near bodies");

    rayOrigins.clear();
    rayDirections.clear();

    auto live = std::size_t{0};

    for (auto& slot : disturbed)
    {
        if (!slot.live)
        {
            continue;
        }

        simpleVehicleWheelRays(options.vehicle, slot.state, rayOrigins, rayDirections);
        live++;
    }

    // A city where nothing has been hit and nobody is near asks the world nothing at all.
    if (live == 0)
    {
        return;
    }

    world.castRays(rayOrigins, rayDirections, simpleVehicleRayLength(options.vehicle), rayHits);

    const auto box = simpleVehicleBox(options.vehicle);
    const auto deltaTime = input.deltaTimeSeconds;
    const auto nearTier = options.embodyRadiusMetres > 0.0 && !input.vehicles.empty();
    auto cursor = std::size_t{0};

    for (auto& slot : disturbed)
    {
        if (!slot.live)
        {
            continue;
        }

        auto& agent = pool[slot.agent];

        const auto hits = std::span<const SurfaceHit>(rayHits).subspan(cursor, simpleWheelCount);
        cursor += simpleWheelCount;

        // For a body following its plan, the plan is what `updateCruising` left in the pose fields
        // this tick. Read before the body overwrites them.
        const auto planPosition = agent.positionMetres;
        const auto planHeading = agent.heading;
        const auto planSpeed = agent.speedMetresPerSecond;

        auto driver = SimpleVehicleInput{};

        if (agent.mode == AgentMode::Embodied)
        {
            driveFollowingPlan(agent, slot, driver);
        }
        else if (agent.mode == AgentMode::Pursuing)
        {
            drivePursuit(agent, slot, driver);
        }
        else
        {
            driveRecovery(agent, slot, driver, input);
        }

        manifoldScratch = collideBody(world, slot.state.body, box);

        if (agent.mode == AgentMode::Embodied)
        {
            // The plan has taken the body somewhere the lane model cannot see — a chord through a
            // building, a lane under something low. Resolved, that depth is a launch; dropped, the
            // car is the point it was before the tier existed, its pose still the plan's, and it is
            // stood up again once the plan is somewhere a body can stand.
            auto deepest = 0.0;
            for (const auto& point : manifoldScratch.points)
            {
                deepest = std::max(deepest, point.penetration);
            }

            if (deepest > deepPenetrationMetres)
            {
                dropBody(agent);

                continue;
            }
        }

        // A pursuing body has the chase's forces and the same geometry; every other body is the
        // traffic car it always was.
        const auto pursuing = agent.mode == AgentMode::Pursuing;

        stepSimpleVehicle(pursuing ? pursuitVehicle : options.vehicle, slot.state, driver, hits, world.materials(),
                          manifoldScratch, options.contact, deltaTime);

        // Nothing in traffic reaches this. A body that has is being thrown by something the guards
        // above did not catch, and a car flying across the city is worse than one that goes back to
        // being a point where its plan is (a following body) or where the road nearest it is (a
        // wreck) — and it is counted, so the next report has a number. A pursuing body is allowed the
        // speed of a chase.
        const auto speedLimit = pursuing ? options.pursuitSpeedLimitMetresPerSecond : bodySpeedLimitMetresPerSecond;

        if (glm::length(slot.state.body.linearVelocity) > speedLimit)
        {
            if (agent.mode != AgentMode::Embodied)
            {
                const auto origin = simpleVehicleOrigin(options.vehicle, slot.state);
                const auto found = projectOntoNetwork(lanes, origin, glm::dvec3(0.0), 12.0);

                if (found.found)
                {
                    agent.positionMetres = origin;
                    agent.orientation = slot.state.body.orientation;
                    agent.velocityMetresPerSecond = glm::dvec3(0.0);
                    demote(agent, found, 0.0);
                    droppedTotal++;
                }
                else
                {
                    dropBody(agent);
                    recycle(agent, input.focusMetres,
                            input.vehicles.empty() ? glm::dvec3(0.0) : input.vehicles.front().forward);
                }
            }
            else
            {
                dropBody(agent);
            }

            continue;
        }

        agent.positionMetres = simpleVehicleOrigin(options.vehicle, slot.state);
        agent.orientation = slot.state.body.orientation;
        agent.velocityMetresPerSecond = slot.state.body.linearVelocity;
        // The body's forward is +z in its own frame, the same axis the tyre forces are resolved on.
        // Signed, so a car shoved backwards turns its wheels backwards, and a car sliding square
        // across the road turns them by nothing.
        const auto forward = agent.orientation * glm::dvec3(0.0, 0.0, 1.0);
        const auto alongSpeed = glm::dot(agent.velocityMetresPerSecond, forward);
        agent.rolledMetres += alongSpeed * deltaTime;

        // A body that is not following a plan has no heading but its own. The field is what the
        // pursuit director reads to tell an aim ahead of a unit from one behind it, and a chasing body
        // turns; a following body keeps the plan's, which `updateCruising` wrote this tick.
        if (agent.mode != AgentMode::Embodied)
        {
            agent.heading = forward;
        }

        const auto speed = glm::length(agent.velocityMetresPerSecond);
        agent.stillSeconds = speed < options.stillSpeedMetresPerSecond ? agent.stillSeconds + deltaTime : 0.0;

        const auto bodyUp = agent.orientation * glm::dvec3(0.0, 1.0, 0.0);
        const auto upright = glm::dot(bodyUp, worldUp) > 0.55;

        if (agent.mode == AgentMode::Embodied)
        {
            // How the body stands to its plan: behind it along the road, and off it to the side.
            const auto lag = glm::dot(planPosition - agent.positionMetres, planHeading);
            const auto aside = glm::dot(agent.positionMetres - planPosition, laneSide(planHeading));

            // Knocked, wedged or rolled too far from its plan to be following it. It is a wreck now,
            // and recovery brings it back the way it brings back any other — and hands it a plan
            // again once it has, below.
            if (!upright || std::abs(aside) > bodyLostLateralMetres || std::abs(lag) > bodyLostAlongMetres)
            {
                agent.mode = AgentMode::Disturbed;
                agent.disturbedSeconds = 0.0;
                agent.bodyLagMetres = 0.0;

                continue;
            }

            // The leash. A plan that gets ahead of a body that cannot keep up — held by something the
            // lane model cannot see — is pulled back to it, so the two never part company and the
            // car behind, which follows the plan, does not close on the metal.
            if (lag > planLeadLimitMetres)
            {
                agent.distanceMetres = std::max(0.0, agent.distanceMetres - (lag - planLeadLimitMetres));
                agent.speedMetresPerSecond = std::min(planSpeed, std::max(alongSpeed, 0.0) + 0.5);
            }

            agent.bodyLagMetres = std::clamp(lag, 0.0, planLeadLimitMetres);

            continue;
        }

        agent.disturbedSeconds += deltaTime;

        if (agent.mode == AgentMode::Stopped)
        {
            continue;
        }

        // On its roof, or on its side. There is nothing a driver can do from there.
        if (!upright && agent.stillSeconds > 1.5)
        {
            agent.mode = AgentMode::Stopped;
            agent.siren = false;

            continue;
        }

        // A chase has no recovery in it: the director decides when it is over, and until then the
        // body goes where it is aimed.
        if (agent.mode == AgentMode::Pursuing)
        {
            continue;
        }

        if (agent.mode == AgentMode::Disturbed)
        {
            const auto projection = projectOntoNetwork(lanes, agent.positionMetres, forward, 12.0);

            if (projection.found && upright && slot.state.groundedWheels == simpleWheelCount &&
                agent.disturbedSeconds > options.minimumDisturbedSeconds &&
                std::abs(projection.lateralMetres) < options.rejoinLateralMetres && speed > 0.5)
            {
                const auto place = laneAt(lanes.lanes[projection.lane], projection.distanceMetres);

                // Along the lane and not across it. See `rejoinLateralSpeedMetresPerSecond`: without
                // this the car rejoins on the tick it was pushed, because being pushed does not move
                // it until the next one.
                const auto crossing =
                    place ? std::abs(glm::dot(agent.velocityMetresPerSecond, laneSide(place->direction))) : 1e9;

                if (place && glm::dot(forward, place->direction) > options.rejoinHeadingAgreement &&
                    crossing < options.rejoinLateralSpeedMetresPerSecond)
                {
                    const auto along = glm::dot(agent.velocityMetresPerSecond, place->direction);

                    if (nearTier && glm::distance(agent.positionMetres, input.focusMetres) < options.embodyRadiusMetres)
                    {
                        // Back on its lane and still close to the player: it keeps its body and picks
                        // up a plan, rather than becoming a point that is given a body next tick.
                        reembody(agent, projection, along);
                    }
                    else
                    {
                        demote(agent, projection, along);
                    }

                    continue;
                }
            }

            if (agent.disturbedSeconds > options.recoverySeconds)
            {
                agent.mode = AgentMode::PullingOver;
            }

            continue;
        }

        if (agent.mode == AgentMode::PullingOver && agent.stillSeconds > options.stillSeconds)
        {
            agent.mode = AgentMode::Stopped;
        }
    }
}

std::size_t TrafficPopulation::freeBodySlot() const
{
    for (auto index = std::size_t{0}; index < disturbed.size(); index++)
    {
        if (!disturbed[index].live)
        {
            return index;
        }
    }

    return noSlot;
}

bool TrafficPopulation::promote(TrafficAgent& agent, const glm::dvec3& extraVelocity,
                                const glm::dvec3& extraAngularVelocity)
{
    const auto free = freeBodySlot();

    if (free == noSlot)
    {
        // The pool is full. The one thing worth evicting is a car that has been stopped a long time
        // somewhere nobody is standing, and that decision belongs to the recycle pass rather than
        // here — so this refuses, and the refusal is counted so the cap can be read off a log line
        // rather than guessed at.
        refusedSinceReport++;

        return false;
    }

    auto& slot = disturbed[free];

    slot.live = true;
    slot.agent = agent.id;

    placeSimpleVehicle(options.vehicle, slot.state, agent.positionMetres, agent.orientation,
                       agent.velocityMetresPerSecond + extraVelocity, extraAngularVelocity);

    agent.mode = AgentMode::Disturbed;
    agent.slot = free;
    agent.disturbedSeconds = 0.0;
    agent.stillSeconds = 0.0;
    agent.bodyLagMetres = 0.0;
    agent.changeProgress = 0.0;
    agent.changeAborting = false;
    agent.changeLengthMetres = 0.0;
    agent.changeShiftMetres = 0.0;
    agent.settleLateralMetres = 0.0;
    agent.holdSeconds = 0.0;

    promotedSinceReport++;

    return true;
}

bool TrafficPopulation::embody(TrafficAgent& agent, const PhysicsWorld& world)
{
    const auto free = freeBodySlot();

    if (free == noSlot)
    {
        // The pool is full; the car stays a point. Counted with the refusals, because it is one:
        // this car wanted a body and the cap said no.
        refusedSinceReport++;

        return false;
    }

    auto& slot = disturbed[free];

    // Stood up exactly on its plan, with the plan's velocity, so the first tick of the body is the
    // last tick of the point continued. `placeSimpleVehicle` puts the design position on the road,
    // so the springs start at their static compression and the car does not sag on arrival.
    placeSimpleVehicle(options.vehicle, slot.state, agent.positionMetres, agent.orientation,
                       agent.velocityMetresPerSecond, glm::dvec3(0.0));

    // Only where a body can stand: ground under all four wheels, and nothing of the world already
    // inside the box. A plan on a lane that grazes a kerb, or runs through a wall, stays a point
    // there — as it always was — and is tried again as it moves on.
    rayOrigins.clear();
    rayDirections.clear();
    simpleVehicleWheelRays(options.vehicle, slot.state, rayOrigins, rayDirections);
    world.castRays(rayOrigins, rayDirections, simpleVehicleRayLength(options.vehicle), rayHits);

    auto grounded = rayHits.size() == simpleWheelCount;
    for (const auto& hit : rayHits)
    {
        grounded = grounded && hit.hit;
    }

    manifoldScratch = collideBody(world, slot.state.body, simpleVehicleBox(options.vehicle));

    auto embedded = false;
    for (const auto& point : manifoldScratch.points)
    {
        embedded = embedded || point.penetration > spawnPenetrationLimitMetres;
    }

    if (!grounded || embedded)
    {
        slot = DisturbedSlot{};
        heldAsPointsTotal++;

        return false;
    }

    slot.live = true;
    slot.agent = agent.id;

    agent.mode = AgentMode::Embodied;
    agent.slot = free;
    agent.disturbedSeconds = 0.0;
    agent.bodyLagMetres = 0.0;

    return true;
}

void TrafficPopulation::disembody(TrafficAgent& agent)
{
    if (agent.slot == noSlot || !disturbed[agent.slot].live)
    {
        agent.mode = AgentMode::Cruising;
        agent.bodyLagMetres = 0.0;

        return;
    }

    const auto& state = disturbed[agent.slot].state;

    // The body's pose and not the plan's: `updateCruising` wrote the plan into these fields this
    // tick, and it is the body the point has to carry on from.
    agent.positionMetres = simpleVehicleOrigin(options.vehicle, state);
    agent.orientation = state.body.orientation;
    agent.velocityMetresPerSecond = state.body.linearVelocity;
    agent.bodyLagMetres = 0.0;

    const auto forward = agent.orientation * glm::dvec3(0.0, 0.0, 1.0);
    const auto upright = glm::dot(agent.orientation * glm::dvec3(0.0, 1.0, 0.0), worldUp) > 0.55;
    const auto projection = projectOntoNetwork(lanes, agent.positionMetres, forward, 6.0);
    const auto place =
        projection.found ? laneAt(lanes.lanes[projection.lane], projection.distanceMetres) : std::optional<LanePlace>{};

    if (place && upright && glm::dot(forward, place->direction) > options.rejoinHeadingAgreement)
    {
        // On its lane, pointing along it. The point picks up from the body — a change in progress is
        // dropped, because the projection has decided which lane it is on — and `demote` eases what
        // is left of the body's offset and heading out over the road.
        agent.changeProgress = 0.0;
        agent.changeAborting = false;
        agent.changeLengthMetres = 0.0;
        agent.changeShiftMetres = 0.0;

        demote(agent, projection, glm::dot(agent.velocityMetresPerSecond, place->direction));

        return;
    }

    // Off its lane, or across it, at the moment it went out of range: it is a wreck, and recovery
    // brings it back the way it brings back any other.
    agent.mode = AgentMode::Disturbed;
    agent.disturbedSeconds = 0.0;
    agent.stillSeconds = 0.0;
}

void TrafficPopulation::reembody(TrafficAgent& agent, const LaneProjection& projection, const double speed)
{
    const auto previousLimit = agent.lane < lanes.lanes.size() ? lanes.lanes[agent.lane].speedLimitMetresPerSecond
                                                               : lanes.lanes[projection.lane].speedLimitMetresPerSecond;

    // The body stays where it is; the plan starts on the lane beside it and the body steers onto it.
    agent.mode = AgentMode::Embodied;
    agent.lane = projection.lane;
    agent.distanceMetres = projection.distanceMetres;
    agent.changeTarget = projection.lane;
    agent.changeProgress = 0.0;
    agent.changeAborting = false;
    agent.changeLengthMetres = 0.0;
    agent.changeShiftMetres = 0.0;
    agent.settleLateralMetres = 0.0;
    agent.heading = agent.orientation * glm::dvec3(0.0, 0.0, 1.0);
    agent.speedMetresPerSecond = std::max(0.0, speed);
    agent.disturbedSeconds = 0.0;
    agent.stillSeconds = 0.0;
    agent.bodyLagMetres = 0.0;

    agent.desiredSpeedMetresPerSecond =
        std::max(2.0, agent.desiredSpeedMetresPerSecond + lanes.lanes[projection.lane].speedLimitMetresPerSecond -
                          previousLimit);
}

void TrafficPopulation::dropBody(TrafficAgent& agent)
{
    if (agent.slot != noSlot && agent.slot < disturbed.size())
    {
        disturbed[agent.slot] = DisturbedSlot{};
    }

    agent.slot = noSlot;
    agent.mode = AgentMode::Cruising;
    agent.bodyLagMetres = 0.0;
    agent.disturbedSeconds = 0.0;
    droppedTotal++;
}

void TrafficPopulation::embodyNear(const TrafficUpdate& input, const PhysicsWorld& world)
{
    if (options.embodyRadiusMetres <= 0.0 || input.vehicles.empty())
    {
        return;
    }

    const auto in = options.embodyRadiusMetres;
    const auto out = std::max(options.disembodyRadiusMetres, in + 8.0);

    for (auto& agent : pool)
    {
        // For a body this is the plan's position, which is within the leash of the body's.
        const auto distance = glm::distance(agent.positionMetres, input.focusMetres);

        if (agent.mode == AgentMode::Cruising && distance < in)
        {
            static_cast<void>(embody(agent, world));
        }
        else if (agent.mode == AgentMode::Embodied && distance > out)
        {
            disembody(agent);
        }
    }
}

std::optional<glm::dvec3> TrafficPopulation::plannedPosition(const TrafficAgent& agent, const double aheadMetres) const
{
    if (agent.lane >= lanes.lanes.size())
    {
        return std::nullopt;
    }

    const auto& own = lanes.lanes[agent.lane];
    const auto place = laneAt(own, wrapDistance(own, agent.distanceMetres + aheadMetres));
    if (!place)
    {
        return std::nullopt;
    }

    // The same construction as `refreshCruisingPose`, further along: the change's blend at the
    // progress it will have there, and the settle at what will be left of it.
    auto position = place->positionMetres;
    auto direction = place->tangent;

    if (agent.changeProgress > 0.0 && agent.changeTarget != agent.lane && agent.changeTarget < lanes.lanes.size())
    {
        const auto& other = lanes.lanes[agent.changeTarget];
        const auto target =
            laneAt(other, wrapDistance(other, agent.distanceMetres + agent.changeShiftMetres + aheadMetres));
        if (target)
        {
            const auto rate = agent.changeAborting ? -abortRateFactor : 1.0;
            const auto weight =
                easeInOut(agent.changeProgress + rate * aheadMetres / std::max(agent.changeLengthMetres, 1.0));

            position = place->positionMetres + (target->positionMetres - place->positionMetres) * weight;
            direction = glm::normalize(glm::mix(place->tangent, target->tangent, weight));
        }
    }

    if (agent.settleLateralMetres != 0.0)
    {
        position += laneSide(direction) * (agent.settleLateralMetres * std::exp(-aheadMetres / settleLengthMetres));
    }

    if (agent.yieldLateralMetres != 0.0 || agent.yieldTargetMetres != 0.0)
    {
        const auto eased = agent.yieldLateralMetres + (agent.yieldTargetMetres - agent.yieldLateralMetres) *
                                                          (1.0 - std::exp(-aheadMetres / options.yieldLengthMetres));

        position += laneSide(direction) * eased;
    }

    return position;
}

void TrafficPopulation::driveFollowingPlan(const TrafficAgent& agent, const DisturbedSlot& slot,
                                           SimpleVehicleInput& into) const
{
    const auto origin = simpleVehicleOrigin(options.vehicle, slot.state);
    const auto forward = slot.state.body.orientation * glm::dvec3(0.0, 0.0, 1.0);
    const auto left = slot.state.body.orientation * glm::dvec3(1.0, 0.0, 0.0);
    const auto along = glm::dot(slot.state.body.linearVelocity, forward);

    // Pure pursuit on the plan's own path, a few metres ahead of where the plan is. On the path and
    // not down the plan's tangent: the tangent leaves a curve, and a car aimed along it runs wide by
    // the lookahead squared over twice the radius.
    const auto lookAhead = std::max(followLookAheadMetres, followLookAheadSeconds * std::abs(along));
    const auto aim = plannedPosition(agent, lookAhead).value_or(agent.positionMetres + agent.heading * lookAhead);
    const auto toAim = aim - origin;
    const auto angle = std::atan2(glm::dot(toAim, left), std::max(glm::dot(toAim, forward), 0.5));

    into.steer = std::clamp(angle / options.vehicle.maximumSteerRadians, -1.0, 1.0);

    // Speed: the plan's, plus enough to close whatever the body has fallen behind by. A gain of
    // about one per second is a driver keeping station, not a servo.
    const auto behind = glm::dot(agent.positionMetres - origin, forward);
    const auto wanted = std::max(0.0, agent.speedMetresPerSecond + followGapGainPerSecond * behind);
    const auto error = wanted - along;

    into.throttle = std::clamp(error / 3.0, 0.0, 1.0);
    into.brake = std::clamp(-error / 3.0, 0.0, 1.0);

    // A plan that is standing still holds the body on the brake, so it neither creeps on the
    // rolling-resistance floor nor rolls on a camber.
    if (agent.speedMetresPerSecond < 0.05 && std::abs(along) < 0.4 && behind < 0.5)
    {
        into.throttle = 0.0;
        into.brake = 1.0;
    }
}

void TrafficPopulation::demote(TrafficAgent& agent, const LaneProjection& projection, const double speed)
{
    if (agent.slot != noSlot)
    {
        disturbed[agent.slot] = DisturbedSlot{};
    }

    const auto previousLimit = agent.lane < lanes.lanes.size() ? lanes.lanes[agent.lane].speedLimitMetresPerSecond
                                                               : lanes.lanes[projection.lane].speedLimitMetresPerSecond;

    agent.slot = noSlot;
    agent.mode = AgentMode::Cruising;
    agent.lane = projection.lane;
    agent.distanceMetres = projection.distanceMetres;
    agent.changeTarget = projection.lane;
    agent.speedMetresPerSecond = std::max(0.0, speed);
    agent.disturbedSeconds = 0.0;
    agent.stillSeconds = 0.0;
    agent.bodyLagMetres = 0.0;

    agent.desiredSpeedMetresPerSecond =
        std::max(2.0, agent.desiredSpeedMetresPerSecond + lanes.lanes[projection.lane].speedLimitMetresPerSecond -
                          previousLimit);

    // The body came back a little off the lane's centre and a little off its heading — inside the
    // rejoin tolerances, which is what let it back. Both are eased out over the road from where the
    // body actually is, rather than corrected on this tick.
    agent.heading = agent.orientation * glm::dvec3(0.0, 0.0, 1.0);
    settleOntoLane(agent);
    refreshCruisingPose(agent, 0.0, false);
}

void TrafficPopulation::recycle(TrafficAgent& agent, const glm::dvec3& focusMetres, const glm::dvec3& focusForward)
{
    if (agent.slot != noSlot)
    {
        disturbed[agent.slot] = DisturbedSlot{};
        agent.slot = noSlot;
    }

    if (placeOnNetwork(agent, focusMetres, focusForward, true))
    {
        summary.recycled++;
    }
}

void TrafficPopulation::obstaclesNear(const glm::dvec3& pointMetres, const double radiusMetres,
                                      std::vector<DynamicObstacle>& into) const
{
    const auto reach = radiusMetres * radiusMetres;
    const auto inverseMass = options.vehicle.massKilograms > 0.0 ? 1.0 / options.vehicle.massKilograms : 0.0;

    const auto full = 2.0 * options.vehicle.bodyHalfExtentsMetres;
    const auto mass = options.vehicle.massKilograms;

    // The box's own inertia, inverted once. Every traffic car is the same box, so this is one
    // three-by-three inverse for the whole city rather than one per obstacle.
    const auto inertia = glm::dmat3(1.0 / 12.0 * mass * (full.y * full.y + full.z * full.z), 0.0, 0.0, 0.0,
                                    1.0 / 12.0 * mass * (full.x * full.x + full.z * full.z), 0.0, 0.0, 0.0,
                                    1.0 / 12.0 * mass * (full.x * full.x + full.y * full.y));
    const auto inverseInertiaBody = glm::inverse(inertia);

    for (const auto& agent : pool)
    {
        // The caller owns an external car's body and offers it with its own mass; listing it here
        // as well would collide the player against the same car twice.
        if (agent.mode == AgentMode::External)
        {
            continue;
        }

        const auto offset = agent.positionMetres - pointMetres;
        if (glm::dot(offset, offset) > reach)
        {
            continue;
        }

        const auto rotation = glm::mat3_cast(agent.orientation);

        auto spin = glm::dvec3(0.0);
        if (agent.slot != noSlot && disturbed[agent.slot].live)
        {
            spin = angularVelocity(disturbed[agent.slot].state.body);
        }

        into.push_back(DynamicObstacle{
            .id = agent.id,
            .centre =
                agent.positionMetres + agent.orientation * glm::dvec3(0.0, options.vehicle.bodyCentreHeightMetres, 0.0),
            .orientation = agent.orientation,
            .halfExtents = options.vehicle.bodyHalfExtentsMetres,
            .centreOfMass = agent.positionMetres +
                            agent.orientation * glm::dvec3(0.0, options.vehicle.centreOfMassHeightMetres, 0.0),
            .linearVelocity = agent.velocityMetresPerSecond,
            .angularVelocity = spin,
            .inverseMass = inverseMass,
            // Into the world frame, and it is the only per-obstacle matrix work here.
            .inverseInertia = rotation * inverseInertiaBody * glm::transpose(rotation)});
    }
}

void TrafficPopulation::chargeImpact(const std::uint32_t id, const double metresPerSecond)
{
    if (id >= pool.size() || metresPerSecond <= 0.0)
    {
        return;
    }

    auto& agent = pool[id];

    // An external car's ledger is the caller's own (`PursuitDirector::reportUnitImpact`).
    if (agent.mode == AgentMode::External)
    {
        return;
    }

    agent.impactMetresPerSecond += metresPerSecond;
}

void TrafficPopulation::applyContacts(const ContactManifold& manifold, const ImpactLedger ledger)
{
    for (const auto& body : manifold.bodies)
    {
        if (body.obstacle == noObstacle)
        {
            continue;
        }

        const auto index = static_cast<std::size_t>(body.obstacle);
        if (index >= pool.size())
        {
            continue;
        }

        auto& agent = pool[index];

        // An external car's body is the caller's and so is whatever hit it: the caller's own solve
        // has already moved it, and the velocity here is against a box this module is not stepping.
        if (agent.mode == AgentMode::External)
        {
            continue;
        }

        // A mirror clipped in passing is not a crash. Below the threshold the nudge is dropped
        // outright rather than accumulated, because accumulating it would leave a trail of disturbed
        // cars behind anyone who uses the whole width of their lane.
        const auto impact = glm::length(body.deltaLinear);
        if (impact < options.impactSpeedThresholdMetresPerSecond)
        {
            continue;
        }

        // What the bodywork took, for a damage model that reads it off the agent — unless the caller
        // is keeping that ledger from the closing speed, in which case the solver's number, which is
        // mostly the push-out on any tick the boxes overlap, is applied and not counted. Charged
        // here it wrote a patrol car off for touching a Charger (2026-09-11).
        if (ledger == ImpactLedger::FromSolve)
        {
            agent.impactMetresPerSecond += impact;
        }

        if (agent.slot != noSlot && disturbed[agent.slot].live)
        {
            // Already a body. The solve's answer is added to what it is already doing, which is what
            // makes a second hit on a car that is still sliding do something.
            auto& body_ = disturbed[agent.slot].state.body;

            body_.linearVelocity += body.deltaLinear;
            setAngularVelocity(body_, angularVelocity(body_) + body.deltaAngular);

            // A body that was following its plan is not following it any more; a stopped wreck that
            // has been hit again is moving again. Both are recovery's now. A pursuing car takes the
            // hit and keeps chasing — being rammed is what a chase is.
            const auto wasFollowing = agent.mode == AgentMode::Embodied || agent.mode == AgentMode::Stopped;

            agent.mode = wasFollowing ? AgentMode::Disturbed : agent.mode;
            agent.disturbedSeconds = agent.mode == AgentMode::Disturbed ? 0.0 : agent.disturbedSeconds;
            agent.bodyLagMetres = 0.0;

            continue;
        }

        static_cast<void>(promote(agent, body.deltaLinear, body.deltaAngular));
    }
}

const TrafficReport& TrafficPopulation::update(const TrafficUpdate& input, const PhysicsWorld& world)
{
    RACEENGINE_ZONE_N("traffic population update");

    summary = TrafficReport{};
    summary.agents = pool.size();
    summary.promoted = promotedSinceReport;
    summary.refused = refusedSinceReport;
    promotedSinceReport = 0;
    refusedSinceReport = 0;

    if (pool.empty() || input.deltaTimeSeconds <= 0.0)
    {
        return summary;
    }

    // The decision clock. The car-following model runs every tick because it is arithmetic on
    // numbers that are already in hand; the lookups it runs against — the per-lane ordering, the
    // projections, the lane-change evaluation — run thirty times a second, which is finer than a
    // driver decides at and a twelfth of the work.
    decisionRemainder += input.deltaTimeSeconds;

    const auto decide = decisionRemainder >= options.decisionIntervalSeconds;
    if (decide)
    {
        decisionRemainder = 0.0;
        rebuildLaneOccupants(input);
    }

    // The plan, for every car that has one: a point on its lane and a body following its lane alike.
    for (auto& agent : pool)
    {
        if (agent.mode == AgentMode::Cruising || agent.mode == AgentMode::Embodied)
        {
            updateCruising(agent, input, decide);
        }
    }

    embodyNear(input, world);
    updateBodies(input, world);

    // Which way the player is looking, near enough: the first outside vehicle is theirs. Zero when
    // there is none, which makes every direction "behind" and leaves the distance test alone.
    const auto focusForward = input.vehicles.empty() ? glm::dvec3(0.0) : input.vehicles.front().forward;

    for (auto& agent : pool)
    {
        switch (agent.mode)
        {
        case AgentMode::Cruising:
        case AgentMode::Embodied:
            (agent.mode == AgentMode::Cruising ? summary.cruising : summary.embodied)++;
            if (agent.changeProgress > 0.0)
            {
                summary.changingLanes++;
            }
            break;
        case AgentMode::Disturbed:
        case AgentMode::PullingOver:
            summary.disturbed++;
            break;
        case AgentMode::Stopped:
            summary.stopped++;
            break;
        case AgentMode::Pursuing:
            summary.pursuing++;
            break;
        case AgentMode::External:
            summary.external++;
            break;
        }

        // A car that has been standing still a long time somewhere nobody can see is put back into
        // circulation. **Only somewhere nobody can see** — taken from out of sight and put down out
        // of sight, which is the whole of why this is not visible teleportation. A car in a chase is
        // never standing still by this rule's meaning, and never recycled from under its director.
        const auto chasing = agent.mode == AgentMode::Pursuing || agent.mode == AgentMode::External;

        if (!chasing && agent.stillSeconds > options.recycleAfterSeconds &&
            outOfSight(agent.positionMetres, input.focusMetres, focusForward))
        {
            recycle(agent, input.focusMetres, focusForward);
        }
    }

    summary.heldAsPoints = heldAsPointsTotal;
    summary.dropped = droppedTotal;

    return summary;
}

// --- the police (docs/police-pursuit-brief.md) -------------------------------------------------

void TrafficPopulation::drivePursuit(const TrafficAgent& agent, const DisturbedSlot& slot, SimpleVehicleInput& into) const
{
    const auto origin = simpleVehicleOrigin(options.vehicle, slot.state);
    const auto forward = slot.state.body.orientation * glm::dvec3(0.0, 0.0, 1.0);
    const auto left = slot.state.body.orientation * glm::dvec3(1.0, 0.0, 0.0);
    const auto along = glm::dot(slot.state.body.linearVelocity, forward);

    // The aim, flat: a chase is decided on the road and a body's own pitch must not steer it.
    auto toAim = agent.pursuitAimMetres - origin;
    toAim.y = 0.0;
    const auto distance = glm::length(toAim);
    const auto ahead = glm::dot(toAim, forward);
    const auto aside = glm::dot(toAim, left);

    if (agent.pursuitReverse)
    {
        // Backing up, on the director's word. Toward an aim behind the car it is reverse pure
        // pursuit, and the wheel turns the way it would going forward — the rear follows the front
        // wheels round the same circle. Away from something the car cannot get past going forward,
        // the wheel is turned *away* from the aim, so the nose swings toward it as the car backs off
        // and the next attempt starts pointed the right way.
        // Backing through a three-point turn, the wheel is on the opposite lock to the turn — a left
        // turn backs with the wheel right — so the nose keeps swinging round (docs/police-driving-brief.md
        // §10). A positive steer is a left turn here.
        const auto angle = agent.pursuitTurnSide != 0
                               ? -static_cast<double>(agent.pursuitTurnSide) * options.vehicle.maximumSteerRadians
                           : ahead < 0.0 ? std::atan2(aside, std::max(-ahead, 0.5))
                                         : -std::copysign(0.7 * options.vehicle.maximumSteerRadians, aside);
        into.steer = std::clamp(angle / options.vehicle.maximumSteerRadians, -1.0, 1.0);
        into.reverse = true;

        // Still rolling forward: the brakes first. Then back up at the speed asked for.
        if (along > 0.5)
        {
            into.throttle = 0.0;
            into.brake = 1.0;

            return;
        }

        const auto error = agent.pursuitSpeedMetresPerSecond + along;
        into.throttle = std::clamp(error / 2.0, 0.0, 1.0);
        into.brake = std::clamp(-error / 2.0, 0.0, 1.0);

        return;
    }

    // Facing away from the aim at speed: a straight-line stop before the turn. Full lock at chase
    // speed is a spin, and a spun patrol car is a wreck the director then writes off.
    if (ahead < 0.0 && along > 6.0)
    {
        into.steer = 0.0;
        into.throttle = 0.0;
        into.brake = 1.0;

        return;
    }

    // Pure pursuit at the aim, through the wheelbase: the circle through the car tangent to its
    // heading and through the aim is a curvature, and the curvature is a road wheel angle. `atan2`
    // against a floored forward component, so an aim behind the car asks for full lock toward it
    // rather than for nothing. The first cut steered the *bearing* of the aim straight onto the road
    // wheel, which for an aim a look-ahead down the road asked for a radius a third of the distance
    // to it, and the cornering cap below then held the car to half the speed it was asked for.
    const auto angle = distance < 0.3 ? 0.0 : std::atan2(aside, std::max(ahead, 0.5));
    const auto pursuitCurvature = 2.0 * std::sin(angle) / std::max(distance, 4.0);
    const auto wheelbase = std::max(options.vehicle.wheelbaseMetres, 1.0);
    const auto lateralLimit = 0.85 * pursuitVehicle.frictionCoefficient * 9.80665;

    // The turn-in the director read off the route, on top of the pure pursuit
    // (docs/pursuit-navigation-brief.md, stage 3), zero off a route — the whole limited to the
    // curvature the tyre holds at this speed, so a lateral error at speed is corrected at the
    // tyre's limit and no harder, and the speed is not touched for it (docs/police-driving-brief.md §9).
    const auto tyreCurvature = lateralLimit / std::max(along * along, 1.0);
    const auto held = std::clamp(pursuitCurvature + agent.pursuitCurvatureAhead, -tyreCurvature, tyreCurvature);
    into.steer = std::clamp(std::atan(wheelbase * held) / options.vehicle.maximumSteerRadians, -1.0, 1.0);

    // Turning round with the aim behind: full lock toward the side the director chose — where the
    // arc fits — rather than toward the aim (docs/police-driving-brief.md §14). Positive is left.
    if (ahead < 0.0 && agent.pursuitTurnSide != 0)
    {
        into.steer = static_cast<double>(agent.pursuitTurnSide);
    }

    // The speed the director asked for, capped for a corner in the path — the turn-in, and the aim's
    // own bearing only when it is well off the nose: a body that tried to take full lock at 30 m/s
    // would spin, and a spun patrol car is a wreck the director then has to write off; a small
    // error's demand is not a corner.
    auto pathCurvature = std::abs(agent.pursuitCurvatureAhead);
    if (std::abs(angle) > options.pursuitSteeringCapAngleRadians)
    {
        pathCurvature = std::max(pathCurvature, std::abs(pursuitCurvature));
    }

    const auto corneringCap =
        pathCurvature > 1e-4 ? std::sqrt(lateralLimit / pathCurvature) : options.pursuitSpeedLimitMetresPerSecond;

    const auto wanted = std::clamp(agent.pursuitSpeedMetresPerSecond, 0.0, corneringCap);
    const auto error = wanted - along;

    // The brake is the deceleration the wanted speed asks for over the reaction horizon it is read
    // at (the director's 0.3 s), as a share of what the body can do — a feedforward, so the body
    // meets a corner at the profile's speed and not a few metres a second over it, which ran every
    // bend wide (docs/police-driving-brief.md §12). The throttle stays proportional.
    into.throttle = std::clamp(error / 3.0, 0.0, 1.0);
    into.brake = 0.0;

    if (error < 0.0)
    {
        const auto horizon = std::max(0.3 * along, 2.0);
        const auto needed = (along * along - wanted * wanted) / (2.0 * horizon);
        const auto braking = std::max(pursuitVehicle.maximumBrakeForceNewtons / std::max(pursuitVehicle.massKilograms, 1.0), 1.0);

        into.brake = std::clamp(std::max(needed / braking, -error / 3.0), 0.0, 1.0);
    }

    // An aim behind the car and close: this is where the director wants the car to be and it has
    // overshot. Stop rather than circle.
    if (ahead < 0.0 && distance < 6.0)
    {
        into.throttle = 0.0;
        into.brake = 1.0;
    }
}

bool TrafficPopulation::standUpBody(TrafficAgent& agent, const PhysicsWorld& world)
{
    const auto free = freeBodySlot();
    if (free == noSlot)
    {
        refusedSinceReport++;

        return false;
    }

    auto& slot = disturbed[free];

    placeSimpleVehicle(options.vehicle, slot.state, agent.positionMetres, agent.orientation,
                       agent.velocityMetresPerSecond, glm::dvec3(0.0));

    rayOrigins.clear();
    rayDirections.clear();
    simpleVehicleWheelRays(options.vehicle, slot.state, rayOrigins, rayDirections);
    world.castRays(rayOrigins, rayDirections, simpleVehicleRayLength(options.vehicle), rayHits);

    auto grounded = rayHits.size() == simpleWheelCount;
    for (const auto& hit : rayHits)
    {
        grounded = grounded && hit.hit;
    }

    manifoldScratch = collideBody(world, slot.state.body, simpleVehicleBox(options.vehicle));

    auto embedded = false;
    for (const auto& point : manifoldScratch.points)
    {
        embedded = embedded || point.penetration > spawnPenetrationLimitMetres;
    }

    if (!grounded || embedded)
    {
        slot = DisturbedSlot{};
        heldAsPointsTotal++;

        return false;
    }

    slot.live = true;
    slot.agent = agent.id;
    agent.slot = free;
    agent.disturbedSeconds = 0.0;
    agent.stillSeconds = 0.0;
    agent.bodyLagMetres = 0.0;
    agent.changeProgress = 0.0;
    agent.changeAborting = false;
    agent.changeLengthMetres = 0.0;
    agent.changeShiftMetres = 0.0;
    agent.settleLateralMetres = 0.0;
    agent.holdSeconds = 0.0;

    return true;
}

bool TrafficPopulation::beginPursuit(const std::uint32_t id, const PhysicsWorld& world)
{
    if (id >= pool.size() || !pool[id].police)
    {
        return false;
    }

    auto& agent = pool[id];

    // A car written off by a chase stays written off. Without this a wreck stood upright and stopped
    // is recruited again on the tick it was written off, with its damage forgotten.
    if (agent.wrecked)
    {
        return false;
    }

    switch (agent.mode)
    {
    case AgentMode::Pursuing:
    case AgentMode::External:
        return true;
    case AgentMode::Stopped:
        // A wreck on its roof does not join a chase; one that is upright and merely parked does.
        if (glm::dot(agent.orientation * glm::dvec3(0.0, 1.0, 0.0), worldUp) < 0.55)
        {
            return false;
        }
        [[fallthrough]];
    case AgentMode::Embodied:
    case AgentMode::Disturbed:
    case AgentMode::PullingOver:
        // Already a body: it keeps it and changes job.
        if (agent.slot != noSlot && disturbed[agent.slot].live)
        {
            agent.mode = AgentMode::Pursuing;
            agent.siren = true;
            agent.pursuitAimMetres = agent.positionMetres;

            return true;
        }
        [[fallthrough]];
    case AgentMode::Cruising:
        break;
    }

    if (!standUpBody(agent, world))
    {
        return false;
    }

    agent.mode = AgentMode::Pursuing;
    agent.siren = true;
    agent.pursuitAimMetres = agent.positionMetres;
    promotedSinceReport++;

    return true;
}

void TrafficPopulation::setPursuitAim(const std::uint32_t id, const glm::dvec3& aimMetres,
                                      const double speedMetresPerSecond, const bool siren, const bool reverse,
                                      const double curvatureAhead, const int turnSide)
{
    if (id >= pool.size())
    {
        return;
    }

    auto& agent = pool[id];
    if (agent.mode != AgentMode::Pursuing && agent.mode != AgentMode::External)
    {
        return;
    }

    agent.pursuitAimMetres = aimMetres;
    agent.pursuitSpeedMetresPerSecond = std::max(0.0, speedMetresPerSecond);
    agent.pursuitReverse = reverse;
    agent.pursuitCurvatureAhead = curvatureAhead;
    agent.pursuitTurnSide = turnSide;
    agent.siren = siren;
}

void TrafficPopulation::endPursuit(const std::uint32_t id, const bool wrecked)
{
    if (id >= pool.size())
    {
        return;
    }

    auto& agent = pool[id];
    if (agent.mode != AgentMode::Pursuing && agent.mode != AgentMode::External)
    {
        return;
    }

    agent.siren = false;
    agent.pursuitSpeedMetresPerSecond = 0.0;
    agent.pursuitReverse = false;

    // An external car has no body here. It is left as a point exactly where the caller last put it,
    // and the recovery driver is not asked to drive a car it cannot step: the next `update` sees a
    // cruising car off its lane and the ordinary rules take it from there — or a wreck that stays
    // where it is.
    const auto hasBody = agent.slot != noSlot && disturbed[agent.slot].live;

    // A stopped car with no body is a parked point: not stepped, still an obstacle and still an
    // occupant where it stands, and recycled by the ordinary rule once nobody can see it. That is
    // what a wreck the caller has finished with becomes, and what a car nowhere near a lane becomes.
    const auto park = [&]
    {
        agent.mode = AgentMode::Stopped;
        agent.velocityMetresPerSecond = glm::dvec3(0.0);
        agent.stillSeconds = hasBody ? 0.0 : options.recycleAfterSeconds + 1.0;
    };

    if (wrecked)
    {
        agent.wrecked = true;
        park();

        return;
    }

    if (hasBody)
    {
        agent.mode = AgentMode::Disturbed;
        agent.disturbedSeconds = 0.0;
        agent.stillSeconds = 0.0;

        return;
    }

    const auto forward = agent.orientation * glm::dvec3(0.0, 0.0, 1.0);
    const auto projection = projectOntoNetwork(lanes, agent.positionMetres, forward, 12.0);
    if (projection.found)
    {
        demote(agent, projection, glm::dot(agent.velocityMetresPerSecond, forward));

        return;
    }

    park();
}

bool TrafficPopulation::takeExternal(const std::uint32_t id)
{
    if (id >= pool.size() || pool[id].mode != AgentMode::Pursuing)
    {
        return false;
    }

    auto& agent = pool[id];

    // The pose fields already carry the body's own pose from the last tick; the slot is simply let
    // go, and the caller stands its model up on what is in the agent.
    if (agent.slot != noSlot && agent.slot < disturbed.size())
    {
        disturbed[agent.slot] = DisturbedSlot{};
    }

    agent.slot = noSlot;
    agent.mode = AgentMode::External;
    agent.bodyLagMetres = 0.0;

    return true;
}

void TrafficPopulation::setExternalPose(const std::uint32_t id, const glm::dvec3& positionMetres,
                                        const glm::dquat& orientation, const glm::dvec3& velocityMetresPerSecond,
                                        const double rolledMetres)
{
    if (id >= pool.size() || pool[id].mode != AgentMode::External)
    {
        return;
    }

    auto& agent = pool[id];
    agent.positionMetres = positionMetres;
    agent.orientation = orientation;
    agent.velocityMetresPerSecond = velocityMetresPerSecond;
    agent.heading = orientation * glm::dvec3(0.0, 0.0, 1.0);
    agent.rolledMetres = rolledMetres;
    agent.stillSeconds = 0.0;
}

void TrafficPopulation::releaseExternal(const std::uint32_t id, const PhysicsWorld& world)
{
    if (id >= pool.size() || pool[id].mode != AgentMode::External)
    {
        return;
    }

    auto& agent = pool[id];

    if (standUpBody(agent, world))
    {
        agent.mode = AgentMode::Pursuing;

        return;
    }

    // No body to be had where the caller left it. It ends its chase as a point on the network, the
    // way `endPursuit` ends one for a car with no body.
    agent.mode = AgentMode::Pursuing;
    endPursuit(id, false);
}

double TrafficPopulation::takeImpact(const std::uint32_t id)
{
    if (id >= pool.size())
    {
        return 0.0;
    }

    const auto taken = pool[id].impactMetresPerSecond;
    pool[id].impactMetresPerSecond = 0.0;

    return taken;
}

std::span<const std::uint32_t> TrafficPopulation::policeIds() const
{
    return police;
}

} // namespace raceengine
