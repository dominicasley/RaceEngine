// The police drive with their eyes open (docs/police-driving-brief.md): the corridor in front of a
// unit — the world by rays, the traffic by the occupant lists — and the traffic's yield to a siren.
//
// **Every fixture is a plate with straight lanes built in code**, on the police tests' own rule: the
// rules being tested are the ones most likely to be wrong on a map nobody has looked at.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::AgentMode;
using raceengine::behaviourFor;
using raceengine::bringUpJolt;
using raceengine::buildLaneNetwork;
using raceengine::ConvexCollider;
using raceengine::DriverArchetype;
using raceengine::DriverProfile;
using raceengine::generateProvingGround;
using raceengine::laneAt;
using raceengine::LaneNetwork;
using raceengine::laneSide;
using raceengine::LaneSource;
using raceengine::neighbourShiftAt;
using raceengine::noPoliceBody;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::PursuitDirector;
using raceengine::PursuitOptions;
using raceengine::PursuitPlayer;
using raceengine::PursuitRouter;
using raceengine::PursuitUnit;
using raceengine::tearDownJolt;
using raceengine::TrafficAgent;
using raceengine::TrafficPopulation;
using raceengine::TrafficPopulationOptions;
using raceengine::TrafficUpdate;
using raceengine::TrafficVehicle;

using Catch::Matchers::WithinAbs;

namespace
{

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    ~JoltGuard()
    {
        tearDownJolt();
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard(JoltGuard&&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;
    JoltGuard& operator=(JoltGuard&&) = delete;
};

constexpr auto tick = 1.0 / 360.0;

// A plate, with whatever stands on it: a box in the road is a prop's worth of world.
[[nodiscard]] PhysicsWorld plate(const double length, const double width, const std::vector<ConvexCollider>& boxes)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());

    auto world = PhysicsWorld::create(ground.value(), boxes);
    REQUIRE(world.has_value());

    return std::move(world).value();
}

[[nodiscard]] ConvexCollider box(const double x0, const double x1, const double z0, const double z1, const double height)
{
    auto hull = ConvexCollider{};
    for (const auto x : {x0, x1})
    {
        for (const auto y : {0.0, height})
        {
            for (const auto z : {z0, z1})
            {
                hull.points.push_back(glm::dvec3(x, y, z));
            }
        }
    }

    return hull;
}

[[nodiscard]] std::vector<glm::dvec3> straightLine(const double x, const double fromZ, const double toZ)
{
    auto line = std::vector<glm::dvec3>();
    const auto count = static_cast<int>((toZ - fromZ) / 18.0);

    for (auto index = 0; index <= count; index++)
    {
        line.push_back(glm::dvec3(x, 0.0, fromZ + (toZ - fromZ) * static_cast<double>(index) / static_cast<double>(count)));
    }

    return line;
}

[[nodiscard]] LaneSource lane(const int id, std::vector<glm::dvec3> points, const double limit)
{
    return LaneSource{.id = id,
                      .name = "lane",
                      .speedLimitMetresPerSecond = limit,
                      .allowLaneChanges = true,
                      .allowUTurns = false,
                      .points = std::move(points)};
}

// One lane down the middle of the plate.
[[nodiscard]] LaneNetwork oneLane(const double length, const double limit)
{
    auto built = buildLaneNetwork({lane(1, straightLine(0.0, 20.0, length - 20.0), limit)});
    REQUIRE(built.has_value());

    return std::move(built).value();
}

// Two lanes running +z, the second 3.7 m to +x — the first lane's LEFT — and starting 50 m up the
// road, so the pairing's shift is not zero.
[[nodiscard]] LaneNetwork twoLanes(const double length, const double limit)
{
    auto built = buildLaneNetwork(
        {lane(1, straightLine(0.0, 20.0, length - 20.0), limit), lane(2, straightLine(3.7, 70.0, length - 20.0), limit)});
    REQUIRE(built.has_value());

    return std::move(built).value();
}

// A city: a stated share of it patrol cars (none at zero), the rest regular drivers.
[[nodiscard]] TrafficPopulation city(LaneNetwork network, const double density, const double policeShare)
{
    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = density;
    options.bodyCount = 2;
    options.policeBody = policeShare > 0.0 ? std::uint8_t{1} : noPoliceBody;
    options.policeShare = policeShare;
    options.recoverySeconds = 30.0;

    auto built = TrafficPopulation(
        std::move(network), options,
        {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    built.seed();

    REQUIRE_FALSE(built.agents().empty());

    return built;
}

[[nodiscard]] PursuitPlayer playerAt(const glm::dvec3& position, const double speed)
{
    return PursuitPlayer{.positionMetres = position,
                         .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, speed),
                         .forward = glm::dvec3(0.0, 0.0, 1.0),
                         .lengthMetres = 4.3,
                         .widthMetres = 1.8};
}

[[nodiscard]] TrafficVehicle vehicleFor(const PursuitPlayer& player)
{
    return TrafficVehicle{.positionMetres = player.positionMetres,
                          .velocityMetresPerSecond = player.velocityMetresPerSecond,
                          .forward = player.forward,
                          .lengthMetres = player.lengthMetres};
}

// Tick the city and the director together, the player held where the caller says.
void run(PursuitDirector& director, TrafficPopulation& town, const PhysicsWorld& world, const PursuitPlayer& player,
         const double seconds, double& clock)
{
    const auto steps = static_cast<int>(seconds / tick);
    const auto vehicles = std::array{vehicleFor(player)};

    for (auto index = 0; index < steps; index++)
    {
        town.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                  .simulatedSeconds = clock,
                                  .focusMetres = player.positionMetres,
                                  .vehicles = vehicles},
                    world);
        director.update(tick, player, town, world);
        clock += tick;
    }
}

// Tick the city alone, with whatever outside vehicles the caller holds on the road.
void runCity(TrafficPopulation& town, const PhysicsWorld& world, const std::vector<TrafficVehicle>& vehicles,
             const double seconds, double& clock)
{
    const auto steps = static_cast<int>(seconds / tick);

    for (auto index = 0; index < steps; index++)
    {
        town.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                  .simulatedSeconds = clock,
                                  .focusMetres = glm::dvec3(0.0),
                                  .vehicles = vehicles},
                    world);
        clock += tick;
    }
}

[[nodiscard]] const TrafficAgent* frontCar(const TrafficPopulation& town)
{
    const TrafficAgent* front = nullptr;

    for (const auto& agent : town.agents())
    {
        if (front == nullptr || agent.positionMetres.z > front->positionMetres.z)
        {
            front = &agent;
        }
    }

    return front;
}

[[nodiscard]] const PursuitUnit* nearestUnit(const PursuitDirector& director)
{
    const PursuitUnit* nearest = nullptr;

    for (const auto& unit : director.units())
    {
        if (nearest == nullptr || unit.distanceMetres < nearest->distanceMetres)
        {
            nearest = &unit;
        }
    }

    return nearest;
}

// A cruising car with a clear stretch of its own lane behind it, so a query there reads that car.
[[nodiscard]] const TrafficAgent* carWithRoomBehind(const TrafficPopulation& town, const double fromMetres,
                                                    const double toMetres, const double roomMetres)
{
    for (const auto& agent : town.agents())
    {
        if (agent.mode != AgentMode::Cruising || agent.distanceMetres < fromMetres || agent.distanceMetres > toMetres)
        {
            continue;
        }

        auto clear = true;
        for (const auto& other : town.agents())
        {
            if (&other != &agent && other.lane == agent.lane && other.distanceMetres < agent.distanceMetres &&
                other.distanceMetres > agent.distanceMetres - roomMetres)
            {
                clear = false;
            }
        }

        if (clear)
        {
            return &agent;
        }
    }

    return nullptr;
}

} // namespace

TEST_CASE("the corridor: a box in the road in front of a unit is seen by the rays, caps its speed and sends it round",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    // A city of patrol cars on one lane; the front one has nothing but the player ahead of it.
    auto town = city(oneLane(700.0, 14.0), 3.0, 1.0);
    const auto* front = frontCar(town);
    REQUIRE(front != nullptr);
    const auto z = front->positionMetres.z;

    // The player thirty metres up the road, speeding: seen, an offence, a chase. And a box a metre
    // and a half beyond the player, a metre wide and a metre tall, in the road: the line of drive
    // ends at the player and is clear of it; the corridor's rays, pointed at where the player will be
    // in a second, run straight into it.
    const auto player = playerAt(glm::dvec3(0.0, 0.0, z + 30.0), 28.0);
    const auto withBox = plate(700.0, 60.0, {box(-0.5, 0.5, z + 31.5, z + 32.5, 1.0)});
    const auto without = plate(700.0, 60.0, {});

    auto router = PursuitRouter(town.network(), nullptr);

    auto director = PursuitDirector{};
    director.setRouter(&router);
    auto clock = 0.0;

    run(director, town, withBox, player, 1.5, clock);

    REQUIRE(director.status().active);
    const auto* unit = nearestUnit(director);
    REQUIRE(unit != nullptr);

    const auto& options = director.settings();

    // Seen, inside the reach and inside the route distance; the speed capped to stop short of it,
    // and the unit on a route round it.
    REQUIRE(unit->lineClear);
    REQUIRE(unit->corridorHit);
    REQUIRE(unit->corridorHitMetres < unit->corridorReachMetres);
    REQUIRE(unit->corridorBlocked);
    REQUIRE(unit->corridorCapMetresPerSecond < options.maximumUnitSpeedMetresPerSecond);
    REQUIRE(unit->wantedSpeedMetresPerSecond <= unit->corridorCapMetresPerSecond + 1e-9);
    REQUIRE(unit->routing);

    // The cap is the stopping distance at the unit's share of its braking, less the margin.
    const auto braking = options.corridorBrakeShare * options.cheapBodyBrakingMetresPerSecondSquared;
    const auto expected =
        std::sqrt(2.0 * braking * std::max(unit->corridorHitMetres - options.corridorStopMarginMetres, 0.0));
    REQUIRE_THAT(unit->corridorCapMetresPerSecond, WithinAbs(expected, 1e-9));

    // The control: the same chase on the bare plate sees nothing, caps nothing.
    auto control = city(oneLane(700.0, 14.0), 3.0, 1.0);
    auto controlRouter = PursuitRouter(control.network(), nullptr);
    auto controlDirector = PursuitDirector{};
    controlDirector.setRouter(&controlRouter);
    auto controlClock = 0.0;

    run(controlDirector, control, without, player, 1.5, controlClock);

    REQUIRE(controlDirector.status().active);
    const auto* clear = nearestUnit(controlDirector);
    REQUIRE(clear != nullptr);
    REQUIRE_FALSE(clear->corridorHit);
    REQUIRE_FALSE(clear->corridorBlocked);
    REQUIRE_THAT(clear->corridorCapMetresPerSecond, WithinAbs(options.maximumUnitSpeedMetresPerSecond, 1e-9));
}

TEST_CASE("the corridor's traffic: the lane under a point, the car ahead on it, and the neighbour either way round",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(twoLanes(700.0, 14.0), 8.0, 0.0);
    auto clock = 0.0;

    // Past the first decision tick, so the occupant lists stand.
    runCity(town, world, {}, 0.2, clock);

    // A car on the stretch where the two lanes run side by side, with fifteen metres of its own lane
    // clear behind it.
    const auto* car = carWithRoomBehind(town, 90.0, 560.0, 15.0);
    REQUIRE(car != nullptr);

    const auto& network = town.network();
    const auto& own = network.lanes[car->lane];
    const auto behind = laneAt(own, car->distanceMetres - 12.0);
    REQUIRE(behind.has_value());

    const auto length = 2.0 * town.settings().vehicle.bodyHalfExtentsMetres.z;

    // The right way round: the car is ahead on the lane's centre line, at the gap the occupant lists
    // give it, at its speed.
    const auto forward = town.trafficAround(behind->positionMetres, glm::dvec3(0.0, 0.0, 1.0), 7.0, 60.0);

    REQUIRE(forward.onRoad);
    REQUIRE(forward.lane == car->lane);
    REQUIRE_FALSE(forward.wrongWay);
    REQUIRE(forward.ahead.found);
    REQUIRE_THAT(forward.ahead.gapMetres, WithinAbs(12.0 - length, 0.3));
    REQUIRE_THAT(forward.ahead.speedMetresPerSecond, WithinAbs(car->speedMetresPerSecond, 1e-6));

    // Four lines: the centre, the neighbour's centre a lane's width over on the side the network
    // states, the seam between them at half that, and a seam's width out on the other side.
    const auto side = own.neighbours.front().side;
    REQUIRE(forward.lineCount == 4);
    REQUIRE(forward.lines[0].lane == car->lane);
    REQUIRE(forward.lines[0].side == 0);
    REQUIRE_THAT(forward.lines[0].offsetMetres, WithinAbs(0.0, 0.1));

    const auto* neighbourCentre = static_cast<const raceengine::TrafficLine*>(nullptr);
    const auto* seam = static_cast<const raceengine::TrafficLine*>(nullptr);
    const auto* far = static_cast<const raceengine::TrafficLine*>(nullptr);
    for (auto index = std::size_t{1}; index < forward.lineCount; index++)
    {
        const auto& line = forward.lines[index];
        if (line.lane == own.neighbours.front().lane && line.side == 0)
        {
            neighbourCentre = &line;
        }
        else if (line.side != 0 && line.offsetMetres * side > 0.0)
        {
            seam = &line;
        }
        else
        {
            far = &line;
        }
    }
    REQUIRE(neighbourCentre != nullptr);
    REQUIRE(seam != nullptr);
    REQUIRE(far != nullptr);
    REQUIRE_THAT(neighbourCentre->offsetMetres, WithinAbs(3.7 * side, 0.1));
    REQUIRE_THAT(seam->offsetMetres, WithinAbs(1.85 * side, 0.1));
    REQUIRE_THAT(far->offsetMetres, WithinAbs(-1.85 * side, 0.1));

    // The car on the centre blocks the centre and both seams beside it, not the neighbour's centre.
    REQUIRE(seam->ahead.found);
    REQUIRE(far->ahead.found);
    REQUIRE_THAT(seam->ahead.gapMetres, WithinAbs(forward.ahead.gapMetres, 1e-9));

    // The wrong way round: the same place read against the lane, the sides the other hand.
    const auto reversed = town.trafficAround(behind->positionMetres, glm::dvec3(0.0, 0.0, -1.0), 7.0, 60.0);

    REQUIRE(reversed.onRoad);
    REQUIRE(reversed.lane == car->lane);
    REQUIRE(reversed.wrongWay);
    REQUIRE(reversed.lineCount == 4);
    for (auto index = std::size_t{1}; index < reversed.lineCount; index++)
    {
        const auto& line = reversed.lines[index];
        if (line.lane == own.neighbours.front().lane && line.side == 0)
        {
            REQUIRE_THAT(line.offsetMetres, WithinAbs(-3.7 * side, 0.1));
        }
    }
    // What is ahead the wrong way is what the lane calls behind, coming the other way.
    if (reversed.ahead.found)
    {
        REQUIRE(reversed.ahead.speedMetresPerSecond <= 0.0);
    }
}

TEST_CASE("threading: a car that has pulled aside blocks its lane's centre and not the seam on its other side",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    // One lane, all traffic, and a siren car behind one of them so it yields — a car's width and a
    // half toward the kerb (the stated side, the right, with no neighbour).
    const auto world = plate(700.0, 60.0, {});
    auto town = city(oneLane(700.0, 14.0), 6.0, 0.0);
    auto clock = 0.0;

    runCity(town, world, {}, 0.2, clock);

    const auto* car = carWithRoomBehind(town, 60.0, 640.0, 30.0);
    REQUIRE(car != nullptr);
    const auto id = car->id;

    const auto& network = town.network();
    const auto& own = network.lanes[car->lane];
    const auto behind = laneAt(own, car->distanceMetres - 25.0);
    REQUIRE(behind.has_value());
    const auto siren = std::vector{TrafficVehicle{.positionMetres = behind->positionMetres,
                                                  .velocityMetresPerSecond = behind->direction * 30.0,
                                                  .forward = behind->direction,
                                                  .lengthMetres = 5.0,
                                                  .siren = true}};

    runCity(town, world, siren, 2.5, clock);

    const auto& yielded = town.agents()[id];
    REQUIRE(yielded.yieldLateralMetres < -1.2);

    // Read from twelve metres behind it: the centre is blocked by it, the seam on its right (where
    // it went) is blocked by it, and the seam on its left is clear.
    const auto point = laneAt(own, yielded.distanceMetres - 12.0);
    REQUIRE(point.has_value());
    const auto corridor = town.trafficAround(point->positionMetres, glm::dvec3(0.0, 0.0, 1.0), 7.0, 60.0);

    REQUIRE(corridor.onRoad);
    REQUIRE(corridor.ahead.found);
    REQUIRE(corridor.lineCount == 3);

    for (auto index = std::size_t{1}; index < corridor.lineCount; index++)
    {
        const auto& line = corridor.lines[index];
        if (line.side > 0)
        {
            REQUIRE_FALSE(line.ahead.found);
        }
        else
        {
            REQUIRE(line.ahead.found);
        }
    }
}

TEST_CASE("the corridor's traffic on a chase: a unit with a car ahead is capped to the safe approach, never past it",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    // A busy two-lane road, a quarter of it patrol cars, the player speeding up it sixty metres ahead
    // of the rearmost one — which chases up the road through everything between the two.
    const auto world = plate(700.0, 60.0, {});
    auto town = city(twoLanes(700.0, 14.0), 24.0, 0.25);
    REQUIRE_FALSE(town.policeIds().empty());

    auto director = PursuitDirector{};
    auto clock = 0.0;

    const TrafficAgent* rearmost = nullptr;
    for (const auto id : town.policeIds())
    {
        const auto& candidate = town.agents()[id];
        if (rearmost == nullptr || candidate.positionMetres.z < rearmost->positionMetres.z)
        {
            rearmost = &candidate;
        }
    }
    REQUIRE(rearmost != nullptr);

    const auto player = playerAt(glm::dvec3(0.0, 0.0, rearmost->positionMetres.z + 60.0), 28.0);

    const auto& options = director.settings();
    auto sawTraffic = false;
    auto shifted = false;

    for (auto step = 0; step < 4 * 360; step++)
    {
        run(director, town, world, player, tick, clock);

        for (const auto& unit : director.units())
        {
            REQUIRE(unit.corridorCapMetresPerSecond <= options.maximumUnitSpeedMetresPerSecond + 1e-9);

            if (!unit.trafficAhead)
            {
                continue;
            }

            sawTraffic = true;
            shifted = shifted || unit.laneShifting;

            const auto braking = options.corridorBrakeShare *
                                 (unit.fullModel ? options.fullModelBrakingMetresPerSecondSquared
                                                 : options.cheapBodyBrakingMetresPerSecondSquared);
            const auto safe = std::max(0.0, unit.trafficSpeedMetresPerSecond +
                                                std::sqrt(2.0 * braking *
                                                          std::max(unit.trafficGapMetres - options.followStandstillMetres,
                                                                   0.0)));

            REQUIRE(unit.corridorCapMetresPerSecond <= safe + 1e-9);
            REQUIRE(unit.wantedSpeedMetresPerSecond <= unit.corridorCapMetresPerSecond + 1e-9);

            // A shift is onto a lane beside the unit, a lane's width or so away.
            if (unit.laneShifting)
            {
                REQUIRE(std::abs(unit.laneShiftMetres) > 1.0);
                REQUIRE(std::abs(unit.laneShiftMetres) < 8.0);
            }
        }
    }

    REQUIRE(director.status().active);
    REQUIRE(sawTraffic);
    static_cast<void>(shifted);
}

TEST_CASE("the yield: a cruising car with a siren behind it slows, pulls to the kerb, and comes back when it has gone",
          "[traffic][traffic-yield]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(twoLanes(700.0, 14.0), 4.0, 0.0);
    auto clock = 0.0;

    runCity(town, world, {}, 0.2, clock);

    // A car on the side-by-side stretch, with room behind it for the siren car.
    const auto* car = carWithRoomBehind(town, 120.0, 500.0, 50.0);
    REQUIRE(car != nullptr);
    const auto id = car->id;

    const auto& network = town.network();
    const auto& own = network.lanes[car->lane];
    const auto desired = car->desiredSpeedMetresPerSecond;

    // The kerb is the side with no neighbour: this lane's one neighbour is on `side`, so the car
    // pulls the other way.
    REQUIRE(own.neighbours.size() == 1);
    const auto kerbSign = -own.neighbours.front().side;

    // A patrol car forty metres behind, siren on, coming up at thirty.
    const auto behind = laneAt(own, car->distanceMetres - 40.0);
    REQUIRE(behind.has_value());
    const auto siren = std::vector{TrafficVehicle{.positionMetres = behind->positionMetres,
                                                  .velocityMetresPerSecond = behind->direction * 30.0,
                                                  .forward = behind->direction,
                                                  .lengthMetres = 5.0,
                                                  .siren = true}};

    runCity(town, world, siren, 1.0, clock);

    const auto& yielding = town.agents()[id];
    REQUIRE(yielding.mode == AgentMode::Cruising);
    REQUIRE(yielding.yieldSeconds > 0.0);
    REQUIRE_THAT(yielding.yieldTargetMetres, WithinAbs(kerbSign * town.settings().yieldLateralMetres, 1e-12));
    REQUIRE(kerbSign * yielding.yieldLateralMetres > 0.5);
    REQUIRE(yielding.speedMetresPerSecond < desired - 1.0);

    // The pose carries the offset: the car is that far off its lane's centre, toward the kerb.
    const auto centre = laneAt(network.lanes[yielding.lane], yielding.distanceMetres);
    REQUIRE(centre.has_value());
    const auto aside = glm::dot(yielding.positionMetres - centre->positionMetres, laneSide(centre->tangent));
    REQUIRE_THAT(aside, WithinAbs(yielding.yieldLateralMetres, 0.05));

    // The siren gone: the hold runs out and the car eases back onto its lane, and back up toward its
    // own speed at the model's own brisk rate (which tapers as it gets there).
    runCity(town, world, {}, 8.0, clock);

    const auto& back = town.agents()[id];
    REQUIRE(back.yieldSeconds == 0.0);
    REQUIRE_THAT(back.yieldTargetMetres, WithinAbs(0.0, 1e-12));
    REQUIRE(std::abs(back.yieldLateralMetres) < 0.2);
    REQUIRE(back.speedMetresPerSecond > desired - 2.0);
}

TEST_CASE("the merge, re-checked: a car part way into a lane with a fast car coming up it jerks back into its own",
          "[traffic][traffic-merge]")
{
    const JoltGuard jolt;

    // Two lanes, all traffic, and every driver keen to get on: the first car to start a change onto
    // the other lane is the fixture. A patrol car is then put on that lane, coming up at sixty.
    const auto world = plate(700.0, 60.0, {});
    auto town = city(twoLanes(700.0, 14.0), 10.0, 0.0);
    auto clock = 0.0;

    const TrafficAgent* changing = nullptr;
    for (auto step = 0; step < 30 * 360 && changing == nullptr; step++)
    {
        runCity(town, world, {}, tick, clock);

        for (const auto& agent : town.agents())
        {
            if (agent.mode == AgentMode::Cruising && agent.changeProgress > 0.0 && agent.changeProgress < 0.15 &&
                agent.changeTarget != agent.lane && agent.distanceMetres > 100.0 && agent.distanceMetres < 500.0)
            {
                changing = &agent;
                break;
            }
        }
    }
    REQUIRE(changing != nullptr);
    const auto id = changing->id;
    const auto ownLane = changing->lane;
    const auto targetLane = changing->changeTarget;

    // The fast car forty metres behind the same place on the target lane, siren off — traffic reads
    // any outside vehicle the same way — at sixty metres a second.
    const auto& network = town.network();
    const auto& target = network.lanes[targetLane];
    const auto onTarget = raceengine::wrapDistance(target, changing->distanceMetres + changing->changeShiftMetres);
    const auto place = laneAt(target, onTarget - 40.0);
    REQUIRE(place.has_value());
    const auto fast = std::vector{TrafficVehicle{.positionMetres = place->positionMetres,
                                                 .velocityMetresPerSecond = place->direction * 60.0,
                                                 .forward = place->direction,
                                                 .lengthMetres = 4.5}};

    // The jerk back is decided on the next decision tick and, this early in the change, is over in
    // a tenth of a second — so it is watched for tick by tick.
    auto sawAbort = false;
    auto sawTarget = false;
    for (auto step = 0; step < 108; step++)
    {
        runCity(town, world, fast, tick, clock);

        const auto& watched = town.agents()[id];
        sawAbort = sawAbort || watched.changeAborting;
        sawTarget = sawTarget || watched.lane == targetLane;
    }
    REQUIRE(sawAbort);
    REQUIRE_FALSE(sawTarget);

    // Back in its own lane within a couple of seconds, the change forgotten, and on cooldown.
    runCity(town, world, fast, 2.2, clock);

    const auto& back = town.agents()[id];
    REQUIRE(back.lane == ownLane);
    REQUIRE_FALSE(back.changeAborting);
    REQUIRE(back.changeProgress == 0.0);
    REQUIRE(back.changeTarget == ownLane);
    REQUIRE(back.changeCooldownSeconds > 0.0);

    const auto centre = laneAt(network.lanes[back.lane], back.distanceMetres);
    REQUIRE(centre.has_value());
    const auto aside = glm::dot(back.positionMetres - centre->positionMetres, laneSide(centre->tangent));
    REQUIRE(std::abs(aside - back.yieldLateralMetres - back.settleLateralMetres) < 0.05);
}

TEST_CASE("the radio reckons the runner on: the general location moves along the lane while nobody sees the player",
          "[police][police-radio]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(oneLane(700.0, 14.0), 3.0, 1.0);
    const auto* front = frontCar(town);
    REQUIRE(front != nullptr);
    const auto z = front->positionMetres.z;

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // Seen, speeding, a chase; then gone — out of every patrol car's sight, off the end of the road.
    const auto seen = playerAt(glm::dvec3(0.0, 0.0, z + 40.0), 20.0);
    run(director, town, world, seen, 0.5, clock);
    REQUIRE(director.status().active);
    REQUIRE(director.status().observed);
    const auto broadcastAtSighting = director.status().broadcastMetres;

    const auto gone = playerAt(glm::dvec3(0.0, 0.0, z + 900.0), 20.0);
    run(director, town, world, gone, 3.0, clock);

    const auto& status = director.status();
    REQUIRE_FALSE(status.observed);
    REQUIRE(status.active);
    // Under the six seconds the search waits for, and the broadcast has moved up the lane at the
    // speed the player had: twenty metres a second for three seconds, less the broadcast's step.
    REQUIRE_FALSE(status.searching);
    REQUIRE(status.broadcastMetres.z > broadcastAtSighting.z + 25.0);
    REQUIRE(status.broadcastMetres.z < broadcastAtSighting.z + 70.0);
    REQUIRE(status.broadcasts >= 2);
}

TEST_CASE("head on, a unit drives straight at the player and does not slow", "[police][police-corridor]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(oneLane(700.0, 14.0), 3.0, 1.0);
    const auto* front = frontCar(town);
    REQUIRE(front != nullptr);
    const auto z = front->positionMetres.z;

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // The player eighty metres up the road, coming back down it at thirty: seen, the wrong way (an
    // offence), a chase. Wrecking is the goal (docs/police-driving-brief.md §10): the unit aims at
    // the player, ahead of it, at the maximum, and its driver is on the throttle.
    auto player = playerAt(glm::dvec3(1.5, 0.0, z + 80.0), -30.0);
    player.forward = glm::dvec3(0.0, 0.0, -1.0);
    run(director, town, world, player, 0.4, clock);
    REQUIRE(director.status().active);

    const auto* unit = director.unit(front->id);
    REQUIRE(unit != nullptr);
    REQUIRE(unit->turnPhase == 0);

    const auto& agent = town.agents()[front->id];
    const auto heading = glm::normalize(glm::dvec3(agent.heading.x, 0.0, agent.heading.z));
    const auto toAim = unit->aimMetres - agent.positionMetres;
    REQUIRE(glm::dot(toAim, heading) > 0.0);
    REQUIRE(unit->wantedSpeedMetresPerSecond > 30.0);

    auto pose = raceengine::PursuitCarPose{};
    pose.positionMetres = agent.positionMetres;
    pose.orientation = agent.orientation;
    pose.velocityMetresPerSecond = heading * 20.0;
    const auto drive = director.drive(*unit, pose);
    REQUIRE(drive.brake == 0.0);
    REQUIRE(drive.throttle > 0.5);
}

TEST_CASE("turning round in a narrow street: the room is measured, a three-point turn is done, and the unit comes about",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    // Walls five metres either side of the lane, a metre and a half tall, the length of the plate: a
    // full-lock arc of the Charger's 6.1 m needs fourteen metres, and there are ten.
    auto town = city(oneLane(700.0, 14.0), 3.0, 1.0);
    const auto* front = frontCar(town);
    REQUIRE(front != nullptr);
    const auto z = front->positionMetres.z;
    const auto id = front->id;

    const auto world = plate(700.0, 60.0, {box(5.0, 6.0, 20.0, 680.0, 1.5), box(-6.0, -5.0, 20.0, 680.0, 1.5)});

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // The player stopped forty metres behind the front car, out of its cruising cone: the wanted
    // level starts the chase, the unit joins, and its station is behind it.
    const auto player = playerAt(glm::dvec3(0.0, 0.0, z - 40.0), 0.0);
    run(director, town, world, player, 0.1, clock);
    director.setLevel(3);

    auto sawBacking = false;
    auto about = false;
    auto roomAtStart = 100.0;
    for (auto step = 0; step < 20 * 360 && !about; step++)
    {
        run(director, town, world, player, tick, clock);

        const auto* unit = director.unit(id);
        if (unit == nullptr)
        {
            continue;
        }

        if (unit->turnPhase == 1 && roomAtStart == 100.0)
        {
            roomAtStart = std::max(unit->roomLeftMetres, unit->roomRightMetres);
        }
        sawBacking = sawBacking || unit->turnPhase == 2;

        const auto& agent = town.agents()[id];
        about = glm::dot(agent.heading, glm::dvec3(0.0, 0.0, -1.0)) > 0.7 && unit->turnPhase == 0;
    }

    // The room was read before the turn began, and was short of the arc; the turn backed once; and
    // the unit came about.
    REQUIRE(roomAtStart < 7.0);
    REQUIRE(sawBacking);
    REQUIRE(about);
}

TEST_CASE("the PIT: from the ram level the tail lines up on the rear quarter and steers into it",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(oneLane(700.0, 14.0), 3.0, 1.0);

    // The rearmost patrol car, so the road ahead is long enough for the chase.
    const TrafficAgent* rear = nullptr;
    for (const auto& agent : town.agents())
    {
        if (rear == nullptr || agent.positionMetres.z < rear->positionMetres.z)
        {
            rear = &agent;
        }
    }
    REQUIRE(rear != nullptr);
    const auto z = rear->positionMetres.z;
    const auto id = rear->id;

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // The player forty metres ahead, driving on up the road at twenty: the unit is the tail, ramming
    // from the first level, and comes up beside the rear quarter.
    auto player = playerAt(glm::dvec3(0.0, 0.0, z + 40.0), 20.0);

    // The side the tail goes for is chosen once and held: it must not flip on the way in.
    auto pitted = false;
    auto sideSeen = 0;
    auto sideFlips = 0;
    for (auto step = 0; step < 15 * 360 && !pitted && player.positionMetres.z < 640.0; step++)
    {
        run(director, town, world, player, tick, clock);
        player.positionMetres += player.velocityMetresPerSecond * tick;

        const auto* unit = director.unit(id);
        pitted = unit != nullptr && unit->pitting;

        if (unit != nullptr && unit->pitSide != 0)
        {
            sideFlips += (sideSeen != 0 && unit->pitSide != sideSeen) ? 1 : 0;
            sideSeen = unit->pitSide;
        }
    }
    REQUIRE(sideFlips == 0);

    REQUIRE(director.status().active);
    REQUIRE(director.status().level >= director.settings().ramLevel);
    REQUIRE(pitted);

    // The aim, once striking, is across the unit's line toward the player's centreline.
    const auto* unit = director.unit(id);
    REQUIRE(unit != nullptr);
    const auto& agent = town.agents()[id];
    const auto left = glm::normalize(glm::cross(glm::dvec3(0.0, 1.0, 0.0), player.forward));
    const auto unitAcross = glm::dot(agent.positionMetres - player.positionMetres, left);
    const auto aimAcross = glm::dot(unit->aimMetres - player.positionMetres, left);
    REQUIRE(std::abs(aimAcross) < std::abs(unitAcross));
}

namespace
{

// A route polyline the way the router makes one: the lane's control points every eight metres,
// resampled every four along the chords, distances cumulative.
[[nodiscard]] std::vector<raceengine::RoutePoint> routeOf(const std::vector<glm::dvec3>& controls)
{
    auto points = std::vector<raceengine::RoutePoint>();
    auto distance = 0.0;
    auto carry = 0.0;

    points.push_back(raceengine::RoutePoint{.positionMetres = controls.front(), .distanceMetres = 0.0});

    for (auto index = std::size_t{1}; index < controls.size(); index++)
    {
        const auto& a = controls[index - 1];
        const auto& b = controls[index];
        const auto length = glm::length(glm::dvec3(b.x - a.x, 0.0, b.z - a.z));

        while (carry + 4.0 <= length)
        {
            carry += 4.0;
            distance += 4.0;
            points.push_back(raceengine::RoutePoint{.positionMetres = a + (b - a) * (carry / length),
                                                    .distanceMetres = distance});
        }

        carry -= length;
        if (carry < 0.0)
        {
            carry = 0.0;
        }
    }

    return points;
}

// A straight, a bend of radius R through ninety degrees, a straight — control points every eight
// metres, on the bend at eight metres of arc.
[[nodiscard]] std::vector<glm::dvec3> cornerOf(const double radius)
{
    auto controls = std::vector<glm::dvec3>();
    for (auto z = 0; z < 120; z += 8)
    {
        controls.push_back(glm::dvec3(0.0, 0.0, static_cast<double>(z)));
    }

    const auto arc = radius * 3.14159265358979323846 / 2.0;
    const auto steps = std::max(2, static_cast<int>(arc / 8.0));
    for (auto step = 1; step <= steps; step++)
    {
        const auto angle = (static_cast<double>(step) / static_cast<double>(steps)) * 3.14159265358979323846 / 2.0;
        controls.push_back(glm::dvec3(radius - radius * std::cos(angle), 0.0, 120.0 + radius * std::sin(angle)));
    }

    for (auto x = 8; x < 128; x += 8)
    {
        controls.push_back(glm::dvec3(radius + static_cast<double>(x), 0.0, 120.0 + radius));
    }

    return controls;
}

// A straight with one change of direction in it: the road bends by the angle at one control point.
[[nodiscard]] std::vector<glm::dvec3> kinkOf(const double degrees)
{
    auto controls = std::vector<glm::dvec3>();
    for (auto z = 0; z <= 200; z += 8)
    {
        controls.push_back(glm::dvec3(0.0, 0.0, static_cast<double>(z)));
    }

    const auto angle = degrees * 3.14159265358979323846 / 180.0;
    for (auto step = 1; step <= 25; step++)
    {
        controls.push_back(glm::dvec3(8.0 * step * std::sin(angle), 0.0, 200.0 + 8.0 * step * std::cos(angle)));
    }

    return controls;
}

[[nodiscard]] double lowestCeiling(const raceengine::SpeedProfile& profile)
{
    auto lowest = 1.0e9;
    for (const auto ceiling : profile.ceilingMetresPerSecond)
    {
        lowest = std::min(lowest, ceiling);
    }

    return lowest;
}

} // namespace

TEST_CASE("the profile reads a bend's curvature once, a lone kink by the driver's look-ahead, and a hairpin as a hairpin",
          "[police][police-plan]")
{
    auto limits = raceengine::PlannerLimits{};
    limits.maximumSpeedMetresPerSecond = 70.0;
    limits.entrySpeedMetresPerSecond = 70.0;
    limits.exitSpeedMetresPerSecond = 70.0;
    limits.brakingMetresPerSecondSquared = 9.3;
    limits.accelerationMetresPerSecondSquared = 8.0;

    const auto lateral = limits.corneringLimitG * 9.80665 * limits.cornerMargin;

    // Bends of 40 and 150 m: the ceiling is the tyre's speed on that radius, not on half of it — the
    // old three-point estimate over four metres read twice the curvature at every control point.
    for (const auto radius : {40.0, 150.0})
    {
        const auto profile = raceengine::planSpeedProfile(routeOf(cornerOf(radius)), limits);
        REQUIRE_THAT(lowestCeiling(profile), WithinAbs(std::sqrt(lateral * radius), 0.04 * std::sqrt(lateral * radius)));
    }

    // A hairpin of 12 m is held under the tyre's speed on it: safe, and slow.
    const auto hairpin = raceengine::planSpeedProfile(routeOf(cornerOf(12.0)), limits);
    REQUIRE(lowestCeiling(hairpin) < std::sqrt(lateral * 12.0));
    REQUIRE(lowestCeiling(hairpin) > 4.0);

    // A lone three-degree kink is not a corner; a ten-degree one is.
    const auto slight = raceengine::planSpeedProfile(routeOf(kinkOf(3.0)), limits);
    REQUIRE(lowestCeiling(slight) > 60.0);

    const auto sharp = raceengine::planSpeedProfile(routeOf(kinkOf(10.0)), limits);
    // The look-ahead arc through ten degrees: 0.55 s of speed over 2 sin(5°) at the planned lateral is
    // 23.6 m/s.
    REQUIRE(lowestCeiling(sharp) < 30.0);
    REQUIRE(lowestCeiling(sharp) > 18.0);
}

TEST_CASE("the driver corrects a lateral error at speed with the wheel at the tyre's limit and never with the brakes",
          "[police][police-corridor]")
{
    auto director = PursuitDirector{};

    // A unit at forty metres a second with its aim twenty-two metres ahead and two metres to the
    // left: the old cap read that as a corner and held the unit to thirty-two.
    auto unit = PursuitUnit{};
    unit.wantedSpeedMetresPerSecond = 62.0;
    unit.aimMetres = glm::dvec3(2.0, 0.0, 122.0);

    auto pose = raceengine::PursuitCarPose{};
    pose.positionMetres = glm::dvec3(0.0, 0.0, 100.0);
    pose.velocityMetresPerSecond = glm::dvec3(0.0, 0.0, 40.0);
    pose.wheelbaseMetres = 3.053;
    pose.lockRadians = 0.462;
    pose.corneringLimitG = 0.85;

    const auto drive = director.drive(unit, pose);

    REQUIRE(drive.brake == 0.0);
    REQUIRE(drive.throttle > 0.9);
    REQUIRE(drive.steering < 0.0);

    // The wheel asks for no more than the tyre holds at this speed.
    const auto tyreCurvature = 0.85 * 9.80665 / (40.0 * 40.0);
    REQUIRE(std::abs(drive.steering) <= std::atan(3.053 * tyreCurvature) / 0.462 + 1e-9);

    // The same aim well off the nose is a corner, and is slowed for.
    unit.aimMetres = glm::dvec3(8.0, 0.0, 110.0);
    const auto corner = director.drive(unit, pose);
    REQUIRE(corner.brake > 0.0);
}

TEST_CASE("traffic stops short of a patrol car across its lane, and a car sideways across two lanes is in both lists",
          "[traffic][traffic-yield]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(twoLanes(700.0, 14.0), 4.0, 0.0);
    auto clock = 0.0;

    runCity(town, world, {}, 0.2, clock);

    const auto* car = carWithRoomBehind(town, 120.0, 500.0, 30.0);
    REQUIRE(car != nullptr);
    const auto id = car->id;
    // Read now: `car` is the live agent, and it moves.
    const auto startDistance = car->distanceMetres;
    const auto startLane = car->lane;

    const auto& network = town.network();
    const auto& own = network.lanes[startLane];
    const auto& neighbourRun = own.neighbours.front();

    // A patrol car sixty metres ahead, stopped, sideways across the road — its nose in the
    // neighbour lane, its tail in this one — siren on.
    const auto ahead = laneAt(own, startDistance + 60.0);
    REQUIRE(ahead.has_value());
    const auto side = laneSide(ahead->tangent);
    const auto across = glm::normalize(side * neighbourRun.side);
    const auto patrol = std::vector{TrafficVehicle{.positionMetres = ahead->positionMetres + across * 1.85,
                                                   .velocityMetresPerSecond = glm::dvec3(0.0),
                                                   .forward = across,
                                                   .lengthMetres = 5.0,
                                                   .siren = true}};

    // In both lanes' lists: a query on the neighbour's centre a few metres short of it finds it too.
    runCity(town, world, patrol, 0.1, clock);
    const auto& other = network.lanes[neighbourRun.lane];
    const auto onOther = raceengine::wrapDistance(other, startDistance + 60.0 + neighbourShiftAt(neighbourRun, startDistance + 60.0));
    const auto probe = laneAt(other, onOther - 15.0);
    REQUIRE(probe.has_value());
    const auto seen = town.trafficAround(probe->positionMetres, probe->tangent, 7.0, 40.0);
    REQUIRE(seen.onRoad);
    // The siren car is not traffic to the corridor, so the lines read clear — the lists are read for
    // the yield and the following instead: the car behind it in this lane stops short.
    static_cast<void>(seen);

    runCity(town, world, patrol, 8.0, clock);

    const auto& stopped = town.agents()[id];
    REQUIRE(stopped.mode == AgentMode::Cruising);
    REQUIRE(stopped.speedMetresPerSecond < 0.5);
    REQUIRE(stopped.lane == startLane);

    // Stopped the siren standoff short of it, near enough, and not the driver's own standstill gap.
    const auto gap = 60.0 - (stopped.distanceMetres - startDistance) - 0.5 * 5.0 -
                     town.settings().vehicle.bodyHalfExtentsMetres.z;
    REQUIRE(gap > town.settings().sirenStandoffMetres - 2.0);
    REQUIRE(gap < town.settings().sirenStandoffMetres + 4.0);
}

namespace
{

// A lane that runs straight, bends left through a right angle on a radius, and runs straight on —
// control points every eight metres, as the export has them.
[[nodiscard]] LaneNetwork bentLane(const double radius)
{
    auto built = buildLaneNetwork({lane(1, cornerOf(radius), 14.0)});
    REQUIRE(built.has_value());

    return std::move(built).value();
}

} // namespace

TEST_CASE("the line through a bend: the car holds the arc, inside by no more than the allowance, and never onto a kerb",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    // A 30 m bend: tight enough for the full apex allowance. The corner's arc centre is at (30, 120).
    constexpr auto radius = 30.0;
    const auto world = plate(700.0, 700.0, {});
    auto town = city(bentLane(radius), 8.0, 1.0);
    auto router = PursuitRouter(town.network(), nullptr);

    auto director = PursuitDirector{};
    director.setRouter(&router);
    auto clock = 0.0;

    // The player well past the corner on the second straight, out of reach, so the unit routes;
    // every patrol car is on the first straight, which is where the city seeds two cars at this
    // density, and the chase is started by the wanted level.
    // Driving on along the second straight at twenty, so a patrol car seeded beside it does not
    // arrest it: the fixture holds it in place, and a stopped player with a unit within nine metres
    // is a bust in three seconds.
    auto player = playerAt(glm::dvec3(radius + 110.0, 0.0, 120.0 + radius), 0.0);
    player.velocityMetresPerSecond = glm::dvec3(20.0, 0.0, 0.0);
    player.forward = glm::dvec3(1.0, 0.0, 0.0);
    run(director, town, world, player, 0.1, clock);
    director.setLevel(2);

    auto entryLeadOk = true;
    auto apexSeen = false;
    auto apexInside = 0.0;
    auto apexOut = 0.0;
    auto through = false;

    // The first pass through the bend only: once a unit is out onto the second straight the run
    // ends, before it reaches the player, rams through it, turns and comes back the wrong way.
    for (auto step = 0; step < 25 * 360 && !through; step++)
    {
        run(director, town, world, player, tick, clock);

        for (const auto& unit : director.units())
        {
            if (!unit.routing || unit.plan.empty())
            {
                continue;
            }

            const auto& agent = town.agents()[unit.agent];
            const auto z = agent.positionMetres.z;

            through = through || (apexSeen && agent.positionMetres.x > radius + 10.0);

            // A routing unit feeds no curvature forward any more: the aim on the route carries the
            // corner (docs/police-driving-brief.md §12).
            entryLeadOk = entryLeadOk && unit.curvatureAhead == 0.0;

            // Through the bend, the unit's distance from the arc centre says which side of the lane's
            // true centre it drives on: inside by no more than the allowance and a little, and never
            // so far either way that a wheel is on a kerb at the lane's edge (0.9 m from the centre
            // for this body). The aim itself sits on the route's tangent, outside the arc by
            // construction; it is the car's line that is measured.
            const auto toUnit = glm::dvec2(agent.positionMetres.x - radius, z - 120.0);
            const auto angle = std::atan2(toUnit.y, -toUnit.x);
            if (angle > 0.35 && angle < 1.2 && glm::length(toUnit) < radius + 3.0)
            {
                apexSeen = true;
                const auto insideBy = radius - glm::length(toUnit);
                apexInside = std::max(apexInside, insideBy);
                apexOut = std::max(apexOut, -insideBy);
            }
        }
    }

    REQUIRE(entryLeadOk);
    REQUIRE(apexSeen);
    // Inside by the allowance and the car's own cut, and never so far that a wheel is on a kerb at the
    // lane's edge — 0.9 m from the centre for this body, which is the 0.3 m Dominic asked for.
    REQUIRE(apexInside <= director.settings().apexInsideMetres + 0.45);
    REQUIRE(apexInside < 0.9);
    REQUIRE(apexOut < 0.9);
}

TEST_CASE("the ram: a unit with a stationary player in front of it drives into it at the maximum, and does not stop",
          "[police][police-corridor]")
{
    const JoltGuard jolt;

    const auto world = plate(700.0, 60.0, {});
    auto town = city(oneLane(700.0, 14.0), 3.0, 1.0);

    const TrafficAgent* rear = nullptr;
    for (const auto& agent : town.agents())
    {
        if (rear == nullptr || agent.positionMetres.z < rear->positionMetres.z)
        {
            rear = &agent;
        }
    }
    REQUIRE(rear != nullptr);
    const auto z = rear->positionMetres.z;
    const auto id = rear->id;

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // The player stopped in the road fifty metres ahead; the chase started by the wanted level.
    const auto player = playerAt(glm::dvec3(0.0, 0.0, z + 50.0), 0.0);
    run(director, town, world, player, 0.1, clock);
    director.setLevel(2);

    auto rammed = false;
    auto slowest = 100.0;
    for (auto step = 0; step < 4 * 360; step++)
    {
        run(director, town, world, player, tick, clock);

        const auto* unit = director.unit(id);
        if (unit == nullptr)
        {
            break;
        }

        const auto& agent = town.agents()[id];
        const auto toPlayer = player.positionMetres.z - agent.positionMetres.z;
        if (toPlayer > 8.0 && toPlayer < 40.0)
        {
            rammed = rammed || unit->ramming;
            slowest = std::min(slowest, unit->wantedSpeedMetresPerSecond);
        }
    }

    REQUIRE(rammed);
    // Never asked to slow on the way in: the wanted speed stays the maximum, not the box's crawl.
    REQUIRE(slowest > 60.0);
}
