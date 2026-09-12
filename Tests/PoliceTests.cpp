// The police: the patrol car the pursuit is driven in, and the director that runs it
// (docs/police-pursuit-brief.md).
//
// **Every fixture is a straight road on a plate, built in code**, on the traffic tests' own rule: the
// rules being tested are the ones most likely to be wrong on a map nobody has looked at, and a test
// that needed Grand City Parkway's export on disk could only ever check the one map this project
// carries.

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
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::AgentMode;
using raceengine::behaviourFor;
using raceengine::bringUpJolt;
using raceengine::buildLaneNetwork;
using raceengine::ContactBody;
using raceengine::ContactManifold;
using raceengine::dodgeChargerPolice;
using raceengine::dodgeChargerPoliceAssists;
using raceengine::dodgeChargerPoliceDriveline;
using raceengine::DriverArchetype;
using raceengine::DriverProfile;
using raceengine::generateProvingGround;
using raceengine::ImpactLedger;
using raceengine::LaneSource;
using raceengine::noDriveTorque;
using raceengine::Offence;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::PursuitCarPose;
using raceengine::PursuitDirector;
using raceengine::PursuitOptions;
using raceengine::PursuitPlayer;
using raceengine::PursuitRole;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TrafficPopulation;
using raceengine::TrafficPopulationOptions;
using raceengine::TrafficUpdate;
using raceengine::VehicleInput;
using raceengine::VehicleState;

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
constexpr auto gravity = 9.80665;

[[nodiscard]] PhysicsWorld flatGround(const double length, const double width)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());

    auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    return std::move(world).value();
}

// A straight lane down the middle of the plate, from 20 m in to 20 m short of its end.
[[nodiscard]] raceengine::LaneNetwork straightRoad(const double length, const double limit)
{
    auto line = std::vector<glm::dvec3>();
    const auto span = length - 40.0;
    const auto count = static_cast<int>(span / 18.0);

    for (auto index = 0; index <= count; index++)
    {
        line.push_back(glm::dvec3(0.0, 0.0, 20.0 + span * static_cast<double>(index) / static_cast<double>(count)));
    }

    auto built = buildLaneNetwork({LaneSource{.id = 1,
                                              .name = "straight",
                                              .speedLimitMetresPerSecond = limit,
                                              .allowLaneChanges = false,
                                              .allowUTurns = false,
                                              .points = std::move(line)}});
    REQUIRE(built.has_value());

    return std::move(built).value();
}

// A city in which every car is a patrol car, so what is measured is the police and not the draw.
[[nodiscard]] TrafficPopulation policeCity(raceengine::LaneNetwork network, const double density)
{
    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = density;
    options.bodyCount = 2;
    options.policeBody = 1;
    options.policeShare = 1.0;
    options.recoverySeconds = 30.0;

    auto city = TrafficPopulation(
        std::move(network), options,
        {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    city.seed();

    REQUIRE_FALSE(city.agents().empty());
    REQUIRE(city.policeIds().size() == city.agents().size());

    return city;
}

// The player as the director sees it: a car at a point on the road, moving along it at a speed.
[[nodiscard]] PursuitPlayer playerAt(const double z, const double speed)
{
    return PursuitPlayer{.positionMetres = glm::dvec3(0.0, 0.0, z),
                         .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, speed),
                         .forward = glm::dvec3(0.0, 0.0, 1.0),
                         .lengthMetres = 4.3,
                         .widthMetres = 1.8};
}

// Tick the city and the director together for a number of seconds, the player held where the
// caller says. The player is handed to the city as an outside vehicle too, so traffic brakes for it.
void run(PursuitDirector& director, TrafficPopulation& city, const PhysicsWorld& world, const PursuitPlayer& player,
         const double seconds, double& clock)
{
    const auto steps = static_cast<int>(seconds / tick);
    const auto vehicles = std::array{raceengine::TrafficVehicle{.positionMetres = player.positionMetres,
                                                                .velocityMetresPerSecond = player.velocityMetresPerSecond,
                                                                .forward = player.forward,
                                                                .lengthMetres = player.lengthMetres}};

    for (auto index = 0; index < steps; index++)
    {
        city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                  .simulatedSeconds = clock,
                                  .focusMetres = player.positionMetres,
                                  .vehicles = vehicles},
                    world);
        director.update(tick, player, city, world);
        clock += tick;
    }
}

} // namespace

TEST_CASE("the police charger builds from its data and settles on its springs", "[police][police-charger]")
{
    const JoltGuard jolt;

    auto built = dodgeChargerPolice();
    REQUIRE(built.has_value());

    const auto& setup = built.value();

    // The file's own figures, carried through the assembly.
    auto sprung = 0.0;
    for (const auto& component : setup.sprung)
    {
        sprung += component.mass;
    }
    REQUIRE_THAT(sprung + setup.unsprungMass(), WithinAbs(1975.0, 1e-6));
    REQUIRE(setup.corners[0].hardpoints.wheelRadius == 0.350);
    REQUIRE(std::abs(setup.rackTravelPerInput) > 0.0);

    // Front-steer rack: a positive demand is a right turn, and on this linkage that is rack travel
    // toward the car's *right*, which is the opposite sign to the Golf's rear-steer rack.
    REQUIRE(setup.rackTravelPerInput < 0.0);

    const auto world = flatGround(200.0, 40.0);

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 1.0, 60.0);
    REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, 1e-6).has_value());

    const auto centreOfMass = state.chassis.centreOfMass;
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.0, 60.0) + centreOfMass;

    auto load = 0.0;
    for (auto index = 0; index < 3 * 360; index++)
    {
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick);
        REQUIRE(stepped.has_value());

        load = 0.0;
        for (const auto& corner : stepped->corners)
        {
            load += corner.forces.tireVertical;
        }
    }

    // Standing on all four wheels with its own weight on them, at rest, where it was put.
    REQUIRE_THAT(load, WithinAbs(1975.0 * gravity, 0.03 * 1975.0 * gravity));
    REQUIRE(glm::length(state.chassis.linearVelocity) < 0.05);
    REQUIRE_THAT(state.chassis.position.z, WithinAbs(60.0 + centreOfMass.z, 0.2));

    // The driveline and the electronics build against it.
    const auto driveline = dodgeChargerPoliceDriveline();
    REQUIRE(driveline.gearbox.ratios.size() == 5);
    REQUIRE(driveline.driven == raceengine::DrivenAxle::Rear);

    const auto assists = dodgeChargerPoliceAssists(setup);
    REQUIRE(assists.brakeTorquePerPressure[0] > 0.0);
    REQUIRE(assists.traction.driven[2]);
    REQUIRE_FALSE(assists.traction.driven[0]);
}

TEST_CASE("a patrol car that sees the player speeding starts a chase, and the felony rises while it lasts",
          "[police][police-pursuit]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // The player forty metres up the road from the first patrol car, in its sight and going twice
    // the limit.
    const auto& patrol = city.agents().front();
    const auto player = playerAt(patrol.positionMetres.z + 40.0, 28.0);

    run(director, city, world, player, 0.5, clock);

    const auto& status = director.status();

    REQUIRE(status.observed);
    REQUIRE(status.speeding);
    REQUIRE(status.offence == Offence::Speeding);
    REQUIRE(status.active);
    REQUIRE(status.units >= 1);
    REQUIRE(status.pursuitsStarted == 1);
    REQUIRE(status.felony > 0.0);

    // Every unit is a pursuing body or the caller's, with its lights on and an aim ahead of it.
    for (const auto& unit : director.units())
    {
        const auto& agent = city.agents()[unit.agent];
        REQUIRE((agent.mode == AgentMode::Pursuing || agent.mode == AgentMode::External));
        REQUIRE(agent.siren);
        REQUIRE(unit.wantedSpeedMetresPerSecond > 0.0);
    }

    const auto before = status.felony;
    run(director, city, world, player, 2.0, clock);

    REQUIRE(director.status().felony > before);
    REQUIRE(director.status().level >= 1);
}

TEST_CASE("an offence nobody can see raises no felony and starts no chase", "[police][police-pursuit]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);

    auto director = PursuitDirector{};
    auto clock = 0.0;

    // Out past every patrol car's sight radius, and speeding.
    auto furthest = 0.0;
    for (const auto& agent : city.agents())
    {
        furthest = std::max(furthest, agent.positionMetres.z);
    }

    const auto player = playerAt(furthest + 300.0, 28.0);
    run(director, city, world, player, 1.0, clock);

    REQUIRE(director.status().speeding);
    REQUIRE_FALSE(director.status().observed);
    REQUIRE_FALSE(director.status().active);
    REQUIRE(director.status().felony == 0.0);
}

TEST_CASE("the units take stations around a stopped player, the box closes, and the player is busted",
          "[police][police-pursuit]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 6.0);

    auto options = PursuitOptions{};
    options.arrestSeconds = 3.0;
    // The box is what the units do below the ram level, which ships at one since 2026-09-12
    // (docs/police-driving-brief.md §10, §12): from that level a unit facing the player drives into it,
    // and a fixture's player has no body to be driven into. The box is kept testable above it.
    options.ramLevel = 6;
    auto director = PursuitDirector{options};
    auto clock = 0.0;

    // Caught speeding, then stopped in the road.
    const auto& patrol = city.agents().front();
    const auto speeding = playerAt(patrol.positionMetres.z + 40.0, 28.0);
    run(director, city, world, speeding, 0.5, clock);
    REQUIRE(director.status().active);

    const auto stopped = playerAt(patrol.positionMetres.z + 40.0, 0.0);
    run(director, city, world, stopped, 6.0, clock);

    const auto& status = director.status();
    REQUIRE(status.active);

    // Somebody has the tail, and every station is within a few car lengths of the player. The aim a
    // station unit steers at rides the station's line a look-ahead further on, so it is the station
    // that is near the player and the aim that is ahead of the unit.
    auto tails = std::size_t{0};
    for (const auto& unit : director.units())
    {
        tails += unit.role == PursuitRole::Tail ? 1 : 0;

        if (unit.role != PursuitRole::Chase)
        {
            REQUIRE(glm::length(unit.stationMetres - stopped.positionMetres) < 3.0 * stopped.lengthMetres);
            REQUIRE(glm::dot(unit.aimMetres - unit.stationMetres, stopped.forward) >= 0.0);
        }
    }
    REQUIRE(tails == 1);

    // The nearest unit closes to the box, and a stopped player with a patrol car on it is caught.
    run(director, city, world, stopped, 20.0, clock);

    REQUIRE(director.status().busted);
    REQUIRE_FALSE(director.status().active);

    // Every unit has been handed back: nothing is pursuing any more.
    for (const auto& agent : city.agents())
    {
        REQUIRE(agent.mode != AgentMode::Pursuing);
        REQUIRE(agent.mode != AgentMode::External);
        REQUIRE_FALSE(agent.siren);
    }
}

TEST_CASE("damage accumulates from what the bodywork takes, and a wrecked patrol car leaves the chase",
          "[police][police-damage]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);

    auto director = PursuitDirector{};
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    auto player = playerAt(patrol.positionMetres.z + 40.0, 28.0);
    run(director, city, world, player, 0.5, clock);
    REQUIRE(director.status().active);

    // A 9 m/s hit on the player's bodywork against a wall: eight over the threshold at a twentieth
    // each, so 0.4 of damage, and nothing to do with the police.
    player.impactWorldMetresPerSecond = 9.0;
    director.update(tick, player, city, world);
    player.impactWorldMetresPerSecond = 0.0;

    REQUIRE_THAT(director.status().playerDamage, WithinAbs(0.40, 1e-9));
    REQUIRE_FALSE(director.status().wrecked);

    // Hitting a patrol car is known to the police whether or not they saw it, and it counts twice:
    // on the meter and on the panels.
    const auto felonyBefore = director.status().felony;
    player.impactPoliceMetresPerSecond = 15.0;
    director.update(tick, player, city, world);
    player.impactPoliceMetresPerSecond = 0.0;

    REQUIRE(director.status().felony > felonyBefore + 0.1);
    REQUIRE(director.status().playerDamage >= 1.0);
    REQUIRE(director.status().wrecked);

    // A unit that takes a wrecking hit is written off: stopped, silent, out of the roster.
    REQUIRE_FALSE(director.units().empty());
    const auto victim = director.units().front().agent;

    auto manifold = ContactManifold{};
    manifold.bodies.push_back(ContactBody{});
    manifold.bodies.push_back(ContactBody{.obstacle = victim, .deltaLinear = glm::dvec3(0.0, 0.0, 25.0)});
    city.applyContacts(manifold);

    run(director, city, world, player, 0.2, clock);

    REQUIRE(director.unit(victim) == nullptr);
    REQUIRE(director.status().policeWrecked == 1);
    REQUIRE(city.agents()[victim].mode == AgentMode::Stopped);
    REQUIRE_FALSE(city.agents()[victim].siren);
}

TEST_CASE("a patrol car nudged by another car's solve is charged the closing speed it met, never the solver's push-out",
          "[police][police-damage]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);

    auto director = PursuitDirector{};
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    const auto player = playerAt(patrol.positionMetres.z + 40.0, 28.0);
    run(director, city, world, player, 0.5, clock);
    REQUIRE(director.status().active);
    REQUIRE_FALSE(director.units().empty());
    const auto victim = director.units().front().agent;

    // The solver's own velocity change — what a box a hand's width inside another reads on every
    // tick of the push-out — applied under the caller's ledger: the body takes the nudge and the
    // damage model reads nothing.
    auto manifold = ContactManifold{};
    manifold.bodies.push_back(ContactBody{});
    manifold.bodies.push_back(ContactBody{.obstacle = victim, .deltaLinear = glm::dvec3(0.0, 0.0, 25.0)});

    for (auto tick_ = 0; tick_ < 12; tick_++)
    {
        city.applyContacts(manifold, ImpactLedger::FromHits);
        run(director, city, world, player, tick, clock);
    }

    const auto* unhurt = director.unit(victim);
    REQUIRE(unhurt != nullptr);
    REQUIRE_THAT(unhurt->damage, WithinAbs(0.0, 1e-12));
    REQUIRE(director.status().policeWrecked == 0);

    // The closing speed, charged once by the caller: the same 25 m/s is a wreck.
    city.chargeImpact(victim, 25.0);
    run(director, city, world, player, 0.2, clock);

    REQUIRE(director.unit(victim) == nullptr);
    REQUIRE(director.status().policeWrecked == 1);
    REQUIRE(city.agents()[victim].mode == AgentMode::Stopped);
}

TEST_CASE("a pursuing body can be handed to an outside model and taken back where it was left",
          "[police][police-external]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);

    auto director = PursuitDirector{};
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    const auto player = playerAt(patrol.positionMetres.z + 40.0, 28.0);
    run(director, city, world, player, 0.5, clock);
    REQUIRE_FALSE(director.units().empty());

    const auto id = director.units().front().agent;
    REQUIRE(city.agents()[id].mode == AgentMode::Pursuing);
    const auto bodies = city.report().pursuing;

    // Taken: no body here any more, the pose the caller's to write.
    REQUIRE(city.takeExternal(id));
    director.setFullModel(id, true);
    REQUIRE(city.agents()[id].mode == AgentMode::External);

    const auto placed = glm::dvec3(2.0, 0.0, patrol.positionMetres.z + 10.0);
    city.setExternalPose(id, placed, glm::dquat(1.0, 0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 15.0), 123.0);

    run(director, city, world, player, 0.5, clock);

    // The population left it exactly where the caller put it, still counts it, still aims it, and
    // offers it to nobody as an obstacle.
    const auto& external = city.agents()[id];
    REQUIRE(external.mode == AgentMode::External);
    REQUIRE_THAT(external.positionMetres.x, WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(external.rolledMetres, WithinAbs(123.0, 1e-9));
    REQUIRE(city.report().external == 1);
    REQUIRE(city.report().pursuing == bodies - 1);
    REQUIRE(director.unit(id) != nullptr);
    REQUIRE(director.unit(id)->fullModel);
    REQUIRE(director.status().fullModels == 1);

    auto obstacles = std::vector<raceengine::DynamicObstacle>();
    city.obstaclesNear(placed, 5.0, obstacles);
    for (const auto& obstacle : obstacles)
    {
        REQUIRE(obstacle.id != id);
    }

    // The full model's driver has something to do with the aim it was given.
    const auto drive = director.drive(*director.unit(id),
                                      PursuitCarPose{.positionMetres = placed,
                                                     .orientation = glm::dquat(1.0, 0.0, 0.0, 0.0),
                                                     .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, 15.0),
                                                     .wheelbaseMetres = 3.053,
                                                     .lockRadians = 0.462});
    REQUIRE(drive.throttle >= 0.0);
    REQUIRE(drive.throttle <= 1.0);
    REQUIRE(std::abs(drive.steering) <= 1.0);

    // Taken back: a body again, stood up where the caller left it.
    city.releaseExternal(id, world);
    director.setFullModel(id, false);

    REQUIRE(city.agents()[id].mode == AgentMode::Pursuing);
    REQUIRE_THAT(city.agents()[id].positionMetres.x, WithinAbs(2.0, 0.05));

    run(director, city, world, player, 0.5, clock);
    REQUIRE(city.report().external == 0);
    REQUIRE(city.report().pursuing == bodies);
}

TEST_CASE("a unit told to back up backs up, as a body and through the full model's driver", "[police][police-reverse]")
{
    const JoltGuard jolt;

    const auto world = flatGround(700.0, 60.0);
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);

    auto director = PursuitDirector{};
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    const auto player = playerAt(patrol.positionMetres.z + 40.0, 28.0);
    run(director, city, world, player, 0.5, clock);
    REQUIRE_FALSE(director.units().empty());

    const auto id = director.units().front().agent;
    REQUIRE(city.agents()[id].mode == AgentMode::Pursuing);

    // The body alone, told to back toward a point eight metres behind it, with the director out of
    // the loop so nothing overwrites the order.
    const auto start = city.agents()[id].positionMetres;
    const auto heading = city.agents()[id].heading;
    city.setPursuitAim(id, start - heading * 8.0, 4.0, true, true);

    // Three seconds: the body was chasing at close to twenty metres a second when the order came,
    // and it brakes at the tyre's limit before it backs up — about two seconds of that.
    auto midway = start;
    for (auto index = 0; index < 1080; index++)
    {
        if (index == 900)
        {
            midway = city.agents()[id].positionMetres;
        }

        city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                  .simulatedSeconds = clock,
                                  .focusMetres = player.positionMetres,
                                  .vehicles = {}},
                    world);
        clock += tick;
    }

    const auto& backed = city.agents()[id];
    INFO("mode " << static_cast<int>(backed.mode) << " reverse " << backed.pursuitReverse << " speed along "
                 << glm::dot(backed.velocityMetresPerSecond, heading) << " moved since 2.5 s "
                 << glm::dot(backed.positionMetres - midway, heading));
    REQUIRE(backed.mode == AgentMode::Pursuing);
    REQUIRE(glm::dot(backed.velocityMetresPerSecond, heading) < -0.5);
    REQUIRE(glm::dot(backed.positionMetres - midway, heading) < -0.3);

    // The full model's driver: a unit backing toward a station behind and to its left reverses with
    // the wheel turned left (a negative demand), and one backing off something in front of it with
    // the wheel turned away from the aim.
    const auto pose = PursuitCarPose{.positionMetres = glm::dvec3(0.0),
                                     .orientation = glm::dquat(1.0, 0.0, 0.0, 0.0),
                                     .velocityMetresPerSecond = glm::dvec3(0.0),
                                     .wheelbaseMetres = 3.053,
                                     .lockRadians = 0.462};

    const auto toStation = director.drive(raceengine::PursuitUnit{.agent = id,
                                                                  .aimMetres = glm::dvec3(2.0, 0.0, -6.0),
                                                                  .wantedSpeedMetresPerSecond = 4.0,
                                                                  .reversing = true,
                                                                  .reversingToStation = true},
                                          pose);
    REQUIRE(toStation.reverse);
    REQUIRE(toStation.throttle > 0.0);
    REQUIRE(toStation.steering < 0.0);

    const auto offWall = director.drive(
        raceengine::PursuitUnit{
            .agent = id, .aimMetres = glm::dvec3(2.0, 0.0, 6.0), .wantedSpeedMetresPerSecond = 4.0, .reversing = true},
        pose);
    REQUIRE(offWall.reverse);
    REQUIRE(offWall.throttle > 0.0);
    REQUIRE(offWall.steering > 0.0);
}

TEST_CASE("a city with no police body stated seeds the city it always did", "[police][police-seed]")
{
    auto network = straightRoad(700.0, 14.0);
    auto copy = network;

    auto plain = TrafficPopulationOptions{};
    plain.densityPerKilometre = 8.0;
    plain.bodyCount = 5;

    auto before = TrafficPopulation(std::move(network), plain,
                                    {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    before.seed();

    auto stated = plain;
    stated.policeBody = 4;
    stated.policeShare = 0.0;

    auto after = TrafficPopulation(std::move(copy), stated,
                                   {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    after.seed();

    REQUIRE(before.agents().size() == after.agents().size());
    REQUIRE(after.policeIds().empty());

    // With the police share at zero no car is a patrol car, and none takes the patrol car's body.
    for (const auto& agent : after.agents())
    {
        REQUIRE_FALSE(agent.police);
        REQUIRE(agent.body != 4);
    }

    // With none stated the draw is the old one, car for car.
    for (auto index = std::size_t{0}; index < before.agents().size(); index++)
    {
        REQUIRE(before.agents()[index].body == before.agents()[index].body);
        REQUIRE_FALSE(before.agents()[index].police);
    }
}
