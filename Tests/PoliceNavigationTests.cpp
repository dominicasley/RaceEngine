// The police navigating: the line of drive and the routing switch, the aim on the route, the
// feedforward in both drivers, and the Charger's braking measured (docs/pursuit-navigation-brief.md,
// stages 2 and 3). The same fixtures as PoliceTests.cpp: a straight road on a plate, built in code.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::AgentMode;
using raceengine::AssistSensors;
using raceengine::AssistState;
using raceengine::behaviourFor;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::buildLaneNetwork;
using raceengine::cornerCount;
using raceengine::dodgeChargerPolice;
using raceengine::dodgeChargerPoliceAssists;
using raceengine::DriverArchetype;
using raceengine::DriverProfile;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::LaneSource;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::PursuitCarPose;
using raceengine::PursuitDirector;
using raceengine::PursuitOptions;
using raceengine::PursuitPlayer;
using raceengine::PursuitRouter;
using raceengine::PursuitUnit;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TractionMode;
using raceengine::TrafficPopulation;
using raceengine::TrafficPopulationOptions;
using raceengine::TrafficUpdate;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::VehicleStep;

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

[[nodiscard]] PhysicsWorld plate(const double length, const double width, const std::vector<Feature>& features)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = 2.0;
    descriptor.features = features;

    auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());

    auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    return std::move(world).value();
}

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

    return city;
}

[[nodiscard]] PursuitPlayer playerAt(const glm::dvec3& position, const double speed)
{
    return PursuitPlayer{.positionMetres = position,
                         .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, speed),
                         .forward = glm::dvec3(0.0, 0.0, 1.0),
                         .lengthMetres = 4.3,
                         .widthMetres = 1.8};
}

void run(PursuitDirector& director, TrafficPopulation& city, const PhysicsWorld& world, const PursuitPlayer& player,
         const double seconds, double& clock)
{
    const auto steps = static_cast<int>(std::lround(seconds / tick));
    const auto vehicles =
        std::array{raceengine::TrafficVehicle{.positionMetres = player.positionMetres,
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

} // namespace

TEST_CASE("the straight: a unit out of reach routes along the lane, its aim on the road ahead of it, its speed the cap",
          "[police][police-routing]")
{
    const JoltGuard jolt;

    const auto world = plate(900.0, 60.0, {});
    auto city = policeCity(straightRoad(900.0, 14.0), 3.0);
    auto router = PursuitRouter(city.network(), nullptr);

    // Seen from anywhere on the plate: this fixture is about a unit routing to the player it has,
    // and a unit that cannot see the player routes to the radio's last position instead
    // (docs/pursuit-radio-brief.md; PoliceRadioTests.cpp).
    auto options = PursuitOptions{};
    options.sightRadiusMetres = 400.0;

    auto director = PursuitDirector{options};
    director.setRouter(&router);
    auto clock = 0.0;

    // The player up the road from the first patrol car, in sight and speeding, to start the chase;
    // then three hundred metres up it, out of every unit's reach, for the routing.
    const auto& patrol = city.agents().front();
    const auto seen = playerAt(glm::dvec3(0.0, 0.0, patrol.positionMetres.z + 90.0), 28.0);
    run(director, city, world, seen, 0.5, clock);
    REQUIRE(director.status().active);

    const auto player = playerAt(glm::dvec3(0.0, 0.0, patrol.positionMetres.z + 300.0), 28.0);
    run(director, city, world, player, 1.0, clock);

    const auto& status = director.status();
    REQUIRE(status.active);
    REQUIRE(status.routingUnits >= 1);
    REQUIRE(status.routesBuilt >= 1);

    auto checked = std::size_t{0};

    for (const auto& unit : director.units())
    {
        if (unit.distanceMetres <= director.settings().chaseDistanceMetres || unit.reversing)
        {
            continue;
        }

        const auto& agent = city.agents()[unit.agent];

        REQUIRE(unit.routing);
        REQUIRE(unit.route.found);
        REQUIRE(unit.route.points.size() >= 2);

                // The aim on the lane, ahead of the unit by the look-ahead; nothing to turn in for; and the
        // profile flat at the cap on a straight, for a unit far enough from the player that the
        // braking to the player's pace at the far end has not begun (from 62 to 38 at 10.3 m/s² is
        // 116 m).
        REQUIRE(std::abs(unit.aimMetres.x) < 1.0);
        const auto ahead = unit.aimMetres.z - agent.positionMetres.z;
        const auto speed = glm::length(agent.velocityMetresPerSecond);
        const auto lookAhead = std::max(director.settings().routeLookAheadMetres,
                                        director.settings().routeLookAheadSeconds * speed);
        REQUIRE_THAT(ahead, WithinAbs(lookAhead, 1.0));
        REQUIRE(std::abs(unit.curvatureAhead) < 1e-3);

        if (unit.distanceMetres > 200.0)
        {
            REQUIRE_THAT(unit.wantedSpeedMetresPerSecond,
                         WithinAbs(director.settings().maximumUnitSpeedMetresPerSecond, 1e-6));
        }


        checked++;
    }

    REQUIRE(checked >= 1);

    // The route is rebuilt on its interval and not every tick: at two a second, a second's chase
    // built a handful and never hundreds.
    REQUIRE(status.routesBuilt <= 3 * (status.units + 1));
}

TEST_CASE(
    "the switch: a unit whose line of drive is blocked routes, and keeps its route for the hold after the line clears",
    "[police][police-routing]")
{
    const JoltGuard jolt;

    // A wall down the right of the lane, 0.9 m tall at x = 4: the sight line at 1.2 m clears it, the
    // line of drive at 0.6 m does not.
    const auto world = plate(700.0, 60.0, {Feature{.kind = FeatureKind::Barrier, .from = 0.0, .to = 700.0}});
    auto city = policeCity(straightRoad(700.0, 14.0), 3.0);
    auto router = PursuitRouter(city.network(), nullptr);

    auto director = PursuitDirector{};
    director.setRouter(&router);
    auto clock = 0.0;

    // The player beyond the wall, thirty metres up the road from the first patrol car, off the road
    // and moving: seen, offending, and unreachable in a straight line.
    const auto& patrol = city.agents().front();
    const auto beyond = playerAt(glm::dvec3(10.0, 0.0, patrol.positionMetres.z + 30.0), 28.0);

    run(director, city, world, beyond, 1.0, clock);

    REQUIRE(director.status().active);

    const auto* nearest = nearestUnit(director);
    REQUIRE(nearest != nullptr);
    REQUIRE(nearest->distanceMetres < director.settings().chaseDistanceMetres);
    REQUIRE_FALSE(nearest->lineClear);
    REQUIRE(nearest->routing);
    // No mesh and a player off the network: the route is the straight line as the last resort, and
    // says so.
    REQUIRE(nearest->route.blocked);
    REQUIRE(nearest->route.direct);
    const auto id = nearest->agent;

    // The player stops on the road just ahead of that unit: the line clears, inside reach.
    const auto& agent = city.agents()[id];
    const auto onRoad = playerAt(glm::dvec3(0.0, 0.0, agent.positionMetres.z + 20.0), 0.0);

    // Still routing a third of a second later — the hold is half a second from the sight tick that
    // saw the line clear — and off its route three quarters of a second later.
    run(director, city, world, onRoad, 0.35, clock);
    const auto* held = director.unit(id);
    REQUIRE(held != nullptr);
    REQUIRE(held->lineClear);
    REQUIRE(held->routing);

    run(director, city, world, onRoad, 0.4, clock);
    const auto* released = director.unit(id);
    REQUIRE(released != nullptr);
    REQUIRE(released->lineClear);
    REQUIRE_FALSE(released->routing);
    REQUIRE_THAT(released->curvatureAhead, WithinAbs(0.0, 1e-12));
}

TEST_CASE("the turn-in: the route's curvature ahead adds a road wheel angle through the wheelbase in both drivers",
          "[police][police-routing]")
{
    const JoltGuard jolt;

    // At ten metres a second, where a 20 m radius is inside what the tyre holds (12.9 m/s at 0.85 g):
    // above that the wheel is limited to the tyre's curvature at speed (docs/police-driving-brief.md §9)
    // and the turn-in would read as less than the corner asks.
    const auto director = PursuitDirector{};
    const auto pose = PursuitCarPose{.positionMetres = glm::dvec3(0.0),
                                     .orientation = glm::dquat(1.0, 0.0, 0.0, 0.0),
                                     .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, 10.0),
                                     .wheelbaseMetres = 3.053,
                                     .lockRadians = 0.462};

    // The aim straight ahead: pure pursuit asks for nothing, the feedforward for the corner. Left
    // positive, and a left turn is a negative demand.
    const auto straight =
        director.drive(PursuitUnit{.aimMetres = glm::dvec3(0.0, 0.0, 12.0), .wantedSpeedMetresPerSecond = 20.0}, pose);
    REQUIRE_THAT(straight.steering, WithinAbs(0.0, 1e-9));

    const auto leftCorner = director.drive(PursuitUnit{.aimMetres = glm::dvec3(0.0, 0.0, 12.0),
                                                       .wantedSpeedMetresPerSecond = 20.0,
                                                       .curvatureAhead = 1.0 / 20.0},
                                           pose);
    REQUIRE_THAT(leftCorner.steering, WithinAbs(-std::atan(3.053 / 20.0) / 0.462, 1e-9));

    const auto rightCorner = director.drive(PursuitUnit{.aimMetres = glm::dvec3(0.0, 0.0, 12.0),
                                                        .wantedSpeedMetresPerSecond = 20.0,
                                                        .curvatureAhead = -1.0 / 20.0},
                                            pose);
    REQUIRE_THAT(rightCorner.steering, WithinAbs(-leftCorner.steering, 1e-9));

    // ...and the arc cap reads the total: a 20 m radius at 0.85 g is sqrt(0.85 g 20) = 12.9 m/s, so a
    // wanted 20 m/s comes back as a lift.
    REQUIRE(leftCorner.throttle < straight.throttle);

    // The cheap body's driver: told the same curvature through the population, a pursuing body's
    // steer differs by the same road wheel angle through its own wheelbase.
    const auto world = plate(700.0, 60.0, {});
    auto city = policeCity(straightRoad(700.0, 14.0), 4.0);
    auto guide = PursuitDirector{};
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    const auto player = playerAt(glm::dvec3(0.0, 0.0, patrol.positionMetres.z + 40.0), 28.0);
    run(guide, city, world, player, 0.5, clock);
    REQUIRE_FALSE(guide.units().empty());

    const auto id = guide.units().front().agent;
    REQUIRE(city.agents()[id].mode == AgentMode::Pursuing);
    const auto before = city.agents()[id].positionMetres;
    const auto heading = city.agents()[id].heading;

    // The body told to go straight for a second, then told the same with a corner's curvature ahead.
    for (auto step = 0; step < 360; step++)
    {
        city.setPursuitAim(id, city.agents()[id].positionMetres + city.agents()[id].heading * 30.0, 20.0, true, false,
                           0.0);
        city.update(
            TrafficUpdate{.deltaTimeSeconds = tick, .simulatedSeconds = clock, .focusMetres = before, .vehicles = {}},
            world);
        clock += tick;
    }

    const auto straightHeading = city.agents()[id].heading;
    REQUIRE(glm::dot(straightHeading, heading) > 0.99);

    for (auto step = 0; step < 360; step++)
    {
        city.setPursuitAim(id, city.agents()[id].positionMetres + city.agents()[id].heading * 30.0, 20.0, true, false,
                           1.0 / 20.0);
        city.update(
            TrafficUpdate{.deltaTimeSeconds = tick, .simulatedSeconds = clock, .focusMetres = before, .vehicles = {}},
            world);
        clock += tick;
    }

    // A second of the turn-in at a 20 m radius: the body has turned left, well past what the straight
    // second did.
    const auto turned = city.agents()[id].heading;
    const auto left = glm::normalize(glm::cross(glm::dvec3(0.0, 1.0, 0.0), straightHeading));
    REQUIRE(glm::dot(turned, left) > 0.2);
}

// The Charger's straight-line deceleration, measured: from 30 m/s on the plate, the pedal on the
// floor, the electronics on as the patrol cars have them. What the profile plans braking points with
// (`PursuitOptions::fullModelBrakingMetresPerSecondSquared`); 8 m/s² was the placeholder.
TEST_CASE("the Charger braking fixture: the deceleration the profile is planned with is the one the car has",
          "[police][police-brake]")
{
    const JoltGuard jolt;

    auto built = dodgeChargerPolice();
    REQUIRE(built.has_value());
    const auto& setup = built.value();

    auto assists = dodgeChargerPoliceAssists(setup);
    assists.antilock.enabled = true;
    assists.traction.mode = TractionMode::Full;
    assists.cornering.enabled = true;

    const auto world = plate(900.0, 40.0, {});

    // On its springs, at rest, then rolling at thirty.
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 1.0, 60.0);
    REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, 1e-6).has_value());
    const auto centreOfMass = state.chassis.centreOfMass;
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.0, 60.0) + centreOfMass;

    for (auto index = 0; index < 3 * 360; index++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    const auto entry = 30.0;
    const auto radius = setup.corners[0].hardpoints.wheelRadius;
    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, entry);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = entry / radius;
    }

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};

    const auto sense = [&]
    {
        auto sensors = AssistSensors{};
        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            sensors.wheelSpeeds[index] = state.corners[index].wheelSpeed;
        }
        sensors.yawRate = lastStep.telemetry.yawRate;
        sensors.lateralAcceleration = lastStep.telemetry.acceleration.x;
        sensors.steeringWheelAngle = lastStep.telemetry.steeringWheelAngle;
        sensors.driveTorqueKnown = true;

        return sensors;
    };

    // Rolling before the pedal, so the tone rings have readings and the reference speed has settled.
    for (auto index = 0; index < 180; index++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, brakeCircuitPressures(setup, 0.0), tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    REQUIRE_THAT(state.chassis.linearVelocity.z, WithinAbs(entry, 1.0));

    // The pedal on the floor until the car is under five metres a second.
    auto input = VehicleInput{};
    input.brake = 1.0;

    const auto from = state.chassis.linearVelocity.z;
    const auto start = state.chassis.position.z;
    auto elapsed = 0.0;
    auto stopped = false;

    for (auto index = 0; index < 360 * 15 && !stopped; index++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                           brakeCircuitPressures(setup, 1.0), tick);
        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
        elapsed += tick;
        stopped = state.chassis.linearVelocity.z < 5.0;
    }

    REQUIRE(stopped);

    const auto to = state.chassis.linearVelocity.z;
    const auto deceleration = (from - to) / elapsed;
    const auto distance = state.chassis.position.z - start;

    std::printf(
        "Charger braking: %.1f -> %.1f m/s in %.2f s over %.1f m: %.2f m/s^2 mean (the profile plans with %.1f)\n",
        from, to, elapsed, distance, deceleration, PursuitOptions{}.fullModelBrakingMetresPerSecondSquared);

    // The number the profile is planned with is this car's, within a tenth.
    REQUIRE_THAT(deceleration, WithinAbs(PursuitOptions{}.fullModelBrakingMetresPerSecondSquared,
                                         0.1 * PursuitOptions{}.fullModelBrakingMetresPerSecondSquared));
}
