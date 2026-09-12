// The seam between traffic and the physics it has to be real against: the obstacle a traffic car
// offers the vehicle model, the momentum the two exchange, and the rigid body a car becomes when it
// is hit.
//
// **The first test in this file is the one that protects both frame gates.** Every contact the
// vehicle model resolves now goes through a call that did not exist before traffic did, and both
// goldens are blessed on a circuit with no traffic lanes at all. If an empty obstacle span is not
// exactly a no-op, that is a change to the player's car on every track.

#include <algorithm>
#include <cmath>
#include <cstddef>
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
using raceengine::collideBody;
using raceengine::collideObstacles;
using raceengine::CollisionBox;
using raceengine::ContactManifold;
using raceengine::ContactMaterial;
using raceengine::DriverArchetype;
using raceengine::DriverProfile;
using raceengine::DynamicObstacle;
using raceengine::generateProvingGround;
using raceengine::LaneSource;
using raceengine::noObstacle;
using raceengine::PhysicsWorld;
using raceengine::placeSimpleVehicle;
using raceengine::ProvingGroundDescriptor;
using raceengine::resolveContacts;
using raceengine::RigidBodyState;
using raceengine::simpleVehicleBox;
using raceengine::SimpleVehicleInput;
using raceengine::simpleVehicleOrigin;
using raceengine::simpleVehicleRayLength;
using raceengine::SimpleVehicleSetup;
using raceengine::SimpleVehicleState;
using raceengine::simpleVehicleWheelRays;
using raceengine::stepSimpleVehicle;
using raceengine::SurfaceHit;
using raceengine::tearDownJolt;
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

// A body whose centre of mass is at its own origin, so the collision box's centre in the body frame
// is the box's centre in the world. Everything in this file wants that and nothing here is testing
// the centre-of-mass offset, which the vehicle model's own tests already cover.
[[nodiscard]] RigidBodyState bodyAt(const glm::dvec3& centre, const double mass, const glm::dvec3& velocity = {})
{
    auto state = RigidBodyState{};

    state.position = centre;
    state.mass = mass;
    state.linearVelocity = velocity;
    state.inertia = glm::dmat3(1000.0);
    state.inverseInertia = glm::inverse(state.inertia);

    return state;
}

[[nodiscard]] DynamicObstacle carAt(const glm::dvec3& centre, const std::uint32_t id, const double mass = 1400.0)
{
    return DynamicObstacle{.id = id,
                           .centre = centre,
                           .orientation = glm::dquat(1.0, 0.0, 0.0, 0.0),
                           .halfExtents = glm::dvec3(0.90, 0.72, 2.15),
                           .centreOfMass = centre,
                           .linearVelocity = glm::dvec3(0.0),
                           .angularVelocity = glm::dvec3(0.0),
                           .inverseMass = 1.0 / mass,
                           .inverseInertia = glm::dmat3(1.0 / 2500.0)};
}

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

} // namespace

TEST_CASE("an empty obstacle span leaves a manifold exactly as it was", "[traffic][traffic-contact]")
{
    const JoltGuard jolt;

    const auto world = flatGround(200.0, 60.0);

    const auto box = CollisionBox{.centre = glm::dvec3(0.0, 0.4, 0.0), .halfExtents = glm::dvec3(0.9, 0.6, 2.1)};
    const auto state = bodyAt(glm::dvec3(0.0, 0.5, 0.0), 1400.0);

    auto manifold = collideBody(world, state, box);
    const auto before = manifold;

    collideObstacles(state, box, {}, manifold);

    // Not "about the same" — the same. The two gates are blessed on a track that hands an empty
    // span on every tick of every frame, so anything less than identity here is a change to the
    // player's car everywhere.
    REQUIRE(manifold.points.size() == before.points.size());
    REQUIRE(manifold.bodies.size() == before.bodies.size());

    for (auto index = std::size_t{0}; index < manifold.points.size(); index++)
    {
        REQUIRE(manifold.points[index].position == before.points[index].position);
        REQUIRE(manifold.points[index].normal == before.points[index].normal);
        REQUIRE(manifold.points[index].penetration == before.points[index].penetration);
        REQUIRE(manifold.points[index].body == before.points[index].body);
    }
}

TEST_CASE("two cars that are not touching produce no contact", "[traffic][traffic-contact]")
{
    const auto box = CollisionBox{.centre = glm::dvec3(0.0), .halfExtents = glm::dvec3(0.9, 0.72, 2.15)};
    const auto state = bodyAt(glm::dvec3(0.0, 0.0, 0.0), 1400.0);

    const auto obstacles = std::vector<DynamicObstacle>{carAt(glm::dvec3(0.0, 0.0, 12.0), 7)};

    auto manifold = ContactManifold{};
    collideObstacles(state, box, obstacles, manifold);

    REQUIRE(manifold.points.empty());
    REQUIRE(manifold.bodies.empty());
}

TEST_CASE("a car driven into the back of another produces a face contact", "[traffic][traffic-contact]")
{
    const auto box = CollisionBox{.centre = glm::dvec3(0.0), .halfExtents = glm::dvec3(0.9, 0.72, 2.15)};
    const auto state = bodyAt(glm::dvec3(0.0, 0.0, 0.0), 1400.0);

    // Nose to tail with a 20 cm overlap.
    const auto obstacles = std::vector<DynamicObstacle>{carAt(glm::dvec3(0.0, 0.0, 4.1), 7)};

    auto manifold = ContactManifold{};
    collideObstacles(state, box, obstacles, manifold);

    REQUIRE(manifold.points.size() > 1);

    // Element 0 is the immovable world and the obstacle is behind it, which is the table's own
    // convention and the reason a point naming index 0 against an empty table is still correct.
    REQUIRE(manifold.bodies.size() == 2);
    REQUIRE(manifold.bodies[0].obstacle == noObstacle);
    REQUIRE(manifold.bodies[1].obstacle == 7u);

    for (const auto& point : manifold.points)
    {
        REQUIRE(point.body == 1);
        REQUIRE(point.penetration > 0.0);
        REQUIRE_THAT(point.penetration, WithinAbs(0.2, 0.02));

        // Out of the obstacle and into the car, which is the direction the car has to move to
        // separate. Backwards here, because the obstacle is in front.
        REQUIRE(point.normal.z < -0.9);
    }
}

TEST_CASE("a corner clipped at an angle still reports a contact", "[traffic][traffic-contact]")
{
    const auto box = CollisionBox{.centre = glm::dvec3(0.0), .halfExtents = glm::dvec3(0.9, 0.72, 2.15)};

    auto state = bodyAt(glm::dvec3(0.0, 0.0, 0.0), 1400.0);
    state.orientation = glm::angleAxis(0.6, glm::dvec3(0.0, 1.0, 0.0));

    const auto obstacles = std::vector<DynamicObstacle>{carAt(glm::dvec3(2.2, 0.0, 2.6), 3)};

    auto manifold = ContactManifold{};
    collideObstacles(state, box, obstacles, manifold);

    REQUIRE_FALSE(manifold.points.empty());
    REQUIRE(manifold.bodies.size() == 2);
    REQUIRE(manifold.bodies[1].obstacle == 3u);
}

TEST_CASE("the solver hands the car it hit a velocity of its own", "[traffic][traffic-contact]")
{
    const auto box = CollisionBox{.centre = glm::dvec3(0.0), .halfExtents = glm::dvec3(0.9, 0.72, 2.15)};

    constexpr auto carMass = 1452.0;
    constexpr auto trafficMass = 1400.0;

    auto state = bodyAt(glm::dvec3(0.0, 0.0, 0.0), carMass, glm::dvec3(0.0, 0.0, 8.0));

    const auto obstacles = std::vector<DynamicObstacle>{carAt(glm::dvec3(0.0, 0.0, 4.2), 11, trafficMass)};

    auto manifold = ContactManifold{};
    collideObstacles(state, box, obstacles, manifold);
    REQUIRE_FALSE(manifold.points.empty());

    const auto before = state.linearVelocity.z;

    resolveContacts(state, manifold, ContactMaterial{}, tick);

    const auto& hit = manifold.bodies[1];

    // The car slowed and the thing it hit was pushed forwards.
    REQUIRE(state.linearVelocity.z < before);
    REQUIRE(hit.deltaLinear.z > 0.5);

    // And the pair kept its momentum to the newton second. This is the whole reason the second body
    // is explicit rather than a scale factor on an impulse.
    const auto ledger = carMass * (state.linearVelocity.z - before) + trafficMass * hit.deltaLinear.z;

    REQUIRE_THAT(ledger, WithinAbs(0.0, 1e-9));
}

TEST_CASE("a simple traffic vehicle stands on the road at its design height", "[traffic][traffic-dynamics]")
{
    const JoltGuard jolt;

    const auto world = flatGround(120.0, 60.0);

    const auto setup = SimpleVehicleSetup{};
    auto state = SimpleVehicleState{};

    // Dropped from fifteen centimetres, so what is measured is where it settles rather than where it
    // was put. Not further: a suspension ray is as long as the suspension, so a car dropped from
    // higher than its own travel has nothing under it until it has fallen most of the way — which is
    // correct, and is not what this test is about. The plate spans z in [0, length], so the middle
    // of it is where a car goes.
    placeSimpleVehicle(setup, state, glm::dvec3(0.0, 0.15, 60.0), glm::dquat(1.0, 0.0, 0.0, 0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0));

    auto origins = std::vector<glm::dvec3>();
    auto directions = std::vector<glm::dvec3>();
    auto hits = std::vector<SurfaceHit>();

    for (auto step = 0; step < 4 * 360; step++)
    {
        origins.clear();
        directions.clear();
        simpleVehicleWheelRays(setup, state, origins, directions);
        world.castRays(origins, directions, simpleVehicleRayLength(setup), hits);

        auto manifold = collideBody(world, state.body, simpleVehicleBox(setup));

        stepSimpleVehicle(setup, state, SimpleVehicleInput{}, hits, world.materials(), manifold, ContactMaterial{},
                          tick);
    }

    REQUIRE(state.groundedWheels == 4);

    // **The body origin is on the road, not under it.** The spring anchor carries the static
    // deflection so that a car placed at a lane point — which is authored at road level — rests
    // where it was placed. Getting this wrong buries every traffic car by its own static deflection.
    const auto origin = simpleVehicleOrigin(setup, state);
    REQUIRE_THAT(origin.y, WithinAbs(0.0, 0.02));

    REQUIRE(glm::length(state.body.linearVelocity) < 0.05);

    // Each corner carries about a quarter of the car.
    for (const auto load : state.loadNewtons)
    {
        REQUIRE_THAT(load, WithinAbs(setup.massKilograms * 9.80665 / 4.0, 200.0));
    }
}

TEST_CASE("a simple traffic vehicle brakes to a stop", "[traffic][traffic-dynamics]")
{
    const JoltGuard jolt;

    const auto world = flatGround(400.0, 60.0);

    const auto setup = SimpleVehicleSetup{};
    auto state = SimpleVehicleState{};

    placeSimpleVehicle(setup, state, glm::dvec3(0.0, 0.0, 20.0), glm::dquat(1.0, 0.0, 0.0, 0.0),
                       glm::dvec3(0.0, 0.0, 18.0), glm::dvec3(0.0));

    auto origins = std::vector<glm::dvec3>();
    auto directions = std::vector<glm::dvec3>();
    auto hits = std::vector<SurfaceHit>();

    auto input = SimpleVehicleInput{};
    input.brake = 1.0;

    for (auto step = 0; step < 10 * 360; step++)
    {
        origins.clear();
        directions.clear();
        simpleVehicleWheelRays(setup, state, origins, directions);
        world.castRays(origins, directions, simpleVehicleRayLength(setup), hits);

        auto manifold = collideBody(world, state.body, simpleVehicleBox(setup));

        stepSimpleVehicle(setup, state, input, hits, world.materials(), manifold, ContactMaterial{}, tick);
    }

    REQUIRE(glm::length(state.body.linearVelocity) < 0.2);

    // It stopped rather than being teleported: 18 m/s under a friction circle takes at least the
    // distance a full-grip stop needs.
    const auto travelled = simpleVehicleOrigin(setup, state).z - 20.0;
    REQUIRE(travelled > 12.0);
    REQUIRE(travelled < 60.0);
}

TEST_CASE("a car hit hard enough stops being a point on a lane and drives itself back", "[traffic][traffic-recovery]")
{
    const JoltGuard jolt;

    // A straight road down the middle of a plate, so a car knocked off it has somewhere to be
    // knocked to and somewhere to come back to.
    const auto world = flatGround(400.0, 120.0);

    auto line = std::vector<glm::dvec3>();
    for (auto index = 0; index <= 20; index++)
    {
        line.push_back(glm::dvec3(0.0, 0.0, 20.0 + 18.0 * static_cast<double>(index)));
    }

    auto built = buildLaneNetwork({LaneSource{.id = 1,
                                              .name = "straight",
                                              .speedLimitMetresPerSecond = 12.0,
                                              .allowLaneChanges = false,
                                              .allowUTurns = false,
                                              .points = std::move(line)}});
    REQUIRE(built.has_value());

    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = 6.0;
    options.recoverySeconds = 30.0;

    auto city = TrafficPopulation(
        std::move(built).value(), options,
        {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    city.seed();
    REQUIRE_FALSE(city.agents().empty());

    const auto& subject = city.agents().front();
    const auto id = subject.id;
    REQUIRE(subject.mode == AgentMode::Cruising);

    // A shove sideways, handed over exactly the way the contact solver hands one over: a body table
    // naming the obstacle, with the velocity change the solve settled on.
    auto manifold = ContactManifold{};
    manifold.bodies.push_back(raceengine::ContactBody{});
    manifold.bodies.push_back(raceengine::ContactBody{.obstacle = id, .deltaLinear = glm::dvec3(4.5, 0.0, 0.0)});

    city.applyContacts(manifold);

    const auto step = [&](const int steps, const double from)
    {
        for (auto index = 0; index < steps; index++)
        {
            city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                      .simulatedSeconds = from + static_cast<double>(index) * tick,
                                      .focusMetres = glm::dvec3(0.0),
                                      .vehicles = {}},
                        world);
        }
    };

    step(1, 0.0);

    REQUIRE(city.report().promoted == 1);
    REQUIRE(city.agents()[id].mode == AgentMode::Disturbed);

    // It was pushed to its left, so it has to be off the lane before recovery means anything.
    step(60, tick);
    REQUIRE(std::abs(city.agents()[id].positionMetres.x) > 0.4);

    // And within a few seconds it has steered itself back and gone back to being a point on a lane.
    step(20 * 360, 0.2);

    const auto& recovered = city.agents()[id];

    REQUIRE(recovered.mode == AgentMode::Cruising);
    REQUIRE_THAT(recovered.positionMetres.x, WithinAbs(0.0, 0.6));
    REQUIRE(recovered.speedMetresPerSecond > 1.0);
}

TEST_CASE("a car near the player is a body that follows its lane, and a point again once it is past",
          "[traffic][traffic-embodied]")
{
    const JoltGuard jolt;

    const auto world = flatGround(400.0, 120.0);

    auto line = std::vector<glm::dvec3>();
    for (auto index = 0; index <= 20; index++)
    {
        line.push_back(glm::dvec3(0.0, 0.0, 20.0 + 18.0 * static_cast<double>(index)));
    }

    auto built = buildLaneNetwork({LaneSource{.id = 1,
                                              .name = "straight",
                                              .speedLimitMetresPerSecond = 12.0,
                                              .allowLaneChanges = false,
                                              .allowUTurns = false,
                                              .points = std::move(line)}});
    REQUIRE(built.has_value());

    // Dense enough that cars keep coming past the player for the whole run, and the near tier on:
    // a body inside thirty metres, a point again outside forty.
    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = 26.0;
    options.embodyRadiusMetres = 30.0;
    options.disembodyRadiusMetres = 40.0;

    auto city = TrafficPopulation(
        std::move(built).value(), options,
        {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    city.seed();
    REQUIRE(city.agents().size() > 5);

    // The player, parked on the verge half way along and looking up the road. The tier is keyed on
    // there being one: the same run with no vehicle is the point tier throughout.
    const auto player = std::vector{TrafficVehicle{.positionMetres = glm::dvec3(6.0, 0.0, 200.0),
                                                   .velocityMetresPerSecond = glm::dvec3(0.0),
                                                   .forward = glm::dvec3(0.0, 0.0, 1.0),
                                                   .lengthMetres = 4.3}};

    auto previous = std::vector<glm::dvec3>();
    auto everEmbodied = std::vector<bool>(city.agents().size(), false);
    for (const auto& agent : city.agents())
    {
        previous.push_back(agent.positionMetres);
    }

    auto embodiedTicks = std::size_t{0};
    // A body's distance off its lane (the lane is x = 0), its forward against the road's +z, and
    // the most any car moved in one tick — across both hand-overs, which is where a snap would show.
    auto worstOffLane = 0.0;
    auto worstHeading = 1.0;
    auto worstJump = 0.0;

    for (auto step = 0; step < 12 * 360; step++)
    {
        const auto& report = city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                                       .simulatedSeconds = static_cast<double>(step) * tick,
                                                       .focusMetres = player.front().positionMetres,
                                                       .vehicles = player},
                                         world);

        embodiedTicks += report.embodied > 0 ? 1 : 0;

        const auto agents = city.agents();
        for (auto index = std::size_t{0}; index < agents.size(); index++)
        {
            const auto& agent = agents[index];

            worstJump = std::max(worstJump, glm::distance(agent.positionMetres, previous[index]));
            previous[index] = agent.positionMetres;

            if (agent.mode != AgentMode::Embodied)
            {
                continue;
            }

            everEmbodied[index] = true;
            worstOffLane = std::max(worstOffLane, std::abs(agent.positionMetres.x));
            worstHeading = std::min(worstHeading, (agent.orientation * glm::dvec3(0.0, 0.0, 1.0)).z);
        }
    }

    // Cars were bodies for a good part of the run, none was refused one, and while a body a car
    // stayed on its lane and pointed down it.
    REQUIRE(embodiedTicks > 3 * 360);
    REQUIRE(city.report().refused == 0);
    REQUIRE(worstOffLane < 0.5);
    REQUIRE(worstHeading > 0.98);

    // Neither hand-over moved a car. At 12 m/s a tick is 33 mm; a hand-over that snapped is a metre.
    REQUIRE(worstJump < 0.25);

    // Cars went through the tier and out the other side: more than one was a body, and at least one
    // of those is a point again now.
    auto passedThrough = std::size_t{0};
    auto wasEmbodied = std::size_t{0};

    for (auto index = std::size_t{0}; index < everEmbodied.size(); index++)
    {
        wasEmbodied += everEmbodied[index] ? 1 : 0;
        passedThrough += everEmbodied[index] && city.agents()[index].mode == AgentMode::Cruising ? 1 : 0;
    }

    REQUIRE(wasEmbodied >= 2);
    REQUIRE(passedThrough >= 1);

    // And the tiers are where the radii say: bodies inside the way out, points outside the way in.
    // The plan the radii are read against sits within the leash of the body, hence the slack.
    for (const auto& agent : city.agents())
    {
        const auto distance = glm::distance(agent.positionMetres, player.front().positionMetres);

        if (agent.mode == AgentMode::Embodied)
        {
            REQUIRE(distance < 40.0 + 3.0);
        }

        if (agent.mode == AgentMode::Cruising)
        {
            REQUIRE(distance > 30.0 - 3.0);
        }
    }
}
