// The drivers: what the archetypes actually do when a city of them is run, as opposed to what their
// parameter tables say.
//
// **Each of these is a population statistic rather than an assertion about one car**, and that is
// deliberate. Where a car is seeded is a draw from a generator, so "the third agent changes lanes at
// twelve seconds" is a fact about a seed and not about a driver model. What is a fact about the
// model is that a hundred and thirty cautious drivers at one density flow slower than a hundred and
// thirty aggressive ones, and that cautious drivers on a road that never runs out never change lanes
// at all.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::AgentMode;
using raceengine::behaviourFor;
using raceengine::bringUpJolt;
using raceengine::buildLaneNetwork;
using raceengine::DriverArchetype;
using raceengine::DriverProfile;
using raceengine::generateProvingGround;
using raceengine::laneLength;
using raceengine::neighbourShiftAt;
using raceengine::wrapDistance;
using raceengine::LaneSource;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::tearDownJolt;
using raceengine::TrafficPopulation;
using raceengine::TrafficPopulationOptions;
using raceengine::TrafficSignal;
using raceengine::TrafficUpdate;

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

// The simulation's own tick, which is what a city is advanced at in the game.
constexpr auto tick = 1.0 / 360.0;

[[nodiscard]] std::vector<glm::dvec3> ring(const double radius, const std::size_t points)
{
    auto circle = std::vector<glm::dvec3>();
    circle.reserve(points + 1);

    for (auto index = std::size_t{0}; index <= points; index++)
    {
        const auto angle = 2.0 * std::numbers::pi * static_cast<double>(index % points) / static_cast<double>(points);

        circle.push_back(glm::dvec3(radius * std::cos(angle), 0.0, radius * std::sin(angle)));
    }

    return circle;
}

[[nodiscard]] LaneSource lane(const int id, std::vector<glm::dvec3> points, const double limit)
{
    return LaneSource{.id = id,
                      .name = "ring",
                      .speedLimitMetresPerSecond = limit,
                      .allowLaneChanges = true,
                      .allowUTurns = false,
                      .points = std::move(points)};
}

// One archetype on its own, so what is measured is that driver rather than a mixture.
[[nodiscard]] std::vector<DriverProfile> only(const DriverArchetype archetype)
{
    return {DriverProfile{.name = std::string(raceengine::archetypeName(archetype)),
                          .weight = 1.0,
                          .behaviour = behaviourFor(archetype)}};
}

// A world the traffic never queries. **A precondition of every test in this file except the last
// one**: `TrafficPopulation::update` reaches the world only for cars that have been hit, and none of
// these hits anything, so a plate the size of a tennis court is the honest fixture rather than a
// three hundred metre one that pretends to be the road.
[[nodiscard]] PhysicsWorld unusedWorld()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 40.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 4.0;
    descriptor.features = {};

    auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());

    auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    return std::move(world).value();
}

[[nodiscard]] double meanSpeed(const TrafficPopulation& city)
{
    auto total = 0.0;
    auto counted = std::size_t{0};

    for (const auto& agent : city.agents())
    {
        if (agent.mode != AgentMode::Cruising)
        {
            continue;
        }

        total += agent.speedMetresPerSecond;
        counted++;
    }

    return counted == 0 ? 0.0 : total / static_cast<double>(counted);
}

// The smallest bumper-to-bumper gap anywhere on the one lane, metres. Negative means two cars are
// occupying the same road.
[[nodiscard]] double closestGap(const TrafficPopulation& city, const double carLength)
{
    auto along = std::vector<double>();

    for (const auto& agent : city.agents())
    {
        along.push_back(agent.distanceMetres);
    }

    if (along.size() < 2)
    {
        return carLength;
    }

    std::ranges::sort(along);

    auto smallest = along.front() + laneLength(city.network().lanes.front()) - along.back();

    for (auto index = std::size_t{1}; index < along.size(); index++)
    {
        smallest = std::min(smallest, along[index] - along[index - 1]);
    }

    return smallest - carLength;
}

// Run a single-ring city and hand back the mean speed it settles at.
[[nodiscard]] double settledSpeed(const DriverArchetype archetype, const double density, const double limit,
                                  double& worstGap, std::size_t& cars)
{
    const auto world = unusedWorld();

    auto built = buildLaneNetwork({lane(1, ring(300.0, 240), limit)});
    REQUIRE(built.has_value());
    REQUIRE(built->lanes.front().loop);

    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = density;
    options.maximumAgents = 400;

    auto city = TrafficPopulation(std::move(built).value(), options, only(archetype));
    city.seed();

    cars = city.agents().size();
    REQUIRE(cars > 4);

    const auto carLength = 2.0 * options.vehicle.bodyHalfExtentsMetres.z;
    worstGap = carLength;

    // Ninety seconds, which is several times the time it takes a queue on a ring to settle.
    for (auto step = 0; step < 90 * 360; step++)
    {
        city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                  .simulatedSeconds = static_cast<double>(step) * tick,
                                  .focusMetres = glm::dvec3(0.0),
                                  .vehicles = {}},
                    world);

        if (step > 10 * 360)
        {
            worstGap = std::min(worstGap, closestGap(city, carLength));
        }
    }

    return meanSpeed(city);
}

} // namespace

TEST_CASE("free-flowing traffic runs at each archetype's own speed", "[traffic][traffic-drivers]")
{
    const JoltGuard jolt;

    // Four cars a kilometre: nobody is holding anybody up, so what is measured is what each driver
    // wants rather than what the road allows.
    auto worst = 0.0;
    auto cars = std::size_t{0};

    const auto cautious = settledSpeed(DriverArchetype::Cautious, 4.0, 14.0, worst, cars);
    const auto regular = settledSpeed(DriverArchetype::Regular, 4.0, 14.0, worst, cars);
    const auto aggressive = settledSpeed(DriverArchetype::Aggressive, 4.0, 14.0, worst, cars);

    // The stated ten km/h either side of the limit, and the spread each driver draws from is under
    // 1.5 m/s so the ordering is not a coincidence of one seed.
    REQUIRE_THAT(cautious, WithinAbs(14.0 - 2.8, 1.2));
    REQUIRE_THAT(regular, WithinAbs(14.0, 1.2));
    REQUIRE_THAT(aggressive, WithinAbs(14.0 + 2.8, 1.2));

    REQUIRE(cautious < regular);
    REQUIRE(regular < aggressive);
}

TEST_CASE("at one density the tailgaters flow faster than the cautious", "[traffic][traffic-drivers]")
{
    const JoltGuard jolt;

    // Seventy cars a kilometre on a 1.88 km ring is about fourteen metres of road each, which is
    // dense enough that nobody reaches the speed they want and the headway is what decides.
    auto cautiousGap = 0.0;
    auto aggressiveGap = 0.0;
    auto cautiousCars = std::size_t{0};
    auto aggressiveCars = std::size_t{0};

    const auto cautious = settledSpeed(DriverArchetype::Cautious, 70.0, 14.0, cautiousGap, cautiousCars);
    const auto aggressive = settledSpeed(DriverArchetype::Aggressive, 70.0, 14.0, aggressiveGap, aggressiveCars);

    // The two cities have to be the same city for the comparison to mean anything. The count is
    // short of the density's own arithmetic because a car is only placed where there is twelve
    // metres of clear road for it, which on a ring this size runs out first — that is the
    // population the network can hold, and it is the same for both.
    REQUIRE(cautiousCars > 60);
    REQUIRE_THAT(static_cast<double>(aggressiveCars), WithinAbs(static_cast<double>(cautiousCars), 12.0));

    REQUIRE(aggressive > cautious + 2.0);

    // **And nobody drives through anybody**, in either city, at any time after the first ten
    // seconds. This is the assertion that says the car-following model is a model rather than a
    // decoration: a gap that goes negative is two cars in the same place.
    REQUIRE(cautiousGap > 0.0);
    REQUIRE(aggressiveGap > 0.0);
}

TEST_CASE("cautious drivers do not change lanes and aggressive ones do", "[traffic][traffic-drivers]")
{
    const JoltGuard jolt;

    const auto world = unusedWorld();

    // Two concentric rings 3.7 m apart, which is one road with two lanes on it. Both close on
    // themselves, so neither ever runs out — and a road that never runs out is the one place where
    // "does not change lanes unless they have to" can be tested against never.
    const auto twoLanes = []
    {
        auto built = buildLaneNetwork({lane(1, ring(300.0, 240), 14.0), lane(2, ring(303.7, 240), 14.0)});
        REQUIRE(built.has_value());
        REQUIRE(built->lanes[0].loop);
        REQUIRE(built->lanes[1].loop);
        REQUIRE_FALSE(built->lanes[0].neighbours.empty());

        return std::move(built).value();
    };

    const auto changesUnder = [&](const DriverArchetype archetype)
    {
        auto options = TrafficPopulationOptions{};
        options.densityPerKilometre = 26.0;

        auto city = TrafficPopulation(twoLanes(), options, only(archetype));
        city.seed();

        REQUIRE(city.agents().size() > 60);

        auto changing = std::size_t{0};

        for (auto step = 0; step < 60 * 360; step++)
        {
            const auto& report = city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                                           .simulatedSeconds = static_cast<double>(step) * tick,
                                                           .focusMetres = glm::dvec3(0.0),
                                                           .vehicles = {}},
                                             world);

            changing += report.changingLanes;
        }

        return changing;
    };

    REQUIRE(changesUnder(DriverArchetype::Cautious) == 0);
    REQUIRE(changesUnder(DriverArchetype::Aggressive) > 0);
}

TEST_CASE("traffic stops at a red light and the aggressive driver leaves first",
          "[traffic][traffic-drivers][traffic-signals]")
{
    const JoltGuard jolt;

    const auto world = unusedWorld();

    // Green for ten, amber for two, red for twelve, so the light turns green on every multiple of
    // twenty-four seconds. The queue is measured across one of those edges well after the city has
    // settled.
    constexpr auto stopLine = 900.0;
    constexpr auto cycle = 24.0;
    constexpr auto greenEdge = 4.0 * cycle;

    const auto queueAt = [&](const DriverArchetype archetype, const double afterGreen)
    {
        auto built = buildLaneNetwork({lane(1, ring(300.0, 240), 14.0)});
        REQUIRE(built.has_value());

        built->signals.push_back(TrafficSignal{.lane = 0,
                                               .distanceMetres = stopLine,
                                               .greenSeconds = 10.0,
                                               .amberSeconds = 2.0,
                                               .redSeconds = 12.0,
                                               .offsetSeconds = 0.0});

        auto options = TrafficPopulationOptions{};
        options.densityPerKilometre = 26.0;

        auto city = TrafficPopulation(std::move(built).value(), options, only(archetype));
        city.seed();

        const auto steps = static_cast<int>((greenEdge + afterGreen) / tick);
        for (auto step = 0; step < steps; step++)
        {
            city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                      .simulatedSeconds = static_cast<double>(step) * tick,
                                      .focusMetres = glm::dvec3(0.0),
                                      .vehicles = {}},
                        world);
        }

        // The car at the front of the queue: the one closest behind the line.
        const auto* front = static_cast<const raceengine::TrafficAgent*>(nullptr);
        auto best = 60.0;

        for (const auto& agent : city.agents())
        {
            const auto behind = stopLine - agent.distanceMetres;
            if (behind < 0.0 || behind > best)
            {
                continue;
            }

            best = behind;
            front = &agent;
        }

        REQUIRE(front != nullptr);

        return front->speedMetresPerSecond;
    };

    // **The precondition, asserted rather than assumed**: half a second *before* the light changes,
    // the front car of each queue is stopped. Without this the comparison below would pass on a
    // city where nobody ever reached the light.
    REQUIRE(queueAt(DriverArchetype::Cautious, -0.5) < 0.3);
    REQUIRE(queueAt(DriverArchetype::Aggressive, -0.5) < 0.3);

    // Six tenths of a second after green. The aggressive driver's stated delay is 0.15 s and the
    // cautious one's is 1.6 s, so one of them is away and the other has not moved.
    const auto cautious = queueAt(DriverArchetype::Cautious, 0.6);
    const auto aggressive = queueAt(DriverArchetype::Aggressive, 0.6);

    REQUIRE(cautious < 0.05);
    REQUIRE(aggressive > 0.6);
}

TEST_CASE("a lane change onto a loop whose origin lies inside the run lands beside the car", "[traffic][traffic-drivers]")
{
    const JoltGuard jolt;

    const auto world = unusedWorld();

    // Two concentric rings 3.7 m apart, the outer one's point list started half way round, so the
    // outer lane's origin falls in the middle of the inner lane's run beside it. This is Grand City
    // Parkway's shape: two long carriageways beside a loop the whole way round it, with their origins
    // wherever the export happened to start them.
    auto outer = ring(303.7, 240);
    outer.pop_back();
    std::ranges::rotate(outer, outer.begin() + 120);
    outer.push_back(outer.front());

    auto built = buildLaneNetwork({lane(1, ring(300.0, 240), 14.0), lane(2, std::move(outer), 14.0)});
    REQUIRE(built.has_value());
    REQUIRE(built->lanes[0].loop);
    REQUIRE(built->lanes[1].loop);
    REQUIRE_FALSE(built->lanes[0].neighbours.empty());

    // The run's shift names the same place on the neighbour everywhere along the run — read at the
    // car's own distance, and brought onto the loop, never the loop's far end. Averaged without
    // unwrapping it named a point on the other side of the ring, and a car changing lanes there set
    // off across the middle of it; averaged at all, the outer ring's extra per cent of length put
    // the place twelve metres along from the car at the run's ends.
    for (const auto& run : built->lanes[0].neighbours)
    {
        for (auto distance = run.fromMetres; distance <= run.toMetres; distance += 25.0)
        {
            const auto here = raceengine::laneAt(built->lanes[0], distance);
            const auto there = raceengine::laneAt(
                built->lanes[1], wrapDistance(built->lanes[1], distance + neighbourShiftAt(run, distance)));
            REQUIRE(here.has_value());
            REQUIRE(there.has_value());
            REQUIRE(glm::distance(here->positionMetres, there->positionMetres) < 4.5);
        }
    }

    // And driven: aggressive drivers change lanes for a minute, and no car moves further in one tick
    // than a car can.
    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = 26.0;

    auto city = TrafficPopulation(std::move(built).value(), options, only(DriverArchetype::Aggressive));
    city.seed();

    auto previous = std::vector<glm::dvec3>();
    for (const auto& agent : city.agents())
    {
        previous.push_back(agent.positionMetres);
    }

    auto changes = std::size_t{0};
    auto worstJump = 0.0;

    for (auto step = 0; step < 60 * 360; step++)
    {
        const auto& report = city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                                       .simulatedSeconds = static_cast<double>(step) * tick,
                                                       .focusMetres = glm::dvec3(0.0),
                                                       .vehicles = {}},
                                         world);
        changes += report.changingLanes;

        const auto agents = city.agents();
        for (auto index = std::size_t{0}; index < agents.size(); index++)
        {
            worstJump = std::max(worstJump, glm::distance(agents[index].positionMetres, previous[index]));
            previous[index] = agents[index].positionMetres;
        }
    }

    REQUIRE(changes > 0);
    REQUIRE(worstJump < 0.3);
}

TEST_CASE("a neighbour that runs beside a lane twice is two runs, and a join is eased across",
          "[traffic][traffic-drivers]")
{
    const JoltGuard jolt;

    const auto world = unusedWorld();

    const auto straight = [](const glm::dvec3& from, const glm::dvec3& to, const std::size_t points,
                             std::vector<glm::dvec3>& into)
    {
        for (auto index = std::size_t{0}; index < points; index++)
        {
            const auto along = static_cast<double>(index) / static_cast<double>(points - 1);

            into.push_back(from + (to - from) * along);
        }
    };

    // Lane A runs 600 m up +z. Lane B is Grand City Parkway's lane 8 in miniature: it starts beside A
    // half way up, leaves at A's end on a long detour and comes back to run beside A from the bottom,
    // its own end merging beside its own start — so between z = 150 and z = 200 both passes of B stand
    // beside A, 3.7 m and 4.2 m out, and the nearest sample flips from one pass to the other between
    // two of A's samples: a step of two kilometres in the shift onto B. Each lane's end runs into the
    // other, so the two make a circuit every car goes round, crossing both joins.
    auto a = std::vector<glm::dvec3>();
    straight(glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 600.0), 31, a);

    // B runs on past A's end before it turns away, so A's join lands on a straight of B: a join onto
    // a corner carries its residual in a frame the corner has turned, and the car lands up to the
    // gap times the sine of half the kink along the road — a limit recorded in the brief, not a case
    // the parkway's seven joins reach.
    auto b = std::vector<glm::dvec3>();
    straight(glm::dvec3(3.7, 0.0, 150.0), glm::dvec3(3.7, 0.0, 700.0), 12, b);
    b.push_back(glm::dvec3(60.0, 0.0, 760.0));
    b.push_back(glm::dvec3(500.0, 0.0, 760.0));
    b.push_back(glm::dvec3(500.0, 0.0, -60.0));
    b.push_back(glm::dvec3(60.0, 0.0, -60.0));
    straight(glm::dvec3(4.2, 0.0, 0.0), glm::dvec3(4.2, 0.0, 200.0), 6, b);

    auto built = buildLaneNetwork({lane(1, std::move(a), 14.0), lane(2, std::move(b), 14.0)});
    REQUIRE(built.has_value());
    REQUIRE_FALSE(built->lanes[0].loop);
    REQUIRE_FALSE(built->lanes[1].loop);
    REQUIRE(built->lanes[0].successors.size() == 1);
    REQUIRE(built->lanes[1].successors.size() == 1);

    // A stands beside both passes of B and B beside A twice: two runs each, never one with a step in
    // it. Before 2026-09-11 A carried one run, and a car changing lanes where its shift stepped was
    // blended toward a point sweeping the far pass at two hundred metres a second.
    REQUIRE(built->lanes[0].neighbours.size() == 2);
    REQUIRE(built->lanes[1].neighbours.size() == 2);

    // Read at every metre of every run, the same place on the neighbour is beside the car, and the
    // shift never steps between two samples.
    for (auto laneIndex = std::size_t{0}; laneIndex < 2; laneIndex++)
    {
        const auto& own = built->lanes[laneIndex];

        for (const auto& run : own.neighbours)
        {
            for (auto index = std::size_t{1}; index < run.shiftMetres.size(); index++)
            {
                REQUIRE(std::abs(run.shiftMetres[index] - run.shiftMetres[index - 1]) < 10.0);
            }

            const auto& other = built->lanes[run.lane];

            for (auto distance = run.fromMetres; distance <= run.toMetres; distance += 1.0)
            {
                const auto here = raceengine::laneAt(own, distance);
                const auto there =
                    raceengine::laneAt(other, wrapDistance(other, distance + neighbourShiftAt(run, distance)));
                INFO("lane " << laneIndex << " run onto " << run.lane << " from " << run.fromMetres << " to "
                             << run.toMetres << " at " << distance << " shift " << neighbourShiftAt(run, distance)
                             << " samples " << run.shiftMetres.size());
                REQUIRE(here.has_value());
                REQUIRE(there.has_value());
                REQUIRE(glm::distance(here->positionMetres, there->positionMetres) < 6.0);
            }
        }
    }

    // And driven for a minute: aggressive drivers change lanes, every car crosses both joins — where
    // a lane's last point stands 3.7 and 4.2 m from the lane it runs into, and the car used to be put
    // on the new lane's centre in one tick — and no car moves further in a tick than a car can.
    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = 26.0;

    auto city = TrafficPopulation(std::move(built).value(), options, only(DriverArchetype::Aggressive));
    city.seed();

    auto previous = std::vector<raceengine::TrafficAgent>(city.agents().begin(), city.agents().end());

    auto changes = std::size_t{0};
    auto worstJump = 0.0;
    auto worstBefore = raceengine::TrafficAgent{};
    auto worstAfter = raceengine::TrafficAgent{};
    auto worstStep = 0;

    for (auto step = 0; step < 60 * 360; step++)
    {
        const auto& report = city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                                       .simulatedSeconds = static_cast<double>(step) * tick,
                                                       .focusMetres = glm::dvec3(0.0),
                                                       .vehicles = {}},
                                         world);
        changes += report.changingLanes;

        const auto agents = city.agents();
        for (auto index = std::size_t{0}; index < agents.size(); index++)
        {
            const auto jump = glm::distance(agents[index].positionMetres, previous[index].positionMetres);
            if (jump > worstJump)
            {
                worstJump = jump;
                worstBefore = previous[index];
                worstAfter = agents[index];
                worstStep = step;
            }

            previous[index] = agents[index];
        }
    }

    REQUIRE(changes > 0);
    INFO("worst at step " << worstStep << ": lane " << worstBefore.lane << " -> " << worstAfter.lane << ", distance "
                          << worstBefore.distanceMetres << " -> " << worstAfter.distanceMetres << ", target "
                          << worstBefore.changeTarget << " -> " << worstAfter.changeTarget << ", progress "
                          << worstBefore.changeProgress << " -> " << worstAfter.changeProgress << ", shift "
                          << worstBefore.changeShiftMetres << " -> " << worstAfter.changeShiftMetres << ", settle "
                          << worstBefore.settleLateralMetres << " -> " << worstAfter.settleLateralMetres << ", at ("
                          << worstBefore.positionMetres.x << ", " << worstBefore.positionMetres.z << ") -> ("
                          << worstAfter.positionMetres.x << ", " << worstAfter.positionMetres.z << ")");
    REQUIRE(worstJump < 0.3);
}
