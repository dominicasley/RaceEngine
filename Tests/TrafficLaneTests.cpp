// The lane network, and the three things it derives that no export states: which lanes close on
// themselves, which run into which, and which run alongside which.
//
// **Every fixture here is built in code and nothing reads a file.** That is the whole reason the
// traffic model sits in the engine rather than in the sandbox beside the loader: the derivations
// below are the part most likely to be wrong on a map nobody has looked at, and a test that needed
// Grand City Parkway's export on disk could only ever check the one map this project happens to
// carry.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::advanceAlongLane;
using raceengine::behaviourFor;
using raceengine::buildLaneNetwork;
using raceengine::DriverArchetype;
using raceengine::laneAt;
using raceengine::laneLength;
using raceengine::wrapDistance;
using raceengine::laneSide;
using raceengine::LaneSource;
using raceengine::projectOntoNetwork;
using raceengine::signalAspect;
using raceengine::SignalAspect;
using raceengine::signalGreenFor;
using raceengine::TrafficSignal;

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{

// A straight run of control points. Sparse on purpose — the export's are, and every helper here
// treats a lane as a chord polyline for exactly that reason.
[[nodiscard]] std::vector<glm::dvec3> straight(const glm::dvec3& from, const glm::dvec3& to, const std::size_t points)
{
    auto line = std::vector<glm::dvec3>();
    line.reserve(points);

    for (auto index = std::size_t{0}; index < points; index++)
    {
        const auto along = static_cast<double>(index) / static_cast<double>(points - 1);

        line.push_back(from + (to - from) * along);
    }

    return line;
}

// A closed ring, last point on top of the first. Smooth through the closure, which is what makes it
// a loop a car can drive round rather than a lane that happens to end near where it began.
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

[[nodiscard]] LaneSource lane(const int id, std::vector<glm::dvec3> points, const double limit = 14.0)
{
    return LaneSource{.id = id,
                      .name = "lane " + std::to_string(id),
                      .speedLimitMetresPerSecond = limit,
                      .allowLaneChanges = true,
                      .allowUTurns = false,
                      .points = std::move(points)};
}

} // namespace

TEST_CASE("a lane is sampled along its own chords", "[traffic][traffic-lanes]")
{
    const auto built = buildLaneNetwork({lane(1, straight(glm::dvec3(0.0), glm::dvec3(0.0, 0.0, 400.0), 5))});
    REQUIRE(built.has_value());

    const auto& road = built->lanes.front();

    REQUIRE_THAT(laneLength(road), WithinAbs(400.0, 1e-9));

    const auto middle = laneAt(road, 200.0);
    REQUIRE(middle.has_value());
    REQUIRE_THAT(middle->positionMetres.z, WithinAbs(200.0, 1e-9));
    REQUIRE_THAT(middle->direction.z, WithinAbs(1.0, 1e-12));

    // Clamped at both ends rather than extrapolated: a lane has a beginning and an end, and a
    // caller asking past either wants the end rather than a point in a field.
    const auto past = laneAt(road, 900.0);
    REQUIRE(past.has_value());
    REQUIRE_THAT(past->distanceMetres, WithinAbs(400.0, 1e-9));
}

TEST_CASE("the driver's left is the +x axis", "[traffic][traffic-lanes]")
{
    // The one place the network depends on the frame being right-handed with +y up. Get it backwards
    // and every lane change in the city goes the wrong way, which looks like a data fault for a day.
    const auto side = laneSide(glm::dvec3(0.0, 0.0, 1.0));

    REQUIRE_THAT(side.x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(side.y, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(side.z, WithinAbs(0.0, 1e-12));
}

TEST_CASE("two parallel lanes are paired, on the correct side and with the correct shift", "[traffic][traffic-lanes]")
{
    // Both running +z, the second 3.7 m to +x — which is the first lane's LEFT.
    auto sources =
        std::vector<LaneSource>{lane(1, straight(glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 600.0), 31)),
                                lane(2, straight(glm::dvec3(3.7, 0.0, 50.0), glm::dvec3(3.7, 0.0, 650.0), 31))};

    const auto built = buildLaneNetwork(std::move(sources));
    REQUIRE(built.has_value());

    const auto& first = built->lanes[0];
    const auto& second = built->lanes[1];

    REQUIRE(first.neighbours.size() == 1);
    REQUIRE(second.neighbours.size() == 1);

    REQUIRE(first.neighbours.front().lane == 1);
    REQUIRE_THAT(first.neighbours.front().side, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(first.neighbours.front().offsetMetres, WithinAbs(3.7, 0.05));

    // The second lane starts 50 m further up the road, so the same place on it is 50 m *less* far
    // along. A shift that came out zero would put every lane change into the wrong gap.
    REQUIRE_THAT(first.neighbours.front().distanceShiftMetres, WithinAbs(-50.0, 0.5));

    // And the pairing is symmetric, with the sides opposite.
    REQUIRE(second.neighbours.front().lane == 0);
    REQUIRE_THAT(second.neighbours.front().side, WithinAbs(-1.0, 1e-12));
    REQUIRE_THAT(second.neighbours.front().distanceShiftMetres, WithinAbs(50.0, 0.5));
}

TEST_CASE("lanes further apart than a lane's width are not neighbours", "[traffic][traffic-lanes]")
{
    // Nine metres is two roads, not two lanes of one road. The band is what stops a city's opposite
    // carriageways being offered to each other as lane changes.
    auto sources =
        std::vector<LaneSource>{lane(1, straight(glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 600.0), 31)),
                                lane(2, straight(glm::dvec3(9.0, 0.0, 0.0), glm::dvec3(9.0, 0.0, 600.0), 31))};

    const auto built = buildLaneNetwork(std::move(sources));
    REQUIRE(built.has_value());

    REQUIRE(built->lanes[0].neighbours.empty());
    REQUIRE(built->lanes[1].neighbours.empty());
}

TEST_CASE("a lane that closes on itself is driven round", "[traffic][traffic-lanes]")
{
    const auto built = buildLaneNetwork({lane(1, ring(120.0, 48))});
    REQUIRE(built.has_value());

    const auto& road = built->lanes.front();
    REQUIRE(road.loop);

    const auto total = laneLength(road);

    // Ten metres past the end comes out ten metres past the start, and the step says the lane
    // changed under the car so a caller keeping per-lane ordering knows to rebuild it.
    const auto step = advanceAlongLane(built.value(), 0, total - 4.0, 14.0);

    REQUIRE_FALSE(step.ranOut);
    REQUIRE(step.changedLane);
    REQUIRE(step.lane == 0);
    REQUIRE_THAT(step.distanceMetres, WithinAbs(10.0, 1e-9));
}

TEST_CASE("a loop whose ends do not meet is driven across the gap", "[traffic][traffic-lanes]")
{
    // A ring with its last chord missing, the shape three of Grand City Parkway's lanes are: the end
    // stands 4 to 15 metres short of the start, pointing at it.
    auto open = ring(120.0, 48);
    open.pop_back();
    const auto gap = glm::distance(open.back(), open.front());
    REQUIRE(gap > 10.0);

    const auto built = buildLaneNetwork({lane(1, std::move(open))});
    REQUIRE(built.has_value());

    const auto& road = built->lanes.front();
    REQUIRE(road.loop);

    // The closing chord is road: the length includes it, a distance on it is a point on it, and a
    // step that runs off the end comes out past the start by what is left — no hop.
    const auto arc = road.arcLengthMetres.back();
    const auto total = laneLength(road);
    REQUIRE_THAT(total, WithinAbs(arc + gap, 1e-9));

    const auto midway = raceengine::laneAt(road, arc + 0.5 * gap);
    REQUIRE(midway.has_value());
    REQUIRE_THAT(glm::distance(midway->positionMetres, 0.5 * (road.points.back() + road.points.front())),
                 WithinAbs(0.0, 1e-9));

    const auto justBefore = raceengine::laneAt(road, total - 1e-6);
    REQUIRE(justBefore.has_value());
    REQUIRE_THAT(glm::distance(justBefore->positionMetres, road.points.front()), WithinAbs(0.0, 1e-4));

    const auto step = advanceAlongLane(built.value(), 0, arc - 1.0, gap + 4.0);
    REQUIRE_FALSE(step.ranOut);
    REQUIRE(step.changedLane);
    REQUIRE_THAT(step.distanceMetres, WithinAbs(3.0, 1e-9));

    // And a computed distance past the end, or before the start, is brought back onto the loop.
    REQUIRE_THAT(wrapDistance(road, total + 7.0), WithinAbs(7.0, 1e-9));
    REQUIRE_THAT(wrapDistance(road, -7.0), WithinAbs(total - 7.0, 1e-9));
}

TEST_CASE("a lane whose ends meet at right angles is not a loop", "[traffic][traffic-lanes]")
{
    // Four of Grand City Parkway's twelve lanes are this shape: they end within a few metres of
    // their own start and point somewhere else entirely. Wrapping one puts a car through a building.
    auto square =
        std::vector<glm::dvec3>{glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 200.0), glm::dvec3(200.0, 0.0, 200.0),
                                glm::dvec3(200.0, 0.0, 0.0), glm::dvec3(3.0, 0.0, 0.0)};

    const auto built = buildLaneNetwork({lane(1, std::move(square))});
    REQUIRE(built.has_value());

    REQUIRE_FALSE(built->lanes.front().loop);
}

TEST_CASE("a lane that runs into another is joined to it", "[traffic][traffic-lanes]")
{
    // The second lane starts 5 m before the first one ends, close enough alongside to be the same
    // road. Every carriageway on Grand City Parkway ends this way.
    auto sources =
        std::vector<LaneSource>{lane(1, straight(glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 200.0), 11)),
                                lane(2, straight(glm::dvec3(0.2, 0.0, 195.0), glm::dvec3(0.2, 0.0, 600.0), 21))};

    const auto built = buildLaneNetwork(std::move(sources));
    REQUIRE(built.has_value());

    const auto& first = built->lanes[0];
    REQUIRE(first.successors.size() == 1);
    REQUIRE(first.successors.front().lane == 1);
    REQUIRE_THAT(first.successors.front().distanceMetres, WithinAbs(5.0, 0.5));

    // And a car driving off the end of the first arrives on the second.
    const auto step = advanceAlongLane(built.value(), 0, 198.0, 10.0);

    REQUIRE_FALSE(step.ranOut);
    REQUIRE(step.lane == 1);
    REQUIRE_THAT(step.distanceMetres, WithinAbs(13.0, 0.5));
}

TEST_CASE("a lane that runs into nothing runs out", "[traffic][traffic-lanes]")
{
    const auto built = buildLaneNetwork({lane(1, straight(glm::dvec3(0.0), glm::dvec3(0.0, 0.0, 200.0), 11))});
    REQUIRE(built.has_value());

    const auto step = advanceAlongLane(built.value(), 0, 195.0, 20.0);

    REQUIRE(step.ranOut);
    REQUIRE_THAT(step.distanceMetres, WithinAbs(200.0, 1e-9));
}

TEST_CASE("a point is put back onto the lane it is on", "[traffic][traffic-lanes]")
{
    auto sources =
        std::vector<LaneSource>{lane(1, straight(glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 600.0), 31)),
                                lane(2, straight(glm::dvec3(3.7, 0.0, 0.0), glm::dvec3(3.7, 0.0, 600.0), 31))};

    const auto built = buildLaneNetwork(std::move(sources));
    REQUIRE(built.has_value());

    // A metre to the left of the first lane, 123 m along it. The second lane is 2.7 m further left
    // and must not win.
    const auto found = projectOntoNetwork(built.value(), glm::dvec3(1.0, 0.0, 123.0), glm::dvec3(0.0, 0.0, 1.0), 6.0);

    REQUIRE(found.found);
    REQUIRE(found.lane == 0);
    REQUIRE_THAT(found.distanceMetres, WithinAbs(123.0, 0.01));
    REQUIRE_THAT(found.lateralMetres, WithinAbs(1.0, 0.01));

    // And a car pointing the other way is not on this lane at all.
    const auto against =
        projectOntoNetwork(built.value(), glm::dvec3(1.0, 0.0, 123.0), glm::dvec3(0.0, 0.0, -1.0), 6.0);
    REQUIRE_FALSE(against.found);
}

TEST_CASE("a signal cycles and reports how long it has been green", "[traffic][traffic-lanes]")
{
    const auto light = TrafficSignal{.lane = 0,
                                     .distanceMetres = 100.0,
                                     .greenSeconds = 20.0,
                                     .amberSeconds = 3.0,
                                     .redSeconds = 17.0,
                                     .offsetSeconds = 0.0};

    REQUIRE(signalAspect(light, 0.0) == SignalAspect::Green);
    REQUIRE(signalAspect(light, 19.9) == SignalAspect::Green);
    REQUIRE(signalAspect(light, 20.1) == SignalAspect::Amber);
    REQUIRE(signalAspect(light, 23.1) == SignalAspect::Red);
    // And round again, which is what a driver waiting at one is doing.
    REQUIRE(signalAspect(light, 40.5) == SignalAspect::Green);

    REQUIRE_THAT(signalGreenFor(light, 0.5), WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(signalGreenFor(light, 25.0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(signalGreenFor(light, 41.0), WithinAbs(1.0, 1e-9));
}

TEST_CASE("the three archetypes differ in the ways they are described", "[traffic][traffic-archetypes]")
{
    const auto cautious = behaviourFor(DriverArchetype::Cautious);
    const auto regular = behaviourFor(DriverArchetype::Regular);
    const auto aggressive = behaviourFor(DriverArchetype::Aggressive);

    // "about ten km/h slower than the limit" and "up to ten km/h above it", which is 2.8 m/s either
    // way. Asserted as the stated figure rather than as an ordering, because the figure is the
    // requirement.
    REQUIRE_THAT(cautious.speedOffsetMetresPerSecond, WithinAbs(-2.8, 1e-12));
    REQUIRE_THAT(regular.speedOffsetMetresPerSecond, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(aggressive.speedOffsetMetresPerSecond, WithinAbs(2.8, 1e-12));

    // "leaves lots of room" against "tailgates".
    REQUIRE(cautious.desiredHeadwaySeconds > regular.desiredHeadwaySeconds);
    REQUIRE(regular.desiredHeadwaySeconds > aggressive.desiredHeadwaySeconds);
    REQUIRE(cautious.minimumGapMetres > aggressive.minimumGapMetres);

    // "slow to take off from lights" against "fast to take off".
    REQUIRE(cautious.greenDelaySeconds > regular.greenDelaySeconds);
    REQUIRE(regular.greenDelaySeconds > aggressive.greenDelaySeconds);
    REQUIRE(aggressive.comfortableAccelerationMetresPerSecondSquared >
            cautious.comfortableAccelerationMetresPerSecondSquared);

    // "doesn't change lanes unless they have to" against "changes lanes, drives around people".
    REQUIRE_FALSE(cautious.changesLanesToGetOn);
    REQUIRE(regular.changesLanesToGetOn);
    REQUIRE(aggressive.changesLanesToGetOn);
    REQUIRE(aggressive.politeness < cautious.politeness);
    REQUIRE(aggressive.overtakeUrgeMetresPerSecondSquared > cautious.overtakeUrgeMetresPerSecondSquared);
}

TEST_CASE("the default mix is mostly ordinary drivers", "[traffic][traffic-archetypes]")
{
    const auto profiles = raceengine::defaultDriverProfiles();

    REQUIRE(profiles.size() == raceengine::namedArchetypeCount);

    auto total = 0.0;
    for (const auto& profile : profiles)
    {
        REQUIRE(profile.weight > 0.0);
        total += profile.weight;
    }

    REQUIRE_THAT(total, WithinRel(1.0, 1e-9));
    REQUIRE(profiles[1].weight > profiles[0].weight);
    REQUIRE(profiles[1].weight > profiles[2].weight);
}
