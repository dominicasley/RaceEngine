module;

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.traffic:Lanes;

namespace raceengine
{

// The road network traffic drives on: a set of directed polylines, the links between them, and the
// signals standing on them.
//
// **This partition knows nothing about where the lanes came from.** What a caller hands in is
// `LaneSource` — points, a speed limit and two switches — and where those came from is the caller's
// business. Grand City Parkway's arrive out of Assetto Corsa's CSP traffic export by way of
// `osr.game:TrafficNetwork`; a generated test network arrives out of a fixture. Keeping the import
// on the far side of a plain struct is what lets every derivation below be unit tested with no asset
// on disk, which is the whole reason this lives in the engine rather than in the sandbox.
//
// Metres and seconds throughout, in the same right-handed axes glTF uses: **+x left, +y up, +z
// forward**. That handedness is load-bearing exactly once, in `laneSide` below, and it is written
// down there rather than assumed.

// One lane as the caller states it.
export struct LaneSource
{
    // The caller's own id, carried through so a report can name the lane the export named. It is not
    // an index and is not required to be contiguous — Grand City Parkway numbers twelve lanes 1..13
    // with 11 missing.
    int id = 0;
    std::string name{};

    // What traffic on this lane is *allowed* to do, which is not what any one driver will do: an
    // archetype decides that against these.
    double speedLimitMetresPerSecond = 13.9;
    bool allowLaneChanges = true;
    bool allowUTurns = false;

    // The control points. Two or more, and treated as a chord polyline throughout: CSP's export does
    // not say how it interpolates between them, so inventing a curve here would be inventing road.
    std::vector<glm::dvec3> points;
};

// What a light is showing.
export enum class SignalAspect : std::uint8_t
{
    Green,
    Amber,
    Red
};

// A traffic light, as a stop line on one lane and a fixed cycle.
//
// **Grand City Parkway has none, and that is a property of the export rather than a gap here**: its
// `counts.traffic_lights` is zero and its `intersections` array is empty, because the map is a set of
// one-way parallel carriageways with no crossings in it. The type exists anyway, and the driver
// archetypes read it, because "slow to take off from the lights" is a stated behaviour and a
// behaviour with nothing to exercise it rots. A track that carries lights gets them for free.
export struct TrafficSignal
{
    // Index into `LaneNetwork::lanes`, and where along that lane the stop line is.
    std::size_t lane = 0;
    double distanceMetres = 0.0;

    // The cycle, seconds. `offsetSeconds` is where in the cycle this light stands at time zero,
    // which is how a corridor of lights is made to run as a green wave rather than in lockstep.
    double greenSeconds = 20.0;
    double amberSeconds = 3.0;
    double redSeconds = 20.0;
    double offsetSeconds = 0.0;
};

// Where a lane carries on when it runs out.
export struct LaneLink
{
    std::size_t lane = 0;
    // Where on the target lane the continuation lands, metres from that lane's own start.
    double distanceMetres = 0.0;
    // The dot product of the two headings at the join. One is the two lanes running the same way.
    double headingAgreement = 1.0;
};

// A parallel lane a car may change into, over the stretch where the two run alongside each other.
//
// **A run rather than a whole-lane pairing**, because two lanes on this map are adjacent for part of
// their length and nowhere near each other for the rest: one entry per contiguous stretch, and a
// lane may carry several naming the same neighbour.
export struct LaneNeighbour
{
    std::size_t lane = 0;
    // Which side the neighbour is on: +1 to the driver's left, -1 to the right.
    double side = 1.0;
    // The mean lateral gap over the run, metres. Read by nothing that decides — it is what a lane
    // change *displaces* the car by while it is in progress.
    double offsetMetres = 3.7;

    // The stretch of **this** lane over which the pairing holds, metres from this lane's start.
    double fromMetres = 0.0;
    double toMetres = 0.0;
    // Add this to a distance on this lane to get the same place on the neighbour — the run's mean,
    // kept for a report and as the fallback. It is **not** constant over a run: the outer lane of a
    // bend is longer than the inner, and over a full ring the two differ by a per cent, which is
    // twelve metres at the ends of a run that used one number. Read `neighbourShiftAt` instead.
    double distanceShiftMetres = 0.0;
    // The shift at each sample of the run, with the distance on this lane it was taken at, so the
    // same place on the neighbour is known to the sample everywhere along the run. Unwrapped across
    // a loop neighbour's origin, so a value may exceed the neighbour's length; `wrapDistance` brings
    // it back.
    std::vector<double> shiftAtMetres{};
    std::vector<double> shiftMetres{};
};

// The shift onto the neighbour at a distance on this lane, interpolated between the run's samples
// and held at the ends; the mean where a run carries no table.
export [[nodiscard]] double neighbourShiftAt(const LaneNeighbour& neighbour, const double distanceMetres);

export struct Lane
{
    int id = 0;
    std::string name{};
    double speedLimitMetresPerSecond = 13.9;
    bool allowLaneChanges = true;
    bool allowUTurns = false;

    std::vector<glm::dvec3> points{};
    // Chord distance from the first point to each, one per point, first always zero.
    std::vector<double> arcLengthMetres{};

    // Whether the lane's end joins its own start closely enough to drive round. Derived, not stated:
    // Grand City Parkway flags `loop` false on every lane and seven of them are geometric loops.
    bool loop = false;

    std::vector<LaneLink> successors{};
    std::vector<LaneNeighbour> neighbours{};
};

// One place on a lane: where it is, which way it runs there, and how far along that is.
export struct LanePlace
{
    glm::dvec3 positionMetres{0.0};
    glm::dvec3 direction{0.0, 0.0, 1.0};
    // The heading a car following the lane has here. `direction` is the chord's own and steps at
    // every point of the polyline; this one is blended between the tangents at the chord's two ends
    // (each the mean of the two chords meeting there, carried across the closure on a loop), so it
    // turns continuously along the lane. The derivations read `direction`; a pose reads this.
    glm::dvec3 tangent{0.0, 0.0, 1.0};
    double distanceMetres = 0.0;
};

// One sample of the network, held in a grid so a point can be put back onto a lane without walking
// 35 km of polyline.
export struct LaneSample
{
    std::size_t lane = 0;
    double distanceMetres = 0.0;
    glm::dvec3 positionMetres{0.0};
    glm::dvec3 direction{0.0, 0.0, 1.0};
};

// The uniform grid over those samples, in compressed-row form: `starts` has one entry per cell plus
// a terminator, and `entries` holds indices into `samples` cell by cell.
//
// A grid rather than a tree because the thing being indexed is a road network laid flat: the samples
// are spread evenly over a square kilometre and never clustered, which is the one case where a
// uniform grid beats everything more clever.
export struct LaneGrid
{
    double cellMetres = 20.0;
    glm::dvec3 originMetres{0.0};
    int columns = 0;
    int rows = 0;
    std::vector<std::uint32_t> starts;
    std::vector<std::uint32_t> entries;
};

export struct LaneNetworkOptions
{
    // How close a lane's end has to be to its own start before it is driven round rather than run
    // out of, and how well the two ends have to agree about which way the road goes.
    //
    // **All three numbers are set from the measured closures on Grand City Parkway**, whose twelve
    // lanes span the whole range of what an export does here. Seven of them end within 16 m of their
    // own start and only three of those are loops a car could drive:
    //
    //     lane  gap    entry·exit   exit·chord     what it is
    //       1   4.0 m    +0.999      +1.000        a loop
    //       6  15.3 m    +0.999      +0.981        a loop
    //      12  11.6 m    +0.974      +0.985        a loop
    //      13  12.5 m    +0.238      +0.975        ends beside its start, pointing elsewhere
    //       2  14.3 m    -0.000      +0.998        ends beside its start at right angles
    //      10   5.9 m    -0.001      +0.975        the same
    //       9   2.0 m    -0.000      -1.000        overshoots its start and points away from it
    //
    // So distance alone is not the test and neither is either agreement alone. `exit·chord` asks
    // whether the end is pointing *at* its own start — lane 9 fails it outright — and `entry·exit`
    // asks whether a car arriving would then be going the way the lane leaves, which is what the
    // four right-angle cases fail. A lane that fails either falls through to the junction derivation
    // below and merges into whatever it actually runs into, which is the honest answer for a
    // carriageway that simply ends.
    double loopClosureMetres = 16.0;
    double loopChordAgreement = 0.80;

    // How close a lane's end has to be to another lane before it is treated as running into it. The
    // same six metres and for the same reason; the export's own junction derivation used one metre
    // and found a single link, which is too tight to carry traffic off a dead end.
    double junctionRadiusMetres = 6.0;

    // How much two headings have to agree before a join or a pairing is believed. A merge may arrive
    // at an angle, so the junction test is the looser of the two.
    double junctionHeadingAgreement = 0.70;
    double neighbourHeadingAgreement = 0.85;

    // The lateral band that counts as "the next lane over". Below the minimum the two lines are the
    // same road drawn twice; above the maximum they are two roads.
    double neighbourMinimumMetres = 2.2;
    double neighbourMaximumMetres = 5.5;

    // How far apart the samples are, metres. It sets the resolution of every derivation below and of
    // the grid: four metres is about a car length, which is the scale at which "is there a lane
    // beside me" changes.
    double sampleSpacingMetres = 4.0;

    // A run of adjacency shorter than this is not a lane change opportunity, it is two lanes brushing
    // past each other at a junction. Thirty metres is about two seconds at city speed.
    double minimumNeighbourRunMetres = 30.0;

    double gridCellMetres = 20.0;
};

export struct LaneNetwork
{
    std::vector<Lane> lanes;
    std::vector<TrafficSignal> signals;

    std::vector<LaneSample> samples;
    LaneGrid grid;

    // The sum of every lane's chord length, metres. What a density is quoted against.
    double totalLengthMetres = 0.0;
};

// Build one. Fallible rather than forgiving: a lane with one point has no heading and a car placed
// on it has nowhere to face, and a network that quietly dropped it would be a city with a missing
// street that nobody notices for a month.
export [[nodiscard]] std::expected<LaneNetwork, std::string> buildLaneNetwork(std::vector<LaneSource> sources,
                                                                              const LaneNetworkOptions& options = {});

// The lane's own length, as its points measure it — and on a loop, the closing chord from its last
// point back to its first as well, because that chord is road a car drives: Grand City Parkway's
// loops close with 4 to 15 m between their two ends, and a car that wrapped from one to the other
// in a tick hopped a car length or three every time round.
export [[nodiscard]] double laneLength(const Lane& lane);
// A distance brought onto the lane: modulo its length on a loop (so a distance past the end, or a
// negative one, is the place it means), the distance itself anywhere else. Every distance that is
// *computed* — the same place on a neighbour, a look-ahead — goes through this before it is read.
export [[nodiscard]] double wrapDistance(const Lane& lane, const double distanceMetres);

// Where the lane is at a distance from its start, interpolated along the chord and clamped to both
// ends. Nothing where the lane carries fewer than two points.
export [[nodiscard]] std::optional<LanePlace> laneAt(const Lane& lane, const double distanceMetres);

// The unit vector pointing to the **driver's left** at a place on a lane.
//
// This is the one thing in this partition that depends on the axes being right-handed with +y up:
// `cross(up, forward)` is +x, and +x is the car's left in this engine's frame (`outboardSign`). Get
// it backwards and every lane change goes the wrong way, which is the kind of fault that looks like
// a data problem for a day.
export [[nodiscard]] glm::dvec3 laneSide(const glm::dvec3& direction);

// Advance along the network, following a loop round or a successor on. Where a lane simply runs out,
// `ranOut` is set and the place returned is its far end — the caller decides whether that is a
// despawn or a turn.
export struct LaneStep
{
    std::size_t lane = 0;
    double distanceMetres = 0.0;
    bool ranOut = false;
    // True when this step crossed onto a different lane, so a caller keeping per-lane ordering knows
    // to rebuild it.
    bool changedLane = false;
};

export [[nodiscard]] LaneStep advanceAlongLane(const LaneNetwork& network, const std::size_t lane,
                                               const double distanceMetres, const double byMetres);

// Put a world point back onto the network: which lane it is on, how far along, and how far to the
// side. `heading` is used to reject a lane running the other way and may be zero to accept any.
export struct LaneProjection
{
    bool found = false;
    std::size_t lane = 0;
    double distanceMetres = 0.0;
    // Signed, positive when the point is to the **left** of the lane.
    double lateralMetres = 0.0;
};

export [[nodiscard]] LaneProjection projectOntoNetwork(const LaneNetwork& network, const glm::dvec3& pointMetres,
                                                       const glm::dvec3& heading, const double maximumLateralMetres);

// What a signal is showing at a time. Pure in (signal, seconds): no clock is read, so a captured run
// and a live one see the same lights on the same tick.
export [[nodiscard]] SignalAspect signalAspect(const TrafficSignal& signal, const double seconds);

// How long that signal has been showing green. Zero on the tick it changed, which is what a driver's
// reaction delay is measured against.
export [[nodiscard]] double signalGreenFor(const TrafficSignal& signal, const double seconds);

} // namespace raceengine
