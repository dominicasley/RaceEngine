// Lane network bodies. Declarations are in Api/Lanes.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export` — which produces
// an object and no BMI, so nothing imports it and nothing rebuilds when it changes. The house rule
// and the measurements behind it: docs/build-times.md.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

module raceengine.traffic;

namespace raceengine
{

namespace
{

// Straight up, in the axes this whole partition works in.
constexpr auto worldUp = glm::dvec3(0.0, 1.0, 0.0);

[[nodiscard]] glm::dvec3 segmentDirection(const glm::dvec3& from, const glm::dvec3& to)
{
    const auto step = to - from;
    const auto length = glm::length(step);

    return length > 1e-9 ? step / length : glm::dvec3(0.0, 0.0, 1.0);
}

// Which cell of the grid a point falls in, clamped to the grid rather than refused: a point outside
// the network's own bounds is a legitimate question — a car driven off the road asks it — and the
// nearest cell is the right place to start looking.
[[nodiscard]] int cellIndexFor(const LaneGrid& grid, const glm::dvec3& point)
{
    const auto column =
        std::clamp(static_cast<int>((point.x - grid.originMetres.x) / grid.cellMetres), 0, grid.columns - 1);
    const auto row = std::clamp(static_cast<int>((point.z - grid.originMetres.z) / grid.cellMetres), 0, grid.rows - 1);

    return row * grid.columns + column;
}

// Every sample within `radius` cells of a point, handed one at a time to `visit`.
template <typename Visitor>
void visitNearbySamples(const LaneNetwork& network, const glm::dvec3& point, const int radius, Visitor&& visit)
{
    const auto& grid = network.grid;
    if (grid.columns <= 0 || grid.rows <= 0)
    {
        return;
    }

    const auto centre = cellIndexFor(grid, point);
    const auto centreColumn = centre % grid.columns;
    const auto centreRow = centre / grid.columns;

    for (auto row = centreRow - radius; row <= centreRow + radius; row++)
    {
        if (row < 0 || row >= grid.rows)
        {
            continue;
        }

        for (auto column = centreColumn - radius; column <= centreColumn + radius; column++)
        {
            if (column < 0 || column >= grid.columns)
            {
                continue;
            }

            const auto cell = static_cast<std::size_t>(row * grid.columns + column);

            for (auto entry = grid.starts[cell]; entry < grid.starts[cell + 1]; entry++)
            {
                visit(network.samples[grid.entries[entry]]);
            }
        }
    }
}

// One lane's worth of adjacency bookkeeping, carried while its samples are walked in order.
struct NeighbourRun
{
    bool active = false;
    double fromMetres = 0.0;
    double toMetres = 0.0;
    double lastMetres = 0.0;
    double shiftTotal = 0.0;
    double offsetTotal = 0.0;
    double sideTotal = 0.0;
    double samples = 0.0;
    std::vector<double> shiftAtMetres{};
    std::vector<double> shiftMetres{};
};

void closeRun(Lane& lane, const std::size_t other, NeighbourRun& run, const double minimumRunMetres)
{
    if (!run.active)
    {
        return;
    }

    run.active = false;

    if (run.samples <= 0.0 || run.toMetres - run.fromMetres < minimumRunMetres)
    {
        return;
    }

    lane.neighbours.push_back(LaneNeighbour{.lane = other,
                                            .side = run.sideTotal >= 0.0 ? 1.0 : -1.0,
                                            .offsetMetres = run.offsetTotal / run.samples,
                                            .fromMetres = run.fromMetres,
                                            .toMetres = run.toMetres,
                                            .distanceShiftMetres = run.shiftTotal / run.samples,
                                            .shiftAtMetres = std::move(run.shiftAtMetres),
                                            .shiftMetres = std::move(run.shiftMetres)});
}

} // namespace

double laneLength(const Lane& lane)
{
    if (lane.arcLengthMetres.empty())
    {
        return 0.0;
    }

    const auto arc = lane.arcLengthMetres.back();

    return lane.loop ? arc + glm::distance(lane.points.back(), lane.points.front()) : arc;
}

double neighbourShiftAt(const LaneNeighbour& neighbour, const double distanceMetres)
{
    const auto& at = neighbour.shiftAtMetres;
    const auto& shift = neighbour.shiftMetres;

    if (at.empty() || at.size() != shift.size())
    {
        return neighbour.distanceShiftMetres;
    }

    if (distanceMetres <= at.front())
    {
        return shift.front();
    }

    if (distanceMetres >= at.back())
    {
        return shift.back();
    }

    const auto after = std::ranges::upper_bound(at, distanceMetres);
    const auto index = static_cast<std::size_t>(after - at.begin());
    const auto span = at[index] - at[index - 1];
    const auto along = span > 0.0 ? (distanceMetres - at[index - 1]) / span : 0.0;

    return shift[index - 1] + (shift[index] - shift[index - 1]) * along;
}

double wrapDistance(const Lane& lane, const double distanceMetres)
{
    const auto total = laneLength(lane);

    if (!lane.loop || total <= 0.0)
    {
        return distanceMetres;
    }

    return distanceMetres - std::floor(distanceMetres / total) * total;
}

glm::dvec3 laneSide(const glm::dvec3& direction)
{
    // `cross(up, forward)` is +x, and +x is this engine's left. The whole handedness of the lane
    // change lives in this one line; see the note on `laneSide` in Api/Lanes.cppm.
    const auto side = glm::cross(worldUp, direction);
    const auto length = glm::length(side);

    return length > 1e-9 ? side / length : glm::dvec3(1.0, 0.0, 0.0);
}

std::optional<LanePlace> laneAt(const Lane& lane, const double distanceMetres)
{
    if (lane.points.size() < 2 || lane.arcLengthMetres.size() != lane.points.size())
    {
        return std::nullopt;
    }

    const auto arcEnd = lane.arcLengthMetres.back();
    const auto total = laneLength(lane);
    const auto wanted = std::clamp(distanceMetres, 0.0, total);

    // The closing chord on a loop, from the last point back to the first, is the chord this lane's
    // ends do not draw and a car drives anyway.
    const auto closing = lane.loop ? segmentDirection(lane.points.back(), lane.points.front()) : glm::dvec3(0.0);

    auto from = glm::dvec3(0.0);
    auto to = glm::dvec3(0.0);
    auto along = 0.0;
    auto own = glm::dvec3(0.0);
    auto chordBefore = glm::dvec3(0.0);
    auto chordAfter = glm::dvec3(0.0);

    if (lane.loop && wanted > arcEnd && total > arcEnd)
    {
        from = lane.points.back();
        to = lane.points.front();
        along = (wanted - arcEnd) / (total - arcEnd);
        own = closing;
        chordBefore = segmentDirection(lane.points[lane.points.size() - 2], from);
        chordAfter = segmentDirection(to, lane.points[1]);
    }
    else
    {
        // The last point at or before `wanted`, so an exact hit on a point lands on the segment
        // leaving it rather than the one arriving.
        const auto after = std::ranges::upper_bound(lane.arcLengthMetres, wanted);
        auto segment = static_cast<std::size_t>(after - lane.arcLengthMetres.begin());
        segment = segment == 0 ? 0 : segment - 1;
        segment = std::min(segment, lane.points.size() - 2);

        from = lane.points[segment];
        to = lane.points[segment + 1];
        const auto span = lane.arcLengthMetres[segment + 1] - lane.arcLengthMetres[segment];
        along = span > 0.0 ? (wanted - lane.arcLengthMetres[segment]) / span : 0.0;
        own = segmentDirection(from, to);
        chordBefore = segment > 0 ? segmentDirection(lane.points[segment - 1], from) : closing;
        chordAfter = segment + 2 < lane.points.size() ? segmentDirection(to, lane.points[segment + 2]) : closing;
    }

    // The tangents at the chord's two ends: the mean of the two chords meeting at each, the chord's
    // own at an open end (a zero `closing` blends to the chord itself). A hairpin whose two chords
    // cancel keeps the chord's own rather than a zero.
    const auto mean = [&](const glm::dvec3& first, const glm::dvec3& second)
    {
        const auto sum = first + second;
        const auto length = glm::length(sum);

        return length > 1e-6 ? sum / length : own;
    };

    const auto blended = glm::mix(mean(chordBefore, own), mean(own, chordAfter), along);
    const auto blendedLength = glm::length(blended);

    return LanePlace{.positionMetres = from + (to - from) * along,
                     .direction = own,
                     .tangent = blendedLength > 1e-6 ? blended / blendedLength : own,
                     .distanceMetres = wanted};
}

LaneStep advanceAlongLane(const LaneNetwork& network, const std::size_t lane, const double distanceMetres,
                          const double byMetres)
{
    auto step = LaneStep{.lane = lane, .distanceMetres = distanceMetres + byMetres};

    if (lane >= network.lanes.size())
    {
        step.ranOut = true;

        return step;
    }

    // A loop or a successor may itself be short, so this walks rather than branching once. Bounded
    // because every hop consumes the remainder of a lane whose length is positive; the cap is there
    // for a network of degenerate lanes rather than for anything a road can do.
    for (auto hop = 0; hop < 8; hop++)
    {
        const auto& current = network.lanes[step.lane];
        const auto total = laneLength(current);

        if (step.distanceMetres <= total)
        {
            return step;
        }

        const auto overshoot = step.distanceMetres - total;

        if (current.loop)
        {
            step.distanceMetres = overshoot;
            step.changedLane = true;

            continue;
        }

        if (current.successors.empty())
        {
            step.distanceMetres = total;
            step.ranOut = true;

            return step;
        }

        // The first successor, which is the closest join the derivation found. A network with real
        // junctions in it wants a choice here and this is where it goes; nothing on the one map this
        // engine carries offers one.
        const auto& link = current.successors.front();

        step.lane = link.lane;
        step.distanceMetres = link.distanceMetres + overshoot;
        step.changedLane = true;
    }

    step.ranOut = true;

    return step;
}

LaneProjection projectOntoNetwork(const LaneNetwork& network, const glm::dvec3& pointMetres, const glm::dvec3& heading,
                                  const double maximumLateralMetres)
{
    auto answer = LaneProjection{};

    if (network.samples.empty() || network.grid.cellMetres <= 0.0)
    {
        return answer;
    }

    const auto headingLength = glm::length(heading);
    const auto wanted = headingLength > 1e-9 ? heading / headingLength : glm::dvec3(0.0);

    const auto radius = std::max(1, static_cast<int>(std::ceil(maximumLateralMetres / network.grid.cellMetres)));

    auto best = maximumLateralMetres;

    visitNearbySamples(network, pointMetres, radius,
                       [&](const LaneSample& sample)
                       {
                           if (headingLength > 1e-9 && glm::dot(wanted, sample.direction) < 0.0)
                           {
                               return;
                           }

                           const auto offset = pointMetres - sample.positionMetres;
                           // Along the lane first, so what is measured across it is what is left.
                           const auto along = glm::dot(offset, sample.direction);
                           const auto across = offset - along * sample.direction;
                           const auto lateral = glm::length(across);

                           if (lateral >= best)
                           {
                               return;
                           }

                           best = lateral;
                           answer.found = true;
                           answer.lane = sample.lane;
                           answer.distanceMetres = sample.distanceMetres + along;
                           answer.lateralMetres = glm::dot(across, laneSide(sample.direction));
                       });

    return answer;
}

SignalAspect signalAspect(const TrafficSignal& signal, const double seconds)
{
    const auto cycle = signal.greenSeconds + signal.amberSeconds + signal.redSeconds;
    if (cycle <= 0.0)
    {
        return SignalAspect::Green;
    }

    auto phase = std::fmod(seconds + signal.offsetSeconds, cycle);
    if (phase < 0.0)
    {
        phase += cycle;
    }

    if (phase < signal.greenSeconds)
    {
        return SignalAspect::Green;
    }

    return phase < signal.greenSeconds + signal.amberSeconds ? SignalAspect::Amber : SignalAspect::Red;
}

double signalGreenFor(const TrafficSignal& signal, const double seconds)
{
    const auto cycle = signal.greenSeconds + signal.amberSeconds + signal.redSeconds;
    if (cycle <= 0.0)
    {
        return seconds;
    }

    auto phase = std::fmod(seconds + signal.offsetSeconds, cycle);
    if (phase < 0.0)
    {
        phase += cycle;
    }

    return phase < signal.greenSeconds ? phase : 0.0;
}

std::expected<LaneNetwork, std::string> buildLaneNetwork(std::vector<LaneSource> sources,
                                                         const LaneNetworkOptions& options)
{
    if (sources.empty())
    {
        return std::unexpected("a traffic lane network carries no lanes");
    }

    if (options.sampleSpacingMetres <= 0.0 || options.gridCellMetres <= 0.0)
    {
        return std::unexpected("a traffic lane network needs a positive sample spacing and cell size");
    }

    auto network = LaneNetwork{};
    network.lanes.reserve(sources.size());

    for (auto& source : sources)
    {
        // A single point is not a lane: nothing can be interpolated along it and a car placed on it
        // has no heading. Refused rather than dropped, for the reason the loader that feeds this one
        // refuses it too — a street going missing between the export and the game is a fault that
        // reads as a hole in the map months later.
        if (source.points.size() < 2)
        {
            return std::unexpected("traffic lane " + std::to_string(source.id) + " carries fewer than two points");
        }

        auto lane = Lane{.id = source.id,
                         .name = std::move(source.name),
                         .speedLimitMetresPerSecond = source.speedLimitMetresPerSecond,
                         .allowLaneChanges = source.allowLaneChanges,
                         .allowUTurns = source.allowUTurns,
                         .points = std::move(source.points)};

        lane.arcLengthMetres.reserve(lane.points.size());
        lane.arcLengthMetres.push_back(0.0);

        for (auto index = std::size_t{1}; index < lane.points.size(); index++)
        {
            lane.arcLengthMetres.push_back(lane.arcLengthMetres.back() +
                                           glm::distance(lane.points[index - 1], lane.points[index]));
        }

        if (laneLength(lane) <= 0.0)
        {
            return std::unexpected("traffic lane " + std::to_string(source.id) + " has no length");
        }

        network.totalLengthMetres += laneLength(lane);
        network.lanes.push_back(std::move(lane));
    }

    // --- The samples, and the grid over them -------------------------------------------------
    //
    // Built before any derivation because all three derivations below are lookups against it. Every
    // lane's samples are contiguous and in order, which is what lets the neighbour walk below carry
    // a run forward without an index of its own.
    auto minimum = glm::dvec3(std::numeric_limits<double>::max());
    auto maximum = glm::dvec3(std::numeric_limits<double>::lowest());

    for (auto laneIndex = std::size_t{0}; laneIndex < network.lanes.size(); laneIndex++)
    {
        const auto& lane = network.lanes[laneIndex];
        const auto total = laneLength(lane);

        // Stepped off the index rather than accumulated, so the thousandth sample of a 7.8 km lane
        // stands where the arithmetic says and not where a thousand additions drifted to.
        for (auto step = std::size_t{0};; step++)
        {
            const auto distance = static_cast<double>(step) * options.sampleSpacingMetres;
            if (distance > total)
            {
                break;
            }

            const auto place = laneAt(lane, distance);
            if (!place)
            {
                break;
            }

            network.samples.push_back(LaneSample{.lane = laneIndex,
                                                 .distanceMetres = place->distanceMetres,
                                                 .positionMetres = place->positionMetres,
                                                 .direction = place->direction});

            minimum = glm::min(minimum, place->positionMetres);
            maximum = glm::max(maximum, place->positionMetres);
        }
    }

    if (network.samples.empty())
    {
        return std::unexpected("a traffic lane network produced no samples");
    }

    network.grid.cellMetres = options.gridCellMetres;
    network.grid.originMetres = minimum;
    network.grid.columns = std::max(1, static_cast<int>((maximum.x - minimum.x) / options.gridCellMetres) + 1);
    network.grid.rows = std::max(1, static_cast<int>((maximum.z - minimum.z) / options.gridCellMetres) + 1);

    const auto cells = static_cast<std::size_t>(network.grid.columns) * static_cast<std::size_t>(network.grid.rows);

    // Counting pass, then a prefix sum, then a placing pass: the compressed-row form is built without
    // a vector per cell, which on a map this size would be three thousand allocations.
    auto counts = std::vector<std::uint32_t>(cells + 1, 0);
    for (const auto& sample : network.samples)
    {
        counts[static_cast<std::size_t>(cellIndexFor(network.grid, sample.positionMetres))]++;
    }

    network.grid.starts.assign(cells + 1, 0);
    for (auto cell = std::size_t{0}; cell < cells; cell++)
    {
        network.grid.starts[cell + 1] = network.grid.starts[cell] + counts[cell];
    }

    auto cursor = network.grid.starts;
    network.grid.entries.assign(network.samples.size(), 0);

    for (auto index = std::size_t{0}; index < network.samples.size(); index++)
    {
        const auto cell = static_cast<std::size_t>(cellIndexFor(network.grid, network.samples[index].positionMetres));

        network.grid.entries[cursor[cell]] = static_cast<std::uint32_t>(index);
        cursor[cell]++;
    }

    // --- Loop closure ---------------------------------------------------------------------------
    //
    // Three tests, and the table in `LaneNetworkOptions` is the measurement that says why none of
    // them can be dropped: near enough, pointing at its own start, and leaving the way it arrived.
    for (auto& lane : network.lanes)
    {
        const auto first = lane.points.front();
        const auto last = lane.points.back();
        const auto gap = glm::distance(first, last);

        if (gap > options.loopClosureMetres)
        {
            continue;
        }

        const auto entry = segmentDirection(lane.points[0], lane.points[1]);
        const auto exit = segmentDirection(lane.points[lane.points.size() - 2], last);

        // A lane whose two ends are the same point closes on the heading test alone; there is no
        // chord to take a direction from.
        const auto chord = gap > 1e-6 ? segmentDirection(last, first) : exit;

        lane.loop = glm::dot(exit, chord) >= options.loopChordAgreement &&
                    glm::dot(entry, exit) >= options.junctionHeadingAgreement;
    }

    // --- Successors -----------------------------------------------------------------------------
    //
    // A lane that does not close on itself may still run into another one, which is what every
    // carriageway on Grand City Parkway does. Derived from geometry because CSP stores no lane graph
    // at all: the export's own `junctions` block is the same test at a one-metre tolerance, which
    // finds a single link on a map with eleven joins in it.
    const auto junctionRadius =
        std::max(1, static_cast<int>(std::ceil(options.junctionRadiusMetres / network.grid.cellMetres)));

    for (auto laneIndex = std::size_t{0}; laneIndex < network.lanes.size(); laneIndex++)
    {
        auto& lane = network.lanes[laneIndex];
        if (lane.loop)
        {
            continue;
        }

        const auto last = lane.points.back();
        const auto exit = segmentDirection(lane.points[lane.points.size() - 2], last);

        auto bestDistance = options.junctionRadiusMetres;
        auto found = false;
        auto link = LaneLink{};

        visitNearbySamples(network, last, junctionRadius,
                           [&](const LaneSample& sample)
                           {
                               if (sample.lane == laneIndex)
                               {
                                   return;
                               }

                               const auto agreement = glm::dot(exit, sample.direction);
                               if (agreement < options.junctionHeadingAgreement)
                               {
                                   return;
                               }

                               // The join has to have road left on the far side of it, or traffic is
                               // handed from one dead end straight onto another.
                               const auto& target = network.lanes[sample.lane];
                               if (laneLength(target) - sample.distanceMetres < options.junctionRadiusMetres)
                               {
                                   return;
                               }

                               const auto gap = glm::distance(last, sample.positionMetres);
                               if (gap >= bestDistance)
                               {
                                   return;
                               }

                               bestDistance = gap;
                               found = true;
                               // **Projected onto the sample rather than taken from it.** A sample
                               // stands every four metres, so the sample's own distance is the join
                               // rounded down to the nearest sample — which puts every car merging
                               // onto a lane up to four metres behind where it actually arrived.
                               link =
                                   LaneLink{.lane = sample.lane,
                                            .distanceMetres = sample.distanceMetres +
                                                              glm::dot(last - sample.positionMetres, sample.direction),
                                            .headingAgreement = agreement};
                           });

        if (found)
        {
            lane.successors.push_back(link);
        }
    }

    // --- Neighbours -----------------------------------------------------------------------------
    //
    // The stretches over which two lanes run alongside each other, which is what a lane change needs
    // and what nothing in the export states. Walked in sample order per lane so a pairing that comes
    // and goes produces separate runs rather than one run spanning the gap between them.
    const auto neighbourRadius =
        std::max(1, static_cast<int>(std::ceil(options.neighbourMaximumMetres / network.grid.cellMetres)));

    auto runs = std::vector<NeighbourRun>(network.lanes.size());
    auto matchedShift = std::vector<double>(network.lanes.size(), 0.0);
    auto matchedOffset = std::vector<double>(network.lanes.size(), 0.0);
    auto matchedSide = std::vector<double>(network.lanes.size(), 0.0);
    auto matched = std::vector<bool>(network.lanes.size(), false);

    for (auto laneIndex = std::size_t{0}; laneIndex < network.lanes.size(); laneIndex++)
    {
        std::ranges::fill(runs, NeighbourRun{});

        for (const auto& sample : network.samples)
        {
            if (sample.lane != laneIndex)
            {
                continue;
            }

            std::ranges::fill(matched, false);
            std::ranges::fill(matchedShift, 0.0);
            std::ranges::fill(matchedOffset, std::numeric_limits<double>::max());
            std::ranges::fill(matchedSide, 0.0);

            visitNearbySamples(network, sample.positionMetres, neighbourRadius,
                               [&](const LaneSample& other)
                               {
                                   if (other.lane == laneIndex)
                                   {
                                       return;
                                   }

                                   if (glm::dot(sample.direction, other.direction) < options.neighbourHeadingAgreement)
                                   {
                                       return;
                                   }

                                   const auto offset = other.positionMetres - sample.positionMetres;
                                   const auto along = glm::dot(offset, sample.direction);
                                   const auto across = offset - along * sample.direction;
                                   const auto lateral = glm::length(across);

                                   // **Beside this sample, not merely near it.** The grid's reach is
                                   // a cell each way, and a neighbour that begins up the road was
                                   // matched from forty metres before it began — its first sample
                                   // being the laterally nearest — so the run named a place before
                                   // the lane starts, which `laneAt` clamps to its first point: a
                                   // car changing lanes there was blended toward a point forty
                                   // metres ahead (2026-09-11, docs/traffic-system-brief.md §20).
                                   // Within a sample spacing along is what "the same place" means.
                                   if (std::abs(along) > options.sampleSpacingMetres)
                                   {
                                       return;
                                   }

                                   if (lateral < options.neighbourMinimumMetres ||
                                       lateral > options.neighbourMaximumMetres)
                                   {
                                       return;
                                   }

                                   if (lateral >= matchedOffset[other.lane])
                                   {
                                       return;
                                   }

                                   matched[other.lane] = true;
                                   matchedOffset[other.lane] = lateral;
                                   matchedSide[other.lane] = glm::dot(across, laneSide(sample.direction));
                                   // Where the same place on this lane falls on the neighbour: the
                                   // neighbour's own distance, plus however far ahead of the sample
                                   // the pairing landed.
                                   matchedShift[other.lane] = other.distanceMetres - along - sample.distanceMetres;
                               });

            for (auto other = std::size_t{0}; other < network.lanes.size(); other++)
            {
                auto& run = runs[other];

                if (!matched[other])
                {
                    closeRun(network.lanes[laneIndex], other, run, options.minimumNeighbourRunMetres);

                    continue;
                }

                // A pairing that skipped more than one sample is a second run rather than a
                // continuation: two lanes that part company at a junction and meet again later are
                // not adjacent in between.
                if (run.active && sample.distanceMetres - run.lastMetres > 2.5 * options.sampleSpacingMetres)
                {
                    closeRun(network.lanes[laneIndex], other, run, options.minimumNeighbourRunMetres);
                }

                if (!run.active)
                {
                    run = NeighbourRun{.active = true, .fromMetres = sample.distanceMetres};
                }

                // **Unwrapped against the run so far when the neighbour is a loop.** The shift is the
                // neighbour's distance less this lane's, and the neighbour's distance drops by its
                // whole length where its origin falls inside this run — Grand City Parkway's two
                // long carriageways run beside a loop the whole way round it. Averaged raw, half the
                // samples say one thing and half say that less a loop, and the mean names a place on
                // the far side of the map. The nearest representative modulo the loop keeps the run's
                // shifts continuous; the consumer brings `distance + shift` back onto the loop.
                auto shift = matchedShift[other];

                if (run.active && run.samples > 0.0 && network.lanes[other].loop)
                {
                    const auto period = laneLength(network.lanes[other]);
                    const auto reference = run.shiftTotal / run.samples;

                    shift = period > 0.0 ? reference + std::remainder(shift - reference, period) : shift;
                }

                // **A shift that steps is another pass of the same neighbour, and another run.** A
                // lane can run beside this one twice — Grand City Parkway's lane 8 is a near-ring whose
                // end merges beside its own start, and both stand beside lane 9 for a dozen metres —
                // and the nearest sample flips from one pass to the other between two samples four
                // metres apart, a step of two kilometres in the shift. Interpolated across (2026-09-11,
                // docs/traffic-system-brief.md §20) that sent a changing car's blend point sweeping
                // the far pass at two hundred metres a second. Along one pass the shift drifts by the
                // lanes' length difference, a few metres over a run; more than a sample's worth and
                // then some is a different place, so the run closes here and a new one opens on this
                // sample. A run too short to change lanes in is discarded by the rule below, which is
                // the right answer where the two passes alternate.
                if (run.active && !run.shiftMetres.empty() &&
                    std::abs(shift - run.shiftMetres.back()) > 2.5 * options.sampleSpacingMetres)
                {
                    closeRun(network.lanes[laneIndex], other, run, options.minimumNeighbourRunMetres);
                    run = NeighbourRun{.active = true, .fromMetres = sample.distanceMetres};
                }

                run.toMetres = sample.distanceMetres;
                run.lastMetres = sample.distanceMetres;
                run.shiftTotal += shift;
                run.shiftAtMetres.push_back(sample.distanceMetres);
                run.shiftMetres.push_back(shift);
                run.offsetTotal += matchedOffset[other];
                run.sideTotal += matchedSide[other];
                run.samples += 1.0;
            }
        }

        for (auto other = std::size_t{0}; other < network.lanes.size(); other++)
        {
            closeRun(network.lanes[laneIndex], other, runs[other], options.minimumNeighbourRunMetres);
        }
    }

    return network;
}

} // namespace raceengine
