// What the acceptance test's discontinuity actually is: `./EngineTests "[.kerb-discontinuity]"`.
//
// `the imported car crosses a kerb continuously` was E2's acceptance signal, and two different
// mechanisms were on record for it. `docs/known-red.md` said the W3 chamfer-edge defect — the bed of
// independent springs dropping samples out with the divisor left alone. The test's own comment said
// the spike-rejection lottery — a sample admitted for one tick because the *median* the ceiling is
// measured from moved when the touching set changed. Those are different faults and only one of them
// is fixed by an enveloping model, so which it was had to be measured before anything was built on
// either.
//
// **It was neither.** Over the whole crossing, zero of 3539 ticks reject a single sample, and the
// load dip cannot produce a step because a sample leaving contact leaves carrying zero weight. What
// the worst tick actually contains is **one ray of 31,851 that came back with no hit** while its two
// neighbours in the same row reported 19.5 and 18.8 mm of road. Removing that one ray's worth of
// nonsense took the worst patch-centre jump from 25.85 mm to 2.42 and the worst load jump from 713 N
// to 275, and the test passes on its merits. The dead band the ray fell into is characterised by the
// second case here and repaired in `raceengineJoltCastRays`.
//
// Kept because the question it answers gets asked again every time this test moves: it re-drives the
// exact scenario and prints the anatomy of the worst tick — per-sample depths, the median, the
// rejection ceiling, how many rays came back touching against how many the aggregate kept, and how
// many came back at all. A rejection shows as two of those counts disagreeing; a lost ray shows as a
// depth of -1000.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::aggregateContactPatch;
using raceengine::bodyToWorld;
using raceengine::bringUpJolt;
using raceengine::contactPatchSamples;
using raceengine::ContactPatchSampling;
using raceengine::Corner;
using raceengine::CornerSide;
using raceengine::Feature;
using raceengine::FeatureKind;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::SurfaceHit;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::WheelPose;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed,
            const double startZ)
{
    state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight, startZ);

    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = speed / tyreRadius;
    }
}

// The acceptance test's ground, restated here rather than shared, so this probe cannot silently stop
// measuring the same thing if that test's descriptor is edited.
[[nodiscard]] ProvingGroundDescriptor kerbGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 200.0;
    descriptor.width = 30.0;
    descriptor.cellSize = 0.10;
    descriptor.kerbInnerEdge = 0.60;
    descriptor.features = {Feature{.kind = FeatureKind::Kerb, .from = 40.0, .to = 160.0}};

    return descriptor;
}

// One tick's worth of what the sampler saw, rebuilt from the pose the solution reports. The vehicle
// does not publish per-sample data, so the rays are re-cast against the same world from the same
// pose — which is the pose `stepVehicle` built its own from, assembled the same way.
struct Anatomy
{
    std::vector<double> depths;
    std::vector<glm::dvec3> origins;
    std::uint32_t touching = 0;
    std::uint32_t kept = 0;
    std::uint32_t missed = 0;
    double median = 0.0;
    double ceiling = 0.0;
    bool rejected = false;
};

[[nodiscard]] Anatomy anatomyOf(const PhysicsWorld& world, const VehicleSetup& setup, const VehicleState& state,
                                const raceengine::CornerSolution& solution, const std::size_t index)
{
    const auto& corner = setup.corners[index];
    const auto outboard = outboardSign(corner.hardpoints.side);
    const auto& suspension = solution.suspension;

    const auto pose = WheelPose{
        .centre = bodyToWorld(state.chassis, suspension.wheelCentre),
        .spinAxis = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(outboard, 0.0, 0.0)),
        .forward = state.chassis.orientation * (suspension.uprightOrientation * glm::dvec3(0.0, 0.0, 1.0)),
        .radius = corner.hardpoints.wheelRadius};

    const auto geometry = contactPatchSamples(pose, setup.sampling);

    auto hits = std::vector<SurfaceHit>{};
    world.castRays(geometry.origins, geometry.directions, setup.sampling.searchDistance * 2.0, hits);

    auto anatomy = Anatomy{};
    auto positive = std::vector<double>{};

    for (auto sample = std::size_t{0}; sample < hits.size(); sample++)
    {
        const auto depth = hits[sample].hit ? hits[sample].point.y - geometry.tireSurface[sample].y : -1.0;
        anatomy.depths.push_back(depth);
        anatomy.origins.push_back(geometry.origins[sample]);
        anatomy.missed += hits[sample].hit ? 0u : 1u;

        if (depth > 0.0)
        {
            positive.push_back(depth);
            anatomy.touching++;
        }
    }

    if (!positive.empty())
    {
        auto sorted = positive;
        std::sort(sorted.begin(), sorted.end());
        anatomy.median = sorted[sorted.size() / 2];
        anatomy.ceiling = anatomy.median + setup.sampling.spikeRejection;

        for (const auto depth : positive)
        {
            anatomy.kept += depth <= anatomy.ceiling ? 1u : 0u;
        }
    }

    anatomy.rejected = anatomy.kept != anatomy.touching;

    return anatomy;
}

void printAnatomy(const char* label, const Anatomy& anatomy)
{
    std::printf("  %-10s touching %u, kept %u, median %.2f mm, ceiling %.2f mm%s\n", label, anatomy.touching,
                anatomy.kept, anatomy.median * 1000.0, anatomy.ceiling * 1000.0,
                anatomy.rejected ? "   <-- A SAMPLE WAS REJECTED" : "");
    std::printf("  %-10s depths mm:", "");
    for (const auto depth : anatomy.depths)
    {
        std::printf(depth > 0.0 ? " %7.2f*" : " %7.2f ", depth * 1000.0);
    }
    std::printf("\n");

    for (auto index = std::size_t{0}; index < anatomy.depths.size(); index++)
    {
        if (anatomy.depths[index] < -0.5)
        {
            std::printf("  %-10s sample %zu came back with NO HIT, from origin (%.10f, %.10f, %.10f)\n", "", index,
                        anatomy.origins[index].x, anatomy.origins[index].y, anatomy.origins[index].z);
        }
    }
}

} // namespace

TEST_CASE("what the kerb crossing's worst tick is actually made of", "[.kerb-discontinuity]")
{
    const JoltGuard jolt;

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto descriptor = kerbGround();
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), 8.0, 20.0);

    auto input = VehicleInput{};
    input.steering = 0.01 * outboardSign(CornerSide::Right);

    constexpr auto wheel = static_cast<std::size_t>(Corner::FrontLeft);

    struct Tick
    {
        int step = 0;
        double centreOffset = 0.0;
        double load = 0.0;
        double jump = 0.0;
        double loadJump = 0.0;
        std::uint32_t contacting = 0;
    };

    auto history = std::vector<Tick>{};
    auto anatomies = std::vector<Anatomy>{};
    auto states = std::vector<VehicleState>{};
    auto solutions = std::vector<raceengine::CornerSolution>{};

    auto previousOffset = 0.0;
    auto previousLoad = 0.0;
    auto first = true;

    for (auto step = 0; step < 3600; step++)
    {
        const auto before = state;
        const auto stepped = stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick);
        REQUIRE(stepped.has_value());

        const auto& corner = stepped->corners[wheel];
        if (!corner.patch.inContact)
        {
            first = true;
            continue;
        }

        const auto offset = corner.patch.centre.x - corner.suspension.contactPatch.x;

        if (step > 60 && !first)
        {
            history.push_back(Tick{.step = step,
                                   .centreOffset = offset,
                                   .load = corner.forces.tireVertical,
                                   .jump = std::abs(offset - previousOffset),
                                   .loadJump = std::abs(corner.forces.tireVertical - previousLoad),
                                   .contacting = corner.patch.contactingSamples});
            states.push_back(before);
            solutions.push_back(corner);
        }

        previousOffset = offset;
        previousLoad = corner.forces.tireVertical;
        first = false;
    }

    REQUIRE(history.size() > 2000);

    // The worst centre jump, which is the assertion that fails.
    auto worst = std::size_t{0};
    for (auto index = std::size_t{0}; index < history.size(); index++)
    {
        worst = history[index].jump > history[worst].jump ? index : worst;
    }

    auto sortedJumps = std::vector<double>{};
    for (const auto& entry : history)
    {
        sortedJumps.push_back(entry.jump);
    }
    std::sort(sortedJumps.begin(), sortedJumps.end());

    std::printf("\n=== the centre-jump distribution over %zu contacting ticks ===\n", history.size());
    std::printf("  worst %.2f mm, p99.9 %.2f mm, p99 %.2f mm, median %.2f mm\n", 1000.0 * sortedJumps.back(),
                1000.0 * sortedJumps[sortedJumps.size() * 999 / 1000],
                1000.0 * sortedJumps[sortedJumps.size() * 99 / 100], 1000.0 * sortedJumps[sortedJumps.size() / 2]);

    auto overEight = std::size_t{0};
    for (const auto jump : sortedJumps)
    {
        overEight += jump > 0.008 ? 1u : 0u;
    }
    std::printf("  %zu ticks over 8 mm\n", overEight);

    auto sortedLoads = std::vector<double>{};
    for (const auto& entry : history)
    {
        sortedLoads.push_back(entry.loadJump);
    }
    std::sort(sortedLoads.begin(), sortedLoads.end());

    std::printf("  load jumps: worst %.0f N, p99.9 %.0f N, p99 %.0f N, median %.0f N\n", sortedLoads.back(),
                sortedLoads[sortedLoads.size() * 999 / 1000], sortedLoads[sortedLoads.size() * 99 / 100],
                sortedLoads[sortedLoads.size() / 2]);

    std::printf("\n=== the worst tick, step %d, and the four either side of it ===\n", history[worst].step);

    const auto from = worst > 2 ? worst - 2 : std::size_t{0};
    const auto to = std::min(worst + 3, history.size());

    for (auto index = from; index < to; index++)
    {
        const auto anatomy = anatomyOf(world.value(), setup.value(), states[index], solutions[index], wheel);

        std::printf("\nstep %d%s  offset %.2f mm  jump %.2f mm  load %.0f N  loadJump %.0f N  contacting %u\n",
                    history[index].step, index == worst ? "  <== WORST" : "", 1000.0 * history[index].centreOffset,
                    1000.0 * history[index].jump, history[index].load, history[index].loadJump,
                    history[index].contacting);
        printAnatomy("sampler:", anatomy);
    }

    // And the same question asked of the whole run rather than of one tick: how many ticks did the
    // aggregate reject a sample on, and what do the jumps on those ticks look like against the rest?
    auto rejectedTicks = std::size_t{0};
    auto worstRejected = 0.0;
    auto worstClean = 0.0;

    for (auto index = std::size_t{0}; index < history.size(); index++)
    {
        const auto anatomy = anatomyOf(world.value(), setup.value(), states[index], solutions[index], wheel);

        if (anatomy.rejected)
        {
            rejectedTicks++;
            worstRejected = std::max(worstRejected, history[index].jump);
        }
        else
        {
            worstClean = std::max(worstClean, history[index].jump);
        }
    }

    std::printf("\n=== spike rejection over the whole crossing ===\n");
    std::printf("  %zu of %zu ticks rejected at least one sample.\n", rejectedTicks, history.size());
    std::printf("  worst centre jump on a tick that rejected:      %.2f mm\n", 1000.0 * worstRejected);
    std::printf("  worst centre jump on a tick that rejected none: %.2f mm\n", 1000.0 * worstClean);
    std::printf("\n  If the second number is under the 12 mm bound and the first is not, the fault is\n"
                "  the rejection rule and not the spring bed. If both are over, the bed is in it too.\n");

    // **How many rays came back with no hit at all**, which is a third mechanism neither record
    // names. A ray starting a metre above the tread and looking two metres down at a road twenty
    // millimetres above that tread cannot miss for any reason that is about the tyre.
    auto missTicks = std::size_t{0};
    auto missed = std::size_t{0};
    auto worstMissJump = 0.0;
    auto worstHitJump = 0.0;

    for (auto index = std::size_t{0}; index < history.size(); index++)
    {
        const auto anatomy = anatomyOf(world.value(), setup.value(), states[index], solutions[index], wheel);

        if (anatomy.missed > 0)
        {
            missTicks++;
            missed += anatomy.missed;
            worstMissJump = std::max(worstMissJump, history[index].jump);
        }
        else
        {
            worstHitJump = std::max(worstHitJump, history[index].jump);
        }
    }

    std::printf("\n=== rays that came back with no hit ===\n");
    std::printf("  %zu of %zu ticks lost at least one ray; %zu rays of %zu in total.\n", missTicks, history.size(),
                missed, 9 * history.size());
    std::printf("  worst centre jump on a tick that lost a ray:  %.2f mm\n", 1000.0 * worstMissJump);
    std::printf("  worst centre jump on a tick that lost none:   %.2f mm\n", 1000.0 * worstHitJump);
}

TEST_CASE("whether the lost ray is a degeneracy in the mesh or in the query", "[.kerb-discontinuity]")
{
    // The worst tick loses one ray while its two neighbours in the same row report 19.5 and 18.8 mm
    // of road. Whatever that is, it is not the tyre. This asks what it is: the ray is re-cast from
    // exactly where it was, and then swept a few micrometres either way in x and in z. A miss that
    // survives the sweep is a query fault; a miss confined to a line is the mesh's own edge, and the
    // line it is confined to says which edge.
    const JoltGuard jolt;

    const auto descriptor = kerbGround();
    const auto world = PhysicsWorld::create(generateProvingGround(descriptor).value());
    REQUIRE(world.has_value());

    // The lost ray's origin at step 2009, printed by the case above. Restated as a literal because
    // re-deriving it means re-driving two thousand ticks to ask a question about one ray.
    const auto origin = glm::dvec3(2.3742377382, 1.0298479792, 63.4999740182);

    std::printf("\n=== sweeping the lost ray's origin, one micrometre at a time ===\n");
    std::printf("  the kerb's inner edge is x = %.4f and the mesh's cell size is %.4f, so mesh lines\n"
                "  in x sit on multiples of %.4f from the ground's own -x edge.\n",
                descriptor.kerbInnerEdge, descriptor.cellSize, descriptor.cellSize);
    std::printf("\n%16s %16s %10s %14s\n", "x", "z", "hit", "road y");

    for (const auto offset : {-1e-4, -1e-5, -1e-6, -1e-7, 0.0, 1e-7, 1e-6, 1e-5, 1e-4})
    {
        const auto from = std::vector<glm::dvec3>{origin + glm::dvec3(offset, 0.0, 0.0)};
        const auto direction = std::vector<glm::dvec3>{glm::dvec3(0.0, -1.0, 0.0)};

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(from, direction, 2.0, hits);

        std::printf("%16.7f %16.7f %10s %14.4f\n", from.front().x, from.front().z, hits.front().hit ? "yes" : "NO",
                    hits.front().hit ? hits.front().point.y : 0.0);
    }

    for (const auto offset : {-1e-4, -1e-6, 1e-6, 1e-4})
    {
        const auto from = std::vector<glm::dvec3>{origin + glm::dvec3(0.0, 0.0, offset)};
        const auto direction = std::vector<glm::dvec3>{glm::dvec3(0.0, -1.0, 0.0)};

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(from, direction, 2.0, hits);

        std::printf("%16.7f %16.7f %10s %14.4f\n", from.front().x, from.front().z, hits.front().hit ? "yes" : "NO",
                    hits.front().hit ? hits.front().point.y : 0.0);
    }

    // The band, resolved. If it is a shared-edge degeneracy it is a line and a micrometre either way
    // clears it; anything wider than that is not the edge and the width says what it is instead.
    std::printf("\n=== the dead band in z, at 5 um ===\n");
    auto bandFrom = 0.0;
    auto bandTo = 0.0;

    for (auto step = 0; step <= 400; step++)
    {
        const auto z = 63.4990 + 5e-6 * static_cast<double>(step);

        const auto from = std::vector<glm::dvec3>{glm::dvec3(origin.x, origin.y, z)};
        const auto direction = std::vector<glm::dvec3>{glm::dvec3(0.0, -1.0, 0.0)};

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(from, direction, 2.0, hits);

        if (!hits.front().hit)
        {
            bandFrom = bandFrom > 0.0 ? bandFrom : z;
            bandTo = z;
        }
    }

    std::printf("  misses run from z = %.7f to z = %.7f, a band %.1f um wide, centred %.1f um from\n"
                "  the mesh row line at z = 63.5.\n",
                bandFrom, bandTo, 1e6 * (bandTo - bandFrom), 1e6 * (0.5 * (bandFrom + bandTo) - 63.5));

    // Whether the ray's *length* is what decides it. Jolt carries the length in the direction vector
    // and reports a fraction of it, so a longer ray is a coarser fraction — which would make this a
    // precision fault in the query rather than anything about the mesh.
    std::printf("\n=== the same ray at four search lengths ===\n");
    for (const auto reach : {0.5, 1.0, 2.0, 10.0, 100.0})
    {
        const auto from = std::vector<glm::dvec3>{origin};
        const auto direction = std::vector<glm::dvec3>{glm::dvec3(0.0, -1.0, 0.0)};

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(from, direction, reach, hits);

        std::printf("  reach %7.2f m: %s\n", reach, hits.front().hit ? "hit" : "MISS");
    }

    // And whether it is this z line or every z line.
    std::printf("\n=== how many of the ground's row lines have a dead band on them ===\n");
    auto deadLines = std::size_t{0};
    for (auto line = 0; line < 200; line++)
    {
        const auto z = 50.0 + 0.1 * static_cast<double>(line);

        const auto from = std::vector<glm::dvec3>{glm::dvec3(origin.x, origin.y, z)};
        const auto direction = std::vector<glm::dvec3>{glm::dvec3(0.0, -1.0, 0.0)};

        auto hits = std::vector<SurfaceHit>{};
        world->castRays(from, direction, 2.0, hits);

        deadLines += hits.front().hit ? 0u : 1u;
    }
    std::printf("  %zu of 200 row lines between z = 50 and z = 70 lose a ray cast exactly on them.\n", deadLines);

    // And the same question asked over a whole grid of the ground, to find out how common it is.
    // A ray on a mesh line is not rare under a wheel: the sampler's grid pitch and the ground's are
    // both regular, so once they align they stay aligned across a whole row.
    auto lost = std::size_t{0};
    auto total = std::size_t{0};

    for (auto ix = 0; ix < 401; ix++)
    {
        for (auto iz = 0; iz < 41; iz++)
        {
            const auto x = -2.0 + 0.01 * static_cast<double>(ix);
            const auto z = 70.0 + 0.05 * static_cast<double>(iz);

            const auto from = std::vector<glm::dvec3>{glm::dvec3(x, 1.5, z)};
            const auto direction = std::vector<glm::dvec3>{glm::dvec3(0.0, -1.0, 0.0)};

            auto hits = std::vector<SurfaceHit>{};
            world->castRays(from, direction, 3.0, hits);

            total++;
            lost += hits.front().hit ? 0u : 1u;
        }
    }

    std::printf("\n  a %zu-ray sweep of the ground at a 10 mm pitch lost %zu of them.\n", total, lost);
}
