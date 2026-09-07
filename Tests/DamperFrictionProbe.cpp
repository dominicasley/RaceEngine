// What seal friction is worth to a car that has just been hit, measured because the seat said so.
//
// Dominic's report on 2026-08-27, after driving the build that first carried it: *"the rear end
// settles realistically now after hitting a wall. Before it would bounce like a ball."* That is a
// claim about **ring-down**, and it is the one thing a Coulomb term is supposed to be worth: friction
// dissipates a fixed amount per unit of distance travelled whatever the speed, so it keeps working
// on a small residual oscillation long after a viscous damper — whose force goes to zero with the
// velocity — has stopped mattering.
//
// This probe turns that into a number and an A/B, because a seat report with a named mechanism still
// has to be isolated. It drops the car onto its wheels and counts how long the rear takes to stop
// moving, with the shipped friction (107 N front / 25 N rear since 2026-08-29) and with it zeroed
// and nothing else touched.
//
// **There are two ring-downs here now, and they claim different things** (2026-09-05).
//
//   - The **historical** one is a 1.5 m/s downward step, which is the impulse every ring-down figure
//     in `docs/` is quoted on. It is kept exactly as it was, and it is **not** an isolated
//     measurement of seal friction: measured rather than assumed, that impulse puts all four corners
//     into their bump stops for 282 corner-ticks and two of them onto their droop stops, lifts a
//     rear wheel clean off the road, and hands the rear stop **577 J** where the rear damper's seal
//     takes **6.3**. What it characterises is a whole corner at a large disturbance, and it says so.
//
//   - The **stop-free control** is the same experiment at an impulse below the measured engagement
//     boundary, where the only dissipative elements in the corner are the damper curve and the seal
//     friction this probe is named after. It asserts its own stop counts are zero, which is the
//     precondition its claim rests on — a precondition, not a number about today's car.
//
// Hidden behind a dot tag — `./EngineTests "[.damper-friction]"`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DamperForceSolution;
using raceengine::damperShaftCompression;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::seedTyreGasPressures;
using raceengine::solveDamperGeometry;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

// The production tick rate, hertz. Every figure this probe has ever published is at this rate, and
// it is the default everywhere below; the one case that varies it says so in its own name.
constexpr auto simulationRate = 360.0;

constexpr auto tick = 1.0 / simulationRate;

// The corner every amplitude, every energy and every settling time below is measured at. The seat
// report was about the rear end, and the rear is where the sourced 25 N lives.
constexpr auto rearCorner = std::size_t{2};

// How long the car is left to settle, and how long the ring-down is watched, seconds. Stated as
// durations rather than as tick counts so the timestep-refinement case measures the same experiment
// on a finer grid rather than a shorter one.
constexpr auto settleSeconds = 8.0;
constexpr auto windowSeconds = 8.0;
constexpr auto tailSeconds = 0.5;

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

[[nodiscard]] ProvingGroundDescriptor plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 200.0;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<raceengine::Feature>{};

    return descriptor;
}

// How far past its own gap a stop was driven, and how often.
//
// **Counted on penetration and not on force**, which matters on the way out: the stop's force law
// clamps at zero (`TravelStop::force`), so a corner releasing fast enough that the viscous term
// beats the elastic one reports no force while it is still inside the rubber. A count taken on the
// force would call that tick stop-free and it is not. The forces are recorded too, because the force
// is what the corner felt.
struct Engagement
{
    // Corner-ticks, summed over all four corners — so a tick with two corners in the stop counts
    // twice. The rear's own count is beside it because the rear is what this probe measures.
    int cornerTicks = 0;
    int rearTicks = 0;

    // The same two counts taken on a **non-zero force** instead, which is what the 2026-09-05 record
    // counted and is always the smaller of the two: the force law clamps at zero on a fast release,
    // so a corner can be inside the rubber and reporting nothing. Kept so this probe's numbers and
    // that entry's can be read against each other rather than argued about.
    int forceTicks = 0;
    int rearForceTicks = 0;

    double peakPast = 0.0;
    double rearPeakPast = 0.0;
    double peakForce = 0.0;

    // The closest any corner came to the stop over the whole window, millimetres past the gap — so
    // it is **negative when the stop was never touched**, and its magnitude is then the unused
    // travel separating this event from engagement. That is the margin a stop-free control is
    // chosen on: a fraction of an impulse is a property of the search, a millimetre of clearance is
    // a property of the car.
    double closest = -1.0e9;

    // The first engagement of the window: which tick, which corner, how deep and how hard. Zero
    // means the stop was never touched. Corners are examined in index order within a tick, so a
    // simultaneous pair reports the lower index.
    int firstTick = 0;
    std::size_t firstCorner = cornerCount;
    double firstPast = 0.0;
    double firstForce = 0.0;

    void observe(const int step, const std::size_t corner, const double past, const double force)
    {
        closest = std::max(closest, past * 1000.0);

        if (past <= 0.0)
        {
            return;
        }

        cornerTicks++;

        if (force > 0.0)
        {
            forceTicks++;
        }

        if (corner == rearCorner)
        {
            rearTicks++;
            rearPeakPast = std::max(rearPeakPast, past * 1000.0);

            if (force > 0.0)
            {
                rearForceTicks++;
            }
        }

        peakPast = std::max(peakPast, past * 1000.0);
        peakForce = std::max(peakForce, force);

        if (firstTick == 0)
        {
            firstTick = step;
            firstCorner = corner;
            firstPast = past * 1000.0;
            firstForce = force;
        }
    }

    [[nodiscard]] bool touched() const
    {
        return cornerTicks > 0;
    }
};

struct RingDown
{
    // The classical settling time: the last moment the corner was further than the band from where
    // it ends up, seconds. **Not** "the last moment anything moved" — that first attempt measured
    // when the last tenth of a micron happened and reported the whole window every time.
    //
    // **Reported, and no longer asserted on** (2026-09-05). It is a tick index times the timestep,
    // so it is the true crossing floored to 2.78 ms, and a strict inequality between two of them
    // resolves nothing smaller than that. Kept because every figure in `docs/` is quoted in it.
    double settlingTime = 0.0;
    // The same crossing, interpolated between the two ticks that straddle it, seconds. The quantity
    // `settlingTime` is a floor of, and what the direction claim is asserted on.
    double crossing = 0.0;
    // How far the corner still swings over the final half second, millimetres.
    double residual = 0.0;
    // The largest excursion of the whole event, millimetres, so a shorter settle cannot be bought by
    // a softer hit.
    double peak = 0.0;

    // The largest shaft speed the rear damper saw, metres per second — the axis the friction law is
    // written on, and the number that says which part of a velocity-dependent friction curve an
    // event of this size lives in.
    double peakShaftSpeed = 0.0;

    // What the rear damper took out of the corner over the window, joules, split at the same seam
    // the force law is: the curve's own viscous force, and the Coulomb term on top of it. Both are
    // a force along the shaft times the shaft velocity, so both are non-negative by construction.
    double frictionEnergy = 0.0;
    double viscousEnergy = 0.0;
    double peakFrictionForce = 0.0;
    double peakViscousForce = 0.0;

    // And what the rear **bump stop** took, the same way — net work into the stop over a window that
    // starts and ends out of contact, which is therefore its dissipation. The two peak forces beside
    // it are the stop's own elastic and viscous halves, read out of the stop's own force law rather
    // than restated here: at zero rate that law returns the elastic term alone.
    double stopWork = 0.0;
    double peakStopElastic = 0.0;
    double peakStopViscous = 0.0;

    // The rear tyre's vertical load over the window, newtons. A spread equal to the peak means the
    // wheel left the road.
    double tyreLoadPeak = 0.0;
    double tyreLoadSpread = 0.0;

    Engagement bump;
    Engagement droop;
};

// A tenth of the impact's own scale, which on the historical drop is about 2 mm of travel. Wide
// enough that tick noise cannot hold it open and narrow enough that a bouncing corner cannot hide
// inside it.
constexpr auto settledBand = 0.002;

// The historical impulse, and the one every ring-down figure in `docs/` older than 2026-09-05 is
// quoted on. Kept exactly as it was.
constexpr auto historicalImpulse = 1.5;

// **The stop-free control's impulse, chosen on stop-freedom and on nothing else.** The boundary is
// measured rather than guessed: the search case below brackets it to a millimetre per second at
// **0.3078 m/s**, where the first thing to touch is a **front** bump stop on the **viscous-only**
// arm — the arm with less dissipation reaches further, which is what makes the boundary a property
// of the pair rather than of the shipped car alone.
//
// This is a round number 19% below that boundary, and at it the nearest corner still has **4.19 mm**
// of unused travel before a 20 mm gap. It was **not** chosen to make the friction effect look big: a
// larger impulse gives a larger effect, and 0.30 m/s would have been the tempting pick at 2.5% from
// the boundary.
constexpr auto stopFreeImpulse = 0.25;

// The car after its own weight has stopped moving it, with the cavity air stated — everything a
// ring-down starts from and nothing about the ring-down itself.
struct Settled
{
    VehicleState state;
    double rest = 0.0;

    // What the settle itself did to the stops, reported once so "stop-free" can be read as the
    // statement it is: the *experiment* never touches a stop. The drop onto the wheels that gets the
    // car there is fixture start-up, it is over long before the impulse, and it is the same in both
    // arms.
    Engagement bump;
    Engagement droop;
};

// How far this corner is past each stop's own gap, on the damper shaft, exactly as the force pass
// measures it. Geometry only: `damperShaftCompression` reads the solved length and nothing else, so
// the rate this function never supplies reaches nothing it returns.
struct StopPenetration
{
    double bump = 0.0;
    double droop = 0.0;
};

[[nodiscard]] StopPenetration penetrationOf(const raceengine::CornerSetup& corner,
                                            const raceengine::SuspensionState& suspension)
{
    const auto geometry = solveDamperGeometry(corner, suspension);
    const auto compression = damperShaftCompression(
        corner, DamperForceSolution{.length = geometry.length, .lengthPerAngle = geometry.lengthPerAngle});

    return StopPenetration{.bump = compression - corner.bumpStop.gap, .droop = -compression - corner.droopStop.gap};
}

void observeStops(const VehicleSetup& setup, const raceengine::VehicleStep& stepped, const int step, Engagement& bump,
                  Engagement& droop)
{
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto past = penetrationOf(setup.corners[index], stepped.corners[index].suspension);

        bump.observe(step, index, past.bump, stepped.corners[index].forces.bumpStop);
        droop.observe(step, index, past.droop, -stepped.corners[index].forces.droopStop);
    }
}

// Settle the car. Split out of the ring-down so a search over impulses pays for it once per arm and
// every impulse starts from the same bytes — a determinism property as much as a speed one,
// `stepVehicle` being a pure function of what it is handed.
[[nodiscard]] Settled settle(const VehicleSetup& setup, const PhysicsWorld& world, const double rate = simulationRate)
{
    const auto deltaTime = 1.0 / rate;
    const auto steps = static_cast<int>(settleSeconds * rate);

    auto result = Settled{};
    result.state.chassis.position = glm::dvec3(0.0, 0.52, 100.0);

    for (auto index = 1; index <= steps; index++)
    {
        const auto stepped = stepVehicle(setup, result.state, VehicleInput{}, noDriveTorque, world, deltaTime);
        REQUIRE(stepped.has_value());

        observeStops(setup, stepped.value(), index, result.bump, result.droop);
    }

    // **State the cavity air, here and not earlier** (2026-09-05). Since the tyre pressure model
    // shipped on (`7551bfc`) the gas is a live node, and a fixture that says nothing about it is
    // measuring a pressure drift as well as whatever it thinks it is measuring: left alone, this
    // one's air falls 65 → 54.21 °C over its own eight-second settle, which is 32.98 psi against a
    // 34 psi ideal and a rear tyre **2.99% softer** than the rate the car states. The seed puts
    // every pressure-dependent number back on the ideal it is quoted at, which is the only
    // configuration in which the pressure model is inert.
    //
    // **After the settle rather than before it**, which is the order `TyrePressureTests` and
    // `TyreThermalProbe` already use — settle the mechanics, then state the thermal condition, then
    // run the experiment. Seeding before the settle states the *initial* condition and not the
    // experiment's: the gas keeps cooling through the settle and arrives at the impulse at
    // 53.34 °C, further from the ideal than doing nothing at all.
    seedTyreGasPressures(setup, result.state);

    // The rear corner's resting travel, which every amplitude below is measured against.
    const auto settled = stepVehicle(setup, result.state, VehicleInput{}, noDriveTorque, world, deltaTime);
    REQUIRE(settled.has_value());
    result.rest = settled->corners[rearCorner].suspension.wheelTravel;

    return result;
}

// Hit the settled car: a downward velocity step, which is what a body dropping back onto its wheels
// after a collision does to the suspension. A wall is not needed to ask the question — what the seat
// noticed is the *decay*, and a clean impulse measures that without a contact solver in the middle
// of it.
[[nodiscard]] RingDown ringDown(const VehicleSetup& setup, const PhysicsWorld& world, const Settled& start,
                                const double impulse, const double band = settledBand,
                                const double rate = simulationRate)
{
    const auto deltaTime = 1.0 / rate;
    const auto steps = static_cast<int>(windowSeconds * rate);
    const auto tail = static_cast<int>(tailSeconds * rate);

    auto state = start.state;
    state.chassis.linearVelocity = glm::dvec3(0.0, -impulse, 0.0);

    const auto& rear = setup.corners[rearCorner];

    auto result = RingDown{};
    auto outsideBand = 0;
    // The two samples that straddle the last crossing: the excursion on the last tick outside the
    // band, and on the first tick back inside it.
    auto lastOutside = 0.0;
    auto firstInside = 0.0;
    auto loadLow = 0.0;
    auto loadSeen = false;

    for (auto index = 1; index <= steps; index++)
    {
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, deltaTime);
        REQUIRE(stepped.has_value());

        observeStops(setup, stepped.value(), index, result.bump, result.droop);

        const auto& solution = stepped->corners[rearCorner];
        const auto shaft = solution.damperVelocity;

        // The seam the force law itself is written on: the curve's viscous force at this shaft
        // velocity, and whatever `solveDamperForce` put on top of it, which is the Coulomb term.
        // Read back rather than restated, so this cannot disagree with the production law.
        const auto viscousForce = rear.damper.at(shaft);
        const auto frictionForce = solution.forces.damper - viscousForce;

        result.peakShaftSpeed = std::max(result.peakShaftSpeed, std::abs(shaft));
        result.frictionEnergy += frictionForce * shaft * deltaTime;
        result.viscousEnergy += viscousForce * shaft * deltaTime;
        result.peakFrictionForce = std::max(result.peakFrictionForce, std::abs(frictionForce));
        result.peakViscousForce = std::max(result.peakViscousForce, std::abs(viscousForce));

        // The stop's own halves, from the stop's own law: at zero rate it returns the elastic term
        // alone, so the difference against the full force is the viscous term including the clamp.
        const auto elastic = rear.bumpStop.force(penetrationOf(rear, solution.suspension).bump, 0.0);

        result.stopWork += solution.forces.bumpStop * shaft * deltaTime;
        result.peakStopElastic = std::max(result.peakStopElastic, elastic);
        result.peakStopViscous = std::max(result.peakStopViscous, std::abs(solution.forces.bumpStop - elastic));

        const auto load = solution.forces.tireVertical;
        result.tyreLoadPeak = std::max(result.tyreLoadPeak, load);
        loadLow = loadSeen ? std::min(loadLow, load) : load;
        loadSeen = true;

        const auto excursion = std::abs(solution.suspension.wheelTravel - start.rest);

        result.peak = std::max(result.peak, excursion * 1000.0);

        if (excursion > band)
        {
            outsideBand = index;
            lastOutside = excursion;
            firstInside = 0.0;
        }
        else if (outsideBand == index - 1 && firstInside == 0.0)
        {
            firstInside = excursion;
        }

        if (index > steps - tail)
        {
            result.residual = std::max(result.residual, excursion * 1000.0);
        }
    }

    result.tyreLoadSpread = result.tyreLoadPeak - loadLow;
    result.settlingTime = static_cast<double>(outsideBand) * deltaTime;

    // Where inside that last tick the corner actually crossed the band, linearly between the two
    // samples either side of it. The envelope is on its way down and nearly straight over one tick
    // at this ring-down's frequency, so this is a much better estimate of the crossing than the
    // floor is — and it is the *same* crossing, with the same band and the same definition, not a
    // different observable. Falls back to the floor if the window ended still outside the band,
    // which on this drop it never does.
    const auto straddled = firstInside > 0.0 && lastOutside > firstInside;
    const auto fraction = straddled ? (lastOutside - band) / (lastOutside - firstInside) : 0.0;

    result.crossing = straddled ? (static_cast<double>(outsideBand) + fraction) * deltaTime : result.settlingTime;

    return result;
}

[[nodiscard]] VehicleSetup withoutFriction(const VehicleSetup& setup)
{
    auto result = setup;
    for (auto& corner : result.corners)
    {
        corner.damperFriction = 0.0;
    }

    return result;
}

void reportArm(const std::string& label, const RingDown& run)
{
    WARN(label << ": peak " << run.peak << " mm, settles to the band in " << run.settlingTime << " s (crossing "
               << run.crossing << " s), residual " << run.residual << " mm");
    WARN("    rear shaft peak " << run.peakShaftSpeed * 1000.0 << " mm/s, friction " << run.frictionEnergy << " J ("
                                << run.peakFrictionForce << " N peak), viscous " << run.viscousEnergy << " J ("
                                << run.peakViscousForce << " N peak); rear tyre load peak " << run.tyreLoadPeak
                                << " N, spread " << run.tyreLoadSpread << " N");
    WARN("    bump stop " << run.bump.cornerTicks << " corner-ticks (" << run.bump.rearTicks << " rear), of which "
                          << run.bump.forceTicks << " (" << run.bump.rearForceTicks
                          << " rear) carry a force; peak " << run.bump.peakPast << " mm past (" << run.bump.rearPeakPast
                          << " rear) / " << run.bump.peakForce << " N; droop " << run.droop.cornerTicks
                          << " corner-ticks (" << run.droop.rearTicks << " rear), peak " << run.droop.peakPast
                          << " mm past / " << run.droop.peakForce << " N");
    WARN("    closest approach: bump " << run.bump.closest << " mm past the gap, droop " << run.droop.closest
                                       << " mm (negative is clearance); the rear stop took " << run.stopWork
                                       << " J, peak " << run.peakStopElastic << " N elastic / " << run.peakStopViscous
                                       << " N viscous");
}

void reportDelta(const RingDown& shipped, const RingDown& viscous)
{
    WARN("  delta (shipped - viscous only): crossing "
         << (shipped.crossing - viscous.crossing) * 1000.0 << " ms, which is "
         << (shipped.crossing / viscous.crossing - 1.0) * 100.0 << "% of the viscous-only settle; peak "
         << shipped.peak - viscous.peak << " mm (" << (shipped.peak / viscous.peak - 1.0) * 100.0 << "%); residual "
         << shipped.residual - viscous.residual << " mm (" << (shipped.residual / viscous.residual - 1.0) * 100.0
         << "%)");
    WARN("  rear damper dissipation: friction "
         << shipped.frictionEnergy << " J of " << shipped.frictionEnergy + shipped.viscousEnergy << " J total, "
         << shipped.frictionEnergy / std::max(shipped.frictionEnergy + shipped.viscousEnergy, 1e-12) * 100.0
         << "%; the viscous-only arm dissipates " << viscous.viscousEnergy << " J");
}

} // namespace

TEST_CASE("damper friction: what it is worth to a car that has just been hit", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto without = withoutFriction(built.value());

    const auto shippedStart = settle(built.value(), world.value());
    const auto viscousStart = settle(without, world.value());

    const auto shipped = ringDown(built.value(), world.value(), shippedStart, historicalImpulse);
    const auto viscous = ringDown(without, world.value(), viscousStart, historicalImpulse);

    WARN("=== rear ring-down after a " << historicalImpulse
                                       << " m/s downward impulse, cavity gas seeded to the ideal ===");
    WARN("  the settle onto the wheels itself touched the bump stop on "
         << shippedStart.bump.cornerTicks << " corner-ticks and the droop stop on " << shippedStart.droop.cornerTicks
         << ", all of it over long before the impulse");
    reportArm("shipped (25 N rear seal friction)", shipped);
    reportArm("viscous only (0 N)               ", viscous);
    reportDelta(shipped, viscous);

    // **This case is a WHOLE-CORNER LARGE-DISTURBANCE characterisation and is labelled as one**
    // (2026-09-05). Asserted rather than remarked: this impulse *does* engage the bump stop, so the
    // ring-down it measures is a damper, a seal, a tyre and a stop carrying a placed 40000 N·s/m at
    // five times a front corner's critical damping. The friction's share is not separable here and
    // this probe no longer implies it is; what isolates the seal is the stop-free control below.
    //
    // Nothing was weakened to make room for that: same impulse, same band, same corner, same three
    // assertions, same numbers to the bit.
    CHECK(shipped.bump.touched());

    // The claim the seat made, asserted: the corner stops sooner with friction than without it, and
    // it is not bought by a softer hit. The numbers are **not** pinned — what is pinned is the
    // direction, because that is what the report said and it is what a Coulomb term must do. A pinned
    // figure here would be a characterisation of borrowed class figures nobody has measured on this
    // car — the front is a Passat B8 strut's, the rear a compact-class monotube's.
    //
    // **On the interpolated crossing and not on `settlingTime`, since 2026-09-05, and this is a
    // resolution change rather than a loosening**: same band, same corner, same last-crossing
    // definition, same strict inequality, and no tolerance anywhere. `settlingTime` is that crossing
    // floored to one 2.78 ms tick, so it reports a margin of one, two or three ticks for the same
    // physics depending on where the crossing happens to fall inside a tick — measured on the seeded
    // plant, the viscous arm's crossing sits **0.21 ms** from the next tick boundary. The effect
    // itself is 7.14 ms. A detector whose reading changes by a whole quantum for a 0.21 ms change in
    // the car is not measuring the car, and in 2026-08-29..09-05 it stopped: the tyre pressure model
    // shipping on moved both arms by about a tick and the floored margin collapsed to an exact tie
    // while the crossing still read −1.49 ms in the sourced direction.
    CHECK(shipped.crossing < viscous.crossing);
    // 2026-08-29, with the rear sourced at 25 N: the residual's direction claim went with the 107 N
    // that made it — the sub-micron difference collapsed from -33% to +0.4%, below this probe's
    // resolution — so what is asserted is that the corner ends dead still, a hundredth of the
    // settled band, and the direction rides the band crossing alone.
    CHECK(shipped.residual < 0.02);
    CHECK(shipped.peak > 0.9 * viscous.peak);
}

TEST_CASE("damper friction: the impulse at which this ring-down first touches a stop", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto without = withoutFriction(built.value());

    const auto shippedStart = settle(built.value(), world.value());
    const auto viscousStart = settle(without, world.value());

    // Stop-free means stop-free in **both** arms, because the control runs them side by side: the
    // boundary that matters is whichever arm engages at the lower impulse.
    const auto engages = [&](const double impulse)
    {
        const auto a = ringDown(built.value(), world.value(), shippedStart, impulse);
        const auto b = ringDown(without, world.value(), viscousStart, impulse);

        return a.bump.touched() || a.droop.touched() || b.bump.touched() || b.droop.touched();
    };

    // A coarse scan first, so the bracket the bisection starts from is measured rather than assumed,
    // and so the printed table says what the transition looks like rather than only where it is.
    WARN("=== stop engagement against impulse, both arms, 8 s window ===");

    auto stopFree = 0.0;
    auto engaging = 0.0;

    for (auto tenths = 1; tenths <= 15; tenths++)
    {
        const auto impulse = 0.1 * static_cast<double>(tenths);
        const auto shipped = ringDown(built.value(), world.value(), shippedStart, impulse);
        const auto viscous = ringDown(without, world.value(), viscousStart, impulse);

        WARN("  " << impulse << " m/s: shipped bump " << shipped.bump.cornerTicks << " droop "
                  << shipped.droop.cornerTicks << " (peak travel " << shipped.peak << " mm, bump clearance "
                  << shipped.bump.closest << " mm); viscous bump " << viscous.bump.cornerTicks << " droop "
                  << viscous.droop.cornerTicks << " (peak travel " << viscous.peak << " mm, bump clearance "
                  << viscous.bump.closest << " mm)");

        const auto hit =
            shipped.bump.touched() || shipped.droop.touched() || viscous.bump.touched() || viscous.droop.touched();

        if (hit && engaging <= 0.0)
        {
            engaging = impulse;

            // Which of the four candidate engagements happened first — arm, stop, corner and tick.
            // Ties inside a tick go to whichever is examined first here, which is stated rather than
            // resolved because a tie is not a fact about the car.
            const auto candidates = std::array{std::pair{"shipped bump", &shipped.bump},
                                               std::pair{"shipped droop", &shipped.droop},
                                               std::pair{"viscous bump", &viscous.bump},
                                               std::pair{"viscous droop", &viscous.droop}};

            const std::pair<const char*, const Engagement*>* earliest = nullptr;

            for (const auto& candidate : candidates)
            {
                if (candidate.second->touched() &&
                    (earliest == nullptr || candidate.second->firstTick < earliest->second->firstTick))
                {
                    earliest = &candidate;
                }
            }

            REQUIRE(earliest != nullptr);

            WARN("    first engagement: " << earliest->first << ", corner " << earliest->second->firstCorner
                                          << ", tick " << earliest->second->firstTick << " ("
                                          << static_cast<double>(earliest->second->firstTick) * tick << " s), "
                                          << earliest->second->firstPast << " mm past the gap at "
                                          << earliest->second->firstForce << " N");
        }

        if (!hit)
        {
            stopFree = impulse;
        }
    }

    REQUIRE(engaging > 0.0);
    REQUIRE(stopFree > 0.0);
    REQUIRE(stopFree < engaging);

    // Bisect the coarse bracket to a millimetre per second, which is three orders below the impulse
    // and one below anything a fixture would choose. A diagnostic and not an optimiser: it locates
    // the transition and stops.
    auto low = stopFree;
    auto high = engaging;

    while (high - low > 0.001)
    {
        const auto middle = 0.5 * (low + high);

        if (engages(middle))
        {
            high = middle;
        }
        else
        {
            low = middle;
        }
    }

    WARN("  boundary: largest stop-free impulse tested " << low << " m/s, smallest engaging " << high << " m/s");

    const auto edgeShipped = ringDown(built.value(), world.value(), shippedStart, high);
    const auto edgeViscous = ringDown(without, world.value(), viscousStart, high);
    const auto& engaged = edgeShipped.bump.touched() ? edgeShipped : edgeViscous;
    const auto& cleared = edgeShipped.bump.touched() ? edgeViscous : edgeShipped;

    WARN("  at " << high << " m/s the engaging arm is the " << (edgeShipped.bump.touched() ? "shipped" : "viscous-only")
                 << " one — corner " << engaged.bump.firstCorner << " on tick " << engaged.bump.firstTick << ", "
                 << engaged.bump.firstPast << " mm past the gap at " << engaged.bump.firstForce << " N, for "
                 << engaged.bump.cornerTicks << " corner-ticks; the other arm still clears the gap by "
                 << -cleared.bump.closest << " mm");

    const auto justFree = ringDown(without, world.value(), viscousStart, low);
    WARN("  at " << low << " m/s the nearer arm clears the bump gap by " << -justFree.bump.closest
                 << " mm and the droop gap by " << -justFree.droop.closest << " mm");

    const auto control = ringDown(without, world.value(), viscousStart, stopFreeImpulse);
    WARN("  the chosen control at " << stopFreeImpulse << " m/s sits " << (1.0 - stopFreeImpulse / high) * 100.0
                                    << "% below the boundary and clears the bump gap by " << -control.bump.closest
                                    << " mm");

    // The one thing a search may assert about itself: it found a transition and did not cross it.
    // **No figure here is pinned** — the boundary is a property of today's plant, and the control
    // below states its own precondition directly instead of trusting this number to stay put.
    CHECK(low < high);
}

TEST_CASE("damper friction: the same A/B where no stop is ever touched", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto without = withoutFriction(built.value());

    const auto shippedStart = settle(built.value(), world.value());
    const auto viscousStart = settle(without, world.value());

    const auto shipped = ringDown(built.value(), world.value(), shippedStart, stopFreeImpulse);
    const auto viscous = ringDown(without, world.value(), viscousStart, stopFreeImpulse);

    WARN("=== rear ring-down after a " << stopFreeImpulse << " m/s downward impulse — STOP-FREE CONTROL ===");
    reportArm("shipped (25 N rear seal friction)", shipped);
    reportArm("viscous only (0 N)               ", viscous);
    reportDelta(shipped, viscous);

    // **The precondition this case's whole claim rests on**, asserted rather than assumed: no corner
    // was ever inside a stop, in either arm, on any tick of the window. If a plant change moves the
    // boundary below this impulse, this fails and says so — which is the point of asserting a
    // precondition instead of pinning today's crossing.
    REQUIRE(shipped.bump.cornerTicks == 0);
    REQUIRE(shipped.droop.cornerTicks == 0);
    REQUIRE(viscous.bump.cornerTicks == 0);
    REQUIRE(viscous.droop.cornerTicks == 0);

    // The same direction claim as the historical case, on the same continuous observable, now with
    // nothing dissipative in the corner but the damper curve and the seal.
    CHECK(shipped.crossing < viscous.crossing);
    CHECK(shipped.peak > 0.9 * viscous.peak);

    // **How much of the size of that difference is real, and the answer is: the sign, and not much
    // else.** The settling band is arbitrary — 2 mm is a tenth of the *historical* impulse's
    // excursion and this impulse is six times smaller — so reading the same A/B at a family of bands
    // says which part of the answer is the car and which part is where the envelope happened to
    // cross.
    //
    // Measured, the direction is negative at every band from 1 mm to 5 mm, and the **magnitude spans
    // two and a half orders**: bands below about 2.4 mm put the two arms' last outside-band
    // excursion on *different half cycles*, which is worth 15 to 140 ms; bands above it put both on
    // the same decaying swing, where the difference is 0.4 to 1.7 ms and grows smoothly with the
    // band. The crossing is a continuous observable in the tick and a **quantised** one in the
    // ring-down's own half period, and no amount of interpolation fixes the second.
    for (auto half = 2; half <= 10; half++)
    {
        const auto band = 0.0005 * static_cast<double>(half);
        const auto a = ringDown(built.value(), world.value(), shippedStart, stopFreeImpulse, band);
        const auto b = ringDown(without, world.value(), viscousStart, stopFreeImpulse, band);

        WARN("  band " << band * 1000.0 << " mm: crossing " << a.crossing << " s against " << b.crossing << " s, delta "
                       << (a.crossing - b.crossing) * 1000.0 << " ms");

        CHECK(a.crossing < b.crossing);
    }
}

TEST_CASE("damper friction: the stop-free direction survives a finer timestep", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto without = withoutFriction(built.value());

    // **Whether a sub-millisecond difference is the car or the sampling grid.** In the continuous
    // regime the stop-free effect is a fraction of a tick, so it rests entirely on the interpolation
    // between two samples — and the way to find out whether that is trustworthy is to take more
    // samples. Refining the timestep changes the plant slightly, the corner's damping being solved
    // per tick, so this is not an inertness proof; it is an error estimate. If the effect holds its
    // size and its sign as the grid is refined threefold, it is a property of the car.
    WARN("=== the stop-free A/B at three timesteps, 3 mm band ===");

    for (const auto rate : std::array{360.0, 720.0, 1080.0})
    {
        const auto shippedStart = settle(built.value(), world.value(), rate);
        const auto viscousStart = settle(without, world.value(), rate);

        // At the 3 mm band, which the case above measures to be in the regime where both arms cross
        // on the same swing — the only regime in which a *magnitude* means anything.
        const auto a = ringDown(built.value(), world.value(), shippedStart, stopFreeImpulse, 0.003, rate);
        const auto b = ringDown(without, world.value(), viscousStart, stopFreeImpulse, 0.003, rate);

        WARN("  " << rate << " Hz: crossing " << a.crossing << " s against " << b.crossing << " s, delta "
                  << (a.crossing - b.crossing) * 1000.0 << " ms (" << (a.crossing - b.crossing) * rate
                  << " ticks); friction energy " << a.frictionEnergy << " J, peak travel " << a.peak << " against "
                  << b.peak << " mm");

        REQUIRE(a.bump.cornerTicks == 0);
        REQUIRE(b.bump.cornerTicks == 0);

        CHECK(a.crossing < b.crossing);
    }
}

TEST_CASE("damper friction: the stop-free fixture answers to friction magnitude", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    // **A sensitivity run and NOT a car.** Four of these five rear friction magnitudes are
    // fabricated and only the 25 N is sourced. It answers one question about the *fixture* — does
    // its observable move with the quantity it is supposed to be measuring, or is the shipped arm's
    // margin a coincidence of where the envelope crossed? **Nothing here is a claim about the car
    // and nothing here is an acceptance criterion.** The fronts keep their shipped 107 N throughout,
    // so what is varied is the rear axle alone.
    auto previousWide = 1.0e9;
    auto previousPeak = 1.0e9;
    auto previousEnergy = -1.0;

    WARN("=== stop-free response to rear friction magnitude — NON-PRODUCTION sensitivity ===");

    for (const auto newtons : std::array{0.0, 12.5, 25.0, 50.0, 100.0})
    {
        auto setup = built.value();
        setup.corners[2].damperFriction = newtons;
        setup.corners[3].damperFriction = newtons;

        const auto start = settle(setup, world.value());
        const auto run = ringDown(setup, world.value(), start, stopFreeImpulse);
        const auto wide = ringDown(setup, world.value(), start, stopFreeImpulse, 0.003);

        WARN("  rear " << newtons << " N: crossing " << run.crossing << " s at the 2 mm band, " << wide.crossing
                       << " s at the 3 mm band; peak " << run.peak << " mm, friction " << run.frictionEnergy
                       << " J, bump ticks " << run.bump.cornerTicks);

        REQUIRE(run.bump.cornerTicks == 0);
        REQUIRE(run.droop.cornerTicks == 0);

        // Three observables, and **the third is why the crossing is quoted at the wide band**.
        // Dissipated friction energy must rise with the magnitude and the first excursion must fall;
        // both are monotone across the whole sweep. The crossing is monotone too — but only at a
        // band in the continuous regime. At the 2 mm band it jumps a half cycle between 12.5 and
        // 25 N and then rises again by about 0.8 ms over the rest of the sweep, which is the peak
        // quantisation of the case above and not a failure of the physics.
        CHECK(run.frictionEnergy > previousEnergy);
        CHECK(run.peak < previousPeak);
        CHECK(wide.crossing < previousWide);

        previousEnergy = run.frictionEnergy;
        previousPeak = run.peak;
        previousWide = wide.crossing;
    }
}

TEST_CASE("damper friction: the two fixtures side by side, each at its own scale", "[.damper-friction]")
{
    const JoltGuard jolt;

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto without = withoutFriction(built.value());

    const auto shippedStart = settle(built.value(), world.value());
    const auto viscousStart = settle(without, world.value());

    // **The comparison the two cases above cannot make between them**, because they read the same
    // fixed 2 mm band at excursions that differ by three and a half times: 2 mm is 5.6% of the
    // historical event and 19.8% of the stop-free one, so a settling time taken at it is measured
    // at a different point on each envelope. Here the band is a stated **fraction of each arm's own
    // peak**, which is the only way "the same measurement, six times smaller" means anything.
    //
    // The peak-travel and energy ratios need no such care — they are ratios of the same quantity in
    // both arms — and they are printed beside it so the normalisation can be read against them.
    WARN("=== the two impulses at bands scaled to their own excursion ===");

    for (const auto impulse : std::array{historicalImpulse, stopFreeImpulse})
    {
        const auto reference = ringDown(without, world.value(), viscousStart, impulse);

        WARN("  " << impulse << " m/s: viscous-only peak " << reference.peak << " mm, bump stop "
                  << reference.bump.cornerTicks << " corner-ticks, rear stop work " << reference.stopWork << " J");

        for (const auto fraction : std::array{0.1, 0.2, 0.3})
        {
            const auto band = fraction * reference.peak / 1000.0;
            const auto a = ringDown(built.value(), world.value(), shippedStart, impulse, band);
            const auto b = ringDown(without, world.value(), viscousStart, impulse, band);

            WARN("    band " << fraction * 100.0 << "% of peak (" << band * 1000.0 << " mm): crossing " << a.crossing
                             << " s against " << b.crossing << " s, delta " << (a.crossing - b.crossing) * 1000.0
                             << " ms = " << (a.crossing / b.crossing - 1.0) * 100.0 << "%");

            CHECK(a.crossing < b.crossing);
        }

        const auto shipped = ringDown(built.value(), world.value(), shippedStart, impulse);

        WARN("    peak " << (shipped.peak / reference.peak - 1.0) * 100.0 << "%, friction energy "
                         << shipped.frictionEnergy << " J = "
                         << shipped.frictionEnergy /
                                std::max(shipped.frictionEnergy + shipped.viscousEnergy, 1e-12) * 100.0
                         << "% of the rear damper's dissipation and "
                         << shipped.frictionEnergy /
                                std::max(shipped.frictionEnergy + shipped.viscousEnergy + shipped.stopWork, 1e-12) *
                                100.0
                         << "% of the whole rear corner's");
    }
}
