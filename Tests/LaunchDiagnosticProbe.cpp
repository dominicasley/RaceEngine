// T1: what the auto-clutch actually does on a standing start. `./EngineTests "[.launch]"`.
//
// The T-series brief's diagnosis is that the DSG's *plant* is right — a pair of wet mechanical
// clutches genuinely is a clutch and not a converter — and its *controller* is missing. Anti-stall is
// one rule of a TCU that needs several, and it is the protective rule rather than the productive one.
//
// Before any of that is built, this measures what the missing rule is worth, and it does the whole of
// T1's unattended half:
//
//   - **0-100 km/h on the corrected car**, re-measured rather than quoted.
//   - **The ordering question.** Does clutch slip reach zero before wheel slip does? A clutch that
//     locks while the tyre is still spinning hard has handed full torque to a tyre that cannot take
//     it, and that is a controller fault rather than a tyre one.
//   - **The inertial dump.** Does clutch torque ever exceed what the engine is making? If it does,
//     the engine's stored rotational energy is being transferred as a spike, and one spike past peak
//     slip loses the launch because the tyre never recovers.
//   - **The anti-stall limit cycle.** Tyre bites, revs drop, coupling opens, torque vanishes, revs
//     recover, re-engages. Visible as chatter on the clutch torque trace.
//   - **`Tyre Fx` against `mu_peak * Fz`.** If the tyre makes only a small fraction of the force
//     available to it at high slip, the longitudinal curve's fall-off past peak is too aggressive,
//     and that is a separate finding that must not be absorbed into the clutch one.
//   - **The spread**, over repeats.
//
// What it cannot do is the other half of T1: a human on the clutch pedal, performing by hand the
// launch control the TCU should perform. That needs a seat. This is the baseline it gets compared to.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DrivelineState;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::tyreFriction;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;

constexpr std::array<double, cornerCount> noRoadTorque{};

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

// Long and narrow, and the car starts at one end. The generator lays the ground from z = 0 to
// z = length rather than centring it on the origin, so a car spawned at a negative station is a car
// dropped off the end of the world — which reads as a car that will not accelerate.
[[nodiscard]] ProvingGroundDescriptor straightGround()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 12000.0;
    descriptor.width = 40.0;
    descriptor.cellSize = 4.0;
    descriptor.features = {};

    return descriptor;
}

[[nodiscard]] double designHeight(const VehicleSetup& setup)
{
    auto highest = 0.0;
    for (const auto& corner : setup.corners)
    {
        highest = std::max(highest, corner.hardpoints.wheelCentre.y + corner.hardpoints.wheelRadius);
    }

    return highest;
}

// One tick of the launch, as recorded. Everything the four questions need, on one time base.
struct Tick
{
    double time = 0.0;
    double speed = 0.0;
    double engineSpeed = 0.0;
    double clutchPedal = 0.0;
    double clutchTorque = 0.0;
    double engineTorque = 0.0;
    double clutchSlip = 0.0;
    double drivenSlipRatio = 0.0;
    double drivenSurfaceSpeed = 0.0;
    double forceX = 0.0;
    double loadZ = 0.0;
    double available = 0.0;
    int gear = 1;
};

// A whole standing start, full throttle, shifting at the limiter. `settleTicks` is the only knob and
// it exists so the spread can be measured against something physical: a car that has been on its
// springs for a different length of time starts from a fractionally different attitude, which is the
// nearest thing this deterministic model has to run-to-run variation.
[[nodiscard]] std::vector<Tick> runLaunch(const PhysicsWorld& world, const int settleTicks, const int seconds = 20,
                                          const double shape = 0.0, const double curvature = 0.0,
                                          const double stiffness = 0.0, const double peak = 0.0,
                                          const double gripScale = 1.0)
{
    auto setup = golfGtiMk7().value();
    if (stiffness > 0.0)
    {
        for (auto& corner : setup.corners)
        {
            corner.tyre.longitudinalShape = shape;
            corner.tyre.longitudinalCurvature = curvature;
            corner.tyre.longitudinalStiffness = stiffness;
            if (peak > 0.0)
            {
                corner.tyre.longitudinalPeak = peak;
            }
        }
    }

    for (auto& corner : setup.corners)
    {
        corner.tyre.gripScale = gripScale;
    }
    const auto driveline = golfGtiMk7Driveline();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);
    for (auto step = 0; step < settleTicks; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    auto drivelineState = DrivelineState{};
    startEngine(driveline, drivelineState);

    const auto inertias = wheelInertias(setup);
    auto road = noRoadTorque;

    // **Settle the driveline too, not just the suspension.** `DrivelineState::clutchPedal` defaults
    // to zero, which is a *fully engaged* clutch, and the pedal is rate limited at 4 per second — so
    // a fixture that starts the engine and immediately goes to full throttle spends the first quarter
    // second dumping engine torque through a closed clutch into a stationary tyre while the
    // automation crawls the pedal open. That is a cold start, not a launch, and it flattered nothing:
    // it produced the 338 N·m torque spike at 0.003 s and most of the wheelspin that followed.
    //
    // A car in the game has been idling in gear on its brakes long before anyone floors it, so this
    // does the same: brakes on, no throttle, until the pedal has reached whatever the transmission
    // holds at a standstill.
    {
        auto idling = VehicleInput{};
        idling.brake = 1.0;
        idling.gear = 1;

        for (auto step = 0; step < 360; step++)
        {
            const auto torques = stepDriveline(driveline, drivelineState,
                                               {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                               inertias, road, idling, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(setup, state, idling, torques->wheel, world, tick);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());
        }
    }

    // **The fixture asserts the state it is about to launch from.** The last unverified launch number
    // cost an entire TCU build: `DrivelineState::clutchPedal` defaults to zero, which is a *closed*
    // clutch, so the original probe spent its first quarter second dumping torque into a stationary
    // tyre and every conclusion drawn from it was about that rather than about the controller. None
    // of these would have let it through.
    {
        const auto atRest = glm::length(state.chassis.linearVelocity);
        CAPTURE(atRest, drivelineState.clutchPedal, drivelineState.engineSpeed, drivelineState.gear);

        // Stationary, on the brakes, in gear, engine running at its idle — a car waiting to launch.
        REQUIRE(atRest < 0.05);
        REQUIRE(drivelineState.gear == 1);
        REQUIRE(drivelineState.engineSpeed > 0.8 * driveline.engine.idleSpeed);
        REQUIRE(drivelineState.engineSpeed < 1.5 * driveline.engine.idleSpeed);

        // **The clutch is open**, which is the one that was wrong. Held on the brakes at a standstill
        // a transmission does not sit slipping its clutch against them, so the pedal must be near its
        // stop rather than near zero — and the settle above must have had long enough to get it
        // there against a rate limit of `pedalRate` per second.
        REQUIRE(drivelineState.clutchPedal > 0.85);

        // And the wheels are not turning, so a slip ratio computed on the first tick means something.
        for (const auto& corner : state.corners)
        {
            REQUIRE(std::abs(corner.wheelSpeed) < 0.5);
        }
    }

    auto input = VehicleInput{};
    input.throttle = 1.0;
    input.gear = 1;

    // The engine, gated on the clutch being engaged — the one rule that distinguishes a launch flare
    // (clutch slipping, do not shift) from a car genuinely at the limiter (clutch locked, shift).
    // Established in `[.vehicle-delta]`; the other three candidate schedules all fail on this car.
    // **Not at the limiter, and `[.vehicle-delta]` says why in as many words.** Road speed through the
    // gear is the right signal and it fails at 0.995 of the limiter, because with the fronts spinning
    // the car asymptotes at 15.9 m/s and the first-to-second point sits at 16.2 — so it holds first
    // at the limiter for ever, which reads as a gearbox fault and is a tyre one. Reproduced here
    // exactly by using 0.995 and then read back off the trace: no drive at all past 4.5 s.
    //
    // 0.93 of the limiter is 6320 rpm, past peak power, and drops to about 4100 in second — in the
    // meat of the torque curve. All of this is a stand-in for the operation mode the car does not
    // have: choosing the gear is its job, and the transmission's job is only how that gear engages.
    const auto upshiftSpeed = driveline.engine.limiterSpeed * 0.93;

    const auto& tyre = setup.corners.front().tyre;

    auto recorded = std::vector<Tick>{};
    recorded.reserve(static_cast<std::size_t>(seconds) * 360);

    for (auto step = 1; step <= seconds * 360; step++)
    {
        // **Shifted on road speed, for the same reason the transmission now engages on it.** The
        // engine-gated schedule this inherited upshifted whenever the engine reached the limiter with
        // the clutch closed — and with the fronts spinning, the engine is *at* the limiter while the
        // car is doing nine kilometres an hour, so it went to third by one second and every shift
        // dumped another 480 N·m into a spinning tyre. Any signal taken off the driven wheels has
        // that fault; road speed through the gear does not.
        const auto roadSideSpeed = std::abs(state.chassis.linearVelocity.z) /
                                   setup.corners.front().hardpoints.wheelRadius * driveline.gearbox.finalDrive *
                                   driveline.gearbox.ratio(input.gear);

        if (roadSideSpeed > upshiftSpeed && input.gear < driveline.gearbox.topGear())
        {
            input.gear++;
        }

        const auto torques = stepDriveline(driveline, drivelineState,
                                           {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                            state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                           inertias, road, input, tick);
        REQUIRE(torques.has_value());

        const auto stepped = stepVehicle(setup, state, input, torques->wheel, world, tick);
        REQUIRE(stepped.has_value());

        road = roadTorques(stepped.value());

        // The driven axle is the front pair on this car, and both corners are averaged rather than
        // one being picked: a front-drive hatch on full throttle loads them unequally through torque
        // steer, and one corner's slip is not the axle's.
        const auto drivenSpeed = 0.5 * (state.corners[0].wheelSpeed + state.corners[1].wheelSpeed);
        const auto surface = drivenSpeed * setup.corners.front().hardpoints.wheelRadius;
        const auto roadSpeed = state.chassis.linearVelocity.z;

        const auto forceX =
            stepped->corners[0].contact.tyre.longitudinal + stepped->corners[1].contact.tyre.longitudinal;
        const auto loadZ = stepped->corners[0].forces.tireVertical + stepped->corners[1].forces.tireVertical;

        // What the tyre could have made at this load, at this surface, if it were exactly at its
        // peak. Load sensitivity is applied per corner rather than to the axle total, because the
        // whole point of load sensitivity is that it is not linear in load.
        const auto availableLeft = tyreFriction(tyre, tyre.longitudinalPeak, stepped->corners[0].forces.tireVertical,
                                                stepped->corners[0].patch.gripMultiplier) *
                                   stepped->corners[0].forces.tireVertical;
        const auto availableRight = tyreFriction(tyre, tyre.longitudinalPeak, stepped->corners[1].forces.tireVertical,
                                                 stepped->corners[1].patch.gripMultiplier) *
                                    stepped->corners[1].forces.tireVertical;

        recorded.push_back(Tick{.time = static_cast<double>(step) * tick,
                                .speed = roadSpeed,
                                .engineSpeed = drivelineState.engineSpeed,
                                .clutchPedal = drivelineState.clutchPedal,
                                .clutchTorque = torques->clutch,
                                .engineTorque = torques->engine,
                                .clutchSlip = torques->clutchSlip,
                                // SAE, and guarded at a standstill rather than clamped: below walking
                                // pace the ratio is meaningless and reporting a large number there is
                                // how a launch study comes to fit to its own divisor.
                                .drivenSlipRatio = roadSpeed > 1.0 ? (surface - roadSpeed) / roadSpeed : 0.0,
                                .drivenSurfaceSpeed = surface,
                                .forceX = forceX,
                                .loadZ = loadZ,
                                .available = availableLeft + availableRight,
                                .gear = input.gear});
    }

    return recorded;
}

[[nodiscard]] double timeToReach(const std::vector<Tick>& run, const double speed)
{
    for (const auto& sample : run)
    {
        if (sample.speed >= speed)
        {
            return sample.time;
        }
    }

    return -1.0;
}

} // namespace

TEST_CASE("what the auto-clutch launch actually does", "[.launch]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    const auto run = runLaunch(world.value(), 1440);

    const auto hundred = timeToReach(run, 100.0 / 3.6);
    const auto sixty = timeToReach(run, 26.8224);

    std::printf("\n=== the headline, on the corrected car ===\n");
    std::printf("  0-60 mph   %s\n", sixty < 0.0 ? "never reached" : (std::to_string(sixty) + " s").c_str());
    std::printf("  0-100 kph  %s\n", hundred < 0.0 ? "never reached" : (std::to_string(hundred) + " s").c_str());
    std::printf("\n  **The benchmark this is measured against is the Golf's own.** 6.2 s is the DSG\n"
                "  Performance figure with launch control. A manual without it is 6.4-6.7 s, and this\n"
                "  model has no launch control at all, so 6.2 is the wrong target to chase and the\n"
                "  gap to it is not all clutch.\n");

    SECTION("the first second, tick by tick, because the summaries above are only as good as their detectors")
    {
        // Added after a change that should have moved the clutch-lock time left it at exactly
        // 0.053 s — the same number to the millisecond. A summary statistic that does not move when
        // the thing it summarises is rewritten is measuring something else, and the only way to find
        // out what is to look at the trace it was taken from.
        std::printf("\n=== the launch, every 30 ms ===\n");
        std::printf("\n%7s %9s %9s %10s %11s %10s %9s %6s\n", "t s", "pedal", "eng rpm", "clutch Nm", "clutch slip",
                    "road m/s", "wheel x", "gear");

        for (const auto& sample : run)
        {
            const auto atTick = static_cast<int>(std::llround(sample.time * 360.0));
            if (atTick % 11 != 0 || sample.time > 1.2)
            {
                continue;
            }

            std::printf("%7.3f %9.3f %9.0f %10.1f %11.2f %10.3f %9.2f %6d\n", sample.time, sample.clutchPedal,
                        sample.engineSpeed * 9.549296585513721, sample.clutchTorque, sample.clutchSlip, sample.speed,
                        sample.speed > 0.1 ? sample.drivenSurfaceSpeed / sample.speed : 0.0, sample.gear);
        }
    }

    SECTION("the ordering question: does the clutch lock before the tyre stops spinning")
    {
        // **The diagnostic the brief names as the point.** If clutch slip reaches zero while wheel
        // slip is still large, the controller handed full engine torque to a tyre that could not take
        // it, and the wheelspin that follows is the controller's doing rather than the tyre's.
        auto clutchLocked = -1.0;
        auto slipSettled = -1.0;
        auto peakSlip = 0.0;
        auto peakSlipAt = 0.0;

        for (const auto& sample : run)
        {
            if (clutchLocked < 0.0 && sample.time > 0.05 && std::abs(sample.clutchSlip) < 1.0)
            {
                clutchLocked = sample.time;
            }
            if (sample.speed > 1.0 && sample.drivenSlipRatio > peakSlip)
            {
                peakSlip = sample.drivenSlipRatio;
                peakSlipAt = sample.time;
            }
        }

        // Settled means the driven axle is within 5% of road speed and stays there — checked forward
        // rather than at a single tick, so a momentary crossing on the way through does not count.
        for (auto index = std::size_t{0}; index < run.size(); index++)
        {
            if (run[index].speed <= 1.0 || run[index].drivenSlipRatio >= 0.05)
            {
                continue;
            }

            auto held = true;
            for (auto ahead = index; ahead < std::min(run.size(), index + 360); ahead++)
            {
                held = held && run[ahead].drivenSlipRatio < 0.05;
            }

            if (held)
            {
                slipSettled = run[index].time;
                break;
            }
        }

        std::printf("\n=== ordering ===\n");
        std::printf("  clutch slip falls under 1 rad/s at   %8.3f s\n", clutchLocked);
        std::printf("  driven-wheel slip peaks at           %8.3f s  (x%.2f of road speed)\n", peakSlipAt,
                    1.0 + peakSlip);
        std::printf("  driven-wheel slip settles under 5%%   %8.3f s\n", slipSettled);
        std::printf("\n  %s\n", clutchLocked >= 0.0 && slipSettled >= 0.0 && clutchLocked < slipSettled
                                    ? "**The clutch locks BEFORE the tyre settles.** Full torque reaches a spinning "
                                      "tyre, which is the controller fault the brief predicted."
                                    : "The clutch is still slipping when the tyre settles, so the controller is not "
                                      "handing torque to a spinning tyre and the wheelspin is not its doing.");
    }

    SECTION("the inertial dump: does the clutch ever pass more than the engine is making")
    {
        auto worstExcess = 0.0;
        auto worstAt = 0.0;
        auto excessTicks = 0;

        for (const auto& sample : run)
        {
            const auto excess = std::abs(sample.clutchTorque) - std::abs(sample.engineTorque);
            if (excess > 0.0)
            {
                excessTicks++;
            }
            if (excess > worstExcess)
            {
                worstExcess = excess;
                worstAt = sample.time;
            }
        }

        std::printf("\n=== inertial dump ===\n");
        std::printf("  worst clutch torque over engine torque  %8.1f Nm at %.3f s\n", worstExcess, worstAt);
        std::printf("  ticks where the clutch passed more than the engine made: %d of %zu\n", excessTicks, run.size());
        std::printf("\n  Anything much above zero is the engine's stored rotational energy being dumped\n"
                    "  through a fast lock, and one spike past peak slip loses the launch.\n");
    }

    SECTION("the anti-stall limit cycle: is the coupling chattering")
    {
        // A limit cycle is torque reversing direction repeatedly at a rate no driver input could
        // produce. Counted rather than eyeballed: sign changes in clutch torque per second, over the
        // first three seconds where anti-stall would be active.
        auto reversals = 0;
        auto previous = 0.0;
        auto window = 0;

        for (const auto& sample : run)
        {
            if (sample.time > 3.0)
            {
                break;
            }

            window++;
            if (previous != 0.0 && ((previous > 0.0) != (sample.clutchTorque > 0.0)))
            {
                reversals++;
            }
            previous = sample.clutchTorque;
        }

        std::printf("\n=== anti-stall chatter ===\n");
        std::printf("  clutch torque sign reversals in the first 3 s: %d over %d ticks\n", reversals, window);
        std::printf("  %s\n", reversals > 10 ? "**Chattering.** That is the limit cycle the brief describes."
                                             : "No limit cycle: the coupling is not hunting.");
    }

    SECTION("Tyre Fx against mu_peak x Fz, which is a tyre finding and not a clutch one")
    {
        // **Kept separate from the clutch questions deliberately.** If the tyre makes only a small
        // fraction of what is available to it while slipping hard, the longitudinal curve falls away
        // too fast past its peak — and that would be a second, independent fault that a TCU would
        // mask rather than fix.
        std::printf("\n=== longitudinal utilisation, driven axle ===\n");
        // **Longitudinal load transfer, checked rather than assumed.** A front-drive car accelerating
        // sheds load off the axle that is driving it, and how much is set by the centre of gravity's
        // height over the wheelbase. If the front is not shedding it, the car has grip it should not
        // have — and that would look exactly like a friction coefficient being too high.
        const auto staticFront = 0.614 * 1348.0 * 9.80665;
        std::printf("\n  static front axle load is %.0f N (61.4%% of 1348 kg). Transfer off it should be\n"
                    "  m*a*h/L = 1348 * a * 0.5720 / 2.638 = %.0f N per m/s^2 of acceleration.\n",
                    staticFront, 1348.0 * 0.5720 / 2.638);
        std::printf("\n%8s %10s %12s %12s %12s %10s %11s %10s\n", "t s", "slip", "Fx N", "mu.Fz N", "utilisation",
                    "gear", "front Fz N", "a m/s^2");

        for (const auto& sample : run)
        {
            const auto atSample = static_cast<int>(std::llround(sample.time * 360.0));
            if (atSample % 90 != 0 || sample.time > 6.0)
            {
                continue;
            }

            std::printf("%8.2f %10.3f %12.0f %12.0f %11.0f%% %10d %11.0f %10.2f\n", sample.time, sample.drivenSlipRatio,
                        sample.forceX, sample.available,
                        sample.available > 1.0 ? 100.0 * sample.forceX / sample.available : 0.0, sample.gear,
                        sample.loadZ, sample.forceX / 1348.0);
        }
    }
}

TEST_CASE("what each longitudinal curve is worth on a standing start", "[.launch]")
{
    // **The prediction under test**: correcting the fall-off should make 0-100 quicker, and materially
    // below about 6.3 s means something other than the fall-off is doing the work. The K-held
    // candidate is the control — same slip stiffness as shipped, shape changed alone — so if it lands
    // near the prediction and the high-K ones run away from it, the compensator is slip stiffness and
    // not shape.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    std::printf("\n=== 0-100 against the longitudinal curve ===\n");
    std::printf("\n%-26s %6s %7s %6s %12s %14s\n", "", "C", "E", "K", "0-100 kph", "peak wheel x");

    for (const auto& [name, shape, curvature, stiffness] :
         std::array<std::tuple<const char*, double, double, double>, 4>{
             {{"shipped", 1.65, -1.00, 20.0},
              {"C 1.55 / E +0.50", 1.55, 0.50, 35.0},
              {"C 1.50 / E 0.0", 1.50, 0.00, 28.0},
              {"C 1.46 / E -1.62, K held", 1.46, -1.62, 20.0}}})
    {
        const auto run = runLaunch(world.value(), 1440, 20, shape, curvature, stiffness);

        auto peakSlip = 0.0;
        for (const auto& sample : run)
        {
            if (sample.speed > 1.0)
            {
                peakSlip = std::max(peakSlip, sample.drivenSlipRatio);
            }
        }

        const auto hundred = timeToReach(run, 100.0 / 3.6);
        std::printf("%-26s %6.2f %7.2f %6.1f %11.3f s %13.2f\n", name, shape, curvature, stiffness, hundred,
                    1.0 + peakSlip);
    }
}

TEST_CASE("what longitudinal grip the published 0-100 implies", "[.launch]")
{
    // **Before hunting a compensator, check the premise.** The published 6.2 s belongs to a Golf GTI
    // on road tyres. This model runs the mod's own coefficients, and the mod states DX_REF 1.30 for
    // its Semislicks and **1.26 for its "Street"** — both of which are track-tyre numbers. A real
    // 225/40 R18 performance road tyre peaks nearer 1.0 to 1.15 longitudinally.
    //
    // So the question is not only "what is compensating" but "does the benchmark apply to this
    // tyre". Swept here: what peak longitudinal friction reproduces the published figure, with the
    // corrected curve shape held.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    std::printf("\n=== 0-100 against peak longitudinal friction, at C 1.50 / E 0.0 / K 28 ===\n");
    std::printf("\n%10s %14s %12s\n", "mu_x", "0-100 kph", "note");

    for (const auto peak : {1.30, 1.26, 1.20, 1.15, 1.10, 1.05, 1.00})
    {
        const auto run = runLaunch(world.value(), 1440, 20, 1.50, 0.0, 28.0, peak);
        const auto hundred = timeToReach(run, 100.0 / 3.6);

        const auto* note = peak > 1.29   ? "the mod's Semislicks"
                           : peak > 1.25 ? "the mod's *Street*"
                           : peak > 1.14 ? ""
                                         : "a real road tyre";

        std::printf("%10.2f %13.3f s %12s\n", peak, hundred, note);
    }
}

TEST_CASE("whether one grip scale satisfies both validations at once", "[.launch]")
{
    // **The test that settles whether the two validations disagree.** They were thought to: the
    // skidpad was believed correct at 0.9225 g with mu_y = 1.28, while the launch wanted mu_x near
    // 1.20. But 0.9225 g is superseded — the CG_LOCATION correction took criterion 5 to 0.9963 g and
    // it now measures 0.998, against a real Mk7 GTI's 0.90-0.95 g on OEM tyres. **Both directions say
    // the grip is high, by a similar amount.**
    //
    // `gripScale` is the seam the model already carries for exactly this: a runtime multiplier
    // outside the coefficients, so a thermal or pressure model can move friction without the data
    // being recalibrated. Swept here to ask whether one value lands both criteria.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    const auto plateWorld = PhysicsWorld::create(generateProvingGround(
                                                     []
                                                     {
                                                         auto d = ProvingGroundDescriptor{};
                                                         d.length = 400.0;
                                                         d.width = 400.0;
                                                         d.cellSize = 2.0;
                                                         d.features = {};
                                                         return d;
                                                     }())
                                                     .value());
    REQUIRE(plateWorld.has_value());

    // A skidpad hold, compactly: settle, hold lock, average the last second of body-frame lateral g.
    const auto skidpad = [&](const double gripScale, const double steering)
    {
        auto setup = golfGtiMk7().value();
        for (auto& corner : setup.corners)
        {
            corner.tyre.gripScale = gripScale;
        }

        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, designHeight(setup), 20.0);
        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, plateWorld.value(), tick).has_value());
        }

        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 20.0);
        for (auto& corner : state.corners)
        {
            corner.wheelSpeed = 20.0 / setup.corners.front().hardpoints.wheelRadius;
        }

        auto input = VehicleInput{};
        input.steering = steering;

        auto total = 0.0;
        auto samples = 0;
        for (auto step = 0; step < 1440; step++)
        {
            const auto stepped = stepVehicle(setup, state, input, noDriveTorque, plateWorld.value(), tick);
            REQUIRE(stepped.has_value());

            if (step >= 1080)
            {
                total += std::abs(stepped->telemetry.acceleration.x) / 9.80665;
                samples++;
            }
        }

        return total / static_cast<double>(samples);
    };

    std::printf("\n=== one grip scale, both criteria ===\n");
    std::printf("  real Mk7 GTI: skidpad 0.90-0.95 g on OEM tyres, 0-100 in 6.2 s (DSG, launch control)\n");
    std::printf("\n%10s %8s %8s %14s %14s\n", "gripScale", "mu_y", "mu_x", "skidpad g", "0-100 kph");

    for (const auto scale : {1.00, 0.95, 0.93, 0.90, 0.85})
    {
        auto peak = 0.0;
        for (const auto steering : {0.45, 0.60, 0.85})
        {
            peak = std::max(peak, skidpad(scale, steering));
        }

        const auto run = runLaunch(world.value(), 1440, 20, 0.0, 0.0, 0.0, 0.0, scale);

        std::printf("%10.2f %8.3f %8.3f %13.4f %13.3f s\n", scale, 1.28 * scale, 1.30 * scale, peak,
                    timeToReach(run, 100.0 / 3.6));
    }
}

TEST_CASE("is criterion 5 lock-limited or grip-limited", "[.launch]")
{
    // **Testing a claim I made too quickly.** The compact skidpad above barely moved when grip was cut
    // 5%, and I read that as the car running out of steering lock before it runs out of grip. The
    // steering probe says otherwise: full demand is 378 deg at the rim over a 13.80:1 ratio, which is
    // **27.4 deg at the road wheel**, and the linear-range understeer gradient off that probe is about
    // 5.9 deg/g — an ordinary road-car number. Neither is the signature of a car short of lock.
    //
    // So this sweeps demand to full lock at two grip levels and reports what is actually happening:
    // whether lateral acceleration is still climbing at the stop (lock-limited) or has asymptoted
    // (grip-limited), and whether the top of the sweep responds to grip at all.
    const JoltGuard jolt;

    const auto plateWorld = PhysicsWorld::create(generateProvingGround(
                                                     []
                                                     {
                                                         auto d = ProvingGroundDescriptor{};
                                                         d.length = 1200.0;
                                                         d.width = 1200.0;
                                                         d.cellSize = 4.0;
                                                         d.features = {};
                                                         return d;
                                                     }())
                                                     .value());
    REQUIRE(plateWorld.has_value());

    const auto hold = [&](const double gripScale, const double steering)
    {
        auto setup = golfGtiMk7().value();
        for (auto& corner : setup.corners)
        {
            corner.tyre.gripScale = gripScale;
        }

        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, designHeight(setup), 400.0);
        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, plateWorld.value(), tick).has_value());
        }

        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 20.0);
        for (auto& corner : state.corners)
        {
            corner.wheelSpeed = 20.0 / setup.corners.front().hardpoints.wheelRadius;
        }

        auto input = VehicleInput{};
        input.steering = steering;

        // **Speed is held, and without that this is not a skidpad.** Coasting, the car scrubs itself
        // down — 20 m/s to 5.9 by full lock — so what gets averaged is a spiral decaying to whatever
        // equilibrium cornering drag leaves, at a radius nobody chose. It also inverts the grip
        // response: a lower-grip tyre scrubs less, keeps more speed and therefore records *more*
        // lateral acceleration, which is how a grip cut appeared to make the car faster round a
        // corner. A real skidpad is driven, and the throttle it takes is part of the test.
        auto drive = std::array<double, cornerCount>{};

        auto lateral = 0.0;
        auto yaw = 0.0;
        auto speed = 0.0;
        auto frontSlip = 0.0;
        auto rearSlip = 0.0;
        auto samples = 0;

        for (auto step = 0; step < 2160; step++)
        {
            // Proportional on road speed, split across the driven axle and applied as wheel torque.
            // It consumes front grip exactly as a real front-drive car's does on a skidpad, which is
            // a property of the test being honest rather than a contamination of it.
            const auto held = 2000.0 * (20.0 - glm::length(state.chassis.linearVelocity));
            const auto perWheel =
                std::clamp(held, -6000.0, 6000.0) * setup.corners.front().hardpoints.wheelRadius / 2.0;
            drive = {perWheel, perWheel, 0.0, 0.0};

            const auto stepped = stepVehicle(setup, state, input, drive, plateWorld.value(), tick);
            REQUIRE(stepped.has_value());

            if (step >= 1800)
            {
                lateral += std::abs(stepped->telemetry.acceleration.x) / 9.80665;
                yaw += std::abs(stepped->telemetry.yawRate);
                speed += glm::length(state.chassis.linearVelocity);
                frontSlip += 0.5 * (std::abs(stepped->telemetry.wheels[0].slipAngle) +
                                    std::abs(stepped->telemetry.wheels[1].slipAngle));
                rearSlip += 0.5 * (std::abs(stepped->telemetry.wheels[2].slipAngle) +
                                   std::abs(stepped->telemetry.wheels[3].slipAngle));
                samples++;
            }
        }

        const auto n = static_cast<double>(samples);
        return std::array<double, 5>{lateral / n, yaw / n, speed / n, frontSlip / n, rearSlip / n};
    };

    for (const auto gripScale : {1.00, 0.85})
    {
        std::printf("\n=== steer sweep at 20 m/s, gripScale %.2f ===\n", gripScale);
        // **The consistency check that says whether this is a steady state at all.** In steady
        // cornering `a_y = v * yawRate`. If the measured lateral acceleration disagrees with that
        // product, the car is not tracking a circle — it is sliding, or spinning, or still slowing —
        // and every other column is describing a transient rather than a limit.
        std::printf("\n%8s %10s %9s %9s %10s %10s %11s %11s\n", "demand", "wheel deg", "lat g", "speed", "v*yaw g",
                    "radius m", "front slip", "rear slip");

        for (const auto steering : {0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00})
        {
            const auto [lateral, yaw, speed, front, rear] = hold(gripScale, steering);
            const auto radius = yaw > 1e-6 ? speed / yaw : 0.0;

            std::printf("%8.2f %10.2f %9.4f %9.2f %10.4f %10.1f %10.2f %11.2f\n", steering, steering * 27.4, lateral,
                        speed, speed * yaw / 9.80665, radius, front * 57.29578, rear * 57.29578);
        }
    }
}

TEST_CASE("is hold() the oscillator, or is the car", "[.launch]")
{
    // **Step 0 of the settling investigation, and it is a check on the instrument.** `hold()` was
    // rewritten an hour ago and three wrong calls this session came from reasoning off an unverified
    // fixture. Zeroing the integral gain was a partial check; the decisive one is to watch the
    // controller's own outputs at full rate. **If the inputs are flat and the lateral acceleration is
    // not, it is the car. If the inputs wobble, it is the loop.**
    const JoltGuard jolt;

    const auto plateWorld = PhysicsWorld::create(generateProvingGround(
                                                     []
                                                     {
                                                         auto d = ProvingGroundDescriptor{};
                                                         d.length = 1200.0;
                                                         d.width = 1200.0;
                                                         d.cellSize = 4.0;
                                                         d.features = {};
                                                         return d;
                                                     }())
                                                     .value());
    REQUIRE(plateWorld.has_value());

    auto setup = golfGtiMk7().value();

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, designHeight(setup), 400.0);
    for (auto step = 0; step < 1440; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, plateWorld.value(), tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 20.0);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = 20.0 / setup.corners.front().hardpoints.wheelRadius;
    }

    // The failing case: 0.15 of demand, well below the limit, where the car has no business sliding.
    auto input = VehicleInput{};
    input.steering = 0.15;

    auto integral = 0.0;

    std::printf("\n=== the last second of a ten-second hold at 0.15 demand, every tick ===\n");
    std::printf("  steering demand is CONSTANT at 0.15 by construction. The only input that varies is\n"
                "  the drive torque holding speed, so that is the one that could be driving anything.\n");
    std::printf("\n%8s %11s %10s %10s %10s %10s %10s\n", "t s", "drive Nm", "speed", "lat g", "yaw deg/s", "roll deg",
                "Fz FL N");

    for (auto step = 0; step < 3600; step++)
    {
        const auto error = 20.0 - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;
        const auto drive = std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0};

        const auto stepped = stepVehicle(setup, state, input, drive, plateWorld.value(), tick);
        REQUIRE(stepped.has_value());

        if (step >= 3240 && step % 12 == 0)
        {
            std::printf("%8.3f %11.1f %10.3f %10.4f %10.3f %10.3f %10.0f\n", static_cast<double>(step) * tick, perWheel,
                        glm::length(state.chassis.linearVelocity), stepped->telemetry.acceleration.x / 9.80665,
                        stepped->telemetry.yawRate * 57.29578, stepped->telemetry.roll * 57.29578,
                        stepped->telemetry.wheels[0].verticalLoad);
        }
    }
}

TEST_CASE("how repeatable the launch is", "[.launch]")
{
    // **The spread, and the honest answer has two halves.** The model is deterministic, so the same
    // inputs give the same run to the bit and a repeat measures nothing. What can vary is the state
    // the car launches *from*, so the settle is perturbed instead — which is the nearest thing this
    // model has to the run-to-run variation a real standing start has, and it bounds how much of any
    // difference against a human launch is noise.
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(straightGround()).value());
    REQUIRE(world.has_value());

    std::printf("\n=== repeats ===\n");
    std::printf("\n%12s %12s %12s\n", "settle ticks", "0-100 s", "0-60 mph s");

    auto times = std::vector<double>{};

    for (const auto settle : {1080, 1260, 1440, 1620, 1800, 2160})
    {
        const auto run = runLaunch(world.value(), settle, 15);
        const auto hundred = timeToReach(run, 100.0 / 3.6);

        times.push_back(hundred);
        std::printf("%12d %12.3f %12.3f\n", settle, hundred, timeToReach(run, 26.8224));
    }

    // Identical inputs twice, which is the determinism half and must be exact.
    const auto first = runLaunch(world.value(), 1440, 15);
    const auto again = runLaunch(world.value(), 1440, 15);
    REQUIRE(timeToReach(first, 100.0 / 3.6) == timeToReach(again, 100.0 / 3.6));

    std::sort(times.begin(), times.end());
    std::printf("\n  identical inputs reproduce exactly. Across settle states the spread is %.3f s\n"
                "  (%.3f to %.3f), which is the noise floor any human-launch comparison sits on.\n",
                times.back() - times.front(), times.front(), times.back());
}
