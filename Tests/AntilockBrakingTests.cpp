#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.input;
import raceengine.physics;

using raceengine::advanceYawMomentDelay;
using raceengine::AntilockChannelState;
using raceengine::AntilockSetup;
using raceengine::AssistSensors;
using raceengine::AssistSetup;
using raceengine::AssistState;
using raceengine::brakeCircuitPressures;
using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::Feature;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Assists;
using raceengine::ModulatorPhase;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::SurfaceMesh;
using raceengine::tearDownJolt;
using raceengine::updateAssists;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::YawMomentDelayState;

// The anti-lock system, measured on the car it is fitted to.
//
// **Every case here runs the assist layer explicitly.** The rest of the suite does not, and must not:
// the car underneath the electronics is what the other 433 tests validate, and a suite that quietly
// assisted every fixture could no longer see it. `golfGtiMk7Assists` therefore hands back a car with
// everything switched off, and each case below says what it is switching on.

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;
constexpr auto gravity = 9.80665;
constexpr auto degrees = 57.29577951308232;
constexpr auto hundred = 100.0 / 3.6;

// **The plate runs z from 0 to its length and x from -width/2 to +width/2.** Not symmetric, and a
// fixture that assumes it is starts the car in mid-air and measures a stop with no brakes in it.
// 600 m is room for a 100-0 that covers 40 m plus the 60 m of settling and rolling in front of it.
constexpr auto plateLength = 600.0;
constexpr auto plateWidth = 60.0;
constexpr auto startZ = 20.0;

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

// The pedal is off, so every caliper is at atmosphere. Named rather than written as `{}` at the
// call, because a zero pressure array means "no brakes" and that should be legible.
constexpr auto noBrakePressure = std::array<double, cornerCount>{};

// A flat plate with a stated grip on each side of the centreline. Split-mu is the case the whole
// system exists for, and it needs a surface that is genuinely different left and right rather than
// a car put on two wheels.
[[nodiscard]] SurfaceMesh gripPlate(const double leftGrip, const double rightGrip)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = plateLength;
    descriptor.width = plateWidth;
    descriptor.cellSize = 2.0;
    descriptor.features = std::vector<Feature>{};

    auto mesh = generateProvingGround(descriptor);
    REQUIRE(mesh.has_value());

    mesh->materials.resize(2);
    mesh->materials[0].gripMultiplier = leftGrip;
    mesh->materials[0].bumpiness = 0.0;
    mesh->materials[1].gripMultiplier = rightGrip;
    mesh->materials[1].bumpiness = 0.0;

    for (auto triangle = std::size_t{0}; triangle < mesh->triangleCount(); triangle++)
    {
        const auto centroid =
            (mesh->vertices[mesh->indices[triangle * 3 + 0]] + mesh->vertices[mesh->indices[triangle * 3 + 1]] +
             mesh->vertices[mesh->indices[triangle * 3 + 2]]) /
            3.0;

        mesh->surfaces[triangle] = centroid.x < 0.0 ? std::uint32_t{0} : std::uint32_t{1};
    }

    return mesh.value();
}

struct StopResult
{
    double distance = 0.0;
    double time = 0.0;
    double lateralTravel = 0.0;
    double finalYaw = 0.0;
    double peakYawRate = 0.0;
    double deceleration = 0.0;
    double meanTrueSlip = 0.0;
    double peakFrontSlip = 0.0;
    std::array<std::uint32_t, cornerCount> cycles{};
    // Turning points in the *pressure* itself, which is the cycle a driver feels and the one the
    // published frequencies are about. The state machine's dump counter is a different and faster
    // event: it re-enters a dump several times inside one pressure excursion.
    std::array<std::uint32_t, cornerCount> pressurePeaks{};
    // Seconds each channel spent anywhere other than passive, which is what a cycling *rate* has to
    // be divided by: cycles over the whole stop reads low whenever the system spent part of it doing
    // nothing, and on this car the front channel spends most of a dry stop doing exactly that.
    std::array<double, cornerCount> engagedTime{};
    bool stopped = false;
    bool onPlate = true;
    bool grounded = true;

    // How long the yaw moment build-up delay was engaged for, seconds, and the lowest ceiling it
    // held the high front channel at. **Zero on every shipped car**, because the feature is off —
    // and reported at all because a feature that measures as inert has to say whether it never
    // engaged or engaged and did nothing, and those are different bugs.
    double yawDelayTime = 0.0;
    double yawDelayLowest = 0.0;
    double peakLateral = 0.0;

    // When each channel first let pressure go, seconds after the pedal moved. The gap between the
    // two fronts on a split surface is the whole window a yaw moment build-up delay has to work in.
    std::array<double, cornerCount> firstCycle{};
};

void settle(const VehicleSetup& setup, VehicleState& state, const PhysicsWorld& world, const double speed)
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

// One stop, in neutral, with the pedal held. Nothing drives the wheels, so what is measured is the
// brake system and the tyre.
// `pedalRamp` is how long the pedal takes to reach `pedal`, seconds. **Zero — a step — is the
// default and every case written before 2026-08-30 uses it**, and with it the arithmetic below is
// the arithmetic it always was, to the bit. A ramp exists because a step pedal is not a driver: it
// puts every wheel on the car past its locking level inside one physics tick, which is fine for
// measuring a stop and is fatal to any mechanism whose trigger is *one* wheel locking before
// another.
[[nodiscard]] StopResult stop(const VehicleSetup& setup, const PhysicsWorld& world, AssistSetup assists,
                              const double entry, const double pedal, const double steering = 0.0,
                              const double pedalRamp = 0.0)
{
    auto state = VehicleState{};
    settle(setup, state, world, entry);

    auto assistState = AssistState{};
    auto lastStep = VehicleStep{};
    auto result = StopResult{};

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

        return sensors;
    };

    // Rolling before the pedal moves, so the tone rings have produced readings and the reference
    // speed estimator is not asked to start up and brake in the same instant.
    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped = stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    // --- preconditions. Nothing below is worth quoting until these hold. ---
    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(lastStep.telemetry.wheels[index].inContact);
    }
    REQUIRE(std::abs(state.chassis.linearVelocity.z - entry) < 0.5);
    REQUIRE(std::abs(state.chassis.position.y - designHeight) < 0.1);
    REQUIRE(std::abs(state.chassis.position.x) < 0.05);

    const auto start = state.chassis.position;
    const auto entrySpeed = state.chassis.linearVelocity.z;

    auto input = VehicleInput{};
    input.brake = pedal;
    input.steering = steering;

    auto samples = 0;
    auto rising = std::array<bool, cornerCount>{};
    auto previousPressure = std::array<double, cornerCount>{};

    for (auto step = 0; step < 360 * 30; step++)
    {
        const auto applied = pedalRamp > 0.0 ? pedal * std::min(1.0, result.time / pedalRamp) : pedal;
        input.brake = applied;

        const auto command = updateAssists(assists, assistState, sense(), {.brake = applied, .throttle = 0.0},
                                           brakeCircuitPressures(setup, applied), tick);

        const auto stepped = stepVehicle(setup, state, input, noDriveTorque, world, tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        result.time += tick;
        samples++;
        result.peakYawRate = std::max(result.peakYawRate, std::abs(lastStep.telemetry.yawRate));
        result.peakLateral = std::max(result.peakLateral, std::abs(lastStep.telemetry.acceleration.x));

        if (assistState.antilock.yawDelay.engaged)
        {
            result.yawDelayTime += tick;
            result.yawDelayLowest = result.yawDelayTime <= tick
                                        ? assistState.antilock.yawDelay.ceiling
                                        : std::min(result.yawDelayLowest, assistState.antilock.yawDelay.ceiling);
        }

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            result.cycles[index] = command.channels.antilockCycles[index];

            if (result.firstCycle[index] == 0.0 && command.channels.antilockCycles[index] > 0)
            {
                result.firstCycle[index] = result.time;
            }

            result.meanTrueSlip += std::abs(lastStep.telemetry.wheels[index].slipRatio) / cornerCount;

            if (index < 2)
            {
                result.peakFrontSlip =
                    std::max(result.peakFrontSlip, std::abs(lastStep.telemetry.wheels[index].slipRatio));
            }

            if (command.channels.antilockPhase[index] != ModulatorPhase::Passive)
            {
                result.engagedTime[index] += tick;
            }

            // A hundredth of full system pressure of hysteresis, so numerical wobble is not counted
            // as a cycle. At 100 bar that is a bar, which is below anything a foot could feel — and
            // it is not what sets the count: re-run at 0.002 the answer is the same to the cycle, so
            // the frequency below is the pressure's and not the threshold's.
            const auto pressure = command.channels.pressure[index];
            if (pressure > previousPressure[index] + 0.01)
            {
                rising[index] = true;
                previousPressure[index] = pressure;
            }
            else if (pressure < previousPressure[index] - 0.01)
            {
                if (rising[index])
                {
                    result.pressurePeaks[index]++;
                    rising[index] = false;
                }
                previousPressure[index] = pressure;
            }

            result.grounded = result.grounded && lastStep.telemetry.wheels[index].inContact;
        }

        if (std::abs(state.chassis.position.x) > 0.5 * plateWidth - 2.0 || state.chassis.position.z > plateLength - 5.0)
        {
            result.onPlate = false;
        }

        if (state.chassis.linearVelocity.z <= 0.0)
        {
            result.stopped = true;
            break;
        }
    }

    result.distance = state.chassis.position.z - start.z;
    result.lateralTravel = state.chassis.position.x - start.x;
    result.finalYaw = lastStep.telemetry.yaw;
    result.deceleration = result.time > 0.0 ? entrySpeed / result.time : 0.0;
    result.meanTrueSlip /= samples > 0 ? static_cast<double>(samples) : 1.0;

    return result;
}

[[nodiscard]] AssistSetup withAntilock(const VehicleSetup& setup)
{
    auto assists = golfGtiMk7Assists(setup);
    assists.antilock.enabled = true;

    return assists;
}

// **Criterion 5's two cases compare ensembles, not single stops, since 2026-08-29.** A full-pedal
// stop on mu 0.35 is chaotic — the assisted trajectory especially: across a ±0.7% entry-speed band
// its peak yaw rate spans 0.111 to 0.330 and its lateral travel 2.1 to 13.4 m, mode-hopping between
// keeping its steering and washing out. Eight documented flips on fraction-of-a-per-cent plant
// changes (`docs/known-red.md`) were single rolls of that dice being divided by each other. The
// fifteen entry speeds below are deterministic, inside the fixture's own settling tolerance, and
// each member is the same stop the criteria always measured; the criteria assert MEDIANS, which the
// isolation A/B (`docs/known-red.md`, 2026-08-29) showed move by 2-8% under a plant change that
// swung single rolls by over 70%. `[.abs-ensemble]` prints the member-level table.
[[nodiscard]] std::vector<StopResult> steeringEnsemble(const VehicleSetup& setup, const PhysicsWorld& world,
                                                       const bool assisted, const double steering)
{
    auto runs = std::vector<StopResult>{};
    runs.reserve(15);

    for (auto k = -7; k <= 7; k++)
    {
        const auto entry = hundred * (1.0 + 0.001 * static_cast<double>(k));
        const auto assists = assisted ? withAntilock(setup) : golfGtiMk7Assists(setup);
        runs.push_back(stop(setup, world, assists, entry, 1.0, steering));
    }

    return runs;
}

// The middle member of an odd-sized ensemble. For the median to change cluster, half the members
// have to change cluster with it — a distribution-level result rather than a re-roll.
template <typename Projection> [[nodiscard]] double medianOf(const std::vector<StopResult>& runs, Projection projection)
{
    auto values = std::vector<double>{};
    values.reserve(runs.size());

    for (const auto& run : runs)
    {
        values.push_back(projection(run));
    }

    std::sort(values.begin(), values.end());

    return values[values.size() / 2];
}

} // namespace

TEST_CASE("this car's brakes exceed its grip, so the pedal has an optimum short of the floor",
          "[assists][antilock][braking]")
{
    // **The measurement every other dry case has to be read against, and it is the opposite of what
    // it was.** Until 2026-08-23 this car was *brake-torque limited*: `brakes.ini`'s 4200 N.m capped
    // it at 0.997 g, the front wheels could not be locked at any pedal position, and full pedal was
    // therefore the optimum by accident. That figure was taken back to source, found not to be a
    // measured number, and replaced — the derivation is in `PublishedCarsImpl`.
    //
    // What the correction buys is not a shorter stop. It is a car that behaves like a car: past the
    // pedal that saturates the tyre, more pressure makes the stop **longer**, which is the curve every
    // real brake system has and the reason an anti-lock system exists at all.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    // **Neither `MAX_TORQUE` nor `FRONT_SHARE` is read any more** (2026-08-23,
    // `docs/brake-model-brief.md`): the peak is what this car's calipers, pads, discs and hydraulics
    // make, and the split is what its rear proportioning valve makes of that. 5600 was itself a
    // chosen number and is gone with the 4200 it corrected. What is asserted here is the *property*
    // the correction was made for, not the figure — the figure belongs to `GolfGtiTests`.
    const auto total = 2.0 * (setup->corners[0].brakeTorque + setup->corners[2].brakeTorque);

    CAPTURE(total, setup->corners[0].brakeTorque, setup->corners[2].brakeTorque);

    // Above the 4624 N.m lower bound below which the fronts cannot be locked at any pedal position,
    // which is the whole reason this car's brake data was taken back to source.
    REQUIRE(total > 4624.0);

    // And the front axle carries more than the mod's stated 0.75 at a fully applied pedal, because
    // the valve limits the rear where the load has left it.
    REQUIRE(2.0 * setup->corners[0].brakeTorque / total > 0.75);

    const auto sprung = raceengine::computeMassProperties(setup->sprung);
    REQUIRE(sprung.has_value());
    const auto mass = sprung->mass + setup->unsprungMass();

    // What the pedal could ask for if grip were unlimited. It has to be *above* what the tyre can
    // deliver, or the brakes are the limit and the whole curve below cannot exist.
    const auto brakeBound = total / tyreRadius / mass;

    CAPTURE(mass, brakeBound / gravity);
    REQUIRE(brakeBound / gravity > 1.2);

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    SECTION("the deceleration turns over, which a brake-limited car cannot do")
    {
        auto best = 1e9;
        auto bestPedal = 0.0;
        auto atFloor = 0.0;

        // **Swept from a tenth of the pedal, and that range is not incidental.** A sweep ranged for
        // one brake torque measures every other candidate off its optimum: at the mod's 4200 N.m the
        // optimum was exactly at the floor, so a sweep starting at 0.30 found it; with brakes derived
        // from the hardware it is near 0.40 and a sweep starting at 0.55 would report a car with no
        // optimum at all. `docs/brake-model-brief.md` records both wrong conclusions that cost.
        for (auto step = 2; step <= 20; step++)
        {
            const auto pedal = 0.05 * static_cast<double>(step);
            const auto run = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, pedal);

            REQUIRE(run.stopped);
            REQUIRE(run.onPlate);

            // **`grounded` was asserted here and is not any more**, and this is not a loosened bound:
            // it has moved to a case of its own, `the car keeps all four wheels on the ground through
            // a hard stop`, which is `[!shouldfail]` with its cause written down. It began failing
            // here on 2026-08-23, when the load-sensitivity split took the rear axle's transient load
            // through zero, and leaving it in this sweep would have stopped the optimum-short-of-the-
            // floor claim below from gating anything at all.
            CAPTURE(run.grounded);

            if (run.distance < best)
            {
                best = run.distance;
                bestPedal = pedal;
            }

            if (pedal >= 1.0)
            {
                atFloor = run.distance;
            }
        }

        CAPTURE(best, bestPedal, atFloor);

        // The optimum is short of the floor. On the old data it was exactly *at* the floor, because
        // the brakes ran out before the tyre did.
        REQUIRE(bestPedal < 0.95);

        // And going past it costs real distance rather than a rounding error.
        REQUIRE(atFloor > best);
        REQUIRE((atFloor - best) / best > 0.02);
    }

    SECTION("and the front wheels can now be locked, which is what an anti-lock system needs")
    {
        // The whole point of the correction. At 4200 N.m the front axle had 3150 against the 3468 it
        // needs to lock at this car's braking limit, so no pedal position could reach it and every
        // dry anti-lock measurement was vacuous.
        const auto run = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, 1.0);

        REQUIRE(run.stopped);
        CAPTURE(run.peakFrontSlip);

        // Well past the tyre's longitudinal peak, which is where locking begins.
        REQUIRE(run.peakFrontSlip > 0.5);
    }
}

TEST_CASE("the imported car stops from 100 km/h in the distance the real one does",
          "[assists][antilock][braking][!shouldfail]")
{
    // The published reference is auto motor und sport's Supertest: **35.5 m from 100 km/h, kalt**
    // (1.108 g mean), verified at source on 2026-08-25. Auto Bild Sportscars' 34.6/35.1 m, which
    // this case was originally written against, did not survive that verification. The car's ABS
    // stop is 41.74 m — 17.6% long — and the gap's owners are measured, not guessed
    // (`docs/braking-chain-brief.md`, the utilisation ledger): ~3.5 m of controller tracking gated
    // on the estimator-and-controller co-design, <= 1.8 m of valve (real hardware), and ~1.4 m of
    // capability whose candidates are the held chassis items. The tyre is exonerated by measurement
    // (`docs/tyre-peak-to-tail-brief.md`). The bound below is the verified figure plus margin.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto run = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

    REQUIRE(run.stopped);
    CAPTURE(run.distance, run.deceleration / gravity);

    REQUIRE(run.distance < 36.0);
}

TEST_CASE("anti-lock braking stays inside the band a sensor-blind controller can reach on dry tarmac",
          "[assists][antilock][braking]")
{
    // **Criterion 2, restated 2026-08-24 late on Dominic's ruling that its first half was
    // erroneous.** It used to assert the anti-lock stop *never beats* a perfect constant-pedal
    // driver, on the premise that beating one means reading something the controller has no sensor
    // for. The premise fails on this car: its optimum pedal is set by the **rear axle locking
    // first**, so a single constant pressure is wrong at every other moment of the stop, and a
    // per-channel modulating system legitimately beats any value of it. The transient decomposition
    // proved it with the erroneous premise's own instrument (`docs/braking-chain-brief.md`,
    // Progress): a controller with perfect knowledge forced through the modulator's own valve rates
    // stops in 38.75 m — 3.4% *better* than the fine-swept driver — without seeing anything a
    // sensor could not in principle deliver.
    //
    // **What the criterion was actually built to catch survives, as the band's far side.** Measured
    // on this car: any controller working through this valve lands no better than −3.4% against the
    // perfect driver, while perfect per-wheel knowledge with a free actuator lands at −8%. A
    // two-sided 6% band therefore never blocks a legitimate controller and still fires on a
    // model-peeker — beating the driver by more than the valve allows IS the signature the old
    // assertion was groping for. **The −6% side is justified by that measurement and must be
    // re-derived if the modulator's gradients ever change**, because the legitimate floor moves
    // with the valve.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    // The best a driver can do without cycling the pedal: the shortest constant-pressure stop there
    // is. **Swept across the whole pedal**, because where the optimum sits is a property of the brake
    // torque this fixture is measuring against — it was at the floor when the brakes could not lock a
    // wheel and it is near 0.40 now that they can. Ranged for one candidate, this comparison silently
    // becomes "ABS against a driver braking badly", and it read that way for exactly one build:
    // the assisted stop came out *shorter* than the sweep's best and tripped the never-beats-a-driver
    // assertion, which is the assertion doing its job.
    auto best = 1e9;
    auto bestPedal = 0.0;

    // **Finely where the optimum is, and that is not fastidiousness** (2026-08-24). At 0.05 steps this
    // sweep put its best at 0.35 and the assisted stop came out 2.4% *shorter*, tripping the assertion
    // below — for the second time, and for the same reason the paragraph above already records. The
    // optimum pedal moves with everything: brake torque, grip, and now the car's corrected mass and
    // gearing. A sweep whose resolution is coarser than the difference being argued about measures the
    // sweep rather than the controller.
    for (const auto pedal : {0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38,
                             0.39, 0.40, 0.41, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.70, 0.80, 1.0})
    {
        const auto run = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, pedal);
        REQUIRE(run.stopped);
        REQUIRE(run.onPlate);

        if (run.distance < best)
        {
            best = run.distance;
            bestPedal = pedal;
        }
    }

    const auto assisted = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

    REQUIRE(assisted.stopped);
    REQUIRE(assisted.onPlate);
    // Reported rather than required, for the reason given in the sweep above: all four wheels staying
    // down through a hard stop is its own `[!shouldfail]` case now.
    CAPTURE(assisted.grounded);

    const auto penalty = (assisted.distance - best) / best;

    CAPTURE(best, bestPedal, assisted.distance, penalty);

    // The cheating watch: better than the driver is allowed, better than the valve permits is not.
    // Perfect knowledge through this valve measures −3.4%; a free actuator measures −8%; a controller
    // past this bound is reading something its sensors cannot deliver.
    REQUIRE(penalty > -0.06);

    // **How much it costs the other way has its own case below**, so the two halves of this
    // criterion cannot fail together: this side is a statement about whether the controller is
    // cheating, the other about how well it hunts.
}

TEST_CASE("and the anti-lock stop costs only a few percent against that driver", "[assists][antilock][braking]")
{
    // **Criterion 2's second half — the hunting bound — and the half the 2026-08-24 restatement left
    // untouched**: however legitimate beating the driver is, *losing* to one by more than a few
    // percent still means the controller hunts badly, and that statement survives the premise the
    // first half lost.
    //
    // **Criterion 2's second half, and it holds again as of 2026-08-23 — closed by something that has
    // nothing to do with the anti-lock system.** It was opened the same day at 8.7% against a bound of
    // 6%, when brakes derived from this car's own calipers put 2.4 times a front wheel's lock pressure
    // at a fully applied pedal where `brakes.ini`'s figure put 1.2 times, and the modulator's hunting
    // range widened with it.
    //
    // **What closed it was the load-sensitivity split**, and *both* sides of the ratio moved, which is
    // exactly the kind of thing a ratio hides. The penalty went 8.71% to 4.92% because the constant-
    // pedal reference got **worse** (39.84 m to 40.32) and the assisted stop got **better** (43.31 m
    // to 42.30) at the same time.
    //
    // One mechanism does both. Braking unloads the rear axle to well below the tyre's nominal load,
    // and a flatter longitudinal exponent gives *less* grip below nominal, not more — so the rear
    // starts locking at the constant pedal a driver would hold, and the driver has no way to know. The
    // assisted stop cycles its rear channel and never sits on the lock, so it keeps the front axle's
    // gain — which is where the flatter exponent pays, because braking puts the fronts well *above*
    // nominal load. A controller looks better the moment the thing it is compared against loses the
    // one advantage it had.
    //
    // Left as a plain green case rather than deleted: it is the one thing watching for a controller
    // that has learned to see the model, and it should fire again if either side of the comparison
    // moves. The account of both directions is in `docs/known-red.md`.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    auto best = 1e9;

    for (const auto pedal : {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0})
    {
        const auto run = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, pedal);
        REQUIRE(run.stopped);
        REQUIRE(run.onPlate);

        best = std::min(best, run.distance);
    }

    const auto assisted = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);
    REQUIRE(assisted.stopped);

    const auto penalty = (assisted.distance - best) / best;

    CAPTURE(best, assisted.distance, penalty);
    REQUIRE(penalty < 0.06);
}

TEST_CASE("anti-lock braking is worth something on a uniformly slippery surface",
          "[assists][antilock][braking][!shouldfail]")
{
    // **Criterion 3, and it does not hold. Measured, not assumed, and the cause is measured too.**
    //
    // On a mu 0.35 plate the anti-lock stop is 130.9 m against 131.3 m with the wheels locked — 0.3%,
    // where the tyre says there is 27% to be had (peak 1783 N at kappa 0.03 against 1299 N locked, at
    // 5000 N and this surface). Two things account for it and neither is the controller's tuning:
    //
    //  1. **This tyre's peak moves with the surface.** The Magic Formula here scales friction by the
    //     surface without scaling stiffness, so the longitudinal peak sits at kappa 0.09 on dry
    //     tarmac and at kappa 0.03 on this one. The published anti-lock control range is 8% to 30%
    //     slip, so a controller calibrated to it works three times past the peak on anything
    //     slippery. Fixing that means changing the tyre model, which is out of scope.
    //
    //  2. **The reference speed is biased low by about 19%**, because every wheel on the car is in
    //     control at once and the fastest of them is never at road speed. Measured over a low-mu
    //     stop: the ECU believed 10.5% slip while the tyres were at 30.9%. That is the honest
    //     failure of a wheel-speed-only estimator and is the same reason production systems that
    //     have to work on ice add a longitudinal accelerometer — which is a sensor this brief does
    //     not give the controller, and adding one is a design decision rather than a fix.
    //
    // **It is not the hydraulics.** Swept across a 4x range of dump rate and an 8.7x range of
    // re-apply rate, the low-mu result never moves outside +1.4% to -2.3%. `[.assist-probe]` has the
    // table.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    const auto locked = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, 1.0);
    const auto assisted = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

    REQUIRE(locked.stopped);
    REQUIRE(assisted.stopped);

    // The wheels really do lock without it, which is the precondition for the comparison meaning
    // anything at all.
    CAPTURE(locked.meanTrueSlip, assisted.meanTrueSlip, locked.distance, assisted.distance);
    REQUIRE(locked.meanTrueSlip > 0.8);
    REQUIRE(assisted.meanTrueSlip < locked.meanTrueSlip);

    // Materially shorter. It is not.
    REQUIRE(assisted.distance < 0.9 * locked.distance);
}

TEST_CASE("the anti-lock system keeps the car straight on a split surface", "[assists][antilock][braking][splitmu]")
{
    // **Criterion 4, and the case the whole system exists for.** Tarmac under the left wheels, a
    // mu 0.35 surface under the right. Without it the car makes a braking imbalance across both
    // axles, the rears lock first, and it spins. With it the rear axle is select-low — both rear
    // wheels held to the pressure the low-grip one can take — so the axle cannot develop an
    // imbalance, and the fronts are what is left.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(world.has_value());

    const auto plain = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, 1.0);
    const auto assisted = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

    REQUIRE(plain.stopped);
    REQUIRE(assisted.stopped);
    REQUIRE(assisted.onPlate);

    CAPTURE(plain.distance, assisted.distance, plain.finalYaw * degrees, assisted.finalYaw * degrees);

    SECTION("without it the car ends up pointing somewhere else entirely")
    {
        // Seventy degrees off its entry heading by the time it stops, which is a spin and not a stop.
        // A quarter turn is the threshold: past that the car has lost the ability to be steered out
        // of it, because the front tyres are no longer pointing anywhere near the direction of travel.
        REQUIRE(std::abs(plain.finalYaw) > 45.0 / degrees);
    }

    SECTION("with it the car is better off than without it, which is the reason it is fitted")
    {
        // The comparison this case is really about — the system either helps on a split surface or it
        // has no reason to exist. **How much better off has its own case now**, because on 2026-08-23
        // it stopped holding: the margin was 0.640 of the unassisted heading error and is 0.839.
        //
        // No steering correction is applied anywhere in this fixture, which is the harshest possible
        // reading of "controllable" — a real driver would be holding it.
        REQUIRE(std::abs(assisted.finalYaw) < std::abs(plain.finalYaw));
    }

    SECTION("and it stops substantially shorter, because the high-grip side is still braking")
    {
        REQUIRE(assisted.distance < 0.85 * plain.distance);
    }
}

TEST_CASE("and on a split surface it stays inside a quarter turn", "[assists][antilock][braking][splitmu]")
{
    // **Closed 2026-08-24 evening by the slip-aware recovery law, at 19.3 degrees against the
    // 45 degree bound — and this closure is the controller's, not the car's.** The two before it were
    // the car's (mass, then droop travel re-opening it), which is why the note below survives. The
    // mechanism: the old recovery law left every wheel in anti-lock control at about three times its
    // peak slip — `docs/braking-chain-brief.md`'s instrument measured it — and a tyre dragged that
    // far past its longitudinal peak has little lateral force left to resist a yaw with. Held near
    // the peak instead, the rear axle keeps its cornering stiffness and the same yaw moment builds a
    // fifth of the heading error: 73.96 degrees to 19.27, with the unassisted run at 96.3.
    //
    // **Yaw-moment build-up limitation is still absent.** The account below stands: nothing ramps the
    // high-grip front's pressure, so what bounds the heading error now is the rear axle's restored
    // lateral authority rather than any moderation of the moment. A car with less rear axle — or a
    // surface split the other way under a crest — has less of this margin.
    //
    // **Re-opened 2026-08-24 at 73.96 degrees, hours after being closed, and the closure note called
    // it.** That note read: *"the margin here is the car's inertia rather than the controller doing
    // anything about it. If a heavier car is what keeps this inside a quarter turn, a lighter one will
    // put it back outside."* What put it back outside was not a lighter car but a rear axle that now
    // stays on the road.
    //
    // Giving the rear its droop travel back — the stop was a placed 20 mm binding at 23 mm of
    // extension, see `docs/known-red.md` — means the rear wheels follow the surface instead of
    // hanging in the air. On a split-mu surface that is *more* asymmetry reaching the road, not less,
    // because the high-grip side's rear tyre is now working. **The correctness fix un-masked this.**
    //
    // The mechanism below is still the mechanism, and it is still absent.
    //
    // **The history, so nobody re-derives it**: red at 53.7 degrees from 2026-08-23, closed by the
    // mass correction on 2026-08-24, re-opened the same day by the droop correction at 73.96.
    //
    // Nothing about the anti-lock unit changed. What changed is that the car weighs what it weighs:
    // the manufacturer's tare figure plus a driver is 1452 kg where the mod's `TOTALMASS` said 1348,
    // so the model was 7.2% light (`docs/engine-curve-validation-brief.md`). A split surface makes a
    // yaw *moment* out of the difference between the two sides, and that moment is set by the tyres;
    // the yaw *inertia* resisting it scales with the mass that was missing. The heading error a given
    // moment produces in a given time therefore falls, and it fell far enough.
    //
    // **That is a real closure and it is also a thin one to lean on.** The mechanism below is still
    // absent — this system has no yaw-moment build-up limitation — so the margin here is the car's
    // inertia rather than the controller doing anything about it. If a heavier car is what keeps this
    // inside a quarter turn, a lighter one will put it back outside, and the account below is what to
    // read when it does.
    //
    // A split surface makes a yaw moment whatever the electronics do, and this system has **no
    // yaw-moment build-up limitation** — the production feature that deliberately ramps the high-grip
    // front wheel's pressure so the moment arrives slowly enough for a driver to catch. Deriving this
    // car's brakes from its calipers is what made that absence matter: the assisted stop went from
    // 74.4 m to 54.8 m on this surface, and every metre of that came from braking the high-grip side
    // harder, which is the same thing as a larger yaw moment.
    //
    // History, so nobody re-derives it: the bound was 30 degrees when this measured 19.1, was moved to
    // a quarter turn when the 5600 N.m correction took it to 31.3, and the hardware derivation has now
    // taken it to 53.7 with the unassisted run at 84.0. The bound has not moved again and will not:
    // what closes this is the missing controller, not a number.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(world.has_value());

    const auto assisted = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

    REQUIRE(assisted.stopped);
    CAPTURE(assisted.finalYaw * degrees);

    // A quarter turn: past that the car has lost the ability to be steered out of it, because the
    // front tyres are no longer pointing anywhere near the direction of travel.
    REQUIRE(std::abs(assisted.finalYaw) < 45.0 / degrees);
}

TEST_CASE("the car still steers under full braking with the anti-lock system on", "[assists][antilock][braking]")
{
    // **Criterion 5.** A locked tyre makes its friction along the direction it is sliding and has
    // none left to point the car with, so threshold braking at the limit costs the steering. Held
    // near the peak the tyre still has a slip angle to work with.
    //
    // Measured on the mu 0.35 plate rather than on dry, and that is a consequence of the first case
    // in this file: on dry the front wheels never lock, so there is no steering to lose and the
    // comparison would be vacuous. Choosing a surface where the failure is reachable is the whole
    // point of testing the complement.
    //
    // **Ensemble medians since 2026-08-29, and the bounds did not move.** This case compared one
    // assisted roll against one locked roll for a year of plant changes and flipped eight times
    // (`docs/known-red.md`); the ensemble helper's comment carries the measured distributions. The
    // locked arm turned out to be the smooth one — 0.1302 to 0.1306 of peak yaw across the whole
    // band — and the ASSISTED arm the chaotic one, so a flip was almost always the assisted dice
    // re-rolling, not the baseline. On medians the verdict is stable and currently red: 0.2217
    // against a needed 2 x 0.1304, with ten of fifteen members keeping their steering and five
    // washing out. What closes it is the controller keeping its steering advantage on low grip —
    // the recovery law's own territory — and never a loosened bound.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    // A third of lock, held from the moment the pedal goes down.
    const auto steering = 0.35;

    const auto plain = steeringEnsemble(setup.value(), world.value(), false, steering);
    const auto assisted = steeringEnsemble(setup.value(), world.value(), true, steering);

    for (const auto& run : plain)
    {
        REQUIRE(run.stopped);

        // The wheels lock without it — the precondition that makes this a test of steering rather
        // than of two identical runs, and it has to hold for every member or the medians compare
        // a mixture.
        REQUIRE(run.meanTrueSlip > 0.8);
    }

    for (const auto& run : assisted)
    {
        REQUIRE(run.stopped);
        REQUIRE(run.onPlate);
    }

    const auto plainYaw = medianOf(plain, [](const StopResult& run) { return run.peakYawRate; });
    const auto assistedYaw = medianOf(assisted, [](const StopResult& run) { return run.peakYawRate; });
    const auto plainTravel = medianOf(plain, [](const StopResult& run) { return run.lateralTravel; });
    const auto assistedTravel = medianOf(assisted, [](const StopResult& run) { return run.lateralTravel; });

    CAPTURE(plainYaw, assistedYaw, plainTravel, assistedTravel);

    // With the system on the car answers the wheel: more than twice the yaw rate over a stop of the
    // same length. Measured at 5.2 times on 2026-08-23's car; the 2026-08-29 median is 1.70 times.
    REQUIRE(assistedYaw > 2.0 * plainYaw);

    // And it went where it was pointed rather than merely somewhere: a third of lock to the right is
    // a right turn.
    REQUIRE(assistedTravel * plainTravel > 0.0);

    // **Whether the car goes where it points is the other half and it has its own case below**,
    // because that is what stopped holding: the yaw rate is five times the locked run's and the
    // lateral displacement is only 1.3 times it, which is a car rotating more than it is travelling.
    REQUIRE(std::abs(assistedTravel) > std::abs(plainTravel));
}

TEST_CASE("and the steering it keeps puts the car somewhere else, not just at another angle",
          "[assists][antilock][braking]")
{
    // **Criterion 5's second half, closed 2026-08-24 evening by the slip-aware recovery law: 4.14
    // times the locked run's lateral displacement against the bound of 3**, at 2.2 times its yaw rate
    // — which is the right shape as well as the right number, because 5.2 times the yaw rate for 1.3
    // times the displacement was a car rotating about its own centre while it slid, and this is a car
    // going where it is pointed.
    //
    // The paragraph that used to close this comment asked for "a two-stage re-apply, with a memory of
    // the pressure it was holding when the wheel last departed" — and that is very nearly what closed
    // it. `AntilockChannelState::departurePressure` is that memory and the re-apply is two-staged on
    // it; what the prediction missed is that the memory alone is half the law, and the recovery
    // phases also had to stop reading "not departing" as "recovered" (`AntilockSetup::
    // slipAwareRecovery`, and the account in `docs/braking-chain-brief.md`'s Progress).
    //
    // The distinction the bound exists for is a real one. Yaw rate alone cannot tell a car that is
    // being steered from a car that is rotating about its own centre while it slides, which is why
    // both halves of criterion 5 exist and are asserted separately.
    //
    // **Ensemble medians since 2026-08-29, the same change as the case above and the same bounds.**
    // The single assisted roll spanned 2.1 to 13.4 m of travel across a ±0.7% entry-speed band —
    // this criterion's flips were that spread being sampled once. The median is 8.11 m against a
    // needed 3 x 2.999: stable, red, and 10% under the bound rather than a coin toss across it.
    // Under the flip-seven plant A/B the median moved 8%, so a future green here needs the
    // assisted DISTRIBUTION to move, not one member.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    const auto steering = 0.35;

    const auto plain = steeringEnsemble(setup.value(), world.value(), false, steering);
    const auto assisted = steeringEnsemble(setup.value(), world.value(), true, steering);

    for (const auto& run : plain)
    {
        REQUIRE(run.stopped);
    }

    for (const auto& run : assisted)
    {
        REQUIRE(run.stopped);
    }

    const auto plainTravel = medianOf(plain, [](const StopResult& run) { return std::abs(run.lateralTravel); });
    const auto assistedTravel = medianOf(assisted, [](const StopResult& run) { return std::abs(run.lateralTravel); });

    CAPTURE(plainTravel, assistedTravel);
    REQUIRE(assistedTravel > 3.0 * plainTravel);
}

TEST_CASE("the cycling frequency falls out of the hydraulics rather than being prescribed",
          "[assists][antilock][braking]")
{
    // **Criterion 6.** Nothing in `AntilockSetup` states a frequency. What is stated is how fast the
    // modulator can dump and re-apply pressure, and the frequency is that working against how fast
    // the wheel re-accelerates once it is let go. Production systems are quoted at 4 to 20 Hz — the
    // article citing SAE J2538 puts a typical unit at 8 to 15 per wheel per second — and the band
    // asserted here is the wider one, because a number that emerged has no business being pinned
    // tightly.
    //
    // Divided by the time the channel spent *engaged*, not by the whole stop: on this car the front
    // channel is passive for most of a dry stop, and dividing by the stop would report a rate the
    // system never ran at.
    //
    // **This does not hold, and it is `[!shouldfail]` rather than a loosened bound — but it has
    // changed sides** (2026-08-23, `docs/brake-model-brief.md`).
    //
    // It used to fail *below* the band: 3.2 Hz on a mu 0.60 front channel and 3.7 to 4.8 elsewhere,
    // because the modulator's rates were fractions of full system pressure per second and a re-apply
    // of 3 per second cannot buy back a whole range faster than about 4 Hz. Those rates are now
    // gradients in bar per second — a property of the valves rather than of whatever calipers are on
    // the other end of the pipe — and the arithmetic that capped it is gone.
    //
    // What fails now is the **rear** channel on dry tarmac at **24.2 Hz**, above the band, while the
    // front channel on the same stop is inside it. The cause is the other half of the same change:
    // the rear circuit's proportioning valve holds the rear to 54 bar where the front sees 125, so the
    // same gradient traverses the rear's whole range in less than half the time. That is a real
    // property of a valved rear circuit and not a control-law fault, and it is why production units
    // meter the rear channel differently from the front.
    //
    // What closes it is a modulator whose gradients are per *channel* — which is what a real unit's
    // rear outlet valve is — with a source for the rear one. Picking a number because it makes a
    // frequency come out is the prescribing this criterion exists to forbid.
    //
    // **It moved again on 2026-08-24 evening and by the right mechanism** — the slip-aware recovery
    // law took the dry rear from 24.2 Hz to inside the band, because a channel that holds while a
    // wheel climbs back and probes gently near the peak cycles at the wheel's pace rather than at its
    // own valve's. What failed after that was the rear channel on the split surface at 3.48 Hz, below
    // the band — select-low means that channel follows the mu 0.35 wheel, whose road can only slowly
    // spin it back up, and the law now waits for that.
    //
    // **CLOSED 2026-08-27, and the bound was never moved.** Every surface and every channel now lands
    // inside 4 to 20 Hz. What closed it is **damper seal friction**, isolated by running this case
    // with that one number zeroed and watching it fail again exactly as it used to: a shaft with a
    // Coulomb dead band settles a dumping wheel differently, and the split surface's slow channel
    // stopped being the slowest thing in the loop. The figure is sourced rather than picked —
    // 107 N from a measured VW Passat B8 front strut, `PublishedCarsImpl.cpp` — which is the whole
    // difference between this and the prescribing the criterion exists to forbid.
    //
    // **The modulator question above is NOT closed by this.** The rear circuit still runs the front's
    // pressure gradients, and a real unit meters its rear outlet valve separately. This criterion
    // stopped being the thing that shows it; that does not make it right.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    struct Surface
    {
        double left;
        double right;
    };

    // Swept across surfaces rather than measured on one, because the frequency is *supposed* to move
    // with the surface: what ends a dump is the road spinning the wheel back up, and a road that can
    // barely do that takes longer about it. Real systems cycle more slowly on ice for the same
    // reason. Every one of these has to land in the band.
    for (const auto surface : {Surface{1.00, 1.00}, Surface{0.60, 0.60}, Surface{1.00, 0.35}})
    {
        const auto world = PhysicsWorld::create(gripPlate(surface.left, surface.right));
        REQUIRE(world.has_value());

        const auto run = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

        REQUIRE(run.stopped);

        for (const auto wheel : {std::size_t{0}, std::size_t{2}})
        {
            const auto peaks = static_cast<double>(run.pressurePeaks[wheel]);
            const auto dumps = static_cast<double>(run.cycles[wheel]);
            const auto engaged = run.engagedTime[wheel];

            CAPTURE(surface.left, surface.right, wheel, peaks, dumps, engaged);

            if (peaks < 5.0)
            {
                // This channel had nothing to do on this surface, which on the front axle of a car
                // that cannot lock its front wheels is the expected answer rather than a failure.
                // See the first case in this file.
                continue;
            }

            REQUIRE(engaged > 0.5);

            const auto frequency = peaks / engaged;

            CAPTURE(frequency);
            REQUIRE(frequency > 4.0);
            REQUIRE(frequency < 20.0);
        }
    }
}

TEST_CASE("the anti-lock system stops being able to help at walking pace, and nothing says so",
          "[assists][antilock][braking][lowspeed]")
{
    // **Criterion 11.** There is no speed threshold anywhere in `raceengine.assists` — grep it. What
    // there is instead is a sensor that reports one pulse per tooth, so the interval between
    // measurements grows as the wheel slows: 1.5 ms at 100 km/h on a 48-pole ring and 30 ms at 5.
    // A wheel that locks between two pulses is a wheel the controller never saw lock.
    //
    // **The measured dropout is between 7.2 and 10.8 km/h**, which is close to the 5 to 7 km/h real
    // systems are described as disengaging at and is not a number this code was given.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    struct Entry
    {
        double speed;
        bool expectCycling;
    };

    // 14.4 km/h still works the system; 7.2 km/h and below cannot.
    for (const auto sample : {Entry{4.0, true}, Entry{2.0, false}, Entry{1.0, false}})
    {
        const auto run = stop(setup.value(), world.value(), withAntilock(setup.value()), sample.speed, 1.0);

        REQUIRE(run.stopped);

        const auto cycles = run.cycles[0] + run.cycles[1] + run.cycles[2];

        CAPTURE(sample.speed, sample.speed * 3.6, cycles);

        if (sample.expectCycling)
        {
            REQUIRE(cycles > 0);
        }
        else
        {
            REQUIRE(cycles == 0);
        }
    }

    SECTION("and the reason is the pulse interval, which is arithmetic on the ring")
    {
        // At 2 m/s a 48-pole ring on a 0.3186 m wheel puts 21 ms between measurements. A front wheel
        // over-braked on this surface loses 1575 - 0.35 * 5000 * 0.3186 = 1018 N.m into 1.45 kg.m^2,
        // which is 702 rad/s^2: from 6.3 rad/s it is stopped in 9 ms. The lock happens entirely
        // between two pulses, and no control law can act on what its sensor did not sample.
        const auto ring = raceengine::ToneRing{};
        const auto interval = 6.283185307179586 * tyreRadius / static_cast<double>(ring.teeth) / 2.0;
        const auto lockTime = (2.0 / tyreRadius) * 1.45 / (setup->corners[0].brakeTorque - 0.35 * 5000.0 * tyreRadius);

        CAPTURE(interval, lockTime);

        REQUIRE(interval > lockTime);
    }
}

TEST_CASE("the pedal pulsation is the modulator's own displacement", "[assists][antilock][braking][pedal]")
{
    // The brief asks for pulsation that comes from the hydraulics rather than from a synthesised
    // buzz, and this is the seam where that is checkable: `derivePedalFeedback` takes how far the
    // unit has moved the wheel pressure away from the pedal, and the *rate* the driver feels is
    // whatever the modulator happens to be cycling at.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    SECTION("the displacement reaches the pedal, and nothing else invents one")
    {
        const auto slipping = std::array<raceengine::SlippingWheel, 0>{};

        // Nothing slipping, nothing displaced, no cue.
        const auto quiet = raceengine::derivePedalFeedback({}, slipping, 1.0, 0.0, 3300.0, 0.0);
        REQUIRE(quiet.finite);
        REQUIRE(quiet.brake == 0.0);

        // Nothing slipping — which is what a wheel under anti-lock control is, by design — and the
        // unit holding a third of the pressure off. That is the cue.
        const auto pulsing = raceengine::derivePedalFeedback({}, slipping, 1.0, 0.0, 3300.0, 0.33);
        REQUIRE(pulsing.finite);
        REQUIRE(pulsing.brake == Catch::Approx(0.33));

        // And a foot that is not on the pedal is told nothing.
        const auto lifted = raceengine::derivePedalFeedback({}, slipping, 0.0, 0.0, 3300.0, 0.33);
        REQUIRE(lifted.brake == 0.0);
    }

    SECTION("and on the car it oscillates at the rate the modulator is cycling at")
    {
        const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
        REQUIRE(world.has_value());

        auto assists = withAntilock(setup.value());

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), hundred);

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

            return sensors;
        };

        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        auto input = VehicleInput{};
        input.brake = 1.0;

        auto peaks = 0;
        auto cycles = std::uint32_t{0};
        auto rising = false;
        auto previous = 0.0;
        auto seconds = 0.0;
        auto strongest = 0.0;

        for (auto step = 0; step < 360 * 6; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup.value(), 1.0), tick);
            const auto stepped =
                stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            // The front-left channel's displacement, as `SimulatedCar` computes it for the pedal.
            const auto displacement = -command.channels.antilockBrakeTorque[0] / setup->corners[0].brakeTorque;

            // **It IS the pressure, not a function of it**, which is the whole claim: the cue is what
            // the driver asked for less what the unit is holding. Checked every step rather than
            // inferred from a frequency, because a synthesised buzz would match a band and could not
            // match this.
            //
            // `channels.pressure` is in **pascals** since 2026-08-23, when the modulator was given a
            // real domain, so it is put back into fractions of this wheel's own full pressure to be
            // compared with a torque ratio. That division is the units change and nothing else.
            const auto asked = command.channels.driverBrakeTorque[0] / setup->corners[0].brakeTorque;
            const auto held = command.channels.pressure[0] / brakeCircuitPressures(setup.value(), 1.0)[0];
            REQUIRE(displacement == Catch::Approx(asked - held).margin(1e-12));

            // And it is silent whenever the unit is not doing anything.
            if (command.channels.antilockPhase[0] == ModulatorPhase::Passive)
            {
                REQUIRE(displacement == Catch::Approx(0.0).margin(1e-12));
            }

            strongest = std::max(strongest, displacement);
            cycles = command.channels.antilockCycles[0];

            // Turning points, counted as a frequency rather than as a count: a peak per cycle.
            if (displacement > previous + 1e-4)
            {
                rising = true;
            }
            else if (rising && displacement < previous - 1e-4)
            {
                peaks++;
                rising = false;
            }

            previous = displacement;

            // **Only while the unit is actually working**, for the same reason the cycling case
            // divides by engaged time: the run includes the moment before it engages and the moment
            // after it lets go, and counting those reports a rate the pedal never pulsed at. It read
            // 4.83 Hz measured over the whole run against 5 to 20 measured over the part of it the
            // modulator was in.
            if (command.channels.antilockActive[0])
            {
                seconds += tick;
            }

            if (state.chassis.linearVelocity.z < 8.0)
            {
                break;
            }
        }

        const auto rate = seconds > 0.0 ? static_cast<double>(peaks) / seconds : 0.0;

        CAPTURE(peaks, cycles, seconds, rate, strongest);

        // It moved at all, and by a useful amount rather than a rounding error.
        REQUIRE(strongest > 0.1);

        // It pulses rather than sitting still — the point of the cue — but **the rate is not asserted
        // here**. It is the modulator's pressure cycle, which is already a documented red in
        // `docs/known-red.md` for coming out below the published band, and a criterion that fails in
        // two places is noise rather than evidence. Measured 3.3 Hz after the brake torque correction
        // of 2026-08-23, down from 4.8 before it: more torque means deeper excursions, and the
        // re-apply rate is what makes those slow.
        REQUIRE(peaks > 5);
    }
}

TEST_CASE("and on a split surface the anti-lock system takes off most of the heading error",
          "[assists][antilock][braking][splitmu]")
{
    // **Closed 2026-08-24 evening by the slip-aware recovery law, at 0.20 of the unassisted heading
    // error against the 0.7 bound** — 19.27 degrees of 96.28 — by the mechanism the absolute case
    // above records: wheels held near their peaks keep their lateral force, and the rear axle's
    // restored cornering stiffness is what resists the moment. The moment itself is unmoderated;
    // yaw-moment build-up limitation is still absent and the account below still says so.
    //
    // **Re-opened 2026-08-24 at 23% against a bound of 30%**, by the same droop correction and for the
    // same reason as the absolute case above: a rear axle that stays on the road puts more of the
    // surface's asymmetry into the car. Red at 16% on 2026-08-23, closed by the mass correction, and
    // re-opened the same day. The mechanism described below is still missing.
    //
    // The mechanism is the one the absolute case above already names, arriving through a new door. A
    // split surface makes a yaw moment out of the difference between the two sides, and the assisted
    // stop is *supposed* to brake the high-grip side hard — that is where its shorter distance comes
    // from. Giving the longitudinal axis its own, flatter load-sensitivity exponent raises grip where
    // the load is high and lowers it where the load is low, which on a split surface means the
    // high-grip side gains and the low-grip side loses: a bigger difference across the car, and
    // therefore a bigger moment, for exactly the same controller.
    //
    // So this is not a controller regression and tuning the modulator will not close it. What closes
    // it is the same missing production feature as the absolute case — yaw-moment build-up
    // limitation, which ramps the high-grip front wheel's pressure so the moment arrives slowly
    // enough for a driver to catch it.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(world.has_value());

    const auto plain = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, 1.0);
    const auto assisted = stop(setup.value(), world.value(), withAntilock(setup.value()), hundred, 1.0);

    REQUIRE(plain.stopped);
    REQUIRE(assisted.stopped);
    CAPTURE(plain.finalYaw * degrees, assisted.finalYaw * degrees);

    REQUIRE(std::abs(assisted.finalYaw) < 0.7 * std::abs(plain.finalYaw));
}

TEST_CASE("what the slip-aware recovery law is worth, measured against itself switched off",
          "[assists][antilock][braking][recovery]")
{
    // **The acceptance evidence for `AntilockSetup::slipAwareRecovery`, and every assertion is an
    // A/B against the previous law rather than a target.** Dominic's rule for the braking chain —
    // a diagnostic, not a calibration target — is why nothing here pins a utilisation figure as
    // correct: what is asserted is that the law *moves* the channels it was built to move, in the
    // direction the utilisation instrument said they were wrong, without giving back what the
    // previous law had. Switching the law off must make every one of these fail, which is what makes
    // this a regression gate for the law rather than a number pinned to a build.
    //
    // The defect it closes over, measured by `[.brake-utilisation]` on 2026-08-24: the rear axle
    // spent a full-pedal ABS stop at ~3.0 times its own peak slip — an *equilibrium* past the peak,
    // where road torque nearly balances brake torque, which the old recovery law read as "recovered"
    // and re-applied into — delivering 0.74 of its capacity against the fronts' 0.85.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    // One floored ABS stop, instrumented the way the utilisation probe is: per wheel per tick, the
    // tyre's own capacity against what it delivered, and where it sat on its own curve. The runout
    // below 5 m/s is excluded from the means for the probe's reason — the anti-lock unit drops out
    // there by design, so a mean across it is a mean of two different experiments.
    struct Measured
    {
        double rearSlipOverPeak = 0.0;
        double frontUtilisation = 0.0;
        double rearUtilisation = 0.0;
        double carUtilisation = 0.0;
        double distance = 0.0;
        double minimumFrontSpeed = 1e9;
    };

    const auto measure = [&](const bool lawOn)
    {
        auto assists = withAntilock(setup.value());
        assists.antilock.slipAwareRecovery = lawOn;

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), hundred);

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

            return sensors;
        };

        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        auto input = VehicleInput{};
        input.brake = 1.0;

        const auto start = state.chassis.position.z;
        auto result = Measured{};
        auto force = std::array<double, 2>{};
        auto capacity = std::array<double, 2>{};
        auto slipOverPeak = 0.0;
        auto ticks = std::size_t{0};

        for (auto step = 0; step < 360 * 30; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup.value(), 1.0), tick);
            const auto stepped =
                stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            if (state.chassis.linearVelocity.z > 5.0)
            {
                for (auto index = std::size_t{0}; index < cornerCount; index++)
                {
                    const auto& corner = lastStep.corners[index];
                    const auto friction =
                        raceengine::tyreFriction(setup->corners[index].tyre, raceengine::TyreAxis::Longitudinal,
                                                 corner.forces.tireVertical, corner.patch.gripMultiplier);

                    force[index / 2] += std::abs(corner.contact.tyre.longitudinal);
                    capacity[index / 2] += friction * corner.forces.tireVertical;

                    if (index >= 2)
                    {
                        const auto peak = corner.contact.tyre.longitudinalPeakSlip;
                        slipOverPeak += peak > 1e-9 ? std::abs(corner.contact.slip.slipRatio) / peak : 0.0;
                    }
                    else
                    {
                        result.minimumFrontSpeed =
                            std::min(result.minimumFrontSpeed, state.corners[index].wheelSpeed * tyreRadius);
                    }
                }

                ticks++;
            }

            if (state.chassis.linearVelocity.z <= 0.0)
            {
                break;
            }
        }

        result.distance = state.chassis.position.z - start;
        result.rearSlipOverPeak = ticks > 0 ? slipOverPeak / (2.0 * static_cast<double>(ticks)) : 0.0;
        result.frontUtilisation = capacity[0] > 1.0 ? force[0] / capacity[0] : 0.0;
        result.rearUtilisation = capacity[1] > 1.0 ? force[1] / capacity[1] : 0.0;
        result.carUtilisation =
            capacity[0] + capacity[1] > 1.0 ? (force[0] + force[1]) / (capacity[0] + capacity[1]) : 0.0;

        return result;
    };

    const auto off = measure(false);
    const auto on = measure(true);

    CAPTURE(off.rearSlipOverPeak, on.rearSlipOverPeak, off.rearUtilisation, on.rearUtilisation, off.frontUtilisation,
            on.frontUtilisation, off.carUtilisation, on.carUtilisation, off.distance, on.distance);

    // The precondition: with the law off, the equilibrium defect exists — the rear axle really is
    // parked far past its peak. If this stops holding, the defect closed some other way and this
    // whole case wants re-deriving rather than trimming.
    REQUIRE(off.rearSlipOverPeak > 2.0);

    // The law's own claim: the rear axle comes most of the way back to its peak...
    REQUIRE(on.rearSlipOverPeak < 0.6 * off.rearSlipOverPeak);

    // ...without giving back the force the old equilibrium was extracting from the falling side of
    // the curve — the shallow far side of a Magic Formula is what made parking past the peak cheap
    // in pure distance, so holding utilisation while halving the slip is the actual work.
    REQUIRE(on.rearUtilisation > off.rearUtilisation - 0.01);

    // The whole car uses more of what its tyres offer, and the stop that falls out is no longer —
    // the distance is fallout here, not the criterion, which is the brief's rule.
    REQUIRE(on.carUtilisation > off.carUtilisation);
    REQUIRE(on.distance < off.distance + 0.05);

    // And no front wheel locked in either arm while the car was moving: the law changed how pressure
    // comes back, and lock prevention is the one thing a recovery law must not trade.
    REQUIRE(off.minimumFrontSpeed > 0.5);
    REQUIRE(on.minimumFrontSpeed > 0.5);
}

TEST_CASE("the car keeps all four wheels on the ground through a hard stop", "[assists][antilock][braking]")
{
    // **Closed 2026-08-24 by the mass correction**, having been red since 2026-08-23 with a rear wheel
    // leaving the ground 0.19 s into the stop.
    //
    // This is the one of the three whose closure is least surprising and most fragile. The pitch
    // transient described below is *unchanged* — it is a property of the suspension and the brake
    // step, not of the mass — but it was swinging the rear axle down to about 11 N, and eleven newtons
    // is a wheel that is already off in every sense but the arithmetic. Putting the car's missing
    // 104 kg back raises the static rear load the transient swings *about*, so the same swing no
    // longer reaches zero.
    //
    // **So the margin is small and the mechanism is still there.** Read the account below before
    // concluding anything about the suspension from this case passing.
    //
    // **The cause is a pitch transient and not the brake balance**, which is why nothing about the
    // valve or the calipers will close it. Traced tick by tick in `[.brake-model]`, a step to the
    // optimum pedal from a settled 100 km/h roll makes the rear axle load swing between about 80 N
    // and 4000 N at roughly 3.3 Hz for a full second before it settles — a pitch overshoot of very
    // nearly 100% of the steady-state transfer, where the steady state at 0.96 g leaves the rear axle
    // carrying about 2300 N and in no danger at all. A road car's rebound damping exists to stop
    // precisely that.
    //
    // **Who closes it**: whoever takes the rear damper's rebound rates, which came across from AC as
    // a wheel-referred knee and have never been checked against a pitch response. It is not the
    // tyre's and it is not the brake system's. Two things would each make it moot and neither is a
    // fix: a pedal applied over 150 ms as a foot applies one rather than as a step, and a droop stop
    // with more than the 20 mm this car states.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    // Half pedal: past the optimum, well short of the floor, and the position this first showed at.
    const auto run = stop(setup.value(), world.value(), golfGtiMk7Assists(setup.value()), hundred, 0.50);

    REQUIRE(run.stopped);
    REQUIRE(run.onPlate);
    CAPTURE(run.distance);

    REQUIRE(run.grounded);
}

TEST_CASE("the rear channel's re-apply is metered separately, and it ships at the front's rate",
          "[assists][antilock][braking][rear-metering]")
{
    // **The mechanism is sourced and the number is not, and both halves are deliberate** — the
    // recession pattern again. Robert Bosch GmbH's US patent 5,284,385 (filed 1990) states that
    // production ABS reduces the REAR axle's pressure build-up rate against the front's, by
    // lengthening the holding phases of exactly the inlet-valve pulsing `reapplyGradient` averages,
    // because dynamic axle-load transfer relieves the rear axle and a rear build-up decoupled from
    // the deceleration "leads inevitably to vehicle instability". Its only figure is "e.g. to halve
    // it" — an example, not a measurement — so `rearReapplyGradient` ships AT the front's value and
    // the modulator is bit-inert until somebody states a number. `[.rear-metering]` prints what the
    // halving does to the cycling criterion's table.
    SECTION("the shipped rear rate is the front's, pinned with its grade")
    {
        // The grade that flips this pin: a measured rear build rate (none is published anywhere
        // reachable — Burckhardt's book is the one untried place), or Dominic stating the patent's
        // own halving example on the car. A number picked to move a cycling frequency must never
        // flip it — that is the prescribing criterion 6 exists to forbid.
        const auto modulator = raceengine::BrakeModulator{};

        REQUIRE(modulator.rearReapplyGradient == modulator.reapplyGradient);
    }

    SECTION("and the selection is live: a stated rear rate reaches only the rear channel")
    {
        // `advanceAntilockChannel` driven directly through one Reapply step, no world needed. The
        // reference is invalid so the slip-aware taper is 1.0 and the arithmetic is exactly
        // `pressure += gradient * dt`, which makes the halving assertable as an equality.
        auto setup = raceengine::AntilockSetup{};
        setup.enabled = true;

        auto reading = raceengine::WheelSpeedReading{};
        reading.speed = 80.0;
        reading.valid = true;
        reading.pulses = 100;

        const auto step = [&](const raceengine::BrakeChannel channel)
        {
            auto state = raceengine::AntilockChannelState{};
            state.phase = ModulatorPhase::Reapply;
            state.pressure = 2.0e6;
            state.departurePressure = 8.0e6;
            state.lastPulses = reading.pulses;

            return raceengine::advanceAntilockChannel(setup, channel, state, reading, 25.0, 27.0, -5.0, false, 1.0e7,
                                                      0.001) -
                   2.0e6;
        };

        const auto frontRise = step(raceengine::BrakeChannel::FrontLeft);
        const auto rearAtDefault = step(raceengine::BrakeChannel::Rear);

        // Ships bit-identical: the same value selected is the same arithmetic.
        REQUIRE(frontRise == rearAtDefault);
        REQUIRE(frontRise == 3.0e4);

        setup.modulator.rearReapplyGradient = 0.5 * setup.modulator.reapplyGradient;

        REQUIRE(step(raceengine::BrakeChannel::Rear) == 0.5 * frontRise);
        REQUIRE(step(raceengine::BrakeChannel::FrontLeft) == frontRise);
    }
}

TEST_CASE("what separate rear metering does to the cycling table", "[.rear-metering]")
{
    // Criterion 6's own sweep with the rear re-apply metered at fractions of the front's rate:
    // `./EngineTests "[.rear-metering]"`. **A diagnostic and not a calibration target** — the
    // Bosch source at `BrakeModulator::rearReapplyGradient` fixes the mechanism and the sign
    // (rear slower) and gives only "halve" as an example figure; this table is what that example
    // does to the criterion's channels, the stop and the yaw, so the decision about stating it
    // can be made on numbers. Factor 1.00 is the shipped car and doubles as the inertness row.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    struct Surface
    {
        double left;
        double right;
    };

    std::printf("\n=== the cycling table against the rear re-apply factor (criterion 6's fixture) ===\n");
    std::printf("\n%-7s %-11s | %9s %9s %9s %9s | %9s %9s\n", "factor", "surface", "FL Hz", "FL peaks", "R Hz",
                "R peaks", "dist m", "yaw deg");
    std::printf("%s\n", "--------------------------------------------------------------------------------------");

    for (const auto factor : {1.0, 0.75, 0.5, 0.33})
    {
        for (const auto surface : {Surface{1.00, 1.00}, Surface{0.60, 0.60}, Surface{1.00, 0.35}})
        {
            const auto world = PhysicsWorld::create(gripPlate(surface.left, surface.right));
            REQUIRE(world.has_value());

            auto assists = withAntilock(setup.value());
            assists.antilock.modulator.rearReapplyGradient = factor * assists.antilock.modulator.reapplyGradient;

            const auto run = stop(setup.value(), world.value(), assists, hundred, 1.0);

            REQUIRE(run.stopped);

            const auto frequency = [&](const std::size_t wheel)
            {
                const auto peaks = static_cast<double>(run.pressurePeaks[wheel]);

                return run.engagedTime[wheel] > 0.5 && peaks >= 5.0 ? peaks / run.engagedTime[wheel] : 0.0;
            };

            std::printf("%-7.2f %5.2f/%-5.2f | %9.3f %9u %9.3f %9u | %9.2f %9.2f\n", factor, surface.left,
                        surface.right, frequency(0), run.pressurePeaks[0], frequency(2), run.pressurePeaks[2],
                        run.distance, run.finalYaw * degrees);
        }
    }

    std::printf("\n  0.00 Hz means the channel had under 5 pressure peaks or under 0.5 s engaged on that\n");
    std::printf("  surface. The band criterion 6 asserts is 4-20 Hz per working channel; the shipped car's\n");
    std::printf("  red is the front channel on the split surface. Factor 0.50 is Bosch 5,284,385's own\n");
    std::printf("  example; 1.00 is the shipped car.\n");
}

TEST_CASE("what separates a washed-out assisted stop from a steering one", "[.washout]")
{
    // The co-design's opening measurement: `./EngineTests "[.washout]"`. The steering pair's
    // ensemble showed the assisted arm mode-hopping — ten of fifteen members keep their steering
    // on mu 0.35 and five wash out — and this table is the per-member mechanism. For each ensemble
    // member it aggregates, over the braking ticks above 5 m/s: how far past the tyre's own peak
    // the front wheels sat, what the ECU believed that slip was, whether the belief ever armed the
    // slip-aware guards, what the reference's rate read against the truth, and what lateral force
    // the front axle had left. Read it against the washed/steering classification in the first two
    // columns before believing any co-design change.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    const auto steering = 0.35;

    std::printf("\n=== the assisted arm, per member: fronts against their own peak, and what the ECU believed ===\n");
    std::printf("\n%3s %8s %8s | %8s %8s %8s %8s | %8s %8s %8s\n", "k", "peakYaw", "latTrv m", "over-pk", "trueSlip",
                "estSlip", "armed%", "lat N", "refAccEr", "dump%");
    std::printf("%s\n",
                "---------------------------------------------------------------------------------------------");

    for (auto k = -7; k <= 7; k++)
    {
        const auto entry = hundred * (1.0 + 0.001 * static_cast<double>(k));
        const auto assists = withAntilock(setup.value());

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), entry);

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

            return sensors;
        };

        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        auto input = VehicleInput{};
        input.brake = 1.0;
        input.steering = steering;

        const auto start = state.chassis.position;

        auto peakYaw = 0.0;
        auto overPeak = 0.0;
        auto trueSlip = 0.0;
        auto estimatedSlip = 0.0;
        auto armed = 0.0;
        auto pastBandTicks = 0.0;
        auto lateralForce = 0.0;
        auto rateError = 0.0;
        auto dumpTicks = 0.0;
        auto ticks = 0.0;
        auto previousSpeed = state.chassis.linearVelocity.z;

        for (auto step = 0; step < 360 * 30; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup.value(), 1.0), tick);
            const auto stepped =
                stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            peakYaw = std::max(peakYaw, std::abs(lastStep.telemetry.yawRate));

            const auto speed = state.chassis.linearVelocity.z;

            if (speed > 5.0)
            {
                const auto trueAcceleration = (speed - previousSpeed) / tick;

                for (const auto wheel : {std::size_t{0}, std::size_t{1}})
                {
                    const auto& corner = lastStep.corners[wheel];
                    const auto peak = corner.contact.tyre.longitudinalPeakSlip;
                    const auto slip = std::abs(corner.contact.slip.slipRatio);

                    overPeak += peak > 1e-9 ? slip / peak : 0.0;
                    trueSlip += slip;
                    estimatedSlip += command.channels.estimatedSlip[wheel];
                    lateralForce += std::abs(corner.contact.tyre.lateral);

                    // The slip-aware guards arm on the ECU's own belief crossing the calibrated
                    // band while the reference is valid — computed here from the channels exactly
                    // as `advanceAntilockChannel` computes `pastBand`.
                    if (slip > 0.20)
                    {
                        pastBandTicks += 1.0;

                        if (command.channels.referenceValid && command.channels.estimatedSlip[wheel] > 0.20)
                        {
                            armed += 1.0;
                        }
                    }

                    if (command.channels.antilockPhase[wheel] == ModulatorPhase::Dump)
                    {
                        dumpTicks += 1.0;
                    }
                }

                rateError += command.channels.referenceAcceleration - trueAcceleration;
                ticks += 1.0;
            }

            previousSpeed = speed;

            if (speed <= 0.0)
            {
                break;
            }
        }

        REQUIRE(ticks > 0.0);

        std::printf("%3d %8.4f %8.2f | %8.2f %8.3f %8.3f %7.1f%% | %8.0f %8.2f %7.1f%%\n", k, peakYaw,
                    state.chassis.position.x - start.x, overPeak / (2.0 * ticks), trueSlip / (2.0 * ticks),
                    estimatedSlip / (2.0 * ticks), pastBandTicks > 0.0 ? 100.0 * armed / pastBandTicks : 0.0,
                    lateralForce / (2.0 * ticks), rateError / ticks, 100.0 * dumpTicks / (2.0 * ticks));
    }

    std::printf("\n  over-pk: mean front |slip| over the tyre's own peak slip (1.0 = at the peak).\n");
    std::printf("  armed%%: of the ticks a front wheel's TRUE slip was past the 0.20 band, how many the\n");
    std::printf("  ECU's belief ALSO read past it with a valid reference — the slip-aware guards' duty.\n");
    std::printf("  refAccEr: mean (referenceAcceleration - true acceleration), m/s^2; both are negative\n");
    std::printf("  under braking, so a negative error is the ECU believing a HARDER stop than the truth.\n");
    std::printf("  lat N: mean front |lateral| per wheel.\n");
}

TEST_CASE("the front-left channel's cycle, tick by tick, washed against steering", "[.washout-trace]")
{
    // The washout probe's aggregate says the two modes differ in how the front channels CYCLE —
    // washed members settle near 0.41 mean front slip with the fewest dumps — and this prints the
    // cycle itself for the adjacent pair that lands on opposite sides: k=+1 washes (peak yaw
    // 0.1255), k=+2 steers (0.2390), 0.1% of entry speed apart. One row per 100 ms: the front-left
    // wheel's true slip, the ECU's belief, the channel's phase, its pressure as a fraction of the
    // request, and the wheel's lateral force.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    const auto steering = 0.35;
    const auto phaseName = [](const ModulatorPhase phase)
    {
        switch (phase)
        {
        case ModulatorPhase::Passive:
            return "Passive";
        case ModulatorPhase::Hold:
            return "Hold";
        case ModulatorPhase::Dump:
            return "Dump";
        case ModulatorPhase::Recover:
            return "Recover";
        case ModulatorPhase::Reapply:
            return "Reapply";
        }

        return "?";
    };

    for (const auto k : {1, 2})
    {
        const auto entry = hundred * (1.0 + 0.001 * static_cast<double>(k));
        const auto assists = withAntilock(setup.value());

        auto state = VehicleState{};
        settle(setup.value(), state, world.value(), entry);

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

            return sensors;
        };

        for (auto step = 0; step < 180; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
            const auto stepped =
                stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();
        }

        auto input = VehicleInput{};
        input.brake = 1.0;
        input.steering = steering;

        std::printf("\n=== k=%+d (%s) — front-left, one row per 100 ms ===\n", k, k == 1 ? "WASHES" : "STEERS");
        std::printf("%6s %8s %8s %9s %8s %8s %7s\n", "t s", "slip", "estSlip", "phase", "press", "lat N", "cycles");

        auto window = 0;
        auto cycleTotal = std::uint32_t{0};

        for (auto step = 0; step < 360 * 10; step++)
        {
            const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                               brakeCircuitPressures(setup.value(), 1.0), tick);
            const auto stepped =
                stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
            REQUIRE(stepped.has_value());
            lastStep = stepped.value();

            cycleTotal = command.channels.antilockCycles[0];

            if (++window == 36)
            {
                window = 0;

                const auto request = brakeCircuitPressures(setup.value(), 1.0)[0];

                std::printf("%6.1f %8.3f %8.3f %9s %8.2f %8.0f %7u\n", static_cast<double>(step + 1) * tick,
                            std::abs(lastStep.corners[0].contact.slip.slipRatio), command.channels.estimatedSlip[0],
                            phaseName(command.channels.antilockPhase[0]),
                            request > 0.0 ? command.channels.pressure[0] / request : 0.0,
                            std::abs(lastStep.corners[0].contact.tyre.lateral), cycleTotal);
            }

            if (state.chassis.linearVelocity.z <= 5.0)
            {
                break;
            }
        }
    }
}

TEST_CASE("the dry stop's tail with the recovery law on, tick by tick", "[.dry-tail]")
{
    // Where the Reapply stuck-exit's dry cost lives: `./EngineTests "[.dry-tail]"`. The recovery
    // law's A/B case shows every above-5 m/s utilisation IMPROVED by the branch and the law-on
    // stop 0.27 m LONGER than law-off, so the loss is in the tail. One row per 25 ms below
    // 12 m/s: both left wheels' true and estimated slip, phase, pressure fraction and cycles.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(1.0, 1.0));
    REQUIRE(world.has_value());

    const auto assists = withAntilock(setup.value());

    auto state = VehicleState{};
    settle(setup.value(), state, world.value(), hundred);

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

        return sensors;
    };

    for (auto step = 0; step < 180; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {}, noBrakePressure, tick);
        const auto stepped =
            stepVehicle(setup.value(), state, VehicleInput{}, noDriveTorque, world.value(), tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();
    }

    auto input = VehicleInput{};
    input.brake = 1.0;

    const auto phaseLetter = [](const ModulatorPhase phase)
    {
        switch (phase)
        {
        case ModulatorPhase::Passive:
            return 'P';
        case ModulatorPhase::Hold:
            return 'H';
        case ModulatorPhase::Dump:
            return 'D';
        case ModulatorPhase::Recover:
            return 'C';
        case ModulatorPhase::Reapply:
            return 'R';
        }

        return '?';
    };

    std::printf("\n=== dry, law on, below 12 m/s — FL and RL: slip / estSlip / phase / press ===\n");
    std::printf("%6s %6s | %6s %6s %2s %5s %4s | %6s %6s %2s %5s %4s\n", "t s", "v m/s", "slipF", "estF", "ph", "prF",
                "cycF", "slipR", "estR", "ph", "prR", "cycR");

    auto window = 0;

    for (auto step = 0; step < 360 * 30; step++)
    {
        const auto command = updateAssists(assists, assistState, sense(), {.brake = 1.0, .throttle = 0.0},
                                           brakeCircuitPressures(setup.value(), 1.0), tick);
        const auto stepped =
            stepVehicle(setup.value(), state, input, noDriveTorque, world.value(), tick, command.brakes);
        REQUIRE(stepped.has_value());
        lastStep = stepped.value();

        const auto speed = state.chassis.linearVelocity.z;

        if (speed < 12.0 && ++window == 9)
        {
            window = 0;

            const auto request = brakeCircuitPressures(setup.value(), 1.0);

            std::printf(
                "%6.2f %6.2f | %6.3f %6.3f %2c %5.2f %4u | %6.3f %6.3f %2c %5.2f %4u\n",
                static_cast<double>(step + 1) * tick, speed, std::abs(lastStep.corners[0].contact.slip.slipRatio),
                command.channels.estimatedSlip[0], phaseLetter(command.channels.antilockPhase[0]),
                request[0] > 0.0 ? command.channels.pressure[0] / request[0] : 0.0, command.channels.antilockCycles[0],
                std::abs(lastStep.corners[2].contact.slip.slipRatio), command.channels.estimatedSlip[2],
                phaseLetter(command.channels.antilockPhase[2]),
                request[2] > 0.0 ? command.channels.pressure[2] / request[2] : 0.0, command.channels.antilockCycles[2]);
        }

        if (speed <= 0.0)
        {
            break;
        }
    }
}

TEST_CASE("what the steering-criteria ensemble distributions look like", "[.abs-ensemble]")
{
    // The distributions behind criterion 5's two cases: `./EngineTests "[.abs-ensemble]"`.
    //
    // **This is a diagnostic and not a calibration target.** Both steering criteria compare one
    // assisted stop against one locked stop, and both trajectories are chaotic: eight documented
    // flips on fraction-of-a-per-cent plant changes (`docs/known-red.md`), every one a re-roll of
    // one arm or the other and never a change in the mechanism under test. This table is what
    // either arm's single number is a sample OF — fifteen stops per arm across a ±0.7% entry-speed
    // band, which is smaller than the fixture's own settling tolerance. Read it before trusting
    // any single-roll verdict on `:746` or `:795`, and before choosing an ensemble statistic.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    const auto steering = 0.35;

    std::printf("\n=== criterion 5's two arms, fifteen entry speeds each, mu 0.35, full pedal, 0.35 lock ===\n");
    std::printf("\n%-9s %9s | %12s %14s %10s %9s %9s\n", "arm", "entry m/s", "peakYawRate", "lateralTravel", "finalYaw",
                "distance", "meanSlip");
    std::printf("%s\n", "---------------------------------------------------------------------------------");

    for (const auto assisted : {false, true})
    {
        const auto runs = steeringEnsemble(setup.value(), world.value(), assisted, steering);

        for (auto member = std::size_t{0}; member < runs.size(); member++)
        {
            const auto& run = runs[member];
            const auto entry = hundred * (1.0 + 0.001 * (static_cast<double>(member) - 7.0));

            REQUIRE(run.stopped);

            std::printf("%-9s %9.3f | %12.5f %14.3f %10.3f %9.2f %9.3f\n", assisted ? "assisted" : "locked", entry,
                        run.peakYawRate, run.lateralTravel, run.finalYaw * degrees, run.distance, run.meanTrueSlip);
        }

        std::printf("%-9s %9s | %12.5f %14.3f  (medians; travel as magnitude)\n", assisted ? "assisted" : "locked",
                    "median", medianOf(runs, [](const StopResult& run) { return run.peakYawRate; }),
                    medianOf(runs, [](const StopResult& run) { return std::abs(run.lateralTravel); }));
    }
}

TEST_CASE("what yaw moment build-up delay is worth, against the share it is metered at", "[.yaw-delay]")
{
    // `./EngineTests "[.yaw-delay]"`. **A diagnostic and not a calibration target.** The mechanism
    // is Limpert's (`AntilockSetup::yawMomentDelay` carries the passage): on a split surface the
    // high wheel's pressure is built in stages from the moment the low wheel first dumps, and hands
    // over to individual control the moment the high wheel reaches its own locking level. The book
    // publishes no rate for the staging and says outright that the whole feature is "a compromise
    // between good steering response and minimized stopping distance", with "differences ... between
    // ABS manufacturers". So this table exists so that the decision — whether this car should have
    // it, and metered how hard — can be made on numbers by whoever drives the car.
    //
    // **The feature ships OFF.** The `off` row is the shipped car and doubles as the inertness row.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    struct Surface
    {
        double left;
        double right;
        const char* name;
    };

    // The mirrored split is a control rather than a second data point: the mechanism is symmetric,
    // so the two SPLIT rows must be each other's mirror image. They also settle which side of the
    // plate each front channel is on, which is not something to assume in a left-handed frame.
    const auto surfaces = std::array{Surface{1.00, 1.00, "dry"}, Surface{0.60, 0.60, "damp"},
                                     Surface{0.35, 0.35, "slippery"}, Surface{1.00, 0.35, "SPLIT"},
                                     Surface{0.35, 1.00, "SPLIT-rev"}};

    // **On a pedal that is applied rather than stepped**, which the third table below shows is the
    // whole difference between this feature acting and this feature being invisible. A tenth of a
    // second is roughly what a real application takes and is the figure `docs/known-red.md` already
    // quotes against the four-wheels-stop fixture's own step pedal.
    const auto ramp = 0.10;

    std::printf("\n=== the split-mu and control table against the delay's apply share (0.10 s pedal ramp) ===\n");
    std::printf("\n%-8s %-9s | %8s %8s %8s %9s | %8s %8s %8s | %7s %9s %8s\n", "share", "surface", "dist m",
                "yaw deg", "lat m", "peak yaw/s", "FL Hz", "FR Hz", "R Hz", "delay s", "FL 1st ms", "FR 1st ms");
    std::printf("%s\n",
                "--------------------------------------------------------------------------------------------------"
                "----------");

    for (const auto share : {-1.0, 0.10, 0.25, 0.50, 1.00})
    {
        for (const auto& surface : surfaces)
        {
            const auto world = PhysicsWorld::create(gripPlate(surface.left, surface.right));
            REQUIRE(world.has_value());

            auto assists = withAntilock(setup.value());
            if (share > 0.0)
            {
                assists.antilock.yawMomentDelay = true;
                assists.antilock.yawDelayApplyShare = share;
            }

            const auto run = stop(setup.value(), world.value(), assists, hundred, 1.0, 0.0, ramp);
            REQUIRE(run.stopped);

            const auto frequency = [&](const std::size_t wheel)
            {
                const auto peaks = static_cast<double>(run.pressurePeaks[wheel]);

                return run.engagedTime[wheel] > 0.5 && peaks >= 5.0 ? peaks / run.engagedTime[wheel] : 0.0;
            };

            if (share > 0.0)
            {
                std::printf("%-8.2f ", share);
            }
            else
            {
                std::printf("%-8s ", "off");
            }

            std::printf("%-9s | %8.2f %8.2f %8.2f %9.3f | %8.3f %8.3f %8.3f | %7.3f %9.2f %8.2f\n", surface.name,
                        run.distance, run.finalYaw * degrees, run.lateralTravel, run.peakYawRate, frequency(0),
                        frequency(1), frequency(2), run.yawDelayTime, 1000.0 * run.firstCycle[0],
                        1000.0 * run.firstCycle[1]);
        }
    }

    std::printf("\n  0.000 Hz means the channel had under 5 pressure peaks or under 0.5 s engaged. Criterion 6's\n");
    std::printf("  band is 4-20 Hz per working channel and its red is the front channel on the split surface.\n");
    std::printf("  The three uniform surfaces are controls: the delay engages on a left-to-right asymmetry,\n");
    std::printf("  so anything it does to them is the engagement test firing where there is no split.\n");

    // **Why every row above is identical**, and it is the fixture rather than the feature: with a
    // step pedal both front channels first let pressure go one physics tick apart even on the split
    // surface, so Limpert's own handover — the high wheel reaching its locking level — fires before
    // the staging can do anything. A real pedal takes about a tenth of a second, and this car's
    // brakes are 2.4 times its front lock pressure at full travel, so how long the low wheel has the
    // road to itself is entirely a property of how the pedal is applied.
    std::printf("\n=== the window the mechanism has, against how the pedal is applied (SPLIT surface) ===\n");
    std::printf("\n%-7s %-7s %-6s | %8s %8s %8s | %9s %9s %8s\n", "ramp s", "pedal", "delay", "dist m", "yaw deg",
                "lat m", "FL 1st ms", "FR 1st ms", "held s");
    std::printf("%s\n", "----------------------------------------------------------------------------------------");

    const auto world = PhysicsWorld::create(gripPlate(1.00, 0.35));
    REQUIRE(world.has_value());

    for (const auto application : {0.0, 0.05, 0.10, 0.20})
    {
        for (const auto pedal : {0.35, 0.60, 1.00})
        {
            for (const auto on : {false, true})
            {
                auto assists = withAntilock(setup.value());
                assists.antilock.yawMomentDelay = on;

                const auto run = stop(setup.value(), world.value(), assists, hundred, pedal, 0.0, application);
                REQUIRE(run.stopped);

                std::printf("%-7.2f %-7.2f %-6s | %8.2f %8.2f %8.2f | %9.2f %9.2f %8.3f\n", application, pedal,
                            on ? "ON" : "off", run.distance, run.finalYaw * degrees, run.lateralTravel,
                            1000.0 * run.firstCycle[0], 1000.0 * run.firstCycle[1], run.yawDelayTime);
            }
        }
    }

    std::printf("\n  `held s` is how long the delay held the high front channel below what the driver asked.\n");
    std::printf("  A first-cycle time of 0.00 ms means that channel never let any pressure go at all.\n");
}

TEST_CASE("what yaw moment build-up delay does to the steering the car keeps", "[.yaw-delay]")
{
    // The steering pair's own fixture, run against the delay. **Ensemble medians**, because a
    // full-pedal stop on mu 0.35 is chaotic and a single roll of it has flipped these criteria eight
    // times (`docs/known-red.md`). The unassisted arm is measured once: it has no anti-lock unit, so
    // no setting of this feature can reach it.
    const auto guard = JoltGuard{};

    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto world = PhysicsWorld::create(gripPlate(0.35, 0.35));
    REQUIRE(world.has_value());

    const auto steering = 0.35;
    const auto plain = steeringEnsemble(setup.value(), world.value(), false, steering);

    const auto plainYaw = medianOf(plain, [](const StopResult& run) { return run.peakYawRate; });
    const auto plainTravel = medianOf(plain, [](const StopResult& run) { return std::abs(run.lateralTravel); });

    std::printf("\n=== the steering pair's medians against the delay's apply share ===\n");
    std::printf("  unassisted median peak yaw rate %.4f /s, lateral travel %.2f m\n", plainYaw, plainTravel);
    std::printf("  the criteria want yaw > 2.0x and travel > 3.0x those\n\n");
    std::printf("%-8s | %10s %8s | %10s %8s | %8s\n", "share", "yaw /s", "x plain", "travel m", "x plain", "dist m");
    std::printf("%s\n", "-------------------------------------------------------------------------");

    for (const auto share : {-1.0, 0.25, 0.50})
    {
        auto runs = std::vector<StopResult>{};
        runs.reserve(15);

        for (auto k = -7; k <= 7; k++)
        {
            const auto entry = hundred * (1.0 + 0.001 * static_cast<double>(k));

            auto assists = withAntilock(setup.value());
            if (share > 0.0)
            {
                assists.antilock.yawMomentDelay = true;
                assists.antilock.yawDelayApplyShare = share;
            }

            runs.push_back(stop(setup.value(), world.value(), assists, entry, 1.0, steering));
        }

        for (const auto& run : runs)
        {
            REQUIRE(run.stopped);
        }

        const auto yaw = medianOf(runs, [](const StopResult& run) { return run.peakYawRate; });
        const auto travel = medianOf(runs, [](const StopResult& run) { return std::abs(run.lateralTravel); });
        const auto distance = medianOf(runs, [](const StopResult& run) { return run.distance; });

        if (share > 0.0)
        {
            std::printf("%-8.2f | ", share);
        }
        else
        {
            std::printf("%-8s | ", "off");
        }

        std::printf("%10.4f %8.2f | %10.2f %8.2f | %8.2f\n", yaw, yaw / plainYaw, travel, travel / plainTravel,
                    distance);
    }

    std::printf("\n  This surface is UNIFORM, so the delay should not engage at all here and these rows\n");
    std::printf("  should agree. A difference is the engagement test firing on two front channels that\n");
    std::printf("  merely dumped a control period apart, which is a real exposure of the mechanism.\n");
}

TEST_CASE("yaw moment build-up delay is off everywhere, and does what the book says when it is not",
          "[assists][antilock][braking][splitmu]")
{
    // The mechanism is Limpert's, §9.3.1: staged pressure build at the "high" wheel from the moment
    // the "low" wheel's first pressure reduction, handing over to individual control when the high
    // wheel reaches its own locking level. `AntilockSetup::yawMomentDelay` carries the passage and
    // the two placed numbers; `[.yaw-delay]` prices the compromise it is.
    //
    // **It ships off**, and this case asserts that first, because the whole build was done on the
    // promise that no golden would move.
    auto setup = AntilockSetup{};
    setup.enabled = true;

    auto state = YawMomentDelayState{};

    auto low = AntilockChannelState{};
    auto high = AntilockChannelState{};

    // The low wheel has let pressure go and is holding 20 bar; the high wheel has not moved and the
    // driver is asking for 110 at both.
    low.cycles = 1;
    low.phase = ModulatorPhase::Reapply;
    low.pressure = 20.0e5;
    const auto requests = std::array<double, 2>{110.0e5, 110.0e5};

    // **The brake application has to begin before either wheel has cycled**, because the trigger is
    // the first pressure reduction *of this application* and `cycles` never resets. One step with
    // both channels quiet is what a driver's foot arriving on the pedal looks like, and it is what
    // the seat lap of 2026-08-30 proved is not optional: without it the delay reads a counter that
    // has been non-zero since the first corner and never arms again.
    const auto begin = [&]
    {
        const auto quiet = AntilockChannelState{};
        return advanceYawMomentDelay(setup, state, quiet, quiet, requests, 0.0, true, 0.001);
    };

    SECTION("off is off, on this setup and on the car's own")
    {
        REQUIRE_FALSE(setup.yawMomentDelay);

        const auto golf = golfGtiMk7();
        REQUIRE(golf.has_value());
        REQUIRE_FALSE(golfGtiMk7Assists(golf.value()).antilock.yawMomentDelay);

        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);
    }

    setup.yawMomentDelay = true;

    SECTION("it engages on one front having reduced pressure and not the other")
    {
        // Neither: nothing to delay for.
        auto quiet = AntilockChannelState{};
        REQUIRE(advanceYawMomentDelay(setup, state, quiet, high, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);
        REQUIRE(state.applied);

        // One: the other is the high wheel, and the ceiling is what the road has just proved the
        // low side will take.
        const auto ceiling = advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001);
        REQUIRE(state.engaged);
        REQUIRE(state.highChannel == 1);
        REQUIRE(ceiling == Catch::Approx(20.0e5));
    }

    SECTION("both having reduced pressure is not a split surface")
    {
        REQUIRE(begin() == 0.0);

        auto other = low;
        REQUIRE(advanceYawMomentDelay(setup, state, low, other, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);
    }

    SECTION("a channel that dumped on an earlier application cannot arm this one")
    {
        // **The case the seat lap found.** `cycles` is cumulative, so by the second corner of a lap
        // both fronts have dumped many times; if the trigger read the raw counter, the delay would
        // be dead for the rest of the session. Baselined at the start of each application, a wheel
        // that cycled a minute ago is simply a wheel that has not cycled yet.
        REQUIRE(begin() == 0.0);
        REQUIRE(state.baseCycles[0] == 0);

        auto stale = AntilockChannelState{};
        stale.cycles = 47;
        stale.phase = ModulatorPhase::Passive;

        auto alsoStale = AntilockChannelState{};
        alsoStale.cycles = 51;

        // Both have cycled before, neither has cycled in this application: nothing to arm on.
        auto fresh = YawMomentDelayState{};
        REQUIRE(advanceYawMomentDelay(setup, fresh, stale, alsoStale, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(fresh.engaged);

        // Now one of them reduces pressure inside this application, and it arms.
        auto reducing = stale;
        reducing.cycles = 48;
        reducing.phase = ModulatorPhase::Reapply;
        reducing.pressure = 20.0e5;

        REQUIRE(advanceYawMomentDelay(setup, fresh, reducing, alsoStale, requests, 0.0, true, 0.001) ==
                Catch::Approx(20.0e5));
        REQUIRE(fresh.engaged);
        REQUIRE(fresh.highChannel == 1);
    }

    SECTION("the staircase climbs at a share of the modulator's own re-apply gradient")
    {
        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) > 0.0);

        const auto step = setup.yawDelayStagePeriod * setup.modulator.reapplyGradient * setup.yawDelayApplyShare;

        // It holds between steps rather than sliding: a controller period short of the stage period
        // leaves it exactly where it was.
        auto elapsed = 0.0;
        while (elapsed + 0.001 < setup.yawDelayStagePeriod)
        {
            REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) == Catch::Approx(20.0e5));
            elapsed += 0.001;
        }

        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) ==
                Catch::Approx(20.0e5 + step));
    }

    SECTION("it follows the low wheel down while that wheel is still reducing")
    {
        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) == Catch::Approx(20.0e5));

        auto dumping = low;
        dumping.phase = ModulatorPhase::Dump;
        dumping.pressure = 8.0e5;

        REQUIRE(advanceYawMomentDelay(setup, state, dumping, high, requests, 0.0, true, 0.001) == Catch::Approx(8.0e5));

        // And not back up again while it is still dumping, whatever that channel reports next.
        dumping.pressure = 15.0e5;
        REQUIRE(advanceYawMomentDelay(setup, state, dumping, high, requests, 0.0, true, 0.001) == Catch::Approx(8.0e5));
    }

    SECTION("it hands over when the high wheel reaches its own locking level")
    {
        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) > 0.0);
        REQUIRE(state.engaged);

        // **Its own first pressure reduction, not its first `Hold`.** A hold is a threshold being
        // watched; the book's handover is the wheel reaching its locking level, which is the same
        // event the trigger reads on the low wheel.
        auto watching = high;
        watching.phase = ModulatorPhase::Hold;
        REQUIRE(advanceYawMomentDelay(setup, state, low, watching, requests, 0.0, true, 0.001) > 0.0);
        REQUIRE(state.engaged);

        auto dumped = high;
        dumped.cycles = 1;
        dumped.phase = ModulatorPhase::Dump;
        REQUIRE(advanceYawMomentDelay(setup, state, low, dumped, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);
        REQUIRE(state.completed);
    }

    SECTION("and it does not re-arm until the foot comes off the pedal")
    {
        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) > 0.0);

        auto dumped = high;
        dumped.cycles = 1;
        REQUIRE(advanceYawMomentDelay(setup, state, low, dumped, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE(state.completed);

        // The low wheel keeps cycling: the delay stays out of it, because the high wheel is
        // individually controlled from here.
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);

        // **The foot coming off is what ends an application**, not the channels going quiet: this
        // modulator returns a channel to `Passive` between cycles, and a stop has dozens of those.
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, false, 0.001) == 0.0);
        REQUIRE_FALSE(state.completed);
        REQUIRE_FALSE(state.applied);

        // And the next application arms again, on its own first pressure reduction.
        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) > 0.0);
        REQUIRE(state.engaged);
    }

    SECTION("it is a braking feature and a foot off the pedal takes it out")
    {
        // Yaw under power is traction control's business, and without this the delay arms on a
        // traction-control brake intervention — measured, one tick of a scripted standing launch.
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, false, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);

        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) > 0.0);
        REQUIRE(state.engaged);

        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, false, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);
    }

    SECTION("a lateral accelerometer switches it off in a corner, which is the book's own rule")
    {
        // "A lateral acceleration sensor switches off the yaw moment delay feature for lateral
        // acceleration exceeding 0.4 g", because there the outer wheel's braking force makes a
        // moment that opposes the lateral force's rather than adding to it.
        REQUIRE(setup.yawDelayLateralLimit == Catch::Approx(0.4 * 9.80665));

        REQUIRE(begin() == 0.0);
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 0.0, true, 0.001) > 0.0);
        REQUIRE(state.engaged);

        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, 5.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);

        // Either sign of it, and it stays off while the car is still cornering.
        REQUIRE(advanceYawMomentDelay(setup, state, low, high, requests, -5.0, true, 0.001) == 0.0);
        REQUIRE_FALSE(state.engaged);
    }
}
