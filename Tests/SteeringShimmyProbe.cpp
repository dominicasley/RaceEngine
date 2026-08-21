// A probe, not a suite: run by hand with `./EngineTests "[.shimmy]"`, hidden the way the device
// probe is. It exists to settle one question from the seat (2026-08-21): at speed the wheel
// oscillates hard enough that letting go is not an option, and at a standstill it does not.
//
// Two hypotheses fit that report and they belong to different layers:
//
//   1. The suspension/tyre model itself oscillates — bump steer, wheel hop, relaxation — in which
//      case the oscillation is in stage one's torque *with the rack held perfectly still*, and no
//      wheel is needed to see it.
//   2. The physics is quiet open loop, and the oscillation only exists with the physical rim in
//      the loop: tyre spring through the rack, against the rim's inertia, through the pipeline's
//      own delays (demand sampled at tick start, torque published at tick end, written at 500 Hz,
//      clipped at the sheet's ceiling). That loop's gain rises with speed — the parked tyre's
//      twist is a couple of N·m where the moving car's trail is tens — which is exactly the
//      standstill/speed asymmetry the seat reports.
//
// The first case writes CSVs of stage one with the rack fixed; the second simulates the rim as a
// free inertia driven through the real mapping code and the real publish cadence. Neither opens a
// device.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::DeviceForceProfile;
using raceengine::ForceMapping;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::mapRackTorque;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::SteeredCorner;
using raceengine::steeredCornerLimit;
using raceengine::SteeringRack;
using raceengine::steeringRackTorque;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::startEngine;
using raceengine::roadTorques;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::VehicleStep;
using raceengine::wheelInertias;

namespace
{

constexpr auto substepTime = 1.0 / 360.0;
constexpr auto rollingRadius = 0.31;

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

// PlayerCar::publishRackTorque's assembly, restated the way RackSignTests restates it.
[[nodiscard]] raceengine::RackTorque rackTorqueOf(const raceengine::VehicleSetup& setup, const VehicleState& state,
                                                  const VehicleStep& stepped, const SteeringRack& rack,
                                                  const double rackTravel, const double rackVelocity)
{
    const auto toBody = glm::conjugate(state.chassis.orientation);

    auto corners = std::array<SteeredCorner, steeredCornerLimit>{};

    for (auto index = std::size_t{0}; index < steeredCornerLimit; index++)
    {
        const auto& solution = stepped.corners[index];
        const auto& suspension = solution.suspension;
        const auto& hardpoints = setup.corners[index].hardpoints;

        const auto worldForce = glm::dvec3(0.0, solution.forces.tireVertical, 0.0) +
                                solution.contact.tyre.longitudinal * solution.contact.forward +
                                solution.contact.tyre.lateral * solution.contact.lateral;

        corners[index] = SteeredCorner{.lowerBallJoint = suspension.lowerBallJoint,
                                       .upperBallJoint = suspension.upperBallJoint,
                                       .steeringArm = suspension.steeringArm,
                                       .rackOuter = hardpoints.steeringRackOuter + glm::dvec3(rackTravel, 0.0, 0.0),
                                       .contactPatch = suspension.contactPatch,
                                       .patchNormal = toBody * solution.patch.normal,
                                       .tyreForce = toBody * worldForce,
                                       .aligningMoment = solution.contact.tyre.aligningMoment};
    }

    return steeringRackTorque(rack, corners, rackVelocity);
}

[[nodiscard]] std::string outputDirectory()
{
    const auto* chosen = std::getenv("OSR_SHIMMY_OUT");

    return chosen != nullptr ? std::string(chosen) : std::string("/tmp");
}

void appendNumber(std::string& text, const double value, const int precision)
{
    auto buffer = std::array<char, 64>{};
    const auto written =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, precision);

    text.append(buffer.data(), written.ptr);
}

void writeFile(const std::string& path, const std::string& text)
{
    auto file = std::ofstream(path);
    REQUIRE(file.is_open());
    file << text;
}

[[nodiscard]] std::string fixed(const double value, const int precision)
{
    auto text = std::string{};
    appendNumber(text, value, precision);

    return text;
}

struct Statistics
{
    double mean = 0.0;
    double deviation = 0.0;
    double lowest = 0.0;
    double highest = 0.0;
};

[[nodiscard]] Statistics statisticsOf(const std::vector<double>& values, const std::size_t from)
{
    auto result = Statistics{};

    if (from >= values.size())
    {
        return result;
    }

    const auto count = static_cast<double>(values.size() - from);
    result.lowest = values[from];
    result.highest = values[from];

    for (auto index = from; index < values.size(); index++)
    {
        result.mean += values[index];
        result.lowest = std::min(result.lowest, values[index]);
        result.highest = std::max(result.highest, values[index]);
    }

    result.mean /= count;

    for (auto index = from; index < values.size(); index++)
    {
        const auto centred = values[index] - result.mean;
        result.deviation += centred * centred;
    }

    result.deviation = std::sqrt(result.deviation / count);

    return result;
}

} // namespace

TEST_CASE("stage one with the rack held still, on a flat road, at speed", "[.shimmy][ffb]")
{
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    auto vehicle = built.value();
    vehicle.rackTravelPerInput = std::abs(vehicle.rackTravelPerInput);

    auto rack = SteeringRack{};
    rack.travelPerInput = vehicle.rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    for (const auto speed : {15.0, 25.0, 35.0})
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, speed);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wheelSpeed = speed / rollingRadius;
        }

        // Two seconds to settle onto its springs at speed, three straight ahead, a quarter-second
        // ramp to a small fixed lock, three held there. Substep resolution throughout, so anything
        // the 120 Hz publish would alias is in the record.
        constexpr auto warmup = 720;
        constexpr auto straight = 1080;
        constexpr auto ramp = 90;
        constexpr auto held = 1080;
        constexpr auto lock = 0.05;

        auto text = std::string("Time [s],Steering Torque [Nm],Tyre Rack Force [N],Vert FL [N],Vert FR [N],"
                                "Lat FL [N],Lat FR [N],Mz FL [Nm],Mz FR [Nm],Last Minus Mean [Nm]\n");

        auto straightTorques = std::vector<double>{};
        auto heldTorques = std::vector<double>{};
        auto publishError = std::vector<double>{};
        auto previousTravel = 0.0;
        auto tickTorques = std::array<double, 3>{};

        for (auto step = 0; step < warmup + straight + ramp + held; step++)
        {
            const auto past = step - (warmup + straight);
            const auto demand = past < 0 ? 0.0 : std::min(lock, lock * static_cast<double>(past) / ramp);

            auto input = VehicleInput{};
            input.steering = demand;

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), substepTime);
            REQUIRE(stepped.has_value());

            const auto travel = demand * vehicle.rackTravelPerInput;
            // The rack velocity PlayerCar publishes is the demand differenced per 120 Hz tick; on a
            // held rack the distinction is nothing, so the substep difference serves.
            const auto velocity = (travel - previousTravel) / substepTime;
            previousTravel = travel;

            const auto torque = rackTorqueOf(vehicle, state, stepped.value(), rack, travel, velocity);
            REQUIRE(torque.finite);

            tickTorques[static_cast<std::size_t>(step % 3)] = torque.steeringTorque;

            if (step < warmup)
            {
                continue;
            }

            const auto lastMinusMean =
                step % 3 == 2 ? tickTorques[2] - (tickTorques[0] + tickTorques[1] + tickTorques[2]) / 3.0 : 0.0;

            if (step % 3 == 2)
            {
                publishError.push_back(lastMinusMean);
            }

            if (step < warmup + straight)
            {
                straightTorques.push_back(torque.steeringTorque);
            }
            else if (past >= ramp)
            {
                heldTorques.push_back(torque.steeringTorque);
            }

            appendNumber(text, static_cast<double>(step - warmup) * substepTime, 6);
            for (const auto value : {torque.steeringTorque, torque.tyreForce,
                                     stepped->corners[0].forces.tireVertical, stepped->corners[1].forces.tireVertical,
                                     stepped->corners[0].contact.tyre.lateral, stepped->corners[1].contact.tyre.lateral,
                                     stepped->corners[0].contact.tyre.aligningMoment,
                                     stepped->corners[1].contact.tyre.aligningMoment, lastMinusMean})
            {
                text += ",";
                appendNumber(text, value, 4);
            }
            text += "\n";
        }

        const auto name = outputDirectory() + "/shimmy-openloop-v" + fixed(speed, 0) + ".csv";
        writeFile(name, text);

        // The verdict numbers: a self-oscillating front end shows as deviation here, with no wheel
        // anywhere near the loop. The held-lock mean against the demand is the spring the rim will
        // see, per radian of rim rotation, which experiment two leans on.
        const auto quietStraight = statisticsOf(straightTorques, straightTorques.size() / 3);
        const auto quietHeld = statisticsOf(heldTorques, heldTorques.size() / 3);
        const auto publish = statisticsOf(publishError, 0);
        const auto rimRadiansPerDemand = glm::radians(rack.lockToLockDegrees) * 0.5;

        WARN("open loop at " << fixed(speed, 0) << " m/s: straight torque " << fixed(quietStraight.mean, 4) << " +- "
                             << fixed(quietStraight.deviation, 4) << " Nm [" << fixed(quietStraight.lowest, 4) << ", "
                             << fixed(quietStraight.highest, 4) << "], held " << fixed(lock, 2) << " lock torque "
                             << fixed(quietHeld.mean, 4) << " +- " << fixed(quietHeld.deviation, 4)
                             << " Nm, spring at the rim "
                             << fixed((quietHeld.mean - quietStraight.mean) / (-lock * rimRadiansPerDemand), 2)
                             << " Nm/rad, last-substep publish error +- " << fixed(publish.deviation, 4) << " Nm -> "
                             << name);
    }
}

namespace
{

// The rim as the pipeline sees it: a free inertia (hands off, which is the case the seat cannot
// hold), fed by the real stage-two mapping at the real cadences. What is modelled of the base is
// its inertia and a token bearing drag; what is *not* modelled is any tuning-menu processing,
// because on this base over hid-fanatecff there is none to model.
struct RimLoopResult
{
    // Peak |rim| over the second after release, and over the last second. The ratio is the verdict:
    // above one the loop is feeding the oscillation, below it the loop is eating it.
    double earlyDegrees = 0.0;
    double lateDegrees = 0.0;
    // What the motor was told to make, past the settling: its spread is the texture in the hands.
    double appliedDeviation = 0.0;
    double appliedSpan = 0.0;
    // The same spread over the final second alone, which is what a *standing* buzz shows up in: a
    // transient's texture washes out of this number and a limit cycle's does not.
    double lateAppliedDeviation = 0.0;
};

struct RimLoopConfiguration
{
    double speed = 25.0;
    double rimInertia = 0.03;
    // The candidate fix: N·m per rad/s of *measured* rim speed, applied against the published
    // torque in stage two, estimated exactly the way a backend would have to (quantised angle,
    // differenced per write, lightly smoothed).
    double softwareDamper = 0.0;
    double releaseDegrees = 2.0;
    double seconds = 8.0;
    // The cadence a 60 Hz display actually produces: `Engine::step` runs the fixed-step loop in
    // catch-up, so two ticks execute back to back and publish microseconds apart, then nothing for
    // a frame. The demand both ticks read is the same device sample, which puts a full frame's rim
    // motion into the first tick's rack velocity and none into the second's — and the writer's
    // reconstruction takes its slope from that back-to-back pair.
    bool burstPublish = false;
    // The two repairs, together: the rack velocity differenced over the device-sample interval
    // rather than the tick, and the reconstruction's slope taken over the publisher's stated
    // interval rather than the wall clock. What the pipeline does as of 2026-08-21; false is what
    // it did before, kept so the failure stays reproducible beside the fix.
    bool fixedPipeline = false;
    // The device the way evdev actually behaves: a report arrives only when the quantised axis
    // value changes, so a wheel that stops moving stops talking. Modelling this is what exposed
    // the first repair's own regression — see `ageHeldVelocity`.
    bool onChangeReports = false;
    // The repair's repair. Holding the last derived velocity across *one* stale tick is the
    // catch-up burst being bridged; holding it for ever turns the rack's friction into a standing
    // torque the moment the wheel stops reporting — the motor pushes the rim a count, the count
    // flips the velocity's sign, and the wheel buzzes on centre indefinitely. Reported from the
    // seat as a constant on-centre vibration at any speed that clears up while turning, which is
    // exactly the regime split: turning streams reports, centre goes quiet.
    bool ageHeldVelocity = false;
    // Publish only the tyres' share of the rack torque, leaving the rack's own friction and
    // damping out of what the motor is asked to render. Those two terms are functions of the
    // *measured* rim velocity delivered back to the rim through the pipeline's delay — and a
    // Coulomb friction behind a delay is a relay oscillator: the motor pushes, the rim moves, the
    // stale measurement flips the sign, the motor reverses. The driver's hands already feel the
    // physical wheel's own friction; the simulated rack's belongs to the car the keyboard drives,
    // not to a torque loop.
    bool tyreOnlyPublish = false;
    // Hands on the rim as a spring toward a held target — compliant, which prescribed motion is
    // not, and compliance is what a relay needs to ignite: a grip cannot be perfectly rigid
    // against a motor that is actively pushing.
    double handsStiffness = 0.0;
    double handsTargetDegrees = 0.0;
    // Nonzero prescribes the rim instead of integrating it: hands turning at this rate, parked,
    // which is the standstill "spikes as you turn" report. The measurement is then the applied
    // torque's texture, not the rim's stability.
    double sweepDegreesPerSecond = 0.0;
    // Nonzero makes the prescribed hands *stop* between movements for this long — the on-centre
    // correction pattern of driving straight. Stopping abruptly is the point: the last derived
    // velocity before the device goes quiet is a mid-motion one, so a held velocity publishes the
    // rack's whole saturated friction as a standing torque through every pause, flipping sign with
    // every correction. That is the on-centre vibration as felt with hands on the rim.
    double sweepDwellSeconds = 0.0;
    double sweepLimitDegrees = 45.0;
    std::string trace;
};

[[nodiscard]] RimLoopResult simulateRimInLoop(const raceengine::VehicleSetup& vehicle, const PhysicsWorld& world,
                                              const SteeringRack& rack, const RimLoopConfiguration& configuration)
{
    // 1/1440 s micro steps: four per physics substep, twelve per engine tick, and the 500 Hz
    // writer lands on every third one — 2.08 ms against the driver's 2 ms hrtimer.
    constexpr auto microTime = 1.0 / 1440.0;
    constexpr auto writerTime = 3.0 * microTime;
    const auto halfLock = glm::radians(rack.lockToLockDegrees) * 0.5;

    // The axis as the device reports it: sixteen bits over the base's 900 degree range.
    const auto countRadians = glm::radians(900.0) / 65536.0;

    auto profile = DeviceForceProfile{};
    auto mapping = ForceMapping{};
    // The sheet that ships: gain 1, ceiling 4. The damper goes through the mapping the way the
    // service sends it, so what is probed is the shipped path and not a restatement of it.
    mapping.ceilingTorque = 4.0;
    mapping.damping = configuration.softwareDamper;

    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, configuration.speed);

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        state.corners[index].wheelSpeed = configuration.speed / rollingRadius;
    }

    const auto parked = configuration.speed < 0.5;
    const auto sweeping = configuration.sweepDegreesPerSecond > 0.0;
    const auto sweepRate = glm::radians(configuration.sweepDegreesPerSecond);
    const auto handsTarget = glm::radians(configuration.handsTargetDegrees);
    // Half-critical: a grip resists and gives, which is what lets a relay ignite against it.
    const auto handsDamping =
        configuration.handsStiffness > 0.0 ? std::sqrt(configuration.handsStiffness * configuration.rimInertia) : 0.0;
    const auto sweepLimit = glm::radians(configuration.sweepLimitDegrees);
    // A correction is brief and the pause after it long, which is what hands do on a straight.
    constexpr auto sweepMoveSeconds = 0.15;

    auto rim = 0.0;
    auto rimSpeed = 0.0;
    auto sweepDirection = 1.0;
    auto sweepPhase = 0.0;
    auto dwelling = false;
    constexpr auto bearingDrag = 0.02;

    // Two ticks per frame in burst mode — a 60 Hz display's catch-up loop — publishing
    // microseconds apart at the frame's start; one tick per frame otherwise, publishing every
    // 8.33 ms, which is what the writer's reconstruction was written against.
    const auto ticksPerFrame = configuration.burstPublish ? 2 : 1;
    constexpr auto burstSpacing = 50e-6;

    auto newestTorque = 0.0;
    auto olderTorque = 0.0;
    auto newestTime = 0.0;
    auto olderTime = 0.0;
    auto published = 0;
    auto commanded = 0.0;
    auto applied = 0.0;
    auto measuredRim = 0.0;
    auto estimatedSpeed = 0.0;
    auto previousTravel = 0.0;
    auto previousVelocity = 0.0;
    auto lastReportedCount = 0.0;
    auto lastReportFrame = 0;
    auto staleTicks = 0;

    const auto warmupFrames = 240 / ticksPerFrame;
    const auto frames = warmupFrames + static_cast<int>(configuration.seconds * 120.0) / ticksPerFrame;
    auto early = 0.0;
    auto late = 0.0;
    auto appliedSamples = std::vector<double>{};
    auto lateApplied = std::vector<double>{};

    auto text = std::string("Time [s],Rim [deg],Rim Speed [deg/s],Published [Nm],Applied [Nm]\n");

    for (auto frame = 0; frame < frames; frame++)
    {
        const auto holding = frame < warmupFrames;
        const auto frameStart = static_cast<double>((frame - warmupFrames) * ticksPerFrame) / 120.0;

        if (frame == warmupFrames && !sweeping)
        {
            rim = glm::radians(configuration.releaseDegrees);
        }

        // Demand sampled where the game samples it: off the quantised axis at the top of the
        // catch-up loop. In burst mode both ticks run microseconds apart, so both read the same
        // device sample — which is precisely what loads a whole frame's rim motion into the first
        // tick's rack velocity and none into the second's.
        const auto counted = std::round(rim / countRadians) * countRadians;
        const auto demand = std::clamp(counted / halfLock, -1.0, 1.0);

        // Whether the device said anything this frame. On-change reporting is what the hardware
        // does: a count that has not moved is a wheel that is not talking, and the stamp the game
        // reads stays where it was.
        const auto arrived = !configuration.onChangeReports || counted != lastReportedCount;
        auto reportGapSeconds = 0.0;

        if (arrived)
        {
            reportGapSeconds = static_cast<double>((frame - lastReportFrame) * ticksPerFrame) / 120.0;
            lastReportFrame = frame;
            lastReportedCount = counted;
        }

        auto input = VehicleInput{};
        input.steering = holding ? 0.0 : demand;
        input.brake = parked ? 1.0 : 0.0;

        for (auto tick = 0; tick < ticksPerFrame; tick++)
        {
            auto tickTorque = 0.0;

            for (auto sub = 0; sub < 3; sub++)
            {
                const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world, substepTime);
                REQUIRE(stepped.has_value());

                if (sub == 2)
                {
                    const auto travel = input.steering * vehicle.rackTravelPerInput;

                    // Faulty original: differenced per tick, which alternates double and zero
                    // across a burst. Fixed: differenced over the device's own report interval,
                    // held across a stale tick — and with `ageHeldVelocity`, held across exactly
                    // one, because the second stale tick means the wheel has stopped talking.
                    auto velocity = 0.0;

                    if (!configuration.fixedPipeline)
                    {
                        velocity = (travel - previousTravel) * 120.0;
                    }
                    else if (tick == 0 && arrived && reportGapSeconds > 0.0)
                    {
                        velocity = (travel - previousTravel) / reportGapSeconds;
                        staleTicks = 0;
                    }
                    else
                    {
                        staleTicks++;
                        velocity =
                            configuration.ageHeldVelocity && staleTicks > 1 ? 0.0 : previousVelocity;
                    }

                    previousTravel = travel;
                    previousVelocity = velocity;

                    // Tyre-only leaves the rack's friction and damping out of what the motor is
                    // asked for, which zeroing the velocity does exactly: both terms are functions
                    // of it and nothing else in the derivation reads it.
                    const auto torque = rackTorqueOf(vehicle, state, stepped.value(), rack, travel,
                                                     configuration.tyreOnlyPublish ? 0.0 : velocity);
                    REQUIRE(torque.finite);
                    tickTorque = torque.steeringTorque;
                }
            }

            olderTorque = newestTorque;
            olderTime = newestTime;
            newestTorque = tickTorque;
            newestTime = frameStart + static_cast<double>(tick) * burstSpacing;
            published++;
        }

        if (holding)
        {
            continue;
        }

        for (auto micro = 0; micro < 12 * ticksPerFrame; micro++)
        {
            const auto now = frameStart + static_cast<double>(micro) * microTime;

            if (micro % 3 == 0)
            {
                // The service's reconstruction, verbatim: continue the last published slope,
                // clamped to one sample's worth of excursion past the newest value. In burst mode
                // the slope's denominator is the burst's microseconds, so the clamp engages almost
                // immediately and the output holds at newest-plus-one-whole-change for the rest of
                // the frame.
                const auto ahead = now - newestTime;
                // Fixed: the slope's denominator is the publisher's stated interval, one tick.
                // Faulty: the wall span between publishes, which across a burst is microseconds.
                const auto span = configuration.fixedPipeline ? 1.0 / 120.0 : newestTime - olderTime;
                const auto change = newestTorque - olderTorque;
                const auto reach = std::abs(change);
                const auto torque = published < 2 || span <= 0.0 || ahead <= 0.0
                                        ? newestTorque
                                        : newestTorque + std::clamp(change * (ahead / span), -reach, reach);

                const auto countedNow = std::round(rim / countRadians) * countRadians;
                const auto instant = (countedNow - measuredRim) / writerTime;
                measuredRim = countedNow;
                const auto blend = writerTime / (writerTime + 0.005);
                estimatedSpeed += (instant - estimatedSpeed) * blend;

                const auto command = mapRackTorque(profile, mapping, torque, estimatedSpeed, commanded, writerTime, 1.0);
                REQUIRE_FALSE(command.rejected);

                commanded = command.commandedTorque;
                applied = command.deliveredTorque;

                if (now > 0.5)
                {
                    appliedSamples.push_back(applied);
                }

                if (now > configuration.seconds - 1.0)
                {
                    lateApplied.push_back(applied);
                }
            }

            if (sweeping)
            {
                if (configuration.sweepDwellSeconds > 0.0)
                {
                    sweepPhase += microTime;

                    if (dwelling && sweepPhase >= configuration.sweepDwellSeconds)
                    {
                        dwelling = false;
                        sweepPhase = 0.0;
                    }
                    else if (!dwelling && sweepPhase >= sweepMoveSeconds)
                    {
                        dwelling = true;
                        sweepPhase = 0.0;
                    }
                }

                if (dwelling)
                {
                    rimSpeed = 0.0;
                }
                else
                {
                    rim += sweepDirection * sweepRate * microTime;
                    rimSpeed = sweepDirection * sweepRate;

                    if (rim > sweepLimit)
                    {
                        sweepDirection = -1.0;
                    }
                    else if (rim < -sweepLimit)
                    {
                        sweepDirection = 1.0;
                    }
                }
            }
            else
            {
                const auto hands = configuration.handsStiffness > 0.0
                                       ? configuration.handsStiffness * (handsTarget - rim) - handsDamping * rimSpeed
                                       : 0.0;

                rimSpeed += microTime * (applied + hands - bearingDrag * rimSpeed) / configuration.rimInertia;
                rim += microTime * rimSpeed;
            }

            if (now >= 0.5 && now < 1.5)
            {
                early = std::max(early, std::abs(rim));
            }
            if (now > configuration.seconds - 1.0)
            {
                late = std::max(late, std::abs(rim));
            }

            if (micro % 3 == 0)
            {
                appendNumber(text, now, 5);
                for (const auto value : {glm::degrees(rim), glm::degrees(rimSpeed), newestTorque, applied})
                {
                    text += ",";
                    appendNumber(text, value, 4);
                }
                text += "\n";
            }
        }
    }

    if (!configuration.trace.empty())
    {
        writeFile(configuration.trace, text);
    }

    const auto texture = statisticsOf(appliedSamples, 0);
    const auto settled = statisticsOf(lateApplied, 0);

    return RimLoopResult{.earlyDegrees = glm::degrees(early),
                         .lateDegrees = glm::degrees(late),
                         .appliedDeviation = texture.deviation,
                         .appliedSpan = texture.highest - texture.lowest,
                         .lateAppliedDeviation = settled.deviation};
}

} // namespace

TEST_CASE("the rim in the loop, hands off, parked and at speed", "[.shimmy][ffb]")
{
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    auto vehicle = built.value();
    vehicle.rackTravelPerInput = std::abs(vehicle.rackTravelPerInput);

    auto rack = SteeringRack{};
    rack.travelPerInput = vehicle.rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    const auto directory = outputDirectory();

    struct Case
    {
        std::string name;
        RimLoopConfiguration configuration;
    };

    auto cases = std::vector<Case>{};
    cases.push_back({"parked, clean publish, released at 20", {.speed = 0.0, .releaseDegrees = 20.0, .trace = {}}});
    cases.push_back({"parked, burst publish, released at 20",
                     {.speed = 0.0, .releaseDegrees = 20.0, .burstPublish = true, .trace = {}}});
    cases.push_back({"parked, hands sweeping 90 deg/s, clean publish",
                     {.speed = 0.0, .sweepDegreesPerSecond = 90.0, .trace = directory + "/shimmy-sweep-clean.csv"}});
    cases.push_back({"parked, hands sweeping 90 deg/s, burst publish",
                     {.speed = 0.0,
                      .burstPublish = true,
                      .sweepDegreesPerSecond = 90.0,
                      .trace = directory + "/shimmy-sweep-burst.csv"}});
    cases.push_back({"25 m/s, clean publish, released at 2", {.speed = 25.0, .trace = {}}});
    cases.push_back(
        {"25 m/s, clean publish, released at 10", {.speed = 25.0, .releaseDegrees = 10.0, .trace = {}}});
    cases.push_back({"25 m/s, burst publish, released at 2",
                     {.speed = 25.0, .burstPublish = true, .trace = directory + "/shimmy-rim-v25-burst.csv"}});
    cases.push_back(
        {"25 m/s, burst publish, released at 10",
         {.speed = 25.0, .releaseDegrees = 10.0, .burstPublish = true, .trace = directory + "/shimmy-rim-v25-burst-r10.csv"}});
    cases.push_back(
        {"25 m/s, burst publish, released at 30", {.speed = 25.0, .releaseDegrees = 30.0, .burstPublish = true, .trace = {}}});
    cases.push_back(
        {"35 m/s, burst publish, released at 10", {.speed = 35.0, .releaseDegrees = 10.0, .burstPublish = true, .trace = {}}});
    cases.push_back({"25 m/s, burst, light rim (0.015), released at 10",
                     {.speed = 25.0, .rimInertia = 0.015, .releaseDegrees = 10.0, .burstPublish = true, .trace = {}}});

    for (const auto damper : {0.15, 0.3, 0.6})
    {
        cases.push_back({"25 m/s, burst, released at 10, damper " + fixed(damper, 2),
                         {.speed = 25.0,
                          .softwareDamper = damper,
                          .releaseDegrees = 10.0,
                          .burstPublish = true,
                          .trace = damper == 0.3 ? directory + "/shimmy-rim-v25-burst-damped.csv" : std::string{}}});
    }

    cases.push_back({"25 m/s, burst, FIXED pipeline, released at 10",
                     {.speed = 25.0,
                      .releaseDegrees = 10.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .trace = directory + "/shimmy-rim-v25-burst-fixed.csv"}});
    cases.push_back({"25 m/s, burst, FIXED pipeline, released at 30",
                     {.speed = 25.0, .releaseDegrees = 30.0, .burstPublish = true, .fixedPipeline = true, .trace = {}}});
    cases.push_back({"35 m/s, burst, FIXED pipeline, released at 10",
                     {.speed = 35.0, .releaseDegrees = 10.0, .burstPublish = true, .fixedPipeline = true, .trace = {}}});
    cases.push_back({"25 m/s, burst, FIXED pipeline + damper 0.15, released at 10",
                     {.speed = 25.0,
                      .softwareDamper = 0.15,
                      .releaseDegrees = 10.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .trace = {}}});
    cases.push_back({"parked, hands sweeping 90 deg/s, burst, FIXED pipeline",
                     {.speed = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .sweepDegreesPerSecond = 90.0,
                      .trace = directory + "/shimmy-sweep-burst-fixed.csv"}});

    // The regression the first repair shipped, reported from the seat as a constant on-centre buzz
    // at any speed that clears while turning: with the device only talking when its count moves, a
    // velocity held for ever turns the rack's friction into a standing torque at centre. Aging the
    // hold to one stale tick is the repair's repair, and the burst cases above must stay stable
    // with it.
    for (const auto speed : {8.0, 25.0})
    {
        cases.push_back({fixed(speed, 0) + " m/s, burst, FIXED, on-change reports, held velocity",
                         {.speed = speed,
                          .releaseDegrees = 0.5,
                          .burstPublish = true,
                          .fixedPipeline = true,
                          .onChangeReports = true,
                          .trace = speed < 10.0 ? directory + "/shimmy-rim-buzz-held.csv" : std::string{}}});
        cases.push_back({fixed(speed, 0) + " m/s, burst, FIXED, on-change reports, aged velocity",
                         {.speed = speed,
                          .releaseDegrees = 0.5,
                          .burstPublish = true,
                          .fixedPipeline = true,
                          .onChangeReports = true,
                          .ageHeldVelocity = true,
                          .trace = speed < 10.0 ? directory + "/shimmy-rim-buzz-aged.csv" : std::string{}}});
    }

    cases.push_back({"25 m/s, burst, FIXED + aged velocity, released at 10",
                     {.speed = 25.0,
                      .releaseDegrees = 10.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .trace = {}}});

    // Hands on the rim, correcting and pausing about centre at speed — the driving-straight
    // pattern the on-centre vibration was reported from. The measurement is the torque's spread
    // over the final second: a held velocity keeps the rack's saturated friction standing through
    // every pause and flips it with every correction; an aged one lets it go.
    cases.push_back({"8 m/s, corrections about centre with pauses, held velocity",
                     {.speed = 8.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .sweepDegreesPerSecond = 60.0,
                      .sweepDwellSeconds = 0.5,
                      .sweepLimitDegrees = 3.0,
                      .trace = directory + "/shimmy-dwell-held.csv"}});
    cases.push_back({"8 m/s, corrections about centre with pauses, aged velocity",
                     {.speed = 8.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .sweepDegreesPerSecond = 60.0,
                      .sweepDwellSeconds = 0.5,
                      .sweepLimitDegrees = 3.0,
                      .trace = directory + "/shimmy-dwell-aged.csv"}});

    // The aggressive shake, as reported: standstill, hands *holding* the rim a couple of degrees
    // off centre — compliantly, which is what a grip is — with the whole shipped pipeline in
    // place. The rack's friction and damping ride the measured velocity back to the motor through
    // the pipeline's delay, and a Coulomb term behind a delay is a relay: it should ignite from
    // the hands' own approach to the target and never stop. The tyre-only publish is the control.
    cases.push_back({"standstill, hands holding 2 deg off, rack resistance in the loop",
                     {.speed = 0.0,
                      .releaseDegrees = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .handsStiffness = 40.0,
                      .handsTargetDegrees = 2.0,
                      .trace = directory + "/shimmy-relay-inloop.csv"}});
    cases.push_back({"standstill, hands holding 2 deg off, tyre-only publish",
                     {.speed = 0.0,
                      .releaseDegrees = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .tyreOnlyPublish = true,
                      .handsStiffness = 40.0,
                      .handsTargetDegrees = 2.0,
                      .trace = directory + "/shimmy-relay-tyreonly.csv"}});
    cases.push_back({"25 m/s, hands holding 2 deg off, rack resistance in the loop",
                     {.speed = 25.0,
                      .releaseDegrees = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .handsStiffness = 40.0,
                      .handsTargetDegrees = 2.0,
                      .trace = {}}});
    cases.push_back({"25 m/s, hands holding 2 deg off, tyre-only publish",
                     {.speed = 25.0,
                      .releaseDegrees = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .tyreOnlyPublish = true,
                      .handsStiffness = 40.0,
                      .handsTargetDegrees = 2.0,
                      .trace = {}}});

    // With the rack resistance out of the loop its Coulomb term stops stabilising the hands-off
    // case too, so the at-speed release is re-run tyre-only to find the damper that stands in.
    for (const auto damper : {0.0, 0.25, 0.5})
    {
        cases.push_back({"25 m/s, burst, FIXED, tyre-only, released at 10, damper " + fixed(damper, 2),
                         {.speed = 25.0,
                          .softwareDamper = damper,
                          .releaseDegrees = 10.0,
                          .burstPublish = true,
                          .fixedPipeline = true,
                          .onChangeReports = true,
                          .ageHeldVelocity = true,
                          .tyreOnlyPublish = true,
                          .trace = {}}});
    }

    // The rig's own shake, per the exit trace of 2026-08-21: parked near centre, the carcass
    // spring through the rack against the rim, the motor clipped at the sheet's ceiling, and the
    // driver's grip the only damping in the loop. Swept across rim inertia (the base's reflected
    // inertia is not a measured number) and the post-clip damper, hands off — the loosest grip
    // there is.
    for (const auto inertia : {0.012, 0.03})
    {
        for (const auto damper : {0.0, 0.25, 0.5})
        {
            cases.push_back({"parked, burst, FIXED, J " + fixed(inertia, 3) + ", released at 15, damper " +
                                 fixed(damper, 2),
                             {.speed = 0.0,
                              .rimInertia = inertia,
                              .softwareDamper = damper,
                              .releaseDegrees = 15.0,
                              .burstPublish = true,
                              .fixedPipeline = true,
                              .onChangeReports = true,
                              .ageHeldVelocity = true,
                              .trace = {}}});
        }
    }

    cases.push_back({"parked, loose hands 2 deg off, light rim, no damper",
                     {.speed = 0.0,
                      .rimInertia = 0.012,
                      .releaseDegrees = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .handsStiffness = 8.0,
                      .handsTargetDegrees = 2.0,
                      .trace = directory + "/shimmy-loose-hands.csv"}});
    cases.push_back({"parked, loose hands 2 deg off, light rim, damper 0.25",
                     {.speed = 0.0,
                      .rimInertia = 0.012,
                      .softwareDamper = 0.25,
                      .releaseDegrees = 0.0,
                      .burstPublish = true,
                      .fixedPipeline = true,
                      .onChangeReports = true,
                      .ageHeldVelocity = true,
                      .handsStiffness = 8.0,
                      .handsTargetDegrees = 2.0,
                      .trace = {}}});

    for (const auto& probe : cases)
    {
        const auto result = simulateRimInLoop(vehicle, world.value(), rack, probe.configuration);

        if (probe.configuration.sweepDegreesPerSecond > 0.0)
        {
            WARN(probe.name << ": applied torque texture +- " << fixed(result.appliedDeviation, 3) << " Nm, span "
                            << fixed(result.appliedSpan, 3) << " Nm, final second +- "
                            << fixed(result.lateAppliedDeviation, 3) << " Nm");

            continue;
        }

        const auto growing = result.lateDegrees > result.earlyDegrees * 1.5;
        const auto decaying = result.lateDegrees < result.earlyDegrees * 0.5;

        WARN(probe.name << ": released at " << fixed(probe.configuration.releaseDegrees, 1) << " deg, early peak "
                        << fixed(result.earlyDegrees, 2) << " deg, final second peak " << fixed(result.lateDegrees, 2)
                        << " deg, final second torque +- " << fixed(result.lateAppliedDeviation, 3) << " Nm -> "
                        << (growing ? "GROWS" : decaying ? "decays" : "holds"));
    }
}

TEST_CASE("stage one at a standstill, engine idling in gear", "[.shimmy][ffb]")
{
    // The third report from the seat: a constant vibration on centre that survives every transport
    // repair and is present *at a standstill*. At rest with the rim centred, stage one has no
    // velocity, no slip and no lock — so if it moves at all, the physics is publishing an
    // oscillation, and the one live thing in a parked idling car is the driveline. The game idles
    // in gear (PlayerCar's gear defaults to 1) with the automation on the clutch, so that exact
    // regime is stepped here, brake held and free, and what stage one publishes is recorded beside
    // what each transport variant would have the motor do with it.
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 400.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    auto vehicle = built.value();
    vehicle.rackTravelPerInput = std::abs(vehicle.rackTravelPerInput);

    auto rack = SteeringRack{};
    rack.travelPerInput = vehicle.rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    const auto driveline = raceengine::golfGtiMk7Driveline();

    for (const auto brake : {1.0, 0.0})
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);

        for (auto step = 0; step < 1440; step++)
        {
            REQUIRE(stepVehicle(vehicle, state, VehicleInput{}, noDriveTorque, world.value(), substepTime).has_value());
        }

        auto engineState = raceengine::DrivelineState{};
        startEngine(driveline, engineState);

        auto road = std::array<double, cornerCount>{};

        // Two seconds in neutral to let the idle settle, then into first the way the game sits at
        // a light: no throttle, no foot on the clutch pedal, the automation holding it.
        auto idling = VehicleInput{};
        idling.gear = 0;
        idling.brake = brake;

        auto inGear = idling;
        inGear.gear = 1;

        auto text = std::string("Time [s],Steering Torque [Nm],Wheel Torque FL [Nm],Long FL [N],Long FR [N],"
                                "Lat FL [N],Lat FR [N],Engine [rad/s]\n");

        auto published = std::vector<double>{};
        auto substepTorques = std::vector<double>{};

        const auto settleTicks = 720;
        const auto measuredTicks = 2160;

        for (auto step = 0; step < settleTicks + measuredTicks; step++)
        {
            const auto& input = step < settleTicks ? idling : inGear;

            const auto torques = stepDriveline(driveline, engineState,
                                               {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed,
                                                state.corners[2].wheelSpeed, state.corners[3].wheelSpeed},
                                               wheelInertias(vehicle), road, input, substepTime);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(vehicle, state, input, torques->wheel, world.value(), substepTime);
            REQUIRE(stepped.has_value());

            road = roadTorques(stepped.value());

            if (step < settleTicks)
            {
                continue;
            }

            const auto torque = rackTorqueOf(vehicle, state, stepped.value(), rack, 0.0, 0.0);
            REQUIRE(torque.finite);

            substepTorques.push_back(torque.steeringTorque);

            if (step % 3 == 2)
            {
                published.push_back(torque.steeringTorque);
            }

            appendNumber(text, static_cast<double>(step - settleTicks) * substepTime, 6);
            for (const auto value :
                 {torque.steeringTorque, torques->wheel[0], stepped->corners[0].contact.tyre.longitudinal,
                  stepped->corners[1].contact.tyre.longitudinal, stepped->corners[0].contact.tyre.lateral,
                  stepped->corners[1].contact.tyre.lateral, engineState.engineSpeed})
            {
                text += ",";
                appendNumber(text, value, 4);
            }
            text += "\n";
        }

        const auto name = outputDirectory() + "/shimmy-idle-brake" + fixed(brake, 0) + ".csv";
        writeFile(name, text);

        // What the transports make of the published series: the ramped forward hold at a clean
        // 120 Hz cadence, and the same at a 60 Hz burst where the whole slope plays out inside
        // every frame. Alternating content is what separates them — a first-order hold run
        // forward amplifies exactly that.
        auto cleanHold = std::vector<double>{};
        auto burstHold = std::vector<double>{};

        for (auto index = std::size_t{1}; index < published.size(); index++)
        {
            const auto change = published[index] - published[index - 1];

            for (const auto fraction : {0.25, 0.5, 0.75, 1.0})
            {
                cleanHold.push_back(published[index] + change * fraction);
            }

            if (index % 2 == 1)
            {
                for (const auto fraction : {0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0})
                {
                    burstHold.push_back(published[index] + change * fraction);
                }
            }
        }

        const auto source = statisticsOf(substepTorques, substepTorques.size() / 3);
        const auto sampled = statisticsOf(published, published.size() / 3);
        const auto clean = statisticsOf(cleanHold, cleanHold.size() / 3);
        const auto burst = statisticsOf(burstHold, burstHold.size() / 3);

        WARN("idling in first, brake " << fixed(brake, 0) << ": stage one at 360 Hz " << fixed(source.mean, 3)
                                       << " +- " << fixed(source.deviation, 3) << " Nm [" << fixed(source.lowest, 3)
                                       << ", " << fixed(source.highest, 3) << "], published at 120 Hz +- "
                                       << fixed(sampled.deviation, 3) << ", after the forward hold +- "
                                       << fixed(clean.deviation, 3) << " clean / " << fixed(burst.deviation, 3)
                                       << " burst -> " << name);
    }
}
