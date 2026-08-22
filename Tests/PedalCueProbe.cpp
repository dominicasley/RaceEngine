// What the pedal motors would actually do, driven by the car: `./EngineTests "[.pedal-cue]"`.
//
// The unit tests pin the derivation against hand-written wheels. This drives the real vehicle and
// the real driveline through the three things a driver does that the cue exists for, and prints what
// each foot would have felt — which is the only way to find out whether the thresholds fire where a
// driver would want them to rather than merely where the arithmetic says.
//
// It needs no pedals and no wheel. That is the point of the split: everything above the write is
// testable on a machine with nothing plugged into it.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::cornerCount;
using raceengine::derivePedalFeedback;
using raceengine::DrivelineState;
using raceengine::DrivenAxle;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::golfGtiMk7Driveline;
using raceengine::mapPedalFeedback;
using raceengine::PedalFeedbackSetup;
using raceengine::PedalMotorMapping;
using raceengine::PedalMotorProfile;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::roadTorques;
using raceengine::SlippingWheel;
using raceengine::startEngine;
using raceengine::stepDriveline;
using raceengine::stepVehicle;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::wheelInertias;

namespace
{

constexpr auto tick = 1.0 / 360.0;

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(raceengine::bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        raceengine::tearDownJolt();
    }
};

} // namespace

TEST_CASE("what a driver's feet would feel, on the three manoeuvres the cue is for", "[.pedal-cue]")
{
    const JoltGuard jolt;

    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 900.0;
    descriptor.width = 200.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    const auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());
    const auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());
    const auto vehicle = built.value();
    const auto driveline = golfGtiMk7Driveline();

    auto motors = PedalMotorProfile{};
    motors.hasMotors = true;

    const auto run = [&](const char* what, const double entrySpeed, const int ticks,
                         const auto& driverAt)
    {
        auto state = VehicleState{};
        state.chassis.position = glm::dvec3(0.0, 0.52, 20.0);
        state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, entrySpeed);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wheelSpeed = entrySpeed / 0.31;
        }

        auto engine = DrivelineState{};
        startEngine(driveline, engine);

        // A driveline placed at speed must have its shaft placed too, or the compliance reads a
        // twist nobody wound in and hands the wheels a few thousand newton metres on the first tick.
        raceengine::placeDriveline(driveline, engine, entrySpeed / 0.31);

        auto road = std::array<double, cornerCount>{};

        std::printf("\n=== %s ===\n", what);
        std::printf("%7s %8s %9s %9s %9s %7s %7s   %s\n", "t [s]", "km/h", "worst-", "worst+", "peak slip", "brake",
                    "accel", "motor codes");

        auto peakBrakeCue = 0.0;
        auto peakThrottleCue = 0.0;

        for (auto step = 0; step < ticks; step++)
        {
            const auto time = static_cast<double>(step) * tick;
            const auto input = driverAt(time);

            const auto torques =
                stepDriveline(driveline, engine,
                              {state.corners[0].wheelSpeed, state.corners[1].wheelSpeed, state.corners[2].wheelSpeed,
                               state.corners[3].wheelSpeed},
                              wheelInertias(vehicle), road, input, tick);
            REQUIRE(torques.has_value());

            const auto stepped = stepVehicle(vehicle, state, input, torques->wheel, world.value(), tick);
            REQUIRE(stepped.has_value());
            road = roadTorques(stepped.value());

            auto wheels = std::array<SlippingWheel, cornerCount>{};
            auto worstNegative = 0.0;
            auto worstPositive = 0.0;
            auto peakSlip = 0.0;

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                const auto& solution = stepped->corners[index];
                const auto driven = driveline.driven == DrivenAxle::All ||
                                    (driveline.driven == DrivenAxle::Front ? index < 2 : index >= 2);

                wheels[index] = SlippingWheel{.slipRatio = solution.contact.slip.slipRatio,
                                              .peakSlipRatio = solution.contact.tyre.longitudinalPeakSlip,
                                              .load = solution.forces.tireVertical,
                                              .inContact = solution.patch.inContact,
                                              .driven = driven};

                worstNegative = std::min(worstNegative, solution.contact.slip.slipRatio);
                worstPositive = std::max(worstPositive, solution.contact.slip.slipRatio);
                peakSlip = std::max(peakSlip, solution.contact.tyre.longitudinalPeakSlip);
            }

            const auto share = 0.25 * state.chassis.mass * 9.80665;
            const auto cue = derivePedalFeedback(PedalFeedbackSetup{}, std::span<const SlippingWheel>(wheels),
                                                 input.brake, input.throttle, share);
            const auto command = mapPedalFeedback(motors, PedalMotorMapping{}, cue);

            peakBrakeCue = std::max(peakBrakeCue, cue.brake);
            peakThrottleCue = std::max(peakThrottleCue, cue.throttle);

            if (step % 60 == 0)
            {
                std::printf("%7.2f %8.1f %9.3f %9.3f %9.3f %7.2f %7.2f   brake %3u  accel %3u\n", time,
                            glm::length(state.chassis.linearVelocity) * 3.6, worstNegative, worstPositive, peakSlip,
                            cue.brake, cue.throttle, command.brake, command.throttle);
            }
        }

        std::printf("  peak cue over the run: brake %.2f, accelerator %.2f\n", peakBrakeCue, peakThrottleCue);
    };

    // A front-drive hatchback at full throttle from rest: the fronts light up, then hook up.
    run("standing start, full throttle", 0.0, 900,
        [](const double)
        {
            auto input = VehicleInput{};
            input.throttle = 1.0;
            input.gear = 1;

            return input;
        });

    // Threshold braking then past it. The pedal ramps so the cue's onset can be read against it.
    run("braking from 120 km/h, ramping past the limit", 33.3, 900,
        [](const double time)
        {
            auto input = VehicleInput{};
            input.gear = 3;
            input.brake = std::min(1.0, 0.35 + 0.35 * time);

            return input;
        });

    // The case that must be silent: a car doing nothing wrong, with a foot on each pedal in turn.
    run("cruising, then gentle pedal use", 25.0, 900,
        [](const double time)
        {
            auto input = VehicleInput{};
            input.gear = 4;
            input.throttle = time < 1.5 ? 0.35 : 0.0;
            input.brake = time > 1.5 ? 0.25 : 0.0;

            return input;
        });
}
