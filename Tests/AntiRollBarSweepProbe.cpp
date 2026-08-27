// What the anti-roll bars are worth on the skidpad: `./EngineTests "[.bar-sweep]"`.
//
// **This is a diagnostic and it is explicitly not a calibration target.** `docs/known-red.md` names
// the anti-roll bars as the honest lever on the one red the geometric load path left — the skidpad's
// 0.8918 g against the real car's 0.90-0.95 band — because `PublishedCarsImpl.cpp` carries AC's
// `ARB FRONT 34000 / REAR 15000` as unverified. Reading a rate off this table and authoring it would
// be fitting the car to the test, which this project has a written rule against. What the table is
// for is the question *before* that one: whether a bar rate inside any credible range can move the
// skidpad at all, and what it costs in roll on a car whose body attitude the seat has already
// accepted.
//
// The two columns are the two halves of that. **Peak g** is the quantity the criterion asserts,
// measured on the criterion's own fixture. **Front share** is the fraction of the lateral load
// transfer the front axle carries, which is the mechanism: the geometric load path took it from
// 53.9% to 58.1%, and more front transfer is more understeer and a lower peak. **Roll gradient** is
// the constraint, because Dominic drove this car on 2026-08-27 and accepted its attitude at
// 2.41-2.61 deg/g — so a bar change that buys peak grip by letting the body lean further is
// spending something that has already been signed off.
//
// One axle is swept at a time with the other held at AC's figure, because a two-axle sweep answers a
// question nobody has asked: what is wanted is the sensitivity of the balance to each bar, not a
// surface to search.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::CornerSide;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::outboardSign;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

namespace
{

constexpr auto tick = 1.0 / 360.0;
constexpr auto gravity = 9.80665;
constexpr auto degrees = 57.29577951308232;
constexpr auto designHeight = 0.572;
constexpr auto tyreRadius = 0.3186;

// AC's own pair, so the sweep can be read against the shipped car without a second lookup.
constexpr auto shippedFront = 34000.0;
constexpr auto shippedRear = 15000.0;

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

ProvingGroundDescriptor plate(const double size = 1200.0)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = size;
    descriptor.width = size;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

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

struct SteadyState
{
    double lateralAcceleration = 0.0;
    double roll = 0.0;
    double frontShare = 0.0;
};

// The skidpad criterion's own driven hold, carried across so this measures the quantity that is red
// rather than a near relative of it: same speed controller, same ten seconds, same last-second
// window. The two additions are the roll angle and the per-corner vertical loads, both read from the
// same samples.
SteadyState hold(const VehicleSetup& setup, const PhysicsWorld& world, const double steering, const double speed)
{
    auto state = VehicleState{};
    settle(setup, state, world, speed, 400.0);

    auto input = VehicleInput{};
    input.steering = steering;

    auto integral = 0.0;
    auto result = SteadyState{};
    auto load = std::array<double, cornerCount>{};
    auto samples = 0;
    auto slowest = std::numeric_limits<double>::max();
    auto fastest = 0.0;

    for (auto step = 0; step < 3600; step++)
    {
        const auto error = speed - glm::length(state.chassis.linearVelocity);
        integral = std::clamp(integral + error * tick, -4.0, 4.0);
        const auto perWheel = std::clamp(2000.0 * error + 6000.0 * integral, -8000.0, 8000.0) *
                              setup.corners.front().hardpoints.wheelRadius / 2.0;
        const auto drive = std::array<double, cornerCount>{perWheel, perWheel, 0.0, 0.0};

        const auto stepped = stepVehicle(setup, state, input, drive, world, tick);
        REQUIRE(stepped.has_value());

        if (step >= 3240)
        {
            constexpr auto toTheRight = outboardSign(CornerSide::Right);
            const auto carried = glm::length(state.chassis.linearVelocity);

            result.lateralAcceleration += stepped->telemetry.acceleration.x * toTheRight;
            result.roll += stepped->telemetry.roll;

            for (auto index = std::size_t{0}; index < cornerCount; index++)
            {
                load[index] += stepped->corners[index].forces.tireVertical;
            }

            slowest = std::min(slowest, carried);
            fastest = std::max(fastest, carried);
            samples++;
        }
    }

    result.lateralAcceleration /= static_cast<double>(samples) * gravity;
    result.roll /= static_cast<double>(samples);

    // The fixture asserts its own precondition, which is this project's standing rule and the one
    // that caught four faults in a month: a hold that did not hold its speed is a spiral, and every
    // number it reports describes a transient rather than a limit.
    CAPTURE(steering, speed, slowest, fastest);
    REQUIRE(slowest > 0.9 * speed);
    REQUIRE(fastest < 1.1 * speed);

    const auto frontTransfer = std::abs(load[0] - load[1]);
    const auto rearTransfer = std::abs(load[2] - load[3]);
    const auto total = frontTransfer + rearTransfer;
    result.frontShare = total > 1e-9 ? frontTransfer / total : 0.0;

    return result;
}

// The peak of the criterion's sweep, over the steering range that brackets it. The full criterion
// runs twelve angles from 0.02 to 1.00 because it also asserts the shape of the gradient; the peak
// itself sits between 0.30 and 0.45 on the shipped car, so five angles around it read the same
// maximum for a fifth of the time.
struct Sample
{
    double peak = 0.0;
    double rollGradient = 0.0;
    double frontShare = 0.0;
};

Sample skidpad(const double frontBar, const double rearBar, const PhysicsWorld& world)
{
    auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    setup->corners[0].antiRollRate = frontBar;
    setup->corners[1].antiRollRate = frontBar;
    setup->corners[2].antiRollRate = rearBar;
    setup->corners[3].antiRollRate = rearBar;

    auto best = Sample{};
    for (const auto steering : {0.22, 0.30, 0.35, 0.45, 0.60})
    {
        const auto held = hold(setup.value(), world, steering, 20.0);
        if (held.lateralAcceleration > best.peak)
        {
            best.peak = held.lateralAcceleration;
            best.rollGradient = std::abs(held.roll * degrees) / std::max(std::abs(held.lateralAcceleration), 1e-6);
            best.frontShare = held.frontShare;
        }
    }

    return best;
}

void report(const char* label, const double frontBar, const double rearBar, const PhysicsWorld& world)
{
    const auto sample = skidpad(frontBar, rearBar, world);

    std::printf("%-10s %9.0f %9.0f | %8.4f %12.4f %14.3f\n", label, frontBar, rearBar, sample.peak, sample.frontShare,
                sample.rollGradient);
    std::fflush(stdout);
}

} // namespace

TEST_CASE("what the anti-roll bars are worth on the skidpad", "[.bar-sweep]")
{
    const JoltGuard jolt;

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    std::printf("\n=== the anti-roll bars against the skidpad, the balance and the attitude ===\n");
    std::printf("\n%-10s %9s %9s | %8s %12s %14s\n", "sweep", "front N/m", "rear N/m", "peak g", "front share",
                "roll deg/g");
    std::printf("%s\n", "-----------------------------------------------------------------------------");

    // The front bar, with the rear held at AC's figure. Softening the front takes load transfer off
    // the front axle, which is the direction that buys peak grip on a nose-heavy front-drive car.
    for (const auto front : {0.0, 8500.0, 17000.0, 25500.0, shippedFront, 42500.0})
    {
        report(front == shippedFront ? "front *" : "front", front, shippedRear, world.value());
    }

    // The rear bar, with the front held. Stiffening the rear moves transfer rearward, which is the
    // same direction by the other end and is the move a front-drive car is normally given.
    for (const auto rear : {0.0, 7500.0, shippedRear, 22500.0, 30000.0, 40000.0})
    {
        report(rear == shippedRear ? "rear *" : "rear", shippedFront, rear, world.value());
    }

    std::printf("\n  * is the shipped car. The band the criterion asserts is 0.90-0.95 g, and the\n");
    std::printf("  attitude the seat accepted on 2026-08-27 is 2.41-2.61 deg/g.\n");

    // **Where the criterion's own number comes from, measured rather than assumed.** The shipped row
    // above reads higher than the 0.8918 g `docs/known-red.md` records, and the difference is not the
    // car: it is which steering angles each sweep visits. A peak taken on a grid is an
    // underestimate of the true peak, always, and by an amount nobody has measured until now. This
    // block prints both grids and the angle each peak lands on, so the size of that bias is a number
    // instead of an argument.
    //
    // Nothing here proposes changing the criterion. Refining a grid moves a measured peak upward by
    // construction, which is the direction that turns this red green — and that is exactly the shape
    // `docs/known-red.md` exists to stop somebody doing quietly.
    auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    const auto peakOver = [&setup, &world](const char* label, const std::vector<double>& angles)
    {
        auto peak = 0.0;
        auto at = 0.0;

        std::printf("\n  %s\n", label);
        for (const auto steering : angles)
        {
            const auto held = hold(setup.value(), world.value(), steering, 20.0);
            std::printf("    steering %.3f -> %.4f g\n", steering, held.lateralAcceleration);
            std::fflush(stdout);

            if (held.lateralAcceleration > peak)
            {
                peak = held.lateralAcceleration;
                at = steering;
            }
        }

        std::printf("    peak %.4f g at %.3f\n", peak, at);

        return std::pair{peak, at};
    };

    std::printf("\n=== the criterion's grid against a finer one, on the shipped car ===\n");

    // `GolfGtiTests.cpp`'s own list, character for character.
    const auto coarse = peakOver("the criterion's twelve angles",
                                 {0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.22, 0.30, 0.45, 0.60, 0.80, 1.00});

    // **Every sample printed, not just the maximum**, because how thin the pass is depends on how
    // flat the peak is — and a peak that is flat over a wide band of steering is a different claim
    // about the car than one that is sharp. This is the scan the criterion's added angles were
    // chosen from.
    const auto fine = peakOver("a dense scan across the peak",
                               {0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.42, 0.45});

    std::printf("\n  criterion %.4f g at %.2f, dense %.4f g at %.2f.\n", coarse.first, coarse.second, fine.first,
                fine.second);
    std::printf("  The gap between those two is the grid's bias, not the car's grip.\n");
}
