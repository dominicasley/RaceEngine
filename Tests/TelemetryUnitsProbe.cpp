// The absolute-units sweep: one value per channel, against a hand calculation, enumerated.
//
// `./EngineTests "[.telemetry-units]"`, hidden like the other probes because what it produces is a
// table to read rather than a pass or a fail.
//
// **Why one value per channel and not a shape.** `Engine RPM [rpm]` carried rad/s for the whole of
// milestone 1 and survived because milestone 1 validated *shapes* — monotonic gradients, bounded
// overshoot, settling times — and a uniform scale factor preserves every shape there is. It only
// surfaced when somebody compared an absolute number against a real car's idle. So this asks the one
// question those tests structurally cannot: is the number in the column the number the column's name
// and unit promise?
//
// Two kinds of failure are possible and they need different fixtures, so there are two:
//
//   1. **The conversion at the CSV boundary** — a radian written as a degree, a metre as a
//      millimetre. Caught by handing `telemetryToCsv` a frame whose every field is a distinct known
//      number and reading the columns back.
//   2. **The quantity in the field** — the right unit applied to the wrong thing, which no
//      conversion check can see. Caught by stepping a real vehicle at an attitude where the world
//      frame and the body frame disagree, and asking whether a channel named for one carries it.

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TelemetryFrame;
using raceengine::telemetryToCsv;
using raceengine::VehicleInput;
using raceengine::VehicleState;

namespace
{

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

[[nodiscard]] std::vector<std::string> split(const std::string& line)
{
    auto fields = std::vector<std::string>{};
    auto start = std::size_t{0};

    while (start <= line.size())
    {
        const auto comma = line.find(',', start);
        fields.push_back(line.substr(start, comma == std::string::npos ? std::string::npos : comma - start));

        if (comma == std::string::npos)
        {
            break;
        }

        start = comma + 1;
    }

    return fields;
}

// Header row and first data row, as parallel vectors.
struct Written
{
    std::vector<std::string> headers;
    std::vector<std::string> values;
};

[[nodiscard]] Written writeOne(const TelemetryFrame& frame)
{
    const auto text = telemetryToCsv({frame});

    const auto firstBreak = text.find('\n');
    const auto secondBreak = text.find('\n', firstBreak + 1);

    return Written{split(text.substr(0, firstBreak)),
                   split(text.substr(firstBreak + 1, secondBreak - firstBreak - 1))};
}

auto failures = 0;

void report(const Written& written, const std::string& column, const double expected, const double tolerance,
            const char* reasoning)
{
    for (auto index = std::size_t{0}; index < written.headers.size(); index++)
    {
        if (written.headers[index] != column)
        {
            continue;
        }

        const auto actual = std::stod(written.values[index]);
        const auto passed = std::abs(actual - expected) <= tolerance;
        failures += passed ? 0 : 1;

        std::printf("  %-4s %-26s %14.4f %14.4f   %s\n", passed ? "PASS" : "FAIL", column.c_str(), actual, expected,
                    reasoning);

        return;
    }

    failures++;
    std::printf("  %-4s %-26s %14s %14s   %s\n", "GONE", column.c_str(), "-", "-", "no column of this name");
}

} // namespace

TEST_CASE("every telemetry column, one value, against a hand calculation", "[.telemetry-units]")
{
    failures = 0;

    // Distinct values everywhere, so a column reading its neighbour is visible rather than plausible.
    auto frame = TelemetryFrame{};
    frame.time = 1.25;
    frame.tick = 450;
    frame.position = glm::dvec3(11.0, 0.55, 22.0);
    // 3-4-5 in the horizontal plane plus a vertical component: |v| is exactly 13 m/s = 46.8 km/h.
    frame.velocity = glm::dvec3(3.0, 12.0, 4.0);
    // One g down, half a g one way, a quarter the other.
    frame.acceleration = glm::dvec3(0.5 * 9.80665, -9.80665, 0.25 * 9.80665);
    frame.yaw = 0.5;
    frame.pitch = 0.05;
    frame.roll = -0.10;
    frame.yawRate = 0.30;
    frame.pitchRate = 0.02;
    frame.rollRate = -0.04;
    // A steering demand and the rim angle it makes. **Two fields now, and two columns**: this used
    // to be one field carrying the demand, which the CSV multiplied by 57.2958 and called degrees.
    frame.steering = 0.5;
    frame.steeringWheelAngle = 0.5 * 0.5 * 13.194689145077131;
    frame.throttle = 0.75;
    frame.brake = 0.25;
    frame.gear = 3;
    frame.shiftPhase = 2;
    // 3000 rpm is 314.159 rad/s, and this is the channel the whole exercise is named after.
    frame.engineSpeed = 3000.0 / 9.549296585513721;
    frame.engineTorque = 350.0;
    frame.clutchTorque = 320.0;
    frame.clutchSlip = 12.0;
    frame.clutchSlipEnergy = 4500.0;

    frame.wheels[0].verticalLoad = 4000.0;
    frame.wheels[0].slipRatio = 0.08;
    frame.wheels[0].slipAngle = 0.10;
    frame.wheels[0].forceLongitudinal = 1200.0;
    frame.wheels[0].forceLateral = 3400.0;
    frame.wheels[0].aligningMoment = -85.0;
    frame.wheels[0].suspensionTravel = 0.0134;
    frame.wheels[0].damperVelocity = 0.075;
    frame.wheels[0].angularVelocity = 48.4;
    frame.wheels[0].camber = -0.0349;
    frame.wheels[0].gripMultiplier = 0.85;
    frame.wheels[0].inContact = true;
    frame.wheels[0].contactingSamples = 7;
    // Metres, like every other length in the frame. Chosen distinct from `suspensionTravel` so a
    // column reading its neighbour cannot land on a plausible number.
    frame.wheels[0].patchDepthSpread = 0.0105;

    const auto written = writeOne(frame);

    std::printf("\n=== part 1: the conversion at the CSV boundary ===\n");
    std::printf("  %-4s %-26s %14s %14s   %s\n", "", "column", "written", "expected", "why that is the right number");

    report(written, "Time [s]", 1.25, 1e-6, "seconds in, seconds out");
    report(written, "Tick", 450.0, 0.5, "a count");
    report(written, "Pos X [m]", 11.0, 1e-4, "metres in, metres out");
    report(written, "Speed [km/h]", 13.0 * 3.6, 1e-3, "|(3,12,4)| = 13 m/s, times 3.6");
    report(written, "Vel Z [m/s]", 4.0, 1e-4, "metres a second in, the same out");

    report(written, "G Force Long [g]", 0.25, 1e-4, "the car's own forward axis, over 9.80665");
    report(written, "G Force Lat [g]", 0.5, 1e-4, "the car's own lateral axis, over 9.80665");
    report(written, "G Force Vert [g]", -1.0, 1e-4, "the car's own vertical axis, over 9.80665");

    report(written, "Yaw [deg]", 0.5 * 57.29577951308232, 1e-3, "0.5 rad");
    report(written, "Roll [deg]", -0.10 * 57.29577951308232, 1e-3, "-0.1 rad");
    report(written, "Yaw Rate [deg/s]", 0.30 * 57.29577951308232, 1e-3, "0.3 rad/s");

    // The column that had to be fixed: it carried the *demand* times a radians conversion, so half
    // lock printed 28.6 where this car's rim is at 189 degrees.
    report(written, "Steering Angle [deg]", 0.5 * 756.0 * 0.5, 1e-2, "0.5 of full lock on a 756 deg rim");
    report(written, "Steering Demand []", 0.5, 1e-6, "the driver's demand, which has no units at all");

    report(written, "Throttle Pos [%]", 75.0, 1e-3, "0.75 as a percentage");
    report(written, "Brake Pos [%]", 25.0, 1e-3, "0.25 as a percentage");
    report(written, "Gear", 3.0, 0.5, "a count");
    report(written, "Engine RPM [rpm]", 3000.0, 0.1, "314.159 rad/s is 3000 rpm");
    report(written, "Engine Torque [Nm]", 350.0, 1e-3, "newton metres in, newton metres out");
    report(written, "Clutch Slip [rad/s]", 12.0, 1e-3, "radians a second, as the name says");
    report(written, "Clutch Slip Energy [J]", 4500.0, 0.1, "joules");

    report(written, "Tyre Load FL [N]", 4000.0, 1e-2, "newtons");
    report(written, "Slip Ratio FL []", 0.08, 1e-5, "dimensionless");
    report(written, "Slip Angle FL [deg]", 0.10 * 57.29577951308232, 1e-3, "0.1 rad");
    report(written, "Aligning Moment FL [Nm]", -85.0, 1e-3, "newton metres");
    report(written, "Susp Pos FL [mm]", 13.4, 1e-2, "0.0134 m in millimetres");
    report(written, "Damper Vel FL [mm/s]", 75.0, 1e-2, "0.075 m/s in millimetres a second");
    report(written, "Wheel Speed FL [rad/s]", 48.4, 1e-3, "radians a second, as the name says");
    report(written, "Camber FL [deg]", -2.0, 1e-2, "-0.0349 rad is two degrees of negative camber");
    report(written, "Grip FL []", 0.85, 1e-4, "dimensionless");
    report(written, "Contact Samples FL []", 7.0, 0.5, "a count, of nine");
    // The sagitta of a 0.31 m tread over a 0.16 m patch is 10.50 mm, which is what a level patch on
    // flat ground reports — so this is the channel's own floor checked against geometry rather than
    // against a number chosen to make the row pass.
    report(written, "Patch Depth Spread FL [mm]", 10.5, 1e-2, "0.0105 m in millimetres, the tread's own rise");

    std::printf("\n  %d of the columns above disagree with their own names.\n", failures);
}

TEST_CASE("and the quantity in the field is the one the column is named for", "[.telemetry-units]")
{
    // Part 2, and it needs a real vehicle because the failure it looks for is invisible in any
    // fixture where the body frame and the world frame agree.
    //
    // The car is placed **heading along world +x**, a quarter turn off the axis a default-constructed
    // state faces. Anything that is genuinely in the body frame is unmoved by that; anything that is
    // a world component wearing a body name swaps with its neighbour.
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
    const auto vehicle = built.value();

    constexpr auto tick = 1.0 / 360.0;
    constexpr auto speed = 20.0;

    const auto driveFacing = [&](const double heading)
    {
        auto state = VehicleState{};
        state.chassis.orientation = glm::angleAxis(heading, glm::dvec3(0.0, 1.0, 0.0));
        // Well inside the ground in both directions: the second run heads along +x, and a start
        // at z = 0 would have it riding the edge of a surface that begins there.
        state.chassis.position = glm::dvec3(0.0, 0.52, 200.0);
        state.chassis.linearVelocity = state.chassis.orientation * glm::dvec3(0.0, 0.0, speed);

        for (auto index = std::size_t{0}; index < cornerCount; index++)
        {
            state.corners[index].wheelSpeed = speed / 0.31;
        }

        auto last = raceengine::VehicleStep{};

        for (auto step = 0; step < 900; step++)
        {
            auto input = VehicleInput{};
            input.steering = std::min(0.15, 0.6 * static_cast<double>(step) * tick);

            const auto stepped = stepVehicle(vehicle, state, input, noDriveTorque, world.value(), tick);
            REQUIRE(stepped.has_value());
            last = stepped.value();
        }

        return last.telemetry;
    };

    const auto alongZ = driveFacing(0.0);
    const auto alongX = driveFacing(1.5707963267948966);

    std::printf("\n=== part 2: the quantity, at two headings a quarter turn apart ===\n");
    std::printf("  a body-frame channel reads the same in both columns. A world component does not.\n\n");
    std::printf("  %-28s %14s %14s   %s\n", "", "facing +z", "facing +x", "");

    const auto both = [](const char* name, const double first, const double second, const char* note)
    {
        const auto agree = std::abs(first - second) <= 0.05 * std::max(1.0, std::abs(first));
        std::printf("  %-4s %-24s %14.4f %14.4f   %s\n", agree ? "PASS" : "FAIL", name, first, second, note);

        return agree;
    };

    std::ignore = both("acceleration -> Long", alongZ.acceleration.z, alongX.acceleration.z,
                       "the CSV writes .z into G Force Long");
    std::ignore = both("acceleration -> Lat", alongZ.acceleration.x, alongX.acceleration.x,
                       "the CSV writes .x into G Force Lat");
    std::ignore = both("yaw rate", alongZ.yawRate, alongX.yawRate, "body rates, converted on the way in");
    std::ignore = both("roll", alongZ.roll, alongX.roll, "read off the body's own axes");
    std::ignore = both("Tyre Load FL", alongZ.wheels[0].verticalLoad, alongX.wheels[0].verticalLoad, "a scalar");
    std::ignore = both("Slip Angle FL", alongZ.wheels[0].slipAngle, alongX.wheels[0].slipAngle, "a scalar");
    std::ignore = both("Tyre Force Y FL", alongZ.wheels[0].forceLateral, alongX.wheels[0].forceLateral, "a scalar");

    std::printf("\n  the same manoeuvre was driven twice, a quarter turn apart. Every channel above\n"
                "  should have read the same both times.\n");
}
